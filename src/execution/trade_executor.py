"""Trade execution facade – routes between paper and live modes.

Default behaviour is **paper mode** (simulation only).  Live execution is
activated ONLY when ``ENABLE_AUTO_TRADING=true`` AND all required env vars
and signer validation pass.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.bot.config import BotConfig
from src.execution.broadcaster import Broadcaster
from src.execution.jupiter_client import (
    LAMPORTS_PER_SOL,
    SOL_MINT,
    JupiterClient,
    SwapTransaction,
)
from src.execution.signer import LocalSigner
from src.execution.trade_executor_live import LiveTradeExecutor, LiveTradeResult
from src.strategy.local_quote import PumpAMMQuoteEngine

logger = logging.getLogger(__name__)

_BASE_TX_FEE_LAMPORTS = 5_000


@dataclass(frozen=True)
class PaperTradeEstimate:
    """Paper-mode execution estimate based on live quote + fee snapshots."""

    success: bool
    input_mint: str
    output_mint: str
    in_amount: int = 0
    out_amount: int = 0
    slippage_bps: int = 0
    priority_fee_lamports: int = 0
    jito_tip_lamports: int = 0
    base_fee_lamports: int = _BASE_TX_FEE_LAMPORTS
    total_network_fee_lamports: int = 0
    error: Optional[str] = None
    raw_priority_fee: Optional[dict] = None
    price_impact_pct: float = 0.0


@dataclass(frozen=True)
class PaperEntryGuardResult:
    """Pre-entry sanity check for paper mode executable liquidity."""

    allowed: bool
    reason: Optional[str] = None
    buy_estimate: Optional[PaperTradeEstimate] = None
    sell_estimate: Optional[PaperTradeEstimate] = None
    entry_cost_sol: float = 0.0
    immediate_exit_net_sol: float = 0.0
    roundtrip_ratio: float = 0.0
    roundtrip_pnl_sol: float = 0.0


class TradeExecutor:
    """Unified execution entry point.

    In **paper mode** (default) this object exposes read-only metadata and
    the engines handle simulation directly.

    In **live mode** (``ENABLE_AUTO_TRADING=true``) it initialises the full
    execution stack: :class:`LocalSigner`, :class:`JupiterClient`, and
    :class:`Broadcaster`, then exposes :meth:`execute_buy` and
    :meth:`execute_sell` which delegate to :class:`LiveTradeExecutor`.
    """

    def __init__(
        self, config: BotConfig, local_quote_engine: Optional[PumpAMMQuoteEngine] = None
    ) -> None:
        self.config = config
        self.local_quote_engine = local_quote_engine
        self.live: bool = False
        self._live_executor: Optional[LiveTradeExecutor] = None
        self._signer: Optional[LocalSigner] = None
        self._paper_jupiter: Optional[JupiterClient] = None
        self._paper_broadcaster: Optional[Broadcaster] = None
        self._paper_quote_disabled_until: datetime | None = None
        self._paper_quote_failure_count: int = 0
        self._paper_quote_last_error: Optional[str] = None
        self.init_diagnostics: list[str] = []
        self.fallback_reason: Optional[str] = None
        # Thread pool for background Jupiter quote pre-fetching (entry guard parallelism).
        self._thread_pool: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="texec"
        )

        self._init_paper_simulation_stack()

        if config.enable_auto_trading:
            self._init_live()
        else:
            self.fallback_reason = "ENABLE_AUTO_TRADING is not true"
            self.init_diagnostics.append(self.fallback_reason)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_paper_simulation_stack(self) -> None:
        """Best-effort initialization for realistic paper execution estimates."""
        if self.config.jupiter_base_url:
            try:
                self._paper_jupiter = JupiterClient(
                    base_url=self.config.jupiter_base_url,
                    api_key=self.config.jupiter_api_key,
                    user_public_key=self.config.bot_public_key,
                    timeout_sec=10,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Paper quote init failed (Jupiter): %s", exc)
                self._paper_jupiter = None
        if self.config.helius_rpc_url:
            try:
                self._paper_broadcaster = Broadcaster(
                    rpc_url=self.config.helius_rpc_url,
                    timeout_sec=10,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Paper fee-estimator init failed (Helius RPC): %s", exc)
                self._paper_broadcaster = None

    def _init_live(self) -> None:
        """Bootstrap the live execution stack with full validation."""
        self.init_diagnostics.clear()
        self.fallback_reason = None
        logger.warning("=" * 60)
        logger.warning("⚠️  LIVE TRADING MODE REQUESTED")
        logger.warning("=" * 60)

        errors: list[str] = []

        # --- Env-var checks ---
        if not self.config.bot_private_key_b58:
            errors.append("BOT_PRIVATE_KEY_B58 is missing or empty")
        if not self.config.helius_rpc_url:
            errors.append("HELIUS_RPC_URL is missing or empty")
        if not self.config.jupiter_base_url:
            errors.append("JUPITER_BASE_URL is missing or empty")
        if not self.config.helius_api_key:
            errors.append("HELIUS_API_KEY is missing or empty")

        if errors:
            for err in errors:
                logger.error("Live-mode env check failed: %s", err)
                self.init_diagnostics.append(err)
            self.fallback_reason = "missing required live env vars"
            logger.error("Falling back to PAPER mode due to missing env vars.")
            return

        # --- Signer validation ---
        signer = LocalSigner(self.config.bot_private_key_b58)
        ok, err_msg = signer.validate()
        if not ok:
            self.fallback_reason = "signer validation failed"
            self.init_diagnostics.append(f"signer validation failed: {err_msg}")
            logger.error("Signer validation failed: %s – falling back to PAPER mode.", err_msg)
            return
        self._signer = signer

        # Log the derived public key (never the private key)
        pubkey = signer.get_public_key()
        logger.info("Signer public key: %s", pubkey)

        # Cross-check with BOT_PUBLIC_KEY if provided
        if self.config.bot_public_key and self.config.bot_public_key != pubkey:
            self.fallback_reason = "BOT_PUBLIC_KEY mismatch"
            self.init_diagnostics.append(
                f"BOT_PUBLIC_KEY mismatch: configured {self.config.bot_public_key} != derived {pubkey}"
            )
            logger.error(
                "BOT_PUBLIC_KEY (%s) does not match derived key (%s) – refusing live mode.",
                self.config.bot_public_key,
                pubkey,
            )
            return

        # --- Build execution stack ---
        try:
            jupiter = JupiterClient(
                base_url=self.config.jupiter_base_url,
                api_key=self.config.jupiter_api_key,
                timeout_sec=30,
            )
            broadcaster = Broadcaster(
                rpc_url=self.config.helius_rpc_url,
                timeout_sec=30,
                sender_url=self.config.helius_sender_url,
                bundle_url=self.config.helius_bundle_url,
                broadcast_mode=self.config.live_broadcast_mode,
                jito_tip_accounts=self.config.jito_tip_accounts,
                confirm_poll_interval_sec=float(self.config.live_confirm_poll_interval_ms) / 1000.0,
                rebroadcast_interval_sec=float(self.config.live_rebroadcast_interval_ms) / 1000.0,
                max_rebroadcast_attempts=self.config.live_max_rebroadcast_attempts,
                sender_idle_ping_sec=float(self.config.live_sender_idle_ping_sec),
                sender_active_warm=bool(self.config.live_sender_active_warm),
                sender_warm_interval_sec=float(self.config.live_sender_warm_interval_sec),
            )
        except Exception as exc:  # noqa: BLE001
            self.fallback_reason = "execution stack init failed"
            self.init_diagnostics.append(f"execution stack init failed: {exc}")
            logger.error(
                "Failed to initialise execution stack: %s – falling back to PAPER mode.",
                exc,
            )
            return

        self._live_executor = LiveTradeExecutor(
            config=self.config,
            signer=signer,
            jupiter=jupiter,
            broadcaster=broadcaster,
            local_quote_engine=self.local_quote_engine,
        )
        self.live = True
        self.fallback_reason = None
        self.init_diagnostics.append("live execution stack initialized")

        logger.warning("🔴 LIVE TRADING ENABLED – real funds will be used!")
        logger.warning(
            "Limits: max_position=%.4f SOL | max_exposure=%.4f SOL | max_daily_loss=%.4f SOL | max_positions=%d",
            self.config.max_position_sol,
            self.config.max_total_exposure_sol,
            self.config.max_daily_loss_sol,
            self.config.max_open_positions,
        )
        logger.warning(
            "Live transport: mode=%s | sender=%s | bundle=%s",
            self.config.live_broadcast_mode,
            bool(self.config.helius_sender_url),
            bool(self.config.helius_bundle_url),
        )

    def close(self) -> None:
        """Close best-effort live/paper transport resources."""
        try:
            if self._paper_jupiter is not None:
                self._paper_jupiter.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._paper_broadcaster is not None:
                self._paper_broadcaster.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._live_executor is not None:
                self._live_executor.jupiter.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._live_executor is not None:
                self._live_executor.broadcaster.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def mode_label(self) -> str:
        """Return ``'LIVE'`` or ``'PAPER'``."""
        return "LIVE" if self.live else "PAPER"

    # ------------------------------------------------------------------
    # Paper estimation helpers (Jupiter quote + Helius fee estimate)
    # ------------------------------------------------------------------

    def submit_buy_quote(self, token_mint: str, size_sol: float) -> "Future[PaperTradeEstimate]":
        """Submit a non-blocking BUY quote to the background thread pool.

        Callers can hold the returned Future and resolve it later (e.g. after the
        candidate ranking window expires) to avoid blocking the event loop.
        """
        return self._thread_pool.submit(
            self.simulate_paper_buy,
            token_mint=token_mint,
            size_sol=size_sol,
        )

    def submit_swap_tx_future(
        self,
        token_mint: str,
        size_sol: float,
        *,
        prefer_jupiter: bool = False,
        strategy: str | None = None,
        source_program: str | None = None,
    ) -> "Future | None":
        """Pre-build a signed-ready Jupiter swap TX in the background (live mode only).

        Calls Jupiter v2 ``/order`` (quote + unsigned TX in one shot) during the
        candidate ranking window so the TX is ready when the candidate flushes.
        At fire time callers only need to sign + broadcast — no Jupiter HTTP calls.

        ``prefer_jupiter=True`` bypasses the native Pump-AMM builder — used by the
        main (mature-pair) lane because the native builder's math overflows on
        mature Pump-AMM pools (preflight 6023). Sniper stays on the native path
        for speed on ultra-fresh pools.

        Returns None if not in live mode or if the live executor is unavailable.
        """
        if not self.live or self._live_executor is None:
            return None
        return self._thread_pool.submit(
            self._live_executor.prefetch_swap_tx,
            token_mint=token_mint,
            size_sol=size_sol,
            prefer_jupiter=prefer_jupiter,
            strategy=strategy,
            source_program=source_program,
        )

    def evaluate_paper_entry_guard(
        self,
        token_mint: str,
        size_sol: float,
        *,
        min_roundtrip_ratio: float | None = None,
        max_price_impact_pct: float | None = None,
        pre_buy_estimate: Optional["PaperTradeEstimate"] = None,
        local_quote_engine: Optional[PumpAMMQuoteEngine] = None,
    ) -> PaperEntryGuardResult:
        """Validate that a paper entry is executable with acceptable round-trip quality.

        The check is purposely strict for paper mode: if buy/sell quotes are unavailable
        or immediate round-trip slippage is extreme, the entry is rejected.

        Parameters
        ----------
        pre_buy_estimate:
            Pre-fetched BUY quote (from :meth:`submit_buy_quote`).  When provided
            the blocking BUY Jupiter call is skipped entirely.
        local_quote_engine:
            If provided and reserves are fresh, the SELL quote is computed locally
            (0 µs, exact constant-product formula) instead of a Jupiter HTTP call.
            Eliminates the last ~300-500 ms Jupiter call from the entry critical path.
        """
        if self.live:
            if not bool(getattr(self.config, "live_entry_roundtrip_guard_enabled", False)):
                return PaperEntryGuardResult(allowed=True)
        elif not self.config.paper_entry_roundtrip_guard_enabled:
            return PaperEntryGuardResult(allowed=True)

        # BUY estimate priority:
        #   1. Pre-fetched Jupiter quote (already done in background, 0ms)
        #   2. Local AMM quote_buy (0µs, no HTTP — used when Jupiter is in backoff)
        #   3. Fresh Jupiter call (fallback only)
        buy_estimate = pre_buy_estimate
        if buy_estimate is None or not buy_estimate.success or buy_estimate.out_amount <= 0:
            if local_quote_engine is not None and local_quote_engine.has_reserves(token_mint):
                amount_lamports = int(size_sol * LAMPORTS_PER_SOL)
                raw_out = local_quote_engine.quote_buy(token_mint, amount_lamports)
                if raw_out and raw_out > 0:
                    fee_lam = _BASE_TX_FEE_LAMPORTS + int(
                        getattr(self.config, "priority_fee_lamports", 0)
                    )
                    buy_estimate = PaperTradeEstimate(
                        success=True,
                        input_mint=SOL_MINT,
                        output_mint=token_mint,
                        in_amount=amount_lamports,
                        out_amount=raw_out,
                        slippage_bps=int(self.config.default_slippage_bps),
                        priority_fee_lamports=int(getattr(self.config, "priority_fee_lamports", 0)),
                        base_fee_lamports=_BASE_TX_FEE_LAMPORTS,
                        total_network_fee_lamports=fee_lam,
                        price_impact_pct=0.0,
                    )
        if buy_estimate is None or not buy_estimate.success or buy_estimate.out_amount <= 0:
            buy_estimate = buy_estimate or self.simulate_paper_buy(
                token_mint=token_mint, size_sol=size_sol
            )

        if not buy_estimate.success or buy_estimate.out_amount <= 0:
            return PaperEntryGuardResult(
                allowed=False,
                reason=f"paper_entry_buy_quote_failed:{buy_estimate.error or 'unknown'}",
                buy_estimate=buy_estimate,
            )

        roundtrip_ratio_floor = (
            float(min_roundtrip_ratio)
            if min_roundtrip_ratio is not None
            else float(self.config.paper_entry_min_roundtrip_ratio)
        )
        price_impact_cap = (
            float(max_price_impact_pct)
            if max_price_impact_pct is not None
            else float(self.config.paper_entry_max_price_impact_pct)
        )

        buy_impact = abs(float(buy_estimate.price_impact_pct or 0.0))
        if buy_impact > price_impact_cap:
            return PaperEntryGuardResult(
                allowed=False,
                reason=(
                    f"paper_entry_price_impact_too_high:{buy_impact:.4f}>{price_impact_cap:.4f}"
                ),
                buy_estimate=buy_estimate,
            )

        # ── Sell quote: local AMM engine (0 µs) or Jupiter fallback (~300-500 ms) ──
        token_amount_out = int(buy_estimate.out_amount)
        sell_estimate: Optional[PaperTradeEstimate] = None
        local_sell_lamports: int = 0

        if local_quote_engine is not None and local_quote_engine.has_reserves(token_mint):
            raw_out = local_quote_engine.quote_sell(token_mint, token_amount_out)
            if raw_out and raw_out > 0:
                # Wrap in a PaperTradeEstimate so downstream code is unchanged
                fee_lam = buy_estimate.total_network_fee_lamports  # reuse buy fee for both legs
                sell_estimate = PaperTradeEstimate(
                    success=True,
                    input_mint=token_mint,
                    output_mint=SOL_MINT,
                    in_amount=token_amount_out,
                    out_amount=raw_out,
                    slippage_bps=int(self.config.default_slippage_bps),
                    priority_fee_lamports=buy_estimate.priority_fee_lamports,
                    base_fee_lamports=buy_estimate.base_fee_lamports,
                    total_network_fee_lamports=fee_lam,
                    price_impact_pct=0.0,  # local engine doesn't model price impact separately
                )
                local_sell_lamports = raw_out

        if sell_estimate is None:
            # Fallback: blocking Jupiter HTTP call
            sell_estimate = self.simulate_paper_sell(
                token_mint=token_mint,
                token_amount=token_amount_out,
            )

        if not sell_estimate.success or sell_estimate.out_amount <= 0:
            return PaperEntryGuardResult(
                allowed=False,
                reason=f"paper_entry_sell_quote_failed:{sell_estimate.error or 'unknown'}",
                buy_estimate=buy_estimate,
                sell_estimate=sell_estimate,
            )

        entry_fee_sol = float(buy_estimate.total_network_fee_lamports) / float(LAMPORTS_PER_SOL)
        exit_fee_sol = float(sell_estimate.total_network_fee_lamports) / float(LAMPORTS_PER_SOL)
        entry_cost_sol = float(size_sol) + entry_fee_sol
        immediate_exit_net_sol = max(
            0.0,
            float(sell_estimate.out_amount) / float(LAMPORTS_PER_SOL) - exit_fee_sol,
        )
        roundtrip_ratio = immediate_exit_net_sol / entry_cost_sol if entry_cost_sol > 0 else 0.0
        roundtrip_pnl_sol = immediate_exit_net_sol - entry_cost_sol

        if roundtrip_ratio < roundtrip_ratio_floor:
            return PaperEntryGuardResult(
                allowed=False,
                reason=(
                    "paper_entry_roundtrip_ratio_too_low:"
                    f"{roundtrip_ratio:.4f}<{roundtrip_ratio_floor:.4f}"
                ),
                buy_estimate=buy_estimate,
                sell_estimate=sell_estimate,
                entry_cost_sol=entry_cost_sol,
                immediate_exit_net_sol=immediate_exit_net_sol,
                roundtrip_ratio=roundtrip_ratio,
                roundtrip_pnl_sol=roundtrip_pnl_sol,
            )

        return PaperEntryGuardResult(
            allowed=True,
            buy_estimate=buy_estimate,
            sell_estimate=sell_estimate,
            entry_cost_sol=entry_cost_sol,
            immediate_exit_net_sol=immediate_exit_net_sol,
            roundtrip_ratio=roundtrip_ratio,
            roundtrip_pnl_sol=roundtrip_pnl_sol,
        )

    def _paper_quote_unavailable_error(self, now: datetime) -> str | None:
        if self._paper_quote_disabled_until is None:
            return None
        if now >= self._paper_quote_disabled_until:
            self._paper_quote_disabled_until = None
            return None
        return (
            "paper_quote_temporarily_unavailable_until:"
            f"{self._paper_quote_disabled_until.isoformat()}"
        )

    def _on_paper_quote_failure(self, exc: Exception) -> None:
        self._paper_quote_failure_count = min(self._paper_quote_failure_count + 1, 10)
        backoff_sec = min(300, 60 * (2 ** min(self._paper_quote_failure_count - 1, 3)))
        self._paper_quote_disabled_until = datetime.now(tz=timezone.utc) + timedelta(
            seconds=backoff_sec
        )
        error_text = str(exc)
        if error_text != self._paper_quote_last_error:
            logger.warning(
                "Paper Jupiter quote unavailable; pausing quote attempts for %ss (error=%s)",
                backoff_sec,
                error_text,
            )
            self._paper_quote_last_error = error_text

    def _on_paper_quote_success(self) -> None:
        self._paper_quote_failure_count = 0
        self._paper_quote_disabled_until = None
        self._paper_quote_last_error = None

    def _estimate_network_fee_lamports(
        self, account_keys: list[str]
    ) -> tuple[int, int, int, dict | None]:
        """Return ``(priority_fee, jito_tip, total_fee, raw_priority_payload)``."""
        priority_fee = max(int(self.config.priority_fee_lamports), 0)
        jito_tip = 0
        raw_priority_payload: dict | None = None

        if self._paper_broadcaster is not None:
            try:
                fee_info = self._paper_broadcaster.get_priority_fee_estimate(
                    account_keys=account_keys
                )
                priority_fee = max(
                    int(fee_info.get("priority_fee_lamports", priority_fee) or priority_fee),
                    0,
                )
                jito_tip = max(int(fee_info.get("jito_tip_lamports", 0) or 0), 0)
                raw_priority_payload = {"result": fee_info.get("raw")}
            except Exception as exc:  # noqa: BLE001
                logger.debug("Paper fee estimate fallback to config fee: %s", exc)

        total = _BASE_TX_FEE_LAMPORTS + priority_fee + jito_tip
        return priority_fee, jito_tip, total, raw_priority_payload

    def simulate_paper_buy(self, token_mint: str, size_sol: float) -> PaperTradeEstimate:
        """Simulate paper BUY with Jupiter quote + Helius fee estimate at signal time."""
        now = datetime.now(tz=timezone.utc)
        unavailable = self._paper_quote_unavailable_error(now)
        if unavailable is not None:
            return PaperTradeEstimate(
                success=False,
                input_mint=SOL_MINT,
                output_mint=token_mint,
                error=unavailable,
            )
        if self._paper_jupiter is None:
            return PaperTradeEstimate(
                success=False,
                input_mint=SOL_MINT,
                output_mint=token_mint,
                error="paper_jupiter_unavailable",
            )

        in_lamports = max(int(round(size_sol * LAMPORTS_PER_SOL)), 1)
        slippage = int(self.config.default_slippage_bps)
        try:
            quote = self._paper_jupiter.get_quote(
                input_mint=SOL_MINT,
                output_mint=token_mint,
                amount=in_lamports,
                slippage_bps=slippage,
            )
        except Exception as exc:  # noqa: BLE001
            self._on_paper_quote_failure(exc)
            return PaperTradeEstimate(
                success=False,
                input_mint=SOL_MINT,
                output_mint=token_mint,
                error=f"jupiter_quote_failed: {exc}",
            )
        self._on_paper_quote_success()

        priority, jito, total_fee, raw_priority = self._estimate_network_fee_lamports(
            account_keys=[SOL_MINT, token_mint],
        )
        return PaperTradeEstimate(
            success=True,
            input_mint=SOL_MINT,
            output_mint=token_mint,
            in_amount=quote.in_amount,
            out_amount=quote.out_amount,
            slippage_bps=slippage,
            priority_fee_lamports=priority,
            jito_tip_lamports=jito,
            total_network_fee_lamports=total_fee,
            raw_priority_fee=raw_priority,
            price_impact_pct=float(quote.price_impact_pct),
        )

    def simulate_paper_sell(self, token_mint: str, token_amount: int) -> PaperTradeEstimate:
        """Simulate paper SELL with Jupiter quote + Helius fee estimate at signal time."""
        now = datetime.now(tz=timezone.utc)
        unavailable = self._paper_quote_unavailable_error(now)
        if unavailable is not None:
            return PaperTradeEstimate(
                success=False,
                input_mint=token_mint,
                output_mint=SOL_MINT,
                error=unavailable,
            )
        if self._paper_jupiter is None:
            return PaperTradeEstimate(
                success=False,
                input_mint=token_mint,
                output_mint=SOL_MINT,
                error="paper_jupiter_unavailable",
            )

        in_amount = max(int(token_amount), 1)
        slippage = int(self.config.default_slippage_bps)
        try:
            quote = self._paper_jupiter.get_quote(
                input_mint=token_mint,
                output_mint=SOL_MINT,
                amount=in_amount,
                slippage_bps=slippage,
            )
        except Exception as exc:  # noqa: BLE001
            self._on_paper_quote_failure(exc)
            return PaperTradeEstimate(
                success=False,
                input_mint=token_mint,
                output_mint=SOL_MINT,
                error=f"jupiter_quote_failed: {exc}",
            )
        self._on_paper_quote_success()

        priority, jito, total_fee, raw_priority = self._estimate_network_fee_lamports(
            account_keys=[token_mint, SOL_MINT],
        )
        return PaperTradeEstimate(
            success=True,
            input_mint=token_mint,
            output_mint=SOL_MINT,
            in_amount=quote.in_amount,
            out_amount=quote.out_amount,
            slippage_bps=slippage,
            priority_fee_lamports=priority,
            jito_tip_lamports=jito,
            total_network_fee_lamports=total_fee,
            raw_priority_fee=raw_priority,
            price_impact_pct=float(quote.price_impact_pct),
        )

    # ------------------------------------------------------------------
    # Execution methods (live only – paper is handled by the engines)
    # ------------------------------------------------------------------

    def _kill_switch_blocks_buys(self) -> bool:
        """Return True when the live kill switch has disabled new buys (drain mode).

        The flag is positive (``live_allow_new_buys``, default True) so a bot
        running without the flag set keeps its existing behavior. Flipping it
        to False drains open positions via sells but blocks all new entries.
        """
        return bool(self.live and not bool(getattr(self.config, "live_allow_new_buys", True)))

    def execute_buy(
        self,
        token_mint: str,
        size_sol: float,
        current_exposure_sol: float = 0.0,
        open_position_count: int = 0,
        *,
        strategy: str | None = None,
        source_program: str | None = None,
    ) -> LiveTradeResult:
        """Execute a live BUY, or raise if not in live mode."""
        if not self.live or self._live_executor is None:
            raise RuntimeError("execute_buy called but live trading is not enabled")
        if self._kill_switch_blocks_buys():
            return LiveTradeResult(success=False, error="live_buys_disabled")
        return self._live_executor.execute_buy(
            token_mint=token_mint,
            size_sol=size_sol,
            current_exposure_sol=current_exposure_sol,
            open_position_count=open_position_count,
            strategy=strategy,
            source_program=source_program,
        )

    def execute_buy_prebuilt(
        self,
        token_mint: str,
        size_sol: float,
        prebuilt_tx: "SwapTransaction",
        current_exposure_sol: float = 0.0,
        open_position_count: int = 0,
    ) -> LiveTradeResult:
        """Fire a pre-built swap TX: sign + broadcast only (~50-150 ms total)."""
        if not self.live or self._live_executor is None:
            raise RuntimeError("execute_buy_prebuilt called but live trading is not enabled")
        if self._kill_switch_blocks_buys():
            return LiveTradeResult(success=False, error="live_buys_disabled")
        return self._live_executor.execute_buy_prebuilt(
            token_mint=token_mint,
            size_sol=size_sol,
            prebuilt_tx=prebuilt_tx,
            current_exposure_sol=current_exposure_sol,
            open_position_count=open_position_count,
        )

    def execute_sell(
        self,
        token_mint: str,
        token_amount: int,
        current_exposure_sol: float = 0.0,
        open_position_count: int = 0,
        close_token_account: bool = False,
        *,
        prefer_jupiter: bool = False,
        strategy: str | None = None,
        source_program: str | None = None,
    ) -> LiveTradeResult:
        """Execute a live SELL, or raise if not in live mode."""
        if not self.live or self._live_executor is None:
            raise RuntimeError("execute_sell called but live trading is not enabled")
        return self._live_executor.execute_sell(
            token_mint=token_mint,
            token_amount=token_amount,
            current_exposure_sol=current_exposure_sol,
            open_position_count=open_position_count,
            close_token_account=close_token_account,
            prefer_jupiter=prefer_jupiter,
            strategy=strategy,
            source_program=source_program,
        )

    def execute_sell_prebuilt(
        self,
        token_mint: str,
        token_amount: int,
        prebuilt_tx: "SwapTransaction",
        current_exposure_sol: float = 0.0,
        close_token_account: bool = False,
        *,
        prefer_jupiter: bool = False,
        strategy: str | None = None,
        source_program: str | None = None,
    ) -> LiveTradeResult:
        """Fire a pre-built sell TX: sign + broadcast only (~50-150ms total)."""
        if not self.live or self._live_executor is None:
            raise RuntimeError("execute_sell_prebuilt called but live trading is not enabled")
        return self._live_executor.execute_sell_prebuilt(
            token_mint=token_mint,
            token_amount=token_amount,
            prebuilt_tx=prebuilt_tx,
            current_exposure_sol=current_exposure_sol,
            close_token_account=close_token_account,
            prefer_jupiter=prefer_jupiter,
            strategy=strategy,
            source_program=source_program,
        )

    async def execute_buy_prebuilt_async(
        self,
        token_mint: str,
        size_sol: float,
        prebuilt_tx: "SwapTransaction",
        current_exposure_sol: float = 0.0,
        open_position_count: int = 0,
    ) -> LiveTradeResult:
        """Async sign + broadcast of a pre-built BUY TX without blocking the event loop."""
        if not self.live or self._live_executor is None:
            raise RuntimeError("execute_buy_prebuilt_async called but live trading is not enabled")
        if self._kill_switch_blocks_buys():
            return LiveTradeResult(success=False, error="live_buys_disabled")
        return await self._live_executor.execute_buy_prebuilt_async(
            token_mint=token_mint,
            size_sol=size_sol,
            prebuilt_tx=prebuilt_tx,
            current_exposure_sol=current_exposure_sol,
            open_position_count=open_position_count,
        )

    async def execute_buy_async(
        self,
        token_mint: str,
        size_sol: float,
        current_exposure_sol: float = 0.0,
        open_position_count: int = 0,
        *,
        prefer_jupiter: bool = False,
        strategy: str | None = None,
        source_program: str | None = None,
    ) -> LiveTradeResult:
        """Async BUY without a pre-built TX: get_order_async → sign → broadcast_async.

        Does not consume a thread-pool slot — uses httpx.AsyncClient directly.
        Used as the fallback path in entry_engine when no prebuilt TX is available.
        """
        if not self.live or self._live_executor is None:
            raise RuntimeError("execute_buy_async called but live trading is not enabled")
        if self._kill_switch_blocks_buys():
            return LiveTradeResult(success=False, error="live_buys_disabled")
        return await self._live_executor.execute_buy_async(
            token_mint=token_mint,
            size_sol=size_sol,
            current_exposure_sol=current_exposure_sol,
            open_position_count=open_position_count,
            prefer_jupiter=prefer_jupiter,
            strategy=strategy,
            source_program=source_program,
        )

    async def execute_sell_prebuilt_async(
        self,
        token_mint: str,
        token_amount: int,
        prebuilt_tx: "SwapTransaction",
        current_exposure_sol: float = 0.0,
        close_token_account: bool = False,
        *,
        prefer_jupiter: bool = False,
        strategy: str | None = None,
        source_program: str | None = None,
    ) -> LiveTradeResult:
        """Async sign + broadcast of a pre-built SELL TX without blocking the event loop."""
        if not self.live or self._live_executor is None:
            raise RuntimeError("execute_sell_prebuilt_async called but live trading is not enabled")
        return await self._live_executor.execute_sell_prebuilt_async(
            token_mint=token_mint,
            token_amount=token_amount,
            prebuilt_tx=prebuilt_tx,
            current_exposure_sol=current_exposure_sol,
            close_token_account=close_token_account,
            prefer_jupiter=prefer_jupiter,
            strategy=strategy,
            source_program=source_program,
        )
