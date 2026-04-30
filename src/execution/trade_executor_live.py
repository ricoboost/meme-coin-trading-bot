"""Live trade executor using Jupiter + Helius.

This module orchestrates real on-chain swap execution:

1. Obtain a Jupiter quote for the desired swap.
2. Build an unsigned swap transaction via Jupiter.
3. Sign the transaction locally with :class:`LocalSigner`.
4. Broadcast and confirm via :class:`Broadcaster` (Helius RPC).

All safeguard checks are enforced **before** any transaction is built.

This file is ~5.5k lines because Pump.fun and Pump-AMM each need their
own native-instruction transaction builders (Jupiter cannot route every
mint, especially fresh launches), and the Sender-mode broadcast path
requires a parallel ``_async`` variant of every transaction-construction
method. The bulk of the size is duplication between sync and async
paths, not unique business logic.

File table of contents (approximate line ranges):

  ~50- ~115   Module dataclasses: LiveExecutionError, LiveTradeResult,
              LiveFillReconciliation, LiveFeePlan, LiveOrderPolicy
  ~118- ~250  LiveTradeExecutor.__init__ + Pump-pool migration tracking
              (dead-pool detection, AMM migration error inference)
  ~250- ~635  Mint/program resolution + preflight simulation
              (Jupiter price-impact gates, native Pump price-impact gates,
              Token-2022 detection)
  ~636- ~970  Honeypot roundtrip simulation, async preflight, build-retry
              on retriable preflight errors
  ~970-~1390  Pure helpers: token-balance probes, ATA derivation, instruction
              builders (createATA, syncNative, closeATA), buy/sell wallet
              balance enforcement
  ~1390-~2360 Native Pump AMM transaction builders — buy and sell, sync
              and async variants. Bulk of the file.
  ~2360-~2610 Reconciliation post-execution (parse confirmed tx, compute
              actual filled amount + tip + slippage)
  ~2610-~2950 Fee plan + slippage + order-policy resolution + Jupiter swap
              instructions with Sender/Bundle fallback
  ~2950-~3970 Sender-mode rebuild path: lift Jupiter swap into a
              compute-budget-correct transaction the Helius Sender will
              accept; custom swap-tx construction
  ~3970-~end  Public API: prefetch_swap_tx, prefetch_sell_tx,
              execute_buy_prebuilt_async, execute_sell_prebuilt_async
"""

from __future__ import annotations

import base64
import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.bot.config import BotConfig, effective_max_price_impact_pct
from src.execution.broadcaster import BroadcastError, BroadcastResult, Broadcaster
from src.execution.jupiter_client import (
    LAMPORTS_PER_SOL,
    SOL_MINT,
    JupiterClient,
    JupiterError,
    SwapTransaction,
)
from src.execution.signer import LocalSigner, SignerError
from src.strategy.local_quote import PumpAMMNativePoolState, PumpAMMQuoteEngine

logger = logging.getLogger(__name__)

_SIGNATURE_FEE_LAMPORTS = 5_000
_TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
_TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
_KNOWN_TOKEN_PROGRAMS = frozenset({_TOKEN_PROGRAM_ID, _TOKEN_2022_PROGRAM_ID})
_ASSOCIATED_TOKEN_PROGRAM_ID = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
_SYSTEM_PROGRAM_ID = "11111111111111111111111111111111"
_PUMP_AMM_PROGRAM_ID = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
_PUMP_AMM_BUY_DISC = bytes([102, 6, 61, 18, 1, 218, 235, 234])
_PUMP_AMM_SELL_DISC = bytes([51, 230, 133, 164, 1, 127, 131, 173])
_PUMP_AMM_EVENT_AUTHORITY_SEED = b"__event_authority"
_PUMP_AMM_GLOBAL_VOLUME_ACCUMULATOR_SEED = b"global_volume_accumulator"
_PUMP_AMM_USER_VOLUME_ACCUMULATOR_SEED = b"user_volume_accumulator"


class LiveExecutionError(Exception):
    """Raised when a live trade execution fails."""


@dataclass(frozen=True)
class LiveTradeResult:
    """Outcome of a live trade attempt."""

    success: bool
    signature: Optional[str] = None
    in_amount: int = 0
    out_amount: int = 0
    slot: int | None = None
    error: Optional[str] = None
    reconciliation_error: Optional[str] = None
    reconciliation: "LiveFillReconciliation | None" = None
    latency_trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LiveFillReconciliation:
    """Exact confirmed-fill reconciliation from getTransaction metadata."""

    signature: str
    wallet_pubkey: str
    token_mint: str
    slot: int | None = None
    fee_lamports: int = 0
    wallet_pre_lamports: int = 0
    wallet_post_lamports: int = 0
    wallet_delta_lamports: int = 0
    token_pre_raw: int = 0
    token_post_raw: int = 0
    token_delta_raw: int = 0
    token_decimals: int = 0
    expected_in_amount: int = 0
    expected_out_amount: int = 0
    actual_output_amount: int = 0
    quote_out_amount_diff: int = 0
    tip_lamports: int = 0
    transaction_error: Any = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LiveFeePlan:
    """Resolved live-network fee plan."""

    priority_fee_lamports: int
    jito_tip_lamports: int
    raw_priority_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class LiveOrderPolicy:
    """Resolved Jupiter order policy for one side of a trade."""

    slippage_bps: int | None
    priority_fee_lamports: int
    jito_tip_lamports: int
    broadcast_fee_type: str | None = None
    raw_priority_payload: dict[str, Any] | None = None
    strategy: str | None = None
    source_program: str | None = None


class LiveTradeExecutor:
    """Execute real swaps on Solana via Jupiter and Helius.

    Parameters
    ----------
    config:
        Bot configuration (supplies limits, slippage, priority fee, etc.).
    signer:
        A validated :class:`LocalSigner`.
    jupiter:
        A :class:`JupiterClient` instance.
    broadcaster:
        A :class:`Broadcaster` instance.
    """

    def __init__(
        self,
        config: BotConfig,
        signer: LocalSigner,
        jupiter: JupiterClient,
        broadcaster: Broadcaster,
        local_quote_engine: PumpAMMQuoteEngine | None = None,
    ) -> None:
        self.config = config
        self.signer = signer
        self.jupiter = jupiter
        self.broadcaster = broadcaster
        self.local_quote_engine = local_quote_engine
        self._mint_token_program_cache: dict[str, str] = {SOL_MINT: _TOKEN_PROGRAM_ID}
        # Mints whose native Pump AMM accounts are no longer valid (graduated
        # to Raydium/Meteora mid-hold, manually migrated, etc.). Detected via
        # InstructionError containing 'MissingAccount' on the pump_sell_ix.
        # Once marked, native-path builders return None so the router falls
        # through to Jupiter /order, which knows the current venue.
        self._pump_dead_mints: set[str] = set()
        self._pump_dead_lock = threading.Lock()
        # Pool liveness probe cache: pool_pubkey -> (monotonic_ts, is_live).
        # Reused across sell attempts within TTL so we don't pay an RPC per TX build.
        self._pool_liveness_cache: dict[str, tuple[float, bool]] = {}
        self._pool_liveness_lock = threading.Lock()

    def _is_pump_dead(self, token_mint: str) -> bool:
        with self._pump_dead_lock:
            return token_mint in self._pump_dead_mints

    def _mark_pump_dead(self, token_mint: str, reason: str) -> None:
        with self._pump_dead_lock:
            if token_mint in self._pump_dead_mints:
                return
            self._pump_dead_mints.add(token_mint)
        logger.warning(
            "🟠 native Pump AMM marked dead for %s (reason=%s) — Jupiter routing forced",
            token_mint[:12],
            reason,
        )

    def _maybe_mark_pump_dead(
        self,
        *,
        trace: dict[str, Any] | None,
        error: str | None,
        token_mint: str | None,
    ) -> None:
        """Mark a mint's native Pump path dead when a native sell fails with a
        migration-class error (``MissingAccount`` / ``AccountNotFound``).

        Only trips when both conditions hold:
        - ``error`` string carries a migration-class phrase.
        - ``trace['venue_path']`` shows the failing TX used the native Pump lane.
        """
        if not token_mint:
            return
        if not self._error_indicates_pump_migration(error):
            return
        venue = str((trace or {}).get("venue_path") or "")
        if not venue.startswith("pump_amm_native"):
            return
        self._mark_pump_dead(token_mint, reason=f"broadcast_error_migration:{str(error)[:80]}")

    def _probe_pump_pool_live(
        self,
        *,
        pool_pubkey: str,
        token_mint: str,
        trace: dict[str, Any] | None = None,
    ) -> bool:
        """Return False iff the Pump pool state account no longer exists on-chain.

        Called before building the first sell TX on a mint. Results are cached
        per pool for ``live_pool_liveness_probe_ttl_sec`` so a burst of exit
        ladder attempts costs one RPC round-trip, not N.

        Failure mode intentionally permissive: any RPC error returns True, so
        we fall through to the normal sell path + circuit breaker instead of
        dropping trades because a probe RPC flaked.
        """
        if not pool_pubkey:
            return True
        if not bool(getattr(self.config, "live_pool_liveness_probe_enabled", True)):
            return True
        ttl = max(
            float(getattr(self.config, "live_pool_liveness_probe_ttl_sec", 30.0) or 30.0),
            1.0,
        )
        now = time.monotonic()
        with self._pool_liveness_lock:
            entry = self._pool_liveness_cache.get(pool_pubkey)
            if entry is not None and (now - entry[0]) < ttl:
                is_live = entry[1]
                if trace is not None:
                    trace["pool_liveness_cached"] = True
                    trace["pool_liveness_alive"] = is_live
                return is_live
        try:
            owner = self.broadcaster.get_account_owner(pool_pubkey, commitment="confirmed")
        except Exception as exc:  # noqa: BLE001
            if trace is not None:
                trace["pool_liveness_probe_error"] = str(exc)[:80]
            logger.debug("pool liveness probe error for %s: %s", pool_pubkey[:12], exc)
            return True
        is_live = owner is not None
        with self._pool_liveness_lock:
            self._pool_liveness_cache[pool_pubkey] = (now, is_live)
        if trace is not None:
            trace["pool_liveness_cached"] = False
            trace["pool_liveness_alive"] = is_live
            trace["pool_liveness_owner"] = owner
        if not is_live:
            self._mark_pump_dead(token_mint, reason="pool_state_missing_pre_sell")
        return is_live

    @staticmethod
    def _error_indicates_pump_migration(error: str | None) -> bool:
        """Return True if a broadcast error string suggests pool-account drift.

        Two failure shapes matter:
        - ``InstructionError: [N, 'MissingAccount']`` — pool moved; accounts gone.
        - ``AccountNotFound`` / ``ProgramAccountNotFound`` — same class, different
          phrasing across RPC providers.
        """
        if not error:
            return False
        text = str(error)
        return (
            "MissingAccount" in text
            or "AccountNotFound" in text
            or "ProgramAccountNotFound" in text
        )

    def _resolve_mint_token_program(self, mint_pubkey: str) -> str:
        """Return the token program that owns `mint_pubkey` (SPL Token or Token-2022).

        Cached in-memory: mint program assignment is immutable on Solana.
        Falls back to SPL Token if the RPC call fails, with a warning. Callers
        that cannot tolerate a wrong guess should handle the failure upstream.
        """
        key = str(mint_pubkey)
        cached = self._mint_token_program_cache.get(key)
        if cached is not None:
            return cached
        try:
            owner = self.broadcaster.get_account_owner(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mint program lookup failed for %s: %s", key, exc)
            owner = None
        if owner in _KNOWN_TOKEN_PROGRAMS:
            resolved = str(owner)
        else:
            if owner:
                logger.warning(
                    "mint %s owned by unknown program %s; defaulting to SPL Token",
                    key,
                    owner,
                )
            resolved = _TOKEN_PROGRAM_ID
        self._mint_token_program_cache[key] = resolved
        return resolved

    def resolve_mint_token_program_strict(self, mint_pubkey: str) -> str | None:
        """Resolve token program with one retry; return None only on RPC unavailable.

        Distinguishes "RPC did not answer" from "RPC answered with an unknown
        program" — the former returns None (fail-closed at callers), the latter
        returns the observed owner string so callers can decide.
        """
        key = str(mint_pubkey)
        cached = self._mint_token_program_cache.get(key)
        if cached is not None:
            return cached
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                owner = self.broadcaster.get_account_owner(key)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt == 0:
                    time.sleep(0.2)
                    continue
                logger.warning("mint program lookup failed twice for %s: %s", key, exc)
                return None
            if owner in _KNOWN_TOKEN_PROGRAMS:
                resolved = str(owner)
                self._mint_token_program_cache[key] = resolved
                return resolved
            return str(owner) if owner else ""
        return None

    def resolve_mint_program_and_extensions_strict(
        self, mint_pubkey: str
    ) -> tuple[str, list[str]] | None:
        """Return ``(owner_program, extensions)`` with one retry; None on RPC fail.

        Extensions are Token-2022 extension names (empty list for classic SPL
        mints). Callers use this to accept metadata-only Token-2022 tokens
        while rejecting ones with risky extensions.
        """
        key = str(mint_pubkey)
        for attempt in range(2):
            try:
                result = self.broadcaster.get_mint_extensions(key)
            except Exception as exc:  # noqa: BLE001
                if attempt == 0:
                    time.sleep(0.2)
                    continue
                logger.warning("mint extensions lookup failed twice for %s: %s", key, exc)
                return None
            if result is None:
                if attempt == 0:
                    time.sleep(0.2)
                    continue
                return None
            owner, extensions = result
            if owner in _KNOWN_TOKEN_PROGRAMS:
                self._mint_token_program_cache[key] = str(owner)
            return str(owner), list(extensions)
        return None

    def _preflight_simulate_sell(
        self,
        signed_tx: bytes,
        *,
        token_mint: str,
        trace: dict | None = None,
    ) -> str | None:
        """Optional simulateTransaction before broadcast. Returns error string or None.

        Gated by ``live_preflight_simulate`` config (default False). When enabled,
        a failing simulation short-circuits before we pay the priority fee + tip.
        Simulation failures that look transient (network errors) are swallowed —
        we'd rather let the real broadcast decide than block a valid sell.
        """
        if not bool(getattr(self.config, "live_preflight_simulate", False)):
            return None
        try:
            sim = self.broadcaster.simulate_transaction(signed_tx)
        except Exception as exc:  # noqa: BLE001
            logger.debug("preflight simulate network error for %s: %s", token_mint[:12], exc)
            if trace is not None:
                trace["preflight_sim_error"] = f"network: {exc}"
            return None
        err = sim.get("err")
        if err is None:
            if trace is not None:
                trace["preflight_sim_err"] = None
            return None
        logs = sim.get("logs") or []
        log_tail = logs[-6:] if isinstance(logs, list) else []
        if trace is not None:
            trace["preflight_sim_err"] = str(err)
            trace["preflight_sim_log_tail"] = log_tail
        logger.warning(
            "preflight simulate FAILED for %s: err=%s last_logs=%s",
            token_mint[:12],
            err,
            log_tail,
        )
        self._maybe_mark_pump_dead(trace=trace, error=str(err), token_mint=token_mint)
        return f"preflight_failed: {err}"

    def _preflight_simulate_buy(
        self,
        signed_tx: bytes,
        *,
        token_mint: str,
        trace: dict | None = None,
    ) -> str | None:
        """Optional simulateTransaction before broadcast for BUY. Returns error string or None.

        Gated by ``live_preflight_simulate`` config (default False). Catches
        deterministic reverts (Jupiter Route CPI into a broken pool, missing
        accounts, Token-2022 mismatches) before we pay the priority fee + tip.
        Simulation network errors are swallowed — we'd rather let the real
        broadcast decide than block a valid buy on transient RPC failure.
        """
        if not bool(getattr(self.config, "live_preflight_simulate", False)):
            return None
        try:
            sim = self.broadcaster.simulate_transaction(signed_tx)
        except Exception as exc:  # noqa: BLE001
            logger.debug("preflight buy simulate network error for %s: %s", token_mint[:12], exc)
            if trace is not None:
                trace["preflight_sim_error"] = f"network: {exc}"
            return None
        err = sim.get("err")
        if err is None:
            if trace is not None:
                trace["preflight_sim_err"] = None
            return None
        logs = sim.get("logs") or []
        log_tail = logs[-6:] if isinstance(logs, list) else []
        if trace is not None:
            trace["preflight_sim_err"] = str(err)
            trace["preflight_sim_log_tail"] = log_tail
        logger.warning(
            "preflight buy simulate FAILED for %s: err=%s last_logs=%s",
            token_mint[:12],
            err,
            log_tail,
        )
        self._maybe_mark_pump_dead(trace=trace, error=str(err), token_mint=token_mint)
        return f"preflight_failed: {err}"

    def _check_jupiter_buy_price_impact(
        self,
        *,
        token_mint: str,
        price_impact_pct: float | None,
        source: str,
        trace: dict[str, Any] | None = None,
        order_policy: "LiveOrderPolicy | None" = None,
    ) -> None:
        """Reject buys when Jupiter reports an insane price impact.

        Jupiter's v2 API occasionally returns corrupt quotes on fresh/stale
        pools — e.g. ``priceImpactPct = -3.00`` (claiming the user gains 300%).
        Such quotes always revert on-chain with an ``IncorrectProgramId`` or
        similar error in RouteV2. Kill the buy before we sign.

        Raises ``LiveExecutionError`` when ``abs(price_impact_pct)`` exceeds
        the configured threshold.
        """
        if price_impact_pct is None:
            return
        if self._should_bypass_jupiter_order_impact(source=source, order_policy=order_policy):
            if trace is not None:
                trace["jupiter_price_impact_bypassed"] = float(price_impact_pct)
                trace["jupiter_price_impact_source"] = source
            return
        threshold = effective_max_price_impact_pct(
            self.config,
            strategy_id=(order_policy.strategy if order_policy is not None else None),
            source_program=(order_policy.source_program if order_policy is not None else None),
            live_mode=True,
        )
        if threshold <= 0.0:
            return
        try:
            impact = float(price_impact_pct)
        except (TypeError, ValueError):
            return
        if abs(impact) <= threshold:
            return
        if trace is not None:
            trace["jupiter_price_impact_rejected"] = impact
            trace["jupiter_price_impact_source"] = source
        logger.warning(
            "🟠 Jupiter price-impact guard tripped for %s: impact=%.4f%% threshold=%.4f%% source=%s",
            token_mint[:12],
            impact * 100,
            threshold * 100,
            source,
        )
        raise LiveExecutionError(
            f"jupiter_price_impact_out_of_range: {impact:.4f} "
            f"(threshold={threshold:.4f}, source={source})"
        )

    def _check_jupiter_sell_price_impact(
        self,
        *,
        token_mint: str,
        price_impact_pct: float | None,
        source: str,
        trace: dict[str, Any] | None = None,
        order_policy: "LiveOrderPolicy | None" = None,
    ) -> None:
        """Reject sells when Jupiter reports a pathological price impact.

        A rugged or broken pool returns quotes like ``priceImpactPct = -100.0``
        (i.e., -10000%) or similarly extreme values. Broadcasting these burns
        priority fees forever. Abort instead — the exit_engine's circuit
        breaker treats ``jupiter_sell_impact_out_of_range`` as a no-route
        signal and flags the mint permanently stuck after 2 hits.
        """
        if price_impact_pct is None:
            return
        if self._should_bypass_jupiter_order_impact(source=source, order_policy=order_policy):
            if trace is not None:
                trace["jupiter_sell_impact_bypassed"] = float(price_impact_pct)
                trace["jupiter_sell_impact_source"] = source
            return
        threshold = float(getattr(self.config, "live_sell_max_price_impact_pct", 0.0) or 0.0)
        if threshold <= 0.0:
            return
        try:
            impact = float(price_impact_pct)
        except (TypeError, ValueError):
            return
        if abs(impact) <= threshold:
            return
        if trace is not None:
            trace["jupiter_sell_impact_rejected"] = impact
            trace["jupiter_sell_impact_source"] = source
        logger.warning(
            "🟠 Jupiter sell price-impact guard tripped for %s: impact=%.4f%% threshold=%.4f%% source=%s",
            token_mint[:12],
            impact * 100,
            threshold * 100,
            source,
        )
        raise LiveExecutionError(
            f"jupiter_sell_impact_out_of_range: {impact:.4f} "
            f"(threshold={threshold:.4f}, source={source})"
        )

    def _should_bypass_jupiter_order_impact(
        self,
        *,
        source: str,
        order_policy: "LiveOrderPolicy | None",
    ) -> bool:
        """Bypass Jupiter impact guard on sniper+PUMP_FUN.

        Covers both ``source="jupiter_order"`` (v2/order reports bogus impact
        like -167% on ultra-fresh bonding-curve tokens) and
        ``source="metis_quote"`` (legitimate high impact on thin fresh pools
        was blocking TP fires at +26%/+39% and forcing stop-outs at -60%
        slippage). Rug defense is layered elsewhere (holder cap, anti-wash,
        age gates, pump_dead_marker).
        """
        if source not in ("jupiter_order", "metis_quote"):
            return False
        if order_policy is None:
            return False
        if (order_policy.strategy or "").lower() != "sniper":
            return False
        if (order_policy.source_program or "").upper() != "PUMP_FUN":
            return False
        return True

    def _check_native_pump_price_impact(
        self,
        *,
        token_mint: str,
        side: str,  # "buy" or "sell"
        amount_in: int,
        amount_out: int,
        reserve_in: int,
        reserve_out: int,
        trace: dict[str, Any] | None = None,
        order_policy: "LiveOrderPolicy | None" = None,
    ) -> None:
        """Reject native Pump-AMM swaps when the quote implies a pathological
        price impact.

        Computed from the local constant-product reserves against the
        pre-slippage quoted amount_out:

            spot_rate = reserve_out / reserve_in
            exec_rate = amount_out / amount_in
            impact    = 1 - (exec_rate / spot_rate)

        For honest pools, impact tracks ``amount_in / (amount_in + reserve_in)``
        and is small. For drained/rugged pools or reserves that have been
        poisoned, impact approaches 1.0. Abort before we sign.

        Threshold is ``live_entry_max_price_impact_pct`` for buys,
        ``live_sell_max_price_impact_pct`` for sells. Passing 0.0 disables the
        check (matches the Jupiter-guard convention).
        """
        if side == "buy":
            threshold = effective_max_price_impact_pct(
                self.config,
                strategy_id=(order_policy.strategy if order_policy is not None else None),
                source_program=(order_policy.source_program if order_policy is not None else None),
                live_mode=True,
            )
            err_code = "native_pump_buy_impact_out_of_range"
        else:
            threshold = float(getattr(self.config, "live_sell_max_price_impact_pct", 0.0) or 0.0)
            err_code = "native_pump_sell_impact_out_of_range"
        if threshold <= 0.0:
            return
        if amount_in <= 0 or amount_out <= 0 or reserve_in <= 0 or reserve_out <= 0:
            return
        try:
            exec_rate = float(amount_out) / float(amount_in)
            spot_rate = float(reserve_out) / float(reserve_in)
            if spot_rate <= 0.0:
                return
            impact = 1.0 - (exec_rate / spot_rate)
        except (ZeroDivisionError, ValueError):
            return
        if trace is not None:
            trace["native_pump_impact_pct"] = impact
        if abs(impact) <= threshold:
            return
        if trace is not None:
            trace["native_pump_impact_rejected"] = impact
        logger.warning(
            "🟠 Native Pump-AMM %s impact guard tripped for %s: impact=%.4f%% threshold=%.4f%%",
            side,
            token_mint[:12],
            impact * 100,
            threshold * 100,
        )
        raise LiveExecutionError(f"{err_code}: {impact:.4f} (threshold={threshold:.4f})")

    def simulate_honeypot_roundtrip(
        self,
        *,
        token_mint: str,
        size_lamports: int,
        trace: dict[str, Any] | None = None,
    ) -> str | None:
        """Atomic buy→sell bundle simulation. Returns honeypot error or None.

        Returns values:
          - ``None``  : simulation succeeded, OR the detection surface does not
                        apply (native Pump-AMM state not cached, no quote).
                        Caller should allow the trade through.
          - ``str``   : failure signal. Caller should reject the entry. Two
                        shapes: ``"honeypot_sell_reverts: ..."`` (rug) and
                        ``"rpc_unavailable: ..."`` (we retried once and the
                        RPC still did not answer — fail-closed).
        """
        if not bool(getattr(self.config, "live_entry_honeypot_sim_enabled", False)):
            return None
        if self.local_quote_engine is None:
            return None
        trace_payload = trace if trace is not None else {}
        state = self.local_quote_engine.get_native_pool_state(token_mint)
        if state is None or not state.token_is_base_quote_is_wsol:
            trace_payload["honeypot_sim_skipped"] = "no_native_state"
            return None
        expected_base_out = self._native_pump_buy_out_amount(token_mint, size_lamports)
        if expected_base_out <= 0:
            trace_payload["honeypot_sim_skipped"] = "no_quote"
            return None
        # Shrink the sell to a fraction of the buy's expected output so a
        # temporary sell-side liquidity shortfall inside the same simulated
        # slot (vs. a real rug) doesn't flag a false positive.
        fraction_bps = max(
            1,
            min(
                10_000,
                int(getattr(self.config, "live_entry_honeypot_sim_fraction_bps", 500) or 500),
            ),
        )
        sell_base_in = max(1, (expected_base_out * fraction_bps) // 10_000)
        buy_policy = self._resolve_order_policy(side="buy")
        sell_policy = self._resolve_order_policy(side="sell")
        try:
            buy_tx = self._build_native_pump_amm_buy_tx(
                token_mint=token_mint,
                amount_lamports=int(size_lamports),
                order_policy=buy_policy,
                trace={"_honeypot_sim": "buy"},
            )
            sell_tx = self._build_native_pump_amm_sell_tx(
                token_mint=token_mint,
                token_amount=int(sell_base_in),
                order_policy=sell_policy,
                close_input_token_account=False,
                trace={"_honeypot_sim": "sell"},
            )
        except LiveExecutionError as exc:
            trace_payload["honeypot_sim_build_failed"] = str(exc)
            return None
        except Exception as exc:  # noqa: BLE001
            trace_payload["honeypot_sim_build_error"] = repr(exc)
            return None
        result = None
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                result = self.broadcaster.simulate_bundle(
                    [buy_tx.raw_transaction, sell_tx.raw_transaction]
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt == 0:
                    time.sleep(0.2)
                    continue
        if result is None:
            err_str = str(last_exc) if last_exc else "unknown"
            trace_payload["honeypot_sim_rpc_error"] = err_str[:200]
            # -32601 = method not found (provider doesn't expose simulateBundle)
            # -32602 = invalid params (our bug or provider-side schema drift)
            # Both are infrastructure issues, NOT transient — fail-open, don't
            # block every candidate.
            if "-32601" in err_str or "-32602" in err_str:
                trace_payload["honeypot_sim_skipped"] = "rpc_not_supported"
                logger.warning(
                    "honeypot_sim: simulateBundle not usable on this RPC (%s) — failing open",
                    err_str[:160],
                )
                return None
            logger.debug(
                "honeypot_sim: simulateBundle transient error for %s: %s",
                token_mint[:12],
                last_exc,
            )
            return f"rpc_unavailable: {last_exc}" if last_exc else "rpc_unavailable"
        summary = str(result.get("summary") or "").lower()
        per_tx = result.get("transactionResults") or []
        if summary == "succeeded" and not any(tx.get("err") for tx in per_tx):
            trace_payload["honeypot_sim_ok"] = True
            return None
        buy_err = per_tx[0].get("err") if len(per_tx) > 0 else None
        sell_err = per_tx[1].get("err") if len(per_tx) > 1 else None
        trace_payload["honeypot_sim_summary"] = summary
        trace_payload["honeypot_sim_buy_err"] = str(buy_err) if buy_err else None
        trace_payload["honeypot_sim_sell_err"] = str(sell_err) if sell_err else None
        # Buy failing is not honeypot-specific (broken pool, wrong route) —
        # the normal preflight guard catches that. Only flag when the SELL
        # fails, which is the canonical honeypot signal.
        if sell_err is not None and buy_err is None:
            logger.warning(
                "🟠 Honeypot simulation tripped for %s: sell_err=%s",
                token_mint[:12],
                sell_err,
            )
            return f"honeypot_sell_reverts: {sell_err}"
        return None

    async def _preflight_simulate_buy_async(
        self,
        signed_tx: bytes,
        *,
        token_mint: str,
        trace: dict | None = None,
    ) -> str | None:
        """Async variant of :meth:`_preflight_simulate_buy` — non-blocking HTTP."""
        if not bool(getattr(self.config, "live_preflight_simulate", False)):
            return None
        try:
            sim = await self.broadcaster.simulate_transaction_async(signed_tx)
        except Exception as exc:  # noqa: BLE001
            logger.debug("preflight buy simulate network error for %s: %s", token_mint[:12], exc)
            if trace is not None:
                trace["preflight_sim_error"] = f"network: {exc}"
            return None
        err = sim.get("err")
        if err is None:
            if trace is not None:
                trace["preflight_sim_err"] = None
            return None
        logs = sim.get("logs") or []
        log_tail = logs[-6:] if isinstance(logs, list) else []
        if trace is not None:
            trace["preflight_sim_err"] = str(err)
            trace["preflight_sim_log_tail"] = log_tail
        logger.warning(
            "preflight buy simulate FAILED for %s: err=%s last_logs=%s",
            token_mint[:12],
            err,
            log_tail,
        )
        self._maybe_mark_pump_dead(trace=trace, error=str(err), token_mint=token_mint)
        return f"preflight_failed: {err}"

    @staticmethod
    def _is_retriable_tx_build_error(preflight_error: str) -> bool:
        """Preflight errors caused by a bad prebuilt-tx structure (not pool state).

        These are fixable by rebuilding the swap via the swap-instructions
        fallback path, which rotates useSharedAccounts / skipUserAccountsRpcCalls
        variants. Pool-state errors (Custom:6xxx, slippage) are NOT retried here.
        """
        if not preflight_error:
            return False
        needles = (
            "PrivilegeEscalation",
            "InvalidSeeds",
            "ProgramFailedToComplete",
            "IncorrectProgramId",
            "MissingAccount",
        )
        return any(n in preflight_error for n in needles)

    async def _rebuild_and_retry_buy_async(
        self,
        *,
        token_mint: str,
        size_sol: float,
        original_error: str,
        parent_trace: dict[str, Any],
    ) -> "LiveTradeResult | None":
        """Rebuild a BUY via swap-instructions + retry once after preflight fail.

        Used when the prebuilt Jupiter Ultra tx fails preflight with a signer/
        PDA-seeds mismatch. The swap-instructions path iterates through
        useSharedAccounts / skip_user_accounts_rpc_calls variants and typically
        lands on one that passes preflight.
        """
        retry_trace: dict[str, Any] = {
            "path": "live_buy_prebuilt_retry",
            "started_at": self._now_iso(),
            "__total_started_monotonic": time.monotonic(),
            "retry_reason": original_error,
        }
        parent_trace["retry_attempted"] = True
        amount_lamports = int(size_sol * LAMPORTS_PER_SOL)
        order_policy = self._resolve_order_policy(side="buy", strategy=None, source_program=None)
        retry_trace["priority_fee_lamports"] = int(order_policy.priority_fee_lamports)
        retry_trace["jito_tip_lamports"] = int(order_policy.jito_tip_lamports)
        try:
            rebuilt = await self._build_custom_swap_tx_async(
                input_mint=SOL_MINT,
                output_mint=token_mint,
                amount=amount_lamports,
                order_policy=order_policy,
                trace=retry_trace,
            )
        except LiveExecutionError as exc:
            logger.warning("LIVE BUY retry rebuild aborted for %s: %s", token_mint[:12], exc)
            retry_trace["retry_outcome"] = "rebuild_aborted"
            parent_trace["retry_trace"] = retry_trace
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("LIVE BUY retry rebuild failed for %s: %s", token_mint[:12], exc)
            retry_trace["retry_outcome"] = f"rebuild_error: {exc}"
            parent_trace["retry_trace"] = retry_trace
            return None

        sign_started = time.monotonic()
        try:
            signed_tx = self.signer.sign_transaction(rebuilt.raw_transaction)
        except SignerError as exc:
            retry_trace["sign_ms"] = (time.monotonic() - sign_started) * 1000.0
            retry_trace["retry_outcome"] = f"signing_failed: {exc}"
            parent_trace["retry_trace"] = retry_trace
            return None
        retry_trace["sign_ms"] = (time.monotonic() - sign_started) * 1000.0

        preflight_sim_started = time.monotonic()
        preflight_err = await self._preflight_simulate_buy_async(
            signed_tx, token_mint=token_mint, trace=retry_trace
        )
        retry_trace["preflight_sim_ms"] = (time.monotonic() - preflight_sim_started) * 1000.0
        if preflight_err is not None:
            retry_trace["retry_outcome"] = f"preflight_failed_again: {preflight_err}"
            parent_trace["retry_trace"] = retry_trace
            logger.warning(
                "LIVE BUY retry still fails preflight for %s: %s",
                token_mint[:12],
                preflight_err,
            )
            return None

        try:
            result: BroadcastResult = await self.broadcaster.broadcast_async(
                signed_tx,
                last_valid_block_height=rebuilt.last_valid_block_height,
            )
        except BroadcastError as exc:
            retry_trace["retry_outcome"] = f"broadcast_failed: {exc}"
            parent_trace["retry_trace"] = retry_trace
            return None
        self._apply_broadcast_trace(retry_trace, result)
        retry_trace["total_execution_ms"] = (
            time.monotonic() - float(retry_trace["__total_started_monotonic"])
        ) * 1000.0
        parent_trace["retry_trace"] = retry_trace
        parent_trace["retry_outcome"] = "confirmed" if result.confirmed else "not_confirmed"

        if not result.confirmed:
            (
                reconciliation,
                reconciliation_error,
            ) = await self._reconcile_failed_transaction_async(
                signature=result.signature,
                token_mint=token_mint,
                expected_in_amount=rebuilt.in_amount,
                expected_out_amount=rebuilt.out_amount,
                slot=result.slot,
                trace=retry_trace,
            )
            return self._result_with_error(
                error=f"not_confirmed: {result.error}",
                signature=result.signature,
                in_amount=rebuilt.in_amount,
                out_amount=rebuilt.out_amount,
                slot=result.slot,
                latency_trace=parent_trace,
                reconciliation=reconciliation,
                reconciliation_error=reconciliation_error,
            )

        reconciliation: LiveFillReconciliation | None = None
        reconciliation_error: str | None = None
        try:
            reconciliation = await self._reconcile_confirmed_fill_async(
                signature=result.signature,
                token_mint=token_mint,
                expected_in_amount=rebuilt.in_amount,
                expected_out_amount=rebuilt.out_amount,
                slot=result.slot,
            )
        except Exception as exc:  # noqa: BLE001
            reconciliation_error = str(exc)

        logger.info(
            "✅ LIVE BUY (retry rebuild) confirmed: sig=%s | in=%d lamports | out=%d tokens | slot=%s",
            result.signature,
            rebuilt.in_amount,
            rebuilt.out_amount,
            result.slot,
        )
        return self._result_success(
            signature=result.signature,
            in_amount=rebuilt.in_amount,
            out_amount=rebuilt.out_amount,
            slot=result.slot,
            latency_trace=parent_trace,
            reconciliation=reconciliation,
            reconciliation_error=reconciliation_error,
        )

    async def _resolve_mint_token_program_async(self, mint_pubkey: str) -> str:
        key = str(mint_pubkey)
        cached = self._mint_token_program_cache.get(key)
        if cached is not None:
            return cached
        try:
            owner = await self.broadcaster.get_account_owner_async(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mint program lookup failed for %s: %s", key, exc)
            owner = None
        if owner in _KNOWN_TOKEN_PROGRAMS:
            resolved = str(owner)
        else:
            if owner:
                logger.warning(
                    "mint %s owned by unknown program %s; defaulting to SPL Token",
                    key,
                    owner,
                )
            resolved = _TOKEN_PROGRAM_ID
        self._mint_token_program_cache[key] = resolved
        return resolved

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(tz=timezone.utc).isoformat()

    @staticmethod
    def _token_raw_balance(
        rows: list[dict[str, Any]], owner: str, token_mint: str
    ) -> tuple[int, int]:
        total_raw = 0
        decimals = 0
        for row in rows:
            if str(row.get("owner") or "") != owner:
                continue
            if str(row.get("mint") or "") != token_mint:
                continue
            ui = row.get("uiTokenAmount") or row.get("ui_token_amount") or {}
            raw_amount = ui.get("amount")
            try:
                total_raw += int(raw_amount or 0)
            except (TypeError, ValueError):
                continue
            try:
                decimals = int(ui.get("decimals") or decimals or 0)
            except (TypeError, ValueError):
                pass
        return total_raw, decimals

    @staticmethod
    def _token_amount_from_inner_instructions(
        meta: dict[str, Any],
        *,
        wallet_pubkey: str,
        token_mint: str,
    ) -> tuple[int, int]:
        """Sum SPL-token transfer amounts for (wallet, mint) from innerInstructions.

        Fallback used when preTokenBalances/postTokenBalances both resolve to 0
        — typically because the ATA was created+closed inside the TX, leaving
        owner-filtered rows empty.  Returns ``(sent_raw, received_raw)`` where
        ``sent_raw`` is tokens leaving the wallet (sell legs) and
        ``received_raw`` is tokens credited to the wallet (buy legs).
        """
        sent = 0
        received = 0
        inner = meta.get("innerInstructions") or []
        for group in inner:
            for ix in group.get("instructions") or []:
                parsed = ix.get("parsed") or {}
                if not isinstance(parsed, dict):
                    continue
                ix_type = str(parsed.get("type") or "")
                if ix_type not in ("transfer", "transferChecked"):
                    continue
                info = parsed.get("info") or {}
                if not isinstance(info, dict):
                    continue
                mint = info.get("mint")
                if mint is not None and str(mint) != token_mint:
                    continue
                raw_amount = info.get("amount")
                if raw_amount is None:
                    token_amount = info.get("tokenAmount") or {}
                    raw_amount = token_amount.get("amount")
                try:
                    amt = int(raw_amount or 0)
                except (TypeError, ValueError):
                    continue
                authority = str(info.get("authority") or info.get("source") or "")
                destination_owner = str(info.get("destinationOwner") or "")
                if authority == wallet_pubkey:
                    sent += amt
                elif destination_owner == wallet_pubkey:
                    received += amt
        return sent, received

    @staticmethod
    def _account_pubkey(entry: Any) -> str:
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            pubkey = entry.get("pubkey") or entry.get("pubKey")
            return str(pubkey or "")
        return ""

    def _actual_tip_lamports_from_tx(self, tx_result: dict[str, Any] | None) -> int:
        if not isinstance(tx_result, dict):
            return 0
        meta = tx_result.get("meta") or {}
        if meta.get("err") is not None:
            return 0
        tx = tx_result.get("transaction") or {}
        message = tx.get("message") or {}
        total = 0
        for instruction in list(message.get("instructions") or []):
            if not isinstance(instruction, dict):
                continue
            parsed = instruction.get("parsed")
            if not isinstance(parsed, dict):
                continue
            if str(parsed.get("type") or "") != "transfer":
                continue
            info = parsed.get("info") or {}
            destination = str(info.get("destination") or "")
            if (
                self.broadcaster.jito_tip_accounts
                and destination not in self.broadcaster.jito_tip_accounts
            ):
                continue
            try:
                total += int(info.get("lamports") or 0)
            except (TypeError, ValueError):
                continue
        return max(total, 0)

    @staticmethod
    def _derive_associated_token_address(
        owner_pubkey: str,
        mint_pubkey: str,
        token_program_id: str = _TOKEN_PROGRAM_ID,
    ):
        from solders.pubkey import Pubkey  # type: ignore[import-untyped]

        owner = Pubkey.from_string(str(owner_pubkey))
        mint = Pubkey.from_string(str(mint_pubkey))
        token_program = Pubkey.from_string(str(token_program_id))
        ata_program = Pubkey.from_string(_ASSOCIATED_TOKEN_PROGRAM_ID)
        ata, _ = Pubkey.find_program_address(
            [bytes(owner), bytes(token_program), bytes(mint)],
            ata_program,
        )
        return ata

    def _close_token_account_instruction(
        self,
        owner_pubkey: str,
        mint_pubkey: str,
        token_program_id: str = _TOKEN_PROGRAM_ID,
    ):
        from solders.instruction import AccountMeta, Instruction  # type: ignore[import-untyped]
        from solders.pubkey import Pubkey  # type: ignore[import-untyped]

        owner = Pubkey.from_string(str(owner_pubkey))
        token_account = self._derive_associated_token_address(
            owner_pubkey,
            mint_pubkey,
            token_program_id,
        )
        return Instruction(
            Pubkey.from_string(str(token_program_id)),
            bytes([9]),  # CloseAccount
            [
                AccountMeta(token_account, False, True),
                AccountMeta(owner, False, True),
                AccountMeta(owner, True, False),
            ],
        )

    @staticmethod
    def _create_associated_token_account_idempotent_instruction(
        payer_pubkey: str,
        owner_pubkey: str,
        mint_pubkey: str,
        token_program_id: str = _TOKEN_PROGRAM_ID,
    ):
        from solders.instruction import AccountMeta, Instruction  # type: ignore[import-untyped]
        from solders.pubkey import Pubkey  # type: ignore[import-untyped]

        payer = Pubkey.from_string(str(payer_pubkey))
        owner = Pubkey.from_string(str(owner_pubkey))
        mint = Pubkey.from_string(str(mint_pubkey))
        token_program = Pubkey.from_string(str(token_program_id))
        ata_program = Pubkey.from_string(_ASSOCIATED_TOKEN_PROGRAM_ID)
        system_program = Pubkey.from_string(_SYSTEM_PROGRAM_ID)
        ata, _ = Pubkey.find_program_address(
            [bytes(owner), bytes(token_program), bytes(mint)],
            ata_program,
        )
        instruction = Instruction(
            ata_program,
            bytes([1]),  # CreateIdempotent
            [
                AccountMeta(payer, True, True),
                AccountMeta(ata, False, True),
                AccountMeta(owner, False, False),
                AccountMeta(mint, False, False),
                AccountMeta(system_program, False, False),
                AccountMeta(token_program, False, False),
            ],
        )
        return ata, instruction

    @staticmethod
    def _sync_native_instruction(
        token_account_pubkey: str, token_program_id: str = _TOKEN_PROGRAM_ID
    ):
        from solders.instruction import AccountMeta, Instruction  # type: ignore[import-untyped]
        from solders.pubkey import Pubkey  # type: ignore[import-untyped]

        return Instruction(
            Pubkey.from_string(str(token_program_id)),
            bytes([17]),  # SyncNative
            [AccountMeta(Pubkey.from_string(str(token_account_pubkey)), False, True)],
        )

    @staticmethod
    def _native_pump_buy_data(
        *, base_amount_out: int, max_quote_amount_in: int, track_volume: bool
    ) -> bytes:
        return _PUMP_AMM_BUY_DISC + struct.pack(
            "<QQB",
            max(int(base_amount_out or 0), 0),
            max(int(max_quote_amount_in or 0), 0),
            1 if track_volume else 0,
        )

    @staticmethod
    def _native_pump_sell_data(*, base_amount_in: int, min_quote_amount_out: int) -> bytes:
        return _PUMP_AMM_SELL_DISC + struct.pack(
            "<QQ",
            max(int(base_amount_in or 0), 0),
            max(int(min_quote_amount_out or 0), 0),
        )

    @staticmethod
    def _find_program_address(seed_parts: list[bytes], program_id: str):
        from solders.pubkey import Pubkey  # type: ignore[import-untyped]

        pda, _ = Pubkey.find_program_address(seed_parts, Pubkey.from_string(str(program_id)))
        return pda

    def _estimated_buy_required_lamports(
        self, size_sol: float, order_policy: "LiveOrderPolicy"
    ) -> int:
        size_lamports = max(int(round(float(size_sol) * float(LAMPORTS_PER_SOL))), 0)
        overhead = (
            max(int(order_policy.priority_fee_lamports or 0), 0)
            + max(int(order_policy.jito_tip_lamports or 0), 0)
            + _SIGNATURE_FEE_LAMPORTS
            + max(
                int(getattr(self.config, "live_buy_ata_rent_buffer_lamports", 0) or 0),
                0,
            )
            + max(int(getattr(self.config, "live_min_wallet_buffer_lamports", 0) or 0), 0)
        )
        return max(size_lamports + overhead, 0)

    def _enforce_buy_wallet_balance(
        self, size_sol: float, order_policy: "LiveOrderPolicy", trace: dict[str, Any]
    ) -> None:
        wallet_pubkey = str(self.signer.get_public_key() or "")
        if not wallet_pubkey:
            raise LiveExecutionError("Signer not validated – refusing live execution")
        started = time.monotonic()
        balance_lamports = self.broadcaster.get_balance(wallet_pubkey, commitment="confirmed")
        trace["wallet_balance_check_ms"] = (time.monotonic() - started) * 1000.0
        required_lamports = self._estimated_buy_required_lamports(size_sol, order_policy)
        trace["wallet_balance_lamports"] = int(balance_lamports)
        trace["wallet_required_lamports"] = int(required_lamports)
        if balance_lamports < required_lamports:
            raise LiveExecutionError(
                f"insufficient wallet balance: have {balance_lamports} lamports, need {required_lamports}"
            )

    async def _enforce_buy_wallet_balance_async(
        self,
        size_sol: float,
        order_policy: "LiveOrderPolicy",
        trace: dict[str, Any],
    ) -> None:
        wallet_pubkey = str(self.signer.get_public_key() or "")
        if not wallet_pubkey:
            raise LiveExecutionError("Signer not validated – refusing live execution")
        started = time.monotonic()
        balance_lamports = await self.broadcaster.get_balance_async(
            wallet_pubkey, commitment="confirmed"
        )
        trace["wallet_balance_check_ms"] = (time.monotonic() - started) * 1000.0
        required_lamports = self._estimated_buy_required_lamports(size_sol, order_policy)
        trace["wallet_balance_lamports"] = int(balance_lamports)
        trace["wallet_required_lamports"] = int(required_lamports)
        if balance_lamports < required_lamports:
            raise LiveExecutionError(
                f"insufficient wallet balance: have {balance_lamports} lamports, need {required_lamports}"
            )

    def _resolve_wallet_token_balance(
        self,
        *,
        token_mint: str,
        requested_token_amount: int,
        close_token_account: bool,
        trace: dict[str, Any],
    ) -> tuple[int, bool]:
        wallet_pubkey = str(self.signer.get_public_key() or "")
        if not wallet_pubkey:
            raise LiveExecutionError("Signer not validated – refusing live execution")
        started = time.monotonic()
        balance_raw, decimals = self.broadcaster.get_owner_token_balance_raw(
            wallet_pubkey,
            token_mint,
            commitment="confirmed",
        )
        trace["wallet_token_balance_check_ms"] = (time.monotonic() - started) * 1000.0
        trace["wallet_token_balance_raw"] = int(balance_raw)
        trace["wallet_token_decimals"] = int(decimals)
        trace["wallet_token_amount_requested_raw"] = int(requested_token_amount)
        if balance_raw <= 0:
            raise LiveExecutionError("wallet_token_balance_zero")
        effective_amount = min(max(int(requested_token_amount or 0), 0), int(balance_raw))
        trace["wallet_token_amount_effective_raw"] = int(effective_amount)
        if effective_amount < int(requested_token_amount or 0):
            trace["wallet_token_balance_clamped"] = True
            trace["wallet_token_balance_shortfall_raw"] = int(requested_token_amount or 0) - int(
                effective_amount
            )
        effective_close_token_account = bool(
            close_token_account or effective_amount >= int(balance_raw)
        )
        trace["wallet_token_close_account_effective"] = bool(effective_close_token_account)
        return int(effective_amount), effective_close_token_account

    async def _resolve_wallet_token_balance_async(
        self,
        *,
        token_mint: str,
        requested_token_amount: int,
        close_token_account: bool,
        trace: dict[str, Any],
    ) -> tuple[int, bool]:
        wallet_pubkey = str(self.signer.get_public_key() or "")
        if not wallet_pubkey:
            raise LiveExecutionError("Signer not validated – refusing live execution")
        started = time.monotonic()
        (
            balance_raw,
            decimals,
        ) = await self.broadcaster.get_owner_token_balance_raw_async(
            wallet_pubkey,
            token_mint,
            commitment="confirmed",
        )
        trace["wallet_token_balance_check_ms"] = (time.monotonic() - started) * 1000.0
        trace["wallet_token_balance_raw"] = int(balance_raw)
        trace["wallet_token_decimals"] = int(decimals)
        trace["wallet_token_amount_requested_raw"] = int(requested_token_amount)
        if balance_raw <= 0:
            raise LiveExecutionError("wallet_token_balance_zero")
        effective_amount = min(max(int(requested_token_amount or 0), 0), int(balance_raw))
        trace["wallet_token_amount_effective_raw"] = int(effective_amount)
        if effective_amount < int(requested_token_amount or 0):
            trace["wallet_token_balance_clamped"] = True
            trace["wallet_token_balance_shortfall_raw"] = int(requested_token_amount or 0) - int(
                effective_amount
            )
        effective_close_token_account = bool(
            close_token_account or effective_amount >= int(balance_raw)
        )
        trace["wallet_token_close_account_effective"] = bool(effective_close_token_account)
        return int(effective_amount), effective_close_token_account

    def _validate_live_sell_viability(
        self,
        *,
        expected_out_lamports: int,
        order_policy: "LiveOrderPolicy",
        close_token_account: bool,
    ) -> None:
        token_account_rent = (
            max(int(getattr(self.config, "live_token_account_rent_lamports", 0) or 0), 0)
            if close_token_account
            and bool(getattr(self.config, "live_close_token_ata_on_full_exit", True))
            else 0
        )
        fee_budget = (
            max(int(order_policy.priority_fee_lamports or 0), 0)
            + max(int(order_policy.jito_tip_lamports or 0), 0)
            + _SIGNATURE_FEE_LAMPORTS
        )
        expected_wallet_delta = int(expected_out_lamports or 0) + token_account_rent - fee_budget
        min_net_wallet_delta = max(
            int(getattr(self.config, "live_min_net_exit_lamports", 0) or 0), 0
        )
        if expected_wallet_delta < min_net_wallet_delta:
            raise LiveExecutionError(
                "dust_exit_blocked: "
                f"expected wallet delta {expected_wallet_delta} lamports < minimum {min_net_wallet_delta}"
            )

    def _local_quote_out_amount(
        self,
        *,
        input_mint: str,
        output_mint: str,
        amount: int,
    ) -> int | None:
        if self.local_quote_engine is None or amount <= 0:
            return None
        if (
            input_mint == SOL_MINT
            and output_mint != SOL_MINT
            and self.local_quote_engine.has_reserves(output_mint)
        ):
            return self.local_quote_engine.quote_buy(output_mint, int(amount))
        if (
            output_mint == SOL_MINT
            and input_mint != SOL_MINT
            and self.local_quote_engine.has_reserves(input_mint)
        ):
            return self.local_quote_engine.quote_sell(input_mint, int(amount))
        return None

    def _conservative_sell_out_lamports(
        self,
        *,
        token_mint: str,
        token_amount: int,
        routed_out_lamports: int,
        trace: dict[str, Any] | None = None,
    ) -> int:
        local_out = self._local_quote_out_amount(
            input_mint=token_mint,
            output_mint=SOL_MINT,
            amount=token_amount,
        )
        if trace is not None and local_out is not None:
            trace["local_quote_sell_out_lamports"] = int(local_out)
            if int(routed_out_lamports or 0) > 0:
                trace["local_quote_sell_vs_route_ratio"] = float(local_out) / float(
                    routed_out_lamports
                )
        routed = max(int(routed_out_lamports or 0), 0)
        if local_out is None or local_out <= 0:
            return routed
        if routed <= 0:
            return int(local_out)
        return min(routed, int(local_out))

    def _native_pump_pool_state(
        self,
        *,
        token_mint: str,
        trace: dict[str, Any] | None = None,
    ) -> PumpAMMNativePoolState | None:
        trace_payload = trace if trace is not None else {}
        trace_payload["native_pump_enabled"] = bool(
            getattr(self.config, "live_enable_native_pump_amm", True)
        )
        if not bool(getattr(self.config, "live_enable_native_pump_amm", True)):
            trace_payload["native_pump_reason"] = "disabled"
            return None
        if self.local_quote_engine is None:
            trace_payload["native_pump_reason"] = "no_local_quote_engine"
            return None
        state = self.local_quote_engine.get_native_pool_state(token_mint)
        if state is None:
            trace_payload["native_pump_reason"] = "state_missing_or_stale"
            return None
        if not state.token_is_base_quote_is_wsol:
            trace_payload["native_pump_reason"] = "unsupported_pool_orientation"
            trace_payload["native_pump_base_mint"] = state.base_mint
            trace_payload["native_pump_quote_mint"] = state.quote_mint
            return None
        reserves = self.local_quote_engine.get_reserves(token_mint)
        if reserves is None:
            trace_payload["native_pump_reason"] = "reserves_missing_or_stale"
            return None
        trace_payload["native_pump_pool"] = state.pool
        trace_payload["native_pump_program_id"] = state.program_id
        trace_payload["native_pump_state_age_ms"] = max(
            0.0, (time.monotonic() - float(state.ts or 0.0)) * 1000.0
        )
        trace_payload["native_pump_base_mint"] = state.base_mint
        trace_payload["native_pump_quote_mint"] = state.quote_mint
        return state

    def _native_pump_buy_out_amount(self, token_mint: str, amount_lamports: int) -> int:
        if self.local_quote_engine is None:
            return 0
        return int(self.local_quote_engine.quote_buy(token_mint, int(amount_lamports)) or 0)

    def _native_pump_sell_out_amount(self, token_mint: str, token_amount: int) -> int:
        if self.local_quote_engine is None:
            return 0
        return int(self.local_quote_engine.quote_sell(token_mint, int(token_amount)) or 0)

    def _build_native_pump_amm_buy_tx(
        self,
        *,
        token_mint: str,
        amount_lamports: int,
        order_policy: LiveOrderPolicy,
        trace: dict[str, Any] | None = None,
    ) -> SwapTransaction:
        from solders.compute_budget import (
            set_compute_unit_limit,
            set_compute_unit_price,
        )  # type: ignore[import-untyped]
        from solders.hash import Hash  # type: ignore[import-untyped]
        from solders.instruction import AccountMeta, Instruction  # type: ignore[import-untyped]
        from solders.message import MessageV0  # type: ignore[import-untyped]
        from solders.null_signer import NullSigner  # type: ignore[import-untyped]
        from solders.pubkey import Pubkey  # type: ignore[import-untyped]
        from solders.system_program import TransferParams, transfer  # type: ignore[import-untyped]
        from solders.transaction import VersionedTransaction  # type: ignore[import-untyped]

        trace_payload = trace if trace is not None else {}
        state = self._native_pump_pool_state(token_mint=token_mint, trace=trace_payload)
        if state is None:
            raise LiveExecutionError(
                f"native_pump_unavailable: {trace_payload.get('native_pump_reason')}"
            )
        user_public_key = str(self.signer.get_public_key() or "")
        if not user_public_key:
            raise LiveExecutionError("Signer not validated – cannot build native Pump AMM buy")

        quote_out_amount = self._native_pump_buy_out_amount(token_mint, amount_lamports)
        if quote_out_amount <= 0:
            raise LiveExecutionError("native_pump_quote_unavailable")
        reserves = (
            self.local_quote_engine.get_reserves(token_mint) if self.local_quote_engine else None
        )
        if reserves is not None:
            self._check_native_pump_price_impact(
                token_mint=token_mint,
                side="buy",
                amount_in=int(amount_lamports),
                amount_out=int(quote_out_amount),
                reserve_in=int(reserves.sol_reserve),
                reserve_out=int(reserves.token_reserve),
                trace=trace_payload,
                order_policy=order_policy,
            )
        slippage_bps = self._custom_tx_slippage_bps(order_policy.slippage_bps)
        base_amount_out = max((quote_out_amount * max(10_000 - slippage_bps, 0)) // 10_000, 1)
        if base_amount_out <= 0:
            raise LiveExecutionError("native_pump_min_out_zero")

        trace_payload["venue_path"] = "pump_amm_native_buy"
        trace_payload["slippage_bps"] = slippage_bps
        trace_payload["slippage_mode"] = "native_local_quote"
        trace_payload["native_pump_quote_out_raw"] = int(quote_out_amount)
        trace_payload["native_pump_base_amount_out_raw"] = int(base_amount_out)
        trace_payload["native_pump_max_quote_amount_in"] = int(amount_lamports)

        payer = Pubkey.from_string(user_public_key)
        tip_account = self._select_tip_account()
        tip_lamports = max(int(order_policy.jito_tip_lamports or 0), 0)
        compute_unit_limit = max(
            int(getattr(self.config, "live_pump_amm_buy_compute_unit_limit", 450_000) or 450_000),
            1,
        )
        compute_unit_price_micro_lamports = self._priority_fee_to_micro_lamports(
            order_policy.priority_fee_lamports,
            compute_unit_limit,
        )
        base_ata, create_base_ata_ix = self._create_associated_token_account_idempotent_instruction(
            user_public_key,
            user_public_key,
            state.base_mint,
            state.base_token_program,
        )
        quote_ata, create_quote_ata_ix = (
            self._create_associated_token_account_idempotent_instruction(
                user_public_key,
                user_public_key,
                state.quote_mint,
                _TOKEN_PROGRAM_ID,
            )
        )
        global_volume_accumulator = state.global_volume_accumulator or str(
            self._find_program_address([_PUMP_AMM_GLOBAL_VOLUME_ACCUMULATOR_SEED], state.program_id)
        )
        user_volume_accumulator = str(
            self._find_program_address(
                [
                    _PUMP_AMM_USER_VOLUME_ACCUMULATOR_SEED,
                    bytes(Pubkey.from_string(user_public_key)),
                ],
                state.program_id,
            )
        )
        pump_buy_ix = Instruction(
            Pubkey.from_string(state.program_id),
            self._native_pump_buy_data(
                base_amount_out=base_amount_out,
                max_quote_amount_in=amount_lamports,
                track_volume=False,
            ),
            [
                AccountMeta(Pubkey.from_string(state.pool), False, True),
                AccountMeta(payer, True, True),
                AccountMeta(Pubkey.from_string(state.global_config), False, False),
                AccountMeta(Pubkey.from_string(state.base_mint), False, False),
                AccountMeta(Pubkey.from_string(state.quote_mint), False, False),
                AccountMeta(base_ata, False, True),
                AccountMeta(quote_ata, False, True),
                AccountMeta(Pubkey.from_string(state.pool_base_token_account), False, True),
                AccountMeta(Pubkey.from_string(state.pool_quote_token_account), False, True),
                AccountMeta(Pubkey.from_string(state.protocol_fee_recipient), False, False),
                AccountMeta(
                    Pubkey.from_string(state.protocol_fee_recipient_token_account),
                    False,
                    True,
                ),
                AccountMeta(Pubkey.from_string(state.base_token_program), False, False),
                AccountMeta(Pubkey.from_string(_TOKEN_PROGRAM_ID), False, False),
                AccountMeta(Pubkey.from_string(_SYSTEM_PROGRAM_ID), False, False),
                AccountMeta(Pubkey.from_string(_ASSOCIATED_TOKEN_PROGRAM_ID), False, False),
                AccountMeta(Pubkey.from_string(state.event_authority), False, False),
                AccountMeta(Pubkey.from_string(state.program_id), False, False),
                AccountMeta(Pubkey.from_string(state.coin_creator_vault_ata), False, True),
                AccountMeta(Pubkey.from_string(state.coin_creator_vault_authority), False, False),
                AccountMeta(Pubkey.from_string(global_volume_accumulator), False, False),
                AccountMeta(Pubkey.from_string(user_volume_accumulator), False, True),
                AccountMeta(Pubkey.from_string(state.fee_config), False, False),
                AccountMeta(Pubkey.from_string(state.fee_program), False, False),
            ],
        )

        blockhash_started = time.monotonic()
        recent_blockhash, last_valid_block_height = self.broadcaster.get_latest_blockhash()
        trace_payload["blockhash_ms"] = (time.monotonic() - blockhash_started) * 1000.0

        instructions = [
            set_compute_unit_limit(int(compute_unit_limit)),
            set_compute_unit_price(int(compute_unit_price_micro_lamports)),
            create_base_ata_ix,
            create_quote_ata_ix,
            transfer(
                TransferParams(
                    from_pubkey=payer,
                    to_pubkey=quote_ata,
                    lamports=max(int(amount_lamports or 0), 0),
                )
            ),
            self._sync_native_instruction(str(quote_ata), _TOKEN_PROGRAM_ID),
            pump_buy_ix,
            self._close_token_account_instruction(
                user_public_key,
                state.quote_mint,
                _TOKEN_PROGRAM_ID,
            ),
            transfer(
                TransferParams(
                    from_pubkey=payer,
                    to_pubkey=Pubkey.from_string(tip_account),
                    lamports=tip_lamports,
                )
            ),
        ]

        compile_started = time.monotonic()
        message = MessageV0.try_compile(
            payer,
            instructions,
            [],
            Hash.from_string(recent_blockhash),
        )
        unsigned_tx = VersionedTransaction(message, [NullSigner(payer)])
        trace_payload["tx_compile_ms"] = (time.monotonic() - compile_started) * 1000.0
        trace_payload["custom_tx_instruction_count"] = len(instructions)
        trace_payload["manual_tip_account"] = tip_account
        trace_payload["manual_tip_lamports"] = tip_lamports
        trace_payload["manual_compute_unit_limit"] = int(compute_unit_limit)
        trace_payload["manual_compute_unit_price_micro_lamports"] = int(
            compute_unit_price_micro_lamports
        )

        return SwapTransaction(
            raw_transaction=bytes(unsigned_tx),
            last_valid_block_height=last_valid_block_height,
            in_amount=int(amount_lamports),
            out_amount=int(base_amount_out),
            built_at=self._now_iso(),
        )

    async def _build_native_pump_amm_buy_tx_async(
        self,
        *,
        token_mint: str,
        amount_lamports: int,
        order_policy: LiveOrderPolicy,
        trace: dict[str, Any] | None = None,
    ) -> SwapTransaction:
        from solders.compute_budget import (
            set_compute_unit_limit,
            set_compute_unit_price,
        )  # type: ignore[import-untyped]
        from solders.hash import Hash  # type: ignore[import-untyped]
        from solders.instruction import AccountMeta, Instruction  # type: ignore[import-untyped]
        from solders.message import MessageV0  # type: ignore[import-untyped]
        from solders.null_signer import NullSigner  # type: ignore[import-untyped]
        from solders.pubkey import Pubkey  # type: ignore[import-untyped]
        from solders.system_program import TransferParams, transfer  # type: ignore[import-untyped]
        from solders.transaction import VersionedTransaction  # type: ignore[import-untyped]

        trace_payload = trace if trace is not None else {}
        state = self._native_pump_pool_state(token_mint=token_mint, trace=trace_payload)
        if state is None:
            raise LiveExecutionError(
                f"native_pump_unavailable: {trace_payload.get('native_pump_reason')}"
            )
        user_public_key = str(self.signer.get_public_key() or "")
        if not user_public_key:
            raise LiveExecutionError("Signer not validated – cannot build native Pump AMM buy")

        quote_out_amount = self._native_pump_buy_out_amount(token_mint, amount_lamports)
        if quote_out_amount <= 0:
            raise LiveExecutionError("native_pump_quote_unavailable")
        reserves = (
            self.local_quote_engine.get_reserves(token_mint) if self.local_quote_engine else None
        )
        if reserves is not None:
            self._check_native_pump_price_impact(
                token_mint=token_mint,
                side="buy",
                amount_in=int(amount_lamports),
                amount_out=int(quote_out_amount),
                reserve_in=int(reserves.sol_reserve),
                reserve_out=int(reserves.token_reserve),
                trace=trace_payload,
                order_policy=order_policy,
            )
        slippage_bps = self._custom_tx_slippage_bps(order_policy.slippage_bps)
        base_amount_out = max((quote_out_amount * max(10_000 - slippage_bps, 0)) // 10_000, 1)
        if base_amount_out <= 0:
            raise LiveExecutionError("native_pump_min_out_zero")

        trace_payload["venue_path"] = "pump_amm_native_buy"
        trace_payload["slippage_bps"] = slippage_bps
        trace_payload["slippage_mode"] = "native_local_quote"
        trace_payload["native_pump_quote_out_raw"] = int(quote_out_amount)
        trace_payload["native_pump_base_amount_out_raw"] = int(base_amount_out)
        trace_payload["native_pump_max_quote_amount_in"] = int(amount_lamports)

        payer = Pubkey.from_string(user_public_key)
        tip_account = self._select_tip_account()
        tip_lamports = max(int(order_policy.jito_tip_lamports or 0), 0)
        compute_unit_limit = max(
            int(getattr(self.config, "live_pump_amm_buy_compute_unit_limit", 450_000) or 450_000),
            1,
        )
        compute_unit_price_micro_lamports = self._priority_fee_to_micro_lamports(
            order_policy.priority_fee_lamports,
            compute_unit_limit,
        )
        base_ata, create_base_ata_ix = self._create_associated_token_account_idempotent_instruction(
            user_public_key,
            user_public_key,
            state.base_mint,
            state.base_token_program,
        )
        quote_ata, create_quote_ata_ix = (
            self._create_associated_token_account_idempotent_instruction(
                user_public_key,
                user_public_key,
                state.quote_mint,
                _TOKEN_PROGRAM_ID,
            )
        )
        global_volume_accumulator = state.global_volume_accumulator or str(
            self._find_program_address([_PUMP_AMM_GLOBAL_VOLUME_ACCUMULATOR_SEED], state.program_id)
        )
        user_volume_accumulator = str(
            self._find_program_address(
                [
                    _PUMP_AMM_USER_VOLUME_ACCUMULATOR_SEED,
                    bytes(Pubkey.from_string(user_public_key)),
                ],
                state.program_id,
            )
        )
        pump_buy_ix = Instruction(
            Pubkey.from_string(state.program_id),
            self._native_pump_buy_data(
                base_amount_out=base_amount_out,
                max_quote_amount_in=amount_lamports,
                track_volume=False,
            ),
            [
                AccountMeta(Pubkey.from_string(state.pool), False, True),
                AccountMeta(payer, True, True),
                AccountMeta(Pubkey.from_string(state.global_config), False, False),
                AccountMeta(Pubkey.from_string(state.base_mint), False, False),
                AccountMeta(Pubkey.from_string(state.quote_mint), False, False),
                AccountMeta(base_ata, False, True),
                AccountMeta(quote_ata, False, True),
                AccountMeta(Pubkey.from_string(state.pool_base_token_account), False, True),
                AccountMeta(Pubkey.from_string(state.pool_quote_token_account), False, True),
                AccountMeta(Pubkey.from_string(state.protocol_fee_recipient), False, False),
                AccountMeta(
                    Pubkey.from_string(state.protocol_fee_recipient_token_account),
                    False,
                    True,
                ),
                AccountMeta(Pubkey.from_string(state.base_token_program), False, False),
                AccountMeta(Pubkey.from_string(_TOKEN_PROGRAM_ID), False, False),
                AccountMeta(Pubkey.from_string(_SYSTEM_PROGRAM_ID), False, False),
                AccountMeta(Pubkey.from_string(_ASSOCIATED_TOKEN_PROGRAM_ID), False, False),
                AccountMeta(Pubkey.from_string(state.event_authority), False, False),
                AccountMeta(Pubkey.from_string(state.program_id), False, False),
                AccountMeta(Pubkey.from_string(state.coin_creator_vault_ata), False, True),
                AccountMeta(Pubkey.from_string(state.coin_creator_vault_authority), False, False),
                AccountMeta(Pubkey.from_string(global_volume_accumulator), False, False),
                AccountMeta(Pubkey.from_string(user_volume_accumulator), False, True),
                AccountMeta(Pubkey.from_string(state.fee_config), False, False),
                AccountMeta(Pubkey.from_string(state.fee_program), False, False),
            ],
        )

        blockhash_started = time.monotonic()
        (
            recent_blockhash,
            last_valid_block_height,
        ) = await self.broadcaster.get_latest_blockhash_async()
        trace_payload["blockhash_ms"] = (time.monotonic() - blockhash_started) * 1000.0

        instructions = [
            set_compute_unit_limit(int(compute_unit_limit)),
            set_compute_unit_price(int(compute_unit_price_micro_lamports)),
            create_base_ata_ix,
            create_quote_ata_ix,
            transfer(
                TransferParams(
                    from_pubkey=payer,
                    to_pubkey=quote_ata,
                    lamports=max(int(amount_lamports or 0), 0),
                )
            ),
            self._sync_native_instruction(str(quote_ata), _TOKEN_PROGRAM_ID),
            pump_buy_ix,
            self._close_token_account_instruction(
                user_public_key,
                state.quote_mint,
                _TOKEN_PROGRAM_ID,
            ),
            transfer(
                TransferParams(
                    from_pubkey=payer,
                    to_pubkey=Pubkey.from_string(tip_account),
                    lamports=tip_lamports,
                )
            ),
        ]

        compile_started = time.monotonic()
        message = MessageV0.try_compile(
            payer,
            instructions,
            [],
            Hash.from_string(recent_blockhash),
        )
        unsigned_tx = VersionedTransaction(message, [NullSigner(payer)])
        trace_payload["tx_compile_ms"] = (time.monotonic() - compile_started) * 1000.0
        trace_payload["custom_tx_instruction_count"] = len(instructions)
        trace_payload["manual_tip_account"] = tip_account
        trace_payload["manual_tip_lamports"] = tip_lamports
        trace_payload["manual_compute_unit_limit"] = int(compute_unit_limit)
        trace_payload["manual_compute_unit_price_micro_lamports"] = int(
            compute_unit_price_micro_lamports
        )

        return SwapTransaction(
            raw_transaction=bytes(unsigned_tx),
            last_valid_block_height=last_valid_block_height,
            in_amount=int(amount_lamports),
            out_amount=int(base_amount_out),
            built_at=self._now_iso(),
        )

    def _build_native_pump_amm_sell_tx(
        self,
        *,
        token_mint: str,
        token_amount: int,
        order_policy: LiveOrderPolicy,
        close_input_token_account: bool = False,
        trace: dict[str, Any] | None = None,
    ) -> SwapTransaction:
        from solders.compute_budget import (
            set_compute_unit_limit,
            set_compute_unit_price,
        )  # type: ignore[import-untyped]
        from solders.hash import Hash  # type: ignore[import-untyped]
        from solders.instruction import AccountMeta, Instruction  # type: ignore[import-untyped]
        from solders.message import MessageV0  # type: ignore[import-untyped]
        from solders.null_signer import NullSigner  # type: ignore[import-untyped]
        from solders.pubkey import Pubkey  # type: ignore[import-untyped]
        from solders.system_program import TransferParams, transfer  # type: ignore[import-untyped]
        from solders.transaction import VersionedTransaction  # type: ignore[import-untyped]

        trace_payload = trace if trace is not None else {}
        state = self._native_pump_pool_state(token_mint=token_mint, trace=trace_payload)
        if state is None:
            raise LiveExecutionError(
                f"native_pump_unavailable: {trace_payload.get('native_pump_reason')}"
            )
        user_public_key = str(self.signer.get_public_key() or "")
        if not user_public_key:
            raise LiveExecutionError("Signer not validated – cannot build native Pump AMM sell")

        quote_out_amount = self._native_pump_sell_out_amount(token_mint, token_amount)
        if quote_out_amount <= 0:
            raise LiveExecutionError("native_pump_quote_unavailable")
        reserves = (
            self.local_quote_engine.get_reserves(token_mint) if self.local_quote_engine else None
        )
        # Refuse native sell when reserves were located via the fragile
        # largest-balance heuristic on a fresh pool — on post-migration
        # Pump-AMM pools the heuristic has been observed to lock onto
        # bonding-curve escrows and return quotes ~48x inflated, causing
        # every preflight to fail with Custom 6004 ExceededSlippage
        # (session 20260419T103007Z, pos 35).  Route the sell through
        # Jupiter instead, where the quote comes from authoritative state.
        if reserves is not None and getattr(reserves, "source", "") == "heuristic_fallback":
            trace_payload["native_pump_fallback_reason"] = "heuristic_reserves_untrusted"
            raise LiveExecutionError("native_pump_quote_untrusted_heuristic")
        if reserves is not None:
            self._check_native_pump_price_impact(
                token_mint=token_mint,
                side="sell",
                amount_in=int(token_amount),
                amount_out=int(quote_out_amount),
                reserve_in=int(reserves.token_reserve),
                reserve_out=int(reserves.sol_reserve),
                trace=trace_payload,
            )
        slippage_bps = self._custom_tx_slippage_bps(order_policy.slippage_bps)
        min_quote_amount_out = max((quote_out_amount * max(10_000 - slippage_bps, 0)) // 10_000, 1)
        if min_quote_amount_out <= 0:
            raise LiveExecutionError("native_pump_min_out_zero")

        trace_payload["venue_path"] = "pump_amm_native_sell"
        trace_payload["slippage_bps"] = slippage_bps
        trace_payload["slippage_mode"] = "native_local_quote"
        trace_payload["native_pump_quote_out_lamports"] = int(quote_out_amount)
        trace_payload["native_pump_min_quote_amount_out"] = int(min_quote_amount_out)
        if reserves is not None:
            trace_payload["native_pump_reserve_source"] = getattr(
                reserves, "source", "heuristic_fallback"
            )
            trace_payload["native_pump_sol_reserve_lamports"] = int(reserves.sol_reserve)
            trace_payload["native_pump_token_reserve_raw"] = int(reserves.token_reserve)

        payer = Pubkey.from_string(user_public_key)
        tip_account = self._select_tip_account()
        tip_lamports = max(int(order_policy.jito_tip_lamports or 0), 0)
        compute_unit_limit = max(
            int(getattr(self.config, "live_pump_amm_sell_compute_unit_limit", 350_000) or 350_000),
            1,
        )
        compute_unit_price_micro_lamports = self._priority_fee_to_micro_lamports(
            order_policy.priority_fee_lamports,
            compute_unit_limit,
        )
        base_ata, create_base_ata_ix = self._create_associated_token_account_idempotent_instruction(
            user_public_key,
            user_public_key,
            state.base_mint,
            state.base_token_program,
        )
        quote_ata, create_quote_ata_ix = (
            self._create_associated_token_account_idempotent_instruction(
                user_public_key,
                user_public_key,
                state.quote_mint,
                _TOKEN_PROGRAM_ID,
            )
        )
        pump_sell_ix = Instruction(
            Pubkey.from_string(state.program_id),
            self._native_pump_sell_data(
                base_amount_in=token_amount,
                min_quote_amount_out=min_quote_amount_out,
            ),
            [
                AccountMeta(Pubkey.from_string(state.pool), False, True),
                AccountMeta(payer, True, True),
                AccountMeta(Pubkey.from_string(state.global_config), False, False),
                AccountMeta(Pubkey.from_string(state.base_mint), False, False),
                AccountMeta(Pubkey.from_string(state.quote_mint), False, False),
                AccountMeta(base_ata, False, True),
                AccountMeta(quote_ata, False, True),
                AccountMeta(Pubkey.from_string(state.pool_base_token_account), False, True),
                AccountMeta(Pubkey.from_string(state.pool_quote_token_account), False, True),
                AccountMeta(Pubkey.from_string(state.protocol_fee_recipient), False, False),
                AccountMeta(
                    Pubkey.from_string(state.protocol_fee_recipient_token_account),
                    False,
                    True,
                ),
                AccountMeta(Pubkey.from_string(state.base_token_program), False, False),
                AccountMeta(Pubkey.from_string(_TOKEN_PROGRAM_ID), False, False),
                AccountMeta(Pubkey.from_string(_SYSTEM_PROGRAM_ID), False, False),
                AccountMeta(Pubkey.from_string(_ASSOCIATED_TOKEN_PROGRAM_ID), False, False),
                AccountMeta(Pubkey.from_string(state.event_authority), False, False),
                AccountMeta(Pubkey.from_string(state.program_id), False, False),
                AccountMeta(Pubkey.from_string(state.coin_creator_vault_ata), False, True),
                AccountMeta(Pubkey.from_string(state.coin_creator_vault_authority), False, False),
                AccountMeta(Pubkey.from_string(state.fee_config), False, False),
                AccountMeta(Pubkey.from_string(state.fee_program), False, False),
            ],
        )

        blockhash_started = time.monotonic()
        recent_blockhash, last_valid_block_height = self.broadcaster.get_latest_blockhash()
        trace_payload["blockhash_ms"] = (time.monotonic() - blockhash_started) * 1000.0

        instructions = [
            set_compute_unit_limit(int(compute_unit_limit)),
            set_compute_unit_price(int(compute_unit_price_micro_lamports)),
            create_base_ata_ix,
            create_quote_ata_ix,
            pump_sell_ix,
        ]
        if close_input_token_account and bool(
            getattr(self.config, "live_close_token_ata_on_full_exit", True)
        ):
            instructions.append(
                self._close_token_account_instruction(
                    user_public_key,
                    state.base_mint,
                    state.base_token_program,
                )
            )
        instructions.extend(
            [
                self._close_token_account_instruction(
                    user_public_key,
                    state.quote_mint,
                    _TOKEN_PROGRAM_ID,
                ),
                transfer(
                    TransferParams(
                        from_pubkey=payer,
                        to_pubkey=Pubkey.from_string(tip_account),
                        lamports=tip_lamports,
                    )
                ),
            ]
        )

        compile_started = time.monotonic()
        message = MessageV0.try_compile(
            payer,
            instructions,
            [],
            Hash.from_string(recent_blockhash),
        )
        unsigned_tx = VersionedTransaction(message, [NullSigner(payer)])
        trace_payload["tx_compile_ms"] = (time.monotonic() - compile_started) * 1000.0
        trace_payload["custom_tx_instruction_count"] = len(instructions)
        trace_payload["manual_tip_account"] = tip_account
        trace_payload["manual_tip_lamports"] = tip_lamports
        trace_payload["manual_close_input_token_account"] = bool(
            close_input_token_account
            and bool(getattr(self.config, "live_close_token_ata_on_full_exit", True))
        )
        trace_payload["manual_compute_unit_limit"] = int(compute_unit_limit)
        trace_payload["manual_compute_unit_price_micro_lamports"] = int(
            compute_unit_price_micro_lamports
        )

        return SwapTransaction(
            raw_transaction=bytes(unsigned_tx),
            last_valid_block_height=last_valid_block_height,
            in_amount=int(token_amount),
            out_amount=int(min_quote_amount_out),
            built_at=self._now_iso(),
        )

    async def _build_native_pump_amm_sell_tx_async(
        self,
        *,
        token_mint: str,
        token_amount: int,
        order_policy: LiveOrderPolicy,
        close_input_token_account: bool = False,
        trace: dict[str, Any] | None = None,
    ) -> SwapTransaction:
        from solders.compute_budget import (
            set_compute_unit_limit,
            set_compute_unit_price,
        )  # type: ignore[import-untyped]
        from solders.hash import Hash  # type: ignore[import-untyped]
        from solders.instruction import AccountMeta, Instruction  # type: ignore[import-untyped]
        from solders.message import MessageV0  # type: ignore[import-untyped]
        from solders.null_signer import NullSigner  # type: ignore[import-untyped]
        from solders.pubkey import Pubkey  # type: ignore[import-untyped]
        from solders.system_program import TransferParams, transfer  # type: ignore[import-untyped]
        from solders.transaction import VersionedTransaction  # type: ignore[import-untyped]

        trace_payload = trace if trace is not None else {}
        state = self._native_pump_pool_state(token_mint=token_mint, trace=trace_payload)
        if state is None:
            raise LiveExecutionError(
                f"native_pump_unavailable: {trace_payload.get('native_pump_reason')}"
            )
        user_public_key = str(self.signer.get_public_key() or "")
        if not user_public_key:
            raise LiveExecutionError("Signer not validated – cannot build native Pump AMM sell")

        quote_out_amount = self._native_pump_sell_out_amount(token_mint, token_amount)
        if quote_out_amount <= 0:
            raise LiveExecutionError("native_pump_quote_unavailable")
        reserves = (
            self.local_quote_engine.get_reserves(token_mint) if self.local_quote_engine else None
        )
        # Refuse native sell when reserves were located via the fragile
        # largest-balance heuristic on a fresh pool — on post-migration
        # Pump-AMM pools the heuristic has been observed to lock onto
        # bonding-curve escrows and return quotes ~48x inflated, causing
        # every preflight to fail with Custom 6004 ExceededSlippage
        # (session 20260419T103007Z, pos 35).  Route the sell through
        # Jupiter instead, where the quote comes from authoritative state.
        if reserves is not None and getattr(reserves, "source", "") == "heuristic_fallback":
            trace_payload["native_pump_fallback_reason"] = "heuristic_reserves_untrusted"
            raise LiveExecutionError("native_pump_quote_untrusted_heuristic")
        if reserves is not None:
            self._check_native_pump_price_impact(
                token_mint=token_mint,
                side="sell",
                amount_in=int(token_amount),
                amount_out=int(quote_out_amount),
                reserve_in=int(reserves.token_reserve),
                reserve_out=int(reserves.sol_reserve),
                trace=trace_payload,
            )
        slippage_bps = self._custom_tx_slippage_bps(order_policy.slippage_bps)
        min_quote_amount_out = max((quote_out_amount * max(10_000 - slippage_bps, 0)) // 10_000, 1)
        if min_quote_amount_out <= 0:
            raise LiveExecutionError("native_pump_min_out_zero")

        trace_payload["venue_path"] = "pump_amm_native_sell"
        trace_payload["slippage_bps"] = slippage_bps
        trace_payload["slippage_mode"] = "native_local_quote"
        trace_payload["native_pump_quote_out_lamports"] = int(quote_out_amount)
        trace_payload["native_pump_min_quote_amount_out"] = int(min_quote_amount_out)
        if reserves is not None:
            trace_payload["native_pump_reserve_source"] = getattr(
                reserves, "source", "heuristic_fallback"
            )
            trace_payload["native_pump_sol_reserve_lamports"] = int(reserves.sol_reserve)
            trace_payload["native_pump_token_reserve_raw"] = int(reserves.token_reserve)

        payer = Pubkey.from_string(user_public_key)
        tip_account = self._select_tip_account()
        tip_lamports = max(int(order_policy.jito_tip_lamports or 0), 0)
        compute_unit_limit = max(
            int(getattr(self.config, "live_pump_amm_sell_compute_unit_limit", 350_000) or 350_000),
            1,
        )
        compute_unit_price_micro_lamports = self._priority_fee_to_micro_lamports(
            order_policy.priority_fee_lamports,
            compute_unit_limit,
        )
        base_ata, create_base_ata_ix = self._create_associated_token_account_idempotent_instruction(
            user_public_key,
            user_public_key,
            state.base_mint,
            state.base_token_program,
        )
        quote_ata, create_quote_ata_ix = (
            self._create_associated_token_account_idempotent_instruction(
                user_public_key,
                user_public_key,
                state.quote_mint,
                _TOKEN_PROGRAM_ID,
            )
        )
        pump_sell_ix = Instruction(
            Pubkey.from_string(state.program_id),
            self._native_pump_sell_data(
                base_amount_in=token_amount,
                min_quote_amount_out=min_quote_amount_out,
            ),
            [
                AccountMeta(Pubkey.from_string(state.pool), False, True),
                AccountMeta(payer, True, True),
                AccountMeta(Pubkey.from_string(state.global_config), False, False),
                AccountMeta(Pubkey.from_string(state.base_mint), False, False),
                AccountMeta(Pubkey.from_string(state.quote_mint), False, False),
                AccountMeta(base_ata, False, True),
                AccountMeta(quote_ata, False, True),
                AccountMeta(Pubkey.from_string(state.pool_base_token_account), False, True),
                AccountMeta(Pubkey.from_string(state.pool_quote_token_account), False, True),
                AccountMeta(Pubkey.from_string(state.protocol_fee_recipient), False, False),
                AccountMeta(
                    Pubkey.from_string(state.protocol_fee_recipient_token_account),
                    False,
                    True,
                ),
                AccountMeta(Pubkey.from_string(state.base_token_program), False, False),
                AccountMeta(Pubkey.from_string(_TOKEN_PROGRAM_ID), False, False),
                AccountMeta(Pubkey.from_string(_SYSTEM_PROGRAM_ID), False, False),
                AccountMeta(Pubkey.from_string(_ASSOCIATED_TOKEN_PROGRAM_ID), False, False),
                AccountMeta(Pubkey.from_string(state.event_authority), False, False),
                AccountMeta(Pubkey.from_string(state.program_id), False, False),
                AccountMeta(Pubkey.from_string(state.coin_creator_vault_ata), False, True),
                AccountMeta(Pubkey.from_string(state.coin_creator_vault_authority), False, False),
                AccountMeta(Pubkey.from_string(state.fee_config), False, False),
                AccountMeta(Pubkey.from_string(state.fee_program), False, False),
            ],
        )

        blockhash_started = time.monotonic()
        (
            recent_blockhash,
            last_valid_block_height,
        ) = await self.broadcaster.get_latest_blockhash_async()
        trace_payload["blockhash_ms"] = (time.monotonic() - blockhash_started) * 1000.0

        instructions = [
            set_compute_unit_limit(int(compute_unit_limit)),
            set_compute_unit_price(int(compute_unit_price_micro_lamports)),
            create_base_ata_ix,
            create_quote_ata_ix,
            pump_sell_ix,
        ]
        if close_input_token_account and bool(
            getattr(self.config, "live_close_token_ata_on_full_exit", True)
        ):
            instructions.append(
                self._close_token_account_instruction(
                    user_public_key,
                    state.base_mint,
                    state.base_token_program,
                )
            )
        instructions.extend(
            [
                self._close_token_account_instruction(
                    user_public_key,
                    state.quote_mint,
                    _TOKEN_PROGRAM_ID,
                ),
                transfer(
                    TransferParams(
                        from_pubkey=payer,
                        to_pubkey=Pubkey.from_string(tip_account),
                        lamports=tip_lamports,
                    )
                ),
            ]
        )

        compile_started = time.monotonic()
        message = MessageV0.try_compile(
            payer,
            instructions,
            [],
            Hash.from_string(recent_blockhash),
        )
        unsigned_tx = VersionedTransaction(message, [NullSigner(payer)])
        trace_payload["tx_compile_ms"] = (time.monotonic() - compile_started) * 1000.0
        trace_payload["custom_tx_instruction_count"] = len(instructions)
        trace_payload["manual_tip_account"] = tip_account
        trace_payload["manual_tip_lamports"] = tip_lamports
        trace_payload["manual_close_input_token_account"] = bool(
            close_input_token_account
            and bool(getattr(self.config, "live_close_token_ata_on_full_exit", True))
        )
        trace_payload["manual_compute_unit_limit"] = int(compute_unit_limit)
        trace_payload["manual_compute_unit_price_micro_lamports"] = int(
            compute_unit_price_micro_lamports
        )

        return SwapTransaction(
            raw_transaction=bytes(unsigned_tx),
            last_valid_block_height=last_valid_block_height,
            in_amount=int(token_amount),
            out_amount=int(min_quote_amount_out),
            built_at=self._now_iso(),
        )

    def _try_native_pump_amm_buy_tx(
        self,
        *,
        token_mint: str,
        amount_lamports: int,
        order_policy: LiveOrderPolicy,
        trace: dict[str, Any],
    ) -> SwapTransaction | None:
        state = self._native_pump_pool_state(token_mint=token_mint, trace=trace)
        if state is None:
            return None
        try:
            return self._build_native_pump_amm_buy_tx(
                token_mint=token_mint,
                amount_lamports=amount_lamports,
                order_policy=order_policy,
                trace=trace,
            )
        except Exception as exc:  # noqa: BLE001
            trace["native_pump_fallback_reason"] = str(exc)
            logger.warning("native Pump AMM buy fallback for %s: %s", token_mint[:12], exc)
            return None

    async def _try_native_pump_amm_buy_tx_async(
        self,
        *,
        token_mint: str,
        amount_lamports: int,
        order_policy: LiveOrderPolicy,
        trace: dict[str, Any],
    ) -> SwapTransaction | None:
        state = self._native_pump_pool_state(token_mint=token_mint, trace=trace)
        if state is None:
            return None
        try:
            return await self._build_native_pump_amm_buy_tx_async(
                token_mint=token_mint,
                amount_lamports=amount_lamports,
                order_policy=order_policy,
                trace=trace,
            )
        except Exception as exc:  # noqa: BLE001
            trace["native_pump_fallback_reason"] = str(exc)
            logger.warning("native Pump AMM buy async fallback for %s: %s", token_mint[:12], exc)
            return None

    def _try_native_pump_amm_sell_tx(
        self,
        *,
        token_mint: str,
        token_amount: int,
        order_policy: LiveOrderPolicy,
        close_input_token_account: bool,
        trace: dict[str, Any],
    ) -> SwapTransaction | None:
        if self._is_pump_dead(token_mint):
            trace["native_pump_fallback_reason"] = "pump_dead_marker"
            return None
        state = self._native_pump_pool_state(token_mint=token_mint, trace=trace)
        if state is None:
            return None
        if not self._probe_pump_pool_live(
            pool_pubkey=state.pool, token_mint=token_mint, trace=trace
        ):
            trace["native_pump_fallback_reason"] = "pool_state_missing_pre_sell"
            return None
        try:
            return self._build_native_pump_amm_sell_tx(
                token_mint=token_mint,
                token_amount=token_amount,
                order_policy=order_policy,
                close_input_token_account=close_input_token_account,
                trace=trace,
            )
        except Exception as exc:  # noqa: BLE001
            trace["native_pump_fallback_reason"] = str(exc)
            logger.warning("native Pump AMM sell fallback for %s: %s", token_mint[:12], exc)
            return None

    async def _try_native_pump_amm_sell_tx_async(
        self,
        *,
        token_mint: str,
        token_amount: int,
        order_policy: LiveOrderPolicy,
        close_input_token_account: bool,
        trace: dict[str, Any],
    ) -> SwapTransaction | None:
        if self._is_pump_dead(token_mint):
            trace["native_pump_fallback_reason"] = "pump_dead_marker"
            return None
        state = self._native_pump_pool_state(token_mint=token_mint, trace=trace)
        if state is None:
            return None
        if not self._probe_pump_pool_live(
            pool_pubkey=state.pool, token_mint=token_mint, trace=trace
        ):
            trace["native_pump_fallback_reason"] = "pool_state_missing_pre_sell"
            return None
        try:
            return await self._build_native_pump_amm_sell_tx_async(
                token_mint=token_mint,
                token_amount=token_amount,
                order_policy=order_policy,
                close_input_token_account=close_input_token_account,
                trace=trace,
            )
        except Exception as exc:  # noqa: BLE001
            trace["native_pump_fallback_reason"] = str(exc)
            logger.warning("native Pump AMM sell async fallback for %s: %s", token_mint[:12], exc)
            return None

    def _reconcile_fill_from_tx(
        self,
        tx_result: dict[str, Any] | None,
        *,
        signature: str,
        token_mint: str,
        expected_in_amount: int,
        expected_out_amount: int,
        fallback_slot: int | None = None,
    ) -> LiveFillReconciliation:
        if not tx_result:
            raise LiveExecutionError("getTransaction returned no result")

        wallet_pubkey = str(self.signer.get_public_key() or "")
        meta = tx_result.get("meta") or {}
        tx = tx_result.get("transaction") or {}
        message = tx.get("message") or {}
        account_keys = list(message.get("accountKeys") or [])
        wallet_index: int | None = None
        for index, entry in enumerate(account_keys):
            if self._account_pubkey(entry) == wallet_pubkey:
                wallet_index = index
                break

        pre_balances = list(meta.get("preBalances") or [])
        post_balances = list(meta.get("postBalances") or [])
        wallet_pre = 0
        wallet_post = 0
        if (
            wallet_index is not None
            and wallet_index < len(pre_balances)
            and wallet_index < len(post_balances)
        ):
            wallet_pre = int(pre_balances[wallet_index] or 0)
            wallet_post = int(post_balances[wallet_index] or 0)
        wallet_delta = wallet_post - wallet_pre

        pre_token_rows = list(meta.get("preTokenBalances") or [])
        post_token_rows = list(meta.get("postTokenBalances") or [])
        token_pre_raw, token_decimals = self._token_raw_balance(
            pre_token_rows, wallet_pubkey, token_mint
        )
        token_post_raw, token_post_decimals = self._token_raw_balance(
            post_token_rows, wallet_pubkey, token_mint
        )
        token_delta = token_post_raw - token_pre_raw

        # Fallback: if owner-filtered pre/post both resolve to 0 but the TX did
        # land (common when the source ATA is closed inside the same TX on a
        # full-exit sell), walk innerInstructions for parsed SPL-token
        # transfers. Without this, a successful sell shows token_delta=0 and
        # gets reported as "no fill" even when SOL was received.
        if token_pre_raw == 0 and token_post_raw == 0:
            inner_sent, inner_received = self._token_amount_from_inner_instructions(
                meta,
                wallet_pubkey=wallet_pubkey,
                token_mint=token_mint,
            )
            if inner_received and not inner_sent:
                token_post_raw = inner_received
                token_delta = inner_received
            elif inner_sent and not inner_received:
                token_pre_raw = inner_sent
                token_delta = -inner_sent

        actual_tip_lamports = self._actual_tip_lamports_from_tx(tx_result)
        actual_output_amount = token_delta if token_delta > 0 else max(wallet_delta, 0)

        return LiveFillReconciliation(
            signature=signature,
            wallet_pubkey=wallet_pubkey,
            token_mint=token_mint,
            slot=int(tx_result.get("slot") or fallback_slot or 0) or fallback_slot,
            fee_lamports=int(meta.get("fee") or 0),
            wallet_pre_lamports=wallet_pre,
            wallet_post_lamports=wallet_post,
            wallet_delta_lamports=wallet_delta,
            token_pre_raw=token_pre_raw,
            token_post_raw=token_post_raw,
            token_delta_raw=token_delta,
            token_decimals=token_post_decimals or token_decimals,
            expected_in_amount=expected_in_amount,
            expected_out_amount=expected_out_amount,
            actual_output_amount=actual_output_amount,
            quote_out_amount_diff=actual_output_amount - int(expected_out_amount or 0),
            tip_lamports=actual_tip_lamports,
            transaction_error=meta.get("err"),
            raw=tx_result,
        )

    def _reconcile_confirmed_fill(
        self,
        *,
        signature: str,
        token_mint: str,
        expected_in_amount: int,
        expected_out_amount: int,
        slot: int | None = None,
    ) -> LiveFillReconciliation:
        return self._reconcile_fill_from_tx(
            self.broadcaster.get_transaction(signature),
            signature=signature,
            token_mint=token_mint,
            expected_in_amount=expected_in_amount,
            expected_out_amount=expected_out_amount,
            fallback_slot=slot,
        )

    async def _reconcile_confirmed_fill_async(
        self,
        *,
        signature: str,
        token_mint: str,
        expected_in_amount: int,
        expected_out_amount: int,
        slot: int | None = None,
    ) -> LiveFillReconciliation:
        tx_result = await self.broadcaster.get_transaction_async(signature)
        return self._reconcile_fill_from_tx(
            tx_result,
            signature=signature,
            token_mint=token_mint,
            expected_in_amount=expected_in_amount,
            expected_out_amount=expected_out_amount,
            fallback_slot=slot,
        )

    def _result_with_error(
        self,
        *,
        error: str,
        signature: str | None = None,
        in_amount: int = 0,
        out_amount: int = 0,
        slot: int | None = None,
        latency_trace: dict[str, Any] | None = None,
        reconciliation_error: str | None = None,
        reconciliation: LiveFillReconciliation | None = None,
    ) -> LiveTradeResult:
        trace = dict(latency_trace or {})
        try:
            started = float(trace.get("__total_started_monotonic", 0.0) or 0.0)
        except (TypeError, ValueError):
            started = 0.0
        if "total_execution_ms" not in trace and started > 0:
            trace["total_execution_ms"] = (time.monotonic() - started) * 1000.0
        trace.pop("__total_started_monotonic", None)
        return LiveTradeResult(
            success=False,
            signature=signature,
            in_amount=in_amount,
            out_amount=out_amount,
            slot=slot,
            error=error,
            reconciliation_error=reconciliation_error,
            reconciliation=reconciliation,
            latency_trace=trace,
        )

    def _result_success(
        self,
        *,
        signature: str,
        in_amount: int,
        out_amount: int,
        slot: int | None,
        latency_trace: dict[str, Any],
        reconciliation: LiveFillReconciliation | None,
        reconciliation_error: str | None = None,
    ) -> LiveTradeResult:
        trace = dict(latency_trace)
        trace.pop("__total_started_monotonic", None)
        return LiveTradeResult(
            success=True,
            signature=signature,
            in_amount=in_amount,
            out_amount=out_amount,
            slot=slot,
            reconciliation=reconciliation,
            reconciliation_error=reconciliation_error,
            latency_trace=trace,
        )

    def _reconcile_failed_transaction(
        self,
        *,
        signature: str | None,
        token_mint: str,
        expected_in_amount: int,
        expected_out_amount: int,
        slot: int | None,
        trace: dict[str, Any],
    ) -> tuple[LiveFillReconciliation | None, str | None]:
        if not signature:
            return None, None
        started = time.monotonic()
        reconciliation: LiveFillReconciliation | None = None
        reconciliation_error: str | None = None
        try:
            reconciliation = self._reconcile_confirmed_fill(
                signature=signature,
                token_mint=token_mint,
                expected_in_amount=expected_in_amount,
                expected_out_amount=expected_out_amount,
                slot=slot,
            )
        except Exception as exc:  # noqa: BLE001
            reconciliation_error = str(exc)
        trace["reconcile_ms"] = (time.monotonic() - started) * 1000.0
        if reconciliation is not None:
            trace["failed_tx_wallet_delta_lamports"] = int(reconciliation.wallet_delta_lamports)
            trace["failed_tx_fee_lamports"] = int(reconciliation.fee_lamports)
        return reconciliation, reconciliation_error

    async def _reconcile_failed_transaction_async(
        self,
        *,
        signature: str | None,
        token_mint: str,
        expected_in_amount: int,
        expected_out_amount: int,
        slot: int | None,
        trace: dict[str, Any],
    ) -> tuple[LiveFillReconciliation | None, str | None]:
        if not signature:
            return None, None
        started = time.monotonic()
        reconciliation: LiveFillReconciliation | None = None
        reconciliation_error: str | None = None
        try:
            reconciliation = await self._reconcile_confirmed_fill_async(
                signature=signature,
                token_mint=token_mint,
                expected_in_amount=expected_in_amount,
                expected_out_amount=expected_out_amount,
                slot=slot,
            )
        except Exception as exc:  # noqa: BLE001
            reconciliation_error = str(exc)
        trace["reconcile_ms"] = (time.monotonic() - started) * 1000.0
        if reconciliation is not None:
            trace["failed_tx_wallet_delta_lamports"] = int(reconciliation.wallet_delta_lamports)
            trace["failed_tx_fee_lamports"] = int(reconciliation.fee_lamports)
        return reconciliation, reconciliation_error

    def _prefetch_age_ms(self, prebuilt_tx: "SwapTransaction") -> float | None:
        built_at = getattr(prebuilt_tx, "built_at", None)
        if not built_at:
            return None
        try:
            built_dt = datetime.fromisoformat(str(built_at).replace("Z", "+00:00"))
        except ValueError:
            return None
        if built_dt.tzinfo is None:
            built_dt = built_dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(tz=timezone.utc) - built_dt).total_seconds() * 1000.0)

    # ------------------------------------------------------------------
    # Pre-flight safeguard checks
    # ------------------------------------------------------------------

    def _preflight_checks(
        self,
        size_sol: float,
        current_exposure_sol: float,
        open_position_count: int,
    ) -> None:
        """Raise :class:`LiveExecutionError` if any safeguard fails."""
        # 1. Signer must be valid
        if self.signer.get_public_key() is None:
            raise LiveExecutionError("Signer not validated – refusing live execution")

        # 2. Position size limit
        if size_sol > self.config.max_position_sol:
            raise LiveExecutionError(
                f"Position size {size_sol:.4f} SOL exceeds max_position_sol "
                f"{self.config.max_position_sol:.4f}"
            )

        # 3. Total exposure limit
        if current_exposure_sol + size_sol > self.config.max_total_exposure_sol:
            raise LiveExecutionError(
                f"Exposure {current_exposure_sol + size_sol:.4f} SOL would exceed "
                f"max_total_exposure_sol {self.config.max_total_exposure_sol:.4f}"
            )

        # 4. Open position count
        if open_position_count >= self.config.max_open_positions:
            raise LiveExecutionError(
                f"Open positions ({open_position_count}) at max ({self.config.max_open_positions})"
            )

        # 5. Required env vars
        if not self.config.helius_rpc_url:
            raise LiveExecutionError("HELIUS_RPC_URL is required for live execution")
        if not self.config.jupiter_base_url:
            raise LiveExecutionError("JUPITER_BASE_URL is required for live execution")

    # ------------------------------------------------------------------
    # Live fee / slippage resolution
    # ------------------------------------------------------------------

    def _sender_jito_tip_floor(self) -> int:
        mode = str(getattr(self.config, "live_broadcast_mode", "staked_rpc") or "staked_rpc")
        if mode in {"helius_sender", "helius_bundle"}:
            return 200_000
        if mode == "helius_sender_swqos":
            return 5_000
        return 0

    def _resolve_fee_plan(self, *, strategy: str | None = None) -> LiveFeePlan:
        """Resolve dynamic priority fee and Jito tip floors for live sending.

        ``strategy="main"`` applies the mature-pair overrides. Mature-pair edges
        are thin (5–15%) and can't absorb sniper-grade landing fees; dynamic
        escalation is disabled by default there.
        """
        is_main = strategy == "main"
        if is_main:
            priority_floor = max(
                int(
                    getattr(
                        self.config,
                        "live_priority_fee_lamports_main",
                        self.config.priority_fee_lamports,
                    )
                    or 0
                ),
                0,
            )
            jito_floor = max(
                int(getattr(self.config, "live_jito_tip_lamports_main", 0) or 0),
                self._sender_jito_tip_floor(),
            )
            use_dyn_priority = bool(
                getattr(self.config, "live_use_dynamic_priority_fee_main", False)
            )
            use_dyn_jito = bool(getattr(self.config, "live_use_dynamic_jito_tip_main", False))
        else:
            priority_floor = max(int(self.config.priority_fee_lamports), 0)
            jito_floor = max(
                int(getattr(self.config, "jito_tip_lamports", 0) or 0),
                self._sender_jito_tip_floor(),
            )
            use_dyn_priority = bool(getattr(self.config, "live_use_dynamic_priority_fee", True))
            use_dyn_jito = bool(getattr(self.config, "live_use_dynamic_jito_tip", True))
        priority_fee = priority_floor
        jito_tip = jito_floor
        raw_priority_payload: dict[str, Any] | None = None
        try:
            fee_info = self.broadcaster.get_priority_fee_estimate(account_keys=[])
            raw_priority_payload = {"result": fee_info.get("raw")}
            if use_dyn_priority:
                dynamic_priority = max(int(fee_info.get("priority_fee_lamports", 0) or 0), 0)
                priority_fee = max(dynamic_priority, priority_floor)
            if use_dyn_jito:
                dynamic_jito = max(int(fee_info.get("jito_tip_lamports", 0) or 0), 0)
                jito_tip = max(dynamic_jito, jito_floor)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "live fee plan fallback to config floors (priority=%d jito=%d): %s",
                priority_floor,
                jito_floor,
                exc,
            )
        return LiveFeePlan(
            priority_fee_lamports=priority_fee,
            jito_tip_lamports=jito_tip,
            raw_priority_payload=raw_priority_payload,
        )

    def _resolve_slippage_bps(
        self,
        *,
        side: str,
        strategy: str | None = None,
        source_program: str | None = None,
    ) -> int | None:
        sniper_pump_fun = (strategy or "").lower() == "sniper" and (
            source_program or ""
        ).upper() == "PUMP_FUN"
        if side == "buy":
            if sniper_pump_fun:
                override = int(
                    getattr(self.config, "live_buy_slippage_bps_sniper_pump_fun", 0) or 0
                )
                if override > 0:
                    return override
            configured = int(getattr(self.config, "live_buy_slippage_bps", 0) or 0)
        else:
            if sniper_pump_fun:
                override = int(
                    getattr(self.config, "live_sell_slippage_bps_sniper_pump_fun", 0) or 0
                )
                if override > 0:
                    return override
            configured = int(getattr(self.config, "live_sell_slippage_bps", 0) or 0)
        if configured > 0:
            return configured
        if bool(getattr(self.config, "live_use_jupiter_auto_slippage", True)):
            return None
        return max(int(getattr(self.config, "default_slippage_bps", 150) or 150), 0)

    def _resolve_order_policy(
        self,
        *,
        side: str,
        strategy: str | None = None,
        source_program: str | None = None,
    ) -> LiveOrderPolicy:
        fee_plan = self._resolve_fee_plan(strategy=strategy)
        return LiveOrderPolicy(
            slippage_bps=self._resolve_slippage_bps(
                side=side, strategy=strategy, source_program=source_program
            ),
            priority_fee_lamports=fee_plan.priority_fee_lamports,
            jito_tip_lamports=fee_plan.jito_tip_lamports,
            broadcast_fee_type=str(
                getattr(self.config, "live_broadcast_fee_type", "maxCap") or "maxCap"
            ),
            raw_priority_payload=fee_plan.raw_priority_payload,
            strategy=strategy,
            source_program=source_program,
        )

    def _refresh_order_policy_for_account_keys(
        self,
        *,
        order_policy: LiveOrderPolicy,
        account_keys: list[str],
        trace: dict[str, Any],
    ) -> LiveOrderPolicy:
        scoped_keys = [str(key) for key in account_keys if str(key)]
        if not scoped_keys:
            return order_policy
        if not bool(getattr(self.config, "live_use_dynamic_priority_fee", True)) and not bool(
            getattr(self.config, "live_use_dynamic_jito_tip", True)
        ):
            return order_policy
        started = time.monotonic()
        try:
            fee_info = self.broadcaster.get_priority_fee_estimate(account_keys=scoped_keys)
        except Exception as exc:  # noqa: BLE001
            trace["priority_fee_account_scoped_error"] = str(exc)
            return order_policy
        trace["priority_fee_account_scoped_ms"] = (time.monotonic() - started) * 1000.0
        trace["priority_fee_scope"] = "account_keys"
        trace["priority_fee_account_key_count"] = len(scoped_keys)
        priority_fee = int(order_policy.priority_fee_lamports)
        jito_tip = int(order_policy.jito_tip_lamports)
        if bool(getattr(self.config, "live_use_dynamic_priority_fee", True)):
            priority_fee = max(priority_fee, int(fee_info.get("priority_fee_lamports", 0) or 0))
        if bool(getattr(self.config, "live_use_dynamic_jito_tip", True)):
            jito_tip = max(jito_tip, int(fee_info.get("jito_tip_lamports", 0) or 0))
        trace["priority_fee_account_scoped_lamports"] = int(priority_fee)
        trace["jito_tip_account_scoped_lamports"] = int(jito_tip)
        return LiveOrderPolicy(
            slippage_bps=order_policy.slippage_bps,
            priority_fee_lamports=priority_fee,
            jito_tip_lamports=jito_tip,
            broadcast_fee_type=order_policy.broadcast_fee_type,
            raw_priority_payload={"result": fee_info.get("raw")},
            strategy=order_policy.strategy,
            source_program=order_policy.source_program,
        )

    async def _refresh_order_policy_for_account_keys_async(
        self,
        *,
        order_policy: LiveOrderPolicy,
        account_keys: list[str],
        trace: dict[str, Any],
    ) -> LiveOrderPolicy:
        scoped_keys = [str(key) for key in account_keys if str(key)]
        if not scoped_keys:
            return order_policy
        if not bool(getattr(self.config, "live_use_dynamic_priority_fee", True)) and not bool(
            getattr(self.config, "live_use_dynamic_jito_tip", True)
        ):
            return order_policy
        started = time.monotonic()
        try:
            fee_info = await self.broadcaster.get_priority_fee_estimate_async(
                account_keys=scoped_keys
            )
        except Exception as exc:  # noqa: BLE001
            trace["priority_fee_account_scoped_error"] = str(exc)
            return order_policy
        trace["priority_fee_account_scoped_ms"] = (time.monotonic() - started) * 1000.0
        trace["priority_fee_scope"] = "account_keys"
        trace["priority_fee_account_key_count"] = len(scoped_keys)
        priority_fee = int(order_policy.priority_fee_lamports)
        jito_tip = int(order_policy.jito_tip_lamports)
        if bool(getattr(self.config, "live_use_dynamic_priority_fee", True)):
            priority_fee = max(priority_fee, int(fee_info.get("priority_fee_lamports", 0) or 0))
        if bool(getattr(self.config, "live_use_dynamic_jito_tip", True)):
            jito_tip = max(jito_tip, int(fee_info.get("jito_tip_lamports", 0) or 0))
        trace["priority_fee_account_scoped_lamports"] = int(priority_fee)
        trace["jito_tip_account_scoped_lamports"] = int(jito_tip)
        return LiveOrderPolicy(
            slippage_bps=order_policy.slippage_bps,
            priority_fee_lamports=priority_fee,
            jito_tip_lamports=jito_tip,
            broadcast_fee_type=order_policy.broadcast_fee_type,
            raw_priority_payload={"result": fee_info.get("raw")},
            strategy=order_policy.strategy,
            source_program=order_policy.source_program,
        )

    @staticmethod
    def _apply_broadcast_trace(trace: dict[str, Any], result: BroadcastResult) -> None:
        trace["broadcast_transport"] = str(getattr(result, "transport", "") or "")
        trace["broadcast_send_attempts"] = int(getattr(result, "send_attempts", 0) or 0)
        trace["broadcast_send_ms"] = float(result.send_latency_ms)
        trace["broadcast_confirm_ms"] = float(result.confirm_latency_ms)
        trace["broadcast_total_ms"] = float(result.total_latency_ms)
        trace["broadcast_sent_at"] = result.sent_at
        trace["broadcast_tip_account"] = getattr(result, "validated_tip_account", None)
        trace["broadcast_tip_lamports"] = int(getattr(result, "validated_tip_lamports", 0) or 0)
        trace["broadcast_has_compute_unit_price"] = bool(
            getattr(result, "validated_has_compute_unit_price", False)
        )
        if result.confirmed_at:
            trace["broadcast_confirmed_at"] = result.confirmed_at
        if getattr(result, "bundle_id", None):
            trace["bundle_id"] = result.bundle_id

    # Jupiter error substrings that indicate the failure has nothing to do
    # with account-config — retrying with a different variant will fail for
    # the same reason. Early-exit the fallback chain on these to cut 400
    # volume and Jupiter rate-limit pressure. Error reasons that *can* vary
    # with account config (missing/uninitialized ATAs, InvalidAccountData
    # on a user account) are intentionally absent — those still benefit
    # from variant iteration.
    _JUPITER_UNROUTABLE_MARKERS: tuple[str, ...] = (
        "COULD_NOT_FIND_ANY_ROUTE",
        "ROUTE_NOT_FOUND",
        "NO_ROUTES_FOUND",
        "INSUFFICIENT_LIQUIDITY",
        "TOKEN_NOT_TRADABLE",
        "Cannot compute other amount threshold",
        "NotEnoughAccountKeys",
    )
    # When Jupiter rate-limits us, hammering the fallback chain just
    # accumulates more 429s. Back off immediately — the caller will retry
    # the whole trade on the next tick.
    _JUPITER_RATE_LIMIT_MARKERS: tuple[str, ...] = (
        "HTTP 429",
        "Too many requests",
    )

    @classmethod
    def _jupiter_error_is_terminal(cls, exc: Exception) -> str | None:
        """Return a short tag if this error shouldn't be retried across variants."""
        text = str(exc)
        for marker in cls._JUPITER_UNROUTABLE_MARKERS:
            if marker in text:
                return "unroutable"
        for marker in cls._JUPITER_RATE_LIMIT_MARKERS:
            if marker in text:
                return "rate_limited"
        return None

    def _requires_custom_jupiter_tx(self) -> bool:
        """Return whether the transport requires a self-built Sender-compliant tx."""
        mode = str(getattr(self.broadcaster, "broadcast_mode", "staked_rpc") or "staked_rpc")
        return mode in {"helius_sender", "helius_sender_swqos", "helius_bundle"}

    def _is_bundle_mode(self) -> bool:
        """Return whether broadcast mode is raw bundle submission."""
        mode = str(getattr(self.broadcaster, "broadcast_mode", "staked_rpc") or "staked_rpc")
        return mode == "helius_bundle"

    def _custom_tx_slippage_bps(self, requested_slippage_bps: int | None) -> int:
        """Swap-instructions path needs an explicit slippage value."""
        if requested_slippage_bps is not None and int(requested_slippage_bps) > 0:
            return int(requested_slippage_bps)
        return max(int(getattr(self.config, "default_slippage_bps", 150) or 150), 1)

    def _swap_instruction_attempt_plan(
        self,
        *,
        input_mint: str,
        output_mint: str,
    ) -> list[dict[str, Any]]:
        configured_shared = bool(getattr(self.config, "live_use_shared_accounts", False))
        is_sell = input_mint != SOL_MINT and output_mint == SOL_MINT
        preferred_shared = False if is_sell else configured_shared
        variants = [
            {
                "use_shared_accounts": preferred_shared,
                "skip_user_accounts_rpc_calls": False,
                "label": "preferred",
            },
            {
                "use_shared_accounts": preferred_shared,
                "skip_user_accounts_rpc_calls": True,
                "label": "preferred_skip_user_accounts_rpc_calls",
            },
            {
                "use_shared_accounts": not preferred_shared,
                "skip_user_accounts_rpc_calls": False,
                "label": "flipped_shared_accounts",
            },
            {
                "use_shared_accounts": not preferred_shared,
                "skip_user_accounts_rpc_calls": True,
                "label": "flipped_shared_accounts_skip_user_accounts_rpc_calls",
            },
        ]
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[bool, bool]] = set()
        for variant in variants:
            key = (
                bool(variant["use_shared_accounts"]),
                bool(variant["skip_user_accounts_rpc_calls"]),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(variant)
        return deduped

    def _get_swap_instructions_with_fallback(
        self,
        *,
        quote_response: dict[str, Any],
        user_public_key: str,
        input_mint: str,
        output_mint: str,
        trace: dict[str, Any],
    ) -> dict[str, Any]:
        attempts = self._swap_instruction_attempt_plan(
            input_mint=input_mint, output_mint=output_mint
        )
        errors: list[str] = []
        for index, attempt in enumerate(attempts, start=1):
            try:
                payload = self.jupiter.get_swap_instructions(
                    quote_response=quote_response,
                    user_public_key=user_public_key,
                    dynamic_compute_unit_limit=True,
                    wrap_and_unwrap_sol=True,
                    as_legacy_transaction=False,
                    use_shared_accounts=bool(attempt["use_shared_accounts"]),
                    skip_user_accounts_rpc_calls=bool(attempt["skip_user_accounts_rpc_calls"]),
                )
                trace["swap_instructions_attempts"] = index
                trace["swap_instructions_variant"] = str(attempt["label"])
                trace["swap_instructions_use_shared_accounts"] = bool(
                    attempt["use_shared_accounts"]
                )
                trace["swap_instructions_skip_user_accounts_rpc_calls"] = bool(
                    attempt["skip_user_accounts_rpc_calls"]
                )
                if errors:
                    trace["swap_instructions_prior_errors"] = errors[-3:]
                return payload
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"{attempt['label']}: shared={bool(attempt['use_shared_accounts'])} "
                    f"skip_user_accounts_rpc_calls={bool(attempt['skip_user_accounts_rpc_calls'])} -> {exc}"
                )
                terminal = self._jupiter_error_is_terminal(exc)
                if terminal is not None:
                    trace["swap_instructions_attempts"] = index
                    trace["swap_instructions_errors"] = errors[-4:]
                    trace["swap_instructions_early_exit"] = terminal
                    raise LiveExecutionError(
                        f"Jupiter swap-instructions {terminal} (early-exit after {index}/{len(attempts)}): {exc}"
                    )
        trace["swap_instructions_attempts"] = len(attempts)
        trace["swap_instructions_errors"] = errors[-4:]
        raise LiveExecutionError(
            "Jupiter swap-instructions failed across all variants: " + " | ".join(errors[-4:])
        )

    async def _get_swap_instructions_with_fallback_async(
        self,
        *,
        quote_response: dict[str, Any],
        user_public_key: str,
        input_mint: str,
        output_mint: str,
        trace: dict[str, Any],
    ) -> dict[str, Any]:
        attempts = self._swap_instruction_attempt_plan(
            input_mint=input_mint, output_mint=output_mint
        )
        errors: list[str] = []
        for index, attempt in enumerate(attempts, start=1):
            try:
                payload = await self.jupiter.get_swap_instructions_async(
                    quote_response=quote_response,
                    user_public_key=user_public_key,
                    dynamic_compute_unit_limit=True,
                    wrap_and_unwrap_sol=True,
                    as_legacy_transaction=False,
                    use_shared_accounts=bool(attempt["use_shared_accounts"]),
                    skip_user_accounts_rpc_calls=bool(attempt["skip_user_accounts_rpc_calls"]),
                )
                trace["swap_instructions_attempts"] = index
                trace["swap_instructions_variant"] = str(attempt["label"])
                trace["swap_instructions_use_shared_accounts"] = bool(
                    attempt["use_shared_accounts"]
                )
                trace["swap_instructions_skip_user_accounts_rpc_calls"] = bool(
                    attempt["skip_user_accounts_rpc_calls"]
                )
                if errors:
                    trace["swap_instructions_prior_errors"] = errors[-3:]
                return payload
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"{attempt['label']}: shared={bool(attempt['use_shared_accounts'])} "
                    f"skip_user_accounts_rpc_calls={bool(attempt['skip_user_accounts_rpc_calls'])} -> {exc}"
                )
                terminal = self._jupiter_error_is_terminal(exc)
                if terminal is not None:
                    trace["swap_instructions_attempts"] = index
                    trace["swap_instructions_errors"] = errors[-4:]
                    trace["swap_instructions_early_exit"] = terminal
                    raise LiveExecutionError(
                        f"Jupiter swap-instructions {terminal} (early-exit after {index}/{len(attempts)}): {exc}"
                    )
        trace["swap_instructions_attempts"] = len(attempts)
        trace["swap_instructions_errors"] = errors[-4:]
        raise LiveExecutionError(
            "Jupiter swap-instructions failed across all variants: " + " | ".join(errors[-4:])
        )

    @staticmethod
    def _extract_compute_unit_limit(instruction_payloads: list[dict[str, Any]]) -> int:
        default_limit = 1_400_000
        for payload in instruction_payloads:
            if str(payload.get("programId") or "") != "ComputeBudget111111111111111111111111111111":
                continue
            try:
                raw = base64.b64decode(str(payload.get("data") or ""))
            except Exception:  # noqa: BLE001
                continue
            if len(raw) >= 5 and raw[:1] == b"\x02":
                try:
                    return max(int.from_bytes(raw[1:5], "little", signed=False), 1)
                except Exception:  # noqa: BLE001
                    continue
        return default_limit

    @staticmethod
    def _instruction_payload_account_keys(
        instructions_payload: dict[str, Any],
    ) -> list[str]:
        keys: list[str] = []

        def _add_instruction(payload: Any) -> None:
            if not isinstance(payload, dict):
                return
            program_id = str(payload.get("programId") or "")
            if program_id:
                keys.append(program_id)
            for account in list(payload.get("accounts") or []):
                if isinstance(account, dict):
                    pubkey = str(account.get("pubkey") or "")
                else:
                    pubkey = str(account or "")
                if pubkey:
                    keys.append(pubkey)

        for payload in list(instructions_payload.get("computeBudgetInstructions") or []):
            _add_instruction(payload)
        for payload in list(instructions_payload.get("setupInstructions") or []):
            _add_instruction(payload)
        for payload in list(instructions_payload.get("otherInstructions") or []):
            _add_instruction(payload)
        _add_instruction(instructions_payload.get("swapInstruction"))
        _add_instruction(instructions_payload.get("cleanupInstruction"))
        for address in list(instructions_payload.get("addressLookupTableAddresses") or []):
            if str(address or ""):
                keys.append(str(address))
        deduped: list[str] = []
        seen: set[str] = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    @staticmethod
    def _priority_fee_to_micro_lamports(priority_fee_lamports: int, compute_unit_limit: int) -> int:
        units = max(int(compute_unit_limit or 0), 1)
        lamports = max(int(priority_fee_lamports or 0), 1)
        return max((lamports * 1_000_000 + units - 1) // units, 1)

    def _select_tip_account(self) -> str:
        """Choose one configured Jito tip account."""
        tip_accounts = tuple(getattr(self.config, "jito_tip_accounts", ()) or ())
        if not tip_accounts:
            tip_accounts = tuple(getattr(self.broadcaster, "jito_tip_accounts", ()) or ())
        if not tip_accounts:
            raise LiveExecutionError("No Jito tip accounts configured for Sender live execution")
        return str(tip_accounts[time.time_ns() % len(tip_accounts)])

    @staticmethod
    def _deserialize_instruction_payload(payload: dict[str, Any]):
        from solders.instruction import AccountMeta, Instruction  # type: ignore[import-untyped]
        from solders.pubkey import Pubkey  # type: ignore[import-untyped]

        program_id = Pubkey.from_string(str(payload.get("programId") or ""))
        data = base64.b64decode(str(payload.get("data") or ""))
        accounts = [
            AccountMeta(
                pubkey=Pubkey.from_string(str(account.get("pubkey") or "")),
                is_signer=bool(account.get("isSigner", False)),
                is_writable=bool(account.get("isWritable", False)),
            )
            for account in list(payload.get("accounts") or [])
        ]
        return Instruction(program_id, data, accounts)

    @staticmethod
    def _is_compute_unit_price_instruction(program_id: str, data: bytes) -> bool:
        return program_id == "ComputeBudget111111111111111111111111111111" and data[:1] == b"\x03"

    def _resolve_loaded_account_keys(self, message: Any, alt_accounts: list[Any]) -> list[Any]:
        lookup_by_key = {str(alt.key): alt for alt in alt_accounts}
        full_keys = list(message.account_keys)
        for lookup in list(getattr(message, "address_table_lookups", []) or []):
            alt = lookup_by_key.get(str(lookup.account_key))
            if alt is None:
                raise LiveExecutionError(f"missing ALT account for {lookup.account_key}")
            addresses = list(alt.addresses)
            for index in list(lookup.writable_indexes):
                full_keys.append(addresses[int(index)])
            for index in list(lookup.readonly_indexes):
                full_keys.append(addresses[int(index)])
        return full_keys

    def _compiled_instruction_to_instruction(
        self, compiled_instruction: Any, message: Any, full_keys: list[Any]
    ):
        from solders.instruction import AccountMeta, Instruction  # type: ignore[import-untyped]

        program_index = int(compiled_instruction.program_id_index)
        if program_index < 0 or program_index >= len(full_keys):
            raise LiveExecutionError(
                f"compiled instruction program index out of range: {program_index}"
            )
        program_id = full_keys[program_index]
        account_indexes = list(bytes(compiled_instruction.accounts))
        accounts = []
        for index in account_indexes:
            if index < 0 or index >= len(full_keys):
                raise LiveExecutionError(
                    f"compiled instruction account index out of range: {index}"
                )
            accounts.append(
                AccountMeta(
                    pubkey=full_keys[index],
                    is_signer=bool(message.is_signer(index)),
                    is_writable=bool(message.is_maybe_writable(index)),
                )
            )
        return Instruction(program_id, bytes(compiled_instruction.data), accounts)

    def _is_existing_tip_transfer(
        self,
        *,
        program_id: str,
        data: bytes,
        account_indexes: list[int],
        full_keys: list[Any],
    ) -> bool:
        if program_id != "11111111111111111111111111111111":
            return False
        if len(data) < 12 or data[:4] != b"\x02\x00\x00\x00":
            return False
        if len(account_indexes) < 2:
            return False
        try:
            destination = str(full_keys[account_indexes[1]])
        except Exception:  # noqa: BLE001
            return False
        if not self.broadcaster.jito_tip_accounts:
            return False
        return destination in self.broadcaster.jito_tip_accounts

    def _rebuild_order_transaction_for_sender(
        self,
        *,
        order: Any,
        input_mint: str,
        output_mint: str,
        order_policy: LiveOrderPolicy,
        close_input_token_account: bool = False,
        trace: dict[str, Any] | None = None,
    ) -> SwapTransaction:
        from solders.compute_budget import set_compute_unit_price  # type: ignore[import-untyped]
        from solders.message import MessageV0  # type: ignore[import-untyped]
        from solders.null_signer import NullSigner  # type: ignore[import-untyped]
        from solders.pubkey import Pubkey  # type: ignore[import-untyped]
        from solders.system_program import TransferParams, transfer  # type: ignore[import-untyped]
        from solders.transaction import VersionedTransaction  # type: ignore[import-untyped]

        user_public_key = str(self.signer.get_public_key() or "")
        if not user_public_key:
            raise LiveExecutionError("Signer not validated – cannot rebuild Jupiter order tx")

        trace_payload = trace if trace is not None else {}
        tx_decode_started = time.monotonic()
        order_tx = VersionedTransaction.from_bytes(order.raw_transaction)
        message = order_tx.message
        trace_payload["jupiter_order_decode_ms"] = (time.monotonic() - tx_decode_started) * 1000.0

        alt_addresses = [
            str(lookup.account_key)
            for lookup in list(getattr(message, "address_table_lookups", []) or [])
        ]
        alt_started = time.monotonic()
        alt_accounts = self.broadcaster.get_address_lookup_table_accounts(alt_addresses)
        trace_payload["alt_fetch_ms"] = (time.monotonic() - alt_started) * 1000.0
        trace_payload["alt_account_count"] = len(alt_accounts)
        full_keys = self._resolve_loaded_account_keys(message, alt_accounts)
        order_policy = self._refresh_order_policy_for_account_keys(
            order_policy=order_policy,
            account_keys=[str(key) for key in full_keys],
            trace=trace_payload,
        )

        compute_unit_price_micro_lamports = self._priority_fee_to_micro_lamports(
            order_policy.priority_fee_lamports,
            1_400_000,
        )
        tip_account = self._select_tip_account()
        tip_lamports = max(int(order_policy.jito_tip_lamports or 0), 0)

        payer = Pubkey.from_string(user_public_key)
        instructions = [set_compute_unit_price(int(compute_unit_price_micro_lamports))]
        preserved_instruction_count = 0
        for compiled_instruction in list(message.instructions):
            program_index = int(compiled_instruction.program_id_index)
            if program_index < 0 or program_index >= len(full_keys):
                raise LiveExecutionError(
                    f"compiled instruction program index out of range while rebuilding order tx: {program_index}"
                )
            program_id = str(full_keys[program_index])
            data = bytes(compiled_instruction.data)
            account_indexes = list(bytes(compiled_instruction.accounts))
            if self._is_compute_unit_price_instruction(program_id, data):
                continue
            if self._is_existing_tip_transfer(
                program_id=program_id,
                data=data,
                account_indexes=account_indexes,
                full_keys=full_keys,
            ):
                continue
            instructions.append(
                self._compiled_instruction_to_instruction(compiled_instruction, message, full_keys)
            )
            preserved_instruction_count += 1

        close_token_account_enabled = (
            close_input_token_account
            and bool(getattr(self.config, "live_close_token_ata_on_full_exit", True))
            and input_mint != SOL_MINT
            and output_mint == SOL_MINT
        )
        if close_token_account_enabled:
            input_token_program = self._resolve_mint_token_program(input_mint)
            trace_payload["input_token_program"] = input_token_program
            instructions.append(
                self._close_token_account_instruction(
                    user_public_key, input_mint, input_token_program
                )
            )
        instructions.append(
            transfer(
                TransferParams(
                    from_pubkey=payer,
                    to_pubkey=Pubkey.from_string(tip_account),
                    lamports=tip_lamports,
                )
            )
        )

        trace_payload["jupiter_path"] = "order_tx_rebuild"
        trace_payload["manual_tip_account"] = tip_account
        trace_payload["manual_tip_lamports"] = tip_lamports
        trace_payload["manual_close_input_token_account"] = bool(close_token_account_enabled)
        trace_payload["manual_compute_unit_price_micro_lamports"] = int(
            compute_unit_price_micro_lamports
        )
        trace_payload["rebuild_preserved_instruction_count"] = preserved_instruction_count

        compile_started = time.monotonic()
        rebuilt_message = MessageV0.try_compile(
            payer,
            instructions,
            alt_accounts,
            message.recent_blockhash,
        )
        unsigned_tx = VersionedTransaction(rebuilt_message, [NullSigner(payer)])
        trace_payload["tx_compile_ms"] = (time.monotonic() - compile_started) * 1000.0
        trace_payload["custom_tx_instruction_count"] = len(instructions)

        return SwapTransaction(
            raw_transaction=bytes(unsigned_tx),
            last_valid_block_height=int(order.last_valid_block_height),
            in_amount=int(order.in_amount),
            out_amount=int(order.out_amount),
            built_at=self._now_iso(),
        )

    def _maybe_use_order_tx_passthrough(
        self,
        *,
        order: Any,
        close_input_token_account: bool = False,
        trace: dict[str, Any] | None = None,
    ) -> SwapTransaction | None:
        trace_payload = trace if trace is not None else {}
        try:
            self.broadcaster._validate_low_latency_transaction(order.raw_transaction)
        except Exception as exc:  # noqa: BLE001
            trace_payload["order_tx_passthrough_valid"] = False
            trace_payload["order_tx_passthrough_reason"] = str(exc)
            return None

        trace_payload["order_tx_passthrough_valid"] = True
        trace_payload["jupiter_path"] = "order_tx_passthrough"
        if close_input_token_account:
            trace_payload["passthrough_skipped_manual_close_input_token_account"] = True
        return SwapTransaction(
            raw_transaction=order.raw_transaction,
            last_valid_block_height=int(order.last_valid_block_height),
            in_amount=int(order.in_amount),
            out_amount=int(order.out_amount),
            built_at=self._now_iso(),
        )

    async def _rebuild_order_transaction_for_sender_async(
        self,
        *,
        order: Any,
        input_mint: str,
        output_mint: str,
        order_policy: LiveOrderPolicy,
        close_input_token_account: bool = False,
        trace: dict[str, Any] | None = None,
    ) -> SwapTransaction:
        from solders.compute_budget import set_compute_unit_price  # type: ignore[import-untyped]
        from solders.message import MessageV0  # type: ignore[import-untyped]
        from solders.null_signer import NullSigner  # type: ignore[import-untyped]
        from solders.pubkey import Pubkey  # type: ignore[import-untyped]
        from solders.system_program import TransferParams, transfer  # type: ignore[import-untyped]
        from solders.transaction import VersionedTransaction  # type: ignore[import-untyped]

        user_public_key = str(self.signer.get_public_key() or "")
        if not user_public_key:
            raise LiveExecutionError("Signer not validated – cannot rebuild Jupiter order tx")

        trace_payload = trace if trace is not None else {}
        tx_decode_started = time.monotonic()
        order_tx = VersionedTransaction.from_bytes(order.raw_transaction)
        message = order_tx.message
        trace_payload["jupiter_order_decode_ms"] = (time.monotonic() - tx_decode_started) * 1000.0

        alt_addresses = [
            str(lookup.account_key)
            for lookup in list(getattr(message, "address_table_lookups", []) or [])
        ]
        alt_started = time.monotonic()
        alt_accounts = await self.broadcaster.get_address_lookup_table_accounts_async(alt_addresses)
        trace_payload["alt_fetch_ms"] = (time.monotonic() - alt_started) * 1000.0
        trace_payload["alt_account_count"] = len(alt_accounts)
        full_keys = self._resolve_loaded_account_keys(message, alt_accounts)
        order_policy = await self._refresh_order_policy_for_account_keys_async(
            order_policy=order_policy,
            account_keys=[str(key) for key in full_keys],
            trace=trace_payload,
        )

        compute_unit_price_micro_lamports = self._priority_fee_to_micro_lamports(
            order_policy.priority_fee_lamports,
            1_400_000,
        )
        tip_account = self._select_tip_account()
        tip_lamports = max(int(order_policy.jito_tip_lamports or 0), 0)

        payer = Pubkey.from_string(user_public_key)
        instructions = [set_compute_unit_price(int(compute_unit_price_micro_lamports))]
        preserved_instruction_count = 0
        for compiled_instruction in list(message.instructions):
            program_index = int(compiled_instruction.program_id_index)
            if program_index < 0 or program_index >= len(full_keys):
                raise LiveExecutionError(
                    f"compiled instruction program index out of range while rebuilding order tx: {program_index}"
                )
            program_id = str(full_keys[program_index])
            data = bytes(compiled_instruction.data)
            account_indexes = list(bytes(compiled_instruction.accounts))
            if self._is_compute_unit_price_instruction(program_id, data):
                continue
            if self._is_existing_tip_transfer(
                program_id=program_id,
                data=data,
                account_indexes=account_indexes,
                full_keys=full_keys,
            ):
                continue
            instructions.append(
                self._compiled_instruction_to_instruction(compiled_instruction, message, full_keys)
            )
            preserved_instruction_count += 1

        close_token_account_enabled = (
            close_input_token_account
            and bool(getattr(self.config, "live_close_token_ata_on_full_exit", True))
            and input_mint != SOL_MINT
            and output_mint == SOL_MINT
        )
        if close_token_account_enabled:
            input_token_program = await self._resolve_mint_token_program_async(input_mint)
            trace_payload["input_token_program"] = input_token_program
            instructions.append(
                self._close_token_account_instruction(
                    user_public_key, input_mint, input_token_program
                )
            )
        instructions.append(
            transfer(
                TransferParams(
                    from_pubkey=payer,
                    to_pubkey=Pubkey.from_string(tip_account),
                    lamports=tip_lamports,
                )
            )
        )

        trace_payload["jupiter_path"] = "order_tx_rebuild"
        trace_payload["manual_tip_account"] = tip_account
        trace_payload["manual_tip_lamports"] = tip_lamports
        trace_payload["manual_close_input_token_account"] = bool(close_token_account_enabled)
        trace_payload["manual_compute_unit_price_micro_lamports"] = int(
            compute_unit_price_micro_lamports
        )
        trace_payload["rebuild_preserved_instruction_count"] = preserved_instruction_count

        compile_started = time.monotonic()
        rebuilt_message = MessageV0.try_compile(
            payer,
            instructions,
            alt_accounts,
            message.recent_blockhash,
        )
        unsigned_tx = VersionedTransaction(rebuilt_message, [NullSigner(payer)])
        trace_payload["tx_compile_ms"] = (time.monotonic() - compile_started) * 1000.0
        trace_payload["custom_tx_instruction_count"] = len(instructions)

        return SwapTransaction(
            raw_transaction=bytes(unsigned_tx),
            last_valid_block_height=int(order.last_valid_block_height),
            in_amount=int(order.in_amount),
            out_amount=int(order.out_amount),
            built_at=self._now_iso(),
        )

    def _get_sender_compliant_order_tx(
        self,
        *,
        input_mint: str,
        output_mint: str,
        amount: int,
        order_policy: LiveOrderPolicy,
        close_input_token_account: bool = False,
        trace: dict[str, Any] | None = None,
    ) -> SwapTransaction:
        order_started = time.monotonic()
        order = self.jupiter.get_order(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=amount,
            slippage_bps=order_policy.slippage_bps,
            user_public_key=self.signer.get_public_key(),
            priority_fee_lamports=order_policy.priority_fee_lamports,
            jito_tip_lamports=order_policy.jito_tip_lamports,
            broadcast_fee_type=order_policy.broadcast_fee_type,
        )
        if trace is not None:
            trace["jupiter_order_ms"] = (time.monotonic() - order_started) * 1000.0
        if input_mint == SOL_MINT:
            self._check_jupiter_buy_price_impact(
                token_mint=output_mint,
                price_impact_pct=order.price_impact_pct,
                source="jupiter_order",
                trace=trace,
                order_policy=order_policy,
            )
        passthrough = self._maybe_use_order_tx_passthrough(
            order=order,
            close_input_token_account=close_input_token_account,
            trace=trace,
        )
        if passthrough is not None:
            return passthrough
        return self._rebuild_order_transaction_for_sender(
            order=order,
            input_mint=input_mint,
            output_mint=output_mint,
            order_policy=order_policy,
            close_input_token_account=close_input_token_account,
            trace=trace,
        )

    async def _get_sender_compliant_order_tx_async(
        self,
        *,
        input_mint: str,
        output_mint: str,
        amount: int,
        order_policy: LiveOrderPolicy,
        close_input_token_account: bool = False,
        trace: dict[str, Any] | None = None,
    ) -> SwapTransaction:
        order_started = time.monotonic()
        order = await self.jupiter.get_order_async(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=amount,
            slippage_bps=order_policy.slippage_bps,
            user_public_key=self.signer.get_public_key(),
            priority_fee_lamports=order_policy.priority_fee_lamports,
            jito_tip_lamports=order_policy.jito_tip_lamports,
            broadcast_fee_type=order_policy.broadcast_fee_type,
        )
        if trace is not None:
            trace["jupiter_order_ms"] = (time.monotonic() - order_started) * 1000.0
        if input_mint == SOL_MINT:
            self._check_jupiter_buy_price_impact(
                token_mint=output_mint,
                price_impact_pct=order.price_impact_pct,
                source="jupiter_order",
                trace=trace,
                order_policy=order_policy,
            )
        passthrough = self._maybe_use_order_tx_passthrough(
            order=order,
            close_input_token_account=close_input_token_account,
            trace=trace,
        )
        if passthrough is not None:
            return passthrough
        return await self._rebuild_order_transaction_for_sender_async(
            order=order,
            input_mint=input_mint,
            output_mint=output_mint,
            order_policy=order_policy,
            close_input_token_account=close_input_token_account,
            trace=trace,
        )

    def _build_custom_swap_tx(
        self,
        *,
        input_mint: str,
        output_mint: str,
        amount: int,
        order_policy: LiveOrderPolicy,
        close_input_token_account: bool = False,
        trace: dict[str, Any] | None = None,
    ) -> SwapTransaction:
        """Build a Sender-compliant unsigned tx from Metis swap instructions."""
        from solders.compute_budget import set_compute_unit_price  # type: ignore[import-untyped]
        from solders.hash import Hash  # type: ignore[import-untyped]
        from solders.message import MessageV0  # type: ignore[import-untyped]
        from solders.null_signer import NullSigner  # type: ignore[import-untyped]
        from solders.pubkey import Pubkey  # type: ignore[import-untyped]
        from solders.system_program import TransferParams, transfer  # type: ignore[import-untyped]
        from solders.transaction import VersionedTransaction  # type: ignore[import-untyped]

        user_public_key = str(self.signer.get_public_key() or "")
        if not user_public_key:
            raise LiveExecutionError("Signer not validated – cannot build custom swap tx")

        trace_payload = trace if trace is not None else {}
        trace_payload["jupiter_path"] = "metis_swap_instructions"
        slippage_bps = self._custom_tx_slippage_bps(order_policy.slippage_bps)
        trace_payload["slippage_bps"] = slippage_bps
        trace_payload["slippage_mode"] = (
            "fixed_custom_tx" if order_policy.slippage_bps is None else "fixed"
        )

        quote_started = time.monotonic()
        quote_response = self.jupiter.get_metis_quote(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=amount,
            slippage_bps=slippage_bps,
            for_jito_bundle=self._is_bundle_mode(),
        )
        trace_payload["jupiter_quote_ms"] = (time.monotonic() - quote_started) * 1000.0
        if input_mint == SOL_MINT:
            self._check_jupiter_buy_price_impact(
                token_mint=output_mint,
                price_impact_pct=quote_response.get("priceImpactPct"),
                source="metis_quote",
                trace=trace_payload,
                order_policy=order_policy,
            )
        else:
            self._check_jupiter_sell_price_impact(
                token_mint=input_mint,
                price_impact_pct=quote_response.get("priceImpactPct"),
                source="metis_quote",
                trace=trace_payload,
                order_policy=order_policy,
            )

        instructions_started = time.monotonic()
        instructions_payload = self._get_swap_instructions_with_fallback(
            quote_response=quote_response,
            user_public_key=user_public_key,
            input_mint=input_mint,
            output_mint=output_mint,
            trace=trace_payload,
        )
        trace_payload["jupiter_swap_instructions_ms"] = (
            time.monotonic() - instructions_started
        ) * 1000.0
        order_policy = self._refresh_order_policy_for_account_keys(
            order_policy=order_policy,
            account_keys=self._instruction_payload_account_keys(instructions_payload),
            trace=trace_payload,
        )

        compute_payloads = list(instructions_payload.get("computeBudgetInstructions") or [])
        compute_limit = self._extract_compute_unit_limit(compute_payloads)
        compute_unit_price_micro_lamports = self._priority_fee_to_micro_lamports(
            order_policy.priority_fee_lamports,
            compute_limit,
        )
        tip_account = self._select_tip_account()
        tip_lamports = max(int(order_policy.jito_tip_lamports or 0), 0)

        payer = Pubkey.from_string(user_public_key)
        instructions = [set_compute_unit_price(int(compute_unit_price_micro_lamports))]
        instructions.extend(
            self._deserialize_instruction_payload(payload)
            for payload in compute_payloads
            if base64.b64decode(str(payload.get("data") or ""))[:1] != b"\x03"
        )
        token_ledger = instructions_payload.get("tokenLedgerInstruction")
        if isinstance(token_ledger, dict) and token_ledger.get("programId"):
            instructions.append(self._deserialize_instruction_payload(token_ledger))
        instructions.extend(
            self._deserialize_instruction_payload(payload)
            for payload in list(instructions_payload.get("setupInstructions") or [])
        )
        instructions.extend(
            self._deserialize_instruction_payload(payload)
            for payload in list(instructions_payload.get("otherInstructions") or [])
        )
        swap_instruction = instructions_payload.get("swapInstruction")
        if not isinstance(swap_instruction, dict) or not swap_instruction.get("programId"):
            raise LiveExecutionError("Jupiter swap-instructions response missing swapInstruction")
        instructions.append(self._deserialize_instruction_payload(swap_instruction))
        cleanup_instruction = instructions_payload.get("cleanupInstruction")
        if isinstance(cleanup_instruction, dict) and cleanup_instruction.get("programId"):
            instructions.append(self._deserialize_instruction_payload(cleanup_instruction))
        if (
            close_input_token_account
            and bool(getattr(self.config, "live_close_token_ata_on_full_exit", True))
            and input_mint != SOL_MINT
            and output_mint == SOL_MINT
        ):
            input_token_program = self._resolve_mint_token_program(input_mint)
            trace_payload["input_token_program"] = input_token_program
            instructions.append(
                self._close_token_account_instruction(
                    user_public_key, input_mint, input_token_program
                )
            )
        instructions.append(
            transfer(
                TransferParams(
                    from_pubkey=payer,
                    to_pubkey=Pubkey.from_string(tip_account),
                    lamports=tip_lamports,
                )
            )
        )

        trace_payload["manual_tip_account"] = tip_account
        trace_payload["manual_tip_lamports"] = tip_lamports
        trace_payload["manual_close_input_token_account"] = bool(
            close_input_token_account
            and bool(getattr(self.config, "live_close_token_ata_on_full_exit", True))
            and input_mint != SOL_MINT
            and output_mint == SOL_MINT
        )
        trace_payload["manual_compute_unit_limit"] = compute_limit
        trace_payload["manual_compute_unit_price_micro_lamports"] = int(
            compute_unit_price_micro_lamports
        )

        alt_started = time.monotonic()
        alt_accounts = self.broadcaster.get_address_lookup_table_accounts(
            list(instructions_payload.get("addressLookupTableAddresses") or [])
        )
        trace_payload["alt_fetch_ms"] = (time.monotonic() - alt_started) * 1000.0
        trace_payload["alt_account_count"] = len(alt_accounts)

        blockhash_started = time.monotonic()
        recent_blockhash, last_valid_block_height = self.broadcaster.get_latest_blockhash()
        trace_payload["blockhash_ms"] = (time.monotonic() - blockhash_started) * 1000.0

        compile_started = time.monotonic()
        message = MessageV0.try_compile(
            payer,
            instructions,
            alt_accounts,
            Hash.from_string(recent_blockhash),
        )
        unsigned_tx = VersionedTransaction(message, [NullSigner(payer)])
        trace_payload["tx_compile_ms"] = (time.monotonic() - compile_started) * 1000.0
        trace_payload["custom_tx_instruction_count"] = len(instructions)

        return SwapTransaction(
            raw_transaction=bytes(unsigned_tx),
            last_valid_block_height=last_valid_block_height,
            in_amount=int(quote_response.get("inAmount") or amount),
            out_amount=int(quote_response.get("outAmount") or 0),
            built_at=self._now_iso(),
        )

    async def _build_custom_swap_tx_async(
        self,
        *,
        input_mint: str,
        output_mint: str,
        amount: int,
        order_policy: LiveOrderPolicy,
        close_input_token_account: bool = False,
        trace: dict[str, Any] | None = None,
    ) -> SwapTransaction:
        """Async variant of :meth:`_build_custom_swap_tx`."""
        from solders.compute_budget import set_compute_unit_price  # type: ignore[import-untyped]
        from solders.hash import Hash  # type: ignore[import-untyped]
        from solders.message import MessageV0  # type: ignore[import-untyped]
        from solders.null_signer import NullSigner  # type: ignore[import-untyped]
        from solders.pubkey import Pubkey  # type: ignore[import-untyped]
        from solders.system_program import TransferParams, transfer  # type: ignore[import-untyped]
        from solders.transaction import VersionedTransaction  # type: ignore[import-untyped]

        user_public_key = str(self.signer.get_public_key() or "")
        if not user_public_key:
            raise LiveExecutionError("Signer not validated – cannot build custom swap tx")

        trace_payload = trace if trace is not None else {}
        trace_payload["jupiter_path"] = "metis_swap_instructions"
        slippage_bps = self._custom_tx_slippage_bps(order_policy.slippage_bps)
        trace_payload["slippage_bps"] = slippage_bps
        trace_payload["slippage_mode"] = (
            "fixed_custom_tx" if order_policy.slippage_bps is None else "fixed"
        )

        quote_started = time.monotonic()
        quote_response = await self.jupiter.get_metis_quote_async(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=amount,
            slippage_bps=slippage_bps,
            for_jito_bundle=self._is_bundle_mode(),
        )
        trace_payload["jupiter_quote_ms"] = (time.monotonic() - quote_started) * 1000.0
        if input_mint == SOL_MINT:
            self._check_jupiter_buy_price_impact(
                token_mint=output_mint,
                price_impact_pct=quote_response.get("priceImpactPct"),
                source="metis_quote",
                trace=trace_payload,
                order_policy=order_policy,
            )
        else:
            self._check_jupiter_sell_price_impact(
                token_mint=input_mint,
                price_impact_pct=quote_response.get("priceImpactPct"),
                source="metis_quote",
                trace=trace_payload,
                order_policy=order_policy,
            )

        instructions_started = time.monotonic()
        instructions_payload = await self._get_swap_instructions_with_fallback_async(
            quote_response=quote_response,
            user_public_key=user_public_key,
            input_mint=input_mint,
            output_mint=output_mint,
            trace=trace_payload,
        )
        trace_payload["jupiter_swap_instructions_ms"] = (
            time.monotonic() - instructions_started
        ) * 1000.0
        order_policy = await self._refresh_order_policy_for_account_keys_async(
            order_policy=order_policy,
            account_keys=self._instruction_payload_account_keys(instructions_payload),
            trace=trace_payload,
        )

        compute_payloads = list(instructions_payload.get("computeBudgetInstructions") or [])
        compute_limit = self._extract_compute_unit_limit(compute_payloads)
        compute_unit_price_micro_lamports = self._priority_fee_to_micro_lamports(
            order_policy.priority_fee_lamports,
            compute_limit,
        )
        tip_account = self._select_tip_account()
        tip_lamports = max(int(order_policy.jito_tip_lamports or 0), 0)

        payer = Pubkey.from_string(user_public_key)
        instructions = [set_compute_unit_price(int(compute_unit_price_micro_lamports))]
        instructions.extend(
            self._deserialize_instruction_payload(payload)
            for payload in compute_payloads
            if base64.b64decode(str(payload.get("data") or ""))[:1] != b"\x03"
        )
        token_ledger = instructions_payload.get("tokenLedgerInstruction")
        if isinstance(token_ledger, dict) and token_ledger.get("programId"):
            instructions.append(self._deserialize_instruction_payload(token_ledger))
        instructions.extend(
            self._deserialize_instruction_payload(payload)
            for payload in list(instructions_payload.get("setupInstructions") or [])
        )
        instructions.extend(
            self._deserialize_instruction_payload(payload)
            for payload in list(instructions_payload.get("otherInstructions") or [])
        )
        swap_instruction = instructions_payload.get("swapInstruction")
        if not isinstance(swap_instruction, dict) or not swap_instruction.get("programId"):
            raise LiveExecutionError("Jupiter swap-instructions response missing swapInstruction")
        instructions.append(self._deserialize_instruction_payload(swap_instruction))
        cleanup_instruction = instructions_payload.get("cleanupInstruction")
        if isinstance(cleanup_instruction, dict) and cleanup_instruction.get("programId"):
            instructions.append(self._deserialize_instruction_payload(cleanup_instruction))
        if (
            close_input_token_account
            and bool(getattr(self.config, "live_close_token_ata_on_full_exit", True))
            and input_mint != SOL_MINT
            and output_mint == SOL_MINT
        ):
            input_token_program = await self._resolve_mint_token_program_async(input_mint)
            trace_payload["input_token_program"] = input_token_program
            instructions.append(
                self._close_token_account_instruction(
                    user_public_key, input_mint, input_token_program
                )
            )
        instructions.append(
            transfer(
                TransferParams(
                    from_pubkey=payer,
                    to_pubkey=Pubkey.from_string(tip_account),
                    lamports=tip_lamports,
                )
            )
        )

        trace_payload["manual_tip_account"] = tip_account
        trace_payload["manual_tip_lamports"] = tip_lamports
        trace_payload["manual_close_input_token_account"] = bool(
            close_input_token_account
            and bool(getattr(self.config, "live_close_token_ata_on_full_exit", True))
            and input_mint != SOL_MINT
            and output_mint == SOL_MINT
        )
        trace_payload["manual_compute_unit_limit"] = compute_limit
        trace_payload["manual_compute_unit_price_micro_lamports"] = int(
            compute_unit_price_micro_lamports
        )

        alt_started = time.monotonic()
        alt_accounts = await self.broadcaster.get_address_lookup_table_accounts_async(
            list(instructions_payload.get("addressLookupTableAddresses") or [])
        )
        trace_payload["alt_fetch_ms"] = (time.monotonic() - alt_started) * 1000.0
        trace_payload["alt_account_count"] = len(alt_accounts)

        blockhash_started = time.monotonic()
        (
            recent_blockhash,
            last_valid_block_height,
        ) = await self.broadcaster.get_latest_blockhash_async()
        trace_payload["blockhash_ms"] = (time.monotonic() - blockhash_started) * 1000.0

        compile_started = time.monotonic()
        message = MessageV0.try_compile(
            payer,
            instructions,
            alt_accounts,
            Hash.from_string(recent_blockhash),
        )
        unsigned_tx = VersionedTransaction(message, [NullSigner(payer)])
        trace_payload["tx_compile_ms"] = (time.monotonic() - compile_started) * 1000.0
        trace_payload["custom_tx_instruction_count"] = len(instructions)

        return SwapTransaction(
            raw_transaction=bytes(unsigned_tx),
            last_valid_block_height=last_valid_block_height,
            in_amount=int(quote_response.get("inAmount") or amount),
            out_amount=int(quote_response.get("outAmount") or 0),
            built_at=self._now_iso(),
        )

    # ------------------------------------------------------------------
    # Pre-fetch swap TX (called during the ranking window, ~2 s before fire)
    # ------------------------------------------------------------------

    def prefetch_swap_tx(
        self,
        token_mint: str,
        size_sol: float,
        *,
        prefer_jupiter: bool = False,
        strategy: str | None = None,
        source_program: str | None = None,
    ) -> "SwapTransaction | None":
        """Quote + build unsigned swap TX in background during the ranking window.

        Called from :meth:`TradeExecutor.submit_swap_tx_future` via ThreadPool.
        By the time the candidate flushes (~2 s later), the TX is already built.
        Fire time then collapses to: sign (~1 ms) + send (~50-150 ms).

        ``prefer_jupiter=True`` skips the native Pump-AMM builder. The main
        (mature-pair) lane uses this because the native builder's Pump-AMM swap
        math overflows on mature pools (preflight Custom 6023).

        Returns None on any failure so callers fall back to ``execute_buy``.
        """
        try:
            amount_lamports = int(size_sol * LAMPORTS_PER_SOL)
            policy = self._resolve_order_policy(
                side="buy", strategy=strategy, source_program=source_program
            )
            trace: dict[str, Any] = {
                "path": "prefetch_buy",
                "strategy": strategy or "default",
            }
            if prefer_jupiter:
                trace["native_pump_skipped"] = "prefer_jupiter"
                native_order = None
            else:
                native_order = self._try_native_pump_amm_buy_tx(
                    token_mint=token_mint,
                    amount_lamports=amount_lamports,
                    order_policy=policy,
                    trace=trace,
                )
            if native_order is not None:
                order = native_order
            elif self._requires_custom_jupiter_tx():
                try:
                    order = self._get_sender_compliant_order_tx(
                        input_mint=SOL_MINT,
                        output_mint=token_mint,
                        amount=amount_lamports,
                        order_policy=policy,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "prefetch_swap_tx sender order rebuild fallback for %s: %s",
                        token_mint[:8],
                        exc,
                    )
                    order = self._build_custom_swap_tx(
                        input_mint=SOL_MINT,
                        output_mint=token_mint,
                        amount=amount_lamports,
                        order_policy=policy,
                    )
            else:
                order = self.jupiter.get_order(
                    input_mint=SOL_MINT,
                    output_mint=token_mint,
                    amount=amount_lamports,
                    slippage_bps=policy.slippage_bps,
                    user_public_key=self.signer.get_public_key(),
                    priority_fee_lamports=policy.priority_fee_lamports,
                    jito_tip_lamports=policy.jito_tip_lamports,
                    broadcast_fee_type=policy.broadcast_fee_type,
                )
                self._check_jupiter_buy_price_impact(
                    token_mint=token_mint,
                    price_impact_pct=order.price_impact_pct,
                    source="jupiter_order",
                    trace=trace,
                    order_policy=policy,
                )
            logger.debug(
                "prefetch_swap_tx: pre-built TX for %s size=%.4f SOL last_valid=%d out=%d tokens",
                token_mint[:8],
                size_sol,
                order.last_valid_block_height,
                order.out_amount,
            )
            return SwapTransaction(
                raw_transaction=order.raw_transaction,
                last_valid_block_height=order.last_valid_block_height,
                in_amount=order.in_amount,
                out_amount=order.out_amount,
                built_at=self._now_iso(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("prefetch_swap_tx failed for %s: %s", token_mint[:8], exc)
            return None

    def prefetch_sell_tx(
        self,
        token_mint: str,
        token_amount: int,
        *,
        prefer_jupiter: bool = False,
        strategy: str | None = None,
        source_program: str | None = None,
    ) -> "SwapTransaction | None":
        """Quote + build unsigned sell TX (token → SOL) in background.

        Called by :class:`LiveSellCache` every 2s per open position.
        At exit time callers only need to sign (~1ms) + broadcast (~50-150ms).

        ``prefer_jupiter=True`` skips the native Pump-AMM sell builder. The main
        (mature-pair) lane uses this because the native sell swap math overflows
        on mature pools (preflight Custom 6023), same failure mode as buy.

        Returns None on any failure so callers fall back to ``execute_sell``.
        """
        try:
            policy = self._resolve_order_policy(
                side="sell", strategy=strategy, source_program=source_program
            )
            trace: dict[str, Any] = {
                "path": "prefetch_sell",
                "strategy": strategy or "default",
            }
            if prefer_jupiter:
                trace["native_pump_skipped"] = "prefer_jupiter"
                native_order = None
            else:
                native_order = self._try_native_pump_amm_sell_tx(
                    token_mint=token_mint,
                    token_amount=token_amount,
                    order_policy=policy,
                    close_input_token_account=bool(
                        getattr(self.config, "live_close_token_ata_on_full_exit", True)
                    ),
                    trace=trace,
                )
            if native_order is not None:
                order = native_order
            elif self._requires_custom_jupiter_tx():
                order = self._build_custom_swap_tx(
                    input_mint=token_mint,
                    output_mint=SOL_MINT,
                    amount=token_amount,
                    order_policy=policy,
                    close_input_token_account=bool(
                        getattr(self.config, "live_close_token_ata_on_full_exit", True)
                    ),
                )
            else:
                order = self.jupiter.get_order(
                    input_mint=token_mint,
                    output_mint=SOL_MINT,
                    amount=token_amount,
                    slippage_bps=policy.slippage_bps,
                    user_public_key=self.signer.get_public_key(),
                    priority_fee_lamports=policy.priority_fee_lamports,
                    jito_tip_lamports=policy.jito_tip_lamports,
                    broadcast_fee_type=policy.broadcast_fee_type,
                )
                self._check_jupiter_sell_price_impact(
                    token_mint=token_mint,
                    price_impact_pct=order.price_impact_pct,
                    source="jupiter_order",
                    trace=trace,
                    order_policy=policy,
                )
            logger.debug(
                "prefetch_sell_tx: pre-built TX for %s amount=%d last_valid=%d out=%d lamps",
                token_mint[:8],
                token_amount,
                order.last_valid_block_height,
                order.out_amount,
            )
            return SwapTransaction(
                raw_transaction=order.raw_transaction,
                last_valid_block_height=order.last_valid_block_height,
                in_amount=order.in_amount,
                out_amount=order.out_amount,
                built_at=self._now_iso(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("prefetch_sell_tx failed for %s: %s", token_mint[:8], exc)
            return None

    async def execute_buy_prebuilt_async(
        self,
        token_mint: str,
        size_sol: float,
        prebuilt_tx: "SwapTransaction",
        current_exposure_sol: float = 0.0,
        open_position_count: int = 0,
    ) -> LiveTradeResult:
        """Async version of execute_buy_prebuilt: sign + broadcast without blocking the event loop."""
        logger.info(
            "🔴 LIVE BUY (prebuilt async): %s | size=%.4f SOL | exposure=%.4f SOL | open=%d",
            token_mint[:12],
            size_sol,
            current_exposure_sol,
            open_position_count,
        )
        trace: dict[str, Any] = {
            "path": "live_buy_prebuilt_async",
            "started_at": self._now_iso(),
            "__total_started_monotonic": time.monotonic(),
        }
        prefetch_age_ms = self._prefetch_age_ms(prebuilt_tx)
        if prefetch_age_ms is not None:
            trace["prefetch_age_ms"] = prefetch_age_ms

        preflight_started = time.monotonic()
        try:
            self._preflight_checks(size_sol, current_exposure_sol, open_position_count)
        except LiveExecutionError as exc:
            logger.error("LIVE BUY prebuilt async preflight failed: %s", exc)
            trace["preflight_ms"] = (time.monotonic() - preflight_started) * 1000.0
            return self._result_with_error(error=str(exc), latency_trace=trace)
        trace["preflight_ms"] = (time.monotonic() - preflight_started) * 1000.0
        order_policy = self._resolve_order_policy(side="buy")
        trace["priority_fee_lamports"] = int(order_policy.priority_fee_lamports)
        trace["jito_tip_lamports"] = int(order_policy.jito_tip_lamports)
        try:
            await self._enforce_buy_wallet_balance_async(size_sol, order_policy, trace)
        except LiveExecutionError as exc:
            logger.error("LIVE BUY prebuilt async wallet check failed: %s", exc)
            return self._result_with_error(error=str(exc), latency_trace=trace)

        sign_started = time.monotonic()
        try:
            signed_tx = self.signer.sign_transaction(prebuilt_tx.raw_transaction)
        except SignerError as exc:
            logger.error("LIVE BUY prebuilt async signing failed: %s", exc)
            trace["sign_ms"] = (time.monotonic() - sign_started) * 1000.0
            return self._result_with_error(error=f"signing_failed: {exc}", latency_trace=trace)
        trace["sign_ms"] = (time.monotonic() - sign_started) * 1000.0

        preflight_sim_started = time.monotonic()
        preflight_err = await self._preflight_simulate_buy_async(
            signed_tx, token_mint=token_mint, trace=trace
        )
        trace["preflight_sim_ms"] = (time.monotonic() - preflight_sim_started) * 1000.0
        if preflight_err is not None:
            if self._is_retriable_tx_build_error(preflight_err):
                retry = await self._rebuild_and_retry_buy_async(
                    token_mint=token_mint,
                    size_sol=size_sol,
                    original_error=preflight_err,
                    parent_trace=trace,
                )
                if retry is not None:
                    return retry
            return self._result_with_error(error=preflight_err, latency_trace=trace)

        try:
            result: BroadcastResult = await self.broadcaster.broadcast_async(
                signed_tx,
                last_valid_block_height=prebuilt_tx.last_valid_block_height,
            )
        except BroadcastError as exc:
            logger.error("LIVE BUY prebuilt async broadcast failed: %s", exc)
            return self._result_with_error(error=f"broadcast_failed: {exc}", latency_trace=trace)
        self._apply_broadcast_trace(trace, result)

        if not result.confirmed:
            logger.warning(
                "LIVE BUY prebuilt async tx sent but NOT confirmed: sig=%s error=%s",
                result.signature,
                result.error,
            )
            (
                reconciliation,
                reconciliation_error,
            ) = await self._reconcile_failed_transaction_async(
                signature=result.signature,
                token_mint=token_mint,
                expected_in_amount=prebuilt_tx.in_amount,
                expected_out_amount=prebuilt_tx.out_amount,
                slot=result.slot,
                trace=trace,
            )
            return self._result_with_error(
                error=f"not_confirmed: {result.error}",
                signature=result.signature,
                in_amount=prebuilt_tx.in_amount,
                out_amount=prebuilt_tx.out_amount,
                slot=result.slot,
                latency_trace=trace,
                reconciliation=reconciliation,
                reconciliation_error=reconciliation_error,
            )

        reconciliation: LiveFillReconciliation | None = None
        reconciliation_error: str | None = None
        reconcile_started = time.monotonic()
        try:
            reconciliation = await self._reconcile_confirmed_fill_async(
                signature=result.signature,
                token_mint=token_mint,
                expected_in_amount=prebuilt_tx.in_amount,
                expected_out_amount=prebuilt_tx.out_amount,
                slot=result.slot,
            )
        except Exception as exc:  # noqa: BLE001
            reconciliation_error = str(exc)
        trace["reconcile_ms"] = (time.monotonic() - reconcile_started) * 1000.0
        trace["total_execution_ms"] = (
            time.monotonic() - float(trace["__total_started_monotonic"])
        ) * 1000.0

        logger.info(
            "✅ LIVE BUY (prebuilt async) confirmed: sig=%s | in=%d lamports | out=%d tokens | slot=%s",
            result.signature,
            prebuilt_tx.in_amount,
            prebuilt_tx.out_amount,
            result.slot,
        )
        return self._result_success(
            signature=result.signature,
            in_amount=prebuilt_tx.in_amount,
            out_amount=prebuilt_tx.out_amount,
            slot=result.slot,
            latency_trace=trace,
            reconciliation=reconciliation,
            reconciliation_error=reconciliation_error,
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
        """Async version of execute_sell_prebuilt: sign + broadcast without blocking the event loop."""
        logger.info(
            "🔴 LIVE SELL (prebuilt async): %s | amount=%d | exposure=%.4f SOL",
            token_mint[:12],
            token_amount,
            current_exposure_sol,
        )
        trace: dict[str, Any] = {
            "path": "live_sell_prebuilt_async",
            "started_at": self._now_iso(),
            "__total_started_monotonic": time.monotonic(),
            "strategy": strategy or "default",
        }
        prefetch_age_ms = self._prefetch_age_ms(prebuilt_tx)
        if prefetch_age_ms is not None:
            trace["prefetch_age_ms"] = prefetch_age_ms

        if self.signer.get_public_key() is None:
            msg = "Signer not validated – refusing live execution"
            logger.error("LIVE SELL prebuilt async preflight failed: %s", msg)
            return self._result_with_error(error=msg, latency_trace=trace)
        try:
            (
                token_amount,
                close_token_account,
            ) = await self._resolve_wallet_token_balance_async(
                token_mint=token_mint,
                requested_token_amount=token_amount,
                close_token_account=close_token_account,
                trace=trace,
            )
        except LiveExecutionError as exc:
            logger.error("LIVE SELL prebuilt async wallet balance check failed: %s", exc)
            return self._result_with_error(error=str(exc), latency_trace=trace)
        order_policy = self._resolve_order_policy(
            side="sell", strategy=strategy, source_program=source_program
        )
        trace["priority_fee_lamports"] = int(order_policy.priority_fee_lamports)
        trace["jito_tip_lamports"] = int(order_policy.jito_tip_lamports)
        trace["slippage_bps"] = order_policy.slippage_bps
        trace["slippage_mode"] = "jupiter_auto" if order_policy.slippage_bps is None else "fixed"
        if int(prebuilt_tx.in_amount) != int(token_amount):
            trace["prefetched_sell_amount_stale"] = True
            trace["prefetched_sell_in_amount_raw"] = int(prebuilt_tx.in_amount)
            order_started = time.monotonic()
            try:
                if prefer_jupiter:
                    trace["native_pump_skipped"] = "prefer_jupiter"
                    native_order = None
                else:
                    native_order = await self._try_native_pump_amm_sell_tx_async(
                        token_mint=token_mint,
                        token_amount=token_amount,
                        order_policy=order_policy,
                        close_input_token_account=close_token_account,
                        trace=trace,
                    )
                if native_order is not None:
                    prebuilt_tx = native_order
                elif self._requires_custom_jupiter_tx():
                    prebuilt_tx = await self._build_custom_swap_tx_async(
                        input_mint=token_mint,
                        output_mint=SOL_MINT,
                        amount=token_amount,
                        order_policy=order_policy,
                        close_input_token_account=close_token_account,
                        trace=trace,
                    )
                else:
                    prebuilt_tx = await self.jupiter.get_order_async(
                        input_mint=token_mint,
                        output_mint=SOL_MINT,
                        amount=token_amount,
                        slippage_bps=order_policy.slippage_bps,
                        user_public_key=self.signer.get_public_key(),
                        priority_fee_lamports=order_policy.priority_fee_lamports,
                        jito_tip_lamports=order_policy.jito_tip_lamports,
                        broadcast_fee_type=order_policy.broadcast_fee_type,
                    )
                    trace["jupiter_order_ms"] = (time.monotonic() - order_started) * 1000.0
            except (JupiterError, LiveExecutionError, BroadcastError) as exc:
                logger.error("LIVE SELL prebuilt async stale-refresh failed: %s", exc)
                if "jupiter_order_ms" not in trace:
                    trace["jupiter_order_ms"] = (time.monotonic() - order_started) * 1000.0
                return self._result_with_error(
                    error=f"jupiter_order_failed: {exc}", latency_trace=trace
                )
        try:
            self._validate_live_sell_viability(
                expected_out_lamports=self._conservative_sell_out_lamports(
                    token_mint=token_mint,
                    token_amount=token_amount,
                    routed_out_lamports=prebuilt_tx.out_amount,
                    trace=trace,
                ),
                order_policy=order_policy,
                close_token_account=close_token_account,
            )
        except LiveExecutionError as exc:
            error = str(exc)
            if error.startswith("dust_exit_blocked") and bool(
                trace.get("wallet_token_close_account_effective")
            ):
                error = f"wallet_token_balance_dust: {error}"
            logger.error("LIVE SELL prebuilt async viability check failed: %s", error)
            return self._result_with_error(error=error, latency_trace=trace)

        sign_started = time.monotonic()
        try:
            signed_tx = self.signer.sign_transaction(prebuilt_tx.raw_transaction)
        except SignerError as exc:
            logger.error("LIVE SELL prebuilt async signing failed: %s", exc)
            trace["sign_ms"] = (time.monotonic() - sign_started) * 1000.0
            return self._result_with_error(error=f"signing_failed: {exc}", latency_trace=trace)
        trace["sign_ms"] = (time.monotonic() - sign_started) * 1000.0

        try:
            result: BroadcastResult = await self.broadcaster.broadcast_async(
                signed_tx,
                last_valid_block_height=prebuilt_tx.last_valid_block_height,
            )
        except BroadcastError as exc:
            logger.error("LIVE SELL prebuilt async broadcast failed: %s", exc)
            return self._result_with_error(error=f"broadcast_failed: {exc}", latency_trace=trace)
        self._apply_broadcast_trace(trace, result)

        if not result.confirmed:
            logger.warning(
                "LIVE SELL prebuilt async tx sent but NOT confirmed: sig=%s error=%s",
                result.signature,
                result.error,
            )
            self._maybe_mark_pump_dead(trace=trace, error=result.error, token_mint=token_mint)
            (
                reconciliation,
                reconciliation_error,
            ) = await self._reconcile_failed_transaction_async(
                signature=result.signature,
                token_mint=token_mint,
                expected_in_amount=prebuilt_tx.in_amount,
                expected_out_amount=prebuilt_tx.out_amount,
                slot=result.slot,
                trace=trace,
            )
            return self._result_with_error(
                error=f"not_confirmed: {result.error}",
                signature=result.signature,
                in_amount=prebuilt_tx.in_amount,
                out_amount=prebuilt_tx.out_amount,
                slot=result.slot,
                latency_trace=trace,
                reconciliation=reconciliation,
                reconciliation_error=reconciliation_error,
            )

        reconciliation: LiveFillReconciliation | None = None
        reconciliation_error: str | None = None
        reconcile_started = time.monotonic()
        try:
            reconciliation = await self._reconcile_confirmed_fill_async(
                signature=result.signature,
                token_mint=token_mint,
                expected_in_amount=prebuilt_tx.in_amount,
                expected_out_amount=prebuilt_tx.out_amount,
                slot=result.slot,
            )
        except Exception as exc:  # noqa: BLE001
            reconciliation_error = str(exc)
        trace["reconcile_ms"] = (time.monotonic() - reconcile_started) * 1000.0
        trace["total_execution_ms"] = (
            time.monotonic() - float(trace["__total_started_monotonic"])
        ) * 1000.0

        logger.info(
            "✅ LIVE SELL (prebuilt async) confirmed: sig=%s | in=%d tokens | out=%d lamports | slot=%s",
            result.signature,
            prebuilt_tx.in_amount,
            prebuilt_tx.out_amount,
            result.slot,
        )
        return self._result_success(
            signature=result.signature,
            in_amount=prebuilt_tx.in_amount,
            out_amount=prebuilt_tx.out_amount,
            slot=result.slot,
            latency_trace=trace,
            reconciliation=reconciliation,
            reconciliation_error=reconciliation_error,
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
        """Fire a pre-built sell TX: sign + broadcast only (~50-150ms total).

        Falls back transparently to ``execute_sell`` on signing or broadcast failure.
        """
        logger.info(
            "🔴 LIVE SELL (prebuilt): %s | amount=%d | exposure=%.4f SOL",
            token_mint[:12],
            token_amount,
            current_exposure_sol,
        )
        trace: dict[str, Any] = {
            "path": "live_sell_prebuilt",
            "started_at": self._now_iso(),
            "__total_started_monotonic": time.monotonic(),
            "strategy": strategy or "default",
        }
        prefetch_age_ms = self._prefetch_age_ms(prebuilt_tx)
        if prefetch_age_ms is not None:
            trace["prefetch_age_ms"] = prefetch_age_ms

        if self.signer.get_public_key() is None:
            msg = "Signer not validated – refusing live execution"
            logger.error("LIVE SELL prebuilt preflight failed: %s", msg)
            return self._result_with_error(error=msg, latency_trace=trace)
        try:
            token_amount, close_token_account = self._resolve_wallet_token_balance(
                token_mint=token_mint,
                requested_token_amount=token_amount,
                close_token_account=close_token_account,
                trace=trace,
            )
        except LiveExecutionError as exc:
            logger.error("LIVE SELL prebuilt wallet balance check failed: %s", exc)
            return self._result_with_error(error=str(exc), latency_trace=trace)
        order_policy = self._resolve_order_policy(
            side="sell", strategy=strategy, source_program=source_program
        )
        trace["priority_fee_lamports"] = int(order_policy.priority_fee_lamports)
        trace["jito_tip_lamports"] = int(order_policy.jito_tip_lamports)
        trace["slippage_bps"] = order_policy.slippage_bps
        trace["slippage_mode"] = "jupiter_auto" if order_policy.slippage_bps is None else "fixed"
        if int(prebuilt_tx.in_amount) != int(token_amount):
            trace["prefetched_sell_amount_stale"] = True
            trace["prefetched_sell_in_amount_raw"] = int(prebuilt_tx.in_amount)
            order_started = time.monotonic()
            try:
                if prefer_jupiter:
                    trace["native_pump_skipped"] = "prefer_jupiter"
                    native_order = None
                else:
                    native_order = self._try_native_pump_amm_sell_tx(
                        token_mint=token_mint,
                        token_amount=token_amount,
                        order_policy=order_policy,
                        close_input_token_account=close_token_account,
                        trace=trace,
                    )
                if native_order is not None:
                    prebuilt_tx = native_order
                elif self._requires_custom_jupiter_tx():
                    prebuilt_tx = self._build_custom_swap_tx(
                        input_mint=token_mint,
                        output_mint=SOL_MINT,
                        amount=token_amount,
                        order_policy=order_policy,
                        close_input_token_account=close_token_account,
                        trace=trace,
                    )
                else:
                    prebuilt_tx = self.jupiter.get_order(
                        input_mint=token_mint,
                        output_mint=SOL_MINT,
                        amount=token_amount,
                        slippage_bps=order_policy.slippage_bps,
                        user_public_key=self.signer.get_public_key(),
                        priority_fee_lamports=order_policy.priority_fee_lamports,
                        jito_tip_lamports=order_policy.jito_tip_lamports,
                        broadcast_fee_type=order_policy.broadcast_fee_type,
                    )
                    trace["jupiter_order_ms"] = (time.monotonic() - order_started) * 1000.0
            except (JupiterError, LiveExecutionError, BroadcastError) as exc:
                logger.error("LIVE SELL prebuilt stale-refresh failed: %s", exc)
                if "jupiter_order_ms" not in trace:
                    trace["jupiter_order_ms"] = (time.monotonic() - order_started) * 1000.0
                return self._result_with_error(
                    error=f"jupiter_order_failed: {exc}", latency_trace=trace
                )
        try:
            self._validate_live_sell_viability(
                expected_out_lamports=self._conservative_sell_out_lamports(
                    token_mint=token_mint,
                    token_amount=token_amount,
                    routed_out_lamports=prebuilt_tx.out_amount,
                    trace=trace,
                ),
                order_policy=order_policy,
                close_token_account=close_token_account,
            )
        except LiveExecutionError as exc:
            error = str(exc)
            if error.startswith("dust_exit_blocked") and bool(
                trace.get("wallet_token_close_account_effective")
            ):
                error = f"wallet_token_balance_dust: {error}"
            logger.error("LIVE SELL prebuilt viability check failed: %s", error)
            return self._result_with_error(error=error, latency_trace=trace)

        sign_started = time.monotonic()
        try:
            signed_tx = self.signer.sign_transaction(prebuilt_tx.raw_transaction)
        except SignerError as exc:
            logger.error("LIVE SELL prebuilt signing failed: %s", exc)
            trace["sign_ms"] = (time.monotonic() - sign_started) * 1000.0
            return self._result_with_error(error=f"signing_failed: {exc}", latency_trace=trace)
        trace["sign_ms"] = (time.monotonic() - sign_started) * 1000.0

        try:
            result: BroadcastResult = self.broadcaster.broadcast(
                signed_tx,
                last_valid_block_height=prebuilt_tx.last_valid_block_height,
            )
        except BroadcastError as exc:
            logger.error("LIVE SELL prebuilt broadcast failed: %s", exc)
            return self._result_with_error(error=f"broadcast_failed: {exc}", latency_trace=trace)
        self._apply_broadcast_trace(trace, result)

        if not result.confirmed:
            logger.warning(
                "LIVE SELL prebuilt tx sent but NOT confirmed: sig=%s error=%s",
                result.signature,
                result.error,
            )
            self._maybe_mark_pump_dead(trace=trace, error=result.error, token_mint=token_mint)
            reconciliation, reconciliation_error = self._reconcile_failed_transaction(
                signature=result.signature,
                token_mint=token_mint,
                expected_in_amount=prebuilt_tx.in_amount,
                expected_out_amount=prebuilt_tx.out_amount,
                slot=result.slot,
                trace=trace,
            )
            return self._result_with_error(
                error=f"not_confirmed: {result.error}",
                signature=result.signature,
                in_amount=prebuilt_tx.in_amount,
                out_amount=prebuilt_tx.out_amount,
                slot=result.slot,
                latency_trace=trace,
                reconciliation=reconciliation,
                reconciliation_error=reconciliation_error,
            )

        reconciliation: LiveFillReconciliation | None = None
        reconciliation_error: str | None = None
        reconcile_started = time.monotonic()
        try:
            reconciliation = self._reconcile_confirmed_fill(
                signature=result.signature,
                token_mint=token_mint,
                expected_in_amount=prebuilt_tx.in_amount,
                expected_out_amount=prebuilt_tx.out_amount,
                slot=result.slot,
            )
        except Exception as exc:  # noqa: BLE001
            reconciliation_error = str(exc)
        trace["reconcile_ms"] = (time.monotonic() - reconcile_started) * 1000.0
        trace["total_execution_ms"] = (
            time.monotonic() - float(trace["__total_started_monotonic"])
        ) * 1000.0

        logger.info(
            "✅ LIVE SELL (prebuilt) confirmed: sig=%s | in=%d tokens | out=%d lamports | slot=%s",
            result.signature,
            prebuilt_tx.in_amount,
            prebuilt_tx.out_amount,
            result.slot,
        )
        return self._result_success(
            signature=result.signature,
            in_amount=prebuilt_tx.in_amount,
            out_amount=prebuilt_tx.out_amount,
            slot=result.slot,
            latency_trace=trace,
            reconciliation=reconciliation,
            reconciliation_error=reconciliation_error,
        )

    def execute_buy_prebuilt(
        self,
        token_mint: str,
        size_sol: float,
        prebuilt_tx: "SwapTransaction",
        current_exposure_sol: float = 0.0,
        open_position_count: int = 0,
    ) -> LiveTradeResult:
        """Fire a pre-built swap TX: sign + broadcast only (~50-150 ms total).

        Skips Jupiter quote and TX build steps entirely.  Safeguard checks still
        run.  Falls back to :meth:`execute_buy` if signing or broadcast fails.

        Parameters
        ----------
        prebuilt_tx:
            TX returned by :meth:`prefetch_swap_tx`.  Must not be expired
            (``last_valid_block_height`` not exceeded).
        """
        logger.info(
            "🔴 LIVE BUY (prebuilt): %s | size=%.4f SOL | exposure=%.4f SOL | open=%d",
            token_mint[:12],
            size_sol,
            current_exposure_sol,
            open_position_count,
        )
        trace: dict[str, Any] = {
            "path": "live_buy_prebuilt",
            "started_at": self._now_iso(),
            "__total_started_monotonic": time.monotonic(),
        }
        prefetch_age_ms = self._prefetch_age_ms(prebuilt_tx)
        if prefetch_age_ms is not None:
            trace["prefetch_age_ms"] = prefetch_age_ms

        preflight_started = time.monotonic()
        try:
            self._preflight_checks(size_sol, current_exposure_sol, open_position_count)
        except LiveExecutionError as exc:
            logger.error("LIVE BUY prebuilt preflight failed: %s", exc)
            trace["preflight_ms"] = (time.monotonic() - preflight_started) * 1000.0
            return self._result_with_error(error=str(exc), latency_trace=trace)
        trace["preflight_ms"] = (time.monotonic() - preflight_started) * 1000.0
        order_policy = self._resolve_order_policy(side="buy")
        trace["priority_fee_lamports"] = int(order_policy.priority_fee_lamports)
        trace["jito_tip_lamports"] = int(order_policy.jito_tip_lamports)
        try:
            self._enforce_buy_wallet_balance(size_sol, order_policy, trace)
        except LiveExecutionError as exc:
            logger.error("LIVE BUY prebuilt wallet check failed: %s", exc)
            return self._result_with_error(error=str(exc), latency_trace=trace)

        # Sign locally — ~1 ms
        sign_started = time.monotonic()
        try:
            signed_tx = self.signer.sign_transaction(prebuilt_tx.raw_transaction)
        except SignerError as exc:
            logger.error("LIVE BUY prebuilt signing failed: %s", exc)
            trace["sign_ms"] = (time.monotonic() - sign_started) * 1000.0
            return self._result_with_error(error=f"signing_failed: {exc}", latency_trace=trace)
        trace["sign_ms"] = (time.monotonic() - sign_started) * 1000.0

        preflight_sim_started = time.monotonic()
        preflight_err = self._preflight_simulate_buy(signed_tx, token_mint=token_mint, trace=trace)
        trace["preflight_sim_ms"] = (time.monotonic() - preflight_sim_started) * 1000.0
        if preflight_err is not None:
            return self._result_with_error(error=preflight_err, latency_trace=trace)

        # Broadcast via Helius — ~50-150 ms
        try:
            result: BroadcastResult = self.broadcaster.broadcast(
                signed_tx,
                last_valid_block_height=prebuilt_tx.last_valid_block_height,
            )
        except BroadcastError as exc:
            logger.error("LIVE BUY prebuilt broadcast failed: %s", exc)
            return self._result_with_error(error=f"broadcast_failed: {exc}", latency_trace=trace)
        self._apply_broadcast_trace(trace, result)

        if not result.confirmed:
            logger.warning(
                "LIVE BUY prebuilt tx sent but NOT confirmed: sig=%s error=%s",
                result.signature,
                result.error,
            )
            reconciliation, reconciliation_error = self._reconcile_failed_transaction(
                signature=result.signature,
                token_mint=token_mint,
                expected_in_amount=prebuilt_tx.in_amount,
                expected_out_amount=prebuilt_tx.out_amount,
                slot=result.slot,
                trace=trace,
            )
            return self._result_with_error(
                error=f"not_confirmed: {result.error}",
                signature=result.signature,
                in_amount=prebuilt_tx.in_amount,
                out_amount=prebuilt_tx.out_amount,
                slot=result.slot,
                latency_trace=trace,
                reconciliation=reconciliation,
                reconciliation_error=reconciliation_error,
            )

        reconciliation: LiveFillReconciliation | None = None
        reconciliation_error: str | None = None
        reconcile_started = time.monotonic()
        try:
            reconciliation = self._reconcile_confirmed_fill(
                signature=result.signature,
                token_mint=token_mint,
                expected_in_amount=prebuilt_tx.in_amount,
                expected_out_amount=prebuilt_tx.out_amount,
                slot=result.slot,
            )
        except Exception as exc:  # noqa: BLE001
            reconciliation_error = str(exc)
        trace["reconcile_ms"] = (time.monotonic() - reconcile_started) * 1000.0
        trace["total_execution_ms"] = (
            time.monotonic() - float(trace["__total_started_monotonic"])
        ) * 1000.0

        logger.info(
            "✅ LIVE BUY (prebuilt) confirmed: sig=%s | in=%d lamports | out=%d tokens | slot=%s",
            result.signature,
            prebuilt_tx.in_amount,
            prebuilt_tx.out_amount,
            result.slot,
        )
        return self._result_success(
            signature=result.signature,
            in_amount=prebuilt_tx.in_amount,
            out_amount=prebuilt_tx.out_amount,
            slot=result.slot,
            latency_trace=trace,
            reconciliation=reconciliation,
            reconciliation_error=reconciliation_error,
        )

    # ------------------------------------------------------------------
    # Async Buy fallback (no pre-built TX) — keeps event loop unblocked
    # ------------------------------------------------------------------

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
        """Async live BUY: Jupiter order + sign + broadcast_async — zero thread pool usage.

        Used when no pre-built TX is available at fire time.  Unlike the sync
        ``execute_buy()`` (which runs in a thread-pool executor), this method
        awaits each async step directly on the event loop:

          get_order_async (~100-200ms, non-blocking) → sign (~1ms) → broadcast_async (~50-150ms)

        This keeps the thread pool free for exit processing and other blocking work.

        ``prefer_jupiter=True`` skips the native Pump-AMM builder — main and
        wallet lanes set this because the native builder's math overflows with
        preflight Custom 6023 on mature/graduated pools.
        """
        logger.info(
            "🔴 LIVE BUY async (no prebuilt): %s | size=%.4f SOL | exposure=%.4f SOL | open=%d",
            token_mint[:12],
            size_sol,
            current_exposure_sol,
            open_position_count,
        )
        trace: dict[str, Any] = {
            "path": "live_buy_async",
            "started_at": self._now_iso(),
            "__total_started_monotonic": time.monotonic(),
        }

        preflight_started = time.monotonic()
        try:
            self._preflight_checks(size_sol, current_exposure_sol, open_position_count)
        except LiveExecutionError as exc:
            logger.error("LIVE BUY async preflight failed: %s", exc)
            trace["preflight_ms"] = (time.monotonic() - preflight_started) * 1000.0
            return self._result_with_error(error=str(exc), latency_trace=trace)
        trace["preflight_ms"] = (time.monotonic() - preflight_started) * 1000.0

        amount_lamports = int(size_sol * LAMPORTS_PER_SOL)
        local_buy_out = self._local_quote_out_amount(
            input_mint=SOL_MINT,
            output_mint=token_mint,
            amount=amount_lamports,
        )
        if local_buy_out is not None:
            trace["local_quote_buy_out_amount"] = int(local_buy_out)
        priority_started = time.monotonic()
        order_policy = self._resolve_order_policy(
            side="buy", strategy=strategy, source_program=source_program
        )
        trace["priority_fee_resolve_ms"] = (time.monotonic() - priority_started) * 1000.0
        trace["priority_fee_lamports"] = int(order_policy.priority_fee_lamports)
        trace["jito_tip_lamports"] = int(order_policy.jito_tip_lamports)
        trace["slippage_bps"] = order_policy.slippage_bps
        trace["slippage_mode"] = "jupiter_auto" if order_policy.slippage_bps is None else "fixed"
        trace["broadcast_fee_type"] = order_policy.broadcast_fee_type
        try:
            await self._enforce_buy_wallet_balance_async(size_sol, order_policy, trace)
        except LiveExecutionError as exc:
            logger.error("LIVE BUY async wallet check failed: %s", exc)
            return self._result_with_error(error=str(exc), latency_trace=trace)
        order_started = time.monotonic()
        try:
            if prefer_jupiter:
                trace["native_pump_skipped"] = "prefer_jupiter"
                native_order = None
            else:
                native_order = await self._try_native_pump_amm_buy_tx_async(
                    token_mint=token_mint,
                    amount_lamports=amount_lamports,
                    order_policy=order_policy,
                    trace=trace,
                )
            if native_order is not None:
                order = native_order
            elif self._requires_custom_jupiter_tx():
                try:
                    order = await self._get_sender_compliant_order_tx_async(
                        input_mint=SOL_MINT,
                        output_mint=token_mint,
                        amount=amount_lamports,
                        order_policy=order_policy,
                        trace=trace,
                    )
                except LiveExecutionError:
                    # Pool-health signals (price impact, preflight) must abort
                    # the buy — do not bypass them via a Metis-quote fallback.
                    raise
                except Exception as exc:  # noqa: BLE001
                    trace["sender_order_fallback_reason"] = str(exc)
                    logger.warning(
                        "LIVE BUY async sender order rebuild fallback for %s: %s",
                        token_mint[:12],
                        exc,
                    )
                    order = await self._build_custom_swap_tx_async(
                        input_mint=SOL_MINT,
                        output_mint=token_mint,
                        amount=amount_lamports,
                        order_policy=order_policy,
                        trace=trace,
                    )
            else:
                order = await self.jupiter.get_order_async(
                    input_mint=SOL_MINT,
                    output_mint=token_mint,
                    amount=amount_lamports,
                    slippage_bps=order_policy.slippage_bps,
                    user_public_key=self.signer.get_public_key(),
                    priority_fee_lamports=order_policy.priority_fee_lamports,
                    jito_tip_lamports=order_policy.jito_tip_lamports,
                    broadcast_fee_type=order_policy.broadcast_fee_type,
                )
                trace["jupiter_order_ms"] = (time.monotonic() - order_started) * 1000.0
                self._check_jupiter_buy_price_impact(
                    token_mint=token_mint,
                    price_impact_pct=order.price_impact_pct,
                    source="jupiter_order",
                    trace=trace,
                    order_policy=order_policy,
                )
        except (JupiterError, LiveExecutionError, BroadcastError) as exc:
            logger.error("LIVE BUY async order failed: %s", exc)
            if "jupiter_order_ms" not in trace:
                trace["jupiter_order_ms"] = (time.monotonic() - order_started) * 1000.0
            return self._result_with_error(
                error=f"jupiter_order_failed: {exc}", latency_trace=trace
            )

        sign_started = time.monotonic()
        try:
            signed_tx = self.signer.sign_transaction(order.raw_transaction)
        except SignerError as exc:
            logger.error("LIVE BUY async signing failed: %s", exc)
            trace["sign_ms"] = (time.monotonic() - sign_started) * 1000.0
            return self._result_with_error(error=f"signing_failed: {exc}", latency_trace=trace)
        trace["sign_ms"] = (time.monotonic() - sign_started) * 1000.0

        preflight_sim_started = time.monotonic()
        preflight_err = await self._preflight_simulate_buy_async(
            signed_tx, token_mint=token_mint, trace=trace
        )
        trace["preflight_sim_ms"] = (time.monotonic() - preflight_sim_started) * 1000.0
        if preflight_err is not None:
            return self._result_with_error(error=preflight_err, latency_trace=trace)

        try:
            result: BroadcastResult = await self.broadcaster.broadcast_async(
                signed_tx,
                last_valid_block_height=order.last_valid_block_height,
            )
        except BroadcastError as exc:
            logger.error("LIVE BUY async broadcast failed: %s", exc)
            return self._result_with_error(error=f"broadcast_failed: {exc}", latency_trace=trace)
        self._apply_broadcast_trace(trace, result)

        if not result.confirmed:
            logger.warning(
                "LIVE BUY async tx sent but NOT confirmed: sig=%s error=%s",
                result.signature,
                result.error,
            )
            (
                reconciliation,
                reconciliation_error,
            ) = await self._reconcile_failed_transaction_async(
                signature=result.signature,
                token_mint=token_mint,
                expected_in_amount=order.in_amount,
                expected_out_amount=order.out_amount,
                slot=result.slot,
                trace=trace,
            )
            return self._result_with_error(
                error=f"not_confirmed: {result.error}",
                signature=result.signature,
                in_amount=order.in_amount,
                out_amount=order.out_amount,
                slot=result.slot,
                latency_trace=trace,
                reconciliation=reconciliation,
                reconciliation_error=reconciliation_error,
            )

        reconciliation: LiveFillReconciliation | None = None
        reconciliation_error: str | None = None
        reconcile_started = time.monotonic()
        try:
            reconciliation = await self._reconcile_confirmed_fill_async(
                signature=result.signature,
                token_mint=token_mint,
                expected_in_amount=order.in_amount,
                expected_out_amount=order.out_amount,
                slot=result.slot,
            )
        except Exception as exc:  # noqa: BLE001
            reconciliation_error = str(exc)
        trace["reconcile_ms"] = (time.monotonic() - reconcile_started) * 1000.0
        trace["total_execution_ms"] = (
            time.monotonic() - float(trace["__total_started_monotonic"])
        ) * 1000.0

        logger.info(
            "✅ LIVE BUY async confirmed: sig=%s | in=%d lamports | out=%d tokens | slot=%s",
            result.signature,
            order.in_amount,
            order.out_amount,
            result.slot,
        )
        return self._result_success(
            signature=result.signature,
            in_amount=order.in_amount,
            out_amount=order.out_amount,
            slot=result.slot,
            latency_trace=trace,
            reconciliation=reconciliation,
            reconciliation_error=reconciliation_error,
        )

    # ------------------------------------------------------------------
    # Buy (SOL → Token)
    # ------------------------------------------------------------------

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
        """Execute a live BUY swap (SOL → token).

        Parameters
        ----------
        token_mint:
            Mint address of the token to buy.
        size_sol:
            Amount of SOL to spend.
        current_exposure_sol:
            Current total SOL exposure across open positions.
        open_position_count:
            Number of currently open positions.

        Returns
        -------
        LiveTradeResult
        """
        logger.info(
            "🔴 LIVE BUY request: %s | size=%.4f SOL | exposure=%.4f SOL | open=%d",
            token_mint[:12],
            size_sol,
            current_exposure_sol,
            open_position_count,
        )
        trace: dict[str, Any] = {
            "path": "live_buy",
            "started_at": self._now_iso(),
            "__total_started_monotonic": time.monotonic(),
        }

        preflight_started = time.monotonic()
        try:
            self._preflight_checks(size_sol, current_exposure_sol, open_position_count)
        except LiveExecutionError as exc:
            logger.error("LIVE BUY preflight failed: %s", exc)
            trace["preflight_ms"] = (time.monotonic() - preflight_started) * 1000.0
            return self._result_with_error(error=str(exc), latency_trace=trace)
        trace["preflight_ms"] = (time.monotonic() - preflight_started) * 1000.0

        # Step 1: Resolve priority fee, then get order (quote + unsigned TX in one call)
        amount_lamports = int(size_sol * LAMPORTS_PER_SOL)
        local_buy_out = self._local_quote_out_amount(
            input_mint=SOL_MINT,
            output_mint=token_mint,
            amount=amount_lamports,
        )
        if local_buy_out is not None:
            trace["local_quote_buy_out_amount"] = int(local_buy_out)
        priority_started = time.monotonic()
        order_policy = self._resolve_order_policy(
            side="buy", strategy=strategy, source_program=source_program
        )
        trace["priority_fee_resolve_ms"] = (time.monotonic() - priority_started) * 1000.0
        trace["priority_fee_lamports"] = int(order_policy.priority_fee_lamports)
        trace["jito_tip_lamports"] = int(order_policy.jito_tip_lamports)
        trace["slippage_bps"] = order_policy.slippage_bps
        trace["slippage_mode"] = "jupiter_auto" if order_policy.slippage_bps is None else "fixed"
        trace["broadcast_fee_type"] = order_policy.broadcast_fee_type
        try:
            self._enforce_buy_wallet_balance(size_sol, order_policy, trace)
        except LiveExecutionError as exc:
            logger.error("LIVE BUY wallet check failed: %s", exc)
            return self._result_with_error(error=str(exc), latency_trace=trace)
        order_started = time.monotonic()
        try:
            native_order = self._try_native_pump_amm_buy_tx(
                token_mint=token_mint,
                amount_lamports=amount_lamports,
                order_policy=order_policy,
                trace=trace,
            )
            if native_order is not None:
                order = native_order
            elif self._requires_custom_jupiter_tx():
                try:
                    order = self._get_sender_compliant_order_tx(
                        input_mint=SOL_MINT,
                        output_mint=token_mint,
                        amount=amount_lamports,
                        order_policy=order_policy,
                        trace=trace,
                    )
                except LiveExecutionError:
                    # Pool-health signals (price impact, preflight) must abort
                    # the buy — do not bypass them via a Metis-quote fallback.
                    raise
                except Exception as exc:  # noqa: BLE001
                    trace["sender_order_fallback_reason"] = str(exc)
                    logger.warning(
                        "LIVE BUY sender order rebuild fallback for %s: %s",
                        token_mint[:12],
                        exc,
                    )
                    order = self._build_custom_swap_tx(
                        input_mint=SOL_MINT,
                        output_mint=token_mint,
                        amount=amount_lamports,
                        order_policy=order_policy,
                        trace=trace,
                    )
            else:
                order = self.jupiter.get_order(
                    input_mint=SOL_MINT,
                    output_mint=token_mint,
                    amount=amount_lamports,
                    slippage_bps=order_policy.slippage_bps,
                    user_public_key=self.signer.get_public_key(),
                    priority_fee_lamports=order_policy.priority_fee_lamports,
                    jito_tip_lamports=order_policy.jito_tip_lamports,
                    broadcast_fee_type=order_policy.broadcast_fee_type,
                )
                trace["jupiter_order_ms"] = (time.monotonic() - order_started) * 1000.0
                self._check_jupiter_buy_price_impact(
                    token_mint=token_mint,
                    price_impact_pct=order.price_impact_pct,
                    source="jupiter_order",
                    trace=trace,
                    order_policy=order_policy,
                )
        except (JupiterError, LiveExecutionError, BroadcastError) as exc:
            logger.error("LIVE BUY order failed: %s", exc)
            if "jupiter_order_ms" not in trace:
                trace["jupiter_order_ms"] = (time.monotonic() - order_started) * 1000.0
            return self._result_with_error(
                error=f"jupiter_order_failed: {exc}", latency_trace=trace
            )

        # Step 2: Sign locally
        sign_started = time.monotonic()
        try:
            signed_tx = self.signer.sign_transaction(order.raw_transaction)
        except SignerError as exc:
            logger.error("LIVE BUY signing failed: %s", exc)
            trace["sign_ms"] = (time.monotonic() - sign_started) * 1000.0
            return self._result_with_error(error=f"signing_failed: {exc}", latency_trace=trace)
        trace["sign_ms"] = (time.monotonic() - sign_started) * 1000.0

        # Step 2b: Optional preflight simulation (cold path). Catches deterministic
        # reverts (Jupiter Route CPI into broken pool, Token-2022 mismatch, stale ATA)
        # before we pay the priority fee + tip.
        preflight_sim_started = time.monotonic()
        preflight_err = self._preflight_simulate_buy(signed_tx, token_mint=token_mint, trace=trace)
        trace["preflight_sim_ms"] = (time.monotonic() - preflight_sim_started) * 1000.0
        if preflight_err is not None:
            return self._result_with_error(error=preflight_err, latency_trace=trace)

        # Step 3: Broadcast via Helius
        try:
            result: BroadcastResult = self.broadcaster.broadcast(
                signed_tx,
                last_valid_block_height=order.last_valid_block_height,
            )
        except BroadcastError as exc:
            logger.error("LIVE BUY broadcast failed: %s", exc)
            return self._result_with_error(error=f"broadcast_failed: {exc}", latency_trace=trace)
        self._apply_broadcast_trace(trace, result)

        if not result.confirmed:
            logger.warning(
                "LIVE BUY tx sent but NOT confirmed: sig=%s error=%s",
                result.signature,
                result.error,
            )
            reconciliation, reconciliation_error = self._reconcile_failed_transaction(
                signature=result.signature,
                token_mint=token_mint,
                expected_in_amount=order.in_amount,
                expected_out_amount=order.out_amount,
                slot=result.slot,
                trace=trace,
            )
            return self._result_with_error(
                error=f"not_confirmed: {result.error}",
                signature=result.signature,
                in_amount=order.in_amount,
                out_amount=order.out_amount,
                slot=result.slot,
                latency_trace=trace,
                reconciliation=reconciliation,
                reconciliation_error=reconciliation_error,
            )

        reconciliation: LiveFillReconciliation | None = None
        reconciliation_error: str | None = None
        reconcile_started = time.monotonic()
        try:
            reconciliation = self._reconcile_confirmed_fill(
                signature=result.signature,
                token_mint=token_mint,
                expected_in_amount=order.in_amount,
                expected_out_amount=order.out_amount,
                slot=result.slot,
            )
        except Exception as exc:  # noqa: BLE001
            reconciliation_error = str(exc)
        trace["reconcile_ms"] = (time.monotonic() - reconcile_started) * 1000.0
        trace["total_execution_ms"] = (
            time.monotonic() - float(trace["__total_started_monotonic"])
        ) * 1000.0

        logger.info(
            "✅ LIVE BUY confirmed: sig=%s | in=%d lamports | out=%d tokens | slot=%s",
            result.signature,
            order.in_amount,
            order.out_amount,
            result.slot,
        )
        return self._result_success(
            signature=result.signature,
            in_amount=order.in_amount,
            out_amount=order.out_amount,
            slot=result.slot,
            latency_trace=trace,
            reconciliation=reconciliation,
            reconciliation_error=reconciliation_error,
        )

    # ------------------------------------------------------------------
    # Sell (Token → SOL)
    # ------------------------------------------------------------------

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
        """Execute a live SELL swap (token → SOL).

        Parameters
        ----------
        token_mint:
            Mint address of the token to sell.
        token_amount:
            Amount of the token to sell (in smallest unit).
        current_exposure_sol:
            Current total SOL exposure (for logging; not enforced on sells).
        open_position_count:
            Number of currently open positions (for logging).
        prefer_jupiter:
            If True, skip the native Pump-AMM sell builder — mature pools overflow
            Custom 6023, so the main lane forces Jupiter routing.
        strategy:
            Strategy-id tag used by :meth:`_resolve_order_policy` to select
            per-lane fee/tip overrides.

        Returns
        -------
        LiveTradeResult
        """
        logger.info(
            "🔴 LIVE SELL request: %s | amount=%d | exposure=%.4f SOL",
            token_mint[:12],
            token_amount,
            current_exposure_sol,
        )
        trace: dict[str, Any] = {
            "path": "live_sell",
            "started_at": self._now_iso(),
            "__total_started_monotonic": time.monotonic(),
            "strategy": strategy or "default",
        }

        # Signer must be valid for sells too
        if self.signer.get_public_key() is None:
            msg = "Signer not validated – refusing live execution"
            logger.error("LIVE SELL preflight failed: %s", msg)
            return self._result_with_error(error=msg, latency_trace=trace)

        if not self.config.helius_rpc_url:
            msg = "HELIUS_RPC_URL is required for live execution"
            logger.error("LIVE SELL preflight failed: %s", msg)
            return self._result_with_error(error=msg, latency_trace=trace)

        try:
            token_amount, close_token_account = self._resolve_wallet_token_balance(
                token_mint=token_mint,
                requested_token_amount=token_amount,
                close_token_account=close_token_account,
                trace=trace,
            )
        except LiveExecutionError as exc:
            logger.error("LIVE SELL wallet balance check failed: %s", exc)
            return self._result_with_error(error=str(exc), latency_trace=trace)

        # Step 1: Resolve priority fee, then get order (quote + unsigned TX in one call)
        priority_started = time.monotonic()
        order_policy = self._resolve_order_policy(
            side="sell", strategy=strategy, source_program=source_program
        )
        trace["priority_fee_resolve_ms"] = (time.monotonic() - priority_started) * 1000.0
        trace["priority_fee_lamports"] = int(order_policy.priority_fee_lamports)
        trace["jito_tip_lamports"] = int(order_policy.jito_tip_lamports)
        trace["slippage_bps"] = order_policy.slippage_bps
        trace["slippage_mode"] = "jupiter_auto" if order_policy.slippage_bps is None else "fixed"
        trace["broadcast_fee_type"] = order_policy.broadcast_fee_type
        order_started = time.monotonic()
        try:
            if prefer_jupiter:
                trace["native_pump_skipped"] = "prefer_jupiter"
                native_order = None
            else:
                native_order = self._try_native_pump_amm_sell_tx(
                    token_mint=token_mint,
                    token_amount=token_amount,
                    order_policy=order_policy,
                    close_input_token_account=close_token_account,
                    trace=trace,
                )
            if native_order is not None:
                order = native_order
            elif self._requires_custom_jupiter_tx():
                order = self._build_custom_swap_tx(
                    input_mint=token_mint,
                    output_mint=SOL_MINT,
                    amount=token_amount,
                    order_policy=order_policy,
                    close_input_token_account=close_token_account,
                    trace=trace,
                )
            else:
                order = self.jupiter.get_order(
                    input_mint=token_mint,
                    output_mint=SOL_MINT,
                    amount=token_amount,
                    slippage_bps=order_policy.slippage_bps,
                    user_public_key=self.signer.get_public_key(),
                    priority_fee_lamports=order_policy.priority_fee_lamports,
                    jito_tip_lamports=order_policy.jito_tip_lamports,
                    broadcast_fee_type=order_policy.broadcast_fee_type,
                )
                trace["jupiter_order_ms"] = (time.monotonic() - order_started) * 1000.0
                self._check_jupiter_sell_price_impact(
                    token_mint=token_mint,
                    price_impact_pct=order.price_impact_pct,
                    source="jupiter_order",
                    trace=trace,
                    order_policy=order_policy,
                )
        except (JupiterError, LiveExecutionError, BroadcastError) as exc:
            logger.error("LIVE SELL order failed: %s", exc)
            if "jupiter_order_ms" not in trace:
                trace["jupiter_order_ms"] = (time.monotonic() - order_started) * 1000.0
            return self._result_with_error(
                error=f"jupiter_order_failed: {exc}", latency_trace=trace
            )
        try:
            self._validate_live_sell_viability(
                expected_out_lamports=self._conservative_sell_out_lamports(
                    token_mint=token_mint,
                    token_amount=token_amount,
                    routed_out_lamports=order.out_amount,
                    trace=trace,
                ),
                order_policy=order_policy,
                close_token_account=close_token_account,
            )
        except LiveExecutionError as exc:
            error = str(exc)
            if error.startswith("dust_exit_blocked") and bool(
                trace.get("wallet_token_close_account_effective")
            ):
                error = f"wallet_token_balance_dust: {error}"
            logger.error("LIVE SELL viability check failed: %s", error)
            return self._result_with_error(error=error, latency_trace=trace)

        # Step 2: Sign locally
        sign_started = time.monotonic()
        try:
            signed_tx = self.signer.sign_transaction(order.raw_transaction)
        except SignerError as exc:
            logger.error("LIVE SELL signing failed: %s", exc)
            trace["sign_ms"] = (time.monotonic() - sign_started) * 1000.0
            return self._result_with_error(error=f"signing_failed: {exc}", latency_trace=trace)
        trace["sign_ms"] = (time.monotonic() - sign_started) * 1000.0

        # Step 2b: Optional preflight simulation (cold path only — prebuilt sells
        # skip this to preserve their latency advantage). Catches deterministic
        # failures (stale ATA, Token-2022 mismatch) before we pay the tip.
        preflight_started = time.monotonic()
        preflight_err = self._preflight_simulate_sell(signed_tx, token_mint=token_mint, trace=trace)
        trace["preflight_ms"] = (time.monotonic() - preflight_started) * 1000.0
        if preflight_err is not None:
            return self._result_with_error(error=preflight_err, latency_trace=trace)

        # Step 3: Broadcast via Helius
        try:
            result: BroadcastResult = self.broadcaster.broadcast(
                signed_tx,
                last_valid_block_height=order.last_valid_block_height,
            )
        except BroadcastError as exc:
            logger.error("LIVE SELL broadcast failed: %s", exc)
            return self._result_with_error(error=f"broadcast_failed: {exc}", latency_trace=trace)
        self._apply_broadcast_trace(trace, result)

        if not result.confirmed:
            logger.warning(
                "LIVE SELL tx sent but NOT confirmed: sig=%s error=%s",
                result.signature,
                result.error,
            )
            self._maybe_mark_pump_dead(trace=trace, error=result.error, token_mint=token_mint)
            reconciliation, reconciliation_error = self._reconcile_failed_transaction(
                signature=result.signature,
                token_mint=token_mint,
                expected_in_amount=order.in_amount,
                expected_out_amount=order.out_amount,
                slot=result.slot,
                trace=trace,
            )
            return self._result_with_error(
                error=f"not_confirmed: {result.error}",
                signature=result.signature,
                in_amount=order.in_amount,
                out_amount=order.out_amount,
                slot=result.slot,
                latency_trace=trace,
                reconciliation=reconciliation,
                reconciliation_error=reconciliation_error,
            )

        reconciliation: LiveFillReconciliation | None = None
        reconciliation_error: str | None = None
        reconcile_started = time.monotonic()
        try:
            reconciliation = self._reconcile_confirmed_fill(
                signature=result.signature,
                token_mint=token_mint,
                expected_in_amount=order.in_amount,
                expected_out_amount=order.out_amount,
                slot=result.slot,
            )
        except Exception as exc:  # noqa: BLE001
            reconciliation_error = str(exc)
        trace["reconcile_ms"] = (time.monotonic() - reconcile_started) * 1000.0
        trace["total_execution_ms"] = (
            time.monotonic() - float(trace["__total_started_monotonic"])
        ) * 1000.0

        logger.info(
            "✅ LIVE SELL confirmed: sig=%s | in=%d tokens | out=%d lamports | slot=%s",
            result.signature,
            order.in_amount,
            order.out_amount,
            result.slot,
        )
        return self._result_success(
            signature=result.signature,
            in_amount=order.in_amount,
            out_amount=order.out_amount,
            slot=result.slot,
            latency_trace=trace,
            reconciliation=reconciliation,
            reconciliation_error=reconciliation_error,
        )
