"""Rule-aware exit logic for paper and live trading."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from src.bot.models import RuntimeFeatures
from src.execution.jupiter_client import LAMPORTS_PER_SOL
from src.execution.trade_executor import PaperTradeEstimate, SOL_MINT, TradeExecutor
from src.portfolio.position_manager import PositionManager
from src.portfolio.proven_winners import append_winner as _append_proven_winner
from src.portfolio.rule_performance import RulePerformanceTracker
from src.strategy.risk_manager import RiskManager
from src.storage.bot_db import BotDB
from src.storage.event_log import EventLogger
from src.strategy.local_quote import PumpAMMQuoteEngine

# Optional late-import to avoid circular dependency
_ExitMLPredictor = None

logger = logging.getLogger(__name__)

BASE_STOPS = {
    "default_recovery": -0.35,
    "default_cluster": -0.30,
    "default_momentum": -0.25,
    "default_sniper": -0.12,
    "default_wallet": -0.15,
    # Mature-pair main-lane profile (post-graduation Raydium V4 canary).
    # Trades smaller moves than fresh-momentum: wider TP, tighter SL than
    # the momentum defaults. Max-hold is enforced by absolute_max_hold_sec
    # (3600s) until a dedicated per-profile max_hold is needed.
    "mature_pair_v1": -0.15,
}
EPS = 1e-12


def _coerce_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _merge_latency_trace(*traces: Any) -> dict[str, Any]:
    """Merge multiple latency trace payloads in order."""
    merged: dict[str, Any] = {}
    for trace in traces:
        if isinstance(trace, dict):
            merged.update(trace)
    return merged


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:  # noqa: BLE001
        return None


def _resolve_live_recorded_at_dt(fallback_dt: datetime, *traces: Any) -> datetime:
    for key in (
        "broadcast_confirmed_at",
        "broadcast_sent_at",
        "confirmed_at",
        "sent_at",
        "started_at",
    ):
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            parsed = _parse_iso_datetime(trace.get(key))
            if parsed is not None:
                return parsed
    return fallback_dt.astimezone(timezone.utc)


def _resolve_live_recorded_at_iso(fallback_dt: datetime, *traces: Any) -> str:
    return _resolve_live_recorded_at_dt(fallback_dt, *traces).isoformat()


def _live_sell_reconciliation_fields(
    result: Any,
    fallback_gross_out_sol: float,
    fallback_token_amount: float,
) -> tuple[float, float, float, float, dict[str, Any]]:
    """Return exact live sell gross/net SOL, token amount, fee SOL, and metadata details."""
    reconciliation = getattr(result, "reconciliation", None)
    details: dict[str, Any] = {
        "live_execution_latency_trace": dict(getattr(result, "latency_trace", {}) or {}),
        "live_reconciliation_error": getattr(result, "reconciliation_error", None),
    }
    if reconciliation is None:
        fallback_gross = float(fallback_gross_out_sol or 0.0)
        return (
            fallback_gross,
            fallback_gross,
            float(fallback_token_amount),
            0.0,
            details,
        )

    net_out_sol = float(reconciliation.wallet_delta_lamports) / float(LAMPORTS_PER_SOL)
    sell_token_amount = (
        abs(float(reconciliation.token_delta_raw))
        if reconciliation.token_delta_raw != 0
        else float(fallback_token_amount)
    )
    exact_fee_sol = float(reconciliation.fee_lamports) / float(LAMPORTS_PER_SOL)
    tip_sol = float(getattr(reconciliation, "tip_lamports", 0) or 0) / float(LAMPORTS_PER_SOL)
    fee_sol = exact_fee_sol + tip_sol
    gross_out_sol = float(fallback_gross_out_sol or 0.0)
    expected_wallet_delta_without_refund = gross_out_sol - fee_sol
    rent_refund_sol = max(0.0, net_out_sol - expected_wallet_delta_without_refund)
    if rent_refund_sol > 0.0:
        gross_out_sol += rent_refund_sol
    details.update(
        {
            "live_exact_fee_lamports": int(reconciliation.fee_lamports),
            "live_tip_lamports": int(getattr(reconciliation, "tip_lamports", 0) or 0),
            "live_effective_fee_lamports": int(round(fee_sol * float(LAMPORTS_PER_SOL))),
            "live_wallet_pre_lamports": int(reconciliation.wallet_pre_lamports),
            "live_wallet_post_lamports": int(reconciliation.wallet_post_lamports),
            "live_wallet_delta_lamports": int(reconciliation.wallet_delta_lamports),
            "live_token_pre_raw": int(reconciliation.token_pre_raw),
            "live_token_post_raw": int(reconciliation.token_post_raw),
            "live_token_delta_raw": int(reconciliation.token_delta_raw),
            "live_token_decimals": int(reconciliation.token_decimals),
            "live_quote_out_amount_diff_raw": int(reconciliation.quote_out_amount_diff),
            "live_fill_slot": reconciliation.slot,
            "live_rent_refund_lamports": int(round(rent_refund_sol * float(LAMPORTS_PER_SOL))),
            "live_actual_gross_out_sol": float(gross_out_sol),
            "live_actual_net_out_sol": float(net_out_sol),
        }
    )
    return gross_out_sol, net_out_sol, sell_token_amount, fee_sol, details


def _allocate_live_cost_basis(
    metadata: dict[str, Any],
    *,
    current_size_sol: float,
    current_token_amount: float,
    sell_size_sol: float,
    sell_token_amount: float,
    close_position: bool,
) -> tuple[float, float, float]:
    remaining_cost_basis_sol = float(
        metadata.get("paper_remaining_cost_basis_sol", current_size_sol) or current_size_sol
    )
    remaining_token_raw = float(
        metadata.get("paper_remaining_token_raw", current_token_amount) or current_token_amount
    )
    reference_token_amount = max(remaining_token_raw, current_token_amount, sell_token_amount, EPS)
    if close_position:
        sell_ratio = 1.0
    else:
        sell_ratio = max(0.0, min(float(sell_token_amount) / reference_token_amount, 1.0))
        if sell_ratio <= EPS and current_size_sol > EPS:
            sell_ratio = max(0.0, min(float(sell_size_sol) / float(current_size_sol), 1.0))
    allocated_cost_basis_sol = (
        remaining_cost_basis_sol
        if close_position
        else min(
            remaining_cost_basis_sol,
            remaining_cost_basis_sol * sell_ratio,
        )
    )
    new_remaining_cost_basis_sol = max(0.0, remaining_cost_basis_sol - allocated_cost_basis_sol)
    new_remaining_token_raw = (
        0.0 if close_position else max(0.0, reference_token_amount - float(sell_token_amount))
    )
    return (
        allocated_cost_basis_sol,
        new_remaining_cost_basis_sol,
        new_remaining_token_raw,
    )


class ExitEngine:
    """Manage staged exits in paper or live mode."""

    def __init__(
        self,
        db: BotDB,
        position_manager: PositionManager,
        rule_performance: RulePerformanceTracker,
        risk_manager: RiskManager,
        event_log: EventLogger,
        trade_executor: Optional[TradeExecutor] = None,
        exit_predictor: Any = None,
        quote_cache: Any = None,
        local_quote_engine: Optional[PumpAMMQuoteEngine] = None,
        live_sell_cache: Any = None,
    ) -> None:
        self.db = db
        self.position_manager = position_manager
        self.rule_performance = rule_performance
        self.risk_manager = risk_manager
        self.event_log = event_log
        self.trade_executor = trade_executor
        self.exit_predictor = exit_predictor  # ExitMLPredictor | None
        self.quote_cache = quote_cache  # PositionQuoteCache | None
        self.local_quote_engine = local_quote_engine  # PumpAMMQuoteEngine | None
        self.live_sell_cache = live_sell_cache  # LiveSellCache | None

        cfg = risk_manager.config
        self.dead_token_hold_sec = int(getattr(cfg, "dead_token_hold_sec", 180))
        self.dead_token_move_pct = float(getattr(cfg, "dead_token_move_pct", 0.02))
        self.dead_token_confirm_ticks = max(1, int(getattr(cfg, "dead_token_confirm_ticks", 2)))
        self.dead_token_max_tx_count_60s = max(
            0, int(getattr(cfg, "dead_token_max_tx_count_60s", 3))
        )
        self.dead_token_max_volume_sol_60s = float(
            getattr(cfg, "dead_token_max_volume_sol_60s", 3.0)
        )

        # Requested staged exits:
        # 50% @ 2x, 30% @ 4x, 20% @ 10x or timeout after stage-2.
        self.exit_tp1_multiple = float(getattr(cfg, "exit_tp1_multiple", 1.0))
        self.exit_tp2_multiple = float(getattr(cfg, "exit_tp2_multiple", 3.0))
        self.exit_tp3_multiple = float(getattr(cfg, "exit_tp3_multiple", 9.0))
        tp1_fraction = max(0.0, min(float(getattr(cfg, "exit_tp1_sell_fraction", 0.5)), 1.0))
        tp2_fraction = max(
            0.0,
            min(float(getattr(cfg, "exit_tp2_sell_fraction", 0.3)), 1.0 - tp1_fraction),
        )
        self.exit_tp1_sell_fraction = tp1_fraction
        self.exit_tp2_sell_fraction = tp2_fraction

        # Tweak 1: protect remainder after TP1.
        self.post_tp1_stop_pnl = float(getattr(cfg, "post_tp1_stop_pnl", 0.02))
        self.exit_rule_stop_overrides = dict(getattr(cfg, "exit_rule_stop_overrides", {}) or {})
        self.sniper_take_profit_pnl = float(getattr(cfg, "sniper_take_profit_pnl", 0.15))
        self.sniper_tp_min_gross_sol_floor = float(
            getattr(cfg, "sniper_tp_min_gross_sol_floor", 0.003)
        )
        self.sniper_tp_min_gross_fee_multiplier = float(
            getattr(cfg, "sniper_tp_min_gross_fee_multiplier", 2.5)
        )
        self.sniper_tp_min_gross_size_ratio = float(
            getattr(cfg, "sniper_tp_min_gross_size_ratio", 0.015)
        )
        self.sniper_stop_pnl = float(getattr(cfg, "sniper_stop_pnl", -0.10))
        self.sniper_max_hold_sec = int(getattr(cfg, "sniper_max_hold_sec", 90))
        self.sniper_tp_confirm_ticks = max(1, int(getattr(cfg, "sniper_tp_confirm_ticks", 1)))
        self.sniper_stop_confirm_ticks = max(1, int(getattr(cfg, "sniper_stop_confirm_ticks", 2)))
        self.sniper_stop_min_hold_sec = max(0, int(getattr(cfg, "sniper_stop_min_hold_sec", 3)))
        self.sniper_tp_jupiter_verify = bool(getattr(cfg, "sniper_tp_jupiter_verify", True))
        # Live-mode TP bypass: at ``pnl >= tp1 * multiplier`` (default 2× TP, i.e. +16%
        # when tp1=8%), skip the volume_sol_30s floor and outlier_jump_guard that
        # otherwise trap clearly profitable sniper positions when a pool goes quiet
        # post-pump. Paper already has an equivalent Jupiter-verify bypass.
        self.sniper_tp_live_bypass_multiplier = float(
            getattr(cfg, "sniper_tp_live_bypass_multiplier", 2.0)
        )
        # ML exit peak lock + veto
        self.ml_exit_peak_lock_enabled = bool(getattr(cfg, "ml_exit_peak_lock_enabled", False))
        self.ml_exit_peak_lock_min_pnl = float(getattr(cfg, "ml_exit_peak_lock_min_pnl", 0.15))
        self.ml_exit_peak_lock_drawdown = float(getattr(cfg, "ml_exit_peak_lock_drawdown", 0.08))
        self.ml_exit_peak_lock_threshold = float(getattr(cfg, "ml_exit_peak_lock_threshold", 0.35))
        self.ml_exit_veto_reasons: set[str] = set(getattr(cfg, "ml_exit_veto_reasons", ()) or ())
        self.ml_exit_veto_threshold = float(getattr(cfg, "ml_exit_veto_threshold", 0.65))
        self.ml_exit_min_hold_sec = int(getattr(cfg, "ml_exit_min_hold_sec", 20))
        self.ml_exit_min_hold_sec_sniper = int(getattr(cfg, "ml_exit_min_hold_sec_sniper", 10))
        self.wallet_take_profit_pnl = float(getattr(cfg, "wallet_take_profit_pnl", 0.15))
        self.wallet_stop_pnl = float(getattr(cfg, "wallet_stop_pnl", -0.15))
        self.wallet_max_hold_sec = int(getattr(cfg, "wallet_max_hold_sec", 180))
        self.wallet_trailing_drawdown = float(getattr(cfg, "wallet_trailing_drawdown", 0.08))
        self.wallet_trailing_arm_confirm_ticks = max(
            1, int(getattr(cfg, "wallet_trailing_arm_confirm_ticks", 2))
        )
        self.wallet_trailing_exit_confirm_ticks = max(
            1, int(getattr(cfg, "wallet_trailing_exit_confirm_ticks", 2))
        )
        self.wallet_tp1_peak = float(getattr(cfg, "wallet_tp1_peak", 0.30))
        self.wallet_tp1_sell_fraction = max(
            0.0, min(float(getattr(cfg, "wallet_tp1_sell_fraction", 0.5)), 1.0)
        )
        self.wallet_tp1_confirm_ticks = max(1, int(getattr(cfg, "wallet_tp1_confirm_ticks", 2)))
        self.wallet_copy_trail_arm_pnl = float(getattr(cfg, "wallet_copy_trail_arm_pnl", 0.05))
        self.wallet_copy_trail_drawdown = float(getattr(cfg, "wallet_copy_trail_drawdown", 0.15))
        self.wallet_copy_hard_stop_pnl = float(getattr(cfg, "wallet_copy_hard_stop_pnl", -0.25))
        self.wallet_copy_max_hold_sec = int(getattr(cfg, "wallet_copy_max_hold_sec", 1800))
        self.wallet_copy_mirror_sell = bool(getattr(cfg, "wallet_copy_mirror_sell", True))
        self.wallet_copy_mirror_sell_profit_threshold = float(
            getattr(cfg, "wallet_copy_mirror_sell_profit_threshold", 0.09)
        )
        self.wallet_copy_tp1_peak = float(getattr(cfg, "wallet_copy_tp1_peak", 0.15))
        self.wallet_copy_tp1_sell_fraction = max(
            0.0, min(float(getattr(cfg, "wallet_copy_tp1_sell_fraction", 0.5)), 1.0)
        )
        self.wallet_copy_tp1_confirm_ticks = max(
            1, int(getattr(cfg, "wallet_copy_tp1_confirm_ticks", 2))
        )
        self.wallet_copy_exit_confirm_ticks = max(
            1, int(getattr(cfg, "wallet_copy_exit_confirm_ticks", 2))
        )
        self.wallet_copy_trail_min_floor = float(getattr(cfg, "wallet_copy_trail_min_floor", 0.02))

        # Tweak 2: trail runner after TP2 and force-close after timeout.
        self.post_tp2_trailing_drawdown = float(getattr(cfg, "post_tp2_trailing_drawdown", 0.25))
        self.post_tp2_timeout_sec = int(getattr(cfg, "post_tp2_timeout_sec", 300))
        self.tp1_confirm_ticks = max(1, int(getattr(cfg, "tp1_confirm_ticks", 2)))
        self.tp2_confirm_ticks = max(1, int(getattr(cfg, "tp2_confirm_ticks", 2)))
        self.tp1_min_volume_sol_30s = float(getattr(cfg, "tp1_min_volume_sol_30s", 2.0))
        self.tp2_min_volume_sol_30s = float(getattr(cfg, "tp2_min_volume_sol_30s", 2.0))
        self.tp3_confirm_ticks = max(1, int(getattr(cfg, "tp3_confirm_ticks", 2)))
        self.tp2_fast_confirm_ticks = max(1, int(getattr(cfg, "tp2_fast_confirm_ticks", 2)))
        self.tp2_fast_min_volume_sol_30s = float(getattr(cfg, "tp2_fast_min_volume_sol_30s", 2.0))
        self.tp3_min_volume_sol_30s = float(getattr(cfg, "tp3_min_volume_sol_30s", 2.0))
        self.exit_price_max_step_multiple = max(
            1.01, float(getattr(cfg, "exit_price_max_step_multiple", 2.5))
        )
        self.exit_outlier_max_pnl_jump = float(getattr(cfg, "exit_outlier_max_pnl_jump", 20.0))
        self.exit_outlier_low_volume_sol_30s = float(
            getattr(cfg, "exit_outlier_low_volume_sol_30s", 2.0)
        )
        self.exit_max_peak_pnl_multiple = float(getattr(cfg, "exit_max_peak_pnl_multiple", 9.0))
        self.stage0_loss_timeout_sec = int(getattr(cfg, "stage0_loss_timeout_sec", 900))
        self.stage0_loss_timeout_max_pnl = float(getattr(cfg, "stage0_loss_timeout_max_pnl", 0.0))
        self.stage0_early_profit_window_sec = int(
            getattr(cfg, "stage0_early_profit_window_sec", 120)
        )
        self.stage0_early_profit_min_pnl = float(getattr(cfg, "stage0_early_profit_min_pnl", 1.0))
        self.stage0_early_profit_max_pnl = float(getattr(cfg, "stage0_early_profit_max_pnl", 1.75))
        self.stage0_early_profit_confirm_ticks = max(
            1, int(getattr(cfg, "stage0_early_profit_confirm_ticks", 2))
        )
        early_profit_fraction = float(getattr(cfg, "stage0_early_profit_sell_fraction", 0.30))
        if 0.0 < early_profit_fraction < 0.05:
            early_profit_fraction = 0.05
        self.stage0_early_profit_sell_fraction = max(0.0, min(early_profit_fraction, 1.0))
        self.stage0_crash_guard_window_sec = int(getattr(cfg, "stage0_crash_guard_window_sec", 120))
        self.stage0_crash_guard_min_pnl = float(getattr(cfg, "stage0_crash_guard_min_pnl", -0.20))
        self.stage0_crash_guard_min_hold_sec = int(
            getattr(cfg, "stage0_crash_guard_min_hold_sec", 20)
        )
        self.stage0_crash_guard_confirm_ticks = max(
            1, int(getattr(cfg, "stage0_crash_guard_confirm_ticks", 2))
        )
        self.pre_tp1_retrace_lock_min_hold_sec = max(
            0,
            int(getattr(cfg, "pre_tp1_retrace_lock_min_hold_sec", 20)),
        )
        self.pre_tp1_retrace_lock_arm_pnl = float(
            getattr(cfg, "pre_tp1_retrace_lock_arm_pnl", 0.35)
        )
        self.pre_tp1_retrace_lock_drawdown = float(
            getattr(cfg, "pre_tp1_retrace_lock_drawdown", 0.30)
        )
        self.pre_tp1_retrace_lock_floor_pnl = float(
            getattr(cfg, "pre_tp1_retrace_lock_floor_pnl", 0.08)
        )
        self.pre_tp1_retrace_lock_confirm_ticks = max(
            1,
            int(getattr(cfg, "pre_tp1_retrace_lock_confirm_ticks", 2)),
        )
        self.stage0_fast_fail_non_positive_sec = int(
            getattr(cfg, "stage0_fast_fail_non_positive_sec", 60)
        )
        self.stage0_fast_fail_non_positive_max_pnl = float(
            getattr(cfg, "stage0_fast_fail_non_positive_max_pnl", 0.0)
        )
        self.stage0_fast_fail_under_profit_sec = int(
            getattr(cfg, "stage0_fast_fail_under_profit_sec", 120)
        )
        self.stage0_fast_fail_under_profit_min_pnl = float(
            getattr(cfg, "stage0_fast_fail_under_profit_min_pnl", 0.10)
        )
        self.stage0_moderate_positive_timeout_sec = int(
            getattr(cfg, "stage0_moderate_positive_timeout_sec", 900)
        )
        self.stage0_moderate_positive_min_pnl = float(
            getattr(cfg, "stage0_moderate_positive_min_pnl", 0.02)
        )
        self.stage0_moderate_positive_max_pnl = float(
            getattr(cfg, "stage0_moderate_positive_max_pnl", 0.99)
        )
        self.stage0_moderate_positive_skip_profiles: frozenset[str] = frozenset(
            str(p) for p in getattr(cfg, "stage0_moderate_positive_skip_profiles", ()) if p
        )
        self.stage1_low_positive_timeout_sec = int(
            getattr(cfg, "stage1_low_positive_timeout_sec", 900)
        )
        self.stage1_low_positive_min_pnl = float(getattr(cfg, "stage1_low_positive_min_pnl", 1.0))
        self.stage1_low_positive_max_pnl = float(getattr(cfg, "stage1_low_positive_max_pnl", 1.9))
        self.stage1_sub2x_timeout_sec = int(getattr(cfg, "stage1_sub2x_timeout_sec", 1200))
        self.stage1_sub2x_min_pnl = float(getattr(cfg, "stage1_sub2x_min_pnl", 0.02))
        self.stage1_sub2x_max_pnl = float(getattr(cfg, "stage1_sub2x_max_pnl", 0.99))
        self.absolute_max_hold_sec = int(getattr(cfg, "absolute_max_hold_sec", 3600))

        # Set of position IDs currently being executed (sell in-flight).
        # Prevents a second exit check from firing on the same position before
        # the first sell completes and the DB write marks it CLOSED.
        self._exiting_position_ids: set[int] = set()
        self._exiting_position_ids_lock = threading.Lock()

        # Per-mint sell locks. Serialize sells across *different* positions on
        # the same token — back-to-back sells on the same ATA can read stale
        # balance or hit InvalidAccountData if a prior sell closed the ATA.
        self._sell_mint_locks: dict[str, threading.Lock] = {}
        self._sell_mint_locks_guard = threading.Lock()

        # Per-mint sell-failure circuit breaker. After N consecutive sell
        # failures on a mint we stop trying for COOLDOWN seconds — the mint is
        # likely a rug, a Token-2022 we can't route, or the pool is gone, and
        # each retry burns priority fee + tip. State is per-process (reset on
        # restart). Tripped state stores monotonic release time.
        self._mint_sell_fail_counts: dict[str, int] = {}
        self._mint_sell_release_at: dict[str, float] = {}
        self._mint_sell_breaker_lock = threading.Lock()
        self._mint_sell_breaker_last_log_at: dict[str, float] = {}
        # Mints classified "no sellable route" (Jupiter returns quote 400
        # "Cannot compute other amount threshold", etc.) are flagged
        # permanently stuck — no cooldown release, they stay skipped for
        # the life of the process.
        self._mint_sell_permanent_stuck: set[str] = set()
        self._mint_sell_no_route_counts: dict[str, int] = {}
        # Preflight slippage 6004 is a distinct class: Jupiter / native-AMM
        # quote vs actual swap drift over the slippage tolerance. Bounded
        # drift normally self-heals, but when our local-quote math is stale
        # or the pool is mis-reserved we keep hitting it every retry. After
        # N repeated 6004 on one mint, flag permanently stuck so we stop
        # burning priority fee + tip on every exit attempt.
        self._mint_sell_slippage_counts: dict[str, int] = {}

    @property
    def _is_live(self) -> bool:
        return self.trade_executor is not None and self.trade_executor.live

    @property
    def _mode_label(self) -> str:
        return "LIVE" if self._is_live else "PAPER"

    def _try_acquire_exit_slot(self, position_id: int) -> bool:
        """Atomically mark one position as currently exiting."""
        with self._exiting_position_ids_lock:
            if position_id in self._exiting_position_ids:
                return False
            self._exiting_position_ids.add(position_id)
            return True

    def _release_exit_slot(self, position_id: int) -> None:
        """Release one in-flight exit slot."""
        with self._exiting_position_ids_lock:
            self._exiting_position_ids.discard(position_id)

    def _acquire_sell_mint_lock(self, token_mint: str) -> threading.Lock:
        """Return (lazily create) the per-mint sell lock."""
        with self._sell_mint_locks_guard:
            lock = self._sell_mint_locks.get(token_mint)
            if lock is None:
                lock = threading.Lock()
                self._sell_mint_locks[token_mint] = lock
            return lock

    def _sell_breaker_tripped(self, token_mint: str) -> bool:
        """Return True if this mint is under an active cool-down from repeated sell failures."""
        with self._mint_sell_breaker_lock:
            if token_mint in self._mint_sell_permanent_stuck:
                return True
            release_at = self._mint_sell_release_at.get(token_mint, 0.0)
            if release_at <= 0.0:
                return False
            if time.monotonic() >= release_at:
                self._mint_sell_release_at.pop(token_mint, None)
                self._mint_sell_fail_counts.pop(token_mint, None)
                return False
            return True

    def _record_sell_breaker_success(self, token_mint: str) -> None:
        with self._mint_sell_breaker_lock:
            self._mint_sell_fail_counts.pop(token_mint, None)
            self._mint_sell_release_at.pop(token_mint, None)
            self._mint_sell_no_route_counts.pop(token_mint, None)
            self._mint_sell_slippage_counts.pop(token_mint, None)

    _NO_ROUTE_ERROR_MARKERS = (
        "Cannot compute other amount threshold",
        "Could not find any route",
        "TOKEN_NOT_TRADABLE",
        "No routes found",
        "jupiter_price_impact_out_of_range",
        "jupiter_sell_impact_out_of_range",
    )

    # Anchor custom error 6004 on Pump-AMM / pump-swap programs is
    # ExceededSlippage — quote drifted past min_out during preflight. We
    # match both the raw "Custom': 6004" signature surfaced by the RPC
    # simulate response and the program-level "ExceededSlippage" name.
    _PREFLIGHT_SLIPPAGE_MARKERS = (
        'Custom": 6004',
        "Custom': 6004",
        "Custom(6004)",
        "ExceededSlippage",
        "exceeded_slippage",
    )

    @classmethod
    def _classify_sell_error(cls, error: str | None) -> str:
        """Return 'no_route' | 'preflight_slippage' | 'generic' for an error string."""
        if not error:
            return "generic"
        for marker in cls._NO_ROUTE_ERROR_MARKERS:
            if marker in error:
                return "no_route"
        for marker in cls._PREFLIGHT_SLIPPAGE_MARKERS:
            if marker in error:
                return "preflight_slippage"
        return "generic"

    def _record_sell_breaker_failure(
        self,
        token_mint: str,
        *,
        error_class: str = "generic",
    ) -> int:
        """Record one sell failure; trip the breaker if threshold reached.

        When ``error_class == "no_route"`` the mint is flagged permanently
        stuck after a smaller threshold — the pool has no sellable route,
        retrying burns fees indefinitely and the breaker's cooldown release
        would just restart the loop.

        Returns the new failure count so callers can log it.
        """
        cfg = self.risk_manager.config
        threshold = max(1, int(getattr(cfg, "live_sell_circuit_breaker_threshold", 3) or 3))
        cooldown = max(
            1.0,
            float(getattr(cfg, "live_sell_circuit_breaker_cooldown_sec", 300.0) or 300.0),
        )
        no_route_threshold = max(1, min(threshold, 2))
        slippage_threshold = max(1, int(getattr(cfg, "live_sell_slippage_stuck_threshold", 5) or 5))
        with self._mint_sell_breaker_lock:
            count = self._mint_sell_fail_counts.get(token_mint, 0) + 1
            self._mint_sell_fail_counts[token_mint] = count
            if error_class == "no_route":
                no_route_count = self._mint_sell_no_route_counts.get(token_mint, 0) + 1
                self._mint_sell_no_route_counts[token_mint] = no_route_count
                if no_route_count >= no_route_threshold:
                    self._mint_sell_permanent_stuck.add(token_mint)
            elif error_class == "preflight_slippage":
                slip_count = self._mint_sell_slippage_counts.get(token_mint, 0) + 1
                self._mint_sell_slippage_counts[token_mint] = slip_count
                if slip_count >= slippage_threshold:
                    self._mint_sell_permanent_stuck.add(token_mint)
            if count >= threshold:
                self._mint_sell_release_at[token_mint] = time.monotonic() + cooldown
        return count

    def _exit_leg_already_recorded(
        self, position_id: int, reason: str, close_position: bool
    ) -> bool:
        """Return whether this exact sell leg already exists for one position."""
        existing = self.db.fetchone(
            """
            SELECT id
            FROM trade_legs
            WHERE position_id = ?
              AND action = 'SELL'
              AND COALESCE(reason, '') = ?
              AND close_position = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (position_id, reason, 1 if close_position else 0),
        )
        return existing is not None

    def _record_failed_live_exit_fee_burn(
        self,
        *,
        position: dict[str, Any],
        metadata: dict[str, Any],
        features: RuntimeFeatures,
        result: Any,
        strategy_id: str,
        reason: str,
        decision_latency_trace: dict[str, Any],
    ) -> float:
        reconciliation = getattr(result, "reconciliation", None)
        if reconciliation is None:
            return 0.0
        if int(getattr(reconciliation, "token_delta_raw", 0) or 0) != 0:
            return 0.0
        burn_lamports = max(0, -int(getattr(reconciliation, "wallet_delta_lamports", 0) or 0))
        if burn_lamports <= 0:
            return 0.0
        burn_sol = float(burn_lamports) / float(LAMPORTS_PER_SOL)
        existing_realized = float(position.get("realized_pnl_sol", 0.0) or 0.0)
        updated_realized = existing_realized - burn_sol
        metadata["last_failed_exit_fee_burn_sol"] = burn_sol
        execution_latency_trace = dict(getattr(result, "latency_trace", {}) or {})
        latency_trace = _merge_latency_trace(decision_latency_trace, execution_latency_trace)
        execution_recorded_at = _resolve_live_recorded_at_iso(features.entry_time, latency_trace)
        metadata["last_failed_exit_fee_burn_at"] = execution_recorded_at
        metadata["last_failed_exit_reason"] = reason
        metadata["live_reconciliation_error"] = getattr(result, "reconciliation_error", None)
        self.position_manager.update_position_after_exit(
            position_id=int(position["id"]),
            exit_stage=int(position.get("exit_stage", 0) or 0),
            realized_pnl_sol=updated_realized,
            status=str(position.get("status") or "OPEN"),
            remaining_size_sol=float(position.get("size_sol", 0.0) or 0.0),
            remaining_amount_received=float(position.get("amount_received", 0.0) or 0.0),
            metadata=metadata,
            unrealized_pnl_sol=float(position.get("unrealized_pnl_sol", 0.0) or 0.0),
        )
        self.db.record_trade_leg(
            position_id=int(position["id"]),
            token_mint=features.token_mint,
            action="SELL_FEE",
            mode="live",
            strategy_id=strategy_id,
            selected_rule_id=str(position.get("selected_rule_id") or ""),
            selected_regime=str(position.get("selected_regime") or ""),
            close_position=False,
            quote_used=False,
            quote_source="live_failed_tx_reconciled",
            cost_basis_sol=0.0,
            leg_size_sol=0.0,
            token_amount_raw=0.0,
            gross_sol=0.0,
            net_sol=0.0,
            fee_sol=burn_sol,
            realized_leg_pnl_sol=-burn_sol,
            realized_total_pnl_sol=updated_realized,
            reason="live_sell_failed_fee_burn",
            tx_signature=result.signature,
            created_at=execution_recorded_at,
            metadata={
                "failed_reason": reason,
                "trigger_event_time": features.entry_time.isoformat(),
                "execution_recorded_at": execution_recorded_at,
                "pipeline_latency_trace": decision_latency_trace,
                "execution_latency_trace": execution_latency_trace,
                "latency_trace": latency_trace,
                "reconciliation_error": getattr(result, "reconciliation_error", None),
                "live_exact_fee_lamports": int(getattr(reconciliation, "fee_lamports", 0) or 0),
                "live_tip_lamports": int(getattr(reconciliation, "tip_lamports", 0) or 0),
                "live_wallet_delta_lamports": int(
                    getattr(reconciliation, "wallet_delta_lamports", 0) or 0
                ),
            },
        )
        self.risk_manager.record_daily_loss(
            -burn_sol,
            realized_at=_resolve_live_recorded_at_dt(features.entry_time, latency_trace),
        )
        self.rule_performance.record_exit(
            str(position.get("selected_rule_id") or "unknown"),
            pnl_sol=-burn_sol,
            hit_2x=False,
            hit_5x=False,
            stop_out=False,
            close_position=False,
        )
        self.event_log.log(
            "live_exit_failed_fee_burn",
            {
                "token_mint": features.token_mint,
                "selected_rule_id": position.get("selected_rule_id"),
                "strategy_id": strategy_id,
                "reason": reason,
                "signature": result.signature,
                "trigger_event_time": features.entry_time.isoformat(),
                "execution_recorded_at": execution_recorded_at,
                "fee_burn_sol": burn_sol,
                "wallet_delta_lamports": int(
                    getattr(reconciliation, "wallet_delta_lamports", 0) or 0
                ),
                "exact_fee_lamports": int(getattr(reconciliation, "fee_lamports", 0) or 0),
                "tip_lamports": int(getattr(reconciliation, "tip_lamports", 0) or 0),
                "pipeline_latency_trace": decision_latency_trace,
                "execution_latency_trace": execution_latency_trace,
                "latency_trace": latency_trace,
            },
        )
        return burn_sol

    def _reconcile_external_live_close(
        self,
        *,
        position: dict[str, Any],
        metadata: dict[str, Any],
        features: RuntimeFeatures,
        strategy_id: str,
        reason: str,
        result: Any,
        decision_latency_trace: dict[str, Any],
        forced: bool = False,
    ) -> None:
        execution_latency_trace = dict(getattr(result, "latency_trace", {}) or {})
        latency_trace = _merge_latency_trace(decision_latency_trace, execution_latency_trace)
        execution_recorded_at = _resolve_live_recorded_at_iso(features.entry_time, latency_trace)
        metadata["external_balance_reconciled_at"] = execution_recorded_at
        metadata["external_balance_reconciled_reason"] = str(getattr(result, "error", "") or "")
        metadata["external_balance_raw"] = int(
            execution_latency_trace.get("wallet_token_balance_raw") or 0
        )
        metadata["paper_remaining_cost_basis_sol"] = 0.0
        metadata["paper_remaining_token_raw"] = 0.0
        metadata["last_exit_at"] = execution_recorded_at
        metadata["live_execution_latency_trace"] = execution_latency_trace
        metadata["live_latency_trace"] = latency_trace
        self.position_manager.update_position_after_exit(
            position_id=int(position["id"]),
            exit_stage=int(position.get("exit_stage", 0) or 0),
            realized_pnl_sol=float(position.get("realized_pnl_sol", 0.0) or 0.0),
            status="CLOSED",
            remaining_size_sol=0.0,
            remaining_amount_received=0.0,
            metadata=metadata,
            unrealized_pnl_sol=0.0,
        )
        if self.quote_cache is not None:
            self.quote_cache.unregister(features.token_mint)
        if self.live_sell_cache is not None:
            self.live_sell_cache.unregister(features.token_mint)
        self.risk_manager.set_cooldown(features.token_mint, minutes=30, reason="position_closed")
        self.db.execute(
            """
            INSERT INTO executions (token_mint, action, mode, strategy_id, size_sol, price_sol, tx_signature, status, reason, created_at)
            VALUES (?, 'SELL', 'live', ?, ?, ?, ?, 'RECONCILED', ?, ?)
            """,
            (
                features.token_mint,
                strategy_id,
                float(position.get("size_sol", 0.0) or 0.0),
                float(features.entry_price_sol or 0.0),
                getattr(result, "signature", None),
                f"live_external_reconcile: {reason}",
                execution_recorded_at,
            ),
        )
        self.event_log.log(
            "live_exit_reconciled",
            {
                "token_mint": features.token_mint,
                "selected_rule_id": position.get("selected_rule_id"),
                "strategy_id": strategy_id,
                "reason": reason,
                "forced": bool(forced),
                "trigger_event_time": features.entry_time.isoformat(),
                "execution_recorded_at": execution_recorded_at,
                "external_balance_raw": int(
                    execution_latency_trace.get("wallet_token_balance_raw") or 0
                ),
                "signature": getattr(result, "signature", None),
                "pipeline_latency_trace": decision_latency_trace,
                "execution_latency_trace": execution_latency_trace,
                "latency_trace": latency_trace,
            },
        )

    def _force_close_stuck_rug_position(
        self,
        *,
        position: dict[str, Any],
        metadata: dict[str, Any],
        features: RuntimeFeatures,
        strategy_id: str,
        reason: str,
        last_error: str | None = None,
    ) -> None:
        """Force-close a position whose mint has no sellable route.

        Called when the per-mint sell circuit breaker has flagged the mint as
        permanently stuck. We can't recover the tokens, so realize the full
        remaining cost basis as a loss and mark the DB row CLOSED — freeing
        the portfolio slot and letting the bot keep trading. Without this,
        the position hangs OPEN forever and silently caps concurrency.
        """
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        remaining_size_sol = float(position.get("size_sol", 0.0) or 0.0)
        remaining_token_amount = float(position.get("amount_received", 0.0) or 0.0)
        prior_realized = float(position.get("realized_pnl_sol", 0.0) or 0.0)
        realized_leg_pnl_sol = -remaining_size_sol
        realized_total = prior_realized + realized_leg_pnl_sol

        metadata = dict(metadata or {})
        metadata["last_exit_reason"] = reason
        metadata["last_exit_at"] = now_iso
        metadata["stuck_rug_closed_at"] = now_iso
        metadata["stuck_rug_last_error"] = str(last_error or "")
        metadata["paper_remaining_cost_basis_sol"] = 0.0
        metadata["paper_remaining_token_raw"] = 0.0

        self.position_manager.update_position_after_exit(
            position_id=int(position["id"]),
            exit_stage=int(position.get("exit_stage", 0) or 0),
            realized_pnl_sol=realized_total,
            status="CLOSED",
            remaining_size_sol=0.0,
            remaining_amount_received=0.0,
            metadata=metadata,
            unrealized_pnl_sol=0.0,
        )

        mode_label = "live" if self._is_live else "paper"
        try:
            self.db.record_trade_leg(
                position_id=int(position["id"]),
                token_mint=features.token_mint,
                action="SELL",
                mode=mode_label,
                strategy_id=strategy_id,
                selected_rule_id=position.get("selected_rule_id"),
                selected_regime=position.get("selected_regime"),
                close_position=True,
                stop_out=True,
                cost_basis_sol=remaining_size_sol,
                leg_size_sol=remaining_size_sol,
                token_amount_raw=remaining_token_amount,
                gross_sol=0.0,
                net_sol=0.0,
                fee_sol=0.0,
                realized_leg_pnl_sol=realized_leg_pnl_sol,
                realized_total_pnl_sol=realized_total,
                reason=f"stuck_rug_force_close: {reason}",
                created_at=now_iso,
                metadata={"stuck_rug_last_error": str(last_error or "")},
            )
        except Exception:  # noqa: BLE001
            logger.exception("stuck_rug trade_leg record failed for %s", features.token_mint[:12])

        try:
            self.db.execute(
                """
                INSERT INTO executions (token_mint, action, mode, strategy_id, size_sol, price_sol, tx_signature, status, reason, created_at)
                VALUES (?, 'SELL', ?, ?, ?, ?, NULL, 'FORCE_CLOSED', ?, ?)
                """,
                (
                    features.token_mint,
                    mode_label,
                    strategy_id,
                    remaining_size_sol,
                    float(features.entry_price_sol or 0.0),
                    f"stuck_rug_force_close: {reason}",
                    now_iso,
                ),
            )
        except Exception:  # noqa: BLE001
            logger.exception("stuck_rug execution record failed for %s", features.token_mint[:12])

        if self.quote_cache is not None:
            self.quote_cache.unregister(features.token_mint)
        if self.live_sell_cache is not None:
            self.live_sell_cache.unregister(features.token_mint)
        try:
            self.risk_manager.set_cooldown(
                features.token_mint, minutes=1440, reason="stuck_rug_force_close"
            )
        except Exception:  # noqa: BLE001
            logger.exception("stuck_rug cooldown set failed for %s", features.token_mint[:12])

        logger.error(
            "💀 LIVE position FORCE-CLOSED as stuck rug: %s | slot freed | realized loss=%.4f SOL (last_err=%s)",
            features.token_mint[:12],
            realized_leg_pnl_sol,
            (last_error or "n/a")[:80],
        )
        self.event_log.log(
            "live_exit_stuck_rug_force_close",
            {
                "token_mint": features.token_mint,
                "position_id": int(position["id"]),
                "strategy_id": strategy_id,
                "selected_rule_id": position.get("selected_rule_id"),
                "reason": reason,
                "remaining_size_sol": remaining_size_sol,
                "realized_leg_pnl_sol": realized_leg_pnl_sol,
                "realized_total_pnl_sol": realized_total,
                "last_error": str(last_error or ""),
                "closed_at": now_iso,
            },
        )

    def force_close_drifted_position(
        self,
        position: dict[str, Any],
        *,
        reason: str,
        last_error: str = "",
    ) -> bool:
        """Close a position whose on-chain balance has fully drifted to zero.

        Called by LiveReconciler when it observes `db_amount > 0 AND
        on_chain_amount == 0` (drift ratio ≈ 1.0). Two legitimate triggers:
        (1) user manually sold on Jupiter / phantom outside the bot (confirmed
        on position #35, session 20260419T103007Z — token BUTTERIN closed
        externally at +27.25%), (2) rug-freeze or migration silently removed
        tokens. In either case there are no tokens left to sell; the sell
        loop will burn fees on preflight failures forever. Here we synthesise
        a minimal features snapshot from the DB row and invoke the existing
        stuck-rug closeout path to free the slot.
        """
        token_mint = str(position.get("token_mint") or "")
        if not token_mint:
            return False
        try:
            entry_price = float(position.get("entry_price_sol", 0.0) or 0.0)
        except (TypeError, ValueError):
            entry_price = 0.0
        strategy_id = str(position.get("strategy_id") or "unknown")
        now = datetime.now(tz=timezone.utc)
        features = RuntimeFeatures(
            token_mint=token_mint,
            entry_time=now,
            entry_price_sol=entry_price,
            token_age_sec=None,
            wallet_cluster_30s=0,
            wallet_cluster_120s=0,
            volume_sol_30s=0.0,
            volume_sol_60s=0.0,
            tx_count_30s=0,
            tx_count_60s=0,
            price_change_30s=0.0,
            price_change_60s=0.0,
            triggering_wallet=str(position.get("triggering_wallet") or ""),
            triggering_wallet_score=0.0,
            aggregated_wallet_score=0.0,
            tracked_wallet_present_60s=False,
            tracked_wallet_count_60s=0,
            tracked_wallet_score_sum_60s=0.0,
            raw={"__reconciler_drift_force_close": True},
        )
        metadata_raw = position.get("metadata") or {}
        if isinstance(metadata_raw, str):
            try:
                import json as _json

                metadata = _json.loads(metadata_raw) or {}
            except Exception:  # noqa: BLE001
                metadata = {}
        elif isinstance(metadata_raw, dict):
            metadata = dict(metadata_raw)
        else:
            metadata = {}
        try:
            self._force_close_stuck_rug_position(
                position=position,
                metadata=metadata,
                features=features,
                strategy_id=strategy_id,
                reason=reason,
                last_error=last_error,
            )
            return True
        except Exception:  # noqa: BLE001
            logger.exception("force_close_drifted_position failed for %s", token_mint[:12])
            return False

    def _parse_time(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _exit_profile(
        self, profile_name: str | None, rule_id: str | None = None
    ) -> dict[str, float]:
        base_stop = BASE_STOPS.get(
            str(profile_name or "default_recovery"), BASE_STOPS["default_recovery"]
        )
        if rule_id:
            override = self.exit_rule_stop_overrides.get(str(rule_id))
            if override is not None:
                base_stop = float(override)
        if str(profile_name or "").strip() == "default_sniper":
            return {
                "tp1": self.sniper_take_profit_pnl,
                "tp2": 999.0,
                "tp3": 999.0,
                "tp1_fraction": 1.0,
                "tp2_fraction": 0.0,
                "stop": self.sniper_stop_pnl
                if rule_id not in self.exit_rule_stop_overrides
                else base_stop,
            }
        if str(profile_name or "").strip() == "default_wallet":
            return {
                "tp1": self.wallet_take_profit_pnl,
                "tp2": 999.0,
                "tp3": 999.0,
                "tp1_fraction": 1.0,
                "tp2_fraction": 0.0,
                "stop": self.wallet_stop_pnl
                if rule_id not in self.exit_rule_stop_overrides
                else base_stop,
            }
        if str(profile_name or "").strip() == "mature_pair_v1":
            # Smaller moves, staged partial exits, tighter SL. Inherits the
            # standard TP1/TP2/TP3 flow — no dedicated sniper-style branch.
            return {
                "tp1": 0.15,
                "tp2": 0.30,
                "tp3": 0.60,
                "tp1_fraction": 0.5,
                "tp2_fraction": 0.3,
                "stop": base_stop,
            }
        return {
            "tp1": self.exit_tp1_multiple,
            "tp2": self.exit_tp2_multiple,
            "tp3": self.exit_tp3_multiple,
            "tp1_fraction": self.exit_tp1_sell_fraction,
            "tp2_fraction": self.exit_tp2_sell_fraction,
            "stop": base_stop,
        }

    def _event_signature(self, features: RuntimeFeatures) -> str:
        """Extract the source event signature when available."""
        raw = features.raw or {}
        signature = raw.get("__event_signature")
        if isinstance(signature, str):
            return signature
        return ""

    def _resolve_mark_price(
        self,
        features: RuntimeFeatures,
        metadata: dict[str, Any],
        entry_price: float,
    ) -> tuple[float, bool, float | None, float | None]:
        """Return a smoothed mark price and whether a step guard was applied."""
        raw = features.raw or {}
        raw_price = (
            float(features.entry_price_sol or 0.0) if features.entry_price_sol is not None else None
        )

        reliable_snapshot = raw.get("last_price_sol_reliable")
        if reliable_snapshot is not None:
            try:
                reliable_snapshot = float(reliable_snapshot)
            except (TypeError, ValueError):
                reliable_snapshot = None
        if reliable_snapshot is not None and reliable_snapshot <= 0:
            reliable_snapshot = None

        last_reliable_price = metadata.get("last_reliable_price_sol")
        if last_reliable_price is None:
            last_reliable_price = metadata.get("last_price_sol_seen")
        try:
            last_reliable_price = float(last_reliable_price or 0.0)
        except (TypeError, ValueError):
            last_reliable_price = 0.0

        mark_price = reliable_snapshot
        if mark_price is None and raw_price is not None and raw_price > 0:
            mark_price = float(raw_price)
        if mark_price is None and last_reliable_price > EPS:
            mark_price = last_reliable_price
        if mark_price is None or mark_price <= EPS:
            mark_price = max(entry_price, EPS)

        step_guard_applied = False
        if last_reliable_price > EPS:
            upper_bound = last_reliable_price * self.exit_price_max_step_multiple
            lower_bound = last_reliable_price / self.exit_price_max_step_multiple
            if mark_price > upper_bound:
                mark_price = upper_bound
                step_guard_applied = True
            elif mark_price < lower_bound:
                mark_price = lower_bound
                step_guard_applied = True

        return float(mark_price), step_guard_applied, raw_price, reliable_snapshot

    def _advance_confirm_counter(
        self,
        metadata: dict[str, Any],
        counter_key: str,
        signature_key: str,
        signature: str,
    ) -> int:
        """Increment a confirmation counter once per unique signature."""
        count = int(metadata.get(counter_key, 0) or 0)
        if signature:
            previous_signature = str(metadata.get(signature_key) or "")
            if previous_signature == signature:
                return count
            metadata[signature_key] = signature
        count += 1
        metadata[counter_key] = count
        return count

    def _compute_paper_sell_realization(
        self,
        *,
        token_mint: str,
        metadata: dict[str, Any],
        current_token_amount: float,
        sell_token_amount: float,
        fallback_sell_size_sol: float,
        fallback_pnl_multiple: float,
    ) -> dict[str, Any]:
        """Compute paper SELL realization.

        Quote priority mirrors the TP-verify layer order to prevent TOCTOU divergence
        (verify passes on cached quote, realization gets a fresh different quote):
          1. Local AMM engine — 0 µs, exact constant-product, no staleness window
          2. Position quote cache — pre-fetched Jupiter, 0 ms on hit
          3. Fresh Jupiter call — last resort fallback
        """
        fallback_realized_leg = fallback_sell_size_sol * fallback_pnl_multiple
        result: dict[str, Any] = {
            "used_quote": False,
            "realized_leg_pnl_sol": fallback_realized_leg,
            "sell_token_amount": sell_token_amount,
            "sell_net_sol": max(0.0, fallback_sell_size_sol + fallback_realized_leg),
            "sell_fee_sol": 0.0,
            "cost_basis_allocated_sol": fallback_sell_size_sol,
            "quote_error": None,
        }
        if self.trade_executor is None:
            return result

        remaining_token_raw = float(
            metadata.get("paper_remaining_token_raw", current_token_amount) or current_token_amount
        )
        remaining_cost_basis_sol = float(
            metadata.get("paper_remaining_cost_basis_sol", fallback_sell_size_sol)
            or fallback_sell_size_sol
        )
        token_to_sell = min(max(float(sell_token_amount), 0.0), max(remaining_token_raw, 0.0))
        token_amount_int = int(round(token_to_sell))
        if token_amount_int <= 0:
            return result

        # ── Layer 1: local AMM quote (0 µs, same source as TP verify) ──────────
        # Sanity-check the AMM output against the mark-price fallback.  If the
        # AMM reserves haven't caught up with a fast price move (Yellowstone pool
        # account updates can lag behind trade events), the AMM quote will diverge
        # dramatically from the actual executable price and must be rejected.
        # Threshold: accept AMM if the implied return ratio is within 50 % of the
        # mark-price ratio — e.g. mark shows −15 %, AMM may return −25 %..+5 %.
        # If AMM shows +173 % while mark shows −15 %, that is a stale-reserve bug.
        _AMM_STALENESS_TOLERANCE = 0.50  # 50 % relative deviation allowed
        estimate: Optional[PaperTradeEstimate] = None
        if self.local_quote_engine is not None and self.local_quote_engine.has_reserves(token_mint):
            raw_out = self.local_quote_engine.quote_sell(token_mint, token_amount_int)
            if raw_out and raw_out > 0:
                one_way_fee_sol = float(metadata.get("paper_entry_fee_sol", 0.0) or 0.0)
                if one_way_fee_sol <= 0.0:
                    one_way_fee_sol = float(
                        self.trade_executor.config.priority_fee_lamports
                        + getattr(self.trade_executor.config, "jito_tip_lamports", 0)
                        + 5_000
                    ) / float(LAMPORTS_PER_SOL)
                fee_lam = int(one_way_fee_sol * LAMPORTS_PER_SOL)

                # Validate AMM quote against mark-price ratio before accepting.
                _amm_net_sol = max(0.0, float(raw_out) / float(LAMPORTS_PER_SOL) - one_way_fee_sol)
                _cost_basis = (
                    float(
                        metadata.get("paper_remaining_cost_basis_sol")
                        or fallback_sell_size_sol
                        or 0.0
                    )
                    or fallback_sell_size_sol
                )
                _sell_ratio = (
                    min(1.0, float(token_amount_int) / remaining_token_raw)
                    if remaining_token_raw > EPS
                    else 1.0
                )
                _allocated_basis = _cost_basis * _sell_ratio
                if _allocated_basis > EPS:
                    _amm_return_ratio = _amm_net_sol / _allocated_basis  # e.g. 2.73 at +173%
                    _mark_return_ratio = 1.0 + fallback_pnl_multiple  # e.g. 0.845 at −15%
                    _deviation = abs(_amm_return_ratio - _mark_return_ratio)
                    _tolerance = max(
                        _AMM_STALENESS_TOLERANCE,
                        abs(_mark_return_ratio) * _AMM_STALENESS_TOLERANCE,
                    )
                    if _deviation > _tolerance:
                        logger.debug(
                            "local_amm stale-reserve rejected for %s: "
                            "amm_ratio=%.3f mark_ratio=%.3f deviation=%.3f > tolerance=%.3f",
                            token_mint[:12],
                            _amm_return_ratio,
                            _mark_return_ratio,
                            _deviation,
                            _tolerance,
                        )
                        metadata["paper_sell_quote_source_amm_rejected"] = (
                            f"stale amm_ratio={_amm_return_ratio:.3f} mark_ratio={_mark_return_ratio:.3f}"
                        )
                        raw_out = None  # force fall-through to Layer 2/3

                if raw_out and raw_out > 0:
                    estimate = PaperTradeEstimate(
                        success=True,
                        input_mint=token_mint,
                        output_mint=SOL_MINT,
                        in_amount=token_amount_int,
                        out_amount=raw_out,
                        slippage_bps=int(self.trade_executor.config.default_slippage_bps),
                        priority_fee_lamports=int(self.trade_executor.config.priority_fee_lamports),
                        base_fee_lamports=fee_lam,
                        total_network_fee_lamports=fee_lam,
                        price_impact_pct=0.0,
                    )
                    metadata["paper_sell_quote_source"] = "local_amm"

        # ── Layer 2: pre-cached Jupiter quote (0 ms on hit) ─────────────────────
        # Only use the cache if it was quoted for the same token amount we're selling.
        # After a partial exit the cache still holds the original full-position size;
        # using it for a smaller sell would massively overstate proceeds.
        if estimate is None and self.quote_cache is not None:
            cached = self.quote_cache.get(token_mint)
            if cached is not None and cached.success and cached.out_amount > 0:
                _cached_in = int(getattr(cached, "in_amount", 0) or 0)
                _amount_ok = (
                    _cached_in <= 0  # unknown in_amount – accept with caution
                    or abs(_cached_in - token_amount_int) <= max(1, int(token_amount_int * 0.02))
                )
                if _amount_ok:
                    estimate = cached
                    metadata["paper_sell_quote_source"] = "quote_cache"
                else:
                    metadata["paper_sell_quote_source_cache_skip"] = (
                        f"amount_mismatch cached={_cached_in} selling={token_amount_int}"
                    )

        # ── Layer 3: fresh Jupiter call (fallback only) ──────────────────────────
        if estimate is None:
            estimate = self.trade_executor.simulate_paper_sell(
                token_mint=token_mint,
                token_amount=token_amount_int,
            )
            if not estimate.success or estimate.out_amount <= 0:
                # Jupiter call attempted but failed — PnL falls back to mark price.
                # Label reflects actual source so the accounting is traceable.
                metadata["paper_sell_quote_source"] = "mark_price_fallback"
                result["quote_error"] = estimate.error
                return result
            metadata["paper_sell_quote_source"] = "jupiter_fresh"

        if not estimate.success or estimate.out_amount <= 0:
            result["quote_error"] = estimate.error
            return result

        gross_out_sol = float(estimate.out_amount) / float(LAMPORTS_PER_SOL)
        exit_fee_sol = float(estimate.total_network_fee_lamports) / float(LAMPORTS_PER_SOL)
        net_out_sol = max(0.0, gross_out_sol - exit_fee_sol)

        sell_ratio = (
            min(1.0, max(0.0, float(token_amount_int) / remaining_token_raw))
            if remaining_token_raw > EPS
            else 1.0
        )
        cost_basis_allocated_sol = remaining_cost_basis_sol * sell_ratio
        realized_leg_pnl_sol = net_out_sol - cost_basis_allocated_sol

        new_remaining_raw = max(0.0, remaining_token_raw - float(token_amount_int))
        metadata["paper_remaining_token_raw"] = new_remaining_raw
        metadata["paper_remaining_cost_basis_sol"] = max(
            0.0, remaining_cost_basis_sol - cost_basis_allocated_sol
        )
        metadata["paper_cumulative_fees_sol"] = (
            float(metadata.get("paper_cumulative_fees_sol", 0.0) or 0.0) + exit_fee_sol
        )

        # After a partial sell, update the quote cache to the remaining token amount
        # so that the next partial exit can use Layer-2 with the correct size.
        if self.quote_cache is not None and new_remaining_raw > 0:
            self.quote_cache.update_token_amount(token_mint, int(round(new_remaining_raw)))
        metadata["paper_last_sell_quote_in_raw"] = int(estimate.in_amount)
        metadata["paper_last_sell_quote_out_lamports"] = int(estimate.out_amount)
        metadata["paper_last_sell_fee_sol"] = exit_fee_sol
        metadata["paper_last_sell_priority_fee_lamports"] = int(estimate.priority_fee_lamports)
        metadata["paper_last_sell_jito_tip_lamports"] = int(estimate.jito_tip_lamports)

        result.update(
            {
                "used_quote": True,
                "realized_leg_pnl_sol": realized_leg_pnl_sol,
                "sell_token_amount": float(token_amount_int),
                "sell_net_sol": net_out_sol,
                "sell_fee_sol": exit_fee_sol,
                "cost_basis_allocated_sol": cost_basis_allocated_sol,
                "quote_error": None,
            }
        )
        return result

    def process(self, features: RuntimeFeatures) -> None:
        """Evaluate exits for one token update."""
        mode = self._mode_label
        mode_db = "live" if self._is_live else "paper"
        exit_pipeline_trace = _merge_latency_trace(
            (features.raw or {}).get("__latency_trace"),
            (features.raw or {}).get("__exit_latency_trace"),
        )
        process_started_at = datetime.now(tz=timezone.utc)
        if exit_pipeline_trace:
            exit_pipeline_trace["exit_engine_started_at"] = process_started_at.isoformat()
        process_started = time.monotonic()

        for position in self.position_manager.list_open_positions_for_token(features.token_mint):
            if float(position["entry_price_sol"]) <= 0:
                continue

            entry_price = float(position["entry_price_sol"])
            metadata = json.loads(position.get("metadata_json") or "{}")
            strategy_id = str(metadata.get("strategy_id") or position.get("strategy_id") or "main")
            mark_price_sol, step_guard_applied, raw_price_sol, reliable_price_sol = (
                self._resolve_mark_price(
                    features=features,
                    metadata=metadata,
                    entry_price=entry_price,
                )
            )
            if mark_price_sol <= EPS:
                continue

            current_size_sol = float(position.get("size_sol", 0.0) or 0.0)
            current_token_amount = float(position.get("amount_received", 0.0) or 0.0)
            if current_size_sol <= EPS or current_token_amount <= EPS:
                continue

            pnl_multiple = (mark_price_sol / entry_price) - 1.0
            exit_stage = int(position.get("exit_stage", 0) or 0)
            profile = self._exit_profile(
                metadata.get("exit_profile"),
                str(position.get("selected_rule_id") or ""),
            )
            is_synthetic_sweep = bool((features.raw or {}).get("__synthetic_sweep", False))
            event_signature = self._event_signature(features)

            # Initialize invariant position baselines once, then persist in metadata.
            initial_size_sol = float(
                metadata.get("initial_size_sol", current_size_sol) or current_size_sol
            )
            initial_token_amount = float(
                metadata.get("initial_amount_received", current_token_amount)
                or current_token_amount
            )
            metadata["initial_size_sol"] = initial_size_sol
            metadata["initial_amount_received"] = initial_token_amount

            # Keep useful state for risk/perf diagnostics.
            previous_pnl_multiple = float(metadata.get("last_pnl_multiple", pnl_multiple))
            pnl_jump = pnl_multiple - previous_pnl_multiple
            low_volume_tick = float(features.volume_sol_30s) <= self.exit_outlier_low_volume_sol_30s
            outlier_jump_guard = step_guard_applied or (
                low_volume_tick
                and (
                    pnl_jump >= self.exit_outlier_max_pnl_jump
                    or pnl_multiple >= self.exit_outlier_max_pnl_jump
                )
            )
            peak_ceiling = float(self.exit_max_peak_pnl_multiple)
            pnl_above_ceiling = peak_ceiling > 0 and pnl_multiple > peak_ceiling
            if not outlier_jump_guard and not pnl_above_ceiling:
                metadata["max_pnl_multiple_seen"] = max(
                    float(metadata.get("max_pnl_multiple_seen", pnl_multiple)),
                    pnl_multiple,
                )
                metadata["last_reliable_pnl_multiple"] = float(pnl_multiple)
                metadata["last_reliable_price_sol"] = float(mark_price_sol)
            else:
                metadata["outlier_jump_guard_hits"] = (
                    int(metadata.get("outlier_jump_guard_hits", 0) or 0) + 1
                )
                if pnl_above_ceiling:
                    metadata["peak_ceiling_hits"] = (
                        int(metadata.get("peak_ceiling_hits", 0) or 0) + 1
                    )
            metadata["last_pnl_multiple"] = float(pnl_multiple)
            metadata["last_price_sol_seen"] = float(mark_price_sol)
            metadata["last_price_sol_raw_seen"] = float(raw_price_sol or 0.0)
            metadata["last_price_sol_reliable_seen"] = float(reliable_price_sol or mark_price_sol)
            metadata["last_token_update_at"] = features.entry_time.isoformat()
            if pnl_multiple >= 1.0 and not outlier_jump_guard:
                metadata["hit_2x_achieved"] = True
            if pnl_multiple >= 4.0 and not outlier_jump_guard:
                metadata["hit_5x_achieved"] = True

            entry_time = self._parse_time(position.get("entry_time"))
            hold_seconds = None
            if entry_time is not None:
                hold_seconds = max(0.0, (features.entry_time - entry_time).total_seconds())

            # Record tick sample for exit ML training (buffered; labeled on close)
            if self.exit_predictor is not None:
                try:
                    self.exit_predictor.record_tick_sample(
                        position_id=int(position["id"]),
                        features=features,
                        hold_time_sec=float(hold_seconds or 0.0),
                        current_pnl_multiple=float(pnl_multiple),
                        max_pnl_multiple_seen=float(
                            metadata.get("max_pnl_multiple_seen", pnl_multiple)
                        ),
                        exit_stage=exit_stage,
                        strategy_id=strategy_id,
                        entry_snapshot=metadata.get("runtime_features"),
                    )
                except Exception as _exc:
                    logger.debug("exit_predictor.record_tick_sample failed: %s", _exc)

            crash_guard_candidate = (
                exit_stage == 0
                and hold_seconds is not None
                and self.stage0_crash_guard_window_sec > 0
                and self.stage0_crash_guard_min_hold_sec
                <= hold_seconds
                <= self.stage0_crash_guard_window_sec
                and pnl_multiple <= self.stage0_crash_guard_min_pnl
                and not outlier_jump_guard
                and not is_synthetic_sweep
            )
            if not crash_guard_candidate:
                metadata["stage0_crash_confirm_count"] = 0
                metadata.pop("stage0_crash_last_signature", None)

            dead_token_candidate = (
                hold_seconds is not None
                and hold_seconds >= self.dead_token_hold_sec
                and abs(pnl_multiple) <= self.dead_token_move_pct
                and int(features.tx_count_60s) <= self.dead_token_max_tx_count_60s
                and float(features.volume_sol_60s) <= self.dead_token_max_volume_sol_60s
                and not outlier_jump_guard
                and not is_synthetic_sweep
            )
            if not dead_token_candidate:
                metadata["dead_token_confirm_count"] = 0
                metadata.pop("dead_token_last_signature", None)

            pre_tp1_peak_pnl = float(
                metadata.get("max_pnl_multiple_seen", pnl_multiple) or pnl_multiple
            )
            pre_tp1_retrace_lock_armed = (
                exit_stage == 0
                and hold_seconds is not None
                and hold_seconds >= self.pre_tp1_retrace_lock_min_hold_sec
                and pre_tp1_peak_pnl >= self.pre_tp1_retrace_lock_arm_pnl
                and pnl_multiple < float(profile["tp1"])
            )
            pre_tp1_retrace_hit = (
                pnl_multiple <= self.pre_tp1_retrace_lock_floor_pnl
                or pnl_multiple <= pre_tp1_peak_pnl - self.pre_tp1_retrace_lock_drawdown
            )
            pre_tp1_retrace_lock_candidate = (
                pre_tp1_retrace_lock_armed
                and pre_tp1_retrace_hit
                and not outlier_jump_guard
                and not is_synthetic_sweep
            )
            if not pre_tp1_retrace_lock_candidate:
                metadata["pre_tp1_retrace_confirm_count"] = 0
                metadata.pop("pre_tp1_retrace_last_signature", None)

            dynamic_stop = float(profile["stop"])
            if exit_stage >= 1:
                # Protect remaining size after first take-profit.
                dynamic_stop = max(dynamic_stop, self.post_tp1_stop_pnl)

            next_stage = exit_stage
            reason: str | None = None
            stop_out = False
            close_all = False
            sell_size_sol = 0.0
            sell_token_amount = 0.0

            # Minimum hold time before exit ML is evaluated.
            # The model was trained with bootstrap samples at hold_time=0/pnl=0, so it
            # incorrectly associates "just entered" feature state with losing positions.
            # Apply per-strategy minimum: sniper gets a shorter window than main.
            _min_hold_exit_ml = float(
                self.ml_exit_min_hold_sec_sniper
                if strategy_id == "sniper"
                else self.ml_exit_min_hold_sec
            )
            _skip_exit_ml = hold_seconds is not None and hold_seconds < _min_hold_exit_ml

            # Evaluate ML exit predictor once per tick — result reused for peak lock,
            # veto, and early-exit gate below. Must run before rule-based exits set reason.
            _exit_ml = None
            if self.exit_predictor is not None and not _skip_exit_ml:
                try:
                    _exit_ml = self.exit_predictor.evaluate_position(
                        position=position,
                        features=features,
                        mark_price_sol=mark_price_sol,
                        strategy_id=strategy_id,
                    )
                    self.event_log.log(
                        "exit_ml_decision",
                        {
                            "token_mint": features.token_mint,
                            "position_id": int(position["id"]),
                            "strategy_id": strategy_id,
                            "hold_probability": round(_exit_ml.hold_probability, 4),
                            "exit_now": _exit_ml.exit_now,
                            "mode": _exit_ml.mode,
                            "model_ready": _exit_ml.model_ready,
                            "reason": _exit_ml.reason,
                            "hold_time_sec": float(hold_seconds or 0.0),
                            "current_pnl_multiple": round(pnl_multiple, 4),
                            "max_pnl_multiple_seen": round(
                                float(metadata.get("max_pnl_multiple_seen", pnl_multiple)),
                                4,
                            ),
                        },
                    )
                    metadata["exit_ml_hold_probability"] = round(_exit_ml.hold_probability, 4)
                except Exception as _exc:
                    logger.debug("exit_predictor.evaluate_position failed: %s", _exc)

            is_copy_position = strategy_id == "wallet" and bool(metadata.get("wallet_copy", False))

            if is_copy_position:
                _copy_stage = int(metadata.get("wallet_stage", 0))
                _copy_armed = bool(metadata.get("wallet_trailing_armed", False))
                _copy_arm_threshold = self.wallet_copy_trail_arm_pnl

                # Stage 0: TP1 partial sell (banks guaranteed profit before any trail).
                if (
                    _copy_stage == 0
                    and reason is None
                    and self.wallet_copy_tp1_sell_fraction > 0.0
                    and pnl_multiple >= self.wallet_copy_tp1_peak
                    and not step_guard_applied
                ):
                    _copy_tp1_count = self._advance_confirm_counter(
                        metadata=metadata,
                        counter_key="wallet_copy_tp1_confirm_count",
                        signature_key="wallet_copy_tp1_last_signature",
                        signature=event_signature,
                    )
                    if _copy_tp1_count >= self.wallet_copy_tp1_confirm_ticks:
                        next_stage = 1
                        reason = "wallet_copy_tp1_partial"
                        _copy_frac = float(self.wallet_copy_tp1_sell_fraction)
                        sell_size_sol = min(current_size_sol, initial_size_sol * _copy_frac)
                        sell_token_amount = min(
                            current_token_amount, initial_token_amount * _copy_frac
                        )
                        metadata["wallet_stage"] = 1
                        metadata["wallet_tp1_partial_at"] = datetime.now(
                            tz=timezone.utc
                        ).isoformat()
                        metadata["wallet_tp1_partial_pnl"] = round(float(pnl_multiple), 4)
                        metadata["wallet_copy_tp1_confirm_count"] = 0
                        metadata.pop("wallet_copy_tp1_last_signature", None)
                        self.event_log.log(
                            "wallet_copy_tp1_partial",
                            {
                                "token_mint": features.token_mint,
                                "position_id": int(position["id"]),
                                "pnl_multiple": round(pnl_multiple, 4),
                                "tp1_peak": round(self.wallet_copy_tp1_peak, 4),
                                "sell_fraction": round(_copy_frac, 4),
                                "sell_size_sol": round(float(sell_size_sol), 6),
                                "confirm_ticks": int(self.wallet_copy_tp1_confirm_ticks),
                            },
                        )
                elif _copy_stage == 0:
                    metadata["wallet_copy_tp1_confirm_count"] = 0
                    metadata.pop("wallet_copy_tp1_last_signature", None)
                # Spike-guard: only update peak and arm on confirmed ticks, so a
                # 1-tick mark-price outlier can't arm a trailing stop whose floor
                # the post-spike retrace immediately hits.
                _copy_arm_condition = pnl_multiple >= _copy_arm_threshold and not step_guard_applied
                _copy_confirmed = True
                if not _copy_armed:
                    if _copy_arm_condition:
                        _copy_confirm_count = self._advance_confirm_counter(
                            metadata=metadata,
                            counter_key="wallet_copy_arm_confirm_count",
                            signature_key="wallet_copy_arm_last_signature",
                            signature=event_signature,
                        )
                        _copy_confirmed = (
                            _copy_confirm_count >= self.wallet_trailing_arm_confirm_ticks
                        )
                    else:
                        metadata["wallet_copy_arm_confirm_count"] = 0
                        metadata.pop("wallet_copy_arm_last_signature", None)
                        _copy_confirmed = False

                if _copy_armed or _copy_confirmed:
                    _copy_peak = max(
                        float(metadata.get("wallet_peak_pnl_multiple", pnl_multiple)),
                        pnl_multiple,
                    )
                    metadata["wallet_peak_pnl_multiple"] = _copy_peak
                else:
                    _copy_peak = float(metadata.get("wallet_peak_pnl_multiple", 0.0))

                if not _copy_armed and _copy_confirmed:
                    _copy_armed = True
                    metadata["wallet_trailing_armed"] = True
                    metadata["wallet_trailing_armed_at"] = datetime.now(tz=timezone.utc).isoformat()
                    metadata["wallet_copy_arm_confirm_count"] = 0
                    metadata.pop("wallet_copy_arm_last_signature", None)
                    self.event_log.log(
                        "wallet_copy_trailing_armed",
                        {
                            "token_mint": features.token_mint,
                            "position_id": int(position["id"]),
                            "pnl_multiple": round(pnl_multiple, 4),
                            "peak": round(_copy_peak, 4),
                            "arm_threshold": round(_copy_arm_threshold, 4),
                            "trailing_drawdown": round(self.wallet_copy_trail_drawdown, 4),
                            "confirm_ticks": int(self.wallet_trailing_arm_confirm_ticks),
                        },
                    )
                if _copy_armed and reason is None:
                    _copy_floor = max(
                        self.wallet_copy_trail_min_floor,
                        _copy_peak - self.wallet_copy_trail_drawdown,
                    )
                    metadata["wallet_trailing_floor"] = float(_copy_floor)
                    # Confirmed exit: only close after N consecutive ticks below
                    # floor, so a single crash-tick can't dump us at worst-price.
                    if pnl_multiple <= _copy_floor and not step_guard_applied:
                        _copy_exit_count = self._advance_confirm_counter(
                            metadata=metadata,
                            counter_key="wallet_copy_exit_confirm_count",
                            signature_key="wallet_copy_exit_last_signature",
                            signature=event_signature,
                        )
                        if _copy_exit_count >= self.wallet_copy_exit_confirm_ticks:
                            next_stage = 201
                            reason = "wallet_copy_trailing_stop"
                            close_all = True
                            metadata["wallet_copy_exit_confirm_count"] = 0
                            metadata.pop("wallet_copy_exit_last_signature", None)
                    else:
                        metadata["wallet_copy_exit_confirm_count"] = 0
                        metadata.pop("wallet_copy_exit_last_signature", None)

                # Mirror-sell: triggering wallet just sold — see runner marker.
                _raw = features.raw if isinstance(features.raw, dict) else {}
                _mirror_seller = _raw.get("copy_mirror_sell_wallet")
                _mirror_ids = _raw.get("copy_mirror_sell_position_ids") or []
                _sources = metadata.get("copy_source_wallets") or []
                _mirror_match = (
                    reason is None
                    and self.wallet_copy_mirror_sell
                    and bool(_mirror_seller)
                    and int(position["id"]) in {int(_id) for _id in _mirror_ids if _id}
                    and str(_mirror_seller) in {str(w) for w in _sources}
                )
                if _mirror_match:
                    # Rule:
                    #   pnl <= 0          → mirror-sell immediately
                    #   0 < pnl < profit_threshold → keep trailing, ignore their sell
                    #   pnl >= profit_threshold    → mirror-sell (lock in with them)
                    _mirror_close = False
                    _mirror_tag = None
                    if pnl_multiple <= 0.0:
                        _mirror_close = True
                        _mirror_tag = "wallet_copy_mirror_sell_loss"
                    elif pnl_multiple >= self.wallet_copy_mirror_sell_profit_threshold:
                        _mirror_close = True
                        _mirror_tag = "wallet_copy_mirror_sell_profit"
                    if _mirror_close:
                        next_stage = 202
                        reason = _mirror_tag
                        close_all = True
                        metadata["wallet_copy_mirror_sell_wallet"] = str(_mirror_seller)
                        self.event_log.log(
                            "wallet_copy_mirror_sell_triggered",
                            {
                                "token_mint": features.token_mint,
                                "position_id": int(position["id"]),
                                "seller": str(_mirror_seller),
                                "pnl_multiple": round(pnl_multiple, 4),
                                "trigger": _mirror_tag,
                            },
                        )

                # Pre-arm hard stop (only when not yet armed).
                if (
                    reason is None
                    and not _copy_armed
                    and pnl_multiple <= self.wallet_copy_hard_stop_pnl
                ):
                    next_stage = 299
                    reason = "wallet_copy_hard_stop"
                    close_all = True

                # Max-hold fallback (only when not yet armed).
                if reason is None and not _copy_armed and self.wallet_copy_max_hold_sec > 0:
                    if hold_seconds is not None and hold_seconds >= self.wallet_copy_max_hold_sec:
                        next_stage = 298
                        reason = "wallet_copy_timeout"
                        close_all = True

            if strategy_id == "wallet" and not is_copy_position:
                # Wallet exit flow:
                #  Stage 0 (pre-TP1): partial-sell at WALLET_TP1_PEAK, lock profit
                #  Stage 1 (post-TP1): trail remaining with confirmed arm + confirmed exit
                #  Floor = max(tp1, peak - trailing_drawdown)
                _wallet_stage = int(metadata.get("wallet_stage", 0))
                _wallet_armed = bool(metadata.get("wallet_trailing_armed", False))
                _wallet_tp = float(profile["tp1"])

                # Stage 0: TP1 partial sell (banks guaranteed profit before any trail).
                if (
                    _wallet_stage == 0
                    and reason is None
                    and self.wallet_tp1_sell_fraction > 0.0
                    and pnl_multiple >= self.wallet_tp1_peak
                    and not step_guard_applied
                ):
                    _wallet_tp1_count = self._advance_confirm_counter(
                        metadata=metadata,
                        counter_key="wallet_tp1_confirm_count",
                        signature_key="wallet_tp1_last_signature",
                        signature=event_signature,
                    )
                    if _wallet_tp1_count >= self.wallet_tp1_confirm_ticks:
                        next_stage = 1
                        reason = "wallet_tp1_partial"
                        _frac = float(self.wallet_tp1_sell_fraction)
                        sell_size_sol = min(current_size_sol, initial_size_sol * _frac)
                        sell_token_amount = min(current_token_amount, initial_token_amount * _frac)
                        metadata["wallet_stage"] = 1
                        metadata["wallet_tp1_partial_at"] = datetime.now(
                            tz=timezone.utc
                        ).isoformat()
                        metadata["wallet_tp1_partial_pnl"] = round(float(pnl_multiple), 4)
                        metadata["wallet_tp1_confirm_count"] = 0
                        metadata.pop("wallet_tp1_last_signature", None)
                        self.event_log.log(
                            "wallet_tp1_partial",
                            {
                                "token_mint": features.token_mint,
                                "position_id": int(position["id"]),
                                "pnl_multiple": round(pnl_multiple, 4),
                                "tp1_peak": round(self.wallet_tp1_peak, 4),
                                "sell_fraction": round(_frac, 4),
                                "sell_size_sol": round(float(sell_size_sol), 6),
                                "confirm_ticks": int(self.wallet_tp1_confirm_ticks),
                            },
                        )
                elif _wallet_stage == 0:
                    metadata["wallet_tp1_confirm_count"] = 0
                    metadata.pop("wallet_tp1_last_signature", None)

                # Trailing arm (independent of stage — we arm once we see TP1+ pnl).
                # Spike-guard: only update peak and arm on confirmed ticks, so a
                # 1-tick mark-price outlier can't arm a trailing stop whose floor
                # the post-spike retrace immediately hits.
                _wallet_arm_condition = pnl_multiple >= _wallet_tp and not step_guard_applied
                _wallet_confirmed = True
                if not _wallet_armed:
                    if _wallet_arm_condition:
                        _wallet_confirm_count = self._advance_confirm_counter(
                            metadata=metadata,
                            counter_key="wallet_arm_confirm_count",
                            signature_key="wallet_arm_last_signature",
                            signature=event_signature,
                        )
                        _wallet_confirmed = (
                            _wallet_confirm_count >= self.wallet_trailing_arm_confirm_ticks
                        )
                    else:
                        metadata["wallet_arm_confirm_count"] = 0
                        metadata.pop("wallet_arm_last_signature", None)
                        _wallet_confirmed = False

                if _wallet_armed or _wallet_confirmed:
                    _wallet_peak = max(
                        float(metadata.get("wallet_peak_pnl_multiple", pnl_multiple)),
                        pnl_multiple,
                    )
                    metadata["wallet_peak_pnl_multiple"] = _wallet_peak
                else:
                    _wallet_peak = float(metadata.get("wallet_peak_pnl_multiple", 0.0))

                if not _wallet_armed and _wallet_confirmed:
                    _wallet_armed = True
                    metadata["wallet_trailing_armed"] = True
                    metadata["wallet_trailing_armed_at"] = datetime.now(tz=timezone.utc).isoformat()
                    metadata["wallet_arm_confirm_count"] = 0
                    metadata.pop("wallet_arm_last_signature", None)
                    self.event_log.log(
                        "wallet_trailing_armed",
                        {
                            "token_mint": features.token_mint,
                            "position_id": int(position["id"]),
                            "pnl_multiple": round(pnl_multiple, 4),
                            "peak": round(_wallet_peak, 4),
                            "tp_threshold": round(_wallet_tp, 4),
                            "trailing_drawdown": round(self.wallet_trailing_drawdown, 4),
                            "confirm_ticks": int(self.wallet_trailing_arm_confirm_ticks),
                        },
                    )
                if _wallet_armed and reason is None:
                    _wallet_floor = max(_wallet_tp, _wallet_peak - self.wallet_trailing_drawdown)
                    metadata["wallet_trailing_floor"] = float(_wallet_floor)
                    # Confirmed exit: only close after N consecutive ticks below
                    # floor, so a single retrace-tick into an outlier doesn't
                    # dump us into a dead pool.
                    if pnl_multiple <= _wallet_floor and not step_guard_applied:
                        _wallet_exit_count = self._advance_confirm_counter(
                            metadata=metadata,
                            counter_key="wallet_exit_confirm_count",
                            signature_key="wallet_exit_last_signature",
                            signature=event_signature,
                        )
                        if _wallet_exit_count >= self.wallet_trailing_exit_confirm_ticks:
                            next_stage = 201
                            reason = "wallet_trailing_stop"
                            close_all = True
                            metadata["wallet_exit_confirm_count"] = 0
                            metadata.pop("wallet_exit_last_signature", None)
                    else:
                        metadata["wallet_exit_confirm_count"] = 0
                        metadata.pop("wallet_exit_last_signature", None)

            if strategy_id in ("sniper", "wallet"):
                unrealized_pnl_sol = current_size_sol * pnl_multiple
                one_way_fee_est_sol = float(metadata.get("paper_entry_fee_sol", 0.0) or 0.0)
                if one_way_fee_est_sol <= 0.0:
                    one_way_fee_est_sol = (
                        float(
                            self.trade_executor.config.priority_fee_lamports
                            + getattr(self.trade_executor.config, "jito_tip_lamports", 0)
                            + 5_000
                        )
                        / float(LAMPORTS_PER_SOL)
                        if self.trade_executor is not None
                        else 0.000055
                    )
                roundtrip_fee_est_sol = max(0.0, one_way_fee_est_sol * 2.0)
                min_gross_take_profit_sol = max(
                    self.sniper_tp_min_gross_sol_floor,
                    self.sniper_tp_min_gross_fee_multiplier * roundtrip_fee_est_sol,
                    self.sniper_tp_min_gross_size_ratio * current_size_sol,
                )
                metadata["sniper_tp_min_gross_take_profit_sol"] = float(min_gross_take_profit_sol)
                metadata["sniper_tp_roundtrip_fee_est_sol"] = float(roundtrip_fee_est_sol)
                metadata["sniper_tp_unrealized_pnl_sol"] = float(unrealized_pnl_sol)
                # Per-gate evaluation so we can (a) apply a high-PnL live bypass
                # and (b) emit a diagnostic when a gate silently blocks TP.
                _tp1 = float(profile["tp1"])
                _bypass_mult = max(1.0, float(self.sniper_tp_live_bypass_multiplier))
                _live_bypass_threshold = _tp1 * _bypass_mult
                _live_high_pnl_bypass = bool(
                    self._is_live and pnl_multiple >= _live_bypass_threshold
                )
                _gate_pnl = pnl_multiple >= _tp1
                _gate_outlier = not outlier_jump_guard
                _gate_sweep = not is_synthetic_sweep
                _gate_volume = float(features.volume_sol_30s) >= self.tp1_min_volume_sol_30s
                _gate_gross = unrealized_pnl_sol >= min_gross_take_profit_sol

                strict_candidate = (
                    _gate_pnl and _gate_outlier and _gate_sweep and _gate_volume and _gate_gross
                )
                # In live mode at ≥ bypass_mult × tp1, volume/outlier gates are
                # skipped — a +16% winner on a pool that went quiet is still a
                # winner, and those gates were sized for main-lane noise filters.
                bypass_candidate = (
                    _live_high_pnl_bypass and _gate_pnl and _gate_sweep and _gate_gross
                )
                sniper_tp_candidate = strict_candidate or bypass_candidate
                if strategy_id == "wallet":
                    # Wallet uses trailing exit handled above — never close via sniper TP.
                    sniper_tp_candidate = False

                if strategy_id == "sniper" and _gate_pnl and not sniper_tp_candidate:
                    # Log exactly which gate is holding up a profitable position
                    # so we can tune thresholds next session instead of guessing.
                    self.event_log.log(
                        "sniper_tp_skipped",
                        {
                            "token_mint": features.token_mint,
                            "position_id": int(position["id"]),
                            "pnl_multiple": round(pnl_multiple, 4),
                            "tp1": round(_tp1, 4),
                            "live": bool(self._is_live),
                            "bypass_threshold": round(_live_bypass_threshold, 4),
                            "bypass_eligible": _live_high_pnl_bypass,
                            "volume_sol_30s": round(float(features.volume_sol_30s), 4),
                            "volume_floor": round(float(self.tp1_min_volume_sol_30s), 4),
                            "unrealized_pnl_sol": round(float(unrealized_pnl_sol), 6),
                            "min_gross_take_profit_sol": round(float(min_gross_take_profit_sol), 6),
                            "gate_outlier_ok": _gate_outlier,
                            "gate_synthetic_sweep_ok": _gate_sweep,
                            "gate_volume_ok": _gate_volume,
                            "gate_gross_ok": _gate_gross,
                            "outlier_jump_guard": bool(outlier_jump_guard),
                            "step_guard_applied": bool(step_guard_applied),
                            "is_synthetic_sweep": bool(is_synthetic_sweep),
                        },
                    )
                if sniper_tp_candidate:
                    confirm_count = self._advance_confirm_counter(
                        metadata=metadata,
                        counter_key="sniper_tp_confirm_count",
                        signature_key="sniper_tp_last_signature",
                        signature=event_signature,
                    )
                    if confirm_count >= self.sniper_tp_confirm_ticks:
                        # Jupiter quote verification: confirm actual pool price supports TP.
                        # Mark price from individual PUMP_AMM events is very noisy on thin
                        # pools — a small buy can spike mark price without real pool movement.
                        # Jupiter returns the actual executable price; if it can't return at
                        # least the entry cost, the mark price spike is noise — hold instead.
                        _tp_jupiter_verified = True
                        if (
                            not self._is_live
                            and self.trade_executor is not None
                            and self.sniper_tp_jupiter_verify
                        ):
                            # High-PnL bypass: at ≥2.5× TP threshold (e.g. ≥20% when TP=8%),
                            # even large slippage leaves plenty of profit. Skip all Jupiter layers
                            # so rate-limit storms can't trap a clearly profitable position.
                            _HIGH_PNL_BYPASS = float(profile["tp1"]) * 2.5
                            if pnl_multiple >= _HIGH_PNL_BYPASS:
                                logger.debug(
                                    "sniper TP verify bypassed: pnl=%.3f >= bypass=%.3f",
                                    pnl_multiple,
                                    _HIGH_PNL_BYPASS,
                                )
                            else:
                                try:
                                    _remaining_tokens = float(
                                        metadata.get(
                                            "paper_remaining_token_raw",
                                            current_token_amount,
                                        )
                                        or current_token_amount
                                    )
                                    _token_amount_int = int(round(_remaining_tokens))
                                    if _token_amount_int > 0:
                                        # ── Layer 1: local AMM quote (0 µs, exact constant-product) ────────
                                        # Uses cached pool reserves extracted from the Yellowstone gRPC stream.
                                        # Formula: in_after_fee = tokens × (10000-fee)/10000
                                        #          out_lamports = in_after_fee × sol_reserve / (token_reserve + in_after_fee)
                                        # Falls back to price-estimate heuristic when no reserve cache exists.
                                        _SKIP_THRESHOLD = (
                                            0.85  # fast-fail if local < 85% of entry cost
                                        )
                                        _local_lamports: int | None = None
                                        if self.local_quote_engine is not None:
                                            _local_lamports = self.local_quote_engine.quote_sell(
                                                features.token_mint,
                                                _token_amount_int,
                                            )
                                        if _local_lamports is None:
                                            # Fallback: price-estimate heuristic (less accurate but instant)
                                            _SLIPPAGE_BUFFER = 0.005
                                            _current_price_sol = float(
                                                getattr(features, "entry_price_sol", 0) or 0
                                            )
                                            if _current_price_sol > 0:
                                                _local_lamports = int(
                                                    (_token_amount_int / 1_000_000)
                                                    * _current_price_sol
                                                    * (1.0 - _SLIPPAGE_BUFFER)
                                                    * 1_000_000_000
                                                )
                                        if _local_lamports is not None and _local_lamports > 0:
                                            _local_sol_est = _local_lamports / 1_000_000_000
                                            if _local_sol_est < initial_size_sol * _SKIP_THRESHOLD:
                                                # Clearly unprofitable — skip Jupiter entirely.
                                                _tp_jupiter_verified = False
                                                metadata["sniper_tp_confirm_count"] = 0
                                                metadata.pop("sniper_tp_last_signature", None)
                                                _has_local_amm = (
                                                    self.local_quote_engine is not None
                                                    and self.local_quote_engine.has_reserves(
                                                        features.token_mint
                                                    )
                                                )
                                                self.event_log.log(
                                                    "sniper_tp_local_rejected",
                                                    {
                                                        "token_mint": features.token_mint,
                                                        "position_id": int(position["id"]),
                                                        "pnl_multiple": round(pnl_multiple, 4),
                                                        "local_sol_est": round(_local_sol_est, 6),
                                                        "initial_size_sol": round(
                                                            initial_size_sol, 6
                                                        ),
                                                        "local_amm_quote": _has_local_amm,
                                                    },
                                                )

                                        if _tp_jupiter_verified:
                                            # ── Layer 2: pre-cached Jupiter quote (0 ms on cache hit) ───────
                                            _cache_hit = False
                                            _jup_est = None
                                            if self.quote_cache is not None:
                                                _jup_est = self.quote_cache.get(features.token_mint)
                                                if _jup_est is not None:
                                                    _cache_hit = True
                                            if _jup_est is None:
                                                # ── Layer 3: fresh Jupiter call (~300 ms, rare fallback) ─────
                                                _jup_est = self.trade_executor.simulate_paper_sell(
                                                    token_mint=features.token_mint,
                                                    token_amount=_token_amount_int,
                                                )
                                            if not _jup_est.success or _jup_est.out_amount <= 0:
                                                # Jupiter unavailable (rate-limit, timeout, etc.).
                                                # Block this tick but preserve confirm_count — next tick
                                                # will retry immediately without re-counting confirms.
                                                _tp_jupiter_verified = False
                                                self.event_log.log(
                                                    "sniper_tp_jupiter_unavailable",
                                                    {
                                                        "token_mint": features.token_mint,
                                                        "position_id": int(position["id"]),
                                                        "pnl_multiple": round(pnl_multiple, 4),
                                                        "cache_hit": _cache_hit,
                                                        "error": str(
                                                            getattr(_jup_est, "error", None)
                                                        ),
                                                    },
                                                )
                                            elif _jup_est.out_amount > 0:
                                                _jup_out_sol = float(_jup_est.out_amount) / float(
                                                    LAMPORTS_PER_SOL
                                                )
                                                if _jup_out_sol < initial_size_sol:
                                                    # Jupiter confirms no gain — mark price is noise.
                                                    _tp_jupiter_verified = False
                                                    metadata["sniper_tp_confirm_count"] = 0
                                                    metadata.pop("sniper_tp_last_signature", None)
                                                    self.event_log.log(
                                                        "sniper_tp_jupiter_rejected",
                                                        {
                                                            "token_mint": features.token_mint,
                                                            "position_id": int(position["id"]),
                                                            "pnl_multiple": round(pnl_multiple, 4),
                                                            "jup_out_sol": round(_jup_out_sol, 6),
                                                            "initial_size_sol": round(
                                                                initial_size_sol, 6
                                                            ),
                                                            "price_impact_pct": round(
                                                                float(_jup_est.price_impact_pct),
                                                                4,
                                                            ),
                                                            "cache_hit": _cache_hit,
                                                        },
                                                    )
                                except Exception as _exc:
                                    logger.debug("sniper TP jupiter verify failed: %s", _exc)
                        if _tp_jupiter_verified:
                            next_stage = 201
                            reason = "sniper_take_profit"
                            close_all = True
                            metadata["sniper_tp_confirm_count"] = 0
                            metadata.pop("sniper_tp_last_signature", None)
                else:
                    metadata["sniper_tp_confirm_count"] = 0
                    metadata.pop("sniper_tp_last_signature", None)

                sniper_stop_candidate = (
                    reason is None
                    and not is_copy_position
                    and pnl_multiple <= dynamic_stop
                    and not outlier_jump_guard
                    and (hold_seconds is None or hold_seconds >= self.sniper_stop_min_hold_sec)
                )
                if sniper_stop_candidate:
                    confirm_count = self._advance_confirm_counter(
                        metadata=metadata,
                        counter_key="sniper_stop_confirm_count",
                        signature_key="sniper_stop_last_signature",
                        signature=event_signature,
                    )
                    if confirm_count >= self.sniper_stop_confirm_ticks:
                        next_stage = 299
                        reason = "wallet_stop_out" if strategy_id == "wallet" else "sniper_stop_out"
                        stop_out = True
                        close_all = True
                        metadata["sniper_stop_confirm_count"] = 0
                        metadata.pop("sniper_stop_last_signature", None)
                else:
                    metadata["sniper_stop_confirm_count"] = 0
                    metadata.pop("sniper_stop_last_signature", None)
                lane_max_hold_sec = (
                    self.wallet_max_hold_sec
                    if strategy_id == "wallet"
                    else self.sniper_max_hold_sec
                )
                if (
                    reason is None
                    and not is_copy_position
                    and hold_seconds is not None
                    and lane_max_hold_sec > 0
                    and hold_seconds >= lane_max_hold_sec
                    and not (strategy_id == "wallet" and metadata.get("wallet_trailing_armed"))
                ):
                    next_stage = 298
                    reason = "sniper_timeout" if strategy_id == "sniper" else "wallet_timeout"
                    close_all = True
            # Hard risk exits first.
            elif pnl_multiple <= dynamic_stop:
                next_stage = 99
                reason = "stop_out"
                stop_out = True
                close_all = True
            elif (
                hold_seconds is not None
                and self.absolute_max_hold_sec > 0
                and hold_seconds >= self.absolute_max_hold_sec
            ):
                next_stage = 92
                reason = "absolute_max_hold_timeout"
                close_all = True
            elif crash_guard_candidate:
                crash_confirm_count = self._advance_confirm_counter(
                    metadata=metadata,
                    counter_key="stage0_crash_confirm_count",
                    signature_key="stage0_crash_last_signature",
                    signature=event_signature,
                )
                if crash_confirm_count >= self.stage0_crash_guard_confirm_ticks:
                    next_stage = 87
                    reason = "stage0_crash_guard"
                    close_all = True
                    metadata["stage0_crash_confirm_count"] = 0
                    metadata.pop("stage0_crash_last_signature", None)
            elif pre_tp1_retrace_lock_candidate:
                retrace_confirm_count = self._advance_confirm_counter(
                    metadata=metadata,
                    counter_key="pre_tp1_retrace_confirm_count",
                    signature_key="pre_tp1_retrace_last_signature",
                    signature=event_signature,
                )
                if retrace_confirm_count >= self.pre_tp1_retrace_lock_confirm_ticks:
                    next_stage = 90
                    reason = "pre_tp1_retrace_lock"
                    close_all = True
                    metadata["pre_tp1_retrace_confirm_count"] = 0
                    metadata.pop("pre_tp1_retrace_last_signature", None)
            elif (
                exit_stage == 0
                and hold_seconds is not None
                and self.stage0_fast_fail_non_positive_sec > 0
                and hold_seconds >= self.stage0_fast_fail_non_positive_sec
                and pnl_multiple <= self.stage0_fast_fail_non_positive_max_pnl
            ):
                next_stage = 92
                reason = "stage0_fast_fail_non_positive"
                close_all = True
            elif (
                exit_stage == 0
                and hold_seconds is not None
                and self.stage0_fast_fail_under_profit_sec > 0
                and hold_seconds >= self.stage0_fast_fail_under_profit_sec
                and pnl_multiple < self.stage0_fast_fail_under_profit_min_pnl
            ):
                next_stage = 89
                reason = "stage0_fast_fail_under_profit"
                close_all = True
            elif (
                exit_stage == 0
                and hold_seconds is not None
                and self.stage0_loss_timeout_sec > 0
                and hold_seconds >= self.stage0_loss_timeout_sec
                and float(profile["stop"]) < pnl_multiple <= self.stage0_loss_timeout_max_pnl
            ):
                next_stage = 94
                reason = "stage0_timeout_loss_band"
                close_all = True
            elif (
                exit_stage == 0
                and hold_seconds is not None
                and self.stage0_moderate_positive_timeout_sec > 0
                and hold_seconds >= self.stage0_moderate_positive_timeout_sec
                and self.stage0_moderate_positive_min_pnl
                <= pnl_multiple
                <= self.stage0_moderate_positive_max_pnl
                and str(metadata.get("exit_profile") or "")
                not in self.stage0_moderate_positive_skip_profiles
            ):
                next_stage = 93
                reason = "stage0_timeout_moderate_positive"
                close_all = True
            elif dead_token_candidate:
                dead_token_confirm_count = self._advance_confirm_counter(
                    metadata=metadata,
                    counter_key="dead_token_confirm_count",
                    signature_key="dead_token_last_signature",
                    signature=event_signature,
                )
                if dead_token_confirm_count >= self.dead_token_confirm_ticks:
                    next_stage = 98
                    reason = "dead_token_timeout"
                    close_all = True
                    metadata["dead_token_confirm_count"] = 0
                    metadata.pop("dead_token_last_signature", None)
            elif exit_stage == 0:
                early_profit_candidate = (
                    hold_seconds is not None
                    and 0.0 <= hold_seconds <= self.stage0_early_profit_window_sec
                    and self.stage0_early_profit_min_pnl
                    <= pnl_multiple
                    <= self.stage0_early_profit_max_pnl
                    and not outlier_jump_guard
                    and not is_synthetic_sweep
                    and float(features.volume_sol_30s) >= self.tp1_min_volume_sol_30s
                )
                if early_profit_candidate:
                    early_confirm_count = self._advance_confirm_counter(
                        metadata=metadata,
                        counter_key="stage0_early_profit_confirm_count",
                        signature_key="stage0_early_profit_last_signature",
                        signature=event_signature,
                    )
                    if early_confirm_count >= self.stage0_early_profit_confirm_ticks:
                        # Partial de-risk at early profit, then keep runner path alive.
                        # Promote to stage-1 so remaining size can continue to TP2/TP3 logic.
                        if self.stage0_early_profit_sell_fraction >= 0.999:
                            next_stage = 88
                            reason = "stage0_early_profit_take"
                            close_all = True
                        else:
                            next_stage = 1
                            reason = "stage0_early_profit_partial"
                            sell_size_sol = min(
                                current_size_sol,
                                initial_size_sol * self.stage0_early_profit_sell_fraction,
                            )
                            sell_token_amount = min(
                                current_token_amount,
                                initial_token_amount * self.stage0_early_profit_sell_fraction,
                            )
                            metadata["stage1_started_at"] = (
                                metadata.get("stage1_started_at") or features.entry_time.isoformat()
                            )
                        metadata["stage0_early_profit_confirm_count"] = 0
                        metadata.pop("stage0_early_profit_last_signature", None)
                else:
                    metadata["stage0_early_profit_confirm_count"] = 0
                    metadata.pop("stage0_early_profit_last_signature", None)

                if pnl_multiple < float(profile["tp1"]):
                    metadata["tp1_confirm_count"] = 0
                    metadata.pop("tp1_last_signature", None)
                if reason is None and pnl_multiple < float(profile["tp2"]):
                    metadata["tp2_fast_confirm_count"] = 0
                    metadata.pop("tp2_fast_last_signature", None)
                if reason is None and pnl_multiple >= float(profile["tp2"]):
                    fast_path_eligible = (
                        not outlier_jump_guard
                        and not is_synthetic_sweep
                        and float(features.volume_sol_30s) >= self.tp2_fast_min_volume_sol_30s
                    )
                    if fast_path_eligible:
                        confirm_count = self._advance_confirm_counter(
                            metadata=metadata,
                            counter_key="tp2_fast_confirm_count",
                            signature_key="tp2_fast_last_signature",
                            signature=event_signature,
                        )
                        if confirm_count >= self.tp2_fast_confirm_ticks:
                            # Fast path: when price skips straight above TP2, realize TP1+TP2
                            # in one shot so the runner logic can activate immediately.
                            next_stage = 2
                            reason = "exit_stage_2_fast"
                            fast_fraction = float(profile["tp1_fraction"]) + float(
                                profile["tp2_fraction"]
                            )
                            sell_size_sol = min(current_size_sol, initial_size_sol * fast_fraction)
                            sell_token_amount = min(
                                current_token_amount,
                                initial_token_amount * fast_fraction,
                            )
                            metadata["stage1_started_at"] = (
                                metadata.get("stage1_started_at") or features.entry_time.isoformat()
                            )
                            metadata["stage2_started_at"] = (
                                metadata.get("stage2_started_at") or features.entry_time.isoformat()
                            )
                            metadata["stage2_peak_pnl_multiple"] = max(
                                float(metadata.get("stage2_peak_pnl_multiple", pnl_multiple)),
                                pnl_multiple,
                            )
                            metadata["tp1_confirm_count"] = 0
                            metadata["tp2_confirm_count"] = 0
                            metadata["tp2_fast_confirm_count"] = 0
                    else:
                        metadata["tp2_fast_confirm_count"] = 0
                        metadata.pop("tp2_fast_last_signature", None)

                # Fallback: if TP2-fast isn't confirmed this tick, still allow TP1 partial.
                if (
                    reason is None
                    and pnl_multiple >= float(profile["tp1"])
                    and not outlier_jump_guard
                    and not is_synthetic_sweep
                    and float(features.volume_sol_30s) >= self.tp1_min_volume_sol_30s
                ):
                    tp1_confirm_count = self._advance_confirm_counter(
                        metadata=metadata,
                        counter_key="tp1_confirm_count",
                        signature_key="tp1_last_signature",
                        signature=event_signature,
                    )
                    if tp1_confirm_count >= self.tp1_confirm_ticks:
                        next_stage = 1
                        reason = "exit_stage_1"
                        sell_size_sol = min(
                            current_size_sol,
                            initial_size_sol * float(profile["tp1_fraction"]),
                        )
                        sell_token_amount = min(
                            current_token_amount,
                            initial_token_amount * float(profile["tp1_fraction"]),
                        )
                        metadata["stage1_started_at"] = (
                            metadata.get("stage1_started_at") or features.entry_time.isoformat()
                        )
                        metadata["tp2_fast_confirm_count"] = 0
                        metadata["tp1_confirm_count"] = 0
                        metadata["tp2_confirm_count"] = 0
                else:
                    metadata["tp1_confirm_count"] = 0
                    metadata.pop("tp1_last_signature", None)
            elif exit_stage == 1:
                metadata["stage1_started_at"] = (
                    metadata.get("stage1_started_at") or features.entry_time.isoformat()
                )
                tp2_candidate = (
                    pnl_multiple >= float(profile["tp2"])
                    and not outlier_jump_guard
                    and not is_synthetic_sweep
                    and float(features.volume_sol_30s) >= self.tp2_min_volume_sol_30s
                )
                if tp2_candidate:
                    tp2_confirm_count = self._advance_confirm_counter(
                        metadata=metadata,
                        counter_key="tp2_confirm_count",
                        signature_key="tp2_last_signature",
                        signature=event_signature,
                    )
                    if tp2_confirm_count >= self.tp2_confirm_ticks:
                        next_stage = 2
                        reason = "exit_stage_2"
                        sell_size_sol = min(
                            current_size_sol,
                            initial_size_sol * float(profile["tp2_fraction"]),
                        )
                        sell_token_amount = min(
                            current_token_amount,
                            initial_token_amount * float(profile["tp2_fraction"]),
                        )
                        metadata["stage2_started_at"] = (
                            metadata.get("stage2_started_at") or features.entry_time.isoformat()
                        )
                        metadata["stage2_peak_pnl_multiple"] = max(
                            float(metadata.get("stage2_peak_pnl_multiple", pnl_multiple)),
                            pnl_multiple,
                        )
                        metadata["tp2_confirm_count"] = 0
                else:
                    metadata["tp2_confirm_count"] = 0
                    metadata.pop("tp2_last_signature", None)
                if reason is None:
                    stage1_started_at = self._parse_time(metadata.get("stage1_started_at"))
                    if (
                        stage1_started_at is not None
                        and self.stage1_low_positive_timeout_sec > 0
                        and (features.entry_time - stage1_started_at).total_seconds()
                        >= self.stage1_low_positive_timeout_sec
                        and self.stage1_low_positive_min_pnl
                        <= pnl_multiple
                        <= self.stage1_low_positive_max_pnl
                    ):
                        next_stage = 95
                        reason = "stage1_timeout_low_positive"
                        close_all = True
                    elif (
                        stage1_started_at is not None
                        and self.stage1_sub2x_timeout_sec > 0
                        and (features.entry_time - stage1_started_at).total_seconds()
                        >= self.stage1_sub2x_timeout_sec
                        and self.stage1_sub2x_min_pnl <= pnl_multiple <= self.stage1_sub2x_max_pnl
                    ):
                        next_stage = 91
                        reason = "stage1_timeout_sub2x"
                        close_all = True
            elif exit_stage >= 2:
                # Runner management after stage-2: TP3, trailing, then timeout fallback.
                metadata["stage2_started_at"] = (
                    metadata.get("stage2_started_at") or features.entry_time.isoformat()
                )
                if outlier_jump_guard:
                    peak = float(metadata.get("stage2_peak_pnl_multiple", previous_pnl_multiple))
                else:
                    peak = max(
                        float(metadata.get("stage2_peak_pnl_multiple", pnl_multiple)),
                        pnl_multiple,
                    )
                metadata["stage2_peak_pnl_multiple"] = peak

                tp3_candidate = (
                    pnl_multiple >= float(profile["tp3"])
                    and float(features.volume_sol_30s) >= self.tp3_min_volume_sol_30s
                    and not outlier_jump_guard
                    and not is_synthetic_sweep
                )
                if tp3_candidate:
                    confirm_count = int(metadata.get("tp3_confirm_count", 0) or 0) + 1
                    metadata["tp3_confirm_count"] = confirm_count
                    if confirm_count >= self.tp3_confirm_ticks:
                        next_stage = 3
                        reason = "exit_stage_3"
                        close_all = True
                else:
                    metadata["tp3_confirm_count"] = 0
                    if outlier_jump_guard:
                        pass
                    elif (
                        self.post_tp2_trailing_drawdown > 0
                        and pnl_multiple <= peak - self.post_tp2_trailing_drawdown
                    ):
                        next_stage = 97
                        reason = "runner_trailing_stop"
                        close_all = True
                    else:
                        stage2_started_at = self._parse_time(metadata.get("stage2_started_at"))
                        if (
                            stage2_started_at is not None
                            and self.post_tp2_timeout_sec > 0
                            and (features.entry_time - stage2_started_at).total_seconds()
                            >= self.post_tp2_timeout_sec
                        ):
                            next_stage = 96
                            reason = "runner_timeout"
                            close_all = True

            if reason is not None and outlier_jump_guard and not is_synthetic_sweep:
                suppressed_reason = reason
                metadata["suppressed_exit_reason"] = suppressed_reason
                metadata["suppressed_exit_at"] = features.entry_time.isoformat()
                self.event_log.log(
                    f"{mode.lower()}_exit_guard",
                    {
                        "token_mint": features.token_mint,
                        "selected_rule_id": position["selected_rule_id"],
                        "reason": "outlier_guard_suppressed_exit",
                        "suppressed_reason": suppressed_reason,
                        "step_guard_applied": step_guard_applied,
                        "pnl_multiple": pnl_multiple,
                        "previous_pnl_multiple": previous_pnl_multiple,
                        "pnl_jump": pnl_jump,
                        "mark_price_sol": mark_price_sol,
                        "raw_price_sol": raw_price_sol,
                        "reliable_price_sol": reliable_price_sol,
                        "volume_sol_30s": float(features.volume_sol_30s),
                    },
                )
                self.position_manager.update_metadata(int(position["id"]), metadata)
                continue

            # ML peak lock: exit when position is in profit and PnL starts retracing.
            # Fires before veto/gate so we capture the peak before rule-based exits trigger.
            if reason is None and _exit_ml is not None and _exit_ml.mode == "gate":
                try:
                    _max_pnl = float(metadata.get("max_pnl_multiple_seen", pnl_multiple))
                    _pnl_drawdown = _max_pnl - pnl_multiple
                    if (
                        self.ml_exit_peak_lock_enabled
                        and pnl_multiple >= self.ml_exit_peak_lock_min_pnl
                        and _pnl_drawdown >= self.ml_exit_peak_lock_drawdown
                        and _exit_ml.hold_probability < self.ml_exit_peak_lock_threshold
                        and not outlier_jump_guard
                        and not is_synthetic_sweep
                    ):
                        reason = "ml_peak_lock"
                        next_stage = 92
                        close_all = True
                        sell_size_sol = current_size_sol
                        sell_token_amount = current_token_amount
                        self.event_log.log(
                            "exit_ml_peak_lock",
                            {
                                "token_mint": features.token_mint,
                                "position_id": int(position["id"]),
                                "pnl_multiple": round(pnl_multiple, 4),
                                "max_pnl_seen": round(_max_pnl, 4),
                                "pnl_drawdown": round(_pnl_drawdown, 4),
                                "hold_probability": round(_exit_ml.hold_probability, 4),
                            },
                        )
                except Exception as _exc:
                    logger.debug("exit_predictor peak_lock check failed: %s", _exc)

            # ML veto: block stop_out / timeout when model has high recovery confidence.
            if _exit_ml is not None and _exit_ml.mode == "gate" and _exit_ml.model_ready:
                try:
                    if (
                        reason in self.ml_exit_veto_reasons
                        and _exit_ml.hold_probability > self.ml_exit_veto_threshold
                    ):
                        self.event_log.log(
                            "exit_ml_veto",
                            {
                                "token_mint": features.token_mint,
                                "position_id": int(position["id"]),
                                "vetoed_reason": reason,
                                "hold_probability": round(_exit_ml.hold_probability, 4),
                            },
                        )
                        reason = None
                        stop_out = False
                        close_all = False
                except Exception as _exc:
                    logger.debug("exit_predictor veto check failed: %s", _exc)

            # ML exit gate: take profit when model detects the position is at/past peak.
            # Fires at exit_stage 0 (pre-TP1) and stage 1 (post-TP1, runner phase).
            # Labels "ml_profit_take" when in profit, "ml_exit_early" when losing/flat.
            # Skipped for sniper: model trained on PUMP_FUN data lacks PUMP_AMM patterns.
            if (
                reason is None
                and exit_stage in (0, 1)
                and strategy_id not in ("sniper", "wallet")
                and _exit_ml is not None
            ):
                try:
                    if _exit_ml.exit_now and _exit_ml.mode == "gate":
                        if pnl_multiple > 0.02:
                            reason = "ml_profit_take"
                        else:
                            reason = "ml_exit_early"
                        next_stage = 92
                        close_all = True
                        sell_size_sol = current_size_sol
                        sell_token_amount = current_token_amount
                except Exception as _exc:
                    logger.debug("exit_predictor gate check failed: %s", _exc)

            if reason is None:
                if outlier_jump_guard and not is_synthetic_sweep:
                    self.event_log.log(
                        f"{mode.lower()}_exit_guard",
                        {
                            "token_mint": features.token_mint,
                            "selected_rule_id": position["selected_rule_id"],
                            "reason": "outlier_pnl_jump_guard",
                            "step_guard_applied": step_guard_applied,
                            "pnl_multiple": pnl_multiple,
                            "previous_pnl_multiple": previous_pnl_multiple,
                            "pnl_jump": pnl_jump,
                            "mark_price_sol": mark_price_sol,
                            "raw_price_sol": raw_price_sol,
                            "reliable_price_sol": reliable_price_sol,
                            "volume_sol_30s": float(features.volume_sol_30s),
                            "jump_threshold": self.exit_outlier_max_pnl_jump,
                            "volume_threshold": self.exit_outlier_low_volume_sol_30s,
                        },
                    )
                self.position_manager.update_metadata(int(position["id"]), metadata)
                continue

            # Fix A: surface exit-reason decisions that previously only
            # appeared downstream via live_exit/live_exit_failed. When the
            # sell path short-circuits (circuit breaker, slot busy, empty
            # amount), reason was invisible — this event makes every
            # setting of `reason` explicit so sniper_timeout non-firings
            # are observable regardless of what happens afterwards.
            self.event_log.log(
                "exit_reason_set",
                {
                    "token_mint": features.token_mint,
                    "position_id": int(position["id"]),
                    "strategy_id": strategy_id,
                    "mode": mode,
                    "reason": reason,
                    "next_stage": next_stage,
                    "exit_stage": exit_stage,
                    "close_all": bool(close_all),
                    "stop_out": bool(stop_out),
                    "pnl_multiple": round(float(pnl_multiple), 4),
                    "hold_seconds": (
                        round(float(hold_seconds), 3) if hold_seconds is not None else None
                    ),
                    "mark_price_sol": float(mark_price_sol),
                    "volume_sol_30s": round(float(features.volume_sol_30s), 4),
                    "is_synthetic_sweep": bool(is_synthetic_sweep),
                    "outlier_jump_guard": bool(outlier_jump_guard),
                    "step_guard_applied": bool(step_guard_applied),
                    "sell_size_sol": round(float(sell_size_sol), 6),
                    "sell_token_amount": float(sell_token_amount),
                },
            )

            if close_all:
                sell_size_sol = current_size_sol
                sell_token_amount = current_token_amount

            if sell_size_sol <= EPS or sell_token_amount <= EPS:
                # Fix C: surface previously-silent skip. `reason` is set
                # but the sizing came out empty (e.g. partial-exit arithmetic
                # with current_token_amount dust). Without this, position
                # looks stuck with no trace.
                self.event_log.log(
                    "exit_skipped",
                    {
                        "token_mint": features.token_mint,
                        "position_id": int(position["id"]),
                        "strategy_id": strategy_id,
                        "reason": reason,
                        "cause": "empty_sell_amount",
                        "sell_size_sol": float(sell_size_sol),
                        "sell_token_amount": float(sell_token_amount),
                        "current_size_sol": float(current_size_sol),
                        "current_token_amount": float(current_token_amount),
                    },
                )
                continue

            decision_latency_trace = _merge_latency_trace(exit_pipeline_trace)
            decision_latency_trace["exit_decided_at"] = datetime.now(tz=timezone.utc).isoformat()
            decision_latency_trace["exit_decision_ms"] = (
                time.monotonic() - process_started
            ) * 1000.0
            decision_latency_trace["exit_reason"] = reason
            metadata["exit_pipeline_latency_trace"] = decision_latency_trace

            position_id = int(position["id"])
            if not self._try_acquire_exit_slot(position_id):
                logger.debug(
                    "exit_engine: skip duplicate exit for position %d (%s) – already in-flight",
                    position_id,
                    features.token_mint[:12],
                )
                self.event_log.log(
                    "exit_skipped",
                    {
                        "token_mint": features.token_mint,
                        "position_id": position_id,
                        "strategy_id": strategy_id,
                        "reason": reason,
                        "cause": "exit_slot_busy",
                    },
                )
                continue
            try:
                if self._exit_leg_already_recorded(position_id, reason, close_all):
                    logger.warning(
                        "exit_engine: skip duplicate persisted exit for position %d (%s) reason=%s close=%s",
                        position_id,
                        features.token_mint[:12],
                        reason,
                        close_all,
                    )
                    self.event_log.log(
                        "exit_skipped",
                        {
                            "token_mint": features.token_mint,
                            "position_id": position_id,
                            "strategy_id": strategy_id,
                            "reason": reason,
                            "cause": "already_recorded",
                            "close_all": bool(close_all),
                        },
                    )
                    continue

                tx_signature = None
                paper_quote_used = False
                paper_quote_error: str | None = None
                paper_sell_fee_sol = 0.0
                paper_cost_basis_allocated_sol = sell_size_sol
                paper_sell_net_sol = max(0.0, sell_size_sol + (sell_size_sol * pnl_multiple))
                gross_out_sol = paper_sell_net_sol + paper_sell_fee_sol
                _live_out_lamports: int = 0  # actual SOL received from on-chain execution
                if self._is_live:
                    logger.info(
                        "🔴 LIVE exit: %s | pnl=%.4f | reason=%s | sell_size=%.4f",
                        features.token_mint[:12],
                        pnl_multiple,
                        reason,
                        sell_size_sol,
                    )
                    token_amount_int = int(round(sell_token_amount))
                    if token_amount_int <= 0:
                        token_amount_int = int(current_token_amount)
                    if token_amount_int <= 0:
                        logger.warning(
                            "🔴 LIVE exit skipped – invalid token amount for %s",
                            features.token_mint[:12],
                        )
                        self.event_log.log(
                            "exit_skipped",
                            {
                                "token_mint": features.token_mint,
                                "position_id": position_id,
                                "strategy_id": strategy_id,
                                "reason": reason,
                                "cause": "invalid_token_amount",
                                "sell_token_amount": float(sell_token_amount),
                                "current_token_amount": float(current_token_amount),
                            },
                        )
                        continue
                    close_token_account = bool(
                        token_amount_int >= max(1, int(round(float(current_token_amount))))
                    )

                    assert self.trade_executor is not None
                    # Circuit breaker: after N consecutive sell failures on
                    # this mint, skip until cooldown expires. Stops burning
                    # priority fees + jito tips on a dead mint.
                    if self._sell_breaker_tripped(features.token_mint):
                        _is_permanent = features.token_mint in self._mint_sell_permanent_stuck
                        if _is_permanent:
                            # No route will ever recover — free the slot so the
                            # bot can keep buying fresh tokens instead of
                            # silently hitting max_concurrent_positions.
                            self._force_close_stuck_rug_position(
                                position=position,
                                metadata=metadata,
                                features=features,
                                strategy_id=strategy_id,
                                reason=reason,
                                last_error="circuit_breaker_permanent",
                            )
                            continue
                        # Throttle this log to once per 60s per mint; otherwise the
                        # exit engine's ~10s tick cadence produces WARN spam during
                        # a 5-minute cooldown window.
                        _now_mono = time.monotonic()
                        _last_at = self._mint_sell_breaker_last_log_at.get(features.token_mint, 0.0)
                        if _now_mono - _last_at >= 60.0:
                            self._mint_sell_breaker_last_log_at[features.token_mint] = _now_mono
                            logger.warning(
                                "🟡 LIVE exit skipped – sell breaker tripped (cooldown) for %s",
                                features.token_mint[:12],
                            )
                            self.event_log.log(
                                "exit_skipped",
                                {
                                    "token_mint": features.token_mint,
                                    "position_id": position_id,
                                    "strategy_id": strategy_id,
                                    "reason": reason,
                                    "cause": "circuit_breaker_cooldown",
                                },
                            )
                        continue

                    # Use pre-built TX if available (sign+broadcast only, ~50-150ms).
                    # Prebuilt TX is built against current_token_amount by the
                    # refresh loop; accept it if our sell amount is within 0.1%
                    # of the prebuilt amount. Exact-equality was brittle against
                    # float->int conversion and masked valid cache hits.
                    _prebuilt_sell = None
                    if self.live_sell_cache is not None:
                        _current_int = int(current_token_amount)
                        _tolerance = max(1, int(_current_int * 0.001))
                        if abs(token_amount_int - _current_int) <= _tolerance:
                            _prebuilt_sell = self.live_sell_cache.get(features.token_mint)

                    # Retry loop: first attempt may use prebuilt; subsequent
                    # attempts always rebuild fresh (prebuilt is assumed stale
                    # after a failure). Terminal errors (wallet balance
                    # zero/dust) break out of the loop immediately.
                    # Per-mint lock serializes sells across positions on the
                    # same mint so two cycles can't race the same ATA.
                    _max_attempts = max(
                        1,
                        int(getattr(self.risk_manager.config, "live_sell_max_attempts", 3) or 3),
                    )
                    _attempt = 0
                    result = None
                    _prefer_jupiter = strategy_id in ("main", "wallet")
                    _source_program = (
                        self.live_sell_cache.get_source_program(features.token_mint)
                        if self.live_sell_cache is not None
                        else None
                    )
                    with self._acquire_sell_mint_lock(features.token_mint):
                        while _attempt < _max_attempts:
                            _attempt += 1
                            if _attempt == 1 and _prebuilt_sell is not None:
                                result = self.trade_executor.execute_sell_prebuilt(
                                    token_mint=features.token_mint,
                                    token_amount=token_amount_int,
                                    prebuilt_tx=_prebuilt_sell,
                                    close_token_account=close_token_account,
                                    prefer_jupiter=_prefer_jupiter,
                                    strategy=strategy_id,
                                    source_program=_source_program,
                                )
                            else:
                                result = self.trade_executor.execute_sell(
                                    token_mint=features.token_mint,
                                    token_amount=token_amount_int,
                                    close_token_account=close_token_account,
                                    prefer_jupiter=_prefer_jupiter,
                                    strategy=strategy_id,
                                    source_program=_source_program,
                                )
                            if result.success:
                                break
                            _err_str = str(getattr(result, "error", "") or "")
                            if _err_str.startswith(
                                (
                                    "wallet_token_balance_zero",
                                    "wallet_token_balance_dust",
                                )
                            ):
                                break  # terminal — handled by dust/zero branch below
                            if _attempt >= _max_attempts:
                                break
                            _backoff = 0.5 * (2 ** (_attempt - 1))
                            logger.warning(
                                "🟡 LIVE sell attempt %d/%d failed for %s: %s; retrying in %.1fs",
                                _attempt,
                                _max_attempts,
                                features.token_mint[:12],
                                _err_str,
                                _backoff,
                            )
                            time.sleep(_backoff)
                    assert result is not None
                    if result.success:
                        self._record_sell_breaker_success(features.token_mint)
                    if not result.success:
                        execution_latency_trace = dict(getattr(result, "latency_trace", {}) or {})
                        latency_trace = _merge_latency_trace(
                            decision_latency_trace,
                            execution_latency_trace,
                        )
                        if str(getattr(result, "error", "") or "").startswith(
                            ("wallet_token_balance_zero", "wallet_token_balance_dust")
                        ):
                            # Terminal: dust/zero is not a retryable sell failure
                            # (wallet genuinely has nothing left) — don't trip breaker.
                            self._reconcile_external_live_close(
                                position=position,
                                metadata=metadata,
                                features=features,
                                strategy_id=strategy_id,
                                reason=reason,
                                result=result,
                                decision_latency_trace=decision_latency_trace,
                            )
                            continue
                        _err_class = self._classify_sell_error(
                            str(getattr(result, "error", "") or "")
                        )
                        _fail_count = self._record_sell_breaker_failure(
                            features.token_mint, error_class=_err_class
                        )
                        _now_permanent_stuck = (
                            features.token_mint in self._mint_sell_permanent_stuck
                        )
                        logger.error(
                            "🔴 LIVE exit FAILED for %s: %s (sig=%s) mint_fail_count=%d class=%s%s",
                            features.token_mint[:12],
                            result.error,
                            result.signature,
                            _fail_count,
                            _err_class,
                            " [PERMANENTLY STUCK]" if _now_permanent_stuck else "",
                        )
                        if _now_permanent_stuck:
                            self.event_log.log(
                                "live_sell_no_route_stuck",
                                {
                                    "token_mint": features.token_mint,
                                    "error": str(result.error or ""),
                                    "fail_count": _fail_count,
                                    "reason": reason,
                                    "error_class": _err_class,
                                },
                            )
                        execution_recorded_at = _resolve_live_recorded_at_iso(
                            features.entry_time, latency_trace
                        )
                        self.db.execute(
                            """
                            INSERT INTO executions (token_mint, action, mode, strategy_id, size_sol, price_sol, tx_signature, status, reason, created_at)
                            VALUES (?, 'SELL', 'live', ?, ?, ?, ?, 'FAILED', ?, ?)
                            """,
                            (
                                features.token_mint,
                                strategy_id,
                                sell_size_sol,
                                mark_price_sol,
                                result.signature,
                                f"live_sell_failed: {result.error}",
                                execution_recorded_at,
                            ),
                        )
                        self.event_log.log(
                            "live_exit_failed",
                            {
                                "token_mint": features.token_mint,
                                "selected_rule_id": position["selected_rule_id"],
                                "pnl_multiple": pnl_multiple,
                                "error": result.error,
                                "signature": result.signature,
                                "hold_seconds": hold_seconds,
                                "reason": reason,
                                "sell_size_sol": sell_size_sol,
                                "trigger_event_time": features.entry_time.isoformat(),
                                "execution_recorded_at": execution_recorded_at,
                                "pipeline_latency_trace": decision_latency_trace,
                                "execution_latency_trace": execution_latency_trace,
                                "latency_trace": latency_trace,
                                "reconciliation_error": getattr(
                                    result, "reconciliation_error", None
                                ),
                            },
                        )
                        self._record_failed_live_exit_fee_burn(
                            position=position,
                            metadata=metadata,
                            features=features,
                            result=result,
                            strategy_id=strategy_id,
                            reason=reason,
                            decision_latency_trace=decision_latency_trace,
                        )
                        if _now_permanent_stuck:
                            # Breaker just flipped permanent this cycle — close
                            # the DB position now so the slot frees up instead
                            # of waiting for the next exit tick to skip it.
                            self._force_close_stuck_rug_position(
                                position=position,
                                metadata=metadata,
                                features=features,
                                strategy_id=strategy_id,
                                reason=reason,
                                last_error=str(result.error or ""),
                            )
                        continue
                    tx_signature = result.signature
                    _live_out_lamports = result.out_amount
                    metadata["live_execution_latency_trace"] = dict(
                        getattr(result, "latency_trace", {}) or {}
                    )
                    metadata["live_latency_trace"] = _merge_latency_trace(
                        decision_latency_trace,
                        metadata.get("live_execution_latency_trace"),
                    )
                    if getattr(result, "reconciliation_error", None):
                        metadata["live_reconciliation_error"] = str(result.reconciliation_error)
                    logger.info(
                        "✅ LIVE exit confirmed: %s | sig=%s | out=%d lamports",
                        features.token_mint[:12],
                        result.signature,
                        _live_out_lamports,
                    )
                else:
                    logger.info(
                        "📝 PAPER exit: %s | pnl=%.4f | reason=%s | sell_size=%.4f",
                        features.token_mint[:12],
                        pnl_multiple,
                        reason,
                        sell_size_sol,
                    )
                    paper_realization = self._compute_paper_sell_realization(
                        token_mint=features.token_mint,
                        metadata=metadata,
                        current_token_amount=current_token_amount,
                        sell_token_amount=sell_token_amount,
                        fallback_sell_size_sol=sell_size_sol,
                        fallback_pnl_multiple=pnl_multiple,
                    )
                    realized_leg_pnl = float(paper_realization["realized_leg_pnl_sol"])
                    sell_token_amount = float(paper_realization["sell_token_amount"])
                    paper_quote_used = bool(paper_realization["used_quote"])
                    paper_quote_error = (
                        str(paper_realization["quote_error"])
                        if paper_realization["quote_error"] is not None
                        else None
                    )
                    paper_sell_fee_sol = float(paper_realization["sell_fee_sol"])
                    paper_cost_basis_allocated_sol = float(
                        paper_realization["cost_basis_allocated_sol"]
                    )
                    paper_sell_net_sol = float(paper_realization["sell_net_sol"])
                    gross_out_sol = paper_sell_net_sol + paper_sell_fee_sol
                if self._is_live:
                    (
                        live_gross_out_sol,
                        live_net_out_sol,
                        live_sell_token_amount,
                        live_fee_sol,
                        live_fill_details,
                    ) = _live_sell_reconciliation_fields(
                        result,
                        float(_live_out_lamports) / float(LAMPORTS_PER_SOL)
                        if _live_out_lamports > 0
                        else 0.0,
                        sell_token_amount,
                    )
                    metadata.update(live_fill_details)
                    if live_sell_token_amount > 0:
                        sell_token_amount = live_sell_token_amount
                    live_close_position = close_all or sell_token_amount >= max(
                        float(current_token_amount) - EPS, 0.0
                    )
                    (
                        paper_cost_basis_allocated_sol,
                        remaining_cost_basis_sol,
                        remaining_token_raw,
                    ) = _allocate_live_cost_basis(
                        metadata,
                        current_size_sol=current_size_sol,
                        current_token_amount=current_token_amount,
                        sell_size_sol=sell_size_sol,
                        sell_token_amount=sell_token_amount,
                        close_position=live_close_position,
                    )
                    if live_fee_sol > 0:
                        paper_sell_fee_sol = live_fee_sol
                    gross_out_sol = live_gross_out_sol
                    if live_net_out_sol or gross_out_sol:
                        paper_sell_net_sol = live_net_out_sol
                        realized_leg_pnl = live_net_out_sol - paper_cost_basis_allocated_sol
                        metadata["pnl_source"] = "reconciled"
                    else:
                        # Fallback if reconciliation failed and out_amount wasn't available.
                        realized_leg_pnl = sell_size_sol * pnl_multiple
                        metadata["pnl_source"] = "mark_price_fallback"
                        self.event_log.log(
                            "pnl_fallback_mark_price",
                            {
                                "token_mint": features.token_mint,
                                "position_id": position_id,
                                "strategy_id": strategy_id,
                                "selected_rule_id": position.get("selected_rule_id"),
                                "reason": reason,
                                "stage": "partial",
                                "sell_size_sol": float(sell_size_sol),
                                "pnl_multiple": float(pnl_multiple or 0.0),
                                "estimated_pnl_sol": float(realized_leg_pnl),
                            },
                        )
                existing_realized = float(position.get("realized_pnl_sol", 0.0) or 0.0)
                realized_total = existing_realized + realized_leg_pnl
                executed_pnl_multiple = (
                    realized_leg_pnl / paper_cost_basis_allocated_sol
                    if paper_cost_basis_allocated_sol > EPS
                    else None
                )
                quote_source = str(
                    metadata.get("paper_sell_quote_source")
                    or (
                        "live_tx_meta_reconciled"
                        if self._is_live and getattr(result, "reconciliation", None) is not None
                        else ("live_fill" if self._is_live else "mark_price_fallback")
                    )
                )

                remaining_size_sol = max(0.0, current_size_sol - sell_size_sol)
                remaining_token_amount = max(0.0, current_token_amount - sell_token_amount)
                is_closed = close_all or remaining_size_sol <= EPS or remaining_token_amount <= EPS
                if is_closed:
                    remaining_size_sol = 0.0
                    remaining_token_amount = 0.0
                position_status = "CLOSED" if is_closed else "OPEN"
                execution_status = "CLOSED" if is_closed else "PARTIAL"

                metadata["last_exit_reason"] = reason
                metadata["last_exit_at"] = features.entry_time.isoformat()
                metadata["last_exit_stage"] = next_stage
                metadata["remaining_size_sol"] = remaining_size_sol
                metadata["remaining_amount_received"] = remaining_token_amount
                metadata["last_mark_price_sol"] = float(mark_price_sol)
                metadata["last_mark_price_raw_sol"] = float(raw_price_sol or 0.0)
                metadata["last_mark_price_reliable_sol"] = float(
                    reliable_price_sol or mark_price_sol
                )
                metadata["last_quote_source"] = quote_source
                metadata["last_realized_leg_pnl_sol"] = float(realized_leg_pnl)
                metadata["last_executed_pnl_multiple"] = float(executed_pnl_multiple or 0.0)

                # For partial exits the remaining runner leg still has mark-to-market value.
                _remaining_unrealized = 0.0
                if not is_closed and remaining_size_sol > EPS:
                    _remaining_unrealized = remaining_size_sol * pnl_multiple
                else:
                    metadata["paper_remaining_cost_basis_sol"] = 0.0
                    metadata["paper_remaining_token_raw"] = 0.0
                if self._is_live and not is_closed:
                    metadata["paper_remaining_cost_basis_sol"] = max(0.0, remaining_cost_basis_sol)
                    metadata["paper_remaining_token_raw"] = max(0.0, remaining_token_raw)
                execution_recorded_at = (
                    _resolve_live_recorded_at_iso(
                        features.entry_time, metadata.get("live_latency_trace")
                    )
                    if self._is_live
                    else features.entry_time.isoformat()
                )
                execution_recorded_dt = (
                    _resolve_live_recorded_at_dt(
                        features.entry_time, metadata.get("live_latency_trace")
                    )
                    if self._is_live
                    else features.entry_time
                )
                metadata["last_exit_at"] = execution_recorded_at

                self.position_manager.update_position_after_exit(
                    position_id=int(position["id"]),
                    exit_stage=next_stage,
                    realized_pnl_sol=realized_total,
                    status=position_status,
                    remaining_size_sol=remaining_size_sol,
                    remaining_amount_received=remaining_token_amount,
                    metadata=metadata,
                    unrealized_pnl_sol=_remaining_unrealized,
                )

                if (
                    is_closed
                    and executed_pnl_multiple is not None
                    and executed_pnl_multiple >= 0.05
                ):
                    _triggering_wallet = position.get("triggering_wallet")
                    if _triggering_wallet:
                        try:
                            _append_proven_winner(
                                str(_triggering_wallet),
                                position_id=int(position["id"]),
                                realized_pnl_pct=float(executed_pnl_multiple),
                                realized_pnl_sol=float(realized_total),
                            )
                        except Exception:
                            logger.exception("proven_winner save failed")

                # Keep live sell cache in sync with the remaining position size.
                if self.live_sell_cache is not None:
                    if is_closed:
                        self.live_sell_cache.unregister(features.token_mint)
                    elif remaining_token_amount > 0:
                        # After a partial exit (TP1 etc.) the cache still holds the
                        # original full-position TX. Update it to the remaining amount
                        # so the background refresh rebuilds for the correct size.
                        self.live_sell_cache.update_amount(
                            features.token_mint, int(remaining_token_amount)
                        )

                self.db.record_trade_leg(
                    position_id=position_id,
                    token_mint=features.token_mint,
                    action="SELL",
                    mode=mode_db,
                    strategy_id=strategy_id,
                    selected_rule_id=str(position.get("selected_rule_id") or ""),
                    selected_regime=str(position.get("selected_regime") or ""),
                    close_position=is_closed,
                    stop_out=stop_out,
                    hit_2x_achieved=bool(metadata.get("hit_2x_achieved", False)),
                    hit_5x_achieved=bool(metadata.get("hit_5x_achieved", False)),
                    quote_used=paper_quote_used or self._is_live,
                    quote_source=quote_source,
                    quote_error=paper_quote_error,
                    observed_price_sol=mark_price_sol,
                    observed_price_raw_sol=_coerce_float(raw_price_sol),
                    observed_price_reliable_sol=_coerce_float(reliable_price_sol),
                    observed_pnl_multiple=pnl_multiple,
                    executed_pnl_multiple=executed_pnl_multiple,
                    cost_basis_sol=paper_cost_basis_allocated_sol,
                    leg_size_sol=sell_size_sol,
                    token_amount_raw=sell_token_amount,
                    gross_sol=gross_out_sol,
                    net_sol=paper_sell_net_sol,
                    fee_sol=paper_sell_fee_sol,
                    realized_leg_pnl_sol=realized_leg_pnl,
                    realized_total_pnl_sol=realized_total,
                    reason=reason,
                    tx_signature=tx_signature,
                    created_at=execution_recorded_at,
                    metadata={
                        "hold_seconds": hold_seconds,
                        "remaining_size_sol": remaining_size_sol,
                        "remaining_token_amount": remaining_token_amount,
                        "trigger_event_time": features.entry_time.isoformat(),
                        "execution_recorded_at": execution_recorded_at,
                        "pipeline_latency_trace": metadata.get("exit_pipeline_latency_trace"),
                        "execution_latency_trace": metadata.get("live_execution_latency_trace"),
                        "latency_trace": metadata.get("live_latency_trace")
                        or metadata.get("exit_pipeline_latency_trace"),
                        "reconciliation_error": metadata.get("live_reconciliation_error"),
                    },
                )

                # Realized loss counter should be updated for every sell leg.
                self.risk_manager.record_daily_loss(
                    realized_leg_pnl, realized_at=execution_recorded_dt
                )
                if is_closed:
                    self.rule_performance.record_exit(
                        position["selected_rule_id"],
                        pnl_sol=realized_total,
                        hit_2x=bool(metadata.get("hit_2x_achieved", False)),
                        hit_5x=bool(metadata.get("hit_5x_achieved", False)),
                        stop_out=stop_out,
                        close_position=True,
                    )
                    self.risk_manager.set_cooldown(
                        features.token_mint, minutes=30, reason="position_closed"
                    )
                    if self.quote_cache is not None:
                        self.quote_cache.unregister(features.token_mint)

                self.db.execute(
                    """
                    INSERT INTO executions (token_mint, action, mode, strategy_id, size_sol, price_sol, tx_signature, status, reason, created_at)
                    VALUES (?, 'SELL', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        features.token_mint,
                        mode_db,
                        strategy_id,
                        sell_size_sol,
                        mark_price_sol,
                        tx_signature,
                        execution_status,
                        reason,
                        execution_recorded_at,
                    ),
                )

                event_type = f"{mode.lower()}_exit"
                self.event_log.log(
                    event_type,
                    {
                        "token_mint": features.token_mint,
                        "selected_rule_id": position["selected_rule_id"],
                        "pnl_multiple": pnl_multiple,
                        "realized_leg_pnl_sol": realized_leg_pnl,
                        "realized_total_pnl_sol": realized_total,
                        "remaining_size_sol": remaining_size_sol,
                        "remaining_token_amount": remaining_token_amount,
                        "status": execution_status,
                        "reason": reason,
                        "mode": mode,
                        "strategy_id": strategy_id,
                        "tx_signature": tx_signature,
                        "hold_seconds": hold_seconds,
                        "trigger_event_time": features.entry_time.isoformat(),
                        "execution_recorded_at": execution_recorded_at,
                        "paper_quote_used": paper_quote_used,
                        "paper_quote_error": paper_quote_error,
                        "paper_sell_quote_source": quote_source,
                        "paper_sell_fee_sol": paper_sell_fee_sol,
                        "paper_cost_basis_allocated_sol": paper_cost_basis_allocated_sol,
                        "paper_sell_net_sol": paper_sell_net_sol,
                        "mark_price_sol": mark_price_sol,
                        "mark_price_raw_sol": raw_price_sol,
                        "mark_price_reliable_sol": reliable_price_sol,
                        "executed_pnl_multiple": executed_pnl_multiple,
                        "pipeline_latency_trace": metadata.get("exit_pipeline_latency_trace"),
                        "execution_latency_trace": metadata.get("live_execution_latency_trace"),
                        "latency_trace": metadata.get("live_latency_trace")
                        or metadata.get("exit_pipeline_latency_trace"),
                        "reconciliation_error": metadata.get("live_reconciliation_error"),
                        "fill_slot": metadata.get("live_fill_slot"),
                        "actual_wallet_delta_lamports": metadata.get("live_wallet_delta_lamports"),
                        "actual_token_delta_raw": metadata.get("live_token_delta_raw"),
                        "exact_fee_lamports": metadata.get("live_exact_fee_lamports"),
                    },
                )
            finally:
                self._release_exit_slot(position_id)

    def force_close_position(
        self,
        position: dict[str, Any],
        features: RuntimeFeatures,
        reason: str = "session_end",
    ) -> bool:
        """Force-close one open position immediately (used for session shutdown)."""
        mode = self._mode_label
        mode_db = "live" if self._is_live else "paper"
        exit_pipeline_trace = _merge_latency_trace(
            (features.raw or {}).get("__latency_trace"),
            (features.raw or {}).get("__exit_latency_trace"),
        )
        force_started_at = datetime.now(tz=timezone.utc)
        if exit_pipeline_trace:
            exit_pipeline_trace["force_close_started_at"] = force_started_at.isoformat()
        force_started = time.monotonic()

        entry_price = float(position.get("entry_price_sol", 0.0) or 0.0)
        current_size_sol = float(position.get("size_sol", 0.0) or 0.0)
        current_token_amount = float(position.get("amount_received", 0.0) or 0.0)
        if entry_price <= EPS or current_size_sol <= EPS or current_token_amount <= EPS:
            return False

        metadata = json.loads(position.get("metadata_json") or "{}")
        strategy_id = str(metadata.get("strategy_id") or position.get("strategy_id") or "main")
        mark_price, _, _, _ = self._resolve_mark_price(
            features=features,
            metadata=metadata,
            entry_price=entry_price,
        )
        pnl_multiple = (mark_price / entry_price) - 1.0 if entry_price > EPS else 0.0
        hold_seconds = None
        try:
            entry_time = datetime.fromisoformat(
                str(position.get("entry_time", "")).replace("Z", "+00:00")
            )
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
            hold_seconds = max(0.0, (features.entry_time - entry_time).total_seconds())
        except Exception:  # noqa: BLE001
            hold_seconds = None

        decision_latency_trace = _merge_latency_trace(exit_pipeline_trace)
        decision_latency_trace["exit_decided_at"] = datetime.now(tz=timezone.utc).isoformat()
        decision_latency_trace["exit_decision_ms"] = (time.monotonic() - force_started) * 1000.0
        decision_latency_trace["exit_reason"] = reason
        metadata["exit_pipeline_latency_trace"] = decision_latency_trace

        position_id = int(position["id"])
        if not self._try_acquire_exit_slot(position_id):
            logger.debug(
                "exit_engine: skip duplicate force-close for position %d (%s) – already in-flight",
                position_id,
                features.token_mint[:12],
            )
            return False

        try:
            if self._exit_leg_already_recorded(position_id, reason, True):
                logger.warning(
                    "exit_engine: skip duplicate persisted force-close for position %d (%s) reason=%s",
                    position_id,
                    features.token_mint[:12],
                    reason,
                )
                return True

            tx_signature = None
            if self._is_live:
                token_amount_int = max(1, int(round(current_token_amount)))
                assert self.trade_executor is not None
                # Force close is always a full-position sell — use pre-built TX if available.
                _prebuilt_sell = None
                if self.live_sell_cache is not None:
                    _prebuilt_sell = self.live_sell_cache.get(features.token_mint)

                # Per-mint lock: serialize sells across positions on the same
                # mint — a force_close racing a staged-exit on a duplicate mint
                # would both reference the same ATA.
                _prefer_jupiter = strategy_id in ("main", "wallet")
                _source_program = (
                    self.live_sell_cache.get_source_program(features.token_mint)
                    if self.live_sell_cache is not None
                    else None
                )
                with self._acquire_sell_mint_lock(features.token_mint):
                    if _prebuilt_sell is not None:
                        result = self.trade_executor.execute_sell_prebuilt(
                            token_mint=features.token_mint,
                            token_amount=token_amount_int,
                            prebuilt_tx=_prebuilt_sell,
                            close_token_account=True,
                            prefer_jupiter=_prefer_jupiter,
                            strategy=strategy_id,
                            source_program=_source_program,
                        )
                    else:
                        result = self.trade_executor.execute_sell(
                            token_mint=features.token_mint,
                            token_amount=token_amount_int,
                            close_token_account=True,
                            prefer_jupiter=_prefer_jupiter,
                            strategy=strategy_id,
                            source_program=_source_program,
                        )
                if result.success:
                    self._record_sell_breaker_success(features.token_mint)
                if not result.success:
                    execution_latency_trace = dict(getattr(result, "latency_trace", {}) or {})
                    latency_trace = _merge_latency_trace(
                        decision_latency_trace,
                        execution_latency_trace,
                    )
                    if str(getattr(result, "error", "") or "").startswith(
                        ("wallet_token_balance_zero", "wallet_token_balance_dust")
                    ):
                        self._reconcile_external_live_close(
                            position=position,
                            metadata=metadata,
                            features=features,
                            strategy_id=strategy_id,
                            reason=reason,
                            result=result,
                            decision_latency_trace=decision_latency_trace,
                            forced=True,
                        )
                        return True
                    _err_class = self._classify_sell_error(str(getattr(result, "error", "") or ""))
                    self._record_sell_breaker_failure(features.token_mint, error_class=_err_class)
                    if features.token_mint in self._mint_sell_permanent_stuck:
                        self.event_log.log(
                            "live_sell_no_route_stuck",
                            {
                                "token_mint": features.token_mint,
                                "error": str(result.error or ""),
                                "reason": reason,
                                "forced": True,
                                "error_class": _err_class,
                            },
                        )
                    execution_recorded_at = _resolve_live_recorded_at_iso(
                        features.entry_time, latency_trace
                    )
                    self.db.execute(
                        """
                        INSERT INTO executions (token_mint, action, mode, strategy_id, size_sol, price_sol, tx_signature, status, reason, created_at)
                        VALUES (?, 'SELL', 'live', ?, ?, ?, ?, 'FAILED', ?, ?)
                        """,
                        (
                            features.token_mint,
                            strategy_id,
                            current_size_sol,
                            mark_price,
                            result.signature,
                            f"force_close_failed: {result.error}",
                            execution_recorded_at,
                        ),
                    )
                    self.event_log.log(
                        "live_exit_failed",
                        {
                            "token_mint": features.token_mint,
                            "selected_rule_id": position.get("selected_rule_id"),
                            "reason": reason,
                            "error": result.error,
                            "signature": result.signature,
                            "forced": True,
                            "trigger_event_time": features.entry_time.isoformat(),
                            "execution_recorded_at": execution_recorded_at,
                            "pipeline_latency_trace": decision_latency_trace,
                            "execution_latency_trace": execution_latency_trace,
                            "latency_trace": latency_trace,
                            "reconciliation_error": getattr(result, "reconciliation_error", None),
                        },
                    )
                    self._record_failed_live_exit_fee_burn(
                        position=position,
                        metadata=metadata,
                        features=features,
                        result=result,
                        strategy_id=strategy_id,
                        reason=reason,
                        decision_latency_trace=decision_latency_trace,
                    )
                    if features.token_mint in self._mint_sell_permanent_stuck:
                        # Forced-close path: sell permanently dead, free the slot.
                        self._force_close_stuck_rug_position(
                            position=position,
                            metadata=metadata,
                            features=features,
                            strategy_id=strategy_id,
                            reason=reason,
                            last_error=str(result.error or ""),
                        )
                        return True
                    return False
                tx_signature = result.signature
                metadata["live_execution_latency_trace"] = dict(
                    getattr(result, "latency_trace", {}) or {}
                )
                metadata["live_latency_trace"] = _merge_latency_trace(
                    decision_latency_trace,
                    metadata.get("live_execution_latency_trace"),
                )

            paper_quote_used = False
            paper_quote_error: str | None = None
            paper_sell_fee_sol = 0.0
            paper_cost_basis_allocated_sol = current_size_sol
            paper_sell_net_sol = max(0.0, current_size_sol + (current_size_sol * pnl_multiple))
            gross_out_sol = paper_sell_net_sol + paper_sell_fee_sol
            if not self._is_live:
                paper_realization = self._compute_paper_sell_realization(
                    token_mint=features.token_mint,
                    metadata=metadata,
                    current_token_amount=current_token_amount,
                    sell_token_amount=current_token_amount,
                    fallback_sell_size_sol=current_size_sol,
                    fallback_pnl_multiple=pnl_multiple,
                )
                realized_leg_pnl = float(paper_realization["realized_leg_pnl_sol"])
                paper_quote_used = bool(paper_realization["used_quote"])
                paper_quote_error = (
                    str(paper_realization["quote_error"])
                    if paper_realization["quote_error"] is not None
                    else None
                )
                paper_sell_fee_sol = float(paper_realization["sell_fee_sol"])
                paper_cost_basis_allocated_sol = float(
                    paper_realization["cost_basis_allocated_sol"]
                )
                paper_sell_net_sol = float(paper_realization["sell_net_sol"])
                gross_out_sol = paper_sell_net_sol + paper_sell_fee_sol
            else:
                (
                    live_gross_out_sol,
                    live_net_out_sol,
                    live_sell_token_amount,
                    live_fee_sol,
                    live_fill_details,
                ) = _live_sell_reconciliation_fields(
                    result,
                    float(result.out_amount) / float(LAMPORTS_PER_SOL)
                    if float(result.out_amount or 0) > 0
                    else 0.0,
                    current_token_amount,
                )
                metadata.update(live_fill_details)
                (
                    paper_cost_basis_allocated_sol,
                    _remaining_cost_basis_sol,
                    _remaining_token_raw,
                ) = _allocate_live_cost_basis(
                    metadata,
                    current_size_sol=current_size_sol,
                    current_token_amount=current_token_amount,
                    sell_size_sol=current_size_sol,
                    sell_token_amount=live_sell_token_amount
                    if live_sell_token_amount > 0
                    else current_token_amount,
                    close_position=True,
                )
                if live_fee_sol > 0:
                    paper_sell_fee_sol = live_fee_sol
                if live_sell_token_amount > 0:
                    current_token_amount = live_sell_token_amount
                gross_out_sol = live_gross_out_sol
                if live_net_out_sol or gross_out_sol:
                    paper_sell_net_sol = live_net_out_sol
                    realized_leg_pnl = live_net_out_sol - paper_cost_basis_allocated_sol
                    metadata["pnl_source"] = "reconciled"
                else:
                    realized_leg_pnl = current_size_sol * pnl_multiple
                    metadata["pnl_source"] = "mark_price_fallback"
                    self.event_log.log(
                        "pnl_fallback_mark_price",
                        {
                            "token_mint": features.token_mint,
                            "position_id": position_id,
                            "strategy_id": strategy_id,
                            "selected_rule_id": position.get("selected_rule_id"),
                            "reason": reason,
                            "stage": "close",
                            "sell_size_sol": float(current_size_sol),
                            "pnl_multiple": float(pnl_multiple or 0.0),
                            "estimated_pnl_sol": float(realized_leg_pnl),
                        },
                    )
            existing_realized = float(position.get("realized_pnl_sol", 0.0) or 0.0)
            realized_total = existing_realized + realized_leg_pnl
            executed_pnl_multiple = (
                realized_leg_pnl / paper_cost_basis_allocated_sol
                if paper_cost_basis_allocated_sol > EPS
                else None
            )
            next_stage = 90

            metadata["last_exit_reason"] = reason
            metadata["last_exit_at"] = features.entry_time.isoformat()
            metadata["last_exit_stage"] = next_stage
            metadata["remaining_size_sol"] = 0.0
            metadata["remaining_amount_received"] = 0.0
            metadata["paper_remaining_cost_basis_sol"] = 0.0
            metadata["paper_remaining_token_raw"] = 0.0
            metadata["last_mark_price_sol"] = float(mark_price)
            metadata["last_quote_source"] = str(
                metadata.get("paper_sell_quote_source")
                or (
                    "live_tx_meta_reconciled"
                    if self._is_live and getattr(result, "reconciliation", None) is not None
                    else ("live_fill" if self._is_live else "mark_price_fallback")
                )
            )
            metadata["last_realized_leg_pnl_sol"] = float(realized_leg_pnl)
            metadata["last_executed_pnl_multiple"] = float(executed_pnl_multiple or 0.0)
            execution_recorded_at = (
                _resolve_live_recorded_at_iso(
                    features.entry_time, metadata.get("live_latency_trace")
                )
                if self._is_live
                else features.entry_time.isoformat()
            )
            execution_recorded_dt = (
                _resolve_live_recorded_at_dt(
                    features.entry_time, metadata.get("live_latency_trace")
                )
                if self._is_live
                else features.entry_time
            )
            metadata["last_exit_at"] = execution_recorded_at

            self.position_manager.update_position_after_exit(
                position_id=position_id,
                exit_stage=next_stage,
                realized_pnl_sol=realized_total,
                status="CLOSED",
                remaining_size_sol=0.0,
                remaining_amount_received=0.0,
                metadata=metadata,
            )

            if executed_pnl_multiple is not None and executed_pnl_multiple >= 0.05:
                _triggering_wallet = position.get("triggering_wallet")
                if _triggering_wallet:
                    try:
                        _append_proven_winner(
                            str(_triggering_wallet),
                            position_id=int(position_id),
                            realized_pnl_pct=float(executed_pnl_multiple),
                            realized_pnl_sol=float(realized_total),
                        )
                    except Exception:
                        logger.exception("proven_winner save failed")

            quote_source = str(
                metadata.get("paper_sell_quote_source")
                or (
                    "live_tx_meta_reconciled"
                    if self._is_live and getattr(result, "reconciliation", None) is not None
                    else ("live_fill" if self._is_live else "mark_price_fallback")
                )
            )
            self.db.record_trade_leg(
                position_id=position_id,
                token_mint=features.token_mint,
                action="SELL",
                mode=mode_db,
                strategy_id=strategy_id,
                selected_rule_id=str(position.get("selected_rule_id") or ""),
                selected_regime=str(position.get("selected_regime") or ""),
                close_position=True,
                stop_out=False,
                hit_2x_achieved=bool(metadata.get("hit_2x_achieved", False)),
                hit_5x_achieved=bool(metadata.get("hit_5x_achieved", False)),
                quote_used=paper_quote_used or self._is_live,
                quote_source=quote_source,
                quote_error=paper_quote_error,
                observed_price_sol=mark_price,
                observed_price_raw_sol=_coerce_float(features.raw.get("last_price_sol_raw")),
                observed_price_reliable_sol=_coerce_float(
                    features.raw.get("last_price_sol_reliable")
                ),
                observed_pnl_multiple=pnl_multiple,
                executed_pnl_multiple=executed_pnl_multiple,
                cost_basis_sol=paper_cost_basis_allocated_sol,
                leg_size_sol=current_size_sol,
                token_amount_raw=current_token_amount,
                gross_sol=gross_out_sol,
                net_sol=paper_sell_net_sol,
                fee_sol=paper_sell_fee_sol,
                realized_leg_pnl_sol=realized_leg_pnl,
                realized_total_pnl_sol=realized_total,
                reason=reason,
                tx_signature=tx_signature,
                created_at=execution_recorded_at,
                metadata={
                    "forced": True,
                    "hold_seconds": hold_seconds,
                    "trigger_event_time": features.entry_time.isoformat(),
                    "execution_recorded_at": execution_recorded_at,
                    "pipeline_latency_trace": metadata.get("exit_pipeline_latency_trace"),
                    "execution_latency_trace": metadata.get("live_execution_latency_trace"),
                    "latency_trace": metadata.get("live_latency_trace")
                    or metadata.get("exit_pipeline_latency_trace"),
                    "reconciliation_error": metadata.get("live_reconciliation_error"),
                },
            )

            self.risk_manager.record_daily_loss(realized_leg_pnl, realized_at=execution_recorded_dt)
            self.rule_performance.record_exit(
                str(position.get("selected_rule_id") or "unknown"),
                pnl_sol=realized_total,
                hit_2x=bool(metadata.get("hit_2x_achieved", False)),
                hit_5x=bool(metadata.get("hit_5x_achieved", False)),
                stop_out=False,
                close_position=True,
            )
            self.risk_manager.set_cooldown(
                features.token_mint, minutes=30, reason="position_closed"
            )
            if self.quote_cache is not None:
                self.quote_cache.unregister(features.token_mint)

            self.db.execute(
                """
                INSERT INTO executions (token_mint, action, mode, strategy_id, size_sol, price_sol, tx_signature, status, reason, created_at)
                VALUES (?, 'SELL', ?, ?, ?, ?, ?, 'CLOSED', ?, ?)
                """,
                (
                    features.token_mint,
                    mode_db,
                    strategy_id,
                    current_size_sol,
                    mark_price,
                    tx_signature,
                    reason,
                    execution_recorded_at,
                ),
            )

            event_type = f"{mode.lower()}_exit"
            self.event_log.log(
                event_type,
                {
                    "token_mint": features.token_mint,
                    "selected_rule_id": position.get("selected_rule_id"),
                    "pnl_multiple": pnl_multiple,
                    "realized_leg_pnl_sol": realized_leg_pnl,
                    "realized_total_pnl_sol": realized_total,
                    "remaining_size_sol": 0.0,
                    "remaining_token_amount": 0.0,
                    "status": "CLOSED",
                    "reason": reason,
                    "mode": mode,
                    "strategy_id": strategy_id,
                    "tx_signature": tx_signature,
                    "hold_seconds": hold_seconds,
                    "forced": True,
                    "trigger_event_time": features.entry_time.isoformat(),
                    "execution_recorded_at": execution_recorded_at,
                    "paper_quote_used": paper_quote_used,
                    "paper_quote_error": paper_quote_error,
                    "paper_sell_quote_source": quote_source,
                    "paper_sell_fee_sol": paper_sell_fee_sol,
                    "paper_cost_basis_allocated_sol": paper_cost_basis_allocated_sol,
                    "paper_sell_net_sol": paper_sell_net_sol,
                    "mark_price_sol": mark_price,
                    "pipeline_latency_trace": metadata.get("exit_pipeline_latency_trace"),
                    "executed_pnl_multiple": executed_pnl_multiple,
                    "execution_latency_trace": metadata.get("live_execution_latency_trace"),
                    "latency_trace": metadata.get("live_latency_trace")
                    or metadata.get("exit_pipeline_latency_trace"),
                    "reconciliation_error": metadata.get("live_reconciliation_error"),
                    "fill_slot": metadata.get("live_fill_slot"),
                    "actual_wallet_delta_lamports": metadata.get("live_wallet_delta_lamports"),
                    "actual_token_delta_raw": metadata.get("live_token_delta_raw"),
                    "exact_fee_lamports": metadata.get("live_exact_fee_lamports"),
                },
            )
            return True
        finally:
            self._release_exit_slot(position_id)
