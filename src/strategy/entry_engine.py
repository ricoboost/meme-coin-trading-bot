"""Entry engine for paper and live trading."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from typing import Optional

from src.bot.models import MatchResult, PositionRecord, RuntimeFeatures
from src.execution.jupiter_client import LAMPORTS_PER_SOL
from src.execution.trade_executor import PaperTradeEstimate, TradeExecutor
from src.portfolio.position_manager import PositionManager
from src.portfolio.rule_performance import RulePerformanceTracker
from src.strategy.risk_manager import RiskManager
from src.storage.bot_db import BotDB
from src.storage.event_log import EventLogger

logger = logging.getLogger(__name__)


def _coerce_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _merge_latency_trace(*traces: Any) -> dict[str, Any]:
    """Merge pipeline and execution latency traces in order."""
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


def _live_entry_fill_fields(
    result: Any, fallback_size_sol: float
) -> tuple[float, float, float, float, dict[str, Any]]:
    """Return exact live fill fields when reconciliation data is available."""
    reconciliation = getattr(result, "reconciliation", None)
    details: dict[str, Any] = {
        "live_execution_latency_trace": dict(getattr(result, "latency_trace", {}) or {}),
        "live_reconciliation_error": getattr(result, "reconciliation_error", None),
    }
    if reconciliation is None:
        live_fee_sol = 0.0
        amount_received = float(getattr(result, "out_amount", 0) or 0)
        live_entry_cost_basis = float(fallback_size_sol)
        token_amount_raw = amount_received
        return (
            amount_received,
            live_fee_sol,
            live_entry_cost_basis,
            token_amount_raw,
            details,
        )

    exact_fee_sol = float(reconciliation.fee_lamports) / float(LAMPORTS_PER_SOL)
    exact_tip_sol = float(getattr(reconciliation, "tip_lamports", 0) or 0) / float(LAMPORTS_PER_SOL)
    wallet_spend_sol = max(
        0.0, -float(reconciliation.wallet_delta_lamports) / float(LAMPORTS_PER_SOL)
    )
    amount_received = (
        float(reconciliation.token_delta_raw)
        if reconciliation.token_delta_raw > 0
        else float(getattr(result, "out_amount", 0) or 0)
    )
    live_entry_cost_basis = (
        wallet_spend_sol if wallet_spend_sol > 0 else float(fallback_size_sol) + exact_fee_sol
    )
    live_fee_sol = max(
        live_entry_cost_basis - float(fallback_size_sol),
        exact_fee_sol + exact_tip_sol,
        0.0,
    )
    token_amount_raw = (
        amount_received if amount_received > 0 else float(getattr(result, "out_amount", 0) or 0)
    )
    details.update(
        {
            "live_exact_fee_lamports": int(reconciliation.fee_lamports),
            "live_tip_lamports": int(getattr(reconciliation, "tip_lamports", 0) or 0),
            "live_effective_fee_lamports": int(round(live_fee_sol * float(LAMPORTS_PER_SOL))),
            "live_wallet_pre_lamports": int(reconciliation.wallet_pre_lamports),
            "live_wallet_post_lamports": int(reconciliation.wallet_post_lamports),
            "live_wallet_delta_lamports": int(reconciliation.wallet_delta_lamports),
            "live_token_pre_raw": int(reconciliation.token_pre_raw),
            "live_token_post_raw": int(reconciliation.token_post_raw),
            "live_token_delta_raw": int(reconciliation.token_delta_raw),
            "live_token_decimals": int(reconciliation.token_decimals),
            "live_quote_out_amount_diff_raw": int(reconciliation.quote_out_amount_diff),
            "live_fill_slot": reconciliation.slot,
        }
    )
    return (
        amount_received,
        live_fee_sol,
        live_entry_cost_basis,
        float(token_amount_raw),
        details,
    )


class EntryEngine:
    """Open positions when rules and risk controls allow.

    Supports both paper (simulated) and live (real on-chain) entries,
    controlled by :attr:`TradeExecutor.live`.
    """

    def __init__(
        self,
        db: BotDB,
        position_manager: PositionManager,
        rule_performance: RulePerformanceTracker,
        event_log: EventLogger,
        trade_executor: Optional[TradeExecutor] = None,
        risk_manager: RiskManager | None = None,
    ) -> None:
        self.db = db
        self.position_manager = position_manager
        self.rule_performance = rule_performance
        self.event_log = event_log
        self.trade_executor = trade_executor
        self.risk_manager = risk_manager

    @property
    def _is_live(self) -> bool:
        return self.trade_executor is not None and self.trade_executor.live

    @property
    def _mode_label(self) -> str:
        return "LIVE" if self._is_live else "PAPER"

    def _record_failed_live_entry_fee_burn(
        self,
        *,
        features: RuntimeFeatures,
        match: MatchResult,
        size_sol: float,
        strategy_id: str,
        result: Any,
        pipeline_latency_trace: dict[str, Any],
        latency_trace: dict[str, Any],
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
        selected_rule = match.selected_rule
        selected_rule_id = str(selected_rule.rule_id if selected_rule is not None else "")
        selected_regime = str(
            selected_rule.regime if selected_rule is not None else match.detected_regime or ""
        )
        execution_recorded_at = _resolve_live_recorded_at_iso(features.entry_time, latency_trace)
        self.db.record_trade_leg(
            position_id=None,
            token_mint=features.token_mint,
            action="BUY_FEE",
            mode="live",
            strategy_id=strategy_id,
            selected_rule_id=selected_rule_id,
            selected_regime=selected_regime,
            close_position=True,
            quote_used=False,
            quote_source="live_failed_tx_reconciled",
            cost_basis_sol=burn_sol,
            leg_size_sol=size_sol,
            token_amount_raw=0.0,
            gross_sol=0.0,
            net_sol=0.0,
            fee_sol=burn_sol,
            realized_leg_pnl_sol=-burn_sol,
            realized_total_pnl_sol=-burn_sol,
            reason="live_buy_failed_fee_burn",
            tx_signature=result.signature,
            created_at=execution_recorded_at,
            metadata={
                "trigger_event_time": features.entry_time.isoformat(),
                "execution_recorded_at": execution_recorded_at,
                "pipeline_latency_trace": pipeline_latency_trace,
                "execution_latency_trace": dict(getattr(result, "latency_trace", {}) or {}),
                "latency_trace": latency_trace,
                "reconciliation_error": getattr(result, "reconciliation_error", None),
                "live_exact_fee_lamports": int(getattr(reconciliation, "fee_lamports", 0) or 0),
                "live_tip_lamports": int(getattr(reconciliation, "tip_lamports", 0) or 0),
                "live_wallet_delta_lamports": int(
                    getattr(reconciliation, "wallet_delta_lamports", 0) or 0
                ),
            },
        )
        if selected_rule is not None:
            self.rule_performance.record_entry(selected_rule.rule_id, selected_rule.regime)
            self.rule_performance.record_exit(
                selected_rule.rule_id,
                pnl_sol=-burn_sol,
                hit_2x=False,
                hit_5x=False,
                stop_out=False,
                close_position=True,
            )
        if self.risk_manager is not None:
            self.risk_manager.record_daily_loss(
                -burn_sol,
                realized_at=_resolve_live_recorded_at_dt(features.entry_time, latency_trace),
            )
        self.event_log.log(
            "live_entry_failed_fee_burn",
            {
                "token_mint": features.token_mint,
                "strategy_id": strategy_id,
                "selected_rule_id": selected_rule_id,
                "size_sol": size_sol,
                "signature": result.signature,
                "trigger_event_time": features.entry_time.isoformat(),
                "execution_recorded_at": execution_recorded_at,
                "fee_burn_sol": burn_sol,
                "wallet_delta_lamports": int(
                    getattr(reconciliation, "wallet_delta_lamports", 0) or 0
                ),
                "exact_fee_lamports": int(getattr(reconciliation, "fee_lamports", 0) or 0),
                "tip_lamports": int(getattr(reconciliation, "tip_lamports", 0) or 0),
                "pipeline_latency_trace": pipeline_latency_trace,
                "execution_latency_trace": dict(getattr(result, "latency_trace", {}) or {}),
                "latency_trace": latency_trace,
            },
        )
        return burn_sol

    # ------------------------------------------------------------------
    # Unified entry
    # ------------------------------------------------------------------

    def execute_entry(
        self,
        features: RuntimeFeatures,
        match: MatchResult,
        size_sol: float,
        current_exposure_sol: float = 0.0,
        open_position_count: int = 0,
        strategy_id: str = "main",
        extra_metadata: dict[str, Any] | None = None,
        paper_entry_estimate: PaperTradeEstimate | None = None,
        prebuilt_tx: Any = None,
        source_program: str | None = None,
    ) -> Optional[PositionRecord]:
        """Execute an entry in the current mode (paper or live).

        Returns the :class:`PositionRecord` on success, or ``None`` on
        live-execution failure.
        """
        if self._is_live:
            return self._execute_live_entry(
                features,
                match,
                size_sol,
                current_exposure_sol=current_exposure_sol,
                open_position_count=open_position_count,
                strategy_id=strategy_id,
                extra_metadata=extra_metadata,
                prebuilt_tx=prebuilt_tx,
                source_program=source_program,
            )
        return self.execute_paper_entry(
            features,
            match,
            size_sol,
            strategy_id=strategy_id,
            extra_metadata=extra_metadata,
            paper_entry_estimate=paper_entry_estimate,
        )

    async def execute_entry_async(
        self,
        features: RuntimeFeatures,
        match: MatchResult,
        size_sol: float,
        current_exposure_sol: float = 0.0,
        open_position_count: int = 0,
        strategy_id: str = "main",
        extra_metadata: dict[str, Any] | None = None,
        paper_entry_estimate: PaperTradeEstimate | None = None,
        prebuilt_tx: Any = None,
        source_program: str | None = None,
    ) -> Optional[PositionRecord]:
        """Async version of execute_entry — uses non-blocking broadcast for live mode."""
        if self._is_live:
            return await self._execute_live_entry_async(
                features,
                match,
                size_sol,
                current_exposure_sol=current_exposure_sol,
                open_position_count=open_position_count,
                strategy_id=strategy_id,
                extra_metadata=extra_metadata,
                prebuilt_tx=prebuilt_tx,
                source_program=source_program,
            )
        return self.execute_paper_entry(
            features,
            match,
            size_sol,
            strategy_id=strategy_id,
            extra_metadata=extra_metadata,
            paper_entry_estimate=paper_entry_estimate,
        )

    # ------------------------------------------------------------------
    # Paper entry (unchanged logic)
    # ------------------------------------------------------------------

    def execute_paper_entry(
        self,
        features: RuntimeFeatures,
        match: MatchResult,
        size_sol: float,
        strategy_id: str = "main",
        extra_metadata: dict[str, Any] | None = None,
        paper_entry_estimate: PaperTradeEstimate | None = None,
    ) -> PositionRecord:
        """Persist one simulated entry."""
        assert match.selected_rule is not None
        logger.info(
            "📝 PAPER entry: %s | strategy=%s | rule=%s | size=%.4f SOL",
            features.token_mint[:12],
            strategy_id,
            match.selected_rule.rule_id,
            size_sol,
        )

        amount_received = (
            size_sol / features.entry_price_sol
            if features.entry_price_sol and features.entry_price_sol > 0
            else 0.0
        )
        entry_fee_sol = 0.0
        entry_quote_in_lamports = 0
        entry_quote_out_raw = 0
        entry_priority_fee_lamports = 0
        entry_jito_tip_lamports = 0
        paper_quote_error = None

        estimate = paper_entry_estimate
        if estimate is None and self.trade_executor is not None:
            estimate = self.trade_executor.simulate_paper_buy(
                token_mint=features.token_mint,
                size_sol=size_sol,
            )
        if estimate is not None:
            if estimate.success and estimate.out_amount > 0:
                amount_received = float(estimate.out_amount)
                entry_fee_sol = float(estimate.total_network_fee_lamports) / float(LAMPORTS_PER_SOL)
                entry_quote_in_lamports = int(estimate.in_amount)
                entry_quote_out_raw = int(estimate.out_amount)
                entry_priority_fee_lamports = int(estimate.priority_fee_lamports)
                entry_jito_tip_lamports = int(estimate.jito_tip_lamports)
            else:
                paper_quote_error = estimate.error

        # When no Jupiter quote is available, use the configured priority fee as a fee floor
        # instead of 0. This prevents paper cost-basis being systematically understated,
        # which was causing paper PnL to overstate live results by 3-8%.
        if entry_fee_sol == 0.0 and self.trade_executor is not None:
            config_priority = int(
                getattr(self.trade_executor.config, "priority_fee_lamports", 50_000)
            )
            entry_fee_sol = float(config_priority + 5_000) / float(LAMPORTS_PER_SOL)

        entry_cost_basis_sol = size_sol + entry_fee_sol
        metadata = {
            "strategy_id": strategy_id,
            "runtime_features": features.raw,
            "detected_regime": match.detected_regime,
            "exit_profile": match.selected_rule.exit_profile,
            "initial_size_sol": size_sol,
            "initial_amount_received": amount_received,
            "remaining_size_sol": size_sol,
            "remaining_amount_received": amount_received,
            "paper_exec_model": "jupiter_quote_plus_helius_fee",
            "paper_entry_quote_in_lamports": entry_quote_in_lamports,
            "paper_entry_quote_out_raw": entry_quote_out_raw,
            "paper_entry_fee_sol": entry_fee_sol,
            "paper_entry_priority_fee_lamports": entry_priority_fee_lamports,
            "paper_entry_jito_tip_lamports": entry_jito_tip_lamports,
            "paper_entry_cost_basis_sol": entry_cost_basis_sol,
            "paper_remaining_cost_basis_sol": entry_cost_basis_sol,
            "paper_remaining_token_raw": float(entry_quote_out_raw or amount_received),
            "paper_cumulative_fees_sol": entry_fee_sol,
            "hit_2x_achieved": False,
            "hit_5x_achieved": False,
            "last_pnl_multiple": 0.0,
            "last_price_sol_seen": features.entry_price_sol or 0.0,
            "last_token_update_at": features.entry_time.isoformat(),
        }
        if paper_quote_error:
            metadata["paper_quote_error"] = paper_quote_error
        if extra_metadata:
            metadata.update(extra_metadata)
        pipeline_latency_trace = _merge_latency_trace(
            features.raw.get("__latency_trace"),
            (extra_metadata or {}).get("pipeline_latency_trace"),
        )
        if pipeline_latency_trace:
            metadata["pipeline_latency_trace"] = pipeline_latency_trace
        observed_entry_price_sol = _coerce_float(features.entry_price_sol)
        observed_entry_price_raw_sol = _coerce_float(features.raw.get("last_price_sol_raw"))
        observed_entry_price_reliable_sol = _coerce_float(
            features.raw.get("last_price_sol_reliable")
        )
        quote_source = (
            "jupiter_quote"
            if estimate is not None and estimate.success and estimate.out_amount > 0
            else "mark_price_fallback"
        )
        metadata.update(
            {
                "entry_quote_source": quote_source,
                "entry_observed_price_sol": observed_entry_price_sol,
                "entry_observed_price_raw_sol": observed_entry_price_raw_sol,
                "entry_observed_price_reliable_sol": observed_entry_price_reliable_sol,
                "tracked_wallet_features_enabled": bool(
                    features.raw.get("tracked_wallet_features_enabled", True)
                ),
            }
        )

        position = PositionRecord(
            token_mint=features.token_mint,
            entry_time=features.entry_time,
            entry_price_sol=features.entry_price_sol or 0.0,
            size_sol=size_sol,
            amount_received=amount_received,
            strategy_id=strategy_id,
            selected_rule_id=match.selected_rule.rule_id,
            selected_regime=match.selected_rule.regime,
            matched_rule_ids=[rule.rule_id for rule in match.matched_rules],
            triggering_wallet=features.triggering_wallet,
            triggering_wallet_score=features.triggering_wallet_score,
            metadata=metadata,
        )
        position_id = self.position_manager.open_position(position)
        self.db.record_trade_leg(
            position_id=position_id,
            token_mint=features.token_mint,
            action="BUY",
            mode="paper",
            strategy_id=strategy_id,
            selected_rule_id=match.selected_rule.rule_id,
            selected_regime=match.selected_rule.regime,
            quote_used=quote_source != "mark_price_fallback",
            quote_source=quote_source,
            quote_error=paper_quote_error,
            observed_price_sol=observed_entry_price_sol,
            observed_price_raw_sol=observed_entry_price_raw_sol,
            observed_price_reliable_sol=observed_entry_price_reliable_sol,
            observed_pnl_multiple=0.0,
            executed_pnl_multiple=0.0,
            cost_basis_sol=entry_cost_basis_sol,
            leg_size_sol=size_sol,
            token_amount_raw=float(entry_quote_out_raw or amount_received),
            gross_sol=size_sol,
            net_sol=size_sol,
            fee_sol=entry_fee_sol,
            reason="entry",
            created_at=features.entry_time.isoformat(),
            metadata={
                "detected_regime": match.detected_regime,
                "matched_rule_ids": [rule.rule_id for rule in match.matched_rules],
                "paper_exec_model": metadata.get("paper_exec_model"),
                "pipeline_latency_trace": pipeline_latency_trace,
                "latency_trace": pipeline_latency_trace,
            },
        )
        self.rule_performance.record_entry(match.selected_rule.rule_id, match.selected_rule.regime)
        self.db.execute(
            """
            INSERT INTO executions (token_mint, action, mode, strategy_id, size_sol, price_sol, tx_signature, status, reason, created_at)
            VALUES (?, 'BUY', 'paper', ?, ?, ?, NULL, 'FILLED', NULL, ?)
            """,
            (
                features.token_mint,
                strategy_id,
                size_sol,
                features.entry_price_sol,
                features.entry_time.isoformat(),
            ),
        )
        self.event_log.log(
            "paper_entry",
            {
                "token_mint": features.token_mint,
                "strategy_id": strategy_id,
                "selected_rule_id": match.selected_rule.rule_id,
                "selected_regime": match.selected_rule.regime,
                "detected_regime": match.detected_regime,
                "matched_rule_ids": [rule.rule_id for rule in match.matched_rules],
                "size_sol": size_sol,
                "paper_entry_fee_sol": entry_fee_sol,
                "paper_entry_cost_basis_sol": entry_cost_basis_sol,
                "paper_entry_quote_out_raw": entry_quote_out_raw,
                "paper_entry_quote_source": quote_source,
                "entry_observed_price_sol": observed_entry_price_sol,
                "entry_observed_price_raw_sol": observed_entry_price_raw_sol,
                "entry_observed_price_reliable_sol": observed_entry_price_reliable_sol,
                "paper_quote_error": paper_quote_error,
                "latency_trace": pipeline_latency_trace,
            },
        )
        return position

    # ------------------------------------------------------------------
    # Live entry
    # ------------------------------------------------------------------

    def _execute_live_entry(
        self,
        features: RuntimeFeatures,
        match: MatchResult,
        size_sol: float,
        current_exposure_sol: float = 0.0,
        open_position_count: int = 0,
        strategy_id: str = "main",
        extra_metadata: dict[str, Any] | None = None,
        prebuilt_tx: Any = None,
        source_program: str | None = None,
    ) -> Optional[PositionRecord]:
        """Execute a real on-chain BUY and persist the position."""
        assert match.selected_rule is not None
        assert self.trade_executor is not None

        logger.info(
            "🔴 LIVE entry: %s | strategy=%s | rule=%s | size=%.4f SOL%s",
            features.token_mint[:12],
            strategy_id,
            match.selected_rule.rule_id,
            size_sol,
            " [prebuilt-tx]" if prebuilt_tx is not None else "",
        )

        if prebuilt_tx is not None:
            result = self.trade_executor.execute_buy_prebuilt(
                token_mint=features.token_mint,
                size_sol=size_sol,
                prebuilt_tx=prebuilt_tx,
                current_exposure_sol=current_exposure_sol,
                open_position_count=open_position_count,
            )
        else:
            result = self.trade_executor.execute_buy(
                token_mint=features.token_mint,
                size_sol=size_sol,
                current_exposure_sol=current_exposure_sol,
                open_position_count=open_position_count,
                strategy=strategy_id,
                source_program=source_program,
            )

        if not result.success:
            pipeline_latency_trace = _merge_latency_trace(
                features.raw.get("__latency_trace"),
                (extra_metadata or {}).get("pipeline_latency_trace"),
            )
            latency_trace = _merge_latency_trace(
                pipeline_latency_trace,
                getattr(result, "latency_trace", {}) or {},
            )
            fee_burn_sol = self._record_failed_live_entry_fee_burn(
                features=features,
                match=match,
                size_sol=size_sol,
                strategy_id=strategy_id,
                result=result,
                pipeline_latency_trace=pipeline_latency_trace,
                latency_trace=latency_trace,
            )
            logger.error(
                "🔴 LIVE entry FAILED for %s: %s",
                features.token_mint[:12],
                result.error,
            )
            execution_recorded_at = _resolve_live_recorded_at_iso(
                features.entry_time, latency_trace
            )
            self.db.execute(
                """
                INSERT INTO executions (token_mint, action, mode, strategy_id, size_sol, price_sol, tx_signature, status, reason, created_at)
                VALUES (?, 'BUY', 'live', ?, ?, ?, ?, 'FAILED', ?, ?)
                """,
                (
                    features.token_mint,
                    strategy_id,
                    size_sol,
                    features.entry_price_sol,
                    result.signature,
                    result.error,
                    execution_recorded_at,
                ),
            )
            self.event_log.log(
                "live_entry_failed",
                {
                    "token_mint": features.token_mint,
                    "strategy_id": strategy_id,
                    "selected_rule_id": match.selected_rule.rule_id,
                    "size_sol": size_sol,
                    "error": result.error,
                    "signature": result.signature,
                    "trigger_event_time": features.entry_time.isoformat(),
                    "execution_recorded_at": execution_recorded_at,
                    "pipeline_latency_trace": pipeline_latency_trace,
                    "execution_latency_trace": dict(getattr(result, "latency_trace", {}) or {}),
                    "latency_trace": latency_trace,
                    "reconciliation_error": getattr(result, "reconciliation_error", None),
                    "failed_fee_burn_sol": fee_burn_sol,
                },
            )
            return None

        (
            amount_received,
            live_fee_sol,
            live_entry_cost_basis,
            live_token_amount_raw,
            live_fill_details,
        ) = _live_entry_fill_fields(result, size_sol)
        if live_fee_sol <= 0.0:
            live_fee_sol = (
                float(
                    self.trade_executor.config.priority_fee_lamports
                    + getattr(self.trade_executor.config, "jito_tip_lamports", 0)
                    + 5_000
                )
                / 1_000_000_000.0
            )
        if live_entry_cost_basis <= 0.0:
            live_entry_cost_basis = size_sol + live_fee_sol

        metadata = {
            "strategy_id": strategy_id,
            "runtime_features": features.raw,
            "detected_regime": match.detected_regime,
            "exit_profile": match.selected_rule.exit_profile,
            "tx_signature": result.signature,
            "live_in_amount": result.in_amount,
            "live_out_amount": result.out_amount,
            "initial_size_sol": size_sol,
            "initial_amount_received": amount_received,
            "remaining_size_sol": size_sol,
            "remaining_amount_received": amount_received,
            # Mirror paper cost-basis fields so exit-engine and dashboard are
            # consistent between paper and live positions.
            "paper_entry_fee_sol": live_fee_sol,
            "paper_entry_cost_basis_sol": live_entry_cost_basis,
            "paper_remaining_cost_basis_sol": live_entry_cost_basis,
            "paper_remaining_token_raw": float(live_token_amount_raw or amount_received),
            "paper_cumulative_fees_sol": live_fee_sol,
            "hit_2x_achieved": False,
            "hit_5x_achieved": False,
            "last_pnl_multiple": 0.0,
            "last_price_sol_seen": features.entry_price_sol or 0.0,
            "last_token_update_at": features.entry_time.isoformat(),
        }
        metadata.update(live_fill_details)
        if extra_metadata:
            metadata.update(extra_metadata)
        pipeline_latency_trace = _merge_latency_trace(
            features.raw.get("__latency_trace"),
            (extra_metadata or {}).get("pipeline_latency_trace"),
        )
        execution_latency_trace = dict(getattr(result, "latency_trace", {}) or {})
        latency_trace = _merge_latency_trace(
            pipeline_latency_trace,
            execution_latency_trace,
        )
        execution_recorded_at = _resolve_live_recorded_at_iso(
            features.entry_time,
            execution_latency_trace,
            latency_trace,
        )
        observed_entry_price_sol = _coerce_float(features.entry_price_sol)
        observed_entry_price_raw_sol = _coerce_float(features.raw.get("last_price_sol_raw"))
        observed_entry_price_reliable_sol = _coerce_float(
            features.raw.get("last_price_sol_reliable")
        )
        metadata.update(
            {
                "entry_quote_source": "live_fill",
                "entry_observed_price_sol": observed_entry_price_sol,
                "entry_observed_price_raw_sol": observed_entry_price_raw_sol,
                "entry_observed_price_reliable_sol": observed_entry_price_reliable_sol,
                "tracked_wallet_features_enabled": bool(
                    features.raw.get("tracked_wallet_features_enabled", True)
                ),
                "trigger_event_time": features.entry_time.isoformat(),
                "execution_recorded_at": execution_recorded_at,
            }
        )
        if pipeline_latency_trace:
            metadata["pipeline_latency_trace"] = pipeline_latency_trace
        metadata["live_latency_trace"] = latency_trace

        position = PositionRecord(
            token_mint=features.token_mint,
            entry_time=features.entry_time,
            entry_price_sol=features.entry_price_sol or 0.0,
            size_sol=size_sol,
            amount_received=amount_received,
            strategy_id=strategy_id,
            selected_rule_id=match.selected_rule.rule_id,
            selected_regime=match.selected_rule.regime,
            matched_rule_ids=[rule.rule_id for rule in match.matched_rules],
            triggering_wallet=features.triggering_wallet,
            triggering_wallet_score=features.triggering_wallet_score,
            metadata=metadata,
        )
        position_id = self.position_manager.open_position(position)
        self.db.record_trade_leg(
            position_id=position_id,
            token_mint=features.token_mint,
            action="BUY",
            mode="live",
            strategy_id=strategy_id,
            selected_rule_id=match.selected_rule.rule_id,
            selected_regime=match.selected_rule.regime,
            quote_used=True,
            quote_source="live_fill",
            observed_price_sol=observed_entry_price_sol,
            observed_price_raw_sol=observed_entry_price_raw_sol,
            observed_price_reliable_sol=observed_entry_price_reliable_sol,
            observed_pnl_multiple=0.0,
            executed_pnl_multiple=0.0,
            cost_basis_sol=live_entry_cost_basis,
            leg_size_sol=size_sol,
            token_amount_raw=float(live_token_amount_raw or amount_received),
            gross_sol=live_entry_cost_basis,
            net_sol=live_entry_cost_basis,
            fee_sol=live_fee_sol,
            reason="entry",
            tx_signature=result.signature,
            created_at=execution_recorded_at,
            metadata={
                "detected_regime": match.detected_regime,
                "matched_rule_ids": [rule.rule_id for rule in match.matched_rules],
                "trigger_event_time": features.entry_time.isoformat(),
                "execution_recorded_at": execution_recorded_at,
                "pipeline_latency_trace": pipeline_latency_trace,
                "execution_latency_trace": execution_latency_trace,
                "latency_trace": latency_trace,
                "reconciliation_error": getattr(result, "reconciliation_error", None),
            },
        )
        self.rule_performance.record_entry(match.selected_rule.rule_id, match.selected_rule.regime)
        self.db.execute(
            """
            INSERT INTO executions (token_mint, action, mode, strategy_id, size_sol, price_sol, tx_signature, status, reason, created_at)
            VALUES (?, 'BUY', 'live', ?, ?, ?, ?, 'FILLED', NULL, ?)
            """,
            (
                features.token_mint,
                strategy_id,
                size_sol,
                features.entry_price_sol,
                result.signature,
                execution_recorded_at,
            ),
        )
        self.event_log.log(
            "live_entry",
            {
                "token_mint": features.token_mint,
                "strategy_id": strategy_id,
                "selected_rule_id": match.selected_rule.rule_id,
                "selected_regime": match.selected_rule.regime,
                "detected_regime": match.detected_regime,
                "matched_rule_ids": [rule.rule_id for rule in match.matched_rules],
                "size_sol": size_sol,
                "signature": result.signature,
                "trigger_event_time": features.entry_time.isoformat(),
                "execution_recorded_at": execution_recorded_at,
                "in_amount": result.in_amount,
                "out_amount": result.out_amount,
                "fill_slot": result.slot,
                "pipeline_latency_trace": pipeline_latency_trace,
                "execution_latency_trace": execution_latency_trace,
                "latency_trace": latency_trace,
                "reconciliation_error": getattr(result, "reconciliation_error", None),
                "actual_token_delta_raw": metadata.get("live_token_delta_raw"),
                "actual_wallet_delta_lamports": metadata.get("live_wallet_delta_lamports"),
                "exact_fee_lamports": metadata.get("live_exact_fee_lamports"),
            },
        )
        logger.info(
            "✅ LIVE entry persisted: %s | sig=%s | tokens=%s",
            features.token_mint[:12],
            result.signature,
            result.out_amount,
        )
        return position

    async def _execute_live_entry_async(
        self,
        features: RuntimeFeatures,
        match: MatchResult,
        size_sol: float,
        current_exposure_sol: float = 0.0,
        open_position_count: int = 0,
        strategy_id: str = "main",
        extra_metadata: dict[str, Any] | None = None,
        prebuilt_tx: Any = None,
        source_program: str | None = None,
    ) -> Optional[PositionRecord]:
        """Async live BUY: uses broadcast_async so the event loop is not blocked."""
        assert match.selected_rule is not None
        assert self.trade_executor is not None

        logger.info(
            "🔴 LIVE entry (async): %s | strategy=%s | rule=%s | size=%.4f SOL%s",
            features.token_mint[:12],
            strategy_id,
            match.selected_rule.rule_id,
            size_sol,
            " [prebuilt-tx]" if prebuilt_tx is not None else "",
        )

        if prebuilt_tx is not None:
            result = await self.trade_executor.execute_buy_prebuilt_async(
                token_mint=features.token_mint,
                size_sol=size_sol,
                prebuilt_tx=prebuilt_tx,
                current_exposure_sol=current_exposure_sol,
                open_position_count=open_position_count,
            )
        else:
            # Async fallback: get_order_async → sign → broadcast_async.
            # Does NOT use the thread pool — the event loop awaits each step
            # directly, keeping threads free for exit processing.
            result = await self.trade_executor.execute_buy_async(
                token_mint=features.token_mint,
                size_sol=size_sol,
                current_exposure_sol=current_exposure_sol,
                open_position_count=open_position_count,
                prefer_jupiter=strategy_id in ("main", "wallet"),
                strategy=strategy_id,
                source_program=source_program,
            )

        if not result.success:
            pipeline_latency_trace = _merge_latency_trace(
                features.raw.get("__latency_trace"),
                (extra_metadata or {}).get("pipeline_latency_trace"),
            )
            latency_trace = _merge_latency_trace(
                pipeline_latency_trace,
                getattr(result, "latency_trace", {}) or {},
            )
            fee_burn_sol = self._record_failed_live_entry_fee_burn(
                features=features,
                match=match,
                size_sol=size_sol,
                strategy_id=strategy_id,
                result=result,
                pipeline_latency_trace=pipeline_latency_trace,
                latency_trace=latency_trace,
            )
            logger.error(
                "🔴 LIVE entry (async) FAILED for %s: %s",
                features.token_mint[:12],
                result.error,
            )
            execution_recorded_at = _resolve_live_recorded_at_iso(
                features.entry_time, latency_trace
            )
            self.db.execute(
                """
                INSERT INTO executions (token_mint, action, mode, strategy_id, size_sol, price_sol, tx_signature, status, reason, created_at)
                VALUES (?, 'BUY', 'live', ?, ?, ?, ?, 'FAILED', ?, ?)
                """,
                (
                    features.token_mint,
                    strategy_id,
                    size_sol,
                    features.entry_price_sol,
                    result.signature,
                    result.error,
                    execution_recorded_at,
                ),
            )
            self.event_log.log(
                "live_entry_failed",
                {
                    "token_mint": features.token_mint,
                    "strategy_id": strategy_id,
                    "selected_rule_id": match.selected_rule.rule_id,
                    "size_sol": size_sol,
                    "error": result.error,
                    "signature": result.signature,
                    "trigger_event_time": features.entry_time.isoformat(),
                    "execution_recorded_at": execution_recorded_at,
                    "pipeline_latency_trace": pipeline_latency_trace,
                    "execution_latency_trace": dict(getattr(result, "latency_trace", {}) or {}),
                    "latency_trace": latency_trace,
                    "reconciliation_error": getattr(result, "reconciliation_error", None),
                    "failed_fee_burn_sol": fee_burn_sol,
                },
            )
            return None

        (
            amount_received,
            live_fee_sol,
            live_entry_cost_basis,
            live_token_amount_raw,
            live_fill_details,
        ) = _live_entry_fill_fields(result, size_sol)
        if live_fee_sol <= 0.0:
            live_fee_sol = (
                float(
                    self.trade_executor.config.priority_fee_lamports
                    + getattr(self.trade_executor.config, "jito_tip_lamports", 0)
                    + 5_000
                )
                / 1_000_000_000.0
            )
        if live_entry_cost_basis <= 0.0:
            live_entry_cost_basis = size_sol + live_fee_sol

        metadata = {
            "strategy_id": strategy_id,
            "runtime_features": features.raw,
            "detected_regime": match.detected_regime,
            "exit_profile": match.selected_rule.exit_profile,
            "tx_signature": result.signature,
            "live_in_amount": result.in_amount,
            "live_out_amount": result.out_amount,
            "initial_size_sol": size_sol,
            "initial_amount_received": amount_received,
            "remaining_size_sol": size_sol,
            "remaining_amount_received": amount_received,
            "paper_entry_fee_sol": live_fee_sol,
            "paper_entry_cost_basis_sol": live_entry_cost_basis,
            "paper_remaining_cost_basis_sol": live_entry_cost_basis,
            "paper_remaining_token_raw": float(live_token_amount_raw or amount_received),
            "paper_cumulative_fees_sol": live_fee_sol,
            "hit_2x_achieved": False,
            "hit_5x_achieved": False,
            "last_pnl_multiple": 0.0,
            "last_price_sol_seen": features.entry_price_sol or 0.0,
            "last_token_update_at": features.entry_time.isoformat(),
        }
        metadata.update(live_fill_details)
        if extra_metadata:
            metadata.update(extra_metadata)
        pipeline_latency_trace = _merge_latency_trace(
            features.raw.get("__latency_trace"),
            (extra_metadata or {}).get("pipeline_latency_trace"),
        )
        execution_latency_trace = dict(getattr(result, "latency_trace", {}) or {})
        latency_trace = _merge_latency_trace(
            pipeline_latency_trace,
            execution_latency_trace,
        )
        execution_recorded_at = _resolve_live_recorded_at_iso(
            features.entry_time,
            execution_latency_trace,
            latency_trace,
        )
        observed_entry_price_sol = _coerce_float(features.entry_price_sol)
        observed_entry_price_raw_sol = _coerce_float(features.raw.get("last_price_sol_raw"))
        observed_entry_price_reliable_sol = _coerce_float(
            features.raw.get("last_price_sol_reliable")
        )
        metadata.update(
            {
                "entry_quote_source": "live_fill",
                "entry_observed_price_sol": observed_entry_price_sol,
                "entry_observed_price_raw_sol": observed_entry_price_raw_sol,
                "entry_observed_price_reliable_sol": observed_entry_price_reliable_sol,
                "tracked_wallet_features_enabled": bool(
                    features.raw.get("tracked_wallet_features_enabled", True)
                ),
                "trigger_event_time": features.entry_time.isoformat(),
                "execution_recorded_at": execution_recorded_at,
            }
        )
        if pipeline_latency_trace:
            metadata["pipeline_latency_trace"] = pipeline_latency_trace
        metadata["live_latency_trace"] = latency_trace

        position = PositionRecord(
            token_mint=features.token_mint,
            entry_time=features.entry_time,
            entry_price_sol=features.entry_price_sol or 0.0,
            size_sol=size_sol,
            amount_received=amount_received,
            strategy_id=strategy_id,
            selected_rule_id=match.selected_rule.rule_id,
            selected_regime=match.selected_rule.regime,
            matched_rule_ids=[rule.rule_id for rule in match.matched_rules],
            triggering_wallet=features.triggering_wallet,
            triggering_wallet_score=features.triggering_wallet_score,
            metadata=metadata,
        )
        position_id = self.position_manager.open_position(position)
        self.db.record_trade_leg(
            position_id=position_id,
            token_mint=features.token_mint,
            action="BUY",
            mode="live",
            strategy_id=strategy_id,
            selected_rule_id=match.selected_rule.rule_id,
            selected_regime=match.selected_rule.regime,
            quote_used=True,
            quote_source="live_fill",
            observed_price_sol=observed_entry_price_sol,
            observed_price_raw_sol=observed_entry_price_raw_sol,
            observed_price_reliable_sol=observed_entry_price_reliable_sol,
            observed_pnl_multiple=0.0,
            executed_pnl_multiple=0.0,
            cost_basis_sol=live_entry_cost_basis,
            leg_size_sol=size_sol,
            token_amount_raw=float(live_token_amount_raw or amount_received),
            gross_sol=live_entry_cost_basis,
            net_sol=live_entry_cost_basis,
            fee_sol=live_fee_sol,
            reason="entry",
            tx_signature=result.signature,
            created_at=execution_recorded_at,
            metadata={
                "detected_regime": match.detected_regime,
                "matched_rule_ids": [rule.rule_id for rule in match.matched_rules],
                "trigger_event_time": features.entry_time.isoformat(),
                "execution_recorded_at": execution_recorded_at,
                "pipeline_latency_trace": pipeline_latency_trace,
                "execution_latency_trace": execution_latency_trace,
                "latency_trace": latency_trace,
                "reconciliation_error": getattr(result, "reconciliation_error", None),
            },
        )
        self.rule_performance.record_entry(match.selected_rule.rule_id, match.selected_rule.regime)
        self.db.execute(
            """
            INSERT INTO executions (token_mint, action, mode, strategy_id, size_sol, price_sol, tx_signature, status, reason, created_at)
            VALUES (?, 'BUY', 'live', ?, ?, ?, ?, 'FILLED', NULL, ?)
            """,
            (
                features.token_mint,
                strategy_id,
                size_sol,
                features.entry_price_sol,
                result.signature,
                execution_recorded_at,
            ),
        )
        self.event_log.log(
            "live_entry",
            {
                "token_mint": features.token_mint,
                "strategy_id": strategy_id,
                "selected_rule_id": match.selected_rule.rule_id,
                "selected_regime": match.selected_rule.regime,
                "detected_regime": match.detected_regime,
                "matched_rule_ids": [rule.rule_id for rule in match.matched_rules],
                "size_sol": size_sol,
                "signature": result.signature,
                "trigger_event_time": features.entry_time.isoformat(),
                "execution_recorded_at": execution_recorded_at,
                "in_amount": result.in_amount,
                "out_amount": result.out_amount,
                "fill_slot": result.slot,
                "pipeline_latency_trace": pipeline_latency_trace,
                "execution_latency_trace": execution_latency_trace,
                "latency_trace": latency_trace,
                "reconciliation_error": getattr(result, "reconciliation_error", None),
                "actual_token_delta_raw": metadata.get("live_token_delta_raw"),
                "actual_wallet_delta_lamports": metadata.get("live_wallet_delta_lamports"),
                "exact_fee_lamports": metadata.get("live_exact_fee_lamports"),
            },
        )
        logger.info(
            "✅ LIVE entry (async) persisted: %s | sig=%s | tokens=%s",
            features.token_mint[:12],
            result.signature,
            result.out_amount,
        )
        return position
