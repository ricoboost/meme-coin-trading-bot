"""Shared dataclasses for the Phase 2 paper bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class RuntimeRule:
    """Normalized runtime rule loaded from Phase 1 research artifacts."""

    rule_id: str
    regime: str
    support: int
    hit_2x_rate: float
    hit_5x_rate: float
    rug_rate: float
    priority: int
    enabled: bool
    score_weight: float
    conditions: dict[str, Any]
    exit_profile: str
    source: str


@dataclass
class CandidateEvent:
    """Token event evaluated by the strategy engine."""

    token_mint: str
    signature: str
    block_time: datetime
    triggering_wallet: str
    side: str
    sol_amount: float
    token_amount: float
    reference_price_sol: float | None = None
    source_program: str | None = None
    tracked_wallets: tuple[str, ...] = ()
    discovery_source: str | None = None
    provider_created_at: datetime | None = None
    stream_received_at: datetime | None = None
    parse_started_at: datetime | None = None
    parse_completed_at: datetime | None = None
    source_slot: int | None = None
    event_time_source: str | None = None


@dataclass
class RuntimeFeatures:
    """Runtime feature snapshot used for regime and rule evaluation."""

    token_mint: str
    entry_time: datetime
    entry_price_sol: float | None
    token_age_sec: float | None
    wallet_cluster_30s: int
    wallet_cluster_120s: int
    volume_sol_30s: float
    volume_sol_60s: float
    tx_count_30s: int
    tx_count_60s: int
    price_change_30s: float | None
    price_change_60s: float | None
    triggering_wallet: str
    triggering_wallet_score: float
    aggregated_wallet_score: float
    tracked_wallet_present_60s: bool
    tracked_wallet_count_60s: int
    tracked_wallet_score_sum_60s: float
    buy_volume_sol_30s: float = 0.0
    buy_volume_sol_60s: float = 0.0
    sell_volume_sol_30s: float = 0.0
    sell_volume_sol_60s: float = 0.0
    buy_tx_count_30s: int = 0
    buy_tx_count_60s: int = 0
    sell_tx_count_30s: int = 0
    sell_tx_count_60s: int = 0
    buy_sell_ratio_30s: float | None = None
    buy_sell_ratio_60s: float | None = None
    net_flow_sol_30s: float = 0.0
    net_flow_sol_60s: float = 0.0
    avg_trade_sol_30s: float = 0.0
    avg_trade_sol_60s: float = 0.0
    round_trip_wallet_count_30s: int = 0
    round_trip_wallet_count_60s: int = 0
    round_trip_wallet_ratio_30s: float = 0.0
    round_trip_wallet_ratio_60s: float = 0.0
    round_trip_volume_sol_30s: float = 0.0
    round_trip_volume_sol_60s: float = 0.0
    real_volume_sol_30s: float = 0.0
    real_volume_sol_60s: float = 0.0
    real_buy_volume_sol_30s: float = 0.0
    real_buy_volume_sol_60s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchResult:
    """Result of rule matching and selection."""

    detected_regime: str
    matched_rules: list[RuntimeRule]
    selected_rule: RuntimeRule | None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class ExitMLDecision:
    """Decision from the exit ML predictor for an open position."""

    exit_now: bool  # True = model recommends immediate early exit
    hold_probability: float  # [0, 1] probability that holding leads to better outcome
    mode: str  # "off" | "shadow" | "gate"
    model_ready: bool  # False = model not yet trained, decision is neutral
    reason: str  # e.g. "exit_ml_early_exit", "exit_ml_hold", "exit_ml_shadow"
    strategy_id: str


@dataclass
class PositionRecord:
    """In-memory representation of a paper position."""

    token_mint: str
    entry_time: datetime
    entry_price_sol: float
    size_sol: float
    amount_received: float
    strategy_id: str
    selected_rule_id: str
    selected_regime: str
    matched_rule_ids: list[str]
    triggering_wallet: str
    triggering_wallet_score: float
    status: str = "OPEN"
    realized_pnl_sol: float = 0.0
    unrealized_pnl_sol: float = 0.0
    exit_stage: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
