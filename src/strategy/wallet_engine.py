"""Wallet-cluster lane: entry triggered by tracked provider_a wallets, not rules."""

from __future__ import annotations

from datetime import datetime, timezone

from src.bot.config import BotConfig
from src.bot.models import MatchResult, RuntimeFeatures, RuntimeRule


class WalletEngine:
    """Entry gate driven by tracked-wallet cluster activity."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.enabled = bool(config.enable_wallet_strategy)
        self.copytrading_enabled = (
            bool(getattr(config, "wallet_copytrading_enabled", False)) and self.enabled
        )
        self.rule = RuntimeRule(
            rule_id="wallet_cluster_v1",
            regime="wallet",
            support=0,
            hit_2x_rate=0.0,
            hit_5x_rate=0.0,
            rug_rate=0.0,
            priority=1,
            enabled=self.enabled,
            score_weight=0.0,
            conditions={},
            exit_profile="default_wallet",
            source="wallet",
        )
        self.copy_rule = RuntimeRule(
            rule_id="wallet_copy_v1",
            regime="wallet",
            support=0,
            hit_2x_rate=0.0,
            hit_5x_rate=0.0,
            rug_rate=0.0,
            priority=1,
            enabled=self.copytrading_enabled,
            score_weight=0.0,
            conditions={},
            exit_profile="wallet_copy",
            source="wallet",
        )

    def proposed_size_sol(self) -> float:
        return max(
            0.0,
            min(
                float(self.config.wallet_position_sol),
                float(self.config.max_position_sol),
            ),
        )

    def copy_proposed_size_sol(self) -> float:
        return max(
            0.0,
            min(
                float(self.config.wallet_copy_position_sol),
                float(self.config.max_position_sol),
            ),
        )

    def copy_wallet_qualifies(
        self,
        wallet: str,
        score: float,
        *,
        in_pool: bool = True,
    ) -> bool:
        if not self.copytrading_enabled:
            return False
        if not wallet:
            return False
        if not in_pool:
            return False
        min_score = float(self.config.wallet_copy_min_wallet_score)
        return float(score or 0.0) >= min_score

    def copy_entry_failures(
        self,
        features: RuntimeFeatures,
        *,
        triggering_wallet: str,
        triggering_wallet_score: float,
        event_block_time: datetime | None,
        now: datetime | None = None,
    ) -> list[str]:
        """Copy-mode gate: only freshness + age + wallet-score checks."""
        failures: list[str] = []
        if not self.copytrading_enabled:
            failures.append("copytrading_disabled")
            return failures

        if not self.copy_wallet_qualifies(triggering_wallet, triggering_wallet_score):
            failures.append(
                f"copy_wallet_score<{float(self.config.wallet_copy_min_wallet_score):.2f} "
                f"({float(triggering_wallet_score or 0.0):.2f})"
            )

        if features.token_age_sec is None:
            failures.append("token_age_sec_missing")
        else:
            age_max = float(self.config.wallet_copy_max_token_age_sec)
            if features.token_age_sec > age_max:
                failures.append(f"token_age_sec>{age_max:.0f} ({features.token_age_sec:.1f})")

        max_event_age = float(self.config.wallet_copy_event_max_age_sec)
        if max_event_age > 0 and event_block_time is not None:
            reference = now or datetime.now(tz=timezone.utc)
            if event_block_time.tzinfo is None:
                event_block_time = event_block_time.replace(tzinfo=timezone.utc)
            age = (reference - event_block_time).total_seconds()
            if age > max_event_age:
                failures.append(f"event_age_sec>{max_event_age:.0f} ({age:.1f})")

        return failures

    def entry_failures(self, features: RuntimeFeatures) -> list[str]:
        failures: list[str] = []
        if not self.enabled:
            failures.append("wallet_disabled")
            return failures

        tracked_cluster = _tracked_cluster_300s(features)
        min_cluster = int(self.config.wallet_min_cluster_300s)
        if tracked_cluster < min_cluster:
            failures.append(f"tracked_wallet_cluster_300s<{min_cluster} ({tracked_cluster})")

        min_buys_90s = int(getattr(self.config, "wallet_min_buys_90s", 0))
        if min_buys_90s > 0:
            buys_90s = _tracked_buys_90s(features)
            if buys_90s < min_buys_90s:
                failures.append(f"tracked_wallet_buys_90s<{min_buys_90s} ({buys_90s})")

        min_score_sum = float(self.config.wallet_min_wallet_score_sum)
        if min_score_sum > 0.0:
            score_sum = float(features.tracked_wallet_score_sum_60s or 0.0)
            if score_sum < min_score_sum:
                failures.append(
                    f"tracked_wallet_score_sum_60s<{min_score_sum:.2f} ({score_sum:.2f})"
                )

        if features.token_age_sec is None:
            failures.append("token_age_sec_missing")
        else:
            age_min = float(self.config.wallet_min_token_age_sec)
            age_max = float(self.config.wallet_max_token_age_sec)
            if features.token_age_sec < age_min:
                failures.append(f"token_age_sec<{age_min:.0f} ({features.token_age_sec:.1f})")
            if features.token_age_sec > age_max:
                failures.append(f"token_age_sec>{age_max:.0f} ({features.token_age_sec:.1f})")

        if features.price_change_30s is not None:
            max_pchg = float(self.config.wallet_max_price_change_30s)
            if features.price_change_30s > max_pchg:
                failures.append(
                    f"price_change_30s>{max_pchg:.4f} ({features.price_change_30s:.4f})"
                )

        if features.price_change_60s is not None:
            min_pchg_60 = float(self.config.wallet_min_price_change_60s)
            if features.price_change_60s < min_pchg_60:
                failures.append(
                    f"price_change_60s<{min_pchg_60:.4f} ({features.price_change_60s:.4f})"
                )

        min_net_flow_60 = float(self.config.wallet_min_net_flow_sol_60s)
        net_flow_60 = float(features.net_flow_sol_60s or 0.0)
        if net_flow_60 < min_net_flow_60:
            failures.append(f"net_flow_sol_60s<{min_net_flow_60:.2f} ({net_flow_60:.2f})")
        return failures

    def build_match(self, *, copy_mode: bool = False) -> MatchResult:
        rule = self.copy_rule if copy_mode else self.rule
        return MatchResult(
            detected_regime="wallet",
            matched_rules=[rule],
            selected_rule=rule,
            rejection_reason=None,
        )


def _tracked_cluster_300s(features: RuntimeFeatures) -> int:
    raw = features.raw or {}
    value = raw.get("tracked_wallet_cluster_300s")
    if value is None:
        value = features.tracked_wallet_count_60s
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _tracked_buys_90s(features: RuntimeFeatures) -> int:
    raw = features.raw or {}
    value = raw.get("tracked_wallet_buys_90s")
    if value is None:
        # Old snapshots missing the field — treat as unknown, don't block.
        return max(int(features.tracked_wallet_count_60s or 0), 1)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
