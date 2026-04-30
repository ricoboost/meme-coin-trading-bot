"""Secondary ultra-short hold sniper strategy helpers."""

from __future__ import annotations

from dataclasses import replace

from src.bot.config import BotConfig
from src.bot.models import MatchResult, RuntimeFeatures, RuntimeRule
from src.strategy.rule_selector import select_rule


class SniperEngine:
    """Deterministic sniper gate + synthetic runtime rule."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.enabled = bool(config.enable_sniper_strategy)
        self.use_runtime_rules = bool(getattr(config, "sniper_use_runtime_rules", False))
        self.rule_id_allowlist = {
            str(rule_id).strip()
            for rule_id in getattr(config, "sniper_rule_ids", ())
            if str(rule_id).strip()
        }
        self.rule = RuntimeRule(
            rule_id="sniper_scalp_v1",
            regime="sniper",
            support=0,
            hit_2x_rate=0.0,
            hit_5x_rate=0.0,
            rug_rate=0.0,
            priority=1,
            enabled=self.enabled,
            score_weight=0.0,
            conditions={},
            exit_profile="default_sniper",
            source="sniper",
        )

    def _eligible_runtime_rules(self, rules: list[RuntimeRule]) -> list[RuntimeRule]:
        """Return enabled runtime rules optionally filtered by sniper allowlist."""
        enabled = [rule for rule in rules if rule.enabled]
        if self.rule_id_allowlist:
            enabled = [rule for rule in enabled if rule.rule_id in self.rule_id_allowlist]
        return enabled

    def _to_sniper_rule(self, rule: RuntimeRule) -> RuntimeRule:
        """Clone one runtime rule and force sniper exit profile semantics."""
        source = str(rule.source or "runtime")
        if ":sniper" not in source:
            source = f"{source}:sniper"
        return replace(rule, exit_profile="default_sniper", source=source)

    def proposed_size_sol(self) -> float:
        """Return configured sniper position size clamped by global max."""
        return max(
            0.0,
            min(
                float(self.config.sniper_position_sol),
                float(self.config.max_position_sol),
            ),
        )

    def entry_failures(self, features: RuntimeFeatures) -> list[str]:
        """Return sniper gate failures for one candidate."""
        failures: list[str] = []
        if not self.enabled:
            failures.append("sniper_disabled")
            return failures

        if features.token_age_sec is None:
            failures.append("token_age_sec_missing")
        else:
            if features.token_age_sec < float(self.config.sniper_min_token_age_sec):
                failures.append(
                    f"token_age_sec<{float(self.config.sniper_min_token_age_sec):.0f} ({features.token_age_sec:.1f})"
                )
            if features.token_age_sec > float(self.config.sniper_max_token_age_sec):
                failures.append(
                    f"token_age_sec>{float(self.config.sniper_max_token_age_sec):.0f} ({features.token_age_sec:.1f})"
                )

        if features.wallet_cluster_30s < int(self.config.sniper_min_cluster_30s):
            failures.append(
                f"wallet_cluster_30s<{int(self.config.sniper_min_cluster_30s)} ({features.wallet_cluster_30s})"
            )
        if features.tx_count_30s < int(self.config.sniper_min_tx_count_30s):
            failures.append(
                f"tx_count_30s<{int(self.config.sniper_min_tx_count_30s)} ({features.tx_count_30s})"
            )
        if features.volume_sol_30s < float(self.config.sniper_min_volume_sol_30s):
            failures.append(
                f"volume_sol_30s<{float(self.config.sniper_min_volume_sol_30s):.4f} ({features.volume_sol_30s:.4f})"
            )
        min_volume_per_tx = float(self.config.sniper_min_volume_per_tx_sol_30s)
        if features.tx_count_30s > 0:
            volume_per_tx = float(features.volume_sol_30s) / float(features.tx_count_30s)
            if volume_per_tx < min_volume_per_tx:
                failures.append(f"volume_per_tx_30s<{min_volume_per_tx:.4f} ({volume_per_tx:.4f})")
        if features.price_change_30s is None:
            failures.append("price_change_30s_missing")
        else:
            if features.price_change_30s < float(self.config.sniper_min_price_change_30s):
                failures.append(
                    f"price_change_30s<{float(self.config.sniper_min_price_change_30s):.4f} ({features.price_change_30s:.4f})"
                )
            if features.price_change_30s > float(self.config.sniper_max_price_change_30s):
                failures.append(
                    f"price_change_30s>{float(self.config.sniper_max_price_change_30s):.4f} ({features.price_change_30s:.4f})"
                )
        return failures

    def build_match(self) -> MatchResult:
        """Build a synthetic match result for entry engine compatibility."""
        return MatchResult(
            detected_regime="sniper",
            matched_rules=[self.rule],
            selected_rule=self.rule,
            rejection_reason=None,
        )

    def build_runtime_match(
        self, features: RuntimeFeatures, rules: list[RuntimeRule]
    ) -> MatchResult:
        """Build sniper match using loaded runtime rules."""
        candidate_rules = self._eligible_runtime_rules(rules)
        if not candidate_rules:
            return MatchResult(
                detected_regime="sniper",
                matched_rules=[],
                selected_rule=None,
                rejection_reason="sniper_no_runtime_rules",
            )

        base_match = select_rule(features, candidate_rules)
        if base_match.selected_rule is None:
            return MatchResult(
                detected_regime=base_match.detected_regime,
                matched_rules=[],
                selected_rule=None,
                rejection_reason=base_match.rejection_reason or "sniper_no_matching_runtime_rule",
            )

        selected_rule = self._to_sniper_rule(base_match.selected_rule)
        matched_rules = [self._to_sniper_rule(rule) for rule in base_match.matched_rules]
        return MatchResult(
            detected_regime=base_match.detected_regime,
            matched_rules=matched_rules,
            selected_rule=selected_rule,
            rejection_reason=None,
        )
