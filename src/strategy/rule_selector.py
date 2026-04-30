"""Select the best eligible rule when multiple match."""

from __future__ import annotations

from src.bot.models import MatchResult, RuntimeFeatures, RuntimeRule
from src.strategy.regime_detector import detect_regime
from src.strategy.rule_matcher import matches_rule
from src.strategy.scoring import selection_score


def select_rule(features: RuntimeFeatures, rules: list[RuntimeRule]) -> MatchResult:
    """Detect regime, match rules, and pick the best one."""
    regime = detect_regime(features)
    matched = [rule for rule in rules if rule.enabled and matches_rule(features, rule)]
    if not matched:
        return MatchResult(
            detected_regime=regime,
            matched_rules=[],
            selected_rule=None,
            rejection_reason="no_matching_rules",
        )
    matched.sort(
        key=lambda rule: selection_score(
            rule=rule,
            regime_match=rule.regime == regime,
            features=features,
        ),
        reverse=True,
    )
    return MatchResult(
        detected_regime=regime,
        matched_rules=matched,
        selected_rule=matched[0],
        rejection_reason=None,
    )
