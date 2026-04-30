"""Small deterministic scoring helpers."""

from __future__ import annotations

from src.bot.models import RuntimeFeatures, RuntimeRule


def selection_score(
    rule: RuntimeRule,
    regime_match: bool,
    features: RuntimeFeatures,
) -> tuple[float, float, float, float, float, float, int, int]:
    """Sort key for choosing among matched rules."""
    source = str(rule.source or "")
    if ":relaxed" in source:
        strict_rule = 0.0
    elif ":pair_adapted" in source:
        strict_rule = 0.5
    else:
        strict_rule = 1.0
    return (
        1.0 if regime_match else 0.0,
        strict_rule,
        1.0 if features.tracked_wallet_present_60s else 0.0,
        float(features.tracked_wallet_count_60s),
        float(features.tracked_wallet_score_sum_60s),
        -rule.rug_rate,
        rule.hit_5x_rate,
        rule.support,
        -rule.priority,
    )
