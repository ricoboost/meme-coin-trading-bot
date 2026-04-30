"""Match runtime features against active rules.

Matcher is schema-driven: each rule condition key is parsed as
``{feature_name}_{min|max}`` and applied to the corresponding feature read
from ``RuntimeFeatures`` (attribute first, then ``features.raw`` dict).
Adding a new feature to the runtime snapshot automatically makes it
available as a rule threshold — no matcher edits required.

Two overrides preserve behavior that doesn't fit the generic pattern:

* ``virtual_sol_growth_60s_min`` falls back to ``net_flow_sol_60s`` as a
  proxy when the raw Pump virtual-reserve delta isn't available.
* ``unique_buyers_30s`` is aliased to ``wallet_cluster_30s`` — same
  quantity under two names used by different rule packs.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from src.bot.models import RuntimeFeatures, RuntimeRule

# Condition stem → real feature name. Used when a rule pack names a feature
# under a synonym that points at an existing runtime field.
_CONDITION_ALIASES: dict[str, str] = {
    "unique_buyers_30s": "wallet_cluster_30s",
}

# Extra raw-snapshot keys populated by monitoring/token_activity.py that aren't
# RuntimeFeatures attributes. Rules can threshold against these via
# `features.raw`. Kept explicit so rules_loader can fail-close against typos.
_EXTRA_RAW_FEATURES: frozenset[str] = frozenset(
    {
        "trade_size_gini_30s",
        "trade_size_gini_60s",
        "inter_arrival_cv_30s",
        "inter_arrival_cv_60s",
        "max_consecutive_buy_streak_30s",
        "max_consecutive_buy_streak_60s",
        "buy_streak_count_30s",
        "buy_streak_count_60s",
        "tracked_wallet_cluster_30s",
        "tracked_wallet_cluster_120s",
        "tracked_wallet_cluster_300s",
        "last_price_sol",
        "last_price_sol_raw",
        "last_price_sol_reliable",
        "virtual_sol_growth_60s",
        "swaps_to_1_sol",
        "swaps_to_5_sol",
        "swaps_to_10_sol",
        "swaps_to_30_sol",
        "launcher_launches",
        "launcher_graduations",
        "launcher_graduation_ratio",
    }
)

KNOWN_FEATURES: frozenset[str] = frozenset(
    {f.name for f in dataclasses.fields(RuntimeFeatures) if f.name != "raw"} | _EXTRA_RAW_FEATURES
)


def _resolve_feature(features: RuntimeFeatures, name: str) -> Any:
    """Return the feature value, preferring dataclass attributes over raw dict."""
    value = getattr(features, name, None)
    if value is not None:
        return value
    return features.raw.get(name)


def _fmt(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _parse_op(cond_key: str) -> tuple[str, str] | None:
    """Split a condition key into (feature_name, op) or return None."""
    for op in ("_min", "_max"):
        if cond_key.endswith(op):
            return cond_key[: -len(op)], op[1:]
    return None


def _special_virtual_sol_growth_60s_min(features: RuntimeFeatures, threshold: Any) -> str | None:
    growth = features.raw.get("virtual_sol_growth_60s")
    if growth is None:
        # Runtime stream doesn't expose Pump virtual reserves yet. Use net_flow
        # as a deterministic proxy until reserve parsing lands.
        growth = features.net_flow_sol_60s
        metric_name = "net_flow_sol_60s_proxy"
    else:
        metric_name = "virtual_sol_growth_60s"
    try:
        if float(growth) < float(threshold):
            return f"{metric_name}<{_fmt(threshold)} ({_fmt(growth)})"
    except (TypeError, ValueError):
        return f"{metric_name}_invalid"
    return None


_SPECIAL_HANDLERS: dict[str, Callable[[RuntimeFeatures, Any], str | None]] = {
    "virtual_sol_growth_60s_min": _special_virtual_sol_growth_60s_min,
}


def validate_rule_conditions(conditions: dict[str, Any]) -> list[str]:
    """Return condition keys the matcher won't understand.

    Fail-closed helper for the rules loader: any key that isn't a registered
    special-handler key AND doesn't parse as ``{known_feature}_{min|max}``
    (after alias resolution) is unknown. The loader drops these rules so a
    CSV typo never silently produces an un-matchable rule at runtime.
    """
    unknown: list[str] = []
    for cond_key in conditions:
        if cond_key in _SPECIAL_HANDLERS:
            continue
        parsed = _parse_op(cond_key)
        if parsed is None:
            unknown.append(cond_key)
            continue
        raw_name, _ = parsed
        feat_name = _CONDITION_ALIASES.get(raw_name, raw_name)
        if feat_name not in KNOWN_FEATURES:
            unknown.append(cond_key)
    return unknown


def matches_rule(features: RuntimeFeatures, rule: RuntimeRule) -> bool:
    """Return whether the runtime features satisfy one rule."""
    return not rule_miss_reasons(features, rule)


def rule_miss_reasons(features: RuntimeFeatures, rule: RuntimeRule) -> list[str]:
    """Return human-readable reasons why a rule did not match."""
    misses: list[str] = []
    for cond_key, threshold in rule.conditions.items():
        if threshold is None:
            continue

        if cond_key in _SPECIAL_HANDLERS:
            miss = _SPECIAL_HANDLERS[cond_key](features, threshold)
            if miss is not None:
                misses.append(miss)
            continue

        parsed = _parse_op(cond_key)
        if parsed is None:
            # Unrecognized operator form. rules_loader.validate_rule_conditions
            # fail-closes against this at load time; ignoring here keeps older
            # clients that may still load raw CSVs from crashing.
            continue

        raw_name, op = parsed
        feat_name = _CONDITION_ALIASES.get(raw_name, raw_name)
        value = _resolve_feature(features, feat_name)
        if value is None:
            direction = ">=" if op == "min" else "<="
            misses.append(f"{raw_name}_missing (needs{direction}{_fmt(threshold)})")
            continue
        try:
            fv = float(value)
            ft = float(threshold)
        except (TypeError, ValueError):
            misses.append(f"{raw_name}_invalid")
            continue
        if op == "min" and fv < ft:
            misses.append(f"{raw_name}<{_fmt(threshold)} ({_fmt(value)})")
        elif op == "max" and fv > ft:
            misses.append(f"{raw_name}>{_fmt(threshold)} ({_fmt(value)})")
    return misses


def closest_rule_misses(
    features: RuntimeFeatures,
    rules: list[RuntimeRule],
    detected_regime: str,
    limit: int = 3,
) -> list[dict[str, object]]:
    """Return the nearest non-matching rules with miss details."""
    candidates: list[tuple[int, int, int, str, RuntimeRule, list[str]]] = []
    for rule in rules:
        if not rule.enabled:
            continue
        misses = rule_miss_reasons(features, rule)
        if not misses:
            continue
        regime_penalty = 0 if rule.regime == detected_regime else 1
        candidates.append((len(misses), regime_penalty, -rule.support, rule.rule_id, rule, misses))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [
        {
            "rule_id": rule.rule_id,
            "regime": rule.regime,
            "support": rule.support,
            "miss_count": miss_count,
            "miss_reasons": misses,
        }
        for miss_count, _, _, _, rule, misses in candidates[:limit]
    ]
