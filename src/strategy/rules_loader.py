"""Load runtime rules from Pump V2 artifacts (preferred) or legacy artifacts."""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.bot.config import BotConfig
from src.bot.models import RuntimeRule
from src.strategy.rule_matcher import validate_rule_conditions
from src.utils.io import read_json

_log = logging.getLogger(__name__)


def _parse_dict_like(value: Any) -> dict[str, Any]:
    """Parse one dict-like value from JSON/py-literal/dict input."""
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return dict(parsed)
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return dict(parsed)
        except (ValueError, SyntaxError):
            return {}
    return {}


def _to_float(value: Any, default: float | None = None) -> float | None:
    """Convert one value to float or return default."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    """Convert one value to int or return default."""
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def infer_regime(conditions: dict[str, Any]) -> str:
    """Infer regime label from conditions (legacy-oriented fallback)."""
    price_min = conditions.get("price_change_30s_min")
    price_max = conditions.get("price_change_30s_max")
    cluster_min = _to_int(conditions.get("wallet_cluster_30s_min"), 0)
    if price_min is None or price_max is None:
        return "unknown"

    price_min = float(price_min)
    price_max = float(price_max)
    if price_max <= -0.2 and cluster_min >= 3:
        return "high_cluster_recovery"
    if price_max <= -0.2:
        return "negative_shock_recovery"
    if price_min <= -0.2 and price_max >= 0.0 and cluster_min >= 3:
        return "high_cluster_recovery"
    if price_min < 0.0 < price_max and cluster_min >= 4:
        return "high_cluster_recovery"
    if price_min >= 0.2:
        return "momentum_burst"
    return "unknown"


def _infer_pump_regime(family: str, conditions: dict[str, Any]) -> str:
    """Infer regime for Pump V2 rows."""
    normalized = family.strip().lower()
    if "momentum" in normalized:
        return "momentum_burst"
    inferred = infer_regime(conditions)
    if inferred != "unknown":
        return inferred
    return "unknown"


def parse_legacy_conditions(rule_value: Any) -> dict[str, Any]:
    """Convert legacy CSV/JSON rule objects into normalized condition fields."""
    raw = _parse_dict_like(rule_value)
    if not raw:
        return {}
    price_range = raw.get("price_change_30s_between")
    if isinstance(price_range, (list, tuple)) and len(price_range) >= 2:
        price_min = price_range[0]
        price_max = price_range[1]
    else:
        price_min = raw.get("price_change_30s_min")
        price_max = raw.get("price_change_30s_max")
    return {
        "wallet_cluster_30s_min": raw.get(
            "wallet_cluster_30s_gte", raw.get("wallet_cluster_30s_min")
        ),
        "volume_sol_30s_min": raw.get("volume_sol_30s_gte", raw.get("volume_sol_30s_min")),
        "tx_count_30s_min": raw.get("tx_count_30s_gte", raw.get("tx_count_30s_min")),
        "token_age_sec_max": raw.get("token_age_sec_lte", raw.get("token_age_sec_max")),
        "price_change_30s_min": price_min,
        "price_change_30s_max": price_max,
    }


def parse_pump_conditions(row: pd.Series) -> dict[str, Any]:
    """Parse and normalize Pump V2 rule conditions from one CSV row."""
    raw: dict[str, Any] = {}
    for column in ("conditions_obj", "conditions_json", "conditions"):
        raw = _parse_dict_like(row.get(column))
        if raw:
            break
    if not raw:
        return {}

    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        numeric = _to_float(value, None)
        if numeric is None:
            continue
        normalized[key] = numeric
    return normalized


def derive_runtime_relaxed_rule(rule: RuntimeRule, config: BotConfig) -> RuntimeRule | None:
    """Build a controlled relaxed runtime variant of a strict legacy rule."""
    if not config.enable_runtime_rule_relaxation:
        return None
    if rule.regime not in {
        "negative_shock_recovery",
        "high_cluster_recovery",
        "momentum_burst",
    }:
        return None

    conditions = dict(rule.conditions)
    cluster_min = _to_int(conditions.get("wallet_cluster_30s_min"), 0)
    tx_count_min = _to_int(conditions.get("tx_count_30s_min"), 0)
    volume_min = _to_float(conditions.get("volume_sol_30s_min"), 0.0) or 0.0
    price_min = conditions.get("price_change_30s_min")
    price_max = conditions.get("price_change_30s_max")
    token_age_max = conditions.get("token_age_sec_max")

    if rule.regime == "negative_shock_recovery":
        conditions["wallet_cluster_30s_min"] = max(1, min(cluster_min, 2))
        conditions["tx_count_30s_min"] = max(1, min(tx_count_min, 2))
        conditions["volume_sol_30s_min"] = max(
            config.derived_rule_volume_floor,
            volume_min * config.derived_rule_volume_scale,
        )
        if price_min is not None:
            conditions["price_change_30s_min"] = min(float(price_min), -0.9)
        if token_age_max is not None:
            conditions["token_age_sec_max"] = min(max(float(token_age_max), 45.0), 180.0)
    elif rule.regime == "high_cluster_recovery":
        conditions["wallet_cluster_30s_min"] = max(2, min(cluster_min, 3))
        conditions["tx_count_30s_min"] = max(2, min(tx_count_min, 3))
        conditions["volume_sol_30s_min"] = max(
            max(config.derived_rule_volume_floor, 3.0),
            volume_min * max(config.derived_rule_volume_scale, 0.6),
        )
        if price_min is not None:
            conditions["price_change_30s_min"] = min(float(price_min), -0.9)
        if price_max is not None:
            conditions["price_change_30s_max"] = max(float(price_max), 0.22)
        else:
            conditions["price_change_30s_max"] = 0.22
        if token_age_max is not None:
            conditions["token_age_sec_max"] = min(max(float(token_age_max), 300.0), 1200.0)
        else:
            conditions["token_age_sec_max"] = 1200.0
    else:
        conditions["wallet_cluster_30s_min"] = max(3, min(cluster_min, 5))
        conditions["tx_count_30s_min"] = max(3, min(tx_count_min, 5))
        conditions["volume_sol_30s_min"] = max(
            max(config.derived_rule_volume_floor, 6.0),
            volume_min * 0.75,
        )
        if price_min is not None:
            conditions["price_change_30s_min"] = max(float(price_min) * 0.85, 0.12)
        else:
            conditions["price_change_30s_min"] = 0.12
        if price_max is not None:
            conditions["price_change_30s_max"] = max(min(float(price_max) * 1.1, 1.2), 0.55)
        else:
            conditions["price_change_30s_max"] = 0.9
        if token_age_max is not None:
            conditions["token_age_sec_max"] = min(max(float(token_age_max), 240.0), 1800.0)
        else:
            conditions["token_age_sec_max"] = 1800.0

    return RuntimeRule(
        rule_id=f"{rule.rule_id}_relaxed",
        regime=rule.regime,
        support=max(rule.support - 1, 1),
        hit_2x_rate=rule.hit_2x_rate * 0.9,
        hit_5x_rate=rule.hit_5x_rate * 0.9,
        rug_rate=min(rule.rug_rate * 1.15, 1.0),
        priority=rule.priority + 1000,
        enabled=rule.enabled,
        score_weight=rule.score_weight * 0.85,
        conditions=conditions,
        exit_profile=rule.exit_profile,
        source=f"{rule.source}:relaxed",
    )


def _scale_toward_zero(value: Any, scale: float) -> float | None:
    """Scale one signed threshold toward zero while preserving direction."""
    if value is None:
        return None
    return float(value) * float(scale)


def derive_pair_first_adapted_rule(rule: RuntimeRule, config: BotConfig) -> RuntimeRule | None:
    """Build a pair-first adapted variant from a strict legacy rule."""
    if config.discovery_mode != "pair_first":
        return None
    if not config.enable_pair_first_rule_adaptation:
        return None
    if "pair_adapted" in str(rule.source):
        return None
    if ":relaxed" in str(rule.source):
        return None
    if rule.regime not in {
        "negative_shock_recovery",
        "high_cluster_recovery",
        "momentum_burst",
    }:
        return None

    conditions = dict(rule.conditions)
    cluster_min = _to_int(conditions.get("wallet_cluster_30s_min"), 0)
    if cluster_min > 0:
        scaled_cluster = int(round(cluster_min * float(config.pair_first_cluster_scale)))
        conditions["wallet_cluster_30s_min"] = max(1, scaled_cluster)

    volume_min = _to_float(conditions.get("volume_sol_30s_min"), 0.0) or 0.0
    if volume_min > 0:
        conditions["volume_sol_30s_min"] = max(
            config.derived_rule_volume_floor,
            volume_min * float(config.pair_first_volume_scale),
        )

    token_age_max = conditions.get("token_age_sec_max")
    if token_age_max is not None:
        conditions["token_age_sec_max"] = max(
            float(token_age_max), float(config.pair_first_token_age_max_sec)
        )

    price_min = conditions.get("price_change_30s_min")
    price_max = conditions.get("price_change_30s_max")
    scaled_min = _scale_toward_zero(price_min, config.pair_first_price_scale)
    scaled_max = _scale_toward_zero(price_max, config.pair_first_price_scale)

    if rule.regime == "negative_shock_recovery":
        if scaled_max is None:
            scaled_max = -0.08
        scaled_max = min(float(scaled_max), -0.05)
        if scaled_min is None:
            scaled_min = scaled_max - 0.2
        if scaled_min > scaled_max:
            scaled_min = scaled_max - 0.1
    elif rule.regime == "high_cluster_recovery":
        if scaled_min is None:
            scaled_min = -0.12
        if scaled_max is None:
            scaled_max = 0.12
        scaled_min = min(float(scaled_min), -0.02)
        scaled_max = max(float(scaled_max), 0.22)
        scaled_max = min(float(scaled_max), 0.35)
        if scaled_min > scaled_max:
            scaled_min = scaled_max - 0.1
    elif rule.regime == "momentum_burst":
        if scaled_min is None:
            scaled_min = 0.08
        scaled_min = max(float(scaled_min), 0.05)
        if scaled_max is None:
            scaled_max = max(0.25, scaled_min + 0.2)
        if scaled_max < scaled_min:
            scaled_max = scaled_min + 0.1

    if scaled_min is not None:
        conditions["price_change_30s_min"] = float(scaled_min)
    if scaled_max is not None:
        conditions["price_change_30s_max"] = float(scaled_max)

    return RuntimeRule(
        rule_id=f"{rule.rule_id}_pair",
        regime=rule.regime,
        support=max(rule.support - 2, 1),
        hit_2x_rate=rule.hit_2x_rate * 0.9,
        hit_5x_rate=rule.hit_5x_rate * 0.9,
        rug_rate=min(rule.rug_rate * 1.1, 1.0),
        priority=rule.priority + 2000,
        enabled=rule.enabled,
        score_weight=rule.score_weight * 0.9,
        conditions=conditions,
        exit_profile=rule.exit_profile,
        source=f"{rule.source}:pair_adapted",
    )


def _load_rule_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        return pd.DataFrame(read_json(path))
    return pd.DataFrame()


def _strict_rule_sort_key(
    rule: RuntimeRule,
) -> tuple[float, float, float, float, int, int]:
    """Sort key for legacy strict rules."""
    return (
        -float(rule.score_weight),
        -float(rule.hit_2x_rate),
        -float(rule.hit_5x_rate),
        float(rule.rug_rate),
        -int(rule.support),
        int(rule.priority),
    )


def _pump_rule_sort_key(
    rule: RuntimeRule,
) -> tuple[float, float, float, float, int, int]:
    """Sort key for Pump V2 rules."""
    return (
        -float(rule.score_weight),
        -float(rule.hit_2x_rate),
        -float(rule.hit_5x_rate),
        float(rule.rug_rate),
        -int(rule.support),
        int(rule.priority),
    )


def _append_runtime_variants(
    rules: list[RuntimeRule], strict_rule: RuntimeRule, config: BotConfig
) -> None:
    """Append one strict legacy rule plus optional pair/relaxed variants."""
    rules.append(strict_rule)
    pair_adapted = derive_pair_first_adapted_rule(strict_rule, config)
    if pair_adapted is not None:
        rules.append(pair_adapted)
    relaxed = derive_runtime_relaxed_rule(strict_rule, config)
    if relaxed is not None:
        rules.append(relaxed)


def _apply_allowlist(rules: list[RuntimeRule], allowlist: tuple[str, ...]) -> list[RuntimeRule]:
    """Filter rules by regime allowlist."""
    if not allowlist:
        return list(rules)
    allowed = set(allowlist)
    return [rule for rule in rules if rule.regime in allowed]


def _load_legacy_rules(config: BotConfig) -> list[RuntimeRule]:
    """Load strict legacy rules from Phase 1 wallet-era artifacts."""
    shortlist = _load_rule_rows(config.trusted_rules_path)
    if shortlist.empty:
        shortlist = _load_rule_rows(config.top_rules_path)

    strict_rules: list[RuntimeRule] = []
    if shortlist.empty:
        strict_rules.append(
            RuntimeRule(
                rule_id="negative_shock_recovery_001",
                regime="negative_shock_recovery",
                support=165,
                hit_2x_rate=0.315,
                hit_5x_rate=0.103,
                rug_rate=0.048,
                priority=1,
                enabled=True,
                score_weight=0.48,
                conditions={
                    "wallet_cluster_30s_min": 2,
                    "volume_sol_30s_min": 5.0,
                    "tx_count_30s_min": 2,
                    "token_age_sec_max": 60,
                    "price_change_30s_min": -0.5,
                    "price_change_30s_max": -0.2,
                },
                exit_profile="default_recovery",
                source="fallback:legacy",
            )
        )
    else:
        for idx, row in shortlist.iterrows():
            support = _to_int(row.get("support"), 0)
            if support < config.min_rule_support:
                continue
            raw_rule = row.get("rule")
            if raw_rule is None:
                raw_rule = row.get("conditions")
            conditions = parse_legacy_conditions(raw_rule)
            if not conditions:
                continue
            regime = infer_regime(conditions)
            rule_id = str(row.get("rule_id") or f"{regime}_{idx + 1:03d}")
            exit_profile = (
                "default_recovery"
                if "recovery" in regime
                else ("default_momentum" if "momentum" in regime else "default_cluster")
            )
            strict_rules.append(
                RuntimeRule(
                    rule_id=rule_id,
                    regime=regime,
                    support=support,
                    hit_2x_rate=float(_to_float(row.get("hit_rate_2x_24h"), 0.0) or 0.0),
                    hit_5x_rate=float(_to_float(row.get("hit_rate_5x_24h"), 0.0) or 0.0),
                    rug_rate=float(_to_float(row.get("rug_rate_24h"), 1.0) or 1.0),
                    priority=idx + 1,
                    enabled=True,
                    score_weight=float(_to_float(row.get("rule_score"), 0.0) or 0.0),
                    conditions=conditions,
                    exit_profile=exit_profile,
                    source=str(config.trusted_rules_path),
                )
            )

    filtered = _apply_allowlist(strict_rules, config.optional_allowed_regimes)
    if filtered:
        strict_rules = filtered
    strict_limit = max(1, int(config.max_strict_rules))
    strict_rules = sorted(strict_rules, key=_strict_rule_sort_key)[:strict_limit]
    return strict_rules


def _parse_pump_format_rows(
    rows: pd.DataFrame,
    *,
    source_label: str,
    min_support: int,
    id_prefix: str,
    default_exit_profile: str = "default_momentum",
) -> list[RuntimeRule]:
    """Shared pump-format CSV → RuntimeRule parser used by both rule packs."""
    parsed: list[RuntimeRule] = []
    for idx, row in rows.iterrows():
        conditions = parse_pump_conditions(row)
        if not conditions:
            continue
        support = _to_int(row.get("support_valid"), _to_int(row.get("support"), 0))
        if support < min_support:
            continue
        family = str(row.get("family") or "").strip()
        regime = _infer_pump_regime(family, conditions)
        precision = (
            _to_float(row.get("precision_valid"), _to_float(row.get("precision"), 0.0)) or 0.0
        )
        score_weight = _to_float(row.get("score_valid"), _to_float(row.get("score"), 0.0)) or 0.0
        priority = _to_int(row.get("pack_rank"), idx + 1)
        pack_name = str(row.get("pack_name") or "").strip()
        rule_id = str(row.get("rule_id") or f"{id_prefix}_{idx + 1:04d}")
        exit_profile = str(row.get("exit_profile") or "").strip() or default_exit_profile
        parsed.append(
            RuntimeRule(
                rule_id=rule_id,
                regime=regime,
                support=support,
                hit_2x_rate=float(precision),
                hit_5x_rate=float(max(0.0, precision * 0.6)),
                rug_rate=float(max(0.0, min(1.0, 1.0 - precision))),
                priority=priority,
                enabled=True,
                score_weight=float(score_weight),
                conditions=conditions,
                exit_profile=exit_profile,
                source=f"{source_label}:{pack_name or 'pump_v2'}",
            )
        )
    return parsed


def _load_pump_rules(config: BotConfig) -> list[RuntimeRule]:
    """Load strict Pump V2 rules from the current rule pack artifact."""
    rows = _load_rule_rows(config.pump_rules_path)
    if rows.empty:
        return []
    strict_rules = _parse_pump_format_rows(
        rows,
        source_label=str(config.pump_rules_path),
        min_support=config.min_rule_support,
        id_prefix="pump_v2",
    )

    filtered = _apply_allowlist(strict_rules, config.optional_allowed_regimes)
    if filtered:
        strict_rules = filtered
    strict_rules = sorted(strict_rules, key=_pump_rule_sort_key)
    return strict_rules[: max(1, int(config.max_active_rules))]


def _load_main_extra_rules(config: BotConfig) -> list[RuntimeRule]:
    """Load hand-curated main-lane rules (e.g. mature_pairs_v1).

    These sit alongside the primary pack and share the same schema. We apply
    a support floor of 1 because mature-pair rules are hand-written (no mined
    support count); pinning them to ``min_rule_support`` would filter them
    out entirely.
    """
    path = getattr(config, "main_rules_path", None)
    if path is None:
        return []
    rows = _load_rule_rows(path)
    if rows.empty:
        return []
    return _parse_pump_format_rows(
        rows,
        source_label=str(path),
        min_support=1,
        id_prefix="main_extra",
        default_exit_profile="mature_pair_v1",
    )


def _load_sniper_extra_rules(config: BotConfig) -> list[RuntimeRule]:
    """Load hand-picked sniper-lane rules (e.g. kaggle_mined_v1).

    Mirrors main_extra semantics: rides alongside the primary pack, opt-in
    via SNIPER_RULE_IDS allowlist. Support floor of 1 because these are
    curated from mined candidates and already passed holdout gates.
    """
    path = getattr(config, "sniper_rules_path", None)
    if path is None:
        return []
    rows = _load_rule_rows(path)
    if rows.empty:
        return []
    return _parse_pump_format_rows(
        rows,
        source_label=str(path),
        min_support=1,
        id_prefix="sniper_extra",
        default_exit_profile="default_sniper",
    )


def load_runtime_rules(config: BotConfig) -> tuple[list[RuntimeRule], dict[str, Any]]:
    """Load and normalize active rules plus optional legacy regime metadata."""
    source_mode = str(config.rules_source_mode or "auto").lower()
    rules: list[RuntimeRule] = []
    selected_source = "none"

    if source_mode in {"pump", "auto"}:
        rules = _load_pump_rules(config)
        if rules:
            selected_source = "pump"

    if not rules and source_mode in {"legacy", "auto"}:
        if source_mode == "legacy" or config.allow_legacy_rule_fallback:
            strict_rules = _load_legacy_rules(config)
            legacy_runtime_rules: list[RuntimeRule] = []
            for strict_rule in strict_rules:
                _append_runtime_variants(legacy_runtime_rules, strict_rule, config)
            rules = legacy_runtime_rules
            if rules:
                selected_source = "legacy"

    if not rules:
        fallback_rule = RuntimeRule(
            rule_id="fallback_conservative_001",
            regime="unknown",
            support=1,
            hit_2x_rate=0.0,
            hit_5x_rate=0.0,
            rug_rate=1.0,
            priority=1,
            enabled=True,
            score_weight=0.0,
            conditions={"tx_count_30s_min": 999999},
            exit_profile="default_cluster",
            source="fallback:no_rules_loaded",
        )
        rules = [fallback_rule]
        selected_source = "fallback"

    disabled_rule_ids = {
        rule_id.strip()
        for rule_id in getattr(config, "disabled_rule_ids", ())
        if str(rule_id).strip()
    }
    if disabled_rule_ids:
        rules = [rule for rule in rules if rule.rule_id not in disabled_rule_ids]

    rules = rules[: max(1, int(config.max_active_rules))]

    # Merge hand-curated main-lane rules (mature_pairs_v1, etc.) AFTER the
    # max_active_rules truncation so a small cap on mined rules doesn't
    # accidentally drop them. They're opt-in via MAIN_RULE_IDS, so leaving
    # them present is harmless when the main lane ignores them.
    main_extra = _load_main_extra_rules(config)
    if main_extra:
        existing_ids = {rule.rule_id for rule in rules}
        rules = list(rules) + [rule for rule in main_extra if rule.rule_id not in existing_ids]
        if selected_source == "none":
            selected_source = "main_extra"

    sniper_extra = _load_sniper_extra_rules(config)
    if sniper_extra:
        existing_ids = {rule.rule_id for rule in rules}
        rules = list(rules) + [rule for rule in sniper_extra if rule.rule_id not in existing_ids]
        if selected_source == "none":
            selected_source = "sniper_extra"

    validated: list[RuntimeRule] = []
    for rule in rules:
        unknown = validate_rule_conditions(rule.conditions)
        if unknown:
            _log.warning(
                "Dropping rule %s (source=%s) with unknown condition keys: %s",
                rule.rule_id,
                rule.source,
                unknown,
            )
            continue
        validated.append(rule)
    rules = validated if validated else rules  # never return empty rule set

    regimes_df = _load_rule_rows(config.regime_comparison_path)
    final_summary = (
        read_json(config.final_summary_path) if config.final_summary_path.exists() else {}
    )
    regime_metadata = {
        "rule_source": selected_source,
        "regime_comparison": regimes_df.to_dict(orient="records") if not regimes_df.empty else [],
        "final_summary": final_summary,
    }
    return rules, regime_metadata
