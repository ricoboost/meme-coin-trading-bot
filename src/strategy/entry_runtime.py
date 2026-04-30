"""Runtime entry helpers for lane gating and candidate scoring."""

from __future__ import annotations

from src.bot.config import BotConfig
from src.bot.models import RuntimeFeatures, RuntimeRule


def determine_entry_lane(
    features: RuntimeFeatures,
    config: BotConfig,
) -> tuple[str | None, list[str]]:
    """Classify one candidate into a valid entry lane or return failures.

    Lanes:
    - ``shock``: pullback/recovery entries after a negative short-term move.
    - ``recovery``: controlled recovery/momentum continuation with tighter flow.
    """
    failures: list[str] = []
    price_change_30s = features.price_change_30s
    if price_change_30s is None:
        return None, ["price_change_30s_missing"]

    overextension_max = float(config.entry_overextension_price_max)
    if price_change_30s > overextension_max:
        fresh_override = (
            features.token_age_sec is not None
            and features.token_age_sec <= float(config.entry_overextension_fresh_age_sec)
            and features.volume_sol_30s >= float(config.entry_overextension_fresh_min_volume_sol)
            and features.tx_count_30s >= int(config.entry_overextension_fresh_min_tx)
        )
        if not fresh_override:
            failures.append(f"price_change_30s>{overextension_max:.4f} ({price_change_30s:.4f})")

    shock_match = (
        float(config.entry_lane_shock_price_min)
        <= price_change_30s
        <= float(config.entry_lane_shock_price_max)
        and features.wallet_cluster_30s >= int(config.entry_lane_shock_min_cluster)
        and features.tx_count_30s >= int(config.entry_lane_shock_min_tx)
        and features.volume_sol_30s >= float(config.entry_lane_shock_min_volume_sol)
    )
    if shock_match:
        return "shock", failures

    recovery_match = (
        float(config.entry_lane_recovery_price_min)
        <= price_change_30s
        <= float(config.entry_lane_recovery_price_max)
        and features.wallet_cluster_30s >= int(config.entry_lane_recovery_min_cluster)
        and features.tx_count_30s >= int(config.entry_lane_recovery_min_tx)
        and features.volume_sol_30s >= float(config.entry_lane_recovery_min_volume_sol)
    )
    if recovery_match:
        recovery_abs_move_min = float(config.entry_lane_recovery_abs_move_min)
        if recovery_abs_move_min > 0 and abs(float(price_change_30s)) < recovery_abs_move_min:
            failures.append(
                f"recovery_abs_move<{recovery_abs_move_min:.4f} ({abs(float(price_change_30s)):.4f})"
            )
            return None, failures

        recovery_max_cluster = int(config.entry_lane_recovery_max_cluster)
        if recovery_max_cluster > 0 and features.wallet_cluster_30s > recovery_max_cluster:
            failures.append(
                f"recovery_cluster>{recovery_max_cluster} ({features.wallet_cluster_30s})"
            )
            return None, failures

        recovery_max_tx = int(config.entry_lane_recovery_max_tx)
        if recovery_max_tx > 0 and features.tx_count_30s > recovery_max_tx:
            failures.append(f"recovery_tx>{recovery_max_tx} ({features.tx_count_30s})")
            return None, failures

        recovery_max_volume_sol = float(config.entry_lane_recovery_max_volume_sol)
        if recovery_max_volume_sol > 0 and features.volume_sol_30s > recovery_max_volume_sol:
            failures.append(
                f"recovery_volume>{recovery_max_volume_sol:.4f} ({features.volume_sol_30s:.4f})"
            )
            return None, failures

        return "recovery", failures

    if bool(getattr(config, "entry_lane_mature_enabled", False)):
        mature_match = (
            float(config.entry_lane_mature_price_min)
            <= price_change_30s
            <= float(config.entry_lane_mature_price_max)
            and features.wallet_cluster_30s >= int(config.entry_lane_mature_min_cluster)
            and features.tx_count_30s >= int(config.entry_lane_mature_min_tx)
            and features.volume_sol_30s >= float(config.entry_lane_mature_min_volume_sol)
        )
        if mature_match:
            return "mature", failures

    if failures:
        return None, failures

    # Nearest lane misses for transparent debugging.
    failures.extend(
        [
            (
                "lane_shock_miss:"
                f"price[{float(config.entry_lane_shock_price_min):.2f},{float(config.entry_lane_shock_price_max):.2f}]"
                f"/cluster>={int(config.entry_lane_shock_min_cluster)}"
                f"/tx>={int(config.entry_lane_shock_min_tx)}"
                f"/vol>={float(config.entry_lane_shock_min_volume_sol):.2f}"
            ),
            (
                "lane_recovery_miss:"
                f"price[{float(config.entry_lane_recovery_price_min):.2f},{float(config.entry_lane_recovery_price_max):.2f}]"
                f"/cluster>={int(config.entry_lane_recovery_min_cluster)}"
                f"/tx>={int(config.entry_lane_recovery_min_tx)}"
                f"/vol>={float(config.entry_lane_recovery_min_volume_sol):.2f}"
            ),
        ]
    )
    if bool(getattr(config, "entry_lane_mature_enabled", False)):
        failures.append(
            "lane_mature_miss:"
            f"price[{float(config.entry_lane_mature_price_min):.2f},{float(config.entry_lane_mature_price_max):.2f}]"
            f"/cluster>={int(config.entry_lane_mature_min_cluster)}"
            f"/tx>={int(config.entry_lane_mature_min_tx)}"
            f"/vol>={float(config.entry_lane_mature_min_volume_sol):.2f}"
        )
    return None, failures


def score_candidate(
    features: RuntimeFeatures,
    rule: RuntimeRule,
    detected_regime: str,
    lane: str,
    config: BotConfig,
) -> tuple[float, dict[str, float]]:
    """Return deterministic candidate score (+ breakdown) for queue ranking."""

    def _norm(value: float, scale: float) -> float:
        if scale <= 0:
            return 0.0
        return min(max(value, 0.0) / scale, 1.0)

    source = str(rule.source or "")
    if ":relaxed" in source:
        strict_factor = 0.70
    elif ":pair_adapted" in source:
        strict_factor = 0.85
    else:
        strict_factor = 1.00

    regime_match = 1.0 if rule.regime == detected_regime else 0.0
    if lane == "shock":
        lane_bonus = 0.06
    elif lane == "mature":
        lane_bonus = 0.04
    else:
        lane_bonus = 0.03

    score = 0.0
    score += 0.42 * float(rule.hit_2x_rate)
    score += 0.34 * float(rule.hit_5x_rate)
    score -= 0.32 * float(rule.rug_rate)
    score += 0.10 * float(rule.score_weight)
    score += 0.08 * regime_match
    score += lane_bonus
    score += 0.10 * _norm(float(features.wallet_cluster_30s), 6.0)
    score += 0.09 * _norm(float(features.tx_count_30s), 10.0)
    score += 0.10 * _norm(float(features.volume_sol_30s), 12.0)

    tracked_wallet_features_enabled = bool(getattr(config, "tracked_wallet_features_enabled", True))
    tracked_wallet_presence_bonus = (
        float(config.entry_score_tracked_wallet_presence_bonus)
        if tracked_wallet_features_enabled and features.tracked_wallet_present_60s
        else 0.0
    )
    tracked_wallet_count_bonus = (
        float(config.entry_score_tracked_wallet_count_weight)
        * _norm(
            float(features.tracked_wallet_count_60s),
            float(config.entry_score_tracked_wallet_count_scale),
        )
        if tracked_wallet_features_enabled
        else 0.0
    )
    tracked_wallet_score_bonus = (
        float(config.entry_score_tracked_wallet_score_weight)
        * _norm(
            float(features.tracked_wallet_score_sum_60s),
            float(config.entry_score_tracked_wallet_score_scale),
        )
        if tracked_wallet_features_enabled
        else 0.0
    )
    tracked_wallet_bonus_total = (
        tracked_wallet_presence_bonus + tracked_wallet_count_bonus + tracked_wallet_score_bonus
    )
    score += tracked_wallet_bonus_total
    score *= strict_factor
    return float(score), {
        "strict_factor": float(strict_factor),
        "tracked_wallet_presence_bonus": float(tracked_wallet_presence_bonus),
        "tracked_wallet_count_bonus": float(tracked_wallet_count_bonus),
        "tracked_wallet_score_bonus": float(tracked_wallet_score_bonus),
        "tracked_wallet_bonus_total": float(tracked_wallet_bonus_total),
        "tracked_wallet_features_enabled": 1.0 if tracked_wallet_features_enabled else 0.0,
    }
