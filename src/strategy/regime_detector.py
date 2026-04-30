"""Deterministic runtime regime detection."""

from __future__ import annotations

from src.bot.models import RuntimeFeatures


def detect_regime(features: RuntimeFeatures) -> str:
    """Infer a regime label from runtime features."""
    if features.price_change_30s is None:
        return "unknown"
    if features.price_change_30s <= -0.08 and features.wallet_cluster_30s >= 3:
        return "high_cluster_recovery"
    if features.price_change_30s <= -0.08:
        return "negative_shock_recovery"
    if (
        -0.2 <= features.price_change_30s <= 0.2
        and features.wallet_cluster_30s >= 3
        and features.tx_count_30s >= 3
    ):
        return "high_cluster_recovery"
    if features.price_change_30s >= 0.2 and features.wallet_cluster_30s >= 8:
        return "momentum_burst"
    if features.price_change_30s >= 0.2:
        return "momentum_burst"
    return "unknown"
