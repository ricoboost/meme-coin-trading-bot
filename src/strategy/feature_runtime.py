"""Runtime feature reconstruction from in-memory token activity."""

from __future__ import annotations

from src.bot.models import CandidateEvent, RuntimeFeatures
from src.monitoring.token_activity import TokenActivityCache


def build_runtime_features(
    event: CandidateEvent,
    token_cache: TokenActivityCache,
    wallet_scores: dict[str, float],
    snapshot_time=None,
    entry_time=None,
) -> RuntimeFeatures | None:
    """Build runtime features for one candidate token event."""
    effective_snapshot_time = snapshot_time or event.block_time
    effective_entry_time = entry_time or effective_snapshot_time
    snapshot = token_cache.snapshot(event.token_mint, effective_snapshot_time)
    if snapshot is None:
        return None
    snapshot_raw = {**snapshot, "__event_signature": event.signature}
    tracked_wallet_scores = [
        float(wallet_scores.get(wallet, 0.0))
        for wallet in event.tracked_wallets
        if wallet in wallet_scores
    ]
    triggering_wallet_score = float(wallet_scores.get(event.triggering_wallet, 0.0))
    if triggering_wallet_score <= 0.0 and tracked_wallet_scores:
        triggering_wallet_score = max(tracked_wallet_scores)
    return RuntimeFeatures(
        token_mint=event.token_mint,
        entry_time=effective_entry_time,
        entry_price_sol=snapshot_raw.get("last_price_sol"),
        token_age_sec=snapshot_raw.get("token_age_sec"),
        wallet_cluster_30s=int(snapshot_raw.get("wallet_cluster_30s", 0)),
        wallet_cluster_120s=int(snapshot_raw.get("wallet_cluster_120s", 0)),
        volume_sol_30s=float(snapshot_raw.get("volume_sol_30s", 0.0)),
        volume_sol_60s=float(snapshot_raw.get("volume_sol_60s", 0.0)),
        tx_count_30s=int(snapshot_raw.get("tx_count_30s", 0)),
        tx_count_60s=int(snapshot_raw.get("tx_count_60s", 0)),
        price_change_30s=snapshot_raw.get("price_change_30s"),
        price_change_60s=snapshot_raw.get("price_change_60s"),
        triggering_wallet=event.triggering_wallet,
        triggering_wallet_score=triggering_wallet_score,
        aggregated_wallet_score=float(snapshot_raw.get("aggregated_wallet_score", 0.0)),
        tracked_wallet_present_60s=bool(snapshot_raw.get("tracked_wallet_present_60s", False)),
        tracked_wallet_count_60s=int(snapshot_raw.get("tracked_wallet_count_60s", 0)),
        tracked_wallet_score_sum_60s=float(snapshot_raw.get("tracked_wallet_score_sum_60s", 0.0)),
        buy_volume_sol_30s=float(snapshot_raw.get("buy_volume_sol_30s", 0.0)),
        buy_volume_sol_60s=float(snapshot_raw.get("buy_volume_sol_60s", 0.0)),
        sell_volume_sol_30s=float(snapshot_raw.get("sell_volume_sol_30s", 0.0)),
        sell_volume_sol_60s=float(snapshot_raw.get("sell_volume_sol_60s", 0.0)),
        buy_tx_count_30s=int(snapshot_raw.get("buy_tx_count_30s", 0)),
        buy_tx_count_60s=int(snapshot_raw.get("buy_tx_count_60s", 0)),
        sell_tx_count_30s=int(snapshot_raw.get("sell_tx_count_30s", 0)),
        sell_tx_count_60s=int(snapshot_raw.get("sell_tx_count_60s", 0)),
        buy_sell_ratio_30s=snapshot_raw.get("buy_sell_ratio_30s"),
        buy_sell_ratio_60s=snapshot_raw.get("buy_sell_ratio_60s"),
        net_flow_sol_30s=float(snapshot_raw.get("net_flow_sol_30s", 0.0)),
        net_flow_sol_60s=float(snapshot_raw.get("net_flow_sol_60s", 0.0)),
        avg_trade_sol_30s=float(snapshot_raw.get("avg_trade_sol_30s", 0.0)),
        avg_trade_sol_60s=float(snapshot_raw.get("avg_trade_sol_60s", 0.0)),
        round_trip_wallet_count_30s=int(snapshot_raw.get("round_trip_wallet_count_30s", 0)),
        round_trip_wallet_count_60s=int(snapshot_raw.get("round_trip_wallet_count_60s", 0)),
        round_trip_wallet_ratio_30s=float(snapshot_raw.get("round_trip_wallet_ratio_30s", 0.0)),
        round_trip_wallet_ratio_60s=float(snapshot_raw.get("round_trip_wallet_ratio_60s", 0.0)),
        round_trip_volume_sol_30s=float(snapshot_raw.get("round_trip_volume_sol_30s", 0.0)),
        round_trip_volume_sol_60s=float(snapshot_raw.get("round_trip_volume_sol_60s", 0.0)),
        real_volume_sol_30s=float(snapshot_raw.get("real_volume_sol_30s", 0.0)),
        real_volume_sol_60s=float(snapshot_raw.get("real_volume_sol_60s", 0.0)),
        real_buy_volume_sol_30s=float(snapshot_raw.get("real_buy_volume_sol_30s", 0.0)),
        real_buy_volume_sol_60s=float(snapshot_raw.get("real_buy_volume_sol_60s", 0.0)),
        raw=snapshot_raw,
    )
