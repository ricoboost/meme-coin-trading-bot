"""Build entry-level BUY features from Helius-derived swap activity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.io import dataset_path, load_app_config, read_parquet, write_parquet
from src.utils.logging_utils import configure_logging


def compute_price_sol(trades: pd.DataFrame) -> pd.Series:
    """Return swap-implied price in SOL per token."""
    token_amount = pd.to_numeric(trades["token_amount"], errors="coerce")
    sol_amount = pd.to_numeric(trades["sol_amount"], errors="coerce")
    price = sol_amount / token_amount.replace(0, np.nan)
    return price.replace([np.inf, -np.inf], np.nan)


def build_token_activity_snapshots(trades: pd.DataFrame) -> pd.DataFrame:
    """Build rolling token activity snapshots from wallet trades."""
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "token_mint",
                "signature",
                "snapshot_at",
                "last_price_sol",
                "volume_sol_30s",
                "volume_sol_60s",
                "tx_count_30s",
                "tx_count_60s",
                "price_change_30s",
                "price_change_60s",
            ]
        )

    all_snapshots: list[pd.DataFrame] = []
    for token_mint, token_df in trades.groupby("token_mint"):
        token_df = token_df.sort_values("block_time").copy()
        token_df["snapshot_at"] = pd.to_datetime(token_df["block_time"], utc=True)
        token_df["last_price_sol"] = compute_price_sol(token_df)
        token_df["sol_amount"] = pd.to_numeric(token_df["sol_amount"], errors="coerce").fillna(0.0)
        timestamps = token_df["snapshot_at"]
        prices = token_df["last_price_sol"]
        volume_30s: list[float] = []
        volume_60s: list[float] = []
        tx_count_30s: list[int] = []
        tx_count_60s: list[int] = []
        price_change_30s: list[float] = []
        price_change_60s: list[float] = []

        for idx, snapshot_at in enumerate(timestamps):
            start_30s = snapshot_at - pd.Timedelta(seconds=30)
            start_60s = snapshot_at - pd.Timedelta(seconds=60)
            mask_30s = (timestamps >= start_30s) & (timestamps <= snapshot_at)
            mask_60s = (timestamps >= start_60s) & (timestamps <= snapshot_at)
            window_30s = token_df.loc[mask_30s]
            window_60s = token_df.loc[mask_60s]
            volume_30s.append(float(window_30s["sol_amount"].sum()))
            volume_60s.append(float(window_60s["sol_amount"].sum()))
            tx_count_30s.append(int(len(window_30s)))
            tx_count_60s.append(int(len(window_60s)))

            current_price = prices.iloc[idx]
            valid_30s = window_30s["last_price_sol"].dropna()
            valid_60s = window_60s["last_price_sol"].dropna()
            base_30s = valid_30s.iloc[0] if not valid_30s.empty else np.nan
            base_60s = valid_60s.iloc[0] if not valid_60s.empty else np.nan
            price_change_30s.append(
                float((current_price / base_30s) - 1)
                if pd.notna(current_price) and pd.notna(base_30s) and base_30s > 0
                else np.nan
            )
            price_change_60s.append(
                float((current_price / base_60s) - 1)
                if pd.notna(current_price) and pd.notna(base_60s) and base_60s > 0
                else np.nan
            )

        token_df["volume_sol_30s"] = volume_30s
        token_df["volume_sol_60s"] = volume_60s
        token_df["tx_count_30s"] = tx_count_30s
        token_df["tx_count_60s"] = tx_count_60s
        token_df["price_change_30s"] = price_change_30s
        token_df["price_change_60s"] = price_change_60s
        all_snapshots.append(
            token_df[
                [
                    "token_mint",
                    "signature",
                    "snapshot_at",
                    "last_price_sol",
                    "volume_sol_30s",
                    "volume_sol_60s",
                    "tx_count_30s",
                    "tx_count_60s",
                    "price_change_30s",
                    "price_change_60s",
                ]
            ]
        )

    snapshots = pd.concat(all_snapshots, ignore_index=True)
    return snapshots.sort_values(["token_mint", "snapshot_at", "signature"]).reset_index(drop=True)


def compute_buy_event_count(entries: pd.DataFrame, seconds: int, column_name: str) -> pd.Series:
    """Count tracked-wallet BUY events in the trailing time window for the same token."""
    counts: dict[int, int] = {}
    for _, token_df in entries.groupby("token_mint"):
        token_df = token_df.sort_values("entry_time")
        timestamps = token_df["entry_time"]
        values: list[int] = []
        for entry_time in timestamps:
            window_start = entry_time - pd.Timedelta(seconds=seconds)
            values.append(int(((timestamps >= window_start) & (timestamps <= entry_time)).sum()))
        counts.update(dict(zip(token_df.index.tolist(), values)))
    return entries.index.to_series().map(counts).fillna(0).astype(int).rename(column_name)


def compute_wallet_baseline_and_count(buys: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing 30d wallet buy baseline size and buy count."""
    result_parts: list[pd.DataFrame] = []
    for _, wallet_df in buys.groupby("wallet"):
        wallet_df = wallet_df.sort_values("entry_time").copy()
        baselines: list[float] = []
        buy_counts: list[int] = []
        times = wallet_df["entry_time"]
        sizes = pd.to_numeric(wallet_df["sol_amount"], errors="coerce")
        for idx, entry_time in enumerate(times):
            window_start = entry_time - pd.Timedelta(days=30)
            prior_mask = (times >= window_start) & (times < entry_time)
            prior_sizes = sizes.loc[prior_mask].dropna()
            baseline = float(prior_sizes.median()) if not prior_sizes.empty else np.nan
            baselines.append(baseline)
            buy_counts.append(int(prior_mask.sum()))
        wallet_df["wallet_baseline_sol_30d"] = baselines
        wallet_df["wallet_buy_count_30d"] = buy_counts
        result_parts.append(wallet_df)
    return pd.concat(result_parts, ignore_index=True)


def main() -> None:
    """CLI entry point."""
    config = load_app_config()
    logger = configure_logging(logger_name=__name__)
    trades = read_parquet(dataset_path(config, "silver", "wallet_trades.parquet"))
    universe = read_parquet(dataset_path(config, "silver", "token_universe.parquet"))
    wallet_pool = read_parquet(dataset_path(config, "bronze", "wallet_pool.parquet"))
    if trades.empty:
        raise SystemExit(
            "wallet_trades.parquet not found or empty. Run normalize_wallet_trades first."
        )

    trades["block_time"] = pd.to_datetime(trades["block_time"], utc=True)
    trades["price_sol"] = compute_price_sol(trades)
    snapshots = build_token_activity_snapshots(trades)
    write_parquet(snapshots, dataset_path(config, "silver", "token_activity_snapshots.parquet"))

    buys = trades[trades["side"] == "BUY"].copy()
    if buys.empty:
        raise SystemExit("No BUY trades available. Run earlier pipeline stages first.")

    buys = buys.rename(columns={"block_time": "entry_time"})
    buys["entry_time"] = pd.to_datetime(buys["entry_time"], utc=True)
    universe["first_seen_at"] = pd.to_datetime(universe["first_seen_at"], utc=True)

    features = buys.merge(universe[["token_mint", "first_seen_at"]], on="token_mint", how="left")
    features = compute_wallet_baseline_and_count(features)
    features["token_age_sec"] = (
        features["entry_time"] - features["first_seen_at"]
    ).dt.total_seconds()
    features["wallet_cluster_30s"] = compute_buy_event_count(features, 30, "wallet_cluster_30s")
    features["wallet_cluster_120s"] = compute_buy_event_count(features, 120, "wallet_cluster_120s")
    features["wallet_size_vs_baseline"] = pd.to_numeric(
        features["sol_amount"], errors="coerce"
    ) / features["wallet_baseline_sol_30d"].replace(0, np.nan)
    features["wallet_size_vs_baseline"] = (
        features["wallet_size_vs_baseline"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    )
    features["entry_price_sol"] = features["price_sol"]

    features = features.merge(
        snapshots.rename(columns={"snapshot_at": "entry_time"}),
        on=["token_mint", "signature", "entry_time"],
        how="left",
    )
    features = features.merge(
        wallet_pool[
            ["wallet", "score", "appears_daily", "appears_weekly", "appears_monthly"]
        ].rename(columns={"score": "wallet_score"}),
        on="wallet",
        how="left",
    )

    selected = features[
        [
            "wallet",
            "token_mint",
            "entry_time",
            "token_age_sec",
            "wallet_cluster_30s",
            "wallet_cluster_120s",
            "volume_sol_30s",
            "volume_sol_60s",
            "tx_count_30s",
            "tx_count_60s",
            "entry_price_sol",
            "price_change_30s",
            "price_change_60s",
            "wallet_size_vs_baseline",
            "wallet_buy_count_30d",
            "wallet_score",
            "appears_daily",
            "appears_weekly",
            "appears_monthly",
        ]
    ].sort_values("entry_time")

    write_parquet(selected, dataset_path(config, "gold", "entry_features.parquet"))
    logger.info(
        "Saved %s token activity snapshots and %s entry feature rows",
        len(snapshots),
        len(selected),
    )


if __name__ == "__main__":
    main()
