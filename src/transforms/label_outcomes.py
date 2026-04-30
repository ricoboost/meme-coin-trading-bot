"""Label forward outcomes from reconstructed future swap-implied prices."""

from __future__ import annotations

import pandas as pd

from src.utils.io import dataset_path, load_app_config, read_parquet, write_parquet
from src.utils.logging_utils import configure_logging


def future_window_stats(
    token_activity: pd.DataFrame,
    entry_time: pd.Timestamp,
    entry_price_sol: float,
    hours: int,
) -> tuple[float | None, float | None]:
    """Return max and min forward returns within the horizon."""
    window_end = entry_time + pd.Timedelta(hours=hours)
    if pd.isna(entry_price_sol) or entry_price_sol <= 0:
        return None, None
    future = token_activity[
        (token_activity["snapshot_at"] >= entry_time)
        & (token_activity["snapshot_at"] <= window_end)
    ]
    prices = pd.to_numeric(future["last_price_sol"], errors="coerce").dropna()
    if prices.empty:
        return None, None
    max_return = (prices.max() / entry_price_sol) - 1
    min_return = (prices.min() / entry_price_sol) - 1
    return float(max_return), float(min_return)


def rug_flag(
    token_activity: pd.DataFrame, entry_time: pd.Timestamp, entry_price_sol: float
) -> bool:
    """Flag rug-like price collapses within 24 hours using tracked swap-implied prices."""
    if pd.isna(entry_price_sol) or entry_price_sol <= 0:
        return False
    future = token_activity[
        (token_activity["snapshot_at"] >= entry_time)
        & (token_activity["snapshot_at"] <= entry_time + pd.Timedelta(hours=24))
    ]
    if future.empty:
        return False
    prices = pd.to_numeric(future["last_price_sol"], errors="coerce").dropna()
    if prices.empty:
        return False
    min_return = (prices.min() / entry_price_sol) - 1
    # TODO: enrich rug detection with liquidity and holder-failure heuristics when better on-chain coverage is added.
    return bool(min_return <= -0.8)


def build_quality_flags(labeled: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Mark rows that pass research-quality filters for downstream analysis."""
    reasons = []
    for row in labeled.itertuples(index=False):
        row_reasons: list[str] = []
        if (
            pd.isna(row.entry_price_sol)
            or row.entry_price_sol < filters["min_entry_price_sol"]
            or row.entry_price_sol > filters["max_entry_price_sol"]
        ):
            row_reasons.append("entry_price_out_of_bounds")
        if pd.isna(row.volume_sol_30s) or row.volume_sol_30s < filters["min_volume_sol_30s"]:
            row_reasons.append("low_volume_30s")
        if pd.isna(row.tx_count_30s) or row.tx_count_30s < filters["min_tx_count_30s"]:
            row_reasons.append("low_tx_count_30s")
        if pd.notna(row.max_return_24h) and row.max_return_24h > filters["max_return_24h_cap"]:
            row_reasons.append("max_return_cap_exceeded")
        reasons.append("|".join(row_reasons))

    labeled = labeled.copy()
    labeled["quality_reason"] = reasons
    labeled["quality_pass"] = labeled["quality_reason"].eq("")
    labeled.loc[~labeled["quality_pass"], ["hit_2x_24h", "hit_5x_24h", "rug_24h"]] = False
    labeled.loc[~labeled["quality_pass"], ["max_return_1h", "max_return_6h", "max_return_24h"]] = (
        pd.NA
    )
    return labeled


def main() -> None:
    """CLI entry point."""
    config = load_app_config()
    logger = configure_logging(logger_name=__name__)
    features = read_parquet(dataset_path(config, "gold", "entry_features.parquet"))
    activity = read_parquet(dataset_path(config, "silver", "token_activity_snapshots.parquet"))
    if features.empty:
        raise SystemExit("entry_features.parquet not found or empty. Run build_features first.")
    if activity.empty:
        raise SystemExit(
            "token_activity_snapshots.parquet not found or empty. Run build_features first."
        )
    quality_filters = config.settings["analysis"]["quality_filters"]

    features["entry_time"] = pd.to_datetime(features["entry_time"], utc=True)
    activity["snapshot_at"] = pd.to_datetime(activity["snapshot_at"], utc=True)

    labeled_rows = []
    for row in features.itertuples(index=False):
        token_activity = activity[activity["token_mint"] == row.token_mint].sort_values(
            "snapshot_at"
        )
        max_1h, _ = future_window_stats(token_activity, row.entry_time, row.entry_price_sol, 1)
        max_6h, _ = future_window_stats(token_activity, row.entry_time, row.entry_price_sol, 6)
        max_24h, _ = future_window_stats(token_activity, row.entry_time, row.entry_price_sol, 24)
        rug = rug_flag(token_activity, row.entry_time, row.entry_price_sol)
        labeled_rows.append(
            {
                **row._asdict(),
                "max_return_1h": max_1h,
                "max_return_6h": max_6h,
                "max_return_24h": max_24h,
                "hit_2x_24h": bool(max_24h is not None and max_24h >= 1.0),
                "hit_5x_24h": bool(max_24h is not None and max_24h >= 4.0),
                # TODO: upgrade label quality with broader market coverage beyond tracked-wallet swaps.
                "rug_24h": rug,
            }
        )

    labeled = pd.DataFrame(labeled_rows)
    labeled = build_quality_flags(labeled, quality_filters)
    write_parquet(labeled, dataset_path(config, "gold", "labeled_entries.parquet"))
    logger.info(
        "Saved %s labeled entries; %s passed quality filters",
        len(labeled),
        int(labeled["quality_pass"].sum()),
    )


if __name__ == "__main__":
    main()
