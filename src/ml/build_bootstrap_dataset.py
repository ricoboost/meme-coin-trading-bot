"""Build a model-ready bootstrap CSV from Pump dataset chunks + labels.

This command aggregates early per-mint transaction behavior from `chunk_*.csv`
and joins labels from `train.csv` (`has_graduated`).

Output columns are aligned with `LiveMLFilter.FEATURE_NAMES` plus:
- `label` (0/1 target)
- `mint`
- `slot_min`
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.live_filter import LiveMLFilter


SLOTS_PER_SECOND = 2.5
SLOT_WINDOW_30S = int(30 * SLOTS_PER_SECOND)  # 75
SLOT_WINDOW_60S = int(60 * SLOTS_PER_SECOND)  # 150
SLOT_WINDOW_120S = int(120 * SLOTS_PER_SECOND)  # 300


@dataclass
class MintStats:
    slot_min: int
    label: int
    first_slot: int = 10**18
    last_slot: int = -1
    tx_30: int = 0
    tx_60: int = 0
    tx_120: int = 0
    vol_30: float = 0.0
    vol_60: float = 0.0
    vol_120: float = 0.0
    buy_30: float = 0.0
    buy_60: float = 0.0
    sell_30: float = 0.0
    sell_60: float = 0.0
    wallet_30_est: int = 0
    wallet_120_est: int = 0
    first_30_slot: int = 10**18
    first_30_tx: int = 10**18
    first_30_price: float = 0.0
    last_30_slot: int = -1
    last_30_tx: int = -1
    last_30_price: float = 0.0
    first_60_slot: int = 10**18
    first_60_tx: int = 10**18
    first_60_price: float = 0.0
    last_60_slot: int = -1
    last_60_tx: int = -1
    last_60_price: float = 0.0


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value_f):
        return default
    return value_f


def _update_first_price(
    stats: MintStats, window: str, slot: int, tx_idx: int, price: float
) -> None:
    if not np.isfinite(price) or price <= 0:
        return
    if window == "30":
        if slot < stats.first_30_slot or (
            slot == stats.first_30_slot and tx_idx < stats.first_30_tx
        ):
            stats.first_30_slot = slot
            stats.first_30_tx = tx_idx
            stats.first_30_price = price
    else:
        if slot < stats.first_60_slot or (
            slot == stats.first_60_slot and tx_idx < stats.first_60_tx
        ):
            stats.first_60_slot = slot
            stats.first_60_tx = tx_idx
            stats.first_60_price = price


def _update_last_price(stats: MintStats, window: str, slot: int, tx_idx: int, price: float) -> None:
    if not np.isfinite(price) or price <= 0:
        return
    if window == "30":
        if slot > stats.last_30_slot or (slot == stats.last_30_slot and tx_idx > stats.last_30_tx):
            stats.last_30_slot = slot
            stats.last_30_tx = tx_idx
            stats.last_30_price = price
    else:
        if slot > stats.last_60_slot or (slot == stats.last_60_slot and tx_idx > stats.last_60_tx):
            stats.last_60_slot = slot
            stats.last_60_tx = tx_idx
            stats.last_60_price = price


def _price_change(first_price: float, last_price: float) -> float:
    if first_price <= 0 or last_price <= 0:
        return 0.0
    value = (last_price / first_price) - 1.0
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -0.99, 20.0))


def _candidate_score(stats: MintStats) -> float:
    """Simple monotonic score used only as bootstrap feature."""
    tx_term = min(1.0, stats.tx_30 / 20.0) * 0.35
    cluster_term = min(1.0, stats.wallet_30_est / 10.0) * 0.25
    vol_term = min(1.0, stats.vol_30 / 10.0) * 0.30
    flow = stats.buy_30 - stats.sell_30
    flow_term = max(0.0, min(1.0, flow / max(stats.vol_30, 1e-9))) * 0.10
    return float(np.clip(tx_term + cluster_term + vol_term + flow_term, 0.0, 1.0))


def _regime_from_stats(stats: MintStats, p30: float) -> str:
    cluster = max(0, min(stats.wallet_30_est, stats.tx_30))
    if p30 <= -0.2 and cluster >= 2 and stats.vol_30 >= 2.0:
        return "negative_shock_recovery"
    if cluster >= 4 and abs(p30) <= 0.35 and stats.vol_30 >= 3.0:
        return "high_cluster_recovery"
    if p30 >= 0.2 and stats.vol_30 >= 3.0:
        return "momentum_burst"
    return "unknown"


def _build_row(mint: str, stats: MintStats) -> dict[str, float | int | str]:
    wallet_cluster_30 = max(0, min(stats.wallet_30_est, stats.tx_30))
    wallet_cluster_120 = max(0, min(stats.wallet_120_est, stats.tx_120))
    tx_30 = max(0, stats.tx_30)
    tx_60 = max(0, stats.tx_60)
    vol_30 = max(0.0, stats.vol_30)
    vol_60 = max(0.0, stats.vol_60)
    buy_30 = max(0.0, stats.buy_30)
    buy_60 = max(0.0, stats.buy_60)
    sell_30 = max(0.0, stats.sell_30)
    sell_60 = max(0.0, stats.sell_60)
    avg_30 = vol_30 / tx_30 if tx_30 > 0 else 0.0
    avg_60 = vol_60 / tx_60 if tx_60 > 0 else 0.0
    ratio_30 = buy_30 / max(sell_30, 1e-9) if buy_30 > 0 else 0.0
    ratio_60 = buy_60 / max(sell_60, 1e-9) if buy_60 > 0 else 0.0

    entry_price = stats.first_30_price if stats.first_30_price > 0 else stats.first_60_price
    p30 = _price_change(stats.first_30_price, stats.last_30_price)
    p60 = _price_change(stats.first_60_price, stats.last_60_price)

    slot_span = max(0, stats.last_slot - stats.slot_min)
    token_age_sec = float(slot_span / SLOTS_PER_SECOND)
    regime = _regime_from_stats(stats, p30)
    lane_shock = 1.0 if p30 <= -0.2 else 0.0
    lane_recovery = 1.0 if -0.2 <= p30 <= 0.35 and abs(p30) >= 0.05 else 0.0

    row: dict[str, float | int | str] = {
        "mint": mint,
        "slot_min": int(stats.slot_min),
        "label": int(stats.label),
        "token_age_sec": token_age_sec,
        "wallet_cluster_30s": float(wallet_cluster_30),
        "wallet_cluster_120s": float(wallet_cluster_120),
        "volume_sol_30s": vol_30,
        "volume_sol_60s": vol_60,
        "tx_count_30s": float(tx_30),
        "tx_count_60s": float(tx_60),
        "buy_volume_sol_30s": buy_30,
        "buy_volume_sol_60s": buy_60,
        "sell_volume_sol_30s": sell_30,
        "sell_volume_sol_60s": sell_60,
        "buy_sell_ratio_30s": ratio_30,
        "buy_sell_ratio_60s": ratio_60,
        "net_flow_sol_30s": buy_30 - sell_30,
        "net_flow_sol_60s": buy_60 - sell_60,
        "avg_trade_sol_30s": avg_30,
        "avg_trade_sol_60s": avg_60,
        "entry_price_sol": entry_price,
        "price_change_30s": p30,
        "price_change_60s": p60,
        "tracked_wallet_present_60s": 0.0,
        "tracked_wallet_count_60s": 0.0,
        "tracked_wallet_score_sum_60s": 0.0,
        "triggering_wallet_score": 0.0,
        "aggregated_wallet_score": 0.0,
        "candidate_score": _candidate_score(stats),
        "rule_support": 0.0,
        "rule_hit_2x_rate": 0.0,
        "rule_hit_5x_rate": 0.0,
        "rule_rug_rate": 0.0,
        "strategy_is_sniper": 0.0,
        "lane_shock": lane_shock,
        "lane_recovery": lane_recovery,
        "regime_negative_shock_recovery": 1.0 if regime == "negative_shock_recovery" else 0.0,
        "regime_high_cluster_recovery": 1.0 if regime == "high_cluster_recovery" else 0.0,
        "regime_momentum_burst": 1.0 if regime == "momentum_burst" else 0.0,
        "regime_unknown": 1.0 if regime == "unknown" else 0.0,
    }

    # Ensure every runtime feature exists.
    for feature_name in LiveMLFilter.FEATURE_NAMES:
        row.setdefault(feature_name, 0.0)
    return row


def build_bootstrap_dataset(
    dataset_dir: Path,
    output_path: Path,
    *,
    labels_file: str,
    tx_glob: str,
    slot_horizon: int,
    chunksize: int,
    max_files: int,
    max_rows: int,
    max_mints: int,
) -> tuple[int, int]:
    labels_path = dataset_dir / labels_file
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file missing: {labels_path}")

    labels = pd.read_csv(
        labels_path,
        usecols=["mint", "slot_min", "has_graduated", "is_valid"],
        low_memory=False,
    )
    labels = labels[labels["is_valid"].astype(bool)].copy()
    labels["mint"] = labels["mint"].astype(str)
    labels["slot_min"] = pd.to_numeric(labels["slot_min"], errors="coerce").fillna(0).astype(int)
    labels["label"] = labels["has_graduated"].astype(bool).astype(int)

    slot_min_map = dict(zip(labels["mint"], labels["slot_min"], strict=False))
    label_map = dict(zip(labels["mint"], labels["label"], strict=False))

    tx_files = sorted(dataset_dir.glob(tx_glob))
    if not tx_files:
        raise FileNotFoundError(f"No transaction files matched: {dataset_dir / tx_glob}")
    if max_files > 0:
        tx_files = tx_files[:max_files]

    stats_by_mint: dict[str, MintStats] = {}
    processed_rows = 0

    usecols = [
        "slot",
        "tx_idx",
        "signing_wallet",
        "direction",
        "base_coin",
        "base_coin_amount",
        "quote_coin_amount",
        "signature",
    ]

    for tx_file in tx_files:
        for chunk in pd.read_csv(tx_file, usecols=usecols, chunksize=chunksize, low_memory=False):
            if max_rows > 0 and processed_rows >= max_rows:
                break

            chunk = chunk.rename(columns={"base_coin": "mint"}).copy()
            chunk["mint"] = chunk["mint"].astype(str)
            chunk["slot"] = pd.to_numeric(chunk["slot"], errors="coerce").fillna(-1).astype(int)
            chunk["tx_idx"] = pd.to_numeric(chunk["tx_idx"], errors="coerce").fillna(0).astype(int)
            chunk["slot_min"] = chunk["mint"].map(slot_min_map)
            chunk = chunk[chunk["slot_min"].notna()].copy()
            if chunk.empty:
                continue

            chunk["slot_min"] = (
                pd.to_numeric(chunk["slot_min"], errors="coerce").fillna(0).astype(int)
            )
            chunk["slot_delta"] = chunk["slot"] - chunk["slot_min"]
            chunk = chunk[(chunk["slot_delta"] >= 0) & (chunk["slot_delta"] <= slot_horizon)].copy()
            if chunk.empty:
                continue

            chunk["direction"] = chunk["direction"].astype(str).str.lower()
            chunk["sol_amount"] = (
                pd.to_numeric(chunk["quote_coin_amount"], errors="coerce").fillna(0.0).astype(float)
                / 1_000_000_000.0
            )
            chunk["token_amount"] = (
                pd.to_numeric(chunk["base_coin_amount"], errors="coerce").fillna(0.0).astype(float)
            )
            chunk["price"] = np.where(
                chunk["token_amount"] > 0,
                chunk["sol_amount"] / chunk["token_amount"],
                0.0,
            )
            chunk["sol_buy"] = np.where(chunk["direction"] == "buy", chunk["sol_amount"], 0.0)
            chunk["sol_sell"] = np.where(chunk["direction"] == "sell", chunk["sol_amount"], 0.0)

            processed_rows += len(chunk)

            span = chunk.groupby("mint", observed=True).agg(
                first_slot=("slot", "min"),
                last_slot=("slot", "max"),
            )
            for mint, row in span.iterrows():
                if mint not in stats_by_mint:
                    if max_mints > 0 and len(stats_by_mint) >= max_mints:
                        continue
                    stats_by_mint[mint] = MintStats(
                        slot_min=_safe_int(slot_min_map.get(mint), 0),
                        label=_safe_int(label_map.get(mint), 0),
                    )
                stats = stats_by_mint[mint]
                stats.first_slot = min(
                    stats.first_slot, _safe_int(row["first_slot"], stats.first_slot)
                )
                stats.last_slot = max(stats.last_slot, _safe_int(row["last_slot"], stats.last_slot))

            chunk_30 = chunk[chunk["slot_delta"] <= SLOT_WINDOW_30S]
            chunk_60 = chunk[chunk["slot_delta"] <= SLOT_WINDOW_60S]
            chunk_120 = chunk[chunk["slot_delta"] <= SLOT_WINDOW_120S]

            if not chunk_30.empty:
                agg_30 = chunk_30.groupby("mint", observed=True).agg(
                    tx=("signature", "count"),
                    vol=("sol_amount", "sum"),
                    buy=("sol_buy", "sum"),
                    sell=("sol_sell", "sum"),
                    wallets=("signing_wallet", "nunique"),
                )
                for mint, row in agg_30.iterrows():
                    stats = stats_by_mint.get(mint)
                    if stats is None:
                        continue
                    stats.tx_30 += _safe_int(row["tx"], 0)
                    stats.vol_30 += _safe_float(row["vol"], 0.0)
                    stats.buy_30 += _safe_float(row["buy"], 0.0)
                    stats.sell_30 += _safe_float(row["sell"], 0.0)
                    stats.wallet_30_est += _safe_int(row["wallets"], 0)

                sorted_30 = chunk_30.sort_values(["mint", "slot", "tx_idx"])
                first_30 = sorted_30.drop_duplicates("mint", keep="first")[
                    ["mint", "slot", "tx_idx", "price"]
                ]
                last_30 = sorted_30.drop_duplicates("mint", keep="last")[
                    ["mint", "slot", "tx_idx", "price"]
                ]
                for row in first_30.itertuples(index=False):
                    stats = stats_by_mint.get(row.mint)
                    if stats is not None:
                        _update_first_price(
                            stats,
                            "30",
                            _safe_int(row.slot),
                            _safe_int(row.tx_idx),
                            _safe_float(row.price),
                        )
                for row in last_30.itertuples(index=False):
                    stats = stats_by_mint.get(row.mint)
                    if stats is not None:
                        _update_last_price(
                            stats,
                            "30",
                            _safe_int(row.slot),
                            _safe_int(row.tx_idx),
                            _safe_float(row.price),
                        )

            if not chunk_60.empty:
                agg_60 = chunk_60.groupby("mint", observed=True).agg(
                    tx=("signature", "count"),
                    vol=("sol_amount", "sum"),
                    buy=("sol_buy", "sum"),
                    sell=("sol_sell", "sum"),
                )
                for mint, row in agg_60.iterrows():
                    stats = stats_by_mint.get(mint)
                    if stats is None:
                        continue
                    stats.tx_60 += _safe_int(row["tx"], 0)
                    stats.vol_60 += _safe_float(row["vol"], 0.0)
                    stats.buy_60 += _safe_float(row["buy"], 0.0)
                    stats.sell_60 += _safe_float(row["sell"], 0.0)

                sorted_60 = chunk_60.sort_values(["mint", "slot", "tx_idx"])
                first_60 = sorted_60.drop_duplicates("mint", keep="first")[
                    ["mint", "slot", "tx_idx", "price"]
                ]
                last_60 = sorted_60.drop_duplicates("mint", keep="last")[
                    ["mint", "slot", "tx_idx", "price"]
                ]
                for row in first_60.itertuples(index=False):
                    stats = stats_by_mint.get(row.mint)
                    if stats is not None:
                        _update_first_price(
                            stats,
                            "60",
                            _safe_int(row.slot),
                            _safe_int(row.tx_idx),
                            _safe_float(row.price),
                        )
                for row in last_60.itertuples(index=False):
                    stats = stats_by_mint.get(row.mint)
                    if stats is not None:
                        _update_last_price(
                            stats,
                            "60",
                            _safe_int(row.slot),
                            _safe_int(row.tx_idx),
                            _safe_float(row.price),
                        )

            if not chunk_120.empty:
                agg_120 = chunk_120.groupby("mint", observed=True).agg(
                    tx=("signature", "count"),
                    vol=("sol_amount", "sum"),
                    wallets=("signing_wallet", "nunique"),
                )
                for mint, row in agg_120.iterrows():
                    stats = stats_by_mint.get(mint)
                    if stats is None:
                        continue
                    stats.tx_120 += _safe_int(row["tx"], 0)
                    stats.vol_120 += _safe_float(row["vol"], 0.0)
                    stats.wallet_120_est += _safe_int(row["wallets"], 0)

            if max_rows > 0 and processed_rows >= max_rows:
                break
        if max_rows > 0 and processed_rows >= max_rows:
            break

    rows = [_build_row(mint, stats) for mint, stats in stats_by_mint.items() if stats.tx_30 > 0]
    if not rows:
        raise RuntimeError("No bootstrap rows produced. Check dataset path and input files.")

    output_df = pd.DataFrame(rows)
    feature_cols = list(LiveMLFilter.FEATURE_NAMES)
    ordered_cols = ["mint", "slot_min", "label", *feature_cols]
    output_df = output_df[ordered_cols]
    output_df = output_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    output_df["label"] = output_df["label"].astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    return len(output_df), processed_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labeled bootstrap CSV for LiveMLFilter.")
    parser.add_argument("--dataset-dir", default="data/external/pump_dataset")
    parser.add_argument("--labels-file", default="train.csv")
    parser.add_argument("--tx-glob", default="chunk_*.csv")
    parser.add_argument("--output", default="data/external/pump_dataset/ml_bootstrap_labeled.csv")
    parser.add_argument("--slot-horizon", type=int, default=SLOT_WINDOW_120S)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max-files", type=int, default=0, help="0 = all files")
    parser.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    parser.add_argument("--max-mints", type=int, default=0, help="0 = no cap")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output)
    max_files = int(args.max_files) if args.max_files > 0 else 0
    max_rows = int(args.max_rows) if args.max_rows > 0 else 0
    max_mints = int(args.max_mints) if args.max_mints > 0 else 0

    row_count, processed_rows = build_bootstrap_dataset(
        dataset_dir=dataset_dir,
        output_path=output_path,
        labels_file=args.labels_file,
        tx_glob=args.tx_glob,
        slot_horizon=max(1, int(args.slot_horizon)),
        chunksize=max(10_000, int(args.chunksize)),
        max_files=max_files,
        max_rows=max_rows,
        max_mints=max_mints,
    )
    print(
        f"Saved bootstrap dataset: {output_path} | rows={row_count} | processed_rows={processed_rows}"
    )


if __name__ == "__main__":
    main()
