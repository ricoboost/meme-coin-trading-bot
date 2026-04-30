"""Build labeled per-mint feature snapshots from the Kaggle pump.fun dump.

Input:
  data/kaggle/archive/train.csv         (labels + first-slot + graduation-slot)
  data/kaggle/archive/chunk_*.csv       (raw swap events)

Output:
  data/kaggle/processed/labeled_snapshots.parquet

Each output row = (mint, window_sec) — features computed on the prefix of
swaps within ``window_sec`` of the mint's first observed swap. Feature keys
mirror ``RuntimeFeatures`` so mined rules drop directly into rule_matcher
without a translation layer.

Usage:
  .venv/bin/python -m src.analysis.pump_kaggle_dataset --limit 15000
  .venv/bin/python -m src.analysis.pump_kaggle_dataset --full
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


KAGGLE_DIR = Path("data/kaggle/archive")
TRAIN_CSV = KAGGLE_DIR / "train.csv"
CHUNK_GLOB = "chunk_*.csv"
OUTPUT_PATH = Path("data/kaggle/processed/labeled_snapshots.parquet")

WINDOWS_SEC: tuple[int, ...] = (30, 60)
CHUNK_READ_SIZE = 250_000
# Swap volumes use quote_coin_amount in lamports (1e9 lamports = 1 SOL).
LAMPORTS_PER_SOL = 1_000_000_000


def _load_labels(sample_limit: int | None, seed: int) -> dict[str, dict[str, Any]]:
    """Return mint → {has_graduated, slot_min, slot_graduated, first_block_time (None)}."""
    df = pd.read_csv(TRAIN_CSV)
    df = df[df["is_valid"] == True]  # noqa: E712 — pandas bool column
    print(
        f"train.csv valid mints: {len(df):,}  graduated: {int(df['has_graduated'].sum()):,}",
        flush=True,
    )
    if sample_limit is not None:
        positives = df[df["has_graduated"]]["mint"].tolist()
        negatives = df[~df["has_graduated"]]["mint"].tolist()
        rng = random.Random(seed)
        rng.shuffle(positives)
        rng.shuffle(negatives)
        half = sample_limit // 2
        pos_sample = positives[: min(half, len(positives))]
        neg_sample = negatives[: sample_limit - len(pos_sample)]
        picked = set(pos_sample + neg_sample)
        df = df[df["mint"].isin(picked)].copy()
        print(
            f"sampled to {len(df):,} mints (pos={len(pos_sample):,}, neg={len(neg_sample):,})",
            flush=True,
        )
    return {
        str(row["mint"]): {
            "has_graduated": bool(row["has_graduated"]),
            "slot_min": int(row["slot_min"]),
            "slot_graduated": None
            if pd.isna(row["slot_graduated"])
            else int(row["slot_graduated"]),
        }
        for _, row in df.iterrows()
    }


def _collect_swaps(target_mints: set[str]) -> dict[str, list[dict[str, Any]]]:
    """Stream chunks and collect swap rows keyed by mint."""
    swaps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_rows = 0
    hit_rows = 0
    t0 = time.time()
    chunk_paths = sorted(KAGGLE_DIR.glob(CHUNK_GLOB))
    if not chunk_paths:
        raise FileNotFoundError(f"no chunks in {KAGGLE_DIR}")
    for idx, path in enumerate(chunk_paths, start=1):
        chunk_t0 = time.time()
        for df in pd.read_csv(path, chunksize=CHUNK_READ_SIZE):
            total_rows += len(df)
            hits = df[df["base_coin"].isin(target_mints)]
            if hits.empty:
                continue
            hit_rows += len(hits)
            # Vectorized extraction: iterrows is ~100x slower at full scale.
            # Coerce to datetime64[ns] explicitly — pandas 3.x defaults strings
            # to datetime64[us], which would make `// 1e9` return values 1000×
            # too small and collapse every window to effectively 0 seconds.
            bt = (
                pd.to_datetime(hits["block_time"])
                .astype("datetime64[ns]")
                .astype("int64")
                .to_numpy()
                // 1_000_000_000
            )
            mints_arr = hits["base_coin"].to_numpy()
            slots = hits["slot"].to_numpy()
            tx_idxs = hits["tx_idx"].to_numpy()
            wallets = hits["signing_wallet"].to_numpy()
            dirs = hits["direction"].to_numpy()
            bases = hits["base_coin_amount"].to_numpy()
            quotes = hits["quote_coin_amount"].to_numpy()
            vts = hits["virtual_token_balance_after"].to_numpy()
            vss = hits["virtual_sol_balance_after"].to_numpy()
            for i in range(len(hits)):
                swaps[mints_arr[i]].append(
                    {
                        "block_time": float(bt[i]),
                        "slot": int(slots[i]),
                        "tx_idx": int(tx_idxs[i]),
                        "signing_wallet": wallets[i],
                        "direction": dirs[i],
                        "base_amount": int(bases[i]),
                        "quote_amount_lamports": int(quotes[i]),
                        "vt_balance": int(vts[i]),
                        "vs_balance": int(vss[i]),
                    }
                )
        elapsed = time.time() - chunk_t0
        print(
            f"  [{idx}/{len(chunk_paths)}] {path.name} scanned in {elapsed:.1f}s  "
            f"(cumulative hits={hit_rows:,}/{total_rows:,})",
            flush=True,
        )
    print(
        f"collected {hit_rows:,} swap rows for {len(swaps):,} mints in {time.time() - t0:.1f}s",
        flush=True,
    )
    return swaps


def _trade_flow_stats(events: list[dict[str, Any]]) -> tuple[float, float, int, int]:
    """Mirror token_activity.trade_flow_stats for offline data."""
    if not events:
        return 0.0, 0.0, 0, 0
    sizes = sorted(e["quote_amount_lamports"] / LAMPORTS_PER_SOL for e in events)
    n = len(sizes)
    total = sum(sizes)
    if n >= 2 and total > 0:
        weighted = sum((i + 1) * size for i, size in enumerate(sizes))
        gini = (2.0 * weighted) / (n * total) - (n + 1) / n
    else:
        gini = 0.0
    gaps = [events[i]["block_time"] - events[i - 1]["block_time"] for i in range(1, n)]
    gaps = [g for g in gaps if g >= 0]
    if len(gaps) >= 2:
        mean_gap = sum(gaps) / len(gaps)
        var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
        cv = (var**0.5) / mean_gap if mean_gap > 0 else 0.0
    else:
        cv = 0.0
    max_streak = 0
    streak_count = 0
    current = 0
    prev_side = None
    for e in events:
        side = e["direction"]
        if side == "buy":
            current += 1
            if prev_side != "buy":
                streak_count += 1
        else:
            if current > max_streak:
                max_streak = current
            current = 0
        prev_side = side
    if current > max_streak:
        max_streak = current
    return float(gini), float(cv), int(max_streak), int(streak_count)


def _compute_snapshot(
    events_window: list[dict[str, Any]],
    mint_events: list[dict[str, Any]],
    window_sec: int,
) -> dict[str, Any]:
    """Compute a feature snapshot from the in-window swap prefix.

    ``mint_events`` is the full sorted swap list for the mint — used only to
    derive full-history anchors (first price).
    """
    win = events_window
    buys = [e for e in win if e["direction"] == "buy"]
    sells = [e for e in win if e["direction"] == "sell"]
    buy_vol = sum(e["quote_amount_lamports"] for e in buys) / LAMPORTS_PER_SOL
    sell_vol = sum(e["quote_amount_lamports"] for e in sells) / LAMPORTS_PER_SOL
    total_vol = buy_vol + sell_vol

    wallet_cluster = len({e["signing_wallet"] for e in buys})

    # Round-trip / real-volume (wallets that both bought and sold within the window)
    buy_wallets = {e["signing_wallet"] for e in buys}
    sell_wallets = {e["signing_wallet"] for e in sells}
    roundtrip_wallets = buy_wallets & sell_wallets
    roundtrip_volume = (
        sum(e["quote_amount_lamports"] for e in win if e["signing_wallet"] in roundtrip_wallets)
        / LAMPORTS_PER_SOL
    )
    real_volume = max(0.0, total_vol - roundtrip_volume)
    real_buy_wallets = buy_wallets - roundtrip_wallets
    real_buy_volume = (
        sum(e["quote_amount_lamports"] for e in buys if e["signing_wallet"] in real_buy_wallets)
        / LAMPORTS_PER_SOL
    )

    # Price = SOL/token = vs_balance / vt_balance (reserve ratio). Use first and
    # last reserve snapshots within the window.
    def _price(e):
        return e["vs_balance"] / e["vt_balance"] if e["vt_balance"] > 0 else 0.0

    first_price = _price(win[0]) if win else 0.0
    last_price = _price(win[-1]) if win else 0.0
    price_change = (last_price / first_price - 1.0) if first_price > 0 else 0.0

    # Virtual-SOL growth: use first and last vs_balance in window
    vs_start = win[0]["vs_balance"] if win else 0
    vs_end = win[-1]["vs_balance"] if win else 0
    virtual_sol_growth = (vs_end - vs_start) / LAMPORTS_PER_SOL

    gini, cv, max_streak, streak_count = _trade_flow_stats(win)

    net_flow = buy_vol - sell_vol
    avg_trade = total_vol / len(win) if win else 0.0
    bsr = buy_vol / sell_vol if sell_vol > 0 else (float("inf") if buy_vol > 0 else 0.0)
    if bsr == float("inf"):
        bsr = 999.0  # cap matches live runtime sentinel behavior

    return {
        f"tx_count_{window_sec}s": len(win),
        f"buy_tx_count_{window_sec}s": len(buys),
        f"sell_tx_count_{window_sec}s": len(sells),
        f"volume_sol_{window_sec}s": float(total_vol),
        f"buy_volume_sol_{window_sec}s": float(buy_vol),
        f"sell_volume_sol_{window_sec}s": float(sell_vol),
        f"net_flow_sol_{window_sec}s": float(net_flow),
        f"avg_trade_sol_{window_sec}s": float(avg_trade),
        f"buy_sell_ratio_{window_sec}s": float(bsr),
        f"wallet_cluster_{window_sec}s": wallet_cluster,
        f"round_trip_wallet_count_{window_sec}s": len(roundtrip_wallets),
        f"round_trip_wallet_ratio_{window_sec}s": float(len(roundtrip_wallets) / len(buy_wallets))
        if buy_wallets
        else 0.0,
        f"round_trip_volume_sol_{window_sec}s": float(roundtrip_volume),
        f"real_volume_sol_{window_sec}s": float(real_volume),
        f"real_buy_volume_sol_{window_sec}s": float(real_buy_volume),
        f"trade_size_gini_{window_sec}s": float(gini),
        f"inter_arrival_cv_{window_sec}s": float(cv),
        f"max_consecutive_buy_streak_{window_sec}s": int(max_streak),
        f"buy_streak_count_{window_sec}s": int(streak_count),
        f"price_change_{window_sec}s": float(price_change),
        f"virtual_sol_growth_{window_sec}s": float(virtual_sol_growth),
    }


def _build_snapshots(
    labels: dict[str, dict[str, Any]],
    swaps: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped_no_swaps = 0
    for mint, meta in labels.items():
        mint_events = swaps.get(mint) or []
        if not mint_events:
            skipped_no_swaps += 1
            continue
        mint_events.sort(key=lambda s: (s["slot"], s["tx_idx"]))
        first_time = mint_events[0]["block_time"]
        first_slot = mint_events[0]["slot"]
        row_base = {
            "mint": mint,
            "has_graduated": meta["has_graduated"],
            "slot_min": meta["slot_min"],
            "slot_graduated": meta["slot_graduated"],
            "total_swaps_observed": len(mint_events),
            "first_observed_slot": first_slot,
            "first_observed_block_time": first_time,
        }
        for window_sec in WINDOWS_SEC:
            cutoff = first_time + window_sec
            window_events = [e for e in mint_events if e["block_time"] <= cutoff]
            if not window_events:
                continue
            snap = _compute_snapshot(window_events, mint_events, window_sec)
            token_age_sec = window_events[-1]["block_time"] - first_time
            rows.append(
                {
                    **row_base,
                    "window_sec": window_sec,
                    "token_age_sec": float(token_age_sec),
                    **snap,
                }
            )
    if skipped_no_swaps:
        print(
            f"  skipped {skipped_no_swaps} mints with no matching swaps in the chunks",
            flush=True,
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Sample N mints (stratified). Omit for full run.",
    )
    ap.add_argument(
        "--full", action="store_true", help="Process all valid mints (ignores --limit)."
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()

    sample_limit = None if args.full else args.limit
    labels = _load_labels(sample_limit=sample_limit, seed=args.seed)
    swaps = _collect_swaps(set(labels))
    rows = _build_snapshots(labels, swaps)
    if not rows:
        print("no snapshots produced — nothing to write", flush=True)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    grad_share = df.loc[df["window_sec"] == WINDOWS_SEC[0], "has_graduated"].mean()
    print(
        f"wrote {len(df):,} snapshots ({df['mint'].nunique():,} mints) → {args.out}  "
        f"graduated share @ {WINDOWS_SEC[0]}s = {grad_share:.3%}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
