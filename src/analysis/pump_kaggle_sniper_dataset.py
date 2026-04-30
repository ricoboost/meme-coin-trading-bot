"""Build sniper-labeled per-mint feature snapshots from the Kaggle pump.fun dump.

Mirrors ``pump_kaggle_dataset.py`` but replaces the ``has_graduated`` label with
short-horizon price-pump labels that match the sniper lane's exit profile
(``default_sniper``: TP +8%, SL −8%, 75 s max hold).

For each mint we compute the standard 30 s / 60 s feature snapshot, then walk
forward through the sorted swap list to capture the max price reached in
[window_end, window_end + horizon]. The entry price is the mint's ``vs/vt``
reserve ratio at ``window_end``.

Labels added per (mint, window_sec) row:
    sniper_tp_08_75s    True if max price in next 75 s ≥ 1.08 × entry
    sniper_tp_15_75s    True if max price in next 75 s ≥ 1.15 × entry
    sniper_tp_08_120s   True if max price in next 120 s ≥ 1.08 × entry
    max_price_mult_75s  float — max(future_price) / entry_price in 75 s

Input:
  data/kaggle/archive/train.csv
  data/kaggle/archive/chunk_*.csv
Output:
  data/kaggle/processed/sniper_labeled_snapshots.parquet

Usage:
  .venv/bin/python -m src.analysis.pump_kaggle_sniper_dataset --limit 15000
  .venv/bin/python -m src.analysis.pump_kaggle_sniper_dataset --full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.pump_kaggle_dataset import (
    WINDOWS_SEC,
    _collect_swaps,
    _compute_snapshot,
    _load_labels,
)


OUTPUT_PATH = Path("data/kaggle/processed/sniper_labeled_snapshots.parquet")

# Horizons (seconds after window end) to evaluate future-price pumps. 75s
# matches the sniper max-hold; 120s gives a slightly looser window that also
# catches entries where the trailing-stop / TP logic fires late.
FUTURE_HORIZONS_SEC: tuple[int, ...] = (75, 120)
TP_THRESHOLDS: tuple[tuple[str, float, int], ...] = (
    ("sniper_tp_08_75s", 1.08, 75),
    ("sniper_tp_15_75s", 1.15, 75),
    ("sniper_tp_08_120s", 1.08, 120),
)


def _price(event: dict[str, Any]) -> float:
    vt = event.get("vt_balance") or 0
    if vt <= 0:
        return 0.0
    return float(event["vs_balance"]) / float(vt)


def _build_snapshots_with_sniper_labels(
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

            # Sniper label: future price in (window_end, window_end + horizon].
            entry_price = _price(window_events[-1])
            horizon_stats: dict[int, float] = {}
            if entry_price > 0:
                for horizon in FUTURE_HORIZONS_SEC:
                    h_cutoff = cutoff + horizon
                    max_p = entry_price
                    for ev in mint_events:
                        t = ev["block_time"]
                        if t <= cutoff:
                            continue
                        if t > h_cutoff:
                            break
                        p = _price(ev)
                        if p > max_p:
                            max_p = p
                    horizon_stats[horizon] = max_p / entry_price
            else:
                for horizon in FUTURE_HORIZONS_SEC:
                    horizon_stats[horizon] = 1.0

            labels_out: dict[str, Any] = {}
            for label_name, mult_threshold, horizon in TP_THRESHOLDS:
                labels_out[label_name] = bool(horizon_stats.get(horizon, 1.0) >= mult_threshold)
            for horizon in FUTURE_HORIZONS_SEC:
                labels_out[f"max_price_mult_{horizon}s"] = float(horizon_stats.get(horizon, 1.0))
            labels_out["entry_price"] = float(entry_price)

            rows.append(
                {
                    **row_base,
                    "window_sec": window_sec,
                    "token_age_sec": float(token_age_sec),
                    **snap,
                    **labels_out,
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
    rows = _build_snapshots_with_sniper_labels(labels, swaps)
    if not rows:
        print("no snapshots produced — nothing to write", flush=True)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)

    w0 = WINDOWS_SEC[0]
    win_df = df.loc[df["window_sec"] == w0]
    print(
        f"wrote {len(df):,} snapshots ({df['mint'].nunique():,} mints) → {args.out}",
        flush=True,
    )
    for label_name, mult_threshold, horizon in TP_THRESHOLDS:
        share = win_df[label_name].mean()
        print(
            f"  label {label_name:22s} (×{mult_threshold} in {horizon}s @ {w0}s snapshot): "
            f"{share:.3%}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
