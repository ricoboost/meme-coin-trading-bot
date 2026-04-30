"""Build a trusted labeled parquet from live ml_samples for v2 rule mining.

IMPORTANT:
  Pre-2026-04-20 ml_samples + DB positions are corrupted by paper-mode biased
  math — never mine from them. This script filters to rows recorded on/after
  the trusted cutoff and writes a parquet compatible with
  `src/analysis/pump_mine_rules_pnl_v3.py`.

Usage:
  .venv/bin/python src/analysis/build_trusted_labeled_parquet.py [--min-per-class N]

The miner will refuse to produce rules if --min-per-class isn't met for the
target strategy. Default floor is 30 wins + 30 losses per strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SAMPLES_PATH = ROOT / "data/live/ml_samples.jsonl"
OUTPUT_PATH = ROOT / "data/gold/bot_pnl_features_v3.parquet"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TRUSTED_CUTOFF = "2026-04-20T00:00:00"
PNL_POSITIVE_THRESHOLD = 0.001

# Columns the v3 miner threshold grids reference. Any feature it doesn't see
# becomes NaN in the grid search (rule simply can't match those rows).
EXPECTED_FEATURES = [
    "token_age_sec",
    "wallet_cluster_30s",
    "wallet_cluster_120s",
    "volume_sol_30s",
    "volume_sol_60s",
    "tx_count_30s",
    "tx_count_60s",
    "buy_volume_sol_30s",
    "buy_volume_sol_60s",
    "sell_volume_sol_30s",
    "sell_volume_sol_60s",
    "buy_sell_ratio_30s",
    "buy_sell_ratio_60s",
    "net_flow_sol_30s",
    "net_flow_sol_60s",
    "avg_trade_sol_30s",
    "avg_trade_sol_60s",
    "price_change_30s",
    "price_change_60s",
    "triggering_wallet_score",
    "aggregated_wallet_score",
    "tracked_wallet_present_60s",
    "tracked_wallet_count_60s",
    "tracked_wallet_score_sum_60s",
    # Sniper-rule features added 2026-04-20 to feature_map
    "buy_streak_count_30s",
    "buy_streak_count_60s",
    "sell_tx_count_30s",
    "round_trip_wallet_count_30s",
    "round_trip_wallet_ratio_30s",
    "round_trip_volume_sol_30s",
    "real_volume_sol_30s",
    "real_buy_volume_sol_30s",
]


def _coerce(v):
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if v is None:
        return np.nan
    try:
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return np.nan
        return f
    except (TypeError, ValueError):
        return np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--min-per-class",
        type=int,
        default=30,
        help="Minimum wins AND losses required per strategy to call mining viable",
    )
    ap.add_argument(
        "--cutoff",
        default=TRUSTED_CUTOFF,
        help="ISO timestamp floor for trusted samples",
    )
    ap.add_argument(
        "--strategy",
        default=None,
        help="Filter to a single strategy (sniper|main). Default: all.",
    )
    args = ap.parse_args()

    if not SAMPLES_PATH.exists():
        print(f"ERROR: {SAMPLES_PATH} not found")
        return 1

    rows = []
    skipped_old = 0
    with SAMPLES_PATH.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            ts = r.get("recorded_at", "")
            if ts < args.cutoff:
                skipped_old += 1
                continue
            if args.strategy and r.get("strategy_id") != args.strategy:
                continue
            fm = r.get("feature_map") or {}
            rec = {col: _coerce(fm.get(col)) for col in EXPECTED_FEATURES}
            rec["position_id"] = r.get("position_id")
            rec["strategy_id"] = r.get("strategy_id") or "unknown"
            pnl = r.get("pnl_sol")
            rec["pnl_sol"] = float(pnl) if pnl is not None else np.nan
            rec["label"] = int(
                1 if (pnl is not None and float(pnl) > PNL_POSITIVE_THRESHOLD) else 0
            )
            rec["source"] = "db_position"
            rec["recorded_at"] = ts
            rows.append(rec)

    df = pd.DataFrame(rows)
    print(f"skipped pre-{args.cutoff} (paper-era biased): {skipped_old}")
    print(f"trusted samples:    {len(df)}")
    if len(df):
        for strat in sorted(df["strategy_id"].unique()):
            sub = df[df["strategy_id"] == strat]
            wins = int(sub["label"].sum())
            losses = int((sub["label"] == 0).sum())
            print(
                f"  {strat:<10} n={len(sub):<4} wins={wins:<4} losses={losses:<4} "
                f"win_rate={wins / max(len(sub), 1):.1%}"
            )

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nwrote {OUTPUT_PATH}")

    # Mining-viability verdict
    ready = {}
    for strat in ("sniper", "main"):
        sub = df[df["strategy_id"] == strat] if len(df) else df
        ready[strat] = (
            int(sub["label"].sum()) >= args.min_per_class
            and int((sub["label"] == 0).sum()) >= args.min_per_class
        )
    print(f"\nmining viable (≥{args.min_per_class}/class):")
    for s, ok in ready.items():
        print(f"  {s}: {'YES' if ok else 'NO — keep collecting'}")

    if any(ready.values()):
        print("\nnext step: .venv/bin/python src/analysis/pump_mine_rules_pnl_v3.py")
    else:
        print("\nnext step: run this script again after more live trades close.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
