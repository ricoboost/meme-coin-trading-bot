"""Tier A derivable-feature backfill + Spearman ranking on ml_samples.jsonl.

Reads the existing 40-feature labeled dataset, adds 9 derivable Tier A features
that are strict transforms of the existing columns (no per-trade data needed),
writes ml_samples_v2.jsonl, then prints:

  - Spearman correlation of each new feature vs pnl_sol
  - Spearman correlation of each baseline feature vs pnl_sol (for comparison)
  - Max |Spearman| of each new feature vs any of the 40 baseline features
  - Shortlist: features with |rho_vs_pnl| >= SHORTLIST_MIN_ABS_RHO and
    max |rho_vs_baseline| <= SHORTLIST_MAX_REDUNDANCY

Usage:
    python -m src.analysis.backfill_features_v2
    python -m src.analysis.backfill_features_v2 --input data/live/ml_samples.jsonl \
        --output data/live/ml_samples_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EPS = 1e-9

SHORTLIST_MIN_ABS_RHO = 0.05
SHORTLIST_MAX_REDUNDANCY = 0.7

BASELINE_FEATURES = [
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
    "entry_price_sol",
    "price_change_30s",
    "price_change_60s",
    "tracked_wallet_present_60s",
    "tracked_wallet_count_60s",
    "tracked_wallet_score_sum_60s",
    "triggering_wallet_score",
    "aggregated_wallet_score",
    "candidate_score",
    "rule_support",
    "rule_hit_2x_rate",
    "rule_hit_5x_rate",
    "rule_rug_rate",
    "strategy_is_sniper",
    "lane_shock",
    "lane_recovery",
    "regime_negative_shock_recovery",
    "regime_high_cluster_recovery",
    "regime_momentum_burst",
    "regime_unknown",
    "market_regime_score",
    "recent_win_rate",
    "candidate_rate_5min_norm",
]

NEW_FEATURES = [
    "dollar_ofi_30s",
    "dollar_ofi_60s",
    "cluster_growth_30_120",
    "age_norm_tx_intensity_30s",
    "age_norm_vol_intensity_30s",
    "buy_vol_share_30s",
    "flow_acceleration_30_60",
    "price_change_acceleration",
    "net_flow_per_tx_30s",
]


def _finite(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def compute_derivable(fm: dict[str, Any]) -> dict[str, float]:
    buy30 = _finite(fm.get("buy_volume_sol_30s"))
    sell30 = _finite(fm.get("sell_volume_sol_30s"))
    buy60 = _finite(fm.get("buy_volume_sol_60s"))
    sell60 = _finite(fm.get("sell_volume_sol_60s"))
    vol30 = _finite(fm.get("volume_sol_30s"))
    vol60 = _finite(fm.get("volume_sol_60s"))
    cluster30 = _finite(fm.get("wallet_cluster_30s"))
    cluster120 = _finite(fm.get("wallet_cluster_120s"))
    tx30 = _finite(fm.get("tx_count_30s"))
    age = _finite(fm.get("token_age_sec"))
    nf30 = _finite(fm.get("net_flow_sol_30s"))
    pc30 = _finite(fm.get("price_change_30s"))
    pc60 = _finite(fm.get("price_change_60s"))

    # vol_30_60 = volume in the [30s, 60s) slice. volume_sol_60s is cumulative
    # over last 60s (and includes the last 30s). Clip at zero to handle noisy rows.
    vol_30_60 = max(vol60 - vol30, 0.0)

    return {
        "dollar_ofi_30s": (buy30 - sell30) / max(buy30 + sell30, EPS),
        "dollar_ofi_60s": (buy60 - sell60) / max(buy60 + sell60, EPS),
        "cluster_growth_30_120": cluster30 / max(cluster120, 1.0),
        "age_norm_tx_intensity_30s": tx30 / max(age, 1.0),
        "age_norm_vol_intensity_30s": vol30 / max(age, 1.0),
        "buy_vol_share_30s": buy30 / max(vol30, EPS),
        "flow_acceleration_30_60": vol30 / max(vol_30_60, EPS),
        "price_change_acceleration": pc30 - pc60,
        "net_flow_per_tx_30s": nf30 / max(tx30, 1.0),
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def backfill(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        fm = row.get("feature_map")
        if not isinstance(fm, dict):
            continue
        derived = compute_derivable(fm)
        new_fm = dict(fm)
        new_fm.update(derived)
        new_row = dict(row)
        new_row["feature_map"] = new_fm
        enriched.append(new_row)
    return enriched


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        fm = row.get("feature_map") or {}
        rec = {name: _finite(fm.get(name)) for name in BASELINE_FEATURES + NEW_FEATURES}
        rec["pnl_sol"] = _finite(row.get("pnl_sol"))
        rec["label"] = int(row.get("label") or 0)
        rec["strategy_id"] = str(row.get("strategy_id") or "")
        records.append(rec)
    return pd.DataFrame.from_records(records)


def spearman_vs_pnl(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = {}
    y = df["pnl_sol"].to_numpy()
    for col in columns:
        x = df[col].to_numpy()
        if np.all(x == x[0]):
            out[col] = float("nan")
            continue
        rho, _ = spearmanr(x, y)
        out[col] = float(rho) if rho == rho else float("nan")
    return pd.Series(out).sort_values(key=lambda s: s.abs(), ascending=False)


def redundancy_against_baseline(df: pd.DataFrame) -> pd.DataFrame:
    # Spearman of each new feature vs each baseline feature.
    rows = []
    for new in NEW_FEATURES:
        x = df[new].to_numpy()
        worst = 0.0
        worst_col = ""
        per_baseline = {}
        for base in BASELINE_FEATURES:
            y = df[base].to_numpy()
            if np.all(x == x[0]) or np.all(y == y[0]):
                per_baseline[base] = float("nan")
                continue
            rho, _ = spearmanr(x, y)
            rho = float(rho) if rho == rho else 0.0
            per_baseline[base] = rho
            if abs(rho) > abs(worst):
                worst = rho
                worst_col = base
        rows.append(
            {
                "new_feature": new,
                "max_abs_rho_vs_baseline": abs(worst),
                "most_redundant_with": worst_col,
                "signed_rho": worst,
            }
        )
    return pd.DataFrame(rows).sort_values("max_abs_rho_vs_baseline", ascending=False)


def print_report(df: pd.DataFrame) -> None:
    print(f"\nDataset: {len(df)} rows")
    print(f"  by strategy: {df['strategy_id'].value_counts().to_dict()}")
    print(f"  label=1 count: {int(df['label'].sum())} / {len(df)}")
    print(
        f"  pnl_sol: mean={df['pnl_sol'].mean():.4f}  median={df['pnl_sol'].median():.4f}  "
        f"std={df['pnl_sol'].std():.4f}"
    )

    print("\n=== Spearman vs pnl_sol — NEW features ===")
    new_vs_pnl = spearman_vs_pnl(df, NEW_FEATURES)
    for name, rho in new_vs_pnl.items():
        print(f"  {rho:+.4f}   {name}")

    print("\n=== Spearman vs pnl_sol — BASELINE features (top 15 by |rho|) ===")
    base_vs_pnl = spearman_vs_pnl(df, BASELINE_FEATURES)
    for name, rho in base_vs_pnl.head(15).items():
        print(f"  {rho:+.4f}   {name}")

    print("\n=== Redundancy of NEW features against baseline 40 ===")
    red = redundancy_against_baseline(df)
    for _, r in red.iterrows():
        print(
            f"  {r['max_abs_rho_vs_baseline']:.3f}   {r['new_feature']:<32} "
            f"most redundant with {r['most_redundant_with']} (rho={r['signed_rho']:+.3f})"
        )

    print("\n=== Shortlist — high |rho_vs_pnl| AND low redundancy ===")
    print(
        f"    criteria: |rho_vs_pnl| >= {SHORTLIST_MIN_ABS_RHO} "
        f"AND max |rho_vs_baseline| <= {SHORTLIST_MAX_REDUNDANCY}"
    )
    red_map = red.set_index("new_feature")["max_abs_rho_vs_baseline"].to_dict()
    shortlisted = []
    for name, rho in new_vs_pnl.items():
        red_score = red_map.get(name, 1.0)
        if abs(rho) >= SHORTLIST_MIN_ABS_RHO and red_score <= SHORTLIST_MAX_REDUNDANCY:
            shortlisted.append((name, rho, red_score))
    if not shortlisted:
        print("  (none — no feature clears both thresholds)")
    else:
        for name, rho, red_score in sorted(shortlisted, key=lambda t: -abs(t[1])):
            print(f"  {rho:+.4f}   {name:<32}  max_redundancy={red_score:.3f}")

    # Per-strategy sanity check on the two lanes.
    for strat in ["main", "sniper"]:
        sub = df[df["strategy_id"] == strat]
        if len(sub) < 50:
            continue
        print(f"\n=== Spearman vs pnl_sol — NEW features (strategy={strat}, n={len(sub)}) ===")
        sub_rho = spearman_vs_pnl(sub, NEW_FEATURES)
        for name, rho in sub_rho.items():
            print(f"  {rho:+.4f}   {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/live/ml_samples.jsonl")
    parser.add_argument("--output", default="data/live/ml_samples_v2.jsonl")
    parser.add_argument(
        "--skip-write",
        action="store_true",
        help="Only compute + rank; don't write ml_samples_v2.jsonl.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    rows = load_rows(in_path)
    if not rows:
        print(f"no rows in {in_path}")
        return 1

    enriched = backfill(rows)
    print(
        f"loaded {len(rows)} rows; enriched {len(enriched)} with {len(NEW_FEATURES)} new features"
    )

    if not args.skip_write:
        write_rows(enriched, out_path)
        print(f"wrote {out_path}")

    df = build_frame(enriched)
    print_report(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
