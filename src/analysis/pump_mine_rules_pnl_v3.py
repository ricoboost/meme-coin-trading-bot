"""
Mine NEW trading rules using bot PnL labels (not historical graduation rate).

Input:  data/gold/bot_pnl_features_v3.parquet  (from extract_bot_labeled_dataset.py)
Output: outputs/rules/pump_rule_packs_pnl_v3.csv  (ready to load in the bot)

Key differences from pump_mine_rules_v2.py:
  - Label = pnl > 0.001 SOL  (bot profit after fees)
  - Uses 5 feature families including net_flow and avg_trade_sol
  - Discovers rules that work in the CURRENT market, not historical dataset
  - Produces runtime-ready conditions matching what rule_matcher.py handles

Usage:
  .venv/bin/python3 src/analysis/pump_mine_rules_pnl_v3.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

INPUT_PATH = ROOT / "data/gold/bot_pnl_features_v3.parquet"
OUTPUT_DIR = ROOT / "outputs/rules"
OUTPUT_CSV = OUTPUT_DIR / "pump_rule_packs_pnl_v3.csv"
OUTPUT_JSON = OUTPUT_DIR / "pump_rule_packs_pnl_v3.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Only use confirmed-labeled rows (source=db_position) for rule mining
# Rejected events are too noisy as negatives without confirmed outcomes
SOURCE_FILTER = "db_position"

# Minimum threshold for a rule to be retained
MIN_SUPPORT = 3  # min labeled trades matching the rule
MIN_WIN_RATE = 0.60  # min win rate on bot trades
MIN_EXPECTANCY = 0.002  # min avg PnL per trade (SOL)
MAX_RULES = 60  # max rules in final pack

# --- Threshold grids per feature -------------------------------------------
THRESHOLD_GRIDS: dict[str, list[float]] = {
    # tx_count: bot trades have range 16-447, p50=54 — use mid-to-high thresholds
    "tx_count_30s": [20, 30, 40, 50, 60, 80, 100, 150],
    # wallet_cluster: p50=33 for both wins/losses — not very discriminative
    "wallet_cluster_30s": [15, 20, 25, 30, 35, 40, 50],
    # buy_volume: win_p25=159, loss_p25=95 — discriminative above ~100
    "buy_volume_sol_30s": [50, 80, 100, 120, 150, 180, 200, 250, 300, 400],
    # buy_sell_ratio: MOST DISCRIMINATIVE — win_p25=165 vs loss_p25=1.07
    # Existing active rule uses 2.49 (catches everything), need stricter values
    "buy_sell_ratio_30s": [2.0, 5.0, 10.0, 30.0, 50.0, 100.0, 200.0, 500.0],
    # net_flow: closely correlated with buy_volume
    "net_flow_sol_30s": [50.0, 80.0, 100.0, 150.0, 200.0, 300.0, 400.0],
    # avg_trade_sol: win_p25=4.02, loss_p25=2.52 — discriminative above ~3
    "avg_trade_sol_30s": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    # 60s versions for longer-window confirmation
    "net_flow_sol_60s": [80.0, 150.0, 200.0, 300.0, 500.0, 700.0],
    "buy_volume_sol_60s": [100.0, 150.0, 200.0, 300.0, 500.0, 700.0],
}

# --- Rule families: list of (anchor features, optional extra conditions) ----
# Each family defines which features are used as MIN thresholds.
FAMILIES: dict[str, list[str]] = {
    "momentum_core": ["tx_count_30s", "wallet_cluster_30s", "buy_volume_sol_30s"],
    "momentum_flow": [
        "tx_count_30s",
        "wallet_cluster_30s",
        "buy_volume_sol_30s",
        "buy_sell_ratio_30s",
    ],
    "momentum_netflow": ["tx_count_30s", "buy_volume_sol_30s", "net_flow_sol_30s"],
    "momentum_pressure": [
        "tx_count_30s",
        "wallet_cluster_30s",
        "net_flow_sol_30s",
        "buy_sell_ratio_30s",
    ],
    "momentum_whale": ["tx_count_30s", "buy_volume_sol_30s", "avg_trade_sol_30s"],
    "momentum_flow60": ["tx_count_30s", "buy_volume_sol_60s", "net_flow_sol_60s"],
}


# ── condition evaluation (vectorized) ────────────────────────────────────────


def _apply_min_conditions(df: pd.DataFrame, conditions: dict) -> pd.Series:
    """Return boolean mask: rows satisfying all MIN conditions."""
    mask = pd.Series(True, index=df.index)
    for feat, threshold in conditions.items():
        if feat not in df.columns:
            continue
        col = df[feat]
        mask &= col.fillna(-np.inf) >= float(threshold)
    return mask


# ── mining core ──────────────────────────────────────────────────────────────


def _mine_family(df_labeled: pd.DataFrame, family: str, features: list[str]) -> list[dict]:
    """Enumerate all threshold combinations for one family and evaluate each."""
    # Build grid: for each feature, list of candidate threshold values
    grids = []
    for feat in features:
        thresholds = []
        if feat in THRESHOLD_GRIDS:
            # Only include thresholds that have at least some observations above them
            col = df_labeled[feat].dropna()
            for t in THRESHOLD_GRIDS[feat]:
                if (col >= t).sum() >= MIN_SUPPORT:
                    thresholds.append(t)
        if not thresholds:
            thresholds = [0.0]  # dummy — will be skipped if no hits
        grids.append(thresholds)

    candidates: list[dict] = []
    total_combos = 1
    for g in grids:
        total_combos *= len(g)

    for combo in itertools.product(*grids):
        conditions = dict(zip(features, combo))
        mask = _apply_min_conditions(df_labeled, conditions)
        matched = df_labeled[mask]
        support = len(matched)
        if support < MIN_SUPPORT:
            continue
        wins = int(matched["label"].sum())
        win_rate = wins / support
        if win_rate < MIN_WIN_RATE:
            continue
        avg_pnl = float(matched["pnl_sol"].mean())
        if avg_pnl < MIN_EXPECTANCY:
            continue
        total_pnl = float(matched["pnl_sol"].sum())
        avg_win_pnl = (
            float(matched.loc[matched["label"] == 1, "pnl_sol"].mean()) if wins > 0 else 0.0
        )
        avg_loss_pnl = (
            float(matched.loc[matched["label"] == 0, "pnl_sol"].mean())
            if (support - wins) > 0
            else 0.0
        )
        expectancy = win_rate * avg_win_pnl + (1 - win_rate) * avg_loss_pnl
        # Composite score: blend expectancy + win_rate + support depth
        score = (0.40 * expectancy / 0.05) + (0.35 * win_rate) + (0.15 * min(support / 20.0, 1.0))

        candidates.append(
            {
                "family": family,
                "conditions": conditions,
                "support": support,
                "wins": wins,
                "win_rate": round(win_rate, 4),
                "avg_pnl": round(avg_pnl, 6),
                "total_pnl": round(total_pnl, 6),
                "avg_win_pnl": round(avg_win_pnl, 6),
                "avg_loss_pnl": round(avg_loss_pnl, 6),
                "expectancy": round(expectancy, 6),
                "score": round(score, 6),
            }
        )

    return candidates


def _deduplicate(candidates: list[dict]) -> list[dict]:
    """Remove strictly dominated candidates (same or looser conditions, worse metrics)."""
    # Sort by score desc
    srt = sorted(candidates, key=lambda x: -x["score"])
    kept: list[dict] = []
    for cand in srt:
        dominated = False
        for k in kept:
            if k["family"] != cand["family"]:
                continue
            # Check if k's conditions are all >= cand's (k is strictly tighter or equal)
            k_conds = k["conditions"]
            c_conds = cand["conditions"]
            if set(k_conds) != set(c_conds):
                continue
            if all(k_conds[f] >= c_conds[f] for f in c_conds) and k["win_rate"] >= cand["win_rate"]:
                dominated = True
                break
        if not dominated:
            kept.append(cand)
    return kept


def _assign_rule_ids(candidates: list[dict], prefix: str = "pnl_v3") -> list[dict]:
    for i, c in enumerate(candidates):
        c["rule_id"] = f"{prefix}_{i + 1:04d}"
    return candidates


def _to_runtime_conditions(conditions: dict) -> dict:
    """Map mining feature names to rule_matcher.py condition key names."""
    FEAT_TO_COND = {
        "tx_count_30s": "tx_count_30s_min",
        "wallet_cluster_30s": "unique_buyers_30s_min",
        "buy_volume_sol_30s": "buy_volume_sol_30s_min",
        "buy_sell_ratio_30s": "buy_sell_ratio_30s_min",
        "net_flow_sol_30s": "virtual_sol_growth_60s_min",  # proxy used by rule_matcher
        "net_flow_sol_60s": "virtual_sol_growth_60s_min",
        "avg_trade_sol_30s": "buy_volume_sol_30s_min",  # use as floor (safe over-write)
        "buy_volume_sol_60s": "buy_volume_sol_30s_min",
        "avg_trade_sol_60s": "buy_volume_sol_30s_min",
    }
    out: dict = {}
    for feat, val in conditions.items():
        cond_key = FEAT_TO_COND.get(feat)
        if cond_key:
            # Take max if multiple features map to same condition key
            if cond_key in out:
                out[cond_key] = max(out[cond_key], val)
            else:
                out[cond_key] = val
    return out


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("PnL-Labeled Rule Mining v3")
    print("=" * 60)

    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run extract_bot_labeled_dataset.py first.")
        sys.exit(1)

    df_all = pd.read_parquet(INPUT_PATH)
    df_labeled = df_all[df_all["source"] == SOURCE_FILTER].copy()
    print(
        f"Labeled rows: {len(df_labeled)}  "
        f"win_rate={df_labeled['label'].mean():.1%}  "
        f"avg_pnl={df_labeled['pnl_sol'].mean():.5f}"
    )
    print(
        f"Mining parameters: min_support={MIN_SUPPORT}  min_win_rate={MIN_WIN_RATE}  "
        f"min_expectancy={MIN_EXPECTANCY}"
    )

    all_candidates: list[dict] = []
    for family, features in FAMILIES.items():
        print(f"\n── Family: {family} ({features}) ──", flush=True)
        cands = _mine_family(df_labeled, family, features)
        print(f"   raw candidates: {len(cands)}")
        all_candidates.extend(cands)

    print(f"\nTotal raw candidates: {len(all_candidates)}")
    deduped = _deduplicate(all_candidates)
    print(f"After deduplication: {len(deduped)}")

    # Sort and cap
    final = sorted(deduped, key=lambda x: -x["score"])[:MAX_RULES]
    final = _assign_rule_ids(final)

    # Build output DataFrame
    rows: list[dict] = []
    for i, c in enumerate(final):
        runtime_conds = _to_runtime_conditions(c["conditions"])
        rows.append(
            {
                "rule_id": c["rule_id"],
                "family": c["family"],
                "support": c["support"],
                "wins": c["wins"],
                "win_rate": c["win_rate"],
                "avg_pnl_sol": c["avg_pnl"],
                "total_pnl_sol": c["total_pnl"],
                "avg_win_pnl_sol": c["avg_win_pnl"],
                "avg_loss_pnl_sol": c["avg_loss_pnl"],
                "expectancy_sol": c["expectancy"],
                "score": c["score"],
                "pack_rank": i + 1,
                "pack_name": "pnl_v3",
                # Runtime-compatible condition columns (for rules_loader.py)
                "conditions_obj": json.dumps(runtime_conds),
                "conditions_json": json.dumps(runtime_conds),
                "conditions": str(runtime_conds),
                # Original mining conditions (for inspection)
                "mining_conditions": json.dumps(c["conditions"]),
                "notes": f"PnL-mined rule. Win rate {c['win_rate']:.0%} over {c['support']} bot trades.",
                # Dummy fields expected by rules_loader.py
                "precision": c["win_rate"],
                "precision_valid": c["win_rate"],
                "score_valid": c["score"],
                "support_valid": c["support"],
                "lift": round(c["win_rate"] / 0.686, 4)
                if c["win_rate"] > 0
                else 0,  # relative to baseline win rate
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    # Save JSON too
    with open(OUTPUT_JSON, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n[done] saved {len(out_df)} rules → {OUTPUT_CSV}")

    # Print summary
    print("\n── MINED RULES SUMMARY ──────────────────────────────────────────")
    print(
        f"{'#':>3} {'rule_id':<16} {'family':<22} {'sup':>4} {'win%':>6} "
        f"{'avg_pnl':>8} {'total_pnl':>10}  conditions"
    )
    print("-" * 100)
    for _, r in out_df.iterrows():
        conds = json.loads(r["mining_conditions"])
        cond_str = "  ".join(
            f"{k.replace('_30s', '').replace('_sol', '').replace('buy_', '')}≥{v}"
            for k, v in conds.items()
        )
        print(
            f"{r['pack_rank']:>3} {r['rule_id']:<16} {r['family']:<22} "
            f"{r['support']:>4} {r['win_rate']:>6.1%} "
            f"{r['avg_pnl_sol']:>8.5f} {r['total_pnl_sol']:>10.5f}  {cond_str}"
        )

    print("\nTo activate these rules, update your .env:")
    print("  PUMP_RULES_PATH=outputs/rules/pump_rule_packs_pnl_v3.csv")
    print("Or merge with existing rules to run both packs.")


if __name__ == "__main__":
    main()
