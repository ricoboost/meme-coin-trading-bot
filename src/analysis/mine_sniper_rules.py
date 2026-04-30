"""
Mine trading rules specifically calibrated for the SNIPER strategy.

The sniper exits at:
  - 8% TP within 75 seconds  → win (sniper_take_profit)
  - timeout at 75s           → loss
  - -10% stop out            → loss

This uses actual sniper trade outcomes instead of historical graduation labels,
so every condition threshold reflects what distinguishes tokens that actually
reach +8% within 75 seconds from those that don't.

Input:  data/live/bot_state.db  (strategy_id='sniper', status='CLOSED')
Output: outputs/rules/sniper_rule_pack.csv  (ready to load in bot via SNIPER_RULE_IDS)

Usage:
  .venv/bin/python3 src/analysis/mine_sniper_rules.py
"""

from __future__ import annotations

import itertools
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data/live/bot_state.db"
OUTPUT_DIR = ROOT / "outputs/rules"
OUTPUT_CSV = OUTPUT_DIR / "sniper_rule_pack.csv"
OUTPUT_JSON = OUTPUT_DIR / "sniper_rule_pack.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Label: only sniper_take_profit counts as a win
# Timeout and stop-out are both losses for sniper purposes
WIN_EXIT_REASON = "sniper_take_profit"

# Minimum thresholds for rule retention
MIN_SUPPORT = 5  # sniper dataset is smaller (359 trades), use lower support
MIN_WIN_RATE = 0.70  # minimum 70% TP hit rate (vs 65% baseline)
MIN_EXPECTANCY = 0.0  # SOL — expectancy filter; sniper uses fixed TP/SL so this is secondary

# Features available in sniper runtime_features (verified from DB)
FEATURE_COLS = [
    "token_age_sec",
    "wallet_cluster_30s",
    "volume_sol_30s",
    "tx_count_30s",
    "tx_count_60s",
    "buy_volume_sol_30s",
    "buy_volume_sol_60s",
    "sell_volume_sol_30s",
    "buy_sell_ratio_30s",
    "net_flow_sol_30s",
    "net_flow_sol_60s",
    "avg_trade_sol_30s",
    "price_change_30s",
    "price_change_60s",
    "top_wallet_buy_share_30s",
    "wallet_cluster_30s",
]

# ── Threshold grids per feature ──────────────────────────────────────────────
# Calibrated from actual sniper data:
#   buy_volume: win_med=273 vs loss_med=227
#   tx_count:   win_med=50  vs loss_med=46
#   token_age:  win_med=46s vs loss_med=52s  ← wins are YOUNGER
#   net_flow:   win_med=273 vs loss_med=224
THRESHOLD_GRIDS: dict[str, list] = {
    # MIN conditions (value must be >= threshold)
    "tx_count_30s": [30, 40, 45, 50, 55, 60, 70, 80],
    "buy_volume_sol_30s": [150, 180, 200, 220, 250, 280, 300, 350],
    "net_flow_sol_30s": [120, 150, 180, 200, 230, 260, 300],
    "wallet_cluster_30s": [20, 25, 30, 35],
    "avg_trade_sol_30s": [3.0, 4.0, 5.0, 6.0, 8.0],
    "buy_volume_sol_60s": [200, 250, 300, 400, 500],
    "net_flow_sol_60s": [150, 200, 250, 300, 400],
    # MAX conditions (value must be <= threshold) — sniper wants YOUNG tokens
    "token_age_sec_max": [40, 45, 50, 55, 60, 65, 70, 80, 100, 120, 150, 180, 240, 300],
}

# Rule families: each defines which MIN features + optional MAX token_age
FAMILIES: dict[str, dict] = {
    # Core: volume + tx + age cap
    "sniper_core": {
        "min": ["tx_count_30s", "buy_volume_sol_30s"],
        "max": ["token_age_sec"],
    },
    # Flow: net flow instead of buy volume
    "sniper_flow": {
        "min": ["tx_count_30s", "net_flow_sol_30s"],
        "max": ["token_age_sec"],
    },
    # Volume60: longer window confirmation
    "sniper_vol60": {
        "min": ["tx_count_30s", "buy_volume_sol_60s"],
        "max": ["token_age_sec"],
    },
    # Pressure: both net_flow + buy_volume
    "sniper_pressure": {
        "min": ["buy_volume_sol_30s", "net_flow_sol_30s"],
        "max": ["token_age_sec"],
    },
    # Whale: avg trade size signals institutional presence
    "sniper_whale": {
        "min": ["buy_volume_sol_30s", "avg_trade_sol_30s"],
        "max": ["token_age_sec"],
    },
    # Age-only: just token freshness + minimal volume
    "sniper_fresh": {"min": ["buy_volume_sol_30s"], "max": ["token_age_sec"]},
}

MAX_RULES = 40


# ── Data loading ──────────────────────────────────────────────────────────────


def load_sniper_positions() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, realized_pnl_sol, metadata_json
        FROM positions
        WHERE status='CLOSED' AND strategy_id='sniper'
        ORDER BY id
    """).fetchall()
    conn.close()

    records: list[dict] = []
    for pos_id, pnl, meta_json in rows:
        try:
            meta = json.loads(meta_json) if meta_json else {}
        except Exception:
            continue
        rf = meta.get("runtime_features") or {}
        if not rf:
            continue

        exit_reason = meta.get("last_exit_reason", "")
        label = 1 if exit_reason == WIN_EXIT_REASON else 0

        rec: dict = {
            "position_id": pos_id,
            "realized_pnl_sol": float(pnl),
            "label": label,
            "exit_reason": exit_reason,
        }
        for col in FEATURE_COLS:
            val = rf.get(col)
            if val is None or val != val:
                rec[col] = np.nan
            else:
                try:
                    f = float(val)
                    rec[col] = np.nan if (f == float("inf") or f == float("-inf")) else f
                except (TypeError, ValueError):
                    rec[col] = np.nan

        # Clip buy_sell_ratio — it saturates at 1000 and is uninformative at extremes
        for col in ["buy_sell_ratio_30s", "buy_sell_ratio_60s"]:
            if col in rec and rec[col] is not None:
                rec[col] = min(rec[col], 1000.0)

        records.append(rec)

    df = pd.DataFrame(records)
    total = len(df)
    wins = int(df["label"].sum())
    print(
        f"[sniper] {total} positions  wins={wins} ({wins / total:.1%})  "
        f"avg_pnl={df['realized_pnl_sol'].mean():.5f}"
    )
    print(f"  exit reasons: {df['exit_reason'].value_counts().to_dict()}")
    return df


# ── Condition evaluation ──────────────────────────────────────────────────────


def _apply_conditions(df: pd.DataFrame, min_conds: dict, max_conds: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for feat, threshold in min_conds.items():
        if feat not in df.columns:
            continue
        mask &= df[feat].fillna(-np.inf) >= float(threshold)
    for feat, threshold in max_conds.items():
        if feat not in df.columns:
            continue
        mask &= df[feat].fillna(np.inf) <= float(threshold)
    return mask


# ── Mining ────────────────────────────────────────────────────────────────────


def _mine_family(df: pd.DataFrame, family_name: str, family_spec: dict) -> list[dict]:
    min_features = family_spec.get("min", [])
    max_features = family_spec.get("max", [])

    # Build grids
    min_grids: list[list] = []
    for feat in min_features:
        grid = []
        if feat in THRESHOLD_GRIDS:
            col = df[feat].dropna()
            for t in THRESHOLD_GRIDS[feat]:
                if (col >= t).sum() >= MIN_SUPPORT:
                    grid.append(t)
        min_grids.append(grid if grid else [0.0])

    max_grids: list[list] = []
    for feat in max_features:
        key = f"{feat}_max"
        grid = []
        if key in THRESHOLD_GRIDS:
            col = df[feat].dropna()
            for t in THRESHOLD_GRIDS[key]:
                if (col <= t).sum() >= MIN_SUPPORT:
                    grid.append(t)
        max_grids.append(grid if grid else [99999.0])

    candidates: list[dict] = []

    for min_combo in itertools.product(*min_grids):
        min_conds = dict(zip(min_features, min_combo))
        for max_combo in itertools.product(*max_grids):
            max_conds = dict(zip(max_features, max_combo))

            mask = _apply_conditions(df, min_conds, max_conds)
            matched = df[mask]
            support = len(matched)
            if support < MIN_SUPPORT:
                continue

            wins = int(matched["label"].sum())
            win_rate = wins / support
            if win_rate < MIN_WIN_RATE:
                continue

            avg_pnl = float(matched["realized_pnl_sol"].mean())

            # Score: win rate is primary; support depth is secondary
            # For sniper, win_rate > support depth because we want consistent TP hits
            score = (
                (0.60 * win_rate)
                + (0.30 * min(support / 30.0, 1.0))
                + (0.10 * min(avg_pnl / 0.02, 1.0))
            )

            candidates.append(
                {
                    "family": family_name,
                    "min_conditions": min_conds,
                    "max_conditions": max_conds,
                    "support": support,
                    "wins": wins,
                    "win_rate": round(win_rate, 4),
                    "avg_pnl_sol": round(avg_pnl, 6),
                    "total_pnl_sol": round(float(matched["realized_pnl_sol"].sum()), 6),
                    "score": round(score, 6),
                }
            )

    return candidates


def _deduplicate(candidates: list[dict]) -> list[dict]:
    """Remove dominated candidates within same family."""
    srt = sorted(candidates, key=lambda x: -x["score"])
    kept: list[dict] = []
    for cand in srt:
        dominated = False
        for k in kept:
            if k["family"] != cand["family"]:
                continue
            k_min = k["min_conditions"]
            c_min = cand["min_conditions"]
            k_max = k["max_conditions"]
            c_max = cand["max_conditions"]
            if set(k_min) != set(c_min) or set(k_max) != set(c_max):
                continue
            # k dominates cand if k is at least as strict on all conditions AND better win rate
            min_tighter = all(k_min.get(f, 0) >= c_min.get(f, 0) for f in c_min)
            max_tighter = all(k_max.get(f, 99999) <= c_max.get(f, 99999) for f in c_max)
            if min_tighter and max_tighter and k["win_rate"] >= cand["win_rate"]:
                dominated = True
                break
        if not dominated:
            kept.append(cand)
    return kept


def _to_runtime_conditions(min_conds: dict, max_conds: dict) -> dict:
    """Map mining feature names to rule_matcher.py condition key names."""
    FEAT_TO_MIN_COND = {
        "tx_count_30s": "tx_count_30s_min",
        "wallet_cluster_30s": "unique_buyers_30s_min",
        "buy_volume_sol_30s": "buy_volume_sol_30s_min",
        "buy_sell_ratio_30s": "buy_sell_ratio_30s_min",
        "net_flow_sol_30s": "virtual_sol_growth_60s_min",  # proxy in rule_matcher
        "net_flow_sol_60s": "virtual_sol_growth_60s_min",
        "avg_trade_sol_30s": "buy_volume_sol_30s_min",  # use as floor
        "buy_volume_sol_60s": "buy_volume_sol_30s_min",
    }
    FEAT_TO_MAX_COND = {
        "token_age_sec": "token_age_sec_max",
    }
    out: dict = {}
    for feat, val in min_conds.items():
        cond_key = FEAT_TO_MIN_COND.get(feat)
        if cond_key:
            out[cond_key] = max(out.get(cond_key, 0.0), val)
    for feat, val in max_conds.items():
        cond_key = FEAT_TO_MAX_COND.get(feat)
        if cond_key:
            # Take the most permissive (highest) max condition if conflict
            if cond_key in out:
                out[cond_key] = max(out[cond_key], val)
            else:
                out[cond_key] = val
    return out


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("Sniper Rule Mining — TP-Calibrated from Live Outcomes")
    print("=" * 60)
    print(f"Label: exit_reason == '{WIN_EXIT_REASON}'")
    print(f"Min support: {MIN_SUPPORT}  Min win rate: {MIN_WIN_RATE:.0%}")
    print()

    df = load_sniper_positions()
    baseline_wr = df["label"].mean()
    print(f"\nBaseline win rate: {baseline_wr:.1%}  (target: >{MIN_WIN_RATE:.0%})\n")

    all_candidates: list[dict] = []
    for family_name, spec in FAMILIES.items():
        print(f"── Family: {family_name} ──", flush=True)
        cands = _mine_family(df, family_name, spec)
        print(f"   raw candidates: {len(cands)}")
        all_candidates.extend(cands)

    print(f"\nTotal raw candidates: {len(all_candidates)}")
    deduped = _deduplicate(all_candidates)
    print(f"After deduplication: {len(deduped)}")

    final = sorted(deduped, key=lambda x: -x["score"])[:MAX_RULES]

    # Assign rule IDs
    for i, c in enumerate(final):
        c["rule_id"] = f"sniper_v1_{i + 1:04d}"

    # Build output DataFrame
    rows: list[dict] = []
    for i, c in enumerate(final):
        runtime_conds = _to_runtime_conditions(c["min_conditions"], c["max_conditions"])
        rows.append(
            {
                "rule_id": c["rule_id"],
                "family": c["family"],
                "support": c["support"],
                "wins": c["wins"],
                "win_rate": c["win_rate"],
                "avg_pnl_sol": c["avg_pnl_sol"],
                "total_pnl_sol": c["total_pnl_sol"],
                "score": c["score"],
                "pack_rank": i + 1,
                "pack_name": "sniper_v1",
                # Runtime-compatible condition columns
                "conditions_obj": json.dumps(runtime_conds),
                "conditions_json": json.dumps(runtime_conds),
                "conditions": str(runtime_conds),
                # Original mining conditions (for inspection)
                "min_conditions": json.dumps(c["min_conditions"]),
                "max_conditions": json.dumps(c["max_conditions"]),
                "notes": (
                    f"Sniper-mined rule. TP hit rate {c['win_rate']:.0%} over {c['support']} sniper trades. "
                    f"Min: {c['min_conditions']}  Max: {c['max_conditions']}"
                ),
                # Fields expected by rules_loader.py
                "precision": c["win_rate"],
                "precision_valid": c["win_rate"],
                "score_valid": c["score"],
                "support_valid": c["support"],
                "lift": round(c["win_rate"] / max(baseline_wr, 0.001), 4),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n[done] saved {len(out_df)} sniper rules → {OUTPUT_CSV}")

    # Summary table
    print("\n── SNIPER RULES SUMMARY ─────────────────────────────────────────────")
    print(
        f"{'#':>3} {'rule_id':<18} {'family':<20} {'sup':>4} {'TP%':>6} "
        f"{'avg_pnl':>8} {'lift':>6}  conditions"
    )
    print("-" * 100)
    for _, r in out_df.iterrows():
        min_c = json.loads(r["min_conditions"])
        max_c = json.loads(r["max_conditions"])
        min_str = "  ".join(
            f"{k.replace('_30s', '').replace('_sol', '').replace('buy_', '')}≥{v}"
            for k, v in min_c.items()
        )
        max_str = "  ".join(f"age≤{v}" for k, v in max_c.items())
        cond_str = f"{min_str}  {max_str}".strip()
        print(
            f"{r['pack_rank']:>3} {r['rule_id']:<18} {r['family']:<20} "
            f"{r['support']:>4} {r['win_rate']:>6.1%} "
            f"{r['avg_pnl_sol']:>8.5f} {r['lift']:>6.2f}  {cond_str}"
        )

    # Top pick recommendation
    print("\n── RECOMMENDED SNIPER_RULE_IDS ──────────────────────────────────────")
    top_rules = out_df.head(5)
    rule_ids = ",".join(top_rules["rule_id"].tolist())
    print(f"SNIPER_RULE_IDS={rule_ids}")
    print()
    print("Or for a single high-conviction rule (highest win rate):")
    best = out_df.nlargest(1, "win_rate").iloc[0]
    print(
        f"SNIPER_RULE_IDS={best['rule_id']}  "
        f"(TP%={best['win_rate']:.0%}, support={best['support']}, lift={best['lift']:.2f}x)"
    )
    print()
    print("To activate: update SNIPER_RULE_IDS in .env and set SNIPER_USE_RUNTIME_RULES=true")
    print("Also update PUMP_RULES_PATH or ensure sniper_rule_pack.csv is used separately.")
    print()
    print("Note: These rules are designed for the sniper's 75s/8%TP profile.")
    print("      They should NOT replace main strategy rules.")


if __name__ == "__main__":
    main()
