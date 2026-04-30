"""
Backtest all 28,867 pre-mined rule candidates against actual bot PnL data.

Instead of using the historical graduation-rate label, this script re-ranks
every candidate using two signals from the live bot:

  1. Bot PnL label  — was the position profitable? (from DB positions)
  2. Observation coverage — how often does this rule fire on all tokens the
     bot saw? (from events.jsonl entry_rejected + candidate_selected snapshots)

Output: outputs/rules/pump_rule_candidates_bot_ranked.csv
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── paths ────────────────────────────────────────────────────────────────────
CANDIDATES_PATH = ROOT / "outputs/rules/pump_rule_candidates_v2.json"
DB_PATH = ROOT / "data/live/bot_state.db"
EVENTS_PATH = ROOT / "data/live/events.jsonl"
OUTPUT_PATH = ROOT / "outputs/rules/pump_rule_candidates_bot_ranked.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── condition key → runtime feature key mapping (mirrors rule_matcher.py) ────
# unique_buyers_30s_min → wallet_cluster_30s (best available proxy)
# virtual_sol_growth_60s_min → net_flow_sol_60s (bot uses this as proxy)
COND_TO_FEAT: dict[str, tuple[str, str]] = {
    # condition key             feature key          direction (min/max)
    "tx_count_30s_min": ("tx_count_30s", "min"),
    "unique_buyers_30s_min": ("wallet_cluster_30s", "min"),
    "buy_volume_sol_30s_min": ("buy_volume_sol_30s", "min"),
    "buy_sell_ratio_30s_min": ("buy_sell_ratio_30s", "min"),
    "virtual_sol_growth_60s_min": ("net_flow_sol_60s", "min"),
    "top_wallet_buy_share_30s_max": ("top_wallet_buy_share_30s", "max"),
    "wallet_cluster_30s_min": ("wallet_cluster_30s", "min"),
    "volume_sol_30s_min": ("volume_sol_30s", "min"),
    "token_age_sec_max": ("token_age_sec", "max"),
    "price_change_30s_min": ("price_change_30s", "min"),
    "price_change_30s_max": ("price_change_30s", "max"),
}

FEATURE_COLS = list({v[0] for v in COND_TO_FEAT.values()})


# ── helpers ───────────────────────────────────────────────────────────────────


def _load_labeled_features() -> pd.DataFrame:
    """Load runtime features + realized_pnl from DB positions (labeled set)."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, realized_pnl_sol, metadata_json FROM positions WHERE status='CLOSED'"
    ).fetchall()
    conn.close()

    records: list[dict] = []
    for pos_id, pnl, meta_json in rows:
        try:
            meta = json.loads(meta_json) if meta_json else {}
        except Exception:
            continue
        rf = meta.get("runtime_features") or {}
        rec: dict = {
            "position_id": pos_id,
            "realized_pnl_sol": float(pnl),
            "label": 1 if float(pnl) > 0.001 else 0,
        }
        for feat in FEATURE_COLS:
            val = rf.get(feat)
            rec[feat] = float(val) if val is not None else np.nan
        records.append(rec)

    df = pd.DataFrame(records)
    print(f"[labeled] loaded {len(df)} closed positions from DB")
    print(f"  win_rate={df['label'].mean():.1%}  avg_pnl={df['realized_pnl_sol'].mean():.5f}")
    return df


def _load_observation_features(max_events: int = 200_000) -> pd.DataFrame:
    """
    Load feature snapshots from events.jsonl for signal-rate estimation.
    Only entry_rejected and candidate_selected events carry feature_snapshots.
    We cap at max_events to keep memory reasonable.
    """
    EVENT_TYPES = {
        "entry_rejected",
        "candidate_selected",
        "sniper_candidate_selected",
        "candidate_ranked",
    }
    records: list[dict] = []
    loaded = 0
    with open(EVENTS_PATH) as fh:
        for line in fh:
            if loaded >= max_events:
                break
            try:
                ev = json.loads(line.strip())
            except Exception:
                continue
            if ev.get("event_type") not in EVENT_TYPES:
                continue
            snap = ev.get("feature_snapshot")
            if not snap:
                continue
            rec: dict = {}
            for feat in FEATURE_COLS:
                val = snap.get(feat)
                rec[feat] = float(val) if val is not None else np.nan
            records.append(rec)
            loaded += 1

    df = pd.DataFrame(records)
    print(f"[observations] loaded {len(df)} feature snapshots from events.jsonl")
    return df


def _matches_rule_vectorized(df: pd.DataFrame, conditions: dict) -> pd.Series:
    """Return boolean Series: True where all conditions are satisfied."""
    mask = pd.Series(True, index=df.index)
    for cond_key, threshold in conditions.items():
        mapping = COND_TO_FEAT.get(cond_key)
        if mapping is None:
            continue
        feat_col, direction = mapping
        if feat_col not in df.columns:
            continue
        col = df[feat_col]
        # Handle infinity safely
        col_safe = col.replace([np.inf, -np.inf], np.nan)
        if direction == "min":
            # NaN rows fail the min check (conservative)
            mask &= col_safe.fillna(-np.inf) >= float(threshold)
        else:  # max
            mask &= col_safe.fillna(np.inf) <= float(threshold)
    return mask


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("Rule Candidate Backtester — Bot PnL Re-ranking")
    print("=" * 60)

    # Load data
    labeled_df = _load_labeled_features()
    obs_df = _load_observation_features(max_events=300_000)

    # Load candidates
    with open(CANDIDATES_PATH) as fh:
        candidates: list[dict] = json.load(fh)
    print(f"\n[candidates] loaded {len(candidates):,} candidates from {CANDIDATES_PATH.name}")

    n_obs = len(obs_df)
    results: list[dict] = []

    for i, cand in enumerate(candidates):
        if i % 2000 == 0:
            print(f"  evaluating {i:,}/{len(candidates):,} ...", flush=True)

        conditions: dict = cand.get("conditions") or {}
        if not conditions:
            continue

        # ── bot PnL evaluation on labeled positions ──────────────────────────
        match_mask = _matches_rule_vectorized(labeled_df, conditions)
        matched = labeled_df[match_mask]
        bot_support = len(matched)

        if bot_support > 0:
            bot_wins = int(matched["label"].sum())
            bot_win_rate = bot_wins / bot_support
            bot_avg_pnl = float(matched["realized_pnl_sol"].mean())
            bot_total_pnl = float(matched["realized_pnl_sol"].sum())
            bot_avg_win_pnl = (
                float(matched.loc[matched["label"] == 1, "realized_pnl_sol"].mean())
                if bot_wins > 0
                else 0.0
            )
            bot_avg_loss_pnl = (
                float(matched.loc[matched["label"] == 0, "realized_pnl_sol"].mean())
                if (bot_support - bot_wins) > 0
                else 0.0
            )
            bot_expectancy = bot_win_rate * bot_avg_win_pnl + (1 - bot_win_rate) * bot_avg_loss_pnl
        else:
            bot_wins = 0
            bot_win_rate = 0.0
            bot_avg_pnl = 0.0
            bot_total_pnl = 0.0
            bot_avg_win_pnl = 0.0
            bot_avg_loss_pnl = 0.0
            bot_expectancy = 0.0

        # ── signal rate on observed tokens ────────────────────────────────────
        if n_obs > 0:
            obs_mask = _matches_rule_vectorized(obs_df, conditions)
            obs_hits = int(obs_mask.sum())
            obs_signal_rate = obs_hits / n_obs
        else:
            obs_hits = 0
            obs_signal_rate = 0.0

        # ── composite ranking score ───────────────────────────────────────────
        # Penalise rules with < 3 bot observations (unreliable)
        support_weight = min(bot_support / 10.0, 1.0)
        ranking_score = bot_expectancy * support_weight

        results.append(
            {
                "rule_id": cand.get("rule_id", ""),
                "family": cand.get("family", ""),
                # Original mining metrics
                "hist_support": cand.get("support", 0),
                "hist_precision": round(cand.get("precision", 0.0), 6),
                "hist_lift": round(cand.get("lift", 0.0), 4),
                "hist_score": round(cand.get("score", 0.0), 6),
                # Bot PnL metrics
                "bot_support": bot_support,
                "bot_wins": bot_wins,
                "bot_win_rate": round(bot_win_rate, 4),
                "bot_avg_pnl": round(bot_avg_pnl, 6),
                "bot_total_pnl": round(bot_total_pnl, 6),
                "bot_avg_win_pnl": round(bot_avg_win_pnl, 6),
                "bot_avg_loss_pnl": round(bot_avg_loss_pnl, 6),
                "bot_expectancy": round(bot_expectancy, 6),
                "ranking_score": round(ranking_score, 6),
                # Signal rate
                "obs_hits": obs_hits,
                "obs_signal_rate": round(obs_signal_rate, 6),
                # Conditions (for inspection)
                "conditions_json": json.dumps(conditions, sort_keys=True),
                "notes": cand.get("notes", ""),
            }
        )

    out_df = pd.DataFrame(results)

    # Sort: best expectancy first (require at least 1 bot hit)
    ranked = out_df.sort_values(
        ["ranking_score", "bot_win_rate", "hist_lift"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    # Save
    ranked.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[done] saved {len(ranked):,} ranked candidates → {OUTPUT_PATH}")

    # Print top-20 summary
    top = ranked.head(20)
    print("\n── TOP 20 RULES BY BOT EXPECTANCY ──────────────────────────────")
    print(
        f"{'rank':>4} {'rule_id':<18} {'bot_sup':>7} {'win%':>6} {'avg_pnl':>9} "
        f"{'total_pnl':>10} {'hist_lift':>10} {'family'}"
    )
    print("-" * 85)
    for _, r in top.iterrows():
        print(
            f"{r['rank']:>4} {r['rule_id']:<18} {r['bot_support']:>7} "
            f"{r['bot_win_rate']:>6.1%} {r['bot_avg_pnl']:>9.5f} "
            f"{r['bot_total_pnl']:>10.5f} {r['hist_lift']:>10.2f} {r['family']}"
        )

    # Also print rules with ≥5 bot hits for reliability
    reliable = ranked[ranked["bot_support"] >= 5].head(20)
    if not reliable.empty:
        print("\n── RELIABLE RULES (bot_support ≥ 5) ────────────────────────────")
        print(
            f"{'rank':>4} {'rule_id':<18} {'bot_sup':>7} {'win%':>6} {'avg_pnl':>9} "
            f"{'total_pnl':>10} {'obs_sig%':>9} {'family'}"
        )
        print("-" * 90)
        for _, r in reliable.iterrows():
            print(
                f"{r['rank']:>4} {r['rule_id']:<18} {r['bot_support']:>7} "
                f"{r['bot_win_rate']:>6.1%} {r['bot_avg_pnl']:>9.5f} "
                f"{r['bot_total_pnl']:>10.5f} {r['obs_signal_rate']:>9.4%} {r['family']}"
            )

    # Zero-hit summary
    zero_hit = (out_df["bot_support"] == 0).sum()
    print(
        f"\n  Rules with 0 bot hits: {zero_hit:,} / {len(out_df):,} "
        f"({zero_hit / len(out_df):.1%}) — conditions too strict for current market"
    )


if __name__ == "__main__":
    main()
