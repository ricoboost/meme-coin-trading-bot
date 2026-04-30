"""
Build a PnL-labeled feature dataset from the live bot's own trading data.

Sources:
  1. DB positions   — runtime_features + realized_pnl_sol (611 labeled samples)
  2. events.jsonl   — feature_snapshots from entry_rejected and candidate events
                      (unlabeled, used as the "rejected" class)

Output: data/gold/bot_pnl_features_v3.parquet

Columns match what pump_mine_rules_pnl_v3.py expects:
  - All feature columns
  - label: 1 = profitable (pnl > 0.001), 0 = loss/rejected
  - pnl_sol: realized PnL (NaN for rejected/unlabeled rows)
  - source: 'db_position' | 'entry_rejected' | 'candidate_selected'
  - strategy_id
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

DB_PATH = ROOT / "data/live/bot_state.db"
EVENTS_PATH = ROOT / "data/live/events.jsonl"
OUTPUT_DIR = ROOT / "data/gold"
OUTPUT_PATH = OUTPUT_DIR / "bot_pnl_features_v3.parquet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Positive PnL threshold — must exceed fees + noise
PNL_POSITIVE_THRESHOLD = 0.001

# Feature columns to extract (union of all relevant features)
FEATURE_COLS = [
    "token_age_sec",
    "wallet_cluster_30s",
    "wallet_cluster_120s",
    "volume_sol_30s",
    "volume_sol_60s",
    "tx_count_30s",
    "tx_count_60s",
    "buy_tx_count_30s",
    "buy_tx_count_60s",
    "sell_tx_count_30s",
    "sell_tx_count_60s",
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
    "entry_price_sol",
    "top_wallet_buy_share_30s",
    "tracked_wallet_present_60s",
    "tracked_wallet_count_60s",
    "tracked_wallet_score_sum_60s",
    "triggering_wallet_score",
    "aggregated_wallet_score",
]


def _feat_from_dict(d: dict) -> dict:
    rec: dict = {}
    for col in FEATURE_COLS:
        val = d.get(col)
        if isinstance(val, bool):
            rec[col] = 1.0 if val else 0.0
        elif val is None or val != val:  # None or NaN
            rec[col] = np.nan
        else:
            try:
                f = float(val)
                rec[col] = np.nan if (f == float("inf") or f == float("-inf")) else f
            except (TypeError, ValueError):
                rec[col] = np.nan
    return rec


# ── 1. DB positions (labeled) ─────────────────────────────────────────────────


def load_db_positions() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, strategy_id, realized_pnl_sol, metadata_json
        FROM positions
        WHERE status = 'CLOSED'
    """).fetchall()
    conn.close()

    records: list[dict] = []
    for pos_id, strategy_id, pnl, meta_json in rows:
        try:
            meta = json.loads(meta_json) if meta_json else {}
        except Exception:
            continue
        rf = meta.get("runtime_features") or {}
        if not rf:
            continue
        rec = _feat_from_dict(rf)
        rec["position_id"] = int(pos_id)
        rec["strategy_id"] = strategy_id or "main"
        rec["pnl_sol"] = float(pnl)
        rec["label"] = int(1 if float(pnl) > PNL_POSITIVE_THRESHOLD else 0)
        rec["source"] = "db_position"
        records.append(rec)

    df = pd.DataFrame(records)
    print(
        f"[db_positions]  {len(df)} rows  win_rate={df['label'].mean():.1%}  "
        f"avg_pnl={df['pnl_sol'].mean():.5f}"
    )
    return df


# ── 2. events.jsonl observations ──────────────────────────────────────────────


def load_event_observations(max_rows: int = 500_000) -> pd.DataFrame:
    ACCEPTED_TYPES = {
        "entry_rejected": "entry_rejected",
        "candidate_selected": "candidate_selected",
        "sniper_candidate_selected": "candidate_selected",
        "candidate_ranked": "candidate_ranked",
    }
    records: list[dict] = []
    loaded = 0

    with open(EVENTS_PATH) as fh:
        for line in fh:
            if loaded >= max_rows:
                break
            try:
                ev = json.loads(line.strip())
            except Exception:
                continue
            ev_type = ev.get("event_type", "")
            src = ACCEPTED_TYPES.get(ev_type)
            if src is None:
                continue
            snap = ev.get("feature_snapshot")
            if not snap:
                continue
            rec = _feat_from_dict(snap)
            rec["position_id"] = None
            rec["strategy_id"] = ev.get("strategy_id") or "unknown"
            rec["pnl_sol"] = np.nan
            # entry_rejected = label 0 (rejected for a reason), others = unknown (NaN)
            rec["label"] = 0 if ev_type == "entry_rejected" else np.nan
            rec["source"] = src
            records.append(rec)
            loaded += 1

    df = pd.DataFrame(records)
    print(f"[events]        {len(df)} rows  ({df['source'].value_counts().to_dict()})")
    return df


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("Bot PnL Labeled Dataset Builder")
    print("=" * 60)

    db_df = load_db_positions()
    ev_df = load_event_observations()

    combined = pd.concat([db_df, ev_df], ignore_index=True)

    # Clip extreme buy_sell_ratio values (Infinity already replaced, but cap outliers)
    for col in ["buy_sell_ratio_30s", "buy_sell_ratio_60s"]:
        if col in combined.columns:
            combined[col] = combined[col].clip(upper=1000.0)

    # Save
    combined.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n[done] saved {len(combined):,} rows → {OUTPUT_PATH}")
    print(f"  labeled rows (source=db_position): {(combined['source'] == 'db_position').sum()}")
    print(f"  labeled win rate (db only): {db_df['label'].mean():.1%}")
    print(f"  feature columns: {len(FEATURE_COLS)}")

    # Stats on key features
    print("\n── Feature coverage in labeled set ──")
    for col in [
        "tx_count_30s",
        "buy_volume_sol_30s",
        "buy_sell_ratio_30s",
        "net_flow_sol_30s",
    ]:
        if col in db_df.columns:
            s = db_df[col].dropna()
            print(
                f"  {col}: min={s.min():.2f}  p50={s.median():.2f}  "
                f"p95={s.quantile(0.95):.2f}  max={s.max():.2f}"
            )


if __name__ == "__main__":
    main()
