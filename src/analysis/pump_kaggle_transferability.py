"""Phase 4: transferability gate.

Does a rule mined on Kaggle pre-PumpSwap data keep predictive power on our
current ml_samples_v2.jsonl? Pass criteria per rule:

  * matches >= MIN_MATCHES (default 20)
  * lift on our live label >= MIN_LIFT (default 1.5)

Survivors are exported to outputs/rules/kaggle_transferred_v1.csv in the
rule-pack schema, ready for canary deploy via SNIPER_RULE_IDS/MAIN_RULE_IDS.

Usage:
  .venv/bin/python -m src.analysis.pump_kaggle_transferability \
      --mined outputs/rules/kaggle_mined_30s.csv \
      --samples data/live/ml_samples_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


DEFAULT_MINED = Path("outputs/rules/kaggle_mined_30s.csv")
DEFAULT_SAMPLES = Path("data/live/ml_samples_v2.jsonl")
DEFAULT_OUT = Path("outputs/rules/kaggle_transferred_v1.csv")

MIN_MATCHES = 20
MIN_LIFT = 1.5
TOP_MINED_BY_HOLDOUT_LIFT = 200  # evaluate top-N from mining; skip tail


def _load_samples(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            fm = rec.get("feature_map") or {}
            fm["label"] = int(rec.get("label", 0))
            fm["pnl_sol"] = float(rec.get("pnl_sol", 0.0))
            fm["strategy_id"] = rec.get("strategy_id") or ""
            rows.append(fm)
    return pd.DataFrame(rows)


def _evaluate(df: pd.DataFrame, feature: str, direction: str, threshold: float) -> dict:
    if feature not in df.columns:
        return {"matches": 0, "precision": 0.0, "lift": 0.0, "in_scope": False}
    vals = df[feature].astype(float).to_numpy()
    label = df["label"].to_numpy()
    base = label.mean() if len(label) else 0.0
    if direction == "min":
        mask = vals >= threshold
    elif direction == "max":
        mask = vals <= threshold
    else:
        return {"matches": 0, "precision": 0.0, "lift": 0.0, "in_scope": False}
    matches = int(mask.sum())
    if matches == 0:
        return {
            "matches": 0,
            "precision": 0.0,
            "lift": 0.0,
            "in_scope": True,
            "positives": 0,
        }
    positives = int(label[mask].sum())
    precision = positives / matches
    lift = (precision / base) if base > 0 else 0.0
    return {
        "matches": matches,
        "positives": positives,
        "precision": precision,
        "lift": lift,
        "in_scope": True,
    }


def _format_rule_pack_row(idx: int, mined_row: pd.Series, live_eval: dict) -> dict:
    """Convert a surviving rule into the pump rule-pack CSV schema."""
    feature = mined_row["feature"]
    direction = mined_row["direction"]
    threshold = float(mined_row["threshold"])
    cond_key = f"{feature}_{direction}"
    conditions = {cond_key: threshold}
    precision = float(live_eval["precision"])
    rule_id = f"kaggle_v1_{idx:03d}"
    return {
        "rule_id": rule_id,
        "family": "kaggle_graduation_v1",
        "pack_name": "kaggle_mined_v1",
        "pack_rank": idx,
        "conditions_obj": json.dumps(conditions),
        "support_valid": int(live_eval["matches"]),
        "precision_valid": precision,
        "score_valid": float(live_eval["lift"]),
        "support": int(mined_row["test_matches"]),
        "precision": float(mined_row["test_precision"]),
        "score": float(mined_row["test_lift"]),
        "exit_profile": "mature_pair_v1",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", type=Path, default=DEFAULT_MINED)
    ap.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-matches", type=int, default=MIN_MATCHES)
    ap.add_argument("--min-lift", type=float, default=MIN_LIFT)
    ap.add_argument("--top", type=int, default=TOP_MINED_BY_HOLDOUT_LIFT)
    ap.add_argument(
        "--strategy-filter",
        default="",
        help="If set, filter ml_samples by strategy_id before evaluating.",
    )
    args = ap.parse_args()

    mined = pd.read_csv(args.mined)
    mined = mined.sort_values("test_lift", ascending=False).head(args.top).reset_index(drop=True)
    print(f"evaluating top {len(mined)} mined rules (by holdout lift)", flush=True)

    samples = _load_samples(args.samples)
    if args.strategy_filter:
        before = len(samples)
        samples = samples[samples["strategy_id"] == args.strategy_filter].reset_index(drop=True)
        print(
            f"filtered samples: {before} -> {len(samples)} (strategy={args.strategy_filter!r})",
            flush=True,
        )
    base = samples["label"].mean() if len(samples) else 0.0
    print(f"samples: {len(samples)}  base win-rate: {base:.3%}", flush=True)

    results: list[dict] = []
    for _, row in mined.iterrows():
        ev = _evaluate(samples, row["feature"], row["direction"], float(row["threshold"]))
        results.append(
            {
                "feature": row["feature"],
                "direction": row["direction"],
                "threshold": float(row["threshold"]),
                "kaggle_test_lift": float(row["test_lift"]),
                "kaggle_test_matches": int(row["test_matches"]),
                "live_matches": ev.get("matches", 0),
                "live_positives": ev.get("positives", 0),
                "live_precision": ev.get("precision", 0.0),
                "live_lift": ev.get("lift", 0.0),
                "in_scope": ev.get("in_scope", False),
            }
        )

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("live_lift", ascending=False).reset_index(drop=True)

    surviving = res_df[
        (res_df["in_scope"])
        & (res_df["live_matches"] >= args.min_matches)
        & (res_df["live_lift"] >= args.min_lift)
    ].reset_index(drop=True)

    print("\n=== Transferability summary ===", flush=True)
    print(
        f"in-scope rules (feature present in samples): {int(res_df['in_scope'].sum())}",
        flush=True,
    )
    print(
        f"passing gate (matches>={args.min_matches} AND lift>={args.min_lift}): {len(surviving)}",
        flush=True,
    )
    if len(res_df):
        print("\nTop 25 by live lift:")
        print(res_df.head(25).to_string(index=False), flush=True)

    if surviving.empty:
        print("\nNO SURVIVORS — rule pack NOT written", flush=True)
        return 0

    rule_rows = [
        _format_rule_pack_row(
            idx + 1,
            pd.Series(
                {
                    "feature": r["feature"],
                    "direction": r["direction"],
                    "threshold": r["threshold"],
                    "test_matches": r["kaggle_test_matches"],
                    "test_precision": r["live_precision"],
                    "test_lift": r["live_lift"],
                }
            ),
            {
                "matches": r["live_matches"],
                "precision": r["live_precision"],
                "lift": r["live_lift"],
            },
        )
        for idx, r in surviving.iterrows()
    ]
    out_df = pd.DataFrame(rule_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"\nwrote {len(out_df)} rule-pack rows → {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
