"""Mine sniper-lane pump rules from the sniper-labeled Kaggle snapshots.

Reuses the lift + holdout machinery from ``pump_kaggle_mine.py`` but targets a
short-horizon price-pump label (``sniper_tp_08_75s``) instead of the
graduation label. Writes the rule-pack CSV schema directly so the output is
loadable by ``rules_loader`` with no transferability post-step.

Output: ``outputs/rules/kaggle_sniper_v1.csv``.

Usage:
  .venv/bin/python -m src.analysis.pump_kaggle_sniper_mine \
      --input data/kaggle/processed/sniper_labeled_snapshots.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.pump_kaggle_mine import (
    MIN_HOLDOUT_N,
    _feature_columns,
    _rule_stats,
)


DEFAULT_INPUT = Path("data/kaggle/processed/sniper_labeled_snapshots.parquet")
DEFAULT_OUTPUT = Path("outputs/rules/kaggle_sniper_v1.csv")

# Sniper is a lift-vs-base play. Base rate of a 30s-age token pumping 8% in
# 75s is much higher than the 1.16% graduation rate, so lift thresholds need
# to be lower than the graduation miner's 3.0 / 2.5.
MIN_TRAIN_LIFT = 1.40
MIN_HOLDOUT_LIFT = 1.30
MIN_TRAIN_PRECISION = 0.0
MIN_HOLDOUT_PRECISION = 0.0

QUANTILES: tuple[float, ...] = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98)

RULE_PACK_COLUMNS: tuple[str, ...] = (
    "rule_id",
    "family",
    "support",
    "support_valid",
    "precision",
    "precision_valid",
    "recall",
    "f1",
    "lift",
    "score",
    "score_valid",
    "pack_name",
    "pack_rank",
    "conditions_obj",
    "conditions_json",
    "conditions",
    "exit_profile",
    "notes",
)


def mine(
    df: pd.DataFrame,
    *,
    window_sec: int,
    label_col: str,
    seed: int,
    min_train_lift: float,
    min_holdout_lift: float,
    min_holdout_n: int,
) -> pd.DataFrame:
    df = df[df["window_sec"] == window_sec].reset_index(drop=True)
    # ``_split_by_mint`` uses a module-level constant LABEL_COL, so do the
    # split manually here to support a custom label column.
    mints = df[["mint", label_col]].drop_duplicates("mint")
    rng = np.random.default_rng(seed)
    mints = mints.sample(frac=1.0, random_state=rng.integers(1 << 31)).reset_index(drop=True)
    pos_ids = mints[mints[label_col]]["mint"].tolist()
    neg_ids = mints[~mints[label_col]]["mint"].tolist()
    split_pos = int(len(pos_ids) * 0.70)
    split_neg = int(len(neg_ids) * 0.70)
    train_mints = set(pos_ids[:split_pos] + neg_ids[:split_neg])
    train = df[df["mint"].isin(train_mints)].copy()
    test = df[~df["mint"].isin(train_mints)].copy()
    train_label = train[label_col].astype(bool).to_numpy()
    test_label = test[label_col].astype(bool).to_numpy()
    train_base = train_label.mean() if len(train_label) else 0.0
    test_base = test_label.mean() if len(test_label) else 0.0
    print(
        f"split {window_sec}s: train={len(train)} (pos={int(train_label.sum())}) "
        f"  test={len(test)} (pos={int(test_label.sum())})",
        flush=True,
    )
    print(f"base rate: train={train_base:.4%}  holdout={test_base:.4%}", flush=True)

    feature_cols = _feature_columns(df, window_sec)
    print(f"candidate features ({len(feature_cols)}): {feature_cols}", flush=True)

    rows: list[dict] = []
    for col in feature_cols:
        train_vals = train[col].to_numpy(dtype=float)
        test_vals = test[col].to_numpy(dtype=float)
        pos_vals = train_vals[train_label]
        neg_vals = train_vals[~train_label]
        if len(pos_vals) < 20:
            continue
        q_pos = np.quantile(pos_vals, QUANTILES)
        q_neg = np.quantile(neg_vals, QUANTILES) if len(neg_vals) >= 20 else np.array([])

        def _try(threshold: float, direction: str, origin: str) -> None:
            if direction == "min":
                mtr = train_vals >= threshold
                mte = test_vals >= threshold
            else:
                mtr = train_vals <= threshold
                mte = test_vals <= threshold
            tr = _rule_stats(mtr, train_label, train_base)
            te = _rule_stats(mte, test_label, test_base)
            if (
                tr["lift"] >= min_train_lift
                and te["lift"] >= min_holdout_lift
                and te["matches"] >= min_holdout_n
                and tr["precision"] >= MIN_TRAIN_PRECISION
                and te["precision"] >= MIN_HOLDOUT_PRECISION
            ):
                rows.append(
                    {
                        "feature": col,
                        "direction": direction,
                        "threshold": float(threshold),
                        "threshold_origin": origin,
                        "train_matches": tr["matches"],
                        "train_precision": tr["precision"],
                        "train_lift": tr["lift"],
                        "test_matches": te["matches"],
                        "test_precision": te["precision"],
                        "test_lift": te["lift"],
                    }
                )

        for q in q_pos:
            _try(float(q), "min", "pos")
            _try(float(q), "max", "pos")
        for q in q_neg:
            _try(float(q), "min", "neg")
            _try(float(q), "max", "neg")

    return pd.DataFrame(rows)


def _format_pack_row(idx: int, row: pd.Series, label_col: str) -> dict:
    feature = str(row["feature"])
    direction = str(row["direction"])
    threshold = float(row["threshold"])
    cond_key = f"{feature}_{direction}"
    conditions = {cond_key: threshold}
    rule_id = f"sniper_kg_v1_{idx:03d}"
    notes = (
        f"Kaggle sniper mining label={label_col} "
        f"holdout lift={float(row['test_lift']):.2f}x "
        f"on {int(row['test_matches'])} mints"
    )
    cond_json = json.dumps(conditions)
    return {
        "rule_id": rule_id,
        "family": "kaggle_sniper_v1",
        "support": int(row["test_matches"]),
        "support_valid": int(row["test_matches"]),
        "precision": float(row["test_precision"]),
        "precision_valid": float(row["test_precision"]),
        "recall": 0.0,
        "f1": 0.0,
        "lift": float(row["test_lift"]),
        "score": float(row["test_lift"]),
        "score_valid": float(row["test_lift"]),
        "pack_name": "kaggle_sniper_v1",
        "pack_rank": idx,
        "conditions_obj": cond_json,
        "conditions_json": cond_json,
        "conditions": cond_json,
        "exit_profile": "default_sniper",
        "notes": notes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--label", default="sniper_tp_08_75s")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-train-lift", type=float, default=MIN_TRAIN_LIFT)
    ap.add_argument("--min-holdout-lift", type=float, default=MIN_HOLDOUT_LIFT)
    ap.add_argument("--min-holdout-n", type=int, default=MIN_HOLDOUT_N)
    ap.add_argument(
        "--top",
        type=int,
        default=10,
        help="Keep at most N top rules in the final pack.",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    if args.label not in df.columns:
        print(
            f"label column {args.label!r} not present; available labels: "
            f"{[c for c in df.columns if c.startswith('sniper_tp_')]}",
            flush=True,
        )
        return 2
    print(f"loaded {len(df):,} snapshots from {args.input}", flush=True)
    rules = mine(
        df,
        window_sec=args.window,
        label_col=args.label,
        seed=args.seed,
        min_train_lift=args.min_train_lift,
        min_holdout_lift=args.min_holdout_lift,
        min_holdout_n=args.min_holdout_n,
    )
    if rules.empty:
        print("no rules passed the lift + holdout gates", flush=True)
        return 0
    rules = rules.sort_values(["test_lift", "test_matches"], ascending=[False, False]).reset_index(
        drop=True
    )
    print(f"\n{len(rules)} candidate rules passed:", flush=True)
    print(rules.head(25).to_string(index=False), flush=True)

    kept = rules.head(args.top).reset_index(drop=True)
    pack_rows = [_format_pack_row(idx + 1, row, args.label) for idx, row in kept.iterrows()]
    out = pd.DataFrame(pack_rows, columns=list(RULE_PACK_COLUMNS))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nwrote {len(out)} rule-pack rows → {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
