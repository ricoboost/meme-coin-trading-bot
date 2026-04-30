"""Mine graduation-predictive rules from the Kaggle snapshot dataset.

Input: data/kaggle/processed/labeled_snapshots.parquet (from pump_kaggle_dataset.py)
Output: outputs/rules/kaggle_mined_v1.csv in the rule-pack CSV schema.

Discipline:
  * 70/30 train/holdout split, stratified on has_graduated.
  * Each candidate rule = single feature + direction (_min/_max) + threshold.
  * Keep rules where:
      - train lift  >= MIN_TRAIN_LIFT     (default 3.0)
      - holdout lift >= MIN_HOLDOUT_LIFT  (default 2.5)
      - holdout matches >= MIN_HOLDOUT_N  (default 50)
  * Threshold search: per feature, evaluate quantile grid over the train set.
  * Prune: drop rules whose condition key isn't in rule_matcher.KNOWN_FEATURES
    (so every survivor is directly loadable by rules_loader).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.strategy.rule_matcher import KNOWN_FEATURES


INPUT_PARQUET = Path("data/kaggle/processed/labeled_snapshots.parquet")
OUTPUT_CSV = Path("outputs/rules/kaggle_mined_v1.csv")

LABEL_COL = "has_graduated"
GROUP_COL = "mint"  # split by mint, not snapshot, so 30s+60s rows for the same mint stay together
DEFAULT_WINDOW = 30

MIN_TRAIN_LIFT = 3.0
MIN_HOLDOUT_LIFT = 2.5
MIN_HOLDOUT_N = 50
MIN_HOLDOUT_PRECISION = 0.0  # 0 = disabled; tune up if needed

QUANTILES: tuple[float, ...] = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98)


def _split_by_mint(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 70/30 split at the mint level."""
    mints = df[[GROUP_COL, LABEL_COL]].drop_duplicates(GROUP_COL)
    rng = np.random.default_rng(seed)
    mints = mints.sample(frac=1.0, random_state=rng.integers(1 << 31)).reset_index(drop=True)
    # stratified: 70% of positives + 70% of negatives in train
    pos = mints[mints[LABEL_COL]][GROUP_COL].tolist()
    neg = mints[~mints[LABEL_COL]][GROUP_COL].tolist()
    split_pos = int(len(pos) * 0.70)
    split_neg = int(len(neg) * 0.70)
    train_mints = set(pos[:split_pos] + neg[:split_neg])
    train = df[df[GROUP_COL].isin(train_mints)].copy()
    test = df[~df[GROUP_COL].isin(train_mints)].copy()
    return train, test


def _feature_columns(df: pd.DataFrame, window_sec: int) -> list[str]:
    """Return known-to-matcher feature columns for the given window."""
    suffix = f"_{window_sec}s"
    cols = []
    for c in df.columns:
        if not c.endswith(suffix):
            continue
        if c not in KNOWN_FEATURES:
            continue
        cols.append(c)
    return cols


def _rule_stats(
    mask: np.ndarray,
    label: np.ndarray,
    base_rate: float,
) -> dict[str, float]:
    matches = int(mask.sum())
    if matches == 0:
        return {"matches": 0, "precision": 0.0, "lift": 0.0, "positives": 0}
    positives = int(label[mask].sum())
    precision = positives / matches
    lift = (precision / base_rate) if base_rate > 0 else 0.0
    return {
        "matches": matches,
        "precision": precision,
        "lift": lift,
        "positives": positives,
    }


def mine(
    df: pd.DataFrame,
    *,
    window_sec: int,
    seed: int,
    min_train_lift: float,
    min_holdout_lift: float,
    min_holdout_n: int,
) -> pd.DataFrame:
    df = df[df["window_sec"] == window_sec].reset_index(drop=True)
    train, test = _split_by_mint(df, seed=seed)
    print(
        f"split {window_sec}s: train={len(train)} (pos={train[LABEL_COL].sum()}) "
        f"  test={len(test)} (pos={test[LABEL_COL].sum()})",
        flush=True,
    )

    train_base = train[LABEL_COL].mean()
    test_base = test[LABEL_COL].mean()
    print(f"base rate: train={train_base:.4%}  holdout={test_base:.4%}", flush=True)

    feature_cols = _feature_columns(df, window_sec)
    print(f"candidate features ({len(feature_cols)}): {feature_cols}", flush=True)

    rows: list[dict] = []
    train_label = train[LABEL_COL].to_numpy()
    test_label = test[LABEL_COL].to_numpy()
    for col in feature_cols:
        train_vals = train[col].to_numpy(dtype=float)
        test_vals = test[col].to_numpy(dtype=float)
        # Only compute quantiles on positives — that's where signal lives.
        positives_vals = train_vals[train_label.astype(bool)]
        if len(positives_vals) < 20:
            continue
        q_values = np.quantile(positives_vals, QUANTILES)
        # Also add quantiles on non-positives for _max rules (low thresholds
        # catch anti-graduates).
        negatives_vals = train_vals[~train_label.astype(bool)]
        q_neg = (
            np.quantile(negatives_vals, QUANTILES) if len(negatives_vals) >= 20 else np.array([])
        )

        for threshold, origin in [(q, "pos") for q in q_values] + [(q, "neg") for q in q_neg]:
            # _min rule: value >= threshold
            mask_train = train_vals >= threshold
            mask_test = test_vals >= threshold
            tr = _rule_stats(mask_train, train_label, train_base)
            te = _rule_stats(mask_test, test_label, test_base)
            if (
                tr["lift"] >= min_train_lift
                and te["lift"] >= min_holdout_lift
                and te["matches"] >= min_holdout_n
            ):
                rows.append(
                    {
                        "feature": col,
                        "direction": "min",
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
            # _max rule: value <= threshold
            mask_train = train_vals <= threshold
            mask_test = test_vals <= threshold
            tr = _rule_stats(mask_train, train_label, train_base)
            te = _rule_stats(mask_test, test_label, test_base)
            if (
                tr["lift"] >= min_train_lift
                and te["lift"] >= min_holdout_lift
                and te["matches"] >= min_holdout_n
            ):
                rows.append(
                    {
                        "feature": col,
                        "direction": "max",
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
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=INPUT_PARQUET)
    ap.add_argument("--output", type=Path, default=OUTPUT_CSV)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-train-lift", type=float, default=MIN_TRAIN_LIFT)
    ap.add_argument("--min-holdout-lift", type=float, default=MIN_HOLDOUT_LIFT)
    ap.add_argument("--min-holdout-n", type=int, default=MIN_HOLDOUT_N)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    print(f"loaded {len(df):,} snapshots from {args.input}", flush=True)
    rules = mine(
        df,
        window_sec=args.window,
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
    print(rules.to_string(index=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rules.to_csv(args.output, index=False)
    print(f"\nwrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
