"""Mine Pump.fun V2 threshold rules from prepared train features.

This module is offline research only. It emits rule artifacts but does not
change live bot runtime behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import load_app_config
from src.utils.logging_utils import configure_logging


@dataclass(frozen=True)
class RuleCandidate:
    """One evaluated threshold-rule candidate."""

    rule_id: str
    family: str
    support: int
    signal_rate: float
    precision: float
    recall: float
    f1: float
    lift: float
    score: float
    conditions: dict[str, float]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "family": self.family,
            "support": self.support,
            "signal_rate": self.signal_rate,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "lift": self.lift,
            "score": self.score,
            "conditions": self.conditions,
            "conditions_json": json.dumps(self.conditions, sort_keys=True),
            "notes": self.notes,
        }


def _quantile_grid(
    series: pd.Series, quantiles: list[float], floor: float | None = None
) -> list[float]:
    """Build sorted unique threshold grid from quantiles."""
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return []
    values = sorted({float(clean.quantile(q)) for q in quantiles})
    if floor is not None:
        values = [max(float(floor), value) for value in values]
    # Stabilize numeric noise.
    return sorted({round(value, 6) for value in values})


def _int_grid(series: pd.Series, quantiles: list[float], manual: list[int]) -> list[int]:
    """Build integer threshold grid."""
    q_values = [int(round(value)) for value in _quantile_grid(series, quantiles)]
    return sorted({int(value) for value in (manual + q_values) if int(value) > 0})


def _float_grid(
    series: pd.Series,
    quantiles: list[float],
    manual: list[float],
    min_value: float = 0.0,
) -> list[float]:
    """Build float threshold grid."""
    q_values = _quantile_grid(series, quantiles)
    values = [float(value) for value in (manual + q_values) if float(value) >= min_value]
    return sorted({round(value, 6) for value in values})


def _score_rule(precision: float, recall: float, lift: float, support: int) -> float:
    """Balanced ranking score for shortlist ordering."""
    support_term = min(1.0, support / 5000.0)
    # Weight precision/lift more heavily than recall for tradable selectivity.
    return (
        (0.45 * precision)
        + (0.25 * recall)
        + (0.20 * min(lift / 10.0, 1.0))
        + (0.10 * support_term)
    )


def _evaluate_mask(
    mask: pd.Series, labels: pd.Series, base_rate: float, positive_total: int
) -> tuple[int, float, float, float, float, float]:
    """Return support, precision, recall, f1, lift, signal_rate."""
    support = int(mask.sum())
    if support <= 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0
    positives = int(labels[mask].sum())
    precision = positives / support
    recall = (positives / positive_total) if positive_total > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    lift = (precision / base_rate) if base_rate > 0 else 0.0
    signal_rate = support / len(labels)
    return support, precision, recall, f1, lift, signal_rate


def _mine_candidates(
    df: pd.DataFrame, support_min: int
) -> tuple[list[RuleCandidate], dict[str, Any]]:
    """Generate threshold-rule candidates from train feature set."""
    labels = df["has_graduated"].astype(int)
    base_rate = float(labels.mean())
    positive_total = int(labels.sum())
    n_total = int(len(df))

    tx_grid = _int_grid(
        df["tx_count_30s"], [0.75, 0.85, 0.90, 0.95, 0.975], [12, 20, 30, 40, 60, 80]
    )
    buyers_grid = _int_grid(
        df["unique_buyers_30s"],
        [0.75, 0.85, 0.90, 0.95, 0.975],
        [6, 10, 15, 20, 30, 40],
    )
    buy_vol_grid = _float_grid(
        df["buy_volume_sol_30s"],
        [0.75, 0.85, 0.90, 0.95, 0.975],
        [3.0, 5.0, 10.0, 20.0, 40.0, 80.0],
        min_value=0.0,
    )
    growth_grid = _float_grid(
        df["virtual_sol_growth_60s"],
        [0.70, 0.80, 0.90, 0.95],
        [3.0, 5.0, 10.0, 20.0, 40.0],
        min_value=0.0,
    )
    ratio_grid = _float_grid(
        df["buy_sell_ratio_30s"],
        [0.60, 0.70, 0.80, 0.90, 0.95],
        [1.2, 1.5, 2.0, 3.0, 5.0],
        min_value=0.0,
    )
    top_share_grid = sorted({0.50, 0.60, 0.70, 0.80, 0.90})

    tx_masks = {value: df["tx_count_30s"] >= value for value in tx_grid}
    buyers_masks = {value: df["unique_buyers_30s"] >= value for value in buyers_grid}
    buy_vol_masks = {value: df["buy_volume_sol_30s"] >= value for value in buy_vol_grid}
    growth_masks = {value: df["virtual_sol_growth_60s"] >= value for value in growth_grid}
    ratio_masks = {value: df["buy_sell_ratio_30s"] >= value for value in ratio_grid}
    top_share_masks = {value: df["top_wallet_buy_share_30s"] <= value for value in top_share_grid}

    candidates: list[RuleCandidate] = []
    candidate_index = 1

    for tx_min in tx_grid:
        for buyers_min in buyers_grid:
            for buy_vol_min in buy_vol_grid:
                base_mask = tx_masks[tx_min] & buyers_masks[buyers_min] & buy_vol_masks[buy_vol_min]
                support, precision, recall, f1, lift, signal_rate = _evaluate_mask(
                    base_mask, labels, base_rate, positive_total
                )
                if support >= support_min:
                    conditions = {
                        "tx_count_30s_min": float(tx_min),
                        "unique_buyers_30s_min": float(buyers_min),
                        "buy_volume_sol_30s_min": float(buy_vol_min),
                    }
                    score = _score_rule(precision, recall, lift, support)
                    candidates.append(
                        RuleCandidate(
                            rule_id=f"pump_v2_{candidate_index:04d}",
                            family="momentum_core",
                            support=support,
                            signal_rate=signal_rate,
                            precision=precision,
                            recall=recall,
                            f1=f1,
                            lift=lift,
                            score=score,
                            conditions=conditions,
                            notes="Core early momentum gate: tx + unique buyers + buy volume.",
                        )
                    )
                    candidate_index += 1

                for growth_min in growth_grid:
                    mask_growth = base_mask & growth_masks[growth_min]
                    support, precision, recall, f1, lift, signal_rate = _evaluate_mask(
                        mask_growth, labels, base_rate, positive_total
                    )
                    if support < support_min:
                        continue
                    conditions = {
                        "tx_count_30s_min": float(tx_min),
                        "unique_buyers_30s_min": float(buyers_min),
                        "buy_volume_sol_30s_min": float(buy_vol_min),
                        "virtual_sol_growth_60s_min": float(growth_min),
                    }
                    score = _score_rule(precision, recall, lift, support)
                    candidates.append(
                        RuleCandidate(
                            rule_id=f"pump_v2_{candidate_index:04d}",
                            family="momentum_growth",
                            support=support,
                            signal_rate=signal_rate,
                            precision=precision,
                            recall=recall,
                            f1=f1,
                            lift=lift,
                            score=score,
                            conditions=conditions,
                            notes="Momentum with bonding-curve reserve growth confirmation.",
                        )
                    )
                    candidate_index += 1

                for ratio_min in ratio_grid:
                    mask_ratio = base_mask & ratio_masks[ratio_min]
                    support, precision, recall, f1, lift, signal_rate = _evaluate_mask(
                        mask_ratio, labels, base_rate, positive_total
                    )
                    if support < support_min:
                        continue
                    conditions = {
                        "tx_count_30s_min": float(tx_min),
                        "unique_buyers_30s_min": float(buyers_min),
                        "buy_volume_sol_30s_min": float(buy_vol_min),
                        "buy_sell_ratio_30s_min": float(ratio_min),
                    }
                    score = _score_rule(precision, recall, lift, support)
                    candidates.append(
                        RuleCandidate(
                            rule_id=f"pump_v2_{candidate_index:04d}",
                            family="momentum_flow",
                            support=support,
                            signal_rate=signal_rate,
                            precision=precision,
                            recall=recall,
                            f1=f1,
                            lift=lift,
                            score=score,
                            conditions=conditions,
                            notes="Momentum with buy/sell pressure ratio confirmation.",
                        )
                    )
                    candidate_index += 1

                for top_share_max in top_share_grid:
                    mask_quality = base_mask & top_share_masks[top_share_max]
                    support, precision, recall, f1, lift, signal_rate = _evaluate_mask(
                        mask_quality, labels, base_rate, positive_total
                    )
                    if support < support_min:
                        continue
                    conditions = {
                        "tx_count_30s_min": float(tx_min),
                        "unique_buyers_30s_min": float(buyers_min),
                        "buy_volume_sol_30s_min": float(buy_vol_min),
                        "top_wallet_buy_share_30s_max": float(top_share_max),
                    }
                    score = _score_rule(precision, recall, lift, support)
                    candidates.append(
                        RuleCandidate(
                            rule_id=f"pump_v2_{candidate_index:04d}",
                            family="momentum_quality",
                            support=support,
                            signal_rate=signal_rate,
                            precision=precision,
                            recall=recall,
                            f1=f1,
                            lift=lift,
                            score=score,
                            conditions=conditions,
                            notes="Momentum with anti-concentration check (whale share cap).",
                        )
                    )
                    candidate_index += 1

    metadata = {
        "n_tokens": n_total,
        "n_positives": positive_total,
        "base_graduation_rate": base_rate,
        "support_min": support_min,
        "grid": {
            "tx_count_30s_min": tx_grid,
            "unique_buyers_30s_min": buyers_grid,
            "buy_volume_sol_30s_min": buy_vol_grid,
            "virtual_sol_growth_60s_min": growth_grid,
            "buy_sell_ratio_30s_min": ratio_grid,
            "top_wallet_buy_share_30s_max": top_share_grid,
        },
    }
    return candidates, metadata


def mine_pump_rules_v2() -> dict[str, Any]:
    """Mine V2 threshold rules from prepared Pump train features."""
    config = load_app_config()
    logger = configure_logging(logger_name=__name__)

    train_path = config.paths["gold_dir"] / "pump_train_features_v2.parquet"
    if not train_path.exists():
        raise SystemExit(
            f"Missing {train_path}. Run `python -m src.analysis.pump_prepare_dataset_v2` first."
        )

    df = pd.read_parquet(train_path)
    required = [
        "has_graduated",
        "is_valid",
        "tx_count_30s",
        "unique_buyers_30s",
        "buy_volume_sol_30s",
        "buy_sell_ratio_30s",
        "virtual_sol_growth_60s",
        "top_wallet_buy_share_30s",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"Train dataset missing required columns: {missing}")

    work = df[df["is_valid"] == 1].copy()
    work = work.dropna(subset=["has_graduated"])
    work["has_graduated"] = work["has_graduated"].astype(int)
    # Keep only rows with minimally meaningful early activity.
    work = work[(work["tx_count_30s"] > 0) & (work["buy_volume_sol_30s"] > 0)]
    if work.empty:
        raise SystemExit("No valid rows available for rule mining.")

    support_min = int(config.settings.get("analysis", {}).get("support_min", 20))
    support_min = max(50, support_min)

    candidates, metadata = _mine_candidates(work, support_min=support_min)
    if not candidates:
        raise SystemExit("No rule candidates found with current support threshold.")

    rows = [candidate.to_dict() for candidate in candidates]
    candidate_df = pd.DataFrame(rows).sort_values(
        by=["score", "precision", "lift", "support"],
        ascending=[False, False, False, False],
    )

    # Remove condition duplicates, keep best score.
    candidate_df = candidate_df.drop_duplicates(subset=["family", "conditions_json"], keep="first")

    shortlist = candidate_df[
        (candidate_df["support"] >= 200)
        & (candidate_df["precision"] >= (metadata["base_graduation_rate"] * 3.5))
        & (candidate_df["lift"] >= 3.5)
    ].copy()
    if shortlist.empty:
        shortlist = candidate_df.head(30).copy()
    else:
        shortlist = shortlist.head(60).copy()

    candidates_json_path = config.paths["rules_dir"] / "pump_rule_candidates_v2.json"
    candidates_csv_path = config.paths["rules_dir"] / "pump_rule_candidates_v2.csv"
    shortlist_json_path = config.paths["rules_dir"] / "pump_rule_shortlist_v2.json"
    shortlist_csv_path = config.paths["rules_dir"] / "pump_rule_shortlist_v2.csv"
    report_path = config.paths["reports_dir"] / "pump_rule_mining_summary_v2.json"

    candidates_json_path.write_text(
        json.dumps(candidate_df.to_dict(orient="records"), indent=2, default=str),
        encoding="utf-8",
    )
    candidate_df.to_csv(candidates_csv_path, index=False)
    shortlist_json_path.write_text(
        json.dumps(shortlist.to_dict(orient="records"), indent=2, default=str),
        encoding="utf-8",
    )
    shortlist.to_csv(shortlist_csv_path, index=False)

    report = {
        "metadata": metadata,
        "candidate_count": int(len(candidate_df)),
        "shortlist_count": int(len(shortlist)),
        "top_10": shortlist.head(10).to_dict(orient="records"),
        "paths": {
            "candidates_csv": str(candidates_csv_path),
            "candidates_json": str(candidates_json_path),
            "shortlist_csv": str(shortlist_csv_path),
            "shortlist_json": str(shortlist_json_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    logger.info(
        "Saved Pump V2 rule artifacts: candidates=%s shortlist=%s",
        len(candidate_df),
        len(shortlist),
    )
    logger.info("Rule summary report: %s", report_path)
    return report


def main() -> None:
    """CLI entrypoint."""
    mine_pump_rules_v2()


if __name__ == "__main__":
    main()
