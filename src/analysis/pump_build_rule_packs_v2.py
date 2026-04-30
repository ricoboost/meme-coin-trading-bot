"""Build runtime-ready Pump V2 rule packs from mined candidates.

Outputs three disjoint packs:
- high_conviction
- balanced
- high_frequency

This module is offline-only and does not modify runtime bot code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import load_app_config
from src.utils.logging_utils import configure_logging


@dataclass(frozen=True)
class PackConfig:
    """Selection policy for one pack tier."""

    name: str
    min_support_valid: int
    min_lift_valid: float
    min_precision_mult: float
    max_buy_volume_30s_min: float | None
    max_tx_count_30s_min: float | None
    max_unique_buyers_30s_min: float | None
    max_rules: int


def _parse_conditions(value: object) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    if not isinstance(value, str) or not value.strip():
        return {}
    parsed = json.loads(value)
    return {str(k): float(v) for k, v in parsed.items()}


def _mask_from_conditions(df: pd.DataFrame, conditions: dict[str, float]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, threshold in conditions.items():
        if key.endswith("_min"):
            col = key[:-4]
            if col not in df.columns:
                return pd.Series(False, index=df.index)
            mask &= df[col].fillna(0) >= float(threshold)
        elif key.endswith("_max"):
            col = key[:-4]
            if col not in df.columns:
                return pd.Series(False, index=df.index)
            mask &= df[col].fillna(0) <= float(threshold)
    return mask


def _eval_rule(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    conditions: dict[str, float],
    base_rate_train: float,
    base_rate_valid: float,
) -> dict[str, float]:
    labels_train = train_df["has_graduated"].astype(int)
    labels_valid = valid_df["has_graduated"].astype(int)
    pos_train = int(labels_train.sum())
    pos_valid = int(labels_valid.sum())

    m_train = _mask_from_conditions(train_df, conditions)
    m_valid = _mask_from_conditions(valid_df, conditions)

    support_train = int(m_train.sum())
    support_valid = int(m_valid.sum())

    tp_train = int(labels_train[m_train].sum()) if support_train > 0 else 0
    tp_valid = int(labels_valid[m_valid].sum()) if support_valid > 0 else 0

    precision_train = (tp_train / support_train) if support_train > 0 else 0.0
    precision_valid = (tp_valid / support_valid) if support_valid > 0 else 0.0
    recall_train = (tp_train / pos_train) if pos_train > 0 else 0.0
    recall_valid = (tp_valid / pos_valid) if pos_valid > 0 else 0.0
    f1_valid = (
        2 * precision_valid * recall_valid / (precision_valid + recall_valid)
        if (precision_valid + recall_valid) > 0
        else 0.0
    )
    lift_train = (precision_train / base_rate_train) if base_rate_train > 0 else 0.0
    lift_valid = (precision_valid / base_rate_valid) if base_rate_valid > 0 else 0.0
    signal_rate_valid = support_valid / len(valid_df) if len(valid_df) > 0 else 0.0
    signal_rate_train = support_train / len(train_df) if len(train_df) > 0 else 0.0
    delta_precision = precision_valid - precision_train

    score_valid = (
        0.40 * precision_valid
        + 0.20 * recall_valid
        + 0.25 * min(lift_valid / 10.0, 1.0)
        + 0.15 * min(support_valid / 4000.0, 1.0)
    )

    return {
        "support_train": float(support_train),
        "support_valid": float(support_valid),
        "precision_train": float(precision_train),
        "precision_valid": float(precision_valid),
        "recall_train": float(recall_train),
        "recall_valid": float(recall_valid),
        "f1_valid": float(f1_valid),
        "lift_train": float(lift_train),
        "lift_valid": float(lift_valid),
        "signal_rate_train": float(signal_rate_train),
        "signal_rate_valid": float(signal_rate_valid),
        "delta_precision": float(delta_precision),
        "score_valid": float(score_valid),
    }


def _condition_value(conditions: dict[str, float], key: str) -> float | None:
    value = conditions.get(key)
    return float(value) if value is not None else None


def _signature(conditions: dict[str, float]) -> str:
    tx_min = _condition_value(conditions, "tx_count_30s_min")
    buyers_min = _condition_value(conditions, "unique_buyers_30s_min")
    buy_vol_min = _condition_value(conditions, "buy_volume_sol_30s_min")
    ratio_min = _condition_value(conditions, "buy_sell_ratio_30s_min")
    growth_min = _condition_value(conditions, "virtual_sol_growth_60s_min")
    share_max = _condition_value(conditions, "top_wallet_buy_share_30s_max")
    return "|".join(
        [
            f"tx:{int(tx_min) if tx_min is not None else 0}",
            f"buyers:{int(buyers_min) if buyers_min is not None else 0}",
            f"buyvol:{round(buy_vol_min or 0.0, 1)}",
            f"ratio:{round(ratio_min or 0.0, 2)}",
            f"growth:{round(growth_min or 0.0, 1)}",
            f"share:{round(share_max or 0.0, 2)}",
        ]
    )


def _pack_filter(row: pd.Series, pack: PackConfig, base_rate_valid: float) -> bool:
    if row["support_valid"] < pack.min_support_valid:
        return False
    if row["lift_valid"] < pack.min_lift_valid:
        return False
    if row["precision_valid"] < (base_rate_valid * pack.min_precision_mult):
        return False
    if row["delta_precision"] < -0.20:
        return False
    conditions = row["conditions_obj"]
    tx_min = _condition_value(conditions, "tx_count_30s_min")
    buyers_min = _condition_value(conditions, "unique_buyers_30s_min")
    buy_vol_min = _condition_value(conditions, "buy_volume_sol_30s_min")
    if (
        pack.max_tx_count_30s_min is not None
        and tx_min is not None
        and tx_min > pack.max_tx_count_30s_min
    ):
        return False
    if (
        pack.max_unique_buyers_30s_min is not None
        and buyers_min is not None
        and buyers_min > pack.max_unique_buyers_30s_min
    ):
        return False
    if (
        pack.max_buy_volume_30s_min is not None
        and buy_vol_min is not None
        and buy_vol_min > pack.max_buy_volume_30s_min
    ):
        return False
    return True


def _select_pack(
    evaluated: pd.DataFrame,
    pack: PackConfig,
    used_rule_ids: set[str],
    base_rate_valid: float,
) -> pd.DataFrame:
    pool = evaluated[~evaluated["rule_id"].isin(used_rule_ids)].copy()
    pool = pool[pool.apply(lambda row: _pack_filter(row, pack, base_rate_valid), axis=1)]
    if pool.empty:
        return pool

    pool = pool.sort_values(
        by=["score_valid", "precision_valid", "lift_valid", "support_valid"],
        ascending=[False, False, False, False],
    )
    pool = pool.drop_duplicates(subset=["condition_signature"], keep="first")
    selected = pool.head(pack.max_rules).copy()
    selected["pack_name"] = pack.name
    selected["pack_rank"] = range(1, len(selected) + 1)
    return selected


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def build_rule_packs_v2() -> dict[str, Any]:
    """Create runtime-ready rule packs from mined candidates."""
    config = load_app_config()
    logger = configure_logging(logger_name=__name__)

    candidate_path = config.paths["rules_dir"] / "pump_rule_candidates_v2.csv"
    train_path = config.paths["gold_dir"] / "pump_train_features_v2.parquet"
    if not candidate_path.exists():
        raise SystemExit(
            f"Missing {candidate_path}. Run `python -m src.analysis.pump_mine_rules_v2` first."
        )
    if not train_path.exists():
        raise SystemExit(
            f"Missing {train_path}. Run `python -m src.analysis.pump_prepare_dataset_v2` first."
        )

    candidates = pd.read_csv(candidate_path)
    train_df = pd.read_parquet(train_path)
    work = train_df[(train_df["is_valid"] == 1)].copy()
    work["has_graduated"] = work["has_graduated"].astype(int)

    cut_slot = work["slot_min"].quantile(0.80)
    split_train = work[work["slot_min"] <= cut_slot].copy()
    split_valid = work[work["slot_min"] > cut_slot].copy()
    if split_train.empty or split_valid.empty:
        raise SystemExit("Temporal split produced empty train or validation set.")

    base_rate_train = float(split_train["has_graduated"].mean())
    base_rate_valid = float(split_valid["has_graduated"].mean())

    # Prefilter candidate pool to keep evaluation tractable and robust.
    pool = candidates.copy()
    pool = pool[(pool["support"] >= 200) & (pool["lift"] >= 2.5)]
    pool = pool.sort_values(
        by=["score", "precision", "support"], ascending=[False, False, False]
    ).head(4000)
    pool = pool.copy()
    pool["conditions_obj"] = pool["conditions_json"].apply(_parse_conditions)
    pool["condition_signature"] = pool["conditions_obj"].apply(_signature)
    pool = pool.drop_duplicates(subset=["condition_signature"], keep="first")

    evaluated_rows: list[dict[str, Any]] = []
    for row in pool.to_dict(orient="records"):
        cond = row["conditions_obj"]
        metrics = _eval_rule(split_train, split_valid, cond, base_rate_train, base_rate_valid)
        evaluated_rows.append(
            {
                **row,
                **metrics,
            }
        )
    evaluated = pd.DataFrame(evaluated_rows)

    packs = [
        PackConfig(
            name="high_conviction",
            min_support_valid=80,
            min_lift_valid=8.0,
            min_precision_mult=8.0,
            max_buy_volume_30s_min=None,
            max_tx_count_30s_min=None,
            max_unique_buyers_30s_min=None,
            max_rules=20,
        ),
        PackConfig(
            name="balanced",
            min_support_valid=180,
            min_lift_valid=4.5,
            min_precision_mult=4.5,
            max_buy_volume_30s_min=80.0,
            max_tx_count_30s_min=80.0,
            max_unique_buyers_30s_min=80.0,
            max_rules=20,
        ),
        PackConfig(
            name="high_frequency",
            min_support_valid=350,
            min_lift_valid=3.0,
            min_precision_mult=3.0,
            max_buy_volume_30s_min=40.0,
            max_tx_count_30s_min=40.0,
            max_unique_buyers_30s_min=40.0,
            max_rules=20,
        ),
    ]

    selected_packs: dict[str, pd.DataFrame] = {}
    used_rule_ids: set[str] = set()
    for pack in packs:
        selected = _select_pack(evaluated, pack, used_rule_ids, base_rate_valid)
        selected_packs[pack.name] = selected
        used_rule_ids.update(selected["rule_id"].tolist())

    combined = (
        pd.concat(selected_packs.values(), ignore_index=True) if selected_packs else pd.DataFrame()
    )
    if combined.empty:
        raise SystemExit("No rules qualified for any pack. Relax pack thresholds and rerun.")

    # Export per-pack and combined artifacts.
    for pack_name, df_pack in selected_packs.items():
        out_csv = config.paths["rules_dir"] / f"pump_rule_pack_{pack_name}_v2.csv"
        out_json = config.paths["rules_dir"] / f"pump_rule_pack_{pack_name}_v2.json"
        df_pack.to_csv(out_csv, index=False)
        out_json.write_text(
            json.dumps(df_pack.to_dict(orient="records"), indent=2, default=str),
            encoding="utf-8",
        )

    combined_csv = config.paths["rules_dir"] / "pump_rule_packs_v2.csv"
    combined_json = config.paths["rules_dir"] / "pump_rule_packs_v2.json"
    combined.to_csv(combined_csv, index=False)
    combined_json.write_text(
        json.dumps(combined.to_dict(orient="records"), indent=2, default=str),
        encoding="utf-8",
    )

    summary = {
        "temporal_split": {
            "train_rows": int(len(split_train)),
            "valid_rows": int(len(split_valid)),
            "cut_slot_min": int(cut_slot),
            "base_rate_train": base_rate_train,
            "base_rate_valid": base_rate_valid,
        },
        "candidate_pool": {
            "after_prefilter": int(len(pool)),
            "evaluated": int(len(evaluated)),
        },
        "pack_counts": {name: int(len(df_pack)) for name, df_pack in selected_packs.items()},
        "combined_count": int(len(combined)),
        "top_rules_by_pack": {
            name: df_pack.head(5)[
                [
                    "rule_id",
                    "family",
                    "support_valid",
                    "precision_valid",
                    "lift_valid",
                    "score_valid",
                    "conditions_json",
                ]
            ].to_dict(orient="records")
            for name, df_pack in selected_packs.items()
        },
        "paths": {
            "combined_csv": str(combined_csv),
            "combined_json": str(combined_json),
            **{
                f"{name}_csv": str(config.paths["rules_dir"] / f"pump_rule_pack_{name}_v2.csv")
                for name in selected_packs
            },
        },
    }
    summary_path = config.paths["reports_dir"] / "pump_rule_packs_summary_v2.json"
    _write_json(summary_path, summary)

    logger.info(
        "Built Pump V2 rule packs: high_conviction=%s balanced=%s high_frequency=%s combined=%s",
        len(selected_packs["high_conviction"]),
        len(selected_packs["balanced"]),
        len(selected_packs["high_frequency"]),
        len(combined),
    )
    logger.info("Pack summary report: %s", summary_path)
    return summary


def main() -> None:
    """CLI entrypoint."""
    build_rule_packs_v2()


if __name__ == "__main__":
    main()
