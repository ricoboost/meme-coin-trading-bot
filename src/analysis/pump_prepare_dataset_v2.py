"""Build Pump.fun V2 research datasets from external CSV chunks.

This module is offline research only. It does not modify bot runtime logic.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.utils.io import load_app_config
from src.utils.logging_utils import configure_logging


@dataclass(frozen=True)
class PumpDatasetPaths:
    """Resolved paths for Pump.fun external dataset assets."""

    root: Path
    chunks_glob: str
    train_csv: Path
    test_csv: Path


def _resolve_dataset_paths(root_dir: Path) -> PumpDatasetPaths:
    dataset_root = root_dir / "data" / "external" / "pump_dataset"
    return PumpDatasetPaths(
        root=dataset_root,
        chunks_glob=str(dataset_root / "chunk_*.csv"),
        train_csv=dataset_root / "train.csv",
        test_csv=dataset_root / "test_unlabeled.csv",
    )


def _escape(path: Path | str) -> str:
    """Escape single-quotes for SQL string literals."""
    return str(path).replace("'", "''")


def _base_features_sql(chunks_glob: str) -> str:
    """Return the SQL statement that builds per-token early features."""
    chunks = _escape(chunks_glob)
    return f"""
    WITH tx_raw AS (
      SELECT
        base_coin AS mint,
        CAST(block_time AS TIMESTAMP) AS block_time,
        CAST(slot AS BIGINT) AS slot,
        CAST(tx_idx AS BIGINT) AS tx_idx,
        CAST(signing_wallet AS VARCHAR) AS signing_wallet,
        LOWER(CAST(direction AS VARCHAR)) AS direction,
        CAST(quote_coin_amount AS DOUBLE) / 1e9 AS sol_amount,
        CAST(base_coin_amount AS DOUBLE) AS token_amount_raw,
        CAST(virtual_sol_balance_after AS DOUBLE) / 1e9 AS virtual_sol_after,
        CAST(virtual_token_balance_after AS DOUBLE) AS virtual_token_after,
        CAST(signature AS VARCHAR) AS signature
      FROM read_csv_auto('{chunks}', HEADER=TRUE)
      WHERE base_coin IS NOT NULL
    ),
    tx AS (
      SELECT *
      FROM tx_raw
      WHERE
        direction IN ('buy', 'sell')
        AND sol_amount > 0
        AND token_amount_raw > 0
    ),
    first_seen AS (
      SELECT
        mint,
        MIN(block_time) AS first_seen_at,
        MIN(slot) AS first_slot
      FROM tx
      GROUP BY 1
    ),
    tx_age AS (
      SELECT
        tx.*,
        fs.first_seen_at,
        fs.first_slot,
        DATE_DIFF('second', fs.first_seen_at, tx.block_time) AS age_sec
      FROM tx
      JOIN first_seen fs USING (mint)
    ),
    token_features AS (
      SELECT
        mint,
        MIN(first_seen_at) AS first_seen_at,
        MIN(first_slot) AS first_slot,
        COUNT(*) AS tx_count_total,
        COUNT(*) FILTER (WHERE direction = 'buy') AS buy_tx_count_total,
        COUNT(*) FILTER (WHERE direction = 'sell') AS sell_tx_count_total,
        COUNT(DISTINCT signing_wallet) AS unique_wallets_total,
        SUM(sol_amount) AS volume_sol_total,
        SUM(CASE WHEN direction = 'buy' THEN sol_amount ELSE 0 END) AS buy_volume_sol_total,
        SUM(CASE WHEN direction = 'sell' THEN sol_amount ELSE 0 END) AS sell_volume_sol_total,
        COUNT(*) FILTER (WHERE age_sec <= 15) AS tx_count_15s,
        COUNT(*) FILTER (WHERE age_sec <= 30) AS tx_count_30s,
        COUNT(*) FILTER (WHERE age_sec <= 60) AS tx_count_60s,
        COUNT(*) FILTER (WHERE age_sec <= 120) AS tx_count_120s,
        COUNT(*) FILTER (WHERE age_sec <= 300) AS tx_count_300s,
        COUNT(*) FILTER (WHERE age_sec <= 30 AND direction = 'buy') AS buy_tx_count_30s,
        COUNT(*) FILTER (WHERE age_sec <= 60 AND direction = 'buy') AS buy_tx_count_60s,
        COUNT(*) FILTER (WHERE age_sec <= 30 AND direction = 'sell') AS sell_tx_count_30s,
        COUNT(*) FILTER (WHERE age_sec <= 60 AND direction = 'sell') AS sell_tx_count_60s,
        COUNT(DISTINCT signing_wallet) FILTER (WHERE age_sec <= 30 AND direction = 'buy') AS unique_buyers_30s,
        COUNT(DISTINCT signing_wallet) FILTER (WHERE age_sec <= 60 AND direction = 'buy') AS unique_buyers_60s,
        SUM(sol_amount) FILTER (WHERE age_sec <= 15) AS volume_sol_15s,
        SUM(sol_amount) FILTER (WHERE age_sec <= 30) AS volume_sol_30s,
        SUM(sol_amount) FILTER (WHERE age_sec <= 60) AS volume_sol_60s,
        SUM(sol_amount) FILTER (WHERE age_sec <= 120) AS volume_sol_120s,
        SUM(CASE WHEN direction = 'buy' THEN sol_amount ELSE 0 END) FILTER (WHERE age_sec <= 30) AS buy_volume_sol_30s,
        SUM(CASE WHEN direction = 'buy' THEN sol_amount ELSE 0 END) FILTER (WHERE age_sec <= 60) AS buy_volume_sol_60s,
        SUM(CASE WHEN direction = 'sell' THEN sol_amount ELSE 0 END) FILTER (WHERE age_sec <= 30) AS sell_volume_sol_30s,
        SUM(CASE WHEN direction = 'sell' THEN sol_amount ELSE 0 END) FILTER (WHERE age_sec <= 60) AS sell_volume_sol_60s,
        MAX(virtual_sol_after) FILTER (WHERE age_sec <= 30) - MIN(virtual_sol_after) FILTER (WHERE age_sec <= 30) AS virtual_sol_growth_30s,
        MAX(virtual_sol_after) FILTER (WHERE age_sec <= 60) - MIN(virtual_sol_after) FILTER (WHERE age_sec <= 60) AS virtual_sol_growth_60s,
        MAX(virtual_sol_after) FILTER (WHERE age_sec <= 120) - MIN(virtual_sol_after) FILTER (WHERE age_sec <= 120) AS virtual_sol_growth_120s,
        MAX(virtual_token_after) FILTER (WHERE age_sec <= 30) - MIN(virtual_token_after) FILTER (WHERE age_sec <= 30) AS virtual_token_change_30s,
        MAX(virtual_token_after) FILTER (WHERE age_sec <= 60) - MIN(virtual_token_after) FILTER (WHERE age_sec <= 60) AS virtual_token_change_60s,
        MAX(CASE WHEN age_sec <= 30 AND direction = 'buy' THEN sol_amount ELSE 0 END) AS max_buy_trade_sol_30s,
        MAX(CASE WHEN age_sec <= 60 AND direction = 'buy' THEN sol_amount ELSE 0 END) AS max_buy_trade_sol_60s,
        MAX(age_sec) AS max_age_sec_observed
      FROM tx_age
      GROUP BY 1
    )
    SELECT
      tf.*,
      CASE
        WHEN COALESCE(tf.sell_volume_sol_30s, 0) > 0 THEN tf.buy_volume_sol_30s / tf.sell_volume_sol_30s
        ELSE NULL
      END AS buy_sell_ratio_30s,
      CASE
        WHEN COALESCE(tf.sell_volume_sol_60s, 0) > 0 THEN tf.buy_volume_sol_60s / tf.sell_volume_sol_60s
        ELSE NULL
      END AS buy_sell_ratio_60s,
      COALESCE(tf.buy_volume_sol_30s, 0) - COALESCE(tf.sell_volume_sol_30s, 0) AS net_flow_sol_30s,
      COALESCE(tf.buy_volume_sol_60s, 0) - COALESCE(tf.sell_volume_sol_60s, 0) AS net_flow_sol_60s,
      CASE WHEN COALESCE(tf.tx_count_30s, 0) > 0 THEN COALESCE(tf.volume_sol_30s, 0) / tf.tx_count_30s ELSE 0 END AS avg_trade_sol_30s,
      CASE WHEN COALESCE(tf.tx_count_60s, 0) > 0 THEN COALESCE(tf.volume_sol_60s, 0) / tf.tx_count_60s ELSE 0 END AS avg_trade_sol_60s,
      CASE
        WHEN COALESCE(tf.buy_volume_sol_30s, 0) > 0 THEN COALESCE(tf.max_buy_trade_sol_30s, 0) / tf.buy_volume_sol_30s
        ELSE NULL
      END AS top_wallet_buy_share_30s,
      CASE
        WHEN COALESCE(tf.buy_volume_sol_60s, 0) > 0 THEN COALESCE(tf.max_buy_trade_sol_60s, 0) / tf.buy_volume_sol_60s
        ELSE NULL
      END AS top_wallet_buy_share_60s
    FROM token_features tf
    """


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _profile_summary(con: duckdb.DuckDBPyConnection, paths: PumpDatasetPaths) -> dict[str, Any]:
    """Collect dataset profile metrics for reporting."""
    chunks = _escape(paths.chunks_glob)
    train = _escape(paths.train_csv)
    test = _escape(paths.test_csv)
    summary_sql = f"""
    WITH c AS (
      SELECT DISTINCT base_coin AS mint
      FROM read_csv_auto('{chunks}', HEADER=TRUE)
    ),
    tr AS (
      SELECT mint, CAST(has_graduated AS DOUBLE) AS has_graduated, CAST(is_valid AS DOUBLE) AS is_valid
      FROM read_csv_auto('{train}', HEADER=TRUE)
    ),
    te AS (
      SELECT mint
      FROM read_csv_auto('{test}', HEADER=TRUE)
    )
    SELECT
      (SELECT COUNT(*) FROM read_csv_auto('{chunks}', HEADER=TRUE)) AS chunk_rows,
      (SELECT COUNT(*) FROM c) AS chunk_unique_mints,
      (SELECT COUNT(*) FROM tr) AS train_rows,
      (SELECT COUNT(*) FROM te) AS test_rows,
      (SELECT AVG(has_graduated) FROM tr) AS train_graduation_rate,
      (SELECT AVG(is_valid) FROM tr) AS train_valid_rate,
      (SELECT COUNT(*) FROM tr LEFT JOIN c USING (mint) WHERE c.mint IS NULL) AS train_missing_in_chunks,
      (SELECT COUNT(*) FROM te LEFT JOIN c USING (mint) WHERE c.mint IS NULL) AS test_missing_in_chunks
    """
    row = con.execute(summary_sql).fetchone()
    return {
        "chunk_rows": int(row[0]),
        "chunk_unique_mints": int(row[1]),
        "train_rows": int(row[2]),
        "test_rows": int(row[3]),
        "train_graduation_rate": float(row[4]) if row[4] is not None else None,
        "train_valid_rate": float(row[5]) if row[5] is not None else None,
        "train_missing_in_chunks": int(row[6]),
        "test_missing_in_chunks": int(row[7]),
    }


def build_pump_research_datasets() -> dict[str, Any]:
    """Build silver/gold Pump.fun research datasets from external CSV dumps."""
    config = load_app_config()
    logger = configure_logging(logger_name=__name__)
    paths = _resolve_dataset_paths(config.root_dir)

    if not paths.root.exists():
        raise SystemExit(f"Pump dataset folder not found: {paths.root}")
    if not paths.train_csv.exists():
        raise SystemExit(f"Missing train.csv: {paths.train_csv}")
    if not paths.test_csv.exists():
        raise SystemExit(f"Missing test_unlabeled.csv: {paths.test_csv}")

    silver_path = config.paths["silver_dir"] / "pump_token_features_v2.parquet"
    train_path = config.paths["gold_dir"] / "pump_train_features_v2.parquet"
    test_path = config.paths["gold_dir"] / "pump_test_features_v2.parquet"
    profile_path = config.paths["reports_dir"] / "pump_dataset_profile_v2.json"
    summary_csv_path = config.paths["reports_dir"] / "pump_feature_summary_v2.csv"

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(os.getenv('PUMP_PREP_DUCKDB_THREADS', '2'))}")
    con.execute(f"PRAGMA memory_limit='{os.getenv('PUMP_PREP_DUCKDB_MEMORY_LIMIT', '10GB')}'")
    con.execute("SET preserve_insertion_order=false")
    temp_dir = os.getenv(
        "PUMP_PREP_DUCKDB_TEMP_DIR",
        str(config.root_dir / "data" / "live" / "duckdb_tmp"),
    )
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{_escape(temp_dir)}'")

    features_sql = _base_features_sql(paths.chunks_glob)
    logger.info("Building silver dataset: %s", silver_path)
    con.execute(
        f"COPY ({features_sql}) TO '{_escape(silver_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    logger.info("Building gold train dataset: %s", train_path)
    con.execute(
        f"""
        COPY (
          SELECT
            tr.*,
            f.* EXCLUDE (mint, first_slot),
            CASE WHEN tr.slot_min IS NOT NULL AND f.first_slot IS NOT NULL THEN tr.slot_min - f.first_slot ELSE NULL END AS slot_offset_from_first_seen
          FROM read_csv_auto('{_escape(paths.train_csv)}', HEADER=TRUE) tr
          LEFT JOIN read_parquet('{_escape(silver_path)}') f
            ON tr.mint = f.mint
        ) TO '{_escape(train_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    logger.info("Building gold test dataset: %s", test_path)
    con.execute(
        f"""
        COPY (
          SELECT
            te.*,
            f.* EXCLUDE (mint, first_slot),
            CASE WHEN te.slot_min IS NOT NULL AND f.first_slot IS NOT NULL THEN te.slot_min - f.first_slot ELSE NULL END AS slot_offset_from_first_seen
          FROM read_csv_auto('{_escape(paths.test_csv)}', HEADER=TRUE) te
          LEFT JOIN read_parquet('{_escape(silver_path)}') f
            ON te.mint = f.mint
        ) TO '{_escape(test_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    profile = _profile_summary(con, paths)
    profile["dataset_root"] = str(paths.root)
    profile["silver_features_path"] = str(silver_path)
    profile["gold_train_path"] = str(train_path)
    profile["gold_test_path"] = str(test_path)
    _write_json(profile_path, profile)

    train_df = pd.read_parquet(train_path)
    summary = (
        train_df.groupby("has_graduated", dropna=False)[
            [
                "tx_count_30s",
                "unique_buyers_30s",
                "buy_volume_sol_30s",
                "sell_volume_sol_30s",
                "buy_sell_ratio_30s",
                "virtual_sol_growth_60s",
                "top_wallet_buy_share_30s",
            ]
        ]
        .median(numeric_only=True)
        .reset_index()
    )
    summary.to_csv(summary_csv_path, index=False)

    logger.info(
        "Built Pump v2 datasets: silver=%s train=%s test=%s",
        silver_path,
        train_path,
        test_path,
    )
    logger.info("Wrote profile report: %s", profile_path)
    return profile


def main() -> None:
    """CLI entrypoint."""
    build_pump_research_datasets()


if __name__ == "__main__":
    main()
