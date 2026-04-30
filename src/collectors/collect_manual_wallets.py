"""Manual wallet list importer.

Reads a user-maintained CSV (default: data/bronze/manual_wallets.csv) and
writes it as a parquet so the orchestrator can union it with the other
wallet sources. Escape hatch for pasting curated wallet lists you've
collected by hand, without needing brittle scrapers.

CSV format (header required):
    wallet,source,note
    <base58-pubkey>,manual_top10,first batch
    <base58-pubkey>,twitter_alpha,curated 2026-01
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.utils.io import dataset_path, load_app_config, write_parquet
from src.utils.logging_utils import configure_logging
from src.utils.time_utils import utcnow

BASE58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import manually-curated wallet CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to CSV with columns: wallet,source[,note]. Default: data/bronze/manual_wallets.csv",
    )
    args = parser.parse_args()

    config = load_app_config()
    logger = configure_logging(logger_name=__name__)
    root = config.root_dir

    csv_path = args.input or (root / "data" / "bronze" / "manual_wallets.csv")
    if not csv_path.exists():
        logger.warning("No manual CSV at %s — writing empty parquet.", csv_path)
        df = pd.DataFrame(columns=["wallet", "source", "note", "collected_at"])
    else:
        df = pd.read_csv(csv_path)
        if "wallet" not in df.columns:
            raise SystemExit(f"{csv_path} must have a 'wallet' column")
        df["wallet"] = df["wallet"].astype(str).str.strip()
        df = df[df["wallet"].str.match(BASE58_RE)].copy()
        if "source" not in df.columns:
            df["source"] = "manual"
        if "note" not in df.columns:
            df["note"] = ""
        df["collected_at"] = utcnow()
        df = df[["wallet", "source", "note", "collected_at"]].drop_duplicates("wallet")
        logger.info("Imported %d manual wallets from %s", len(df), csv_path)

    out_path = dataset_path(config, "bronze", "manual_wallets.parquet")
    write_parquet(df, out_path)
    logger.info("Wrote %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()
