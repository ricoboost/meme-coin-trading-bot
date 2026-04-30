"""Collect wallet transaction history from Helius."""

from __future__ import annotations

import argparse

from tqdm import tqdm

from src.clients.helius_client import HeliusClient
from src.utils.io import append_jsonl, dataset_path, load_app_config, read_parquet
from src.utils.logging_utils import configure_logging
from src.utils.time_utils import lookback_cutoff


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Collect wallet history from Helius.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of wallets processed."
    )
    parser.add_argument("--force", action="store_true", help="Re-collect even if raw JSONL exists.")
    args = parser.parse_args()

    config = load_app_config()
    logger = configure_logging(logger_name=__name__)
    wallet_pool = read_parquet(dataset_path(config, "bronze", "wallet_pool.parquet"))
    if wallet_pool.empty:
        raise SystemExit(
            "wallet_pool.parquet not found or empty. Run collect_provider_a_wallets first."
        )

    cutoff = lookback_cutoff(config.env["LOOKBACK_DAYS"])
    helius_cfg = config.settings["helius"]
    rows = wallet_pool.sort_values(["score", "best_rank"], ascending=[False, True])
    if args.limit:
        rows = rows.head(args.limit)

    with HeliusClient(
        api_key=config.env["HELIUS_API_KEY"],
        base_url=config.env["HELIUS_BASE_URL"] or helius_cfg["base_url"],
        timeout_sec=helius_cfg["request_timeout_sec"],
        page_size=helius_cfg["page_size"],
        max_pages=helius_cfg["max_pages_per_wallet"],
    ) as client:
        for wallet in tqdm(rows["wallet"].tolist(), desc="wallets"):
            target = config.paths["raw_wallet_dir"] / f"{wallet}.jsonl"
            if target.exists() and not args.force:
                logger.info("Skipping %s because raw file already exists", wallet)
                continue
            transactions = client.fetch_wallet_transactions(wallet, cutoff)
            if target.exists():
                target.unlink()
            append_jsonl(target, transactions)
            logger.info("Saved %s transactions for %s", len(transactions), wallet)


if __name__ == "__main__":
    main()
