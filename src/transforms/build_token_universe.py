"""Build the token universe from normalized wallet trades."""

from __future__ import annotations


from src.utils.io import dataset_path, load_app_config, read_parquet, write_parquet
from src.utils.logging_utils import configure_logging


def main() -> None:
    """CLI entry point."""
    config = load_app_config()
    logger = configure_logging(logger_name=__name__)
    trades = read_parquet(dataset_path(config, "silver", "wallet_trades.parquet"))
    if trades.empty:
        raise SystemExit(
            "wallet_trades.parquet not found or empty. Run normalize_wallet_trades first."
        )

    work = trades.dropna(subset=["token_mint"]).copy()
    universe = (
        work.sort_values("block_time")
        .groupby("token_mint", as_index=False)
        .agg(
            first_seen_at=("block_time", "min"),
            first_seen_wallet=("wallet", "first"),
            trade_count=("signature", "count"),
            unique_wallets=("wallet", "nunique"),
        )
        .sort_values(["trade_count", "unique_wallets"], ascending=[False, False])
    )

    write_parquet(universe, dataset_path(config, "silver", "token_universe.parquet"))
    logger.info("Saved %s tokens to universe", len(universe))


if __name__ == "__main__":
    main()
