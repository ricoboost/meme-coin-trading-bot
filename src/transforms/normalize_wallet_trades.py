"""Normalize raw Helius wallet transactions into BUY and SELL trade events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import dataset_path, load_app_config, read_jsonl, write_parquet
from src.utils.logging_utils import configure_logging


SOL_MINT = "So11111111111111111111111111111111111111112"


def _sum_token_amount(
    transfers: list[dict[str, Any]], wallet: str, direction: str
) -> tuple[str | None, float]:
    amount_by_mint: dict[str, float] = {}
    for transfer in transfers or []:
        mint = transfer.get("mint")
        if not mint or mint == SOL_MINT:
            continue
        from_user = transfer.get("fromUserAccount")
        to_user = transfer.get("toUserAccount")
        raw_amount = transfer.get("tokenAmount")
        amount = pd.to_numeric(raw_amount, errors="coerce")
        if pd.isna(amount):
            continue
        if direction == "in" and to_user == wallet:
            amount_by_mint[mint] = amount_by_mint.get(mint, 0.0) + float(amount)
        elif direction == "out" and from_user == wallet:
            amount_by_mint[mint] = amount_by_mint.get(mint, 0.0) + float(amount)
    if not amount_by_mint:
        return None, 0.0
    mint, amount = max(amount_by_mint.items(), key=lambda item: item[1])
    return mint, amount


def _sum_sol_amount(tx: dict[str, Any], wallet: str, direction: str) -> float:
    total = 0.0
    for transfer in tx.get("nativeTransfers", []) or []:
        amount = pd.to_numeric(transfer.get("amount"), errors="coerce")
        if pd.isna(amount):
            continue
        from_user = transfer.get("fromUserAccount")
        to_user = transfer.get("toUserAccount")
        if direction == "out" and from_user == wallet:
            total += float(amount) / 1_000_000_000
        elif direction == "in" and to_user == wallet:
            total += float(amount) / 1_000_000_000
    return total


def extract_trade(wallet: str, tx: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract BUY or SELL rows from one Helius transaction using simple wallet-centric heuristics."""
    # TODO: upgrade swap parsing with program-specific decoders for tighter attribution and fee handling.
    token_transfers = tx.get("tokenTransfers", []) or []
    received_mint, received_amount = _sum_token_amount(token_transfers, wallet, "in")
    sent_mint, sent_amount = _sum_token_amount(token_transfers, wallet, "out")
    sol_out = _sum_sol_amount(tx, wallet, "out")
    sol_in = _sum_sol_amount(tx, wallet, "in")
    block_time = tx.get("timestamp")
    signature = tx.get("signature")
    source_program = tx.get("source") or tx.get("type")

    rows: list[dict[str, Any]] = []
    if received_mint and sol_out > 0 and received_amount > 0:
        rows.append(
            {
                "wallet": wallet,
                "signature": signature,
                "block_time": pd.to_datetime(block_time, unit="s", utc=True),
                "token_mint": received_mint,
                "side": "BUY",
                "sol_amount": sol_out,
                "token_amount": received_amount,
                "source_program": source_program,
            }
        )

    if sent_mint and sol_in > 0 and sent_amount > 0:
        rows.append(
            {
                "wallet": wallet,
                "signature": signature,
                "block_time": pd.to_datetime(block_time, unit="s", utc=True),
                "token_mint": sent_mint,
                "side": "SELL",
                "sol_amount": sol_in,
                "token_amount": sent_amount,
                "source_program": source_program,
            }
        )

    return rows


def wallet_from_path(path: Path) -> str:
    """Infer the wallet address from a raw JSONL file path."""
    return path.stem


def main() -> None:
    """CLI entry point."""
    config = load_app_config()
    logger = configure_logging(logger_name=__name__)
    all_rows: list[dict[str, Any]] = []

    for path in sorted(config.paths["raw_wallet_dir"].glob("*.jsonl")):
        wallet = wallet_from_path(path)
        for tx in read_jsonl(path):
            all_rows.extend(extract_trade(wallet, tx))

    trades = pd.DataFrame(all_rows)
    if trades.empty:
        trades = pd.DataFrame(
            columns=[
                "wallet",
                "signature",
                "block_time",
                "token_mint",
                "side",
                "sol_amount",
                "token_amount",
                "source_program",
            ]
        )
    else:
        trades = trades.drop_duplicates(
            subset=["wallet", "signature", "token_mint", "side"]
        ).sort_values("block_time")

    write_parquet(trades, dataset_path(config, "silver", "wallet_trades.parquet"))
    logger.info("Saved %s normalized wallet trades", len(trades))


if __name__ == "__main__":
    main()
