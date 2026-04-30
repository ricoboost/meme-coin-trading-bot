"""[TEMPLATE — won't run as-is. See docs/COLLECTORS.md.

This file shows the wallet-pool input schema and a sample fetch shape.
The actual scrape/API call has been replaced with placeholders so the
data flow is intact but no specific vendor is named. Plug in your own
data source by rewriting the fetch function — the rest of the pipeline
(refresh_wallet_pool, score_wallets_helius) consumes the parquet output
unchanged.]

Collect Solana alpha wallets from Provider C's Smart Money API.

Calls `POST https://your-wallet-provider.example.com/api/v1/smart-money` filtered to
`chains=["solana"]` and all Smart-Money labels, paginates through recent
trades, aggregates unique `trader_address` values with their labels, and
writes `data/bronze/provider_c_wallets.parquet`.

Set `PROVIDER_C_API_KEY` in `.env`. Without it this collector writes an empty
parquet so the orchestrator doesn't crash.

Auth header is lowercase `apiKey: <key>` per Provider C's docs.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.io import dataset_path, load_app_config, write_parquet
from src.utils.logging_utils import configure_logging
from src.utils.time_utils import utcnow

PROVIDER_C_URL = "https://your-wallet-provider.example.com/api/v1/smart-money"
SMART_MONEY_LABELS = [
    "Fund",
    "Smart Trader",
    "30D Smart Trader",
    "90D Smart Trader",
    "180D Smart Trader",
]


def fetch_page(api_key: str, *, page: int, per_page: int, labels: list[str]) -> dict[str, Any]:
    payload = {
        "chains": ["solana"],
        "filters": {"include_smart_money_labels": labels},
        "pagination": {"page": page, "per_page": per_page},
        "order_by": [{"field": "block_timestamp", "direction": "DESC"}],
    }
    for attempt in range(5):
        try:
            resp = requests.post(
                PROVIDER_C_URL,
                headers={"apiKey": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
            if resp.status_code == 429:
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            return resp.json() or {}
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    return {}


def aggregate_trades(trades: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for t in trades:
        wallet = t.get("trader_address")
        if not isinstance(wallet, str) or not wallet:
            continue
        label = t.get("trader_address_label") or ""
        ts = t.get("block_timestamp") or ""
        entry = out.setdefault(
            wallet,
            {
                "labels": set(),
                "first_seen_ts": ts,
                "trade_count": 0,
            },
        )
        if isinstance(label, str) and label:
            entry["labels"].add(label)
        entry["trade_count"] += 1
        if ts and (not entry["first_seen_ts"] or ts < entry["first_seen_ts"]):
            entry["first_seen_ts"] = ts
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Provider C Solana smart-money wallets.")
    parser.add_argument("--per-page", type=int, default=1000, help="Records per page (max 1000)")
    parser.add_argument("--max-pages", type=int, default=5, help="How many pages to pull")
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=1.1,
        help="Sleep between pages (rate limit buffer)",
    )
    args = parser.parse_args()

    config = load_app_config()
    logger = configure_logging(logger_name=__name__)
    load_dotenv(config.root_dir / ".env")
    api_key = os.getenv("PROVIDER_C_API_KEY", "").strip()

    out_path = dataset_path(config, "bronze", "provider_c_wallets.parquet")
    if not api_key:
        logger.warning("PROVIDER_C_API_KEY not set — writing empty parquet")
        empty = pd.DataFrame(
            columns=[
                "wallet",
                "source",
                "labels",
                "first_seen_ts",
                "trade_count",
                "collected_at",
            ]
        )
        write_parquet(empty, out_path)
        return

    combined: dict[str, dict[str, Any]] = {}
    captured_at = utcnow()

    for page in range(1, args.max_pages + 1):
        try:
            body = fetch_page(api_key, page=page, per_page=args.per_page, labels=SMART_MONEY_LABELS)
        except requests.RequestException as exc:
            logger.warning("Provider C page %d failed: %s", page, exc)
            break
        trades = body.get("data") or []
        logger.info("  page=%d trades=%d", page, len(trades))
        page_agg = aggregate_trades(trades)
        for wallet, info in page_agg.items():
            if wallet in combined:
                prior = combined[wallet]
                prior["labels"].update(info["labels"])
                prior["trade_count"] += info["trade_count"]
                if info["first_seen_ts"] and (
                    not prior["first_seen_ts"] or info["first_seen_ts"] < prior["first_seen_ts"]
                ):
                    prior["first_seen_ts"] = info["first_seen_ts"]
            else:
                combined[wallet] = {**info, "labels": set(info["labels"])}

        pagination = body.get("pagination") or {}
        if pagination.get("is_last_page"):
            break
        if not trades:
            break
        time.sleep(args.sleep_sec)

    if not combined:
        logger.warning("Provider C returned 0 wallets — writing empty parquet")
        df = pd.DataFrame(
            columns=[
                "wallet",
                "source",
                "labels",
                "first_seen_ts",
                "trade_count",
                "collected_at",
            ]
        )
    else:
        rows = [
            {
                "wallet": wallet,
                "source": "provider_c_smart_money",
                "labels": sorted(info["labels"]),
                "first_seen_ts": info["first_seen_ts"],
                "trade_count": int(info["trade_count"]),
                "collected_at": captured_at,
            }
            for wallet, info in combined.items()
        ]
        df = pd.DataFrame(rows).sort_values("trade_count", ascending=False)
        logger.info(
            "Aggregated %d unique Provider C wallets across %d trade records",
            len(df),
            int(df["trade_count"].sum()),
        )

    write_parquet(df, out_path)
    logger.info("Wrote %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()
