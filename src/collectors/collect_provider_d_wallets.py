"""[TEMPLATE — won't run as-is. See docs/COLLECTORS.md.

This file shows the wallet-pool input schema and a sample fetch shape.
The actual scrape/API call has been replaced with placeholders so the
data flow is intact but no specific vendor is named. Plug in your own
data source by rewriting the fetch function — the rest of the pipeline
(refresh_wallet_pool, score_wallets_helius) consumes the parquet output
unchanged.]

Collect Solana wallets from a Provider D Analytics saved query.

The query is user-supplied (see `PROVIDER_D_SOLANA_WALLETS_QUERY_ID` in `.env`). We
do not assume a schema — pick which column holds the wallet address via
`PROVIDER_D_WALLET_COLUMN` (default: `wallet`). Optional label column via
`PROVIDER_D_LABEL_COLUMN`.

Default mode uses the latest cached results for the query — no execution
credits burned. Pass `--execute` to trigger a fresh run and poll until it
completes (costs Provider D credits).

Writes `data/bronze/provider_d_wallets.parquet`. If the API key or query ID is
missing, writes an empty parquet so the orchestrator keeps moving.
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

PROVIDER_D_BASE = "https://your-wallet-provider.example.com/api/v1"


def _headers(api_key: str) -> dict[str, str]:
    return {"X-Provider D-API-Key": api_key}


def fetch_latest_results(api_key: str, query_id: int, *, limit: int) -> list[dict[str, Any]]:
    url = f"{PROVIDER_D_BASE}/query/{query_id}/results"
    resp = requests.get(
        url,
        headers=_headers(api_key),
        params={"limit": limit},
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.json() or {}
    return (body.get("result") or {}).get("rows") or []


def execute_and_wait(
    api_key: str,
    query_id: int,
    *,
    limit: int,
    poll_interval: float,
    max_wait_sec: int,
    logger,
) -> list[dict[str, Any]]:
    exec_resp = requests.post(
        f"{PROVIDER_D_BASE}/query/{query_id}/execute",
        headers=_headers(api_key),
        timeout=30,
    )
    exec_resp.raise_for_status()
    execution_id = (exec_resp.json() or {}).get("execution_id")
    if not execution_id:
        raise RuntimeError("Provider D execute returned no execution_id")
    logger.info("  execution_id=%s — polling", execution_id)

    deadline = time.time() + max_wait_sec
    status_url = f"{PROVIDER_D_BASE}/execution/{execution_id}/status"
    while time.time() < deadline:
        status_resp = requests.get(status_url, headers=_headers(api_key), timeout=30)
        status_resp.raise_for_status()
        state = (status_resp.json() or {}).get("state", "")
        if state == "QUERY_STATE_COMPLETED":
            break
        if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
            raise RuntimeError(f"Provider D execution ended in state={state}")
        time.sleep(poll_interval)
    else:
        raise TimeoutError(f"Provider D execution {execution_id} did not finish in {max_wait_sec}s")

    results_resp = requests.get(
        f"{PROVIDER_D_BASE}/execution/{execution_id}/results",
        headers=_headers(api_key),
        params={"limit": limit},
        timeout=60,
    )
    results_resp.raise_for_status()
    return ((results_resp.json() or {}).get("result") or {}).get("rows") or []


def normalize_rows(
    rows: list[dict[str, Any]],
    *,
    wallet_col: str,
    label_col: str | None,
    captured_at: str,
) -> pd.DataFrame:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        wallet = r.get(wallet_col)
        if not isinstance(wallet, str) or not wallet:
            continue
        entry = out.setdefault(
            wallet,
            {"labels": set(), "trade_count": 0},
        )
        if label_col:
            label = r.get(label_col)
            if isinstance(label, str) and label:
                entry["labels"].add(label)
        entry["trade_count"] += 1

    if not out:
        return pd.DataFrame(
            columns=[
                "wallet",
                "source",
                "labels",
                "first_seen_ts",
                "trade_count",
                "collected_at",
            ]
        )

    records = [
        {
            "wallet": wallet,
            "source": "provider_d",
            "labels": sorted(info["labels"]),
            "first_seen_ts": captured_at,
            "trade_count": int(info["trade_count"]),
            "collected_at": captured_at,
        }
        for wallet, info in out.items()
    ]
    return pd.DataFrame(records).sort_values("trade_count", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Solana wallets from a Provider D saved query."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Trigger a fresh query execution (uses Provider D credits). Default: use latest cached results.",
    )
    parser.add_argument("--limit", type=int, default=10000, help="Max rows to pull from Provider D")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Polling interval (sec) when --execute",
    )
    parser.add_argument(
        "--max-wait-sec", type=int, default=300, help="Max wait for execution to finish"
    )
    args = parser.parse_args()

    config = load_app_config()
    logger = configure_logging(logger_name=__name__)
    load_dotenv(config.root_dir / ".env")

    api_key = os.getenv("PROVIDER_D_API_KEY", "").strip()
    query_id_raw = os.getenv("PROVIDER_D_SOLANA_WALLETS_QUERY_ID", "").strip()
    wallet_col = os.getenv("PROVIDER_D_WALLET_COLUMN", "wallet").strip() or "wallet"
    label_col_env = os.getenv("PROVIDER_D_LABEL_COLUMN", "").strip()
    label_col = label_col_env or None

    out_path = dataset_path(config, "bronze", "provider_d_wallets.parquet")

    def write_empty(reason: str) -> None:
        logger.warning("%s — writing empty parquet", reason)
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

    if not api_key:
        write_empty("PROVIDER_D_API_KEY not set")
        return
    if not query_id_raw:
        write_empty("PROVIDER_D_SOLANA_WALLETS_QUERY_ID not set")
        return
    try:
        query_id = int(query_id_raw)
    except ValueError:
        write_empty(f"PROVIDER_D_SOLANA_WALLETS_QUERY_ID is not an integer: {query_id_raw!r}")
        return

    captured_at = utcnow()
    logger.info(
        "Provider D collection: query_id=%d wallet_col=%s label_col=%s execute=%s",
        query_id,
        wallet_col,
        label_col,
        args.execute,
    )

    try:
        if args.execute:
            rows = execute_and_wait(
                api_key,
                query_id,
                limit=args.limit,
                poll_interval=args.poll_interval,
                max_wait_sec=args.max_wait_sec,
                logger=logger,
            )
        else:
            rows = fetch_latest_results(api_key, query_id, limit=args.limit)
    except (requests.RequestException, RuntimeError, TimeoutError) as exc:
        write_empty(f"Provider D fetch failed: {exc}")
        return

    logger.info("  fetched %d rows from Provider D", len(rows))
    if rows and wallet_col not in rows[0]:
        cols = sorted(rows[0].keys())
        write_empty(
            f"wallet column {wallet_col!r} not found in Provider D result. Available columns: {cols}"
        )
        return

    df = normalize_rows(rows, wallet_col=wallet_col, label_col=label_col, captured_at=captured_at)
    logger.info("Aggregated %d unique Provider D wallets", len(df))
    write_parquet(df, out_path)
    logger.info("Wrote %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()
