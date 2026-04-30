"""[TEMPLATE — won't run as-is. See docs/COLLECTORS.md.

This file shows the wallet-pool input schema and a sample fetch shape.
The actual scrape/API call has been replaced with placeholders so the
data flow is intact but no specific vendor is named. Plug in your own
data source by rewriting the fetch function — the rest of the pipeline
(refresh_wallet_pool, score_wallets_helius) consumes the parquet output
unchanged.]

Collect alpha wallets from PROVIDER_B via the official `your-wallet-provider-cli` tool.

Shells out to `your-wallet-provider-cli track smartmoney` and `your-wallet-provider-cli track kol` (Solana
chain), parses the JSON trade records, and writes the unique maker wallets to
`data/bronze/provider_b_wallets.parquet` tagged with which PROVIDER_B bucket surfaced them.

The endpoints return individual trade records (not leaderboards), so we
aggregate unique `maker` addresses across the response and union their
`maker_info.tags` labels. Sandwich/MEV bots (`sandwich_bot`) are filtered out
because they don't carry directional alpha even though PROVIDER_B surfaces them.

Assumes `your-wallet-provider-cli` is installed globally (npm i -g your-wallet-provider-cli  # install your data provider CLI here) and
`~/.config/provider_b/.env` holds `PROVIDER_B_API_KEY` and `PROVIDER_B_PRIVATE_KEY`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from typing import Any

import pandas as pd

from src.utils.io import dataset_path, load_app_config, write_parquet
from src.utils.logging_utils import configure_logging
from src.utils.time_utils import utcnow

EXCLUDE_TAGS = {"sandwich_bot", "wash_trader", "arbitrager"}
STREAMS = [
    ("smartmoney", "provider_b_smart_money"),
    ("kol", "provider_b_kol"),
]


def run_provider_b_cli(stream: str, *, chain: str, limit: int) -> list[dict[str, Any]]:
    cli = shutil.which("your-wallet-provider-cli")
    if not cli:
        raise SystemExit(
            "your-wallet-provider-cli not found in PATH. Install with: npm i -g your-wallet-provider-cli  # install your data provider CLI here"
        )
    cmd = [cli, "track", stream, "--chain", chain, "--limit", str(limit), "--raw"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise SystemExit(
            f"your-wallet-provider-cli track {stream} failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )
    payload = json.loads(result.stdout)
    records = payload.get("list") or payload.get("data") or []
    if not isinstance(records, list):
        return []
    return records


def aggregate_records(records: list[dict[str, Any]], bucket: str) -> dict[str, dict[str, Any]]:
    """Fold trade records → {wallet: {tags, first_seen_ts, trade_count}}."""
    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        maker = rec.get("maker")
        if not isinstance(maker, str) or not maker:
            continue
        info = rec.get("maker_info") or {}
        tags = [t for t in (info.get("tags") or []) if isinstance(t, str)]
        if any(t in EXCLUDE_TAGS for t in tags):
            continue
        ts_val = rec.get("timestamp")
        try:
            ts = int(ts_val) if ts_val is not None else 0
        except (TypeError, ValueError):
            ts = 0
        entry = out.setdefault(
            maker,
            {
                "tags": set(),
                "first_seen_ts": ts if ts else 0,
                "trade_count": 0,
                "bucket": bucket,
            },
        )
        entry["tags"].update(tags)
        entry["trade_count"] += 1
        if ts and (entry["first_seen_ts"] == 0 or ts < entry["first_seen_ts"]):
            entry["first_seen_ts"] = ts
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect PROVIDER_B alpha wallets via your-wallet-provider-cli."
    )
    parser.add_argument("--chain", default="sol")
    parser.add_argument("--limit", type=int, default=200, help="Page size per stream (max 200)")
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="How many spaced calls to each stream (spreads sampling, bigger pool)",
    )
    parser.add_argument(
        "--pass-interval-sec",
        type=float,
        default=5.0,
        help="Seconds between passes",
    )
    args = parser.parse_args()

    config = load_app_config()
    logger = configure_logging(logger_name=__name__)

    combined: dict[str, dict[str, Any]] = {}
    captured_at = utcnow()

    for stream, bucket in STREAMS:
        for pass_idx in range(args.passes):
            try:
                records = run_provider_b_cli(stream, chain=args.chain, limit=args.limit)
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001 — defensive; CLI can hiccup
                logger.warning(
                    "your-wallet-provider-cli %s pass %d failed: %s",
                    stream,
                    pass_idx + 1,
                    exc,
                )
                records = []
            logger.info(
                "  stream=%s pass=%d/%d records=%d",
                stream,
                pass_idx + 1,
                args.passes,
                len(records),
            )
            aggregated = aggregate_records(records, bucket)
            for wallet, info in aggregated.items():
                if wallet in combined:
                    prior = combined[wallet]
                    prior["tags"].update(info["tags"])
                    prior["trade_count"] += info["trade_count"]
                    if info["first_seen_ts"] and (
                        prior["first_seen_ts"] == 0
                        or info["first_seen_ts"] < prior["first_seen_ts"]
                    ):
                        prior["first_seen_ts"] = info["first_seen_ts"]
                    if prior["bucket"] != info["bucket"]:
                        prior["bucket"] = "provider_b_multi"
                else:
                    combined[wallet] = {**info, "tags": set(info["tags"])}
            if pass_idx + 1 < args.passes:
                time.sleep(args.pass_interval_sec)

    if not combined:
        logger.warning("PROVIDER_B returned 0 wallets — writing empty parquet")
        df = pd.DataFrame(
            columns=[
                "wallet",
                "source",
                "tags",
                "first_seen_ts",
                "trade_count",
                "collected_at",
            ]
        )
    else:
        rows = [
            {
                "wallet": wallet,
                "source": info["bucket"],
                "tags": sorted(info["tags"]),
                "first_seen_ts": int(info["first_seen_ts"]),
                "trade_count": int(info["trade_count"]),
                "collected_at": captured_at,
            }
            for wallet, info in combined.items()
        ]
        df = pd.DataFrame(rows).sort_values("trade_count", ascending=False)
        logger.info(
            "Aggregated %d unique PROVIDER_B wallets across %d trade records",
            len(df),
            int(df["trade_count"].sum()),
        )

    out_path = dataset_path(config, "bronze", "provider_b_wallets.parquet")
    write_parquet(df, out_path)
    logger.info("Wrote %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()
