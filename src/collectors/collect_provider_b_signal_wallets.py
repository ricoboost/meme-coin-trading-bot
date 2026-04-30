"""[TEMPLATE — won't run as-is. See docs/COLLECTORS.md.

This file shows the wallet-pool input schema and a sample fetch shape.
The actual scrape/API call has been replaced with placeholders so the
data flow is intact but no specific vendor is named. Plug in your own
data source by rewriting the fetch function — the rest of the pipeline
(refresh_wallet_pool, score_wallets_helius) consumes the parquet output
unchanged.]

Collect alpha wallets from PROVIDER_B's signal + token-traders two-hop.

Step 1: `your-wallet-provider-cli market signal --chain sol` across all signal types (1–18:
price spikes, smart-money buys, large buys, etc.) surfaces tokens that hit an
alpha-relevant event.

Step 2: For each unique signalled token, `your-wallet-provider-cli token traders --tag
smart_degen` returns pre-labeled alpha wallets that are trading that token.
This is the wallet-level payoff of the signal — who the smart-money label
lit up on.

Output: `data/bronze/provider_b_signal_wallets.parquet` (schema matches the other
PROVIDER_B collector so the scorer can union them).

Suspicious wallets (`is_suspicious=true`) are dropped because PROVIDER_B already
flagged them as likely noise. `wallet_tag_v2` labels are preserved.
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

SIGNAL_TYPES = list(range(1, 19))


def run_cli(args: list[str], *, timeout: int = 60) -> Any:
    cli = shutil.which("your-wallet-provider-cli")
    if not cli:
        raise SystemExit(
            "your-wallet-provider-cli not found. npm i -g your-wallet-provider-cli  # install your data provider CLI here"
        )
    result = subprocess.run([cli] + args, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"your-wallet-provider-cli {' '.join(args)} failed (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def fetch_signal_tokens(logger, *, signal_types: list[int], sleep_sec: float) -> dict[str, int]:
    """Return {token_mint: max_signal_times}."""
    seen: dict[str, int] = {}
    for stype in signal_types:
        try:
            rows = run_cli(
                [
                    "market",
                    "signal",
                    "--chain",
                    "sol",
                    "--signal-type",
                    str(stype),
                    "--raw",
                ]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("signal type=%d failed: %s", stype, exc)
            continue
        if not isinstance(rows, list):
            continue
        logger.info("  signal_type=%d → %d tokens", stype, len(rows))
        for r in rows:
            addr = (r.get("data") or {}).get("address") or r.get("token_address")
            if not isinstance(addr, str) or not addr:
                continue
            times = int(r.get("signal_times") or 1)
            seen[addr] = max(seen.get(addr, 0), times)
        time.sleep(sleep_sec)
    return seen


def fetch_token_traders(token: str, *, tag: str, limit: int) -> list[dict[str, Any]]:
    payload = run_cli(
        [
            "token",
            "traders",
            "--chain",
            "sol",
            "--address",
            token,
            "--tag",
            tag,
            "--limit",
            str(limit),
            "--raw",
        ]
    )
    items = payload.get("list") if isinstance(payload, dict) else None
    return items if isinstance(items, list) else []


def main() -> None:
    parser = argparse.ArgumentParser(description="PROVIDER_B signal → token traders → wallets.")
    parser.add_argument(
        "--max-tokens", type=int, default=50, help="Top-N signalled tokens to expand"
    )
    parser.add_argument(
        "--traders-per-token",
        type=int,
        default=30,
        help="Top traders fetched per token",
    )
    parser.add_argument("--sleep-sec", type=float, default=0.6, help="Sleep between CLI calls")
    parser.add_argument("--tags", nargs="+", default=["smart_degen", "renowned"])
    args = parser.parse_args()

    config = load_app_config()
    logger = configure_logging(logger_name=__name__)
    captured_at = utcnow()

    logger.info("[step 1/2] fetching signal tokens across types %s", SIGNAL_TYPES)
    token_to_signal_times = fetch_signal_tokens(
        logger, signal_types=SIGNAL_TYPES, sleep_sec=args.sleep_sec
    )
    logger.info("  %d unique signalled tokens", len(token_to_signal_times))
    if not token_to_signal_times:
        logger.warning("No signalled tokens — writing empty parquet")
        empty = pd.DataFrame(
            columns=[
                "wallet",
                "source",
                "tags",
                "signal_tokens",
                "realized_profit",
                "buy_tx_count",
                "collected_at",
            ]
        )
        write_parquet(empty, dataset_path(config, "bronze", "provider_b_signal_wallets.parquet"))
        return

    top_tokens = sorted(token_to_signal_times.items(), key=lambda kv: kv[1], reverse=True)[
        : args.max_tokens
    ]
    logger.info(
        "[step 2/2] expanding top %d tokens with traders (tags=%s)",
        len(top_tokens),
        args.tags,
    )

    wallets: dict[str, dict[str, Any]] = {}
    for i, (token, signal_times) in enumerate(top_tokens, 1):
        for tag in args.tags:
            try:
                traders = fetch_token_traders(token, tag=tag, limit=args.traders_per_token)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "  [%d/%d] token=%s tag=%s failed: %s",
                    i,
                    len(top_tokens),
                    token,
                    tag,
                    exc,
                )
                continue
            for tr in traders:
                addr = tr.get("address")
                if not isinstance(addr, str) or not addr:
                    continue
                if tr.get("is_suspicious"):
                    continue
                realized = float(tr.get("realized_profit") or 0.0)
                buys = int(tr.get("buy_tx_count_cur") or 0)
                wallet_tag = tr.get("wallet_tag_v2") or ""
                entry = wallets.setdefault(
                    addr,
                    {
                        "tags": set(),
                        "signal_tokens": set(),
                        "realized_profit": 0.0,
                        "buy_tx_count": 0,
                    },
                )
                entry["tags"].add(tag)
                if wallet_tag:
                    entry["tags"].add(wallet_tag)
                entry["signal_tokens"].add(token)
                entry["realized_profit"] += realized
                entry["buy_tx_count"] += buys
            time.sleep(args.sleep_sec)
        if i % 10 == 0:
            logger.info("  [%d/%d] running wallets=%d", i, len(top_tokens), len(wallets))

    if not wallets:
        logger.warning("No wallets surfaced — writing empty parquet")
        df = pd.DataFrame(
            columns=[
                "wallet",
                "source",
                "tags",
                "signal_tokens",
                "realized_profit",
                "buy_tx_count",
                "collected_at",
            ]
        )
    else:
        rows = [
            {
                "wallet": addr,
                "source": "provider_b_signal",
                "tags": sorted(info["tags"]),
                "signal_tokens": sorted(info["signal_tokens"]),
                "realized_profit": float(info["realized_profit"]),
                "buy_tx_count": int(info["buy_tx_count"]),
                "collected_at": captured_at,
            }
            for addr, info in wallets.items()
        ]
        df = pd.DataFrame(rows).sort_values("realized_profit", ascending=False)
        logger.info(
            "Aggregated %d unique signal-surfaced wallets (realized_profit range: %.0f → %.0f)",
            len(df),
            float(df["realized_profit"].min()),
            float(df["realized_profit"].max()),
        )

    out_path = dataset_path(config, "bronze", "provider_b_signal_wallets.parquet")
    write_parquet(df, out_path)
    logger.info("Wrote %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()
