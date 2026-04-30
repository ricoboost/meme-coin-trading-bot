"""[TEMPLATE — won't run as-is. See docs/COLLECTORS.md.

This file shows the wallet-pool input schema and a sample fetch shape.
The actual scrape/API call has been replaced with placeholders so the
data flow is intact but no specific vendor is named. Plug in your own
data source by rewriting the fetch function — the rest of the pipeline
(refresh_wallet_pool, score_wallets_helius) consumes the parquet output
unchanged.]

Refresh the winner-token seed list that feeds collect_helius_token_buyers.

Pulls `your-wallet-provider-cli market trending` for Solana across short (6h) and long (24h)
windows, filters out wash/honeypot/low-quality tokens, and writes the surviving
mints to `data/bronze/seed_winner_tokens.txt`.

The next stage (`collect_helius_token_buyers`) reads that file and mines the
early buyer wallets of each mint via Helius, which is the Orb-style "catch the
wallets that were early on pumped tokens" signal the user asked for.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from src.utils.io import load_app_config
from src.utils.logging_utils import configure_logging

INTERVALS = ["1h", "6h", "24h"]


def run_provider_b_trending(interval: str, *, limit: int) -> list[dict[str, Any]]:
    cli = shutil.which("your-wallet-provider-cli")
    if not cli:
        raise SystemExit(
            "your-wallet-provider-cli not found. npm i -g your-wallet-provider-cli  # install your data provider CLI here"
        )
    cmd = [
        cli,
        "market",
        "trending",
        "--chain",
        "sol",
        "--interval",
        interval,
        "--limit",
        str(limit),
        "--raw",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise SystemExit(
            f"your-wallet-provider-cli market trending ({interval}) failed: {result.stderr.strip()}"
        )
    payload = json.loads(result.stdout)
    rank = (payload.get("data") or {}).get("rank") or []
    return rank if isinstance(rank, list) else []


def keep_token(
    t: dict[str, Any],
    *,
    min_price_change_pct: float,
    min_smart_degen: int,
    min_volume_usd: float,
    min_liquidity_usd: float,
) -> bool:
    if t.get("is_wash_trading") or t.get("is_honeypot"):
        return False
    if float(t.get("price_change_percent") or 0.0) < min_price_change_pct:
        return False
    if int(t.get("smart_degen_count") or 0) < min_smart_degen:
        return False
    if float(t.get("volume") or 0.0) < min_volume_usd:
        return False
    if float(t.get("liquidity") or 0.0) < min_liquidity_usd:
        return False
    return isinstance(t.get("address"), str) and bool(t["address"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh winner-token seed list via PROVIDER_B trending."
    )
    parser.add_argument(
        "--limit-per-interval",
        type=int,
        default=100,
        help="provider_b limit per window",
    )
    parser.add_argument("--min-price-change-pct", type=float, default=30.0)
    parser.add_argument("--min-smart-degen", type=int, default=3)
    parser.add_argument("--min-volume-usd", type=float, default=25_000.0)
    parser.add_argument("--min-liquidity-usd", type=float, default=10_000.0)
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=60,
        help="Cap to keep downstream Helius cost sane",
    )
    args = parser.parse_args()

    config = load_app_config()
    logger = configure_logging(logger_name=__name__)

    seen: dict[str, dict[str, Any]] = {}
    for interval in INTERVALS:
        try:
            rows = run_provider_b_trending(interval, limit=args.limit_per_interval)
        except SystemExit:
            raise
        logger.info("  interval=%s returned %d tokens", interval, len(rows))
        for t in rows:
            if not keep_token(
                t,
                min_price_change_pct=args.min_price_change_pct,
                min_smart_degen=args.min_smart_degen,
                min_volume_usd=args.min_volume_usd,
                min_liquidity_usd=args.min_liquidity_usd,
            ):
                continue
            mint = t["address"]
            prior = seen.get(mint)
            pchg = float(t.get("price_change_percent") or 0.0)
            sdeg = int(t.get("smart_degen_count") or 0)
            if prior is None or pchg > prior["price_change_pct"]:
                seen[mint] = {
                    "mint": mint,
                    "symbol": t.get("symbol", ""),
                    "price_change_pct": pchg,
                    "smart_degen_count": sdeg,
                    "intervals": (prior["intervals"] if prior else set()) | {interval},
                }
            else:
                prior["intervals"].add(interval)

    if not seen:
        logger.warning("No trending tokens passed filters — leaving seed file untouched")
        return

    ranked = sorted(
        seen.values(),
        key=lambda r: (r["smart_degen_count"], r["price_change_pct"]),
        reverse=True,
    )[: args.max_seeds]

    out_path: Path = config.root_dir / "data" / "bronze" / "seed_winner_tokens.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# auto-generated by src.collectors.refresh_seed_tokens — do not hand-edit",
        f"# source: your-wallet-provider-cli market trending (intervals: {', '.join(INTERVALS)})",
        f"# filters: pchg>={args.min_price_change_pct}%, smart_degen>={args.min_smart_degen}, "
        f"vol>={args.min_volume_usd:.0f}, liq>={args.min_liquidity_usd:.0f}",
    ]
    for r in ranked:
        lines.append(
            f"{r['mint']}  # {r['symbol']} pchg={r['price_change_pct']:+.1f}% "
            f"smart_degen={r['smart_degen_count']} in={sorted(r['intervals'])}"
        )
    out_path.write_text("\n".join(lines) + "\n")
    logger.info(
        "Wrote %s (%d seed tokens; top: %s pchg=%+.1f%% smart_degen=%d)",
        out_path,
        len(ranked),
        ranked[0]["symbol"],
        ranked[0]["price_change_pct"],
        ranked[0]["smart_degen_count"],
    )


if __name__ == "__main__":
    main()
