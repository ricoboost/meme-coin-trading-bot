"""Mine early buyer wallets for a seed list of winning tokens via Helius.

Given a list of "known winner" token mints (e.g. the 5 winners extracted
from data/wallet_failed_entries_12h_enriched.csv), pull the first N SWAP
transactions on each mint via Helius, extract the wallets that bought (signed
with SOL-out + token-in), and write them to data/bronze/helius_token_buyers.parquet.

Seed list is read from data/bronze/seed_winner_tokens.txt (one mint per
line) if present, else falls back to the hard-coded defaults derived from
today's enrichment run. Add more winners to that file as you find them.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.io import dataset_path, load_app_config, write_parquet
from src.utils.logging_utils import configure_logging
from src.utils.time_utils import utcnow

WSOL = "So11111111111111111111111111111111111111112"
LAMPORTS_PER_SOL = 1_000_000_000
DEFAULT_SEEDS = [
    "GF6SFVJ1Wx15HhBpZ2VR18mdi9awpvuG8UAaKu7Rpump",
    "CGiF7jn2Sj23qCzEZDy5zk1NqbBizK7chrrDmFyDpump",
    "3FyHNz9wCauQvjL7LBHrNt6eUtowfGz9htLHKPAUpump",
    "EV5px3Vge5UR5tpyjrAj835WGVCNLUxgo9EKii2Ypump",
    "E89kJBg8MeCN2bnNWPrmTuzRHSrWF1HP8XvaGJUZpump",
]


def load_seed_tokens(root: Path) -> list[str]:
    seed_file = root / "data" / "bronze" / "seed_winner_tokens.txt"
    if seed_file.exists():
        tokens: list[str] = []
        for line in seed_file.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            mint = stripped.split("#", 1)[0].strip()
            if mint:
                tokens.append(mint)
        if tokens:
            return tokens
    return list(DEFAULT_SEEDS)


def fetch_token_swaps(
    mint: str, api_key: str, cache_dir: Path, limit_txs: int = 200
) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{mint}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        # Honor cache only if it covers the current limit. Otherwise we'd
        # silently undersample when the caller bumps limit_txs between runs.
        if len(cached) >= limit_txs or len(cached) < 100:
            # len < 100 means Helius returned everything it had (batch short-
            # circuited), so there's nothing more to fetch.
            return cached

    url = f"https://api.helius.xyz/v0/addresses/{mint}/transactions"
    all_txs: list[dict[str, Any]] = []
    before: str | None = None
    while len(all_txs) < limit_txs:
        params: dict[str, Any] = {"api-key": api_key, "type": "SWAP", "limit": 100}
        if before:
            params["before"] = before
        for attempt in range(5):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    time.sleep(2**attempt)
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 4:
                    resp = None
                    break
                time.sleep(2**attempt)
        if resp is None:
            break
        batch = resp.json()
        if not isinstance(batch, list) or not batch:
            break
        all_txs.extend(batch)
        before = batch[-1].get("signature")
        if len(batch) < 100:
            break
        time.sleep(0.15)
    cache_path.write_text(json.dumps(all_txs))
    return all_txs


def extract_buyer_from_tx(tx: dict[str, Any], target_mint: str) -> str | None:
    """Return the wallet that appears to have bought target_mint in this tx.

    A buyer has SOL going out (nativeTransfers from them and/or WSOL from them)
    AND target_mint going in (tokenTransfers to them). We pick the strongest
    such wallet.
    """
    sol_delta: dict[str, float] = defaultdict(float)
    token_delta: dict[str, float] = defaultdict(float)

    for nt in tx.get("nativeTransfers", []) or []:
        amt = float(nt.get("amount", 0)) / LAMPORTS_PER_SOL
        to_w = nt.get("toUserAccount")
        fr_w = nt.get("fromUserAccount")
        if to_w:
            sol_delta[to_w] += amt
        if fr_w:
            sol_delta[fr_w] -= amt

    for tt in tx.get("tokenTransfers", []) or []:
        mint = tt.get("mint")
        if not mint:
            continue
        amt = float(tt.get("tokenAmount", 0) or 0)
        to_w = tt.get("toUserAccount")
        fr_w = tt.get("fromUserAccount")
        if mint == WSOL:
            if to_w:
                sol_delta[to_w] += amt
            if fr_w:
                sol_delta[fr_w] -= amt
            continue
        if mint == target_mint:
            if to_w:
                token_delta[to_w] += amt
            if fr_w:
                token_delta[fr_w] -= amt

    best: tuple[str, float] | None = None
    for wallet, tok_gain in token_delta.items():
        if tok_gain <= 0:
            continue
        sol_spent = -sol_delta.get(wallet, 0.0)
        if sol_spent <= 0.001:
            continue
        if best is None or sol_spent > best[1]:
            best = (wallet, sol_spent)
    return best[0] if best else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine early buyers of winner tokens via Helius.")
    parser.add_argument("--limit-per-token", type=int, default=400)
    args = parser.parse_args()

    config = load_app_config()
    logger = configure_logging(logger_name=__name__)
    load_dotenv(config.root_dir / ".env")
    api_key = os.getenv("HELIUS_API_KEY") or config.env.get("HELIUS_API_KEY")
    if not api_key:
        raise SystemExit("HELIUS_API_KEY missing from .env")

    seeds = load_seed_tokens(config.root_dir)
    logger.info("Mining %d seed tokens", len(seeds))
    cache_dir = config.root_dir / "data" / "raw" / "helius_token_swaps"

    captured_at = utcnow()
    rows: list[dict[str, Any]] = []
    wallet_hits: dict[str, dict[str, Any]] = {}

    for i, mint in enumerate(seeds, 1):
        txs = fetch_token_swaps(mint, api_key, cache_dir, limit_txs=args.limit_per_token)
        txs.sort(key=lambda t: t.get("timestamp", 0))
        buyers_in_order: list[tuple[str, float]] = []
        for tx in txs:
            buyer = extract_buyer_from_tx(tx, mint)
            if buyer:
                buyers_in_order.append((buyer, float(tx.get("timestamp", 0))))

        seen_in_token: set[str] = set()
        for rank, (wallet, ts) in enumerate(buyers_in_order, 1):
            if wallet in seen_in_token:
                continue
            seen_in_token.add(wallet)
            rows.append(
                {
                    "wallet": wallet,
                    "source": "helius_token_buyer",
                    "seed_token": mint,
                    "buyer_rank_on_token": rank,
                    "first_buy_ts": int(ts),
                    "collected_at": captured_at,
                }
            )
            hit = wallet_hits.setdefault(wallet, {"tokens_hit": 0, "best_rank": rank})
            hit["tokens_hit"] += 1
            hit["best_rank"] = min(hit["best_rank"], rank)

        logger.info(
            "[%d/%d] %s: %d txs, %d unique buyers",
            i,
            len(seeds),
            mint,
            len(txs),
            len(seen_in_token),
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["wallet", "buyer_rank_on_token"])
    out_path = dataset_path(config, "bronze", "helius_token_buyers.parquet")
    write_parquet(df, out_path)
    logger.info(
        "Wrote %s (%d buyer rows, %d unique wallets across %d seed tokens)",
        out_path,
        len(df),
        df["wallet"].nunique() if not df.empty else 0,
        len(seeds),
    )


if __name__ == "__main__":
    main()
