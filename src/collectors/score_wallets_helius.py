"""Helius-backed PnL scorer for candidate wallets.

For every unique wallet that appears in any of the bronze wallet-source
parquets, fetch 30 days of SWAP transactions from Helius, compute realized
PnL per token (closed positions), and aggregate. Output feeds the
refresh_wallet_pool orchestrator.

Caches raw Helius responses per wallet at
data/raw/helius_wallet_swaps_30d/{wallet}.json so re-runs are cheap.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.io import dataset_path, load_app_config, read_parquet, write_parquet
from src.utils.logging_utils import configure_logging
from src.utils.time_utils import utcnow

WSOL = "So11111111111111111111111111111111111111112"
LAMPORTS_PER_SOL = 1_000_000_000
PAGE_LIMIT = 100

SOURCE_PARQUETS = [
    ("provider_a_wallets.parquet", "provider_a"),
    ("helius_token_buyers.parquet", "helius_token_buyer"),
    ("manual_wallets.parquet", "manual"),
    ("provider_b_wallets.parquet", "provider_b"),
    ("provider_b_signal_wallets.parquet", "provider_b_signal"),
    ("provider_c_wallets.parquet", "provider_c"),
]


def load_candidates(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name, fallback_source in SOURCE_PARQUETS:
        path = root / "data" / "bronze" / name
        df = read_parquet(path)
        if df.empty or "wallet" not in df.columns:
            continue
        df = df[["wallet"]].copy()
        df["source"] = fallback_source
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["wallet", "sources"])
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["wallet"]).drop_duplicates(["wallet", "source"])
    grouped = (
        merged.groupby("wallet", as_index=False)["source"]
        .agg(lambda s: sorted(set(s)))
        .rename(columns={"source": "sources"})
    )
    return grouped


def fetch_swaps_30d(
    wallet: str,
    api_key: str,
    cache_dir: Path,
    cutoff_ts: int,
    max_pages: int = 40,
    *,
    cache_max_age_sec: int | None = None,
) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{wallet}.json"
    if cache_path.exists():
        if cache_max_age_sec is not None:
            age = time.time() - cache_path.stat().st_mtime
            if age > cache_max_age_sec:
                cache_path.unlink()
            else:
                return json.loads(cache_path.read_text())
        else:
            return json.loads(cache_path.read_text())

    url = f"https://api.helius.xyz/v0/addresses/{wallet}/transactions"
    all_txs: list[dict[str, Any]] = []
    before: str | None = None
    for _ in range(max_pages):
        params: dict[str, Any] = {
            "api-key": api_key,
            "type": "SWAP",
            "limit": PAGE_LIMIT,
        }
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
        oldest = min((tx.get("timestamp", 0) for tx in batch), default=0)
        if oldest <= cutoff_ts or len(batch) < PAGE_LIMIT:
            break
        time.sleep(0.15)
    cache_path.write_text(json.dumps(all_txs))
    return all_txs


def parse_swap(tx: dict[str, Any], wallet: str) -> dict[str, Any] | None:
    ts = int(tx.get("timestamp", 0))
    if ts == 0:
        return None
    sol_delta = 0.0
    for nt in tx.get("nativeTransfers", []) or []:
        amt = float(nt.get("amount", 0)) / LAMPORTS_PER_SOL
        if nt.get("toUserAccount") == wallet:
            sol_delta += amt
        if nt.get("fromUserAccount") == wallet:
            sol_delta -= amt

    token_delta: dict[str, float] = defaultdict(float)
    for tt in tx.get("tokenTransfers", []) or []:
        mint = tt.get("mint")
        if not mint:
            continue
        amt = float(tt.get("tokenAmount", 0) or 0)
        if tt.get("toUserAccount") == wallet:
            token_delta[mint] += amt
        if tt.get("fromUserAccount") == wallet:
            token_delta[mint] -= amt
        if mint == WSOL:
            if tt.get("toUserAccount") == wallet:
                sol_delta += amt
            if tt.get("fromUserAccount") == wallet:
                sol_delta -= amt

    candidates = [(m, d) for m, d in token_delta.items() if m != WSOL and abs(d) > 1e-12]
    if not candidates:
        return None
    candidates.sort(key=lambda kv: abs(kv[1]), reverse=True)
    mint, tok_amt = candidates[0]
    if tok_amt > 0 and sol_delta < 0:
        side = "buy"
    elif tok_amt < 0 and sol_delta > 0:
        side = "sell"
    else:
        return None
    return {
        "token": mint,
        "sol_delta": sol_delta,
        "token_delta": tok_amt,
        "side": side,
        "ts": ts,
    }


def score_wallet(wallet: str, txs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-token results, then compute the wallet-level score."""
    per_token: dict[str, dict[str, float]] = defaultdict(
        lambda: {"invested": 0.0, "realized": 0.0, "bought": 0.0, "sold": 0.0}
    )
    n_trades = 0
    last_active = 0
    seven_d_cutoff = int((datetime.now(tz=timezone.utc) - timedelta(days=7)).timestamp())
    active_7d = 0

    for tx in txs:
        parsed = parse_swap(tx, wallet)
        if parsed is None:
            continue
        n_trades += 1
        last_active = max(last_active, parsed["ts"])
        if parsed["ts"] >= seven_d_cutoff:
            active_7d += 1
        b = per_token[parsed["token"]]
        if parsed["side"] == "buy":
            b["invested"] += -parsed["sol_delta"]
            b["bought"] += parsed["token_delta"]
        else:
            b["realized"] += parsed["sol_delta"]
            b["sold"] += -parsed["token_delta"]

    total_invested = 0.0
    total_realized = 0.0
    n_closed = 0
    n_winners = 0
    for b in per_token.values():
        if b["bought"] <= 0:
            continue
        total_invested += b["invested"]
        total_realized += b["realized"]
        remaining = b["bought"] - b["sold"]
        closed_enough = remaining / max(b["bought"], 1e-9) < 0.05
        if closed_enough and b["invested"] > 0:
            n_closed += 1
            if b["realized"] > b["invested"]:
                n_winners += 1

    win_rate = (n_winners / n_closed) if n_closed > 0 else 0.0
    return {
        "wallet": wallet,
        "realized_pnl_30d": float(total_realized - total_invested),
        "total_invested_30d": float(total_invested),
        "total_realized_30d": float(total_realized),
        "n_trades_30d": int(n_trades),
        "n_closed_30d": int(n_closed),
        "n_winners_30d": int(n_winners),
        "win_rate_30d": float(win_rate),
        "active_trades_7d": int(active_7d),
        "last_active_ts": int(last_active),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidate wallets via Helius 30d backfill.")
    parser.add_argument("--limit", type=int, default=0, help="Cap wallets scored (0=all)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache + existing scores, re-fetch every wallet",
    )
    parser.add_argument(
        "--rescore-after-days",
        type=float,
        default=7.0,
        help="Reuse existing score if scored_at is younger than this (default 7)",
    )
    parser.add_argument(
        "--drop-inactive-days",
        type=float,
        default=30.0,
        help="Skip wallets whose last_active_ts is older than this — drops them from the scores file without an API call (default 30)",
    )
    args = parser.parse_args()

    config = load_app_config()
    logger = configure_logging(logger_name=__name__)
    load_dotenv(config.root_dir / ".env")
    api_key = os.getenv("HELIUS_API_KEY") or config.env.get("HELIUS_API_KEY")
    if not api_key:
        raise SystemExit("HELIUS_API_KEY missing from .env")

    candidates = load_candidates(config.root_dir)
    out_path = dataset_path(config, "silver", "wallet_scores.parquet")
    if candidates.empty:
        logger.warning("No candidate wallets found across bronze sources.")
        write_parquet(
            pd.DataFrame(
                columns=[
                    "wallet",
                    "realized_pnl_30d",
                    "total_invested_30d",
                    "total_realized_30d",
                    "n_trades_30d",
                    "n_closed_30d",
                    "n_winners_30d",
                    "win_rate_30d",
                    "active_trades_7d",
                    "last_active_ts",
                    "scored_at",
                ]
            ),
            out_path,
        )
        return

    wallets = candidates["wallet"].tolist()
    if args.limit:
        wallets = wallets[: args.limit]

    cache_dir = config.root_dir / "data" / "raw" / "helius_wallet_swaps_30d"
    if args.force:
        for p in cache_dir.glob("*.json"):
            p.unlink()

    # Load prior scores so we can reuse fresh ones and skip dead wallets.
    prior_scores: dict[str, dict[str, Any]] = {}
    if out_path.exists() and not args.force:
        try:
            prior_df = read_parquet(out_path)
            if not prior_df.empty:
                for row in prior_df.to_dict(orient="records"):
                    w = row.get("wallet")
                    if w:
                        prior_scores[w] = row
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load prior scores (%s) — rescoring everything", exc)

    now_dt = datetime.now(tz=timezone.utc)
    now_ts = int(now_dt.timestamp())
    rescore_cutoff_ts = now_ts - int(args.rescore_after_days * 86400)
    inactive_cutoff_ts = now_ts - int(args.drop_inactive_days * 86400)
    cache_max_age_sec = int(args.rescore_after_days * 86400)

    cutoff_ts = int((now_dt - timedelta(days=30)).timestamp())
    scored_at = utcnow()
    results: list[dict[str, Any]] = []
    reused = 0
    rescored = 0
    dropped_inactive = 0
    fetch_failures = 0

    logger.info(
        "scoring %d candidates (reuse_ttl=%.1fd, drop_inactive=%.1fd, prior_scores=%d)",
        len(wallets),
        args.rescore_after_days,
        args.drop_inactive_days,
        len(prior_scores),
    )

    for i, wallet in enumerate(wallets, 1):
        prior = prior_scores.get(wallet) if not args.force else None
        if prior is not None:
            prior_last_active = int(prior.get("last_active_ts") or 0)
            prior_scored_at_ts = _ts_from_scored_at(prior.get("scored_at"))

            # Dead wallets: prior data says inactive beyond threshold — drop entirely, no API call.
            if prior_last_active and prior_last_active < inactive_cutoff_ts:
                dropped_inactive += 1
                continue

            # Fresh score: reuse verbatim, no API call.
            if prior_scored_at_ts is not None and prior_scored_at_ts >= rescore_cutoff_ts:
                results.append(prior)
                reused += 1
                continue

        try:
            txs = fetch_swaps_30d(
                wallet,
                api_key,
                cache_dir,
                cutoff_ts,
                cache_max_age_sec=cache_max_age_sec,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%d/%d] %s FETCH FAIL: %s", i, len(wallets), wallet, exc)
            fetch_failures += 1
            # Keep any prior score we had so a single flaky fetch doesn't erase the wallet.
            if prior is not None:
                results.append(prior)
            continue
        score = score_wallet(wallet, txs)
        score["scored_at"] = scored_at
        results.append(score)
        rescored += 1
        if rescored % 25 == 0:
            logger.info(
                "[%d/%d] rescored=%d reused=%d dropped_inactive=%d fetch_fail=%d last: pnl=%+0.2f trades=%d win_rate=%.2f",
                i,
                len(wallets),
                rescored,
                reused,
                dropped_inactive,
                fetch_failures,
                score["realized_pnl_30d"],
                score["n_trades_30d"],
                score["win_rate_30d"],
            )

    logger.info(
        "done: rescored=%d reused=%d dropped_inactive=%d fetch_fail=%d total_out=%d",
        rescored,
        reused,
        dropped_inactive,
        fetch_failures,
        len(results),
    )

    df = pd.DataFrame(results)
    write_parquet(df, out_path)
    logger.info("Wrote %s (%d rows)", out_path, len(df))


def _ts_from_scored_at(value: Any) -> int | None:
    """Parse the scored_at column back to a unix timestamp.

    Accepts pandas Timestamp, datetime, or ISO string. Returns None if unparseable
    so the caller falls through to a rescore.
    """
    if value is None:
        return None
    try:
        if hasattr(value, "timestamp"):
            return int(value.timestamp())
        if isinstance(value, str):
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except Exception:  # noqa: BLE001
        return None
    return None


if __name__ == "__main__":
    main()
