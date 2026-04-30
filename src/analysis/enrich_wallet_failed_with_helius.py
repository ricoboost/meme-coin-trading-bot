"""Enrich wallet_failed_entries CSV with per-wallet invested/realized SOL via Helius.

Reads data/wallet_failed_entries_12h.csv, pulls SWAP transactions from Helius
Enhanced Transactions API for each unique tracked wallet in the file, parses
them from tokenTransfers + nativeTransfers (events.swap is empty for pump.fun
txs), aggregates BUY / SELL amounts per (wallet, token), and writes an
enriched CSV with totals summed across all tracked wallets that touched each
token.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "data" / "wallet_failed_entries_12h.csv"
OUT_CSV = ROOT / "data" / "wallet_failed_entries_12h_enriched.csv"
RAW_DIR = ROOT / "data" / "raw" / "helius_wallet_swaps"

WSOL = "So11111111111111111111111111111111111111112"
LAMPORTS_PER_SOL = 1_000_000_000
PAGE_LIMIT = 100
MAX_PAGES = 20
CUTOFF_UTC = datetime(2026, 4, 20, 17, 0, 0, tzinfo=timezone.utc)


def _env(key: str) -> str:
    env_path = ROOT / ".env"
    with env_path.open() as fh:
        for line in fh:
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(f"{key} not in .env")


def load_csv() -> tuple[list[dict[str, str]], dict[str, set[str]], set[str]]:
    rows: list[dict[str, str]] = []
    wallets_by_token: dict[str, set[str]] = defaultdict(set)
    all_wallets: set[str] = set()
    with IN_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
            raw = row.get("tracked_wallets", "") or ""
            token = row["token_mint"]
            for w in (x for x in raw.split(";") if x):
                wallets_by_token[token].add(w)
                all_wallets.add(w)
    return rows, wallets_by_token, all_wallets


def fetch_wallet_swaps(wallet: str, api_key: str) -> list[dict[str, Any]]:
    """Fetch SWAP txs for wallet since CUTOFF_UTC. Pages with `before` cursor."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"{wallet}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    url = f"https://api.helius.xyz/v0/addresses/{wallet}/transactions"
    all_txs: list[dict[str, Any]] = []
    before: str | None = None
    for page in range(MAX_PAGES):
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
            except requests.RequestException as exc:
                if attempt == 4:
                    print(f"  [{wallet}] page {page} failed: {exc}", file=sys.stderr)
                    resp = None
                    break
                time.sleep(2**attempt)
        if resp is None:
            break
        batch = resp.json()
        if not isinstance(batch, list) or not batch:
            break
        all_txs.extend(batch)
        oldest_ts = min(tx.get("timestamp", 0) for tx in batch)
        oldest_dt = datetime.fromtimestamp(oldest_ts, tz=timezone.utc)
        before = batch[-1].get("signature")
        if oldest_dt < CUTOFF_UTC:
            break
        if len(batch) < PAGE_LIMIT:
            break
        time.sleep(0.15)

    cache.write_text(json.dumps(all_txs))
    return all_txs


def parse_swap(tx: dict[str, Any], wallet: str) -> dict[str, Any] | None:
    """Return {'token': mint, 'sol_delta': float, 'token_delta': float,
    'side': 'buy'|'sell', 'ts': int} or None."""
    ts = tx.get("timestamp", 0)
    if ts == 0:
        return None

    # SOL delta: native transfers + token transfers of WSOL
    sol_delta = 0.0  # positive = wallet received SOL, negative = paid SOL
    for nt in tx.get("nativeTransfers", []) or []:
        amt = float(nt.get("amount", 0)) / LAMPORTS_PER_SOL
        if nt.get("toUserAccount") == wallet:
            sol_delta += amt
        if nt.get("fromUserAccount") == wallet:
            sol_delta -= amt

    # Token deltas (by mint)
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
        # WSOL shows up here too; fold into sol_delta
        if mint == WSOL:
            if tt.get("toUserAccount") == wallet:
                sol_delta += amt
            if tt.get("fromUserAccount") == wallet:
                sol_delta -= amt

    # Pick the non-WSOL mint with non-zero delta as the target token
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


def aggregate(
    wallets: set[str], api_key: str, tokens: set[str]
) -> dict[tuple[str, str], dict[str, float]]:
    """Return {(wallet, token): {invested_sol, realized_sol, tokens_bought,
    tokens_sold, n_buys, n_sells, still_holding}}."""
    agg: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {
            "invested_sol": 0.0,
            "realized_sol": 0.0,
            "tokens_bought": 0.0,
            "tokens_sold": 0.0,
            "n_buys": 0,
            "n_sells": 0,
        }
    )
    for i, wallet in enumerate(sorted(wallets), 1):
        print(f"[{i}/{len(wallets)}] {wallet}", flush=True)
        txs = fetch_wallet_swaps(wallet, api_key)
        kept = 0
        for tx in txs:
            parsed = parse_swap(tx, wallet)
            if parsed is None:
                continue
            if parsed["token"] not in tokens:
                continue
            key = (wallet, parsed["token"])
            bucket = agg[key]
            if parsed["side"] == "buy":
                bucket["invested_sol"] += -parsed["sol_delta"]
                bucket["tokens_bought"] += parsed["token_delta"]
                bucket["n_buys"] += 1
            else:
                bucket["realized_sol"] += parsed["sol_delta"]
                bucket["tokens_sold"] += -parsed["token_delta"]
                bucket["n_sells"] += 1
            kept += 1
        print(f"   fetched={len(txs)} matched={kept}", flush=True)
    return agg


def write_enriched(
    rows: list[dict[str, str]],
    wallets_by_token: dict[str, set[str]],
    agg: dict[tuple[str, str], dict[str, float]],
) -> None:
    base_fields = list(rows[0].keys())
    extra_fields = [
        "tracked_invested_sol",
        "tracked_realized_sol",
        "tracked_pnl_sol",
        "tracked_n_buys",
        "tracked_n_sells",
        "tracked_still_holding",
        "per_wallet_pnl",
    ]
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=base_fields + extra_fields)
        writer.writeheader()
        for row in rows:
            token = row["token_mint"]
            wallets = wallets_by_token.get(token, set())
            invested = realized = 0.0
            n_buys = n_sells = 0
            still_holding = 0
            per_wallet_parts: list[str] = []
            for w in sorted(wallets):
                b = agg.get((w, token))
                if not b:
                    per_wallet_parts.append(f"{w[:6]}:none")
                    continue
                invested += b["invested_sol"]
                realized += b["realized_sol"]
                n_buys += int(b["n_buys"])
                n_sells += int(b["n_sells"])
                held = b["tokens_bought"] - b["tokens_sold"]
                if held > 1e-9 and b["n_buys"] > 0:
                    still_holding += 1
                pnl_w = b["realized_sol"] - b["invested_sol"]
                per_wallet_parts.append(
                    f"{w[:6]}:{b['invested_sol']:.3f}/{b['realized_sol']:.3f}/{pnl_w:+.3f}"
                )
            pnl = realized - invested
            out = dict(row)
            out["tracked_invested_sol"] = f"{invested:.4f}" if invested else ""
            out["tracked_realized_sol"] = f"{realized:.4f}" if realized else ""
            out["tracked_pnl_sol"] = f"{pnl:+.4f}" if (invested or realized) else ""
            out["tracked_n_buys"] = str(n_buys) if n_buys else ""
            out["tracked_n_sells"] = str(n_sells) if n_sells else ""
            out["tracked_still_holding"] = str(still_holding) if still_holding else ""
            out["per_wallet_pnl"] = " | ".join(per_wallet_parts)
            writer.writerow(out)


def main() -> None:
    api_key = _env("HELIUS_API_KEY")
    rows, wallets_by_token, all_wallets = load_csv()
    tokens = set(wallets_by_token.keys())
    print(
        f"rows={len(rows)} tokens_with_wallets={len(tokens)} unique_wallets={len(all_wallets)}",
        flush=True,
    )
    agg = aggregate(all_wallets, api_key, tokens)
    print(f"aggregated {len(agg)} (wallet, token) pairs with activity", flush=True)
    write_enriched(rows, wallets_by_token, agg)
    print(f"wrote {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
