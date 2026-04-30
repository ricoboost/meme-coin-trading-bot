# Wallet pool collectors

The bot's trading signal depends on a curated wallet pool — the set of
addresses whose buys you treat as informed. Maintaining this pool is the
hardest non-trading problem in the project, and the part most likely to
need re-implementation when you adapt the bot to your own data sources.

## The contract

`src/bot/main.py` loads `data/bronze/wallet_pool.parquet` at startup. The
schema (see `src/collectors/refresh_wallet_pool.py` for the canonical
producer):

| Column | Type | Required | Notes |
|---|---|---|---|
| `wallet` | string | yes | Solana base58 pubkey, 32-44 chars. |
| `score` | float | yes | Composite score; runtime uses it as a signal-strength bonus. |
| `appears_daily` | int | no | "Made the daily leaderboard N times in lookback." Used for legacy bagholder filter. |
| `appears_weekly` | int | no | Same, weekly. |
| `appears_monthly` | int | no | Same, monthly. |
| `best_rank` | int | no | Best leaderboard rank achieved (lower = better). |
| `tags` | list[string] | no | Free-form; consumed by the dashboard for display. |
| `sources` | list[string] | no | Per-wallet provenance — which collectors saw it. |
| `realized_pnl_30d` | float | no | 30-day realized PnL in SOL. |
| `win_rate_30d` | float | no | 30-day win rate (0.0-1.0). |
| `n_trades_30d` | int | no | 30-day trade count. |
| `last_active_ts` | datetime | no | Last on-chain activity timestamp. |

The minimum the bot needs is `wallet` and `score`. Everything else is
used by either the dashboard (display) or the runtime
(`WALLET_POOL_EXCLUDE_BAGHOLDERS` filtering, score-based size boost).

## How the pool is built

```
   ┌──────────────────────────────────────────────┐
   │ Bronze (raw, per-source)                     │
   │  data/bronze/provider_a_wallets.parquet      │
   │  data/bronze/provider_b_wallets.parquet      │
   │  data/bronze/provider_b_signal_wallets.parquet│
   │  data/bronze/provider_c_wallets.parquet      │
   │  data/bronze/provider_d_wallets.parquet      │
   │  data/bronze/manual_wallets.parquet          │
   │  data/bronze/helius_token_buyers.parquet     │
   └────────────────────┬─────────────────────────┘
                        │ refresh_wallet_pool union + score
                        ▼
   ┌──────────────────────────────────────────────┐
   │ Silver (per-wallet metrics from Helius)      │
   │  data/silver/wallet_scores.parquet           │
   └────────────────────┬─────────────────────────┘
                        │ filter + rank
                        ▼
   ┌──────────────────────────────────────────────┐
   │ Bronze output                                │
   │  data/bronze/wallet_pool.parquet             │
   └──────────────────────────────────────────────┘
```

Run order:

```bash
# 1. Pull from each source you've configured
python -m src.collectors.collect_provider_a_wallets   # template — implement your fetch
python -m src.collectors.collect_provider_b_wallets   # template — implement your fetch
python -m src.collectors.collect_provider_c_wallets   # template — implement your fetch
python -m src.collectors.collect_provider_d_wallets   # template — implement your fetch
python -m src.collectors.collect_manual_wallets       # CSV-driven, ready to use
python -m src.collectors.collect_helius_token_buyers  # Helius Enhanced API, ready to use

# 2. Optionally pull per-wallet trade history for scoring
python -m src.collectors.collect_wallet_history --limit 100
python -m src.collectors.score_wallets_helius

# 3. Merge into the runtime pool
python -m src.collectors.refresh_wallet_pool
```

## Bring your own data source

The four `collect_provider_X_wallets.py` files are **templates**. They
demonstrate the schema your collector needs to write — they don't fetch
real data as shipped.

The original implementations were vendor-specific (a leaderboard scraper,
a CLI shell-out, two HTTP API clients). Each has been replaced with a
placeholder. To make one functional:

1. Pick the template whose shape matches your data source:
   - `collect_provider_a_wallets.py` — Playwright scraping pattern (HTML
     leaderboard).
   - `collect_provider_b_wallets.py` — subprocess CLI shell-out pattern.
   - `collect_provider_b_signal_wallets.py` — two-hop discovery pattern
     (signal → token traders → wallets).
   - `collect_provider_c_wallets.py` — paginated HTTP API pattern with
     auth header.
   - `collect_provider_d_wallets.py` — auth'd query API pattern.
2. Replace the URL / CLI placeholder with your real one.
3. Update the parsing of raw rows to extract `wallet` and any optional
   columns you have.
4. Output `data/bronze/<your-source>_wallets.parquet` with the schema
   above.
5. Add `(<your-source>_wallets.parquet, <your-source>)` to the
   `SOURCE_PARQUETS` list in `src/collectors/refresh_wallet_pool.py:35`.

## Manual collector (no scraping needed)

If you have a CSV of wallets you want to track — paste them in:

```bash
mkdir -p data/raw
cat > data/raw/manual_wallets.csv <<'EOF'
wallet,source,note
<base58-pubkey>,manual,my watch list
<another-base58>,manual,from a friend
EOF

python -m src.collectors.collect_manual_wallets
python -m src.collectors.refresh_wallet_pool
```

## Helius token-buyers collector

`src/collectors/collect_helius_token_buyers.py` pulls early buyers of a
seed token list via the Helius Enhanced API. It works out of the box —
just set `HELIUS_API_KEY` and provide a seed list. Useful if you want to
discover wallets that bought a known winner-token early.

## Refresh cadence

The wallet pool is loaded at bot startup. The pool is NOT auto-refreshed
mid-run — you re-run `refresh_wallet_pool` and restart the bot. This is
intentional: a hot-swapped pool would invalidate in-flight rolling
windows, and the wallet-list cardinality usually doesn't change fast
enough to matter intra-day.

If you want continuous refresh, the cleanest way is a cron job that runs
the collectors + `refresh_wallet_pool`, then SIGTERM-and-restarts the
bot once a day during a quiet window. The bot's exit path is clean —
open positions are closed (paper) or persisted-and-handed-off (live).
