# Getting started (paper mode)

Goal: run the bot end-to-end in paper mode, with the dashboard, in under 10
minutes. This walkthrough doesn't touch real funds.

## Prerequisites

- Python 3.11 or newer.
- A free [Helius](https://www.helius.dev/) API key (or any Solana RPC that
  supports the standard methods you'll see in `src/strategy/rug_check.py`).
- A Yellowstone gRPC endpoint. Chainstack or Helius LaserStream both work.
  This is required for `DISCOVERY_MODE=pair_first`, the default.
- ~500 MB free disk for state + logs. Live execution adds a bit more.

You don't need a Solana wallet for paper mode.

## 1. Clone and set up the environment

```bash
git clone <this-repo-url> trader-bot
cd trader-bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

The `playwright` step pulls a Chromium binary used by
`src/collectors/collect_provider_a_wallets.py` (the leaderboard scraper
template). If you don't plan to use that template, you can skip it.

## 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` in your editor. The minimum to set for paper mode:

```bash
HELIUS_API_KEY=<your-helius-key>
HELIUS_BASE_URL=https://api-mainnet.helius-rpc.com

# Required when DISCOVERY_MODE=pair_first (the default)
CHAINSTACK_GRPC_ENDPOINT=<your-grpc-host>:<port>
CHAINSTACK_GRPC_TOKEN=<your-grpc-token>
```

Everything else can stay at defaults for a first run. See
[CONFIG.md](./CONFIG.md) for the full env-var catalog.

## 3. Provide a wallet pool

The bot needs `data/bronze/wallet_pool.parquet` to exist before startup.
This file is intentionally **not** included in the repo — bring your own.

The schema is documented in [COLLECTORS.md](./COLLECTORS.md). The minimum
is a parquet with at least these columns:

- `wallet` — Solana base58 pubkey (string)
- `score` — float, used as a runtime signal-strength bonus

The fastest way to get going is the manual collector:

```bash
# Create data/raw/manual_wallets.csv with wallets you want to track
echo 'wallet,source,note' > data/raw/manual_wallets.csv
echo '<some-base58-pubkey>,manual,test wallet' >> data/raw/manual_wallets.csv

python -m src.collectors.collect_manual_wallets
python -m src.collectors.refresh_wallet_pool
```

Or, if you have your own wallet data source (a leaderboard, an API, a
CSV), edit one of the `collect_provider_X_wallets.py` templates to fetch
it, then run `refresh_wallet_pool` to merge.

## 4. Run paper mode

```bash
python -m src.bot.main --paper
```

You should see startup logs:

```
[INFO] Loading config…
[INFO] Loading rules from outputs/rules/pump_rule_packs_v2.csv
[INFO] Loaded N tracked wallets from data/bronze/wallet_pool.parquet
[INFO] Connecting to Yellowstone gRPC…
[INFO] Subscribed to programs: ...
[INFO] Bot running in PAPER mode
```

Useful flags:

```bash
python -m src.bot.main --paper --verbose                    # debug-level logs
python -m src.bot.main --paper --limit-wallets 10           # subset for quick test
python -m src.bot.main --paper --allowed-regimes momentum_burst
python -m src.bot.main --paper --with-dashboard             # bot + dashboard one process
```

Paper trades are logged with a `📝 PAPER` prefix and persisted with
`mode=paper` in the SQLite store.

## 5. Watch the dashboard

In a separate terminal:

```bash
python -m src.dashboard.main
```

Open http://127.0.0.1:8787. You should see:

- A summary header with mode (paper), exposure, realized PnL, execution count.
- Open and recent positions.
- Per-rule performance.
- Recent rejected trades with the rule-match snapshot, so you can tune
  thresholds without restarting the bot.

## 6. Let it run

Even on a busy wallet pool, signals can take minutes to fire — the cluster
detection requires multiple distinct pool wallets to buy the same token
within `CONSENSUS_WINDOW_SEC` seconds.

Spend at least an hour watching the bot before deciding anything. A full
day is better. A week of paper-mode data is what you'd want before even
considering live mode.

## What's next

- [STRATEGY.md](./STRATEGY.md) — what signals the bot is looking for and why.
- [CONFIG.md](./CONFIG.md) — every knob in the config, grouped by topic.
- [LIVE_TRADING.md](./LIVE_TRADING.md) — when (and whether) to flip to live.
- [DISCLAIMER.md](../DISCLAIMER.md) — re-read this before live mode.
