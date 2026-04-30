# Solana wallet-cluster trading bot

A copy-trading bot for Solana memecoin markets. It tracks a pool of wallets,
detects when several of them buy the same token within a short window, and
executes paper or live trades against the resulting signal.

This is research code. **The author ran a version of it live and lost money.**
See [DISCLAIMER.md](./DISCLAIMER.md) before doing anything with it.

## Project status

This is **research code**, published as a snapshot. It is not actively
maintained, not under development, and not optimized for library reuse.

A few things to set expectations:

- The largest files are state machines that grew organically as the bot
  encountered new market conditions. `src/execution/trade_executor_live.py`
  (~5500 lines) and `src/bot/runner.py` (~4000 lines) were not refactored
  pre-publish, because every code path in them saw real-world inputs and a
  cosmetic refactor would risk breaking working logic in code that will
  never be re-run anyway. Each file has a top-of-file table of contents to
  make navigation tractable.
- The strategy did not work live. See [docs/STRATEGY.md](./docs/STRATEGY.md)
  "Why it didn't work live" for the post-mortem. The bot is published for
  educational value, not as an endorsement of the approach.
- If you're reading the code: start at [`src/bot/main.py`](./src/bot/main.py),
  follow the imports into [`src/bot/runner.py`](./src/bot/runner.py) and
  [`src/strategy/`](./src/strategy/). The `src/transforms/` package is the
  offline dataset pipeline; the runtime bot lives under `src/bot/`,
  `src/strategy/`, `src/execution/`, `src/portfolio/`.
- `pyproject.toml` configures `ruff` with the project's accepted style. The
  repo passes `ruff check` and `ruff format --check` on every file outside
  `src/monitoring/yellowstone_proto/generated/` (auto-generated protobuf).

## What it does

- Maintains a pool of "interesting" wallets (loaded from parquet — bring your
  own data source, see [docs/COLLECTORS.md](./docs/COLLECTORS.md)).
- Streams Solana program activity (Pump.fun, Raydium, etc.) via Yellowstone
  gRPC.
- Detects clusters of pool-wallet buys within a rolling window.
- Evaluates each candidate against a rule set (mined offline from historical
  data) plus a runtime ML qualifier and rug checks.
- Simulates trades in **paper mode** (default) or signs and broadcasts real
  swaps via Jupiter + Helius RPC in **live mode** (opt-in, real funds).
- Persists positions, executions, rule performance, and event logs to SQLite
  + JSONL for the local dashboard at `src/dashboard/`.

## Quickstart (paper mode, ~5 minutes)

```bash
# 1. Python 3.11+, virtualenv recommended
python3 -m venv .venv && source .venv/bin/activate

# 2. Deps + Playwright browsers (for the wallet-leaderboard collector template)
pip install -r requirements.txt
playwright install chromium

# 3. Config
cp .env.example .env
# Open .env and set HELIUS_API_KEY (free tier is fine to start).
# For live program subscription you'll also need CHAINSTACK_GRPC_ENDPOINT
# + CHAINSTACK_GRPC_TOKEN, or you can stub these with any Yellowstone-
# compatible provider.

# 4. Drop a wallet pool parquet into data/bronze/wallet_pool.parquet.
# See docs/COLLECTORS.md for the schema and a starter template.

# 5. Run the bot in paper mode
python -m src.bot.main --paper

# 6. (Optional) Watch the local dashboard in another terminal
python -m src.dashboard.main
# Then open http://127.0.0.1:8787
```

Full walkthrough in [docs/GETTING_STARTED.md](./docs/GETTING_STARTED.md).

## Live trading

Live mode is **opt-in** and trades **real SOL** on Solana mainnet. There is
no testnet path. Read [docs/LIVE_TRADING.md](./docs/LIVE_TRADING.md) and
[DISCLAIMER.md](./DISCLAIMER.md) before enabling it.

## Documentation

- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) — subsystems and data flow.
- [docs/GETTING_STARTED.md](./docs/GETTING_STARTED.md) — paper-mode walkthrough.
- [docs/CONFIG.md](./docs/CONFIG.md) — every env var, grouped by topic.
- [docs/STRATEGY.md](./docs/STRATEGY.md) — what the bot looks for and why.
- [docs/COLLECTORS.md](./docs/COLLECTORS.md) — wallet-pool input schema, BYO
  data source.
- [docs/LIVE_TRADING.md](./docs/LIVE_TRADING.md) — real-funds setup and
  safeguards.
- [DISCLAIMER.md](./DISCLAIMER.md) — read this.

## Status

This code is published as a research artifact. It is not actively maintained,
not under development, and not seeking contributors. Issues and PRs may go
unanswered. See [CONTRIBUTING.md](./CONTRIBUTING.md) if you want to try anyway.

## License

[Apache-2.0](./LICENSE).
