# Architecture

A high-level map of how the bot is wired. For deeper detail, the relevant
modules are linked at each step.

## Data flow

```
                    ┌─────────────────────────────────┐
                    │  Wallet pool (parquet, offline) │
                    │  data/bronze/wallet_pool.parquet│
                    └──────────────┬──────────────────┘
                                   │ load at startup
                                   ▼
   ┌──────────────────────┐   ┌──────────────────────────┐
   │ Yellowstone gRPC     │──▶│  HeliusWebsocketMonitor  │
   │ (Chainstack/Helius)  │   │  src/monitoring/         │
   └──────────────────────┘   └────────────┬─────────────┘
                                           │ swap events (BUY/SELL)
                                           ▼
                              ┌────────────────────────────┐
                              │  BotRunner event loop      │
                              │  src/bot/runner.py         │
                              └────────────┬───────────────┘
                                           │
                ┌──────────────────────────┼─────────────────────────┐
                ▼                          ▼                         ▼
   ┌────────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
   │ build_runtime_features │  │ select_rule          │  │ ML qualifier         │
   │ src/strategy/          │  │ src/strategy/        │  │ src/ml/              │
   │  feature_runtime.py    │  │  rule_matcher.py     │  │  off|shadow|gate     │
   └────────────────────────┘  └──────────────────────┘  └──────────────────────┘
                                           │
                                           ▼
                              ┌────────────────────────────┐
                              │  EntryEngine               │
                              │  src/strategy/             │
                              │   entry_engine.py          │
                              └─┬──────────────────────────┘
                                │ paper                      live
                                ▼                            ▼
                ┌────────────────────────┐  ┌────────────────────────────┐
                │ Paper executor         │  │ TradeExecutor              │
                │ (simulate + persist)   │  │ src/execution/             │
                └────────────┬───────────┘  │  ├─ JupiterClient (quote)  │
                             │              │  ├─ JupiterClient (build)  │
                             │              │  ├─ LocalSigner (solders)  │
                             │              │  └─ Broadcaster (Helius)   │
                             │              └────────────┬───────────────┘
                             │                           │
                             └─────────────┬─────────────┘
                                           ▼
                              ┌────────────────────────────┐
                              │  Portfolio + storage       │
                              │  src/portfolio/            │
                              │  src/storage/  (SQLite)    │
                              └────────────┬───────────────┘
                                           │
                                           ▼
                              ┌────────────────────────────┐
                              │  ExitEngine                │
                              │  staged TPs, stop-loss,    │
                              │  trailing, dead-token, etc.│
                              └────────────┬───────────────┘
                                           │
                                           ▼
                                       (close cycle)
```

## Subsystems

Brief role for each top-level package under `src/`.

| Package | Role |
|---|---|
| `src/bot/` | Orchestrator. `runner.py` is the main event loop, `main.py` is the entry point, `config.py` loads env into a typed config dataclass. |
| `src/monitoring/` | Yellowstone gRPC client. Subscribes to Pump.fun / Raydium / Pump AMM / LaunchLab programs (configurable), parses swap-like transactions, emits BUY/SELL events with optional tracked-wallet annotation. |
| `src/strategy/` | Feature builder, rule matcher, entry engine, exit engine, regime detector, rug-check helpers. The bot's "brain". |
| `src/execution/` | Paper and live executors. Live path uses Jupiter for quotes/swap-tx and Helius RPC for broadcast. Local signing via `solders`. |
| `src/portfolio/` | Position state machine, PnL accounting, risk-limit enforcement, exit-condition evaluation. |
| `src/storage/` | SQLite schema + persistence layer for positions, executions, rule performance, daily risk counters. |
| `src/ml/` | Lightweight binary classifier (logistic by default) used as an entry qualifier. Three modes: `off` (disabled), `shadow` (predict, don't gate), `gate` (predict, reject below threshold). |
| `src/clients/` | API clients. `helius_client.py` for the Helius Enhanced API (transaction history). |
| `src/collectors/` | Offline wallet-data ingestion. `refresh_wallet_pool.py` is the union/scoring step that produces `wallet_pool.parquet`. The four `collect_provider_X_wallets.py` files are templates — bring your own data source. |
| `src/transforms/` | Phase-1 dataset pipelines: normalize wallet trades, build the token universe, build entry features, label outcomes. |
| `src/notifications/` | Optional Telegram alerts on entries/exits. |
| `src/dashboard/` | Self-contained local web UI (FastAPI + Jinja2 + HTMX). Reads `data/live/bot_state.db` and `data/live/events.jsonl` for live monitoring. |
| `src/utils/` | Config loading, logging, time + dataset helpers, app initialization. |

## Threading + concurrency

The bot runs as a single process with a synchronous event loop in
`BotRunner`. The Yellowstone gRPC client offloads protobuf parsing to a
thread pool so the main loop isn't blocked by deserialization. Live
broadcast uses `httpx.AsyncClient` for the Jupiter + Helius RPC path.

Live mode rebroadcasts the same signed transaction every
`LIVE_REBROADCAST_INTERVAL_MS` until confirmation or
`LIVE_MAX_REBROADCAST_ATTEMPTS` is reached. This is intentional — Solana
sometimes drops valid transactions under load and the cheap fix is to
keep broadcasting while polling `getSignatureStatuses` in parallel.

## Why no testnet for live mode

Solana devnet/testnet have no Pump.fun / Raydium / Helius streams that
match mainnet behavior. There is no environment in which "live mode"
runs without real funds. **Paper mode is the only safe way to evaluate
this bot.**

## Persistence

- `data/live/bot_state.db` — SQLite, all positions/executions/risk state.
- `data/live/events.jsonl` — append-only structured event log.
- `data/bronze/`, `data/silver/`, `data/gold/` — phase-1 datasets, owned
  by `src/transforms/`.
- `outputs/` — rule packs, regime metadata, research summaries. The
  runtime loads from `outputs/rules/pump_rule_packs_v2.csv` by default
  (override via `PUMP_RULES_PATH`).

## Security boundaries

- The signing key (`BOT_PRIVATE_KEY_B58`) is loaded only by
  `src/execution/signer.py`. It is never logged, transmitted, or written
  to disk.
- If `BOT_PUBLIC_KEY` is set, the signer cross-checks the derived
  pubkey at startup as a copy-paste-mistake guard.
- `ENABLE_AUTO_TRADING` is the master live-mode switch. With it `false`,
  `--live` is silently downgraded to paper.
- All risk gates (max position size, max exposure, daily-loss cap, max
  open positions) are enforced before signing, not after.
