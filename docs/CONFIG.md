# Configuration reference

All configuration is via environment variables, loaded from `.env` at
startup. The full catalog is in `.env.example` with inline comments. This
doc groups the knobs by topic and explains the trade-offs.

> Required vs optional: items marked **required** are checked at startup
> and the bot refuses to run without them. Everything else has a default
> in `src/bot/config.py` or `src/utils/io.py`.

## RPC + streaming

| Var | Required | Notes |
|---|---|---|
| `HELIUS_API_KEY` | yes | Free tier works for paper mode. |
| `HELIUS_BASE_URL` | no | Default `https://api-mainnet.helius-rpc.com`. |
| `HELIUS_RPC_URL` | live only | Format: `https://mainnet.helius-rpc.com/?api-key=<KEY>`. |
| `JUPITER_BASE_URL` | live only | Default `https://api.jup.ag/swap/v2`. The lite endpoint at `https://lite-api.jup.ag/swap/v1` also works. |
| `JUPITER_API_KEY` | no | Only needed if your provider gates Jupiter v6 by key. |
| `CHAINSTACK_GRPC_ENDPOINT` | pair_first | Format `host:port` or `https://host:port`. |
| `CHAINSTACK_GRPC_TOKEN` | pair_first | Auth token from your gRPC provider. |
| `CHAINSTACK_RECONNECT_MAX_RETRIES` | no | `0` = retry forever (default). |
| `CHAINSTACK_RECONNECT_BACKOFF_INITIAL_SEC` | no | Default `1`. |
| `CHAINSTACK_RECONNECT_BACKOFF_MAX_SEC` | no | Default `30`. |

## Discovery mode

| Var | Default | Notes |
|---|---|---|
| `DISCOVERY_MODE` | `pair_first` | `pair_first` subscribes to programs and scores tokens; `wallet_first` is the legacy tracked-wallet flow. |
| `DISCOVERY_ACCOUNT_INCLUDE` | Pump/Raydium IDs | Comma-separated program IDs to subscribe to. |
| `DISCOVERY_ALLOWED_SOURCES` | `PUMP_FUN,PUMP_AMM,RAYDIUM,RAYDIUM_LAUNCHLAB` | Filter on the parser's source label. |
| `DISCOVERY_REQUIRE_PUMP_SUFFIX` | `false` | Reject tokens whose mint doesn't end in `pump`. |

## Wallet pool

| Var | Default | Notes |
|---|---|---|
| `TRACKED_WALLETS_PATH` | `data/bronze/wallet_pool.parquet` | Loaded at startup. |
| `WALLET_POOL_EXCLUDE_BAGHOLDERS` | `true` | Filters wallets that look like long-term holders rather than active traders. |
| `PROVIDER_A_TOP_N` | `50` | Used by `collect_provider_a_wallets.py` template. |

## Rule loading

| Var | Default | Notes |
|---|---|---|
| `RULES_SOURCE_MODE` | `pump` | `pump` (recommended) loads Pump V2 rule packs. `legacy` loads the older artifact set. |
| `PUMP_RULES_PATH` | `outputs/rules/pump_rule_packs_v2.csv` | Path to the rule pack CSV. |
| `ALLOW_LEGACY_RULE_FALLBACK` | `false` | Allow falling back to legacy artifacts if pump path missing. |
| `OPTIONAL_ALLOWED_REGIMES` | empty | Comma-separated regimes to whitelist. Empty = all. |
| `MAX_STRICT_RULES` | _unset_ | Optional hard cap on rule count. |
| `DISABLED_RULE_IDS` | empty | Comma-separated rule IDs to skip. |

## Position sizing + risk caps (always enforced)

| Var | Default | Notes |
|---|---|---|
| `MAX_POSITION_SOL` | `0.05` | Per-trade size cap. Trades exceeding this are rejected. |
| `MAX_TOTAL_EXPOSURE_SOL` | `0.20` | Sum across all open positions. |
| `MAX_DAILY_LOSS_SOL` | `0.20` | Once realized day-PnL drops below `-this`, no new entries until next day. |
| `MAX_OPEN_POSITIONS` | `5` | Hard cap on concurrent positions. |
| `DEFAULT_SLIPPAGE_BPS` | `150` | Slippage tolerance in basis points (paper sanity + live default). |
| `PRIORITY_FEE_LAMPORTS` | `50000` | Live priority fee. Increase for high-contention windows. |

## Live execution

These only matter when `ENABLE_AUTO_TRADING=true` and `--live` is passed.

| Var | Required | Notes |
|---|---|---|
| `ENABLE_AUTO_TRADING` | live | Master switch. `true` to allow live. |
| `BOT_PRIVATE_KEY_B58` | live | Base58-encoded keypair. NEVER commit this. |
| `BOT_PUBLIC_KEY` | recommended | Cross-check vs derived pubkey at startup. |
| `LIVE_BROADCAST_MODE` | no | `helius_sender` (default) or `rpc`. |
| `HELIUS_SENDER_URL` | sender mode | Default `https://sender.helius-rpc.com/fast`. |
| `HELIUS_BUNDLE_URL` | bundle mode | For Jito Block Engine. |
| `JITO_TIP_LAMPORTS` | no | Default `200000`. |
| `JITO_TIP_ACCOUNTS` | no | Comma-separated Jito tip recipient list. |
| `LIVE_USE_DYNAMIC_PRIORITY_FEE` | no | Default `true`. Pull current p75 fee from RPC. |
| `LIVE_USE_DYNAMIC_JITO_TIP` | no | Default `true`. |
| `LIVE_USE_JUPITER_AUTO_SLIPPAGE` | no | Default `true`. Jupiter computes route-aware slippage. |
| `LIVE_REBROADCAST_INTERVAL_MS` | no | Default `250`. |
| `LIVE_MAX_REBROADCAST_ATTEMPTS` | no | Default `8`. |
| `LIVE_MIN_WALLET_BUFFER_LAMPORTS` | no | Refuse to trade if SOL balance after the buy would drop below this. |

See [LIVE_TRADING.md](./LIVE_TRADING.md) for the full live-mode setup.

## Rug-protection entry guards (live)

| Var | Default | Notes |
|---|---|---|
| `LIVE_ENTRY_REQUIRE_FREEZE_AUTHORITY_NULL` | `true` | Reject mints with live freeze authority. Strongly recommended. |
| `LIVE_ENTRY_REQUIRE_MINT_AUTHORITY_NULL` | `false` | Reject mints with live mint authority. Off by default — many healthy mints keep it for burns. |
| `LIVE_ENTRY_MAX_TOP_HOLDER_PCT` | `0.10` | Top-1 non-pool holder share cap. |
| `LIVE_ENTRY_MAX_TOP5_HOLDER_PCT` | `0.25` | Top-5 non-pool holders summed. |
| `LIVE_ENTRY_HONEYPOT_SIM_ENABLED` | `false` | Pre-buy `simulateBundle(buy, sell)` honeypot check. Requires RPC support. |
| `LIVE_ENTRY_HONEYPOT_SIM_FRACTION_BPS` | `500` | Simulated sell size as a fraction of buy size in bps. |

## Entry quality gates (paper + live)

These reject candidates that look thin even if the rule fires. See
`src/strategy/feature_runtime.py` for how each is computed.

| Var | Notes |
|---|---|
| `MIN_TRIGGER_SOL` | Minimum SOL volume on the triggering tx. |
| `ENTRY_MIN_TOKEN_AGE_SEC` | Minimum token age. |
| `ENTRY_MIN_CLUSTER_30S` | Minimum distinct buyer count over the prior 30s. |
| `ENTRY_MIN_TX_COUNT_30S` | Minimum tx count over the prior 30s. |
| `ENTRY_MIN_VOLUME_SOL_30S` | Minimum volume over the prior 30s. |
| `ENTRY_MIN_AVG_TRADE_SOL_30S` | Minimum average trade size. |

There are lane-specific overrides (`ENTRY_LANE_SHOCK_*`, `ENTRY_LANE_RECOVERY_*`,
`ENTRY_OVEREXTENSION_*`) — see `.env.example` for the full list.

## ML qualifier

| Var | Default | Notes |
|---|---|---|
| `ML_MODE` | `shadow` | `off` / `shadow` (predict, log only) / `gate` (reject below threshold). |
| `ML_THRESHOLD_MAIN` | _unset_ | Probability threshold to pass in `gate` mode. |
| `ML_THRESHOLD_SNIPER` | _unset_ | Threshold for sniper-strategy candidates. |
| `ML_BOOTSTRAP_PATH` | _unset_ | Directory of CSVs to bootstrap from at startup. |
| `ML_BOOTSTRAP_GLOB` | `ml_bootstrap*.csv` | Glob within bootstrap path. |
| `ML_RETRAIN_EVERY` | _unset_ | Retrain after N closed trades. |

## Staged exits

Many knobs. See `.env.example` for `EXIT_TP*`, `STAGE0_*`, `STAGE1_*`,
`POST_TP*`, `PRE_TP1_*`, `PRICE_OUTLIER_*`, `DEAD_TOKEN_*`, `STALE_SWEEP_*`,
`ABSOLUTE_MAX_HOLD_SEC`. Defaults are conservative; tune in paper mode.

## Sniper strategy (optional, layered on top)

| Var | Default | Notes |
|---|---|---|
| `ENABLE_SNIPER_STRATEGY` | `false` | Toggle the sniper sub-strategy. |
| `SNIPER_USE_RUNTIME_RULES` | `false` | If true, sniper uses runtime rules; if false, requires explicit rule IDs. |
| `SNIPER_RULE_IDS` | empty | Comma-separated rule IDs. |
| `SNIPER_POSITION_SOL` | `0.15` | Sniper-specific position size. |
| (and many more `SNIPER_*` knobs) | | See `.env.example`. |

## Notifications

| Var | Default | Notes |
|---|---|---|
| `ENABLE_TELEGRAM` | `false` | Send alerts to Telegram. |
| `TELEGRAM_BOT_TOKEN` | _unset_ | From @BotFather. |
| `TELEGRAM_CHAT_ID` | _unset_ | Where to send. |

## Dashboard

| Var | Default | Notes |
|---|---|---|
| `DASHBOARD_HOST` | `127.0.0.1` | |
| `DASHBOARD_PORT` | `8787` | |
| `DASHBOARD_REFRESH_SEC` | `5` | |
