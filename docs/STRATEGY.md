# Strategy

What this bot is looking for, and what it isn't. Read this before tuning
configuration knobs — it explains what the knobs are gating against.

## The premise

Solana memecoin buys are noisy. The vast majority of trades are bots
front-running each other, wash trading, and noise. The premise of this
bot is that **clusters of independent "good" wallets converging on the
same token within a short window** carry information that random buys
don't.

So the pipeline:

1. Maintain a pool of wallets that have a track record (you bring this).
2. Watch all relevant programs in real time.
3. When N pool wallets buy the same token within a window, treat it as
   a candidate.
4. Apply rules + rug checks + risk gates before sizing a position.

This worked in backtest. It did not work live. Why is the second half of
this doc.

## The signal

### Cluster detection

In `pair_first` mode (the default), the bot subscribes to Pump.fun /
Raydium / Pump AMM / LaunchLab program activity via Yellowstone gRPC,
parses every swap, and maintains a per-token rolling window of buy events.

A candidate is created when:

- ≥ `ENTRY_MIN_CLUSTER_30S` distinct buyers transacted on the token in the
  prior 30 seconds, AND
- ≥ `ENTRY_MIN_TX_COUNT_30S` total swap transactions, AND
- ≥ `ENTRY_MIN_VOLUME_SOL_30S` SOL volume.

Tracked-wallet (pool) buys add a multiplicative bonus to the candidate's
runtime score (`TRACKED_WALLET_SIZE_BOOST_PER_WALLET`, capped by
`TRACKED_WALLET_SIZE_BOOST_CAP`). They are NOT the primary trigger in
pair-first mode — that's a deliberate change from the older
`wallet_first` mode, which over-fired on lone pool-wallet buys.

### Rule matching

Every candidate is fed through a rule set produced offline (the rule
mining pipeline that generated these is not included in this
distribution; the runtime consumes the resulting CSV at
`outputs/rules/pump_rule_packs_v2.csv`). Rules are simple feature
thresholds — token age, 30s volume, cluster size, price-impact bands,
etc. — plus a regime tag and a historical hit-rate.

When multiple rules match a single feature snapshot, selection prefers
(in order): rules in the detected regime, lower rug rate, higher 5x hit
rate, higher support, configured priority.

### Regime detection

The bot tags the current market state into one of:

- `negative_shock_recovery`
- `high_cluster_recovery`
- `momentum_burst`
- `unknown`

Regime is used to pick rules and to scale position size
(`REGIME_SIZE_MULTIPLIER_*`).

### ML qualifier

After the rule fires, an optional logistic classifier predicts whether
this candidate is likely to close positive. Three modes:

- `off` — disabled.
- `shadow` — predict, log probability, don't gate. Good for collecting
  data without affecting decisions.
- `gate` — reject candidates with probability below `ML_THRESHOLD_MAIN`.

The model retrains every `ML_RETRAIN_EVERY` closed trades using the
in-process feature snapshots. Bootstrap CSVs (under `ML_BOOTSTRAP_PATH`)
seed it before live data is available.

### Rug check

Before any trade is sized, candidates pass through `RugChecker`:

- Mint authority null check.
- Freeze authority null check.
- Top-1 / top-5 non-pool holder concentration.
- Jupiter quote sanity (route exists, price impact reasonable).
- (Live only, opt-in) `simulateBundle(buy, sell)` honeypot detection.

Failed rug checks reject the trade outright. A "passed but with caveats"
result reduces the position size.

## Position management

### Entry sizing

Base position is `MAX_POSITION_SOL`, scaled by:

- Regime multiplier (`REGIME_SIZE_MULTIPLIER_*`).
- Tracked-wallet boost (per pool-wallet involved, capped).
- Relaxed-rule penalty (smaller size if a runtime-relaxed rule fired).

Then clamped against `MAX_POSITION_SOL`, current open exposure
(`MAX_TOTAL_EXPOSURE_SOL`), and daily-loss cap (`MAX_DAILY_LOSS_SOL`).

### Staged exits

The exit engine runs multiple parallel rules per position:

- TP1, TP2, TP3 at multiples of entry (`EXIT_TP1_MULTIPLE`, etc.) with
  configurable sell fractions.
- Post-TP1 stop-loss tightening (`POST_TP1_STOP_PNL`).
- Post-TP2 trailing drawdown (`POST_TP2_TRAILING_DRAWDOWN`).
- Stage-0 fast fail conditions for positions that don't move in the first
  N seconds.
- Dead-token timeout for positions where on-chain activity stalls.
- Pre-TP1 retrace lock — if the position pumps then retraces while still
  in profit, lock in a floor.
- Crash guard — if the position was in solid profit and then drops
  sharply, exit before retest.
- Absolute max hold (`ABSOLUTE_MAX_HOLD_SEC`) — close everything older
  than this regardless of state.

The complexity here exists because no single exit rule worked across
regimes. Each was added in response to a specific failure mode observed
in paper or live trading.

## Why it didn't work live

The author's experience running this strategy on Solana memecoin markets
in production. Documenting it here so you don't have to discover it the
same way.

1. **Signal degradation at scale.** Wallet-cluster signals work better on
   sleepy markets than on hot ones. As copy-trading infrastructure
   proliferated on Solana, more bots were watching the same wallets,
   priority-fee-bidding to front-run them. By the time the bot saw and
   acted on a cluster, the wallets it was copying had often already
   exited.
2. **Slippage tax compounds.** Memecoin markets have wide bid-ask
   spreads on the smaller mints. Each round-trip costs you 4-8% in
   slippage. With sub-50% win rate, this eats every edge.
3. **Rug rate is higher than backtest.** Backtest data is survivor-biased
   (you see the tokens that were liquid enough to sell). Live, you'll
   buy mints whose LP gets pulled before you can exit, and they
   disappear from your dataset entirely. The actual rug rate at the
   margin is meaningfully higher than what historical data shows.
4. **Priority fee inflation eats edge.** During hot windows, getting
   your transaction included costs hundreds of thousands of lamports.
   On a 0.05 SOL position with a 10% target gain, a 0.0005 SOL priority
   fee is 10% of your gross gain. It compounds quickly.
5. **Exit conditions are harder than entry.** Knowing when to buy was
   the easy part. Knowing when to exit before the inevitable dump is
   what kills you. The 14+ exit conditions in `src/portfolio/` are scar
   tissue from this — but they're patches over a fundamentally hard
   problem.

If you're going to run this bot, **assume it will lose money.** Treat
any positive PnL as a temporary anomaly, not the new baseline.

## What might actually work

Since you're going to ignore the warning anyway:

- **Fewer trades, higher conviction.** Tighten every threshold. Aim for
  one trade a day, not one trade an hour.
- **Bigger size on the rare confident trades.** The rule-quality tail is
  fat. The top decile of rules historically beat the rest by 3-5x.
- **Faster exits.** Memecoin pumps are minutes, not hours. The default
  exit logic is too slow.
- **Watch a smaller wallet pool, more carefully.** 50 wallets you trust
  beats 500 wallets you don't.
- **Don't co-fire with the public bots.** If you can detect that a
  signal has already been broadcast publicly (your Telegram channel,
  someone else's), skip it — the move is already in.
- **Be ruthless about killing the bot when conditions change.** Memecoin
  market regime shifts are sudden. Stop running for a week and check
  again.
