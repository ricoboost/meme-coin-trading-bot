# Live trading

> **Read [DISCLAIMER.md](../DISCLAIMER.md) first.** Live mode trades real
> SOL on Solana mainnet. The author lost money running this. There is no
> testnet for live execution.

## When to flip to live

Don't, probably.

If you're going to anyway, the prerequisites are:

1. You've run paper mode for **at least a week** on the same wallet pool
   you'll trade with. You've watched the dashboard. You've seen what the
   bot decides, what it rejects, and what it would have made/lost.
2. You've read every line of `src/strategy/`, `src/portfolio/`, and
   `src/execution/`. Not just skimmed — actually read.
3. You can articulate, in two sentences, what edge you think you have
   over the dozens of other bots running similar strategies on the same
   data.
4. You have a kill switch ready (a separate terminal on the same machine,
   ready to SIGINT the bot) and you know what your portfolio looks like
   if the bot dies mid-trade.
5. You've decided on a position size where, if you lose all of it, you
   shrug and don't care.

If any of those is "no" or "kind of", stay in paper mode.

## Setup

### 1. Master switch

```bash
ENABLE_AUTO_TRADING=true
```

Without this, `--live` is silently downgraded to paper. This is
intentional — it means you can't accidentally enable live mode by
forgetting a flag.

### 2. Signing key

```bash
BOT_PRIVATE_KEY_B58=<base58-encoded-keypair>
BOT_PUBLIC_KEY=<derived-pubkey-for-cross-check>
```

The private key is used **only** by `src/execution/signer.py`
(`LocalSigner`). It is never logged, never transmitted to any external
service, and lives in process memory only.

`BOT_PUBLIC_KEY` is optional but recommended: at startup, the signer
derives the pubkey from the private key and refuses to start if it
doesn't match. This catches the common copy-paste error of pasting the
wrong keypair.

**Generating a keypair:**

```bash
solana-keygen new -o trader-bot-key.json
# Convert to base58 — solana-keygen outputs a JSON array, the bot needs
# base58 of the 64-byte secret-key bytes.
python -c "import base58, json; k=json.load(open('trader-bot-key.json')); print(base58.b58encode(bytes(k)).decode())"
```

Fund the resulting pubkey with the SOL you're willing to risk. **Do not**
use a wallet that holds anything else. This bot will spend that wallet's
SOL.

### 3. RPC + Jupiter

```bash
HELIUS_RPC_URL=https://mainnet.helius-rpc.com/?api-key=<your-key>
JUPITER_BASE_URL=https://api.jup.ag/swap/v2
```

The free Helius tier has rate limits that will sometimes drop your live
broadcasts. Consider paying for a tier with reliable `sendTransaction`
throughput and headroom for the rebroadcast loop.

### 4. Risk caps

Set conservatively. **Lower than you think.**

```bash
MAX_POSITION_SOL=0.05         # per-trade
MAX_TOTAL_EXPOSURE_SOL=0.20   # all open positions summed
MAX_DAILY_LOSS_SOL=0.20       # daily realized-PnL floor
MAX_OPEN_POSITIONS=5
DEFAULT_SLIPPAGE_BPS=150
PRIORITY_FEE_LAMPORTS=50000
```

These are enforced **before signing**. A trade that would breach any of
them is rejected and logged.

### 5. Rug-protection guards

```bash
LIVE_ENTRY_REQUIRE_FREEZE_AUTHORITY_NULL=true   # strongly recommended
LIVE_ENTRY_REQUIRE_MINT_AUTHORITY_NULL=false    # default off, see CONFIG.md
LIVE_ENTRY_MAX_TOP_HOLDER_PCT=0.10
LIVE_ENTRY_MAX_TOP5_HOLDER_PCT=0.25
LIVE_ENTRY_HONEYPOT_SIM_ENABLED=false           # opt-in, requires simulateBundle support
```

The freeze-authority check is the cheap one that catches the most rugs.
Top-holder caps catch dev-allocation rugs. Honeypot sim catches the
trickier "you can buy but can't sell" mints — but it requires an RPC
that supports `simulateBundle` (Helius, Jito Block Engine).

## Run

```bash
python -m src.bot.main --live
```

You'll see:

```
[INFO] Bot running in LIVE mode
[INFO] Signer validated — derived pubkey matches BOT_PUBLIC_KEY
[INFO] Risk caps: max_pos=0.05 max_exp=0.20 max_open=5 daily_loss=0.20
```

Live trades are logged with a `🔴 LIVE` prefix and persisted with
`mode=live` in SQLite.

## Safeguards

| Check | Behavior |
|---|---|
| `ENABLE_AUTO_TRADING != true` | `--live` silently downgrades to paper. |
| `BOT_PRIVATE_KEY_B58` missing | Live refuses to start, falls back to paper. |
| `HELIUS_RPC_URL` missing | Same. |
| `JUPITER_BASE_URL` missing | Same. |
| Signer validation fails | Same. |
| `BOT_PUBLIC_KEY` mismatch | Same. |
| Position would exceed `MAX_POSITION_SOL` | Trade rejected. |
| Total exposure would exceed `MAX_TOTAL_EXPOSURE_SOL` | Trade rejected. |
| Open positions at `MAX_OPEN_POSITIONS` | Trade rejected. |
| Net realized day-PnL below `-MAX_DAILY_LOSS_SOL` | Trade rejected. |
| Wallet SOL balance after buy would dip below `LIVE_MIN_WALLET_BUFFER_LAMPORTS` | Trade rejected. |
| Jupiter quote fails | Trade rejected. |
| Tx signing fails | Trade rejected. |
| Broadcast/confirmation fails | Trade recorded as `FAILED`, position state stays consistent. |
| Rug check fails | Trade rejected (or sized down, depending on which check). |

## Kill switch

`Ctrl+C` sends SIGINT. The bot catches it and:

1. Stops accepting new candidates.
2. Lets in-flight broadcasts finish.
3. Persists final state.
4. Exits.

It does **NOT** automatically close open positions on shutdown. You'll
restart the bot to manage them, or close them manually with whatever
wallet you fund.

If the bot crashes hard (segfault, OOM, panic), open positions stay
open on-chain. They'll still be in your SQLite store and the dashboard
when you restart. Have a plan for what you do in that case.

## Operational notes

- **Don't run unattended.** The exit logic is complex and partially
  reactive to on-chain state. If your machine loses network for 10
  minutes during a pump, the bot may not see the moment to exit.
- **Don't run concurrently.** Two instances against the same wallet will
  step on each other's transactions and your priority-fee budget.
- **Monitor the dashboard.** It shows the recent rejected trades and
  why — that's where you'll see config issues before they cost you.
- **Watch your SOL balance.** Live mode pays priority fees and Jito tips
  for every entry and exit. Even when you're not winning, you're paying
  fees.

## Re-read

- [DISCLAIMER.md](../DISCLAIMER.md) — yes, again.
- [STRATEGY.md](./STRATEGY.md) — section "Why it didn't work live."
- [ARCHITECTURE.md](./ARCHITECTURE.md) — "Why no testnet for live mode."
