# Disclaimer

**Read this before you do anything with this software.**

## You will probably lose money

This is a research artifact. The author ran a version of this bot live on
Solana mainnet and **lost money**. The strategy underperformed live trading
across multiple regimes and configurations. It is published for educational
value, not because it is profitable.

If you run this bot with real funds, the most likely outcome is that you
will also lose money — possibly all of it, possibly quickly.

## No warranty, no support, no liability

This software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose, and non-infringement. In no event shall
the authors or copyright holders be liable for any claim, damages, or
other liability, whether in an action of contract, tort, or otherwise,
arising from, out of, or in connection with the software or the use or
other dealings in the software.

This is a personal project. There is no support team. There is no SLA.
Issues may go unanswered. Pull requests may be ignored.

## On-chain transactions are irreversible

There is no undo for a Solana transaction. If the bot:

- Buys a rug-pull token, your SOL is gone.
- Hits a slippage spike at exit, you eat the spread.
- Pays a priority fee that doesn't get included, you lose the fee.
- Submits a malformed transaction, the network may still drop the fee.

There is no testnet for live execution in this codebase. "Live mode" trades
real SOL on Solana mainnet from the moment you enable it.

## Memecoin markets are dangerous

The strategies in this bot target Solana memecoins. Memecoin markets are:

- Highly volatile (entire-position drawdowns within seconds are routine).
- Routinely manipulated (rugs, honeypots, sniper bundles, wash trading).
- Often illiquid at exit (you may not be able to sell what you bought).
- Subject to bot warfare in a way regular DEX trading is not.

The rug-check guards in `src/strategy/rug_check.py` are best-effort. They
do not catch every honeypot, every freeze rug, or every dilution attack.

## Not financial advice

Nothing in this repository is financial advice. The author is not a
financial advisor. The strategies here are not endorsed, validated, or
audited. Do not interpret any code, comment, doc, or example as a
recommendation to trade.

## Your responsibility

If you choose to run this software:

1. Start in **paper mode** (the default). Spend at least a week watching
   what the bot does without spending real money.
2. Read every line of `src/strategy/`, `src/portfolio/`, and
   `src/execution/` before flipping `ENABLE_AUTO_TRADING=true`.
3. Set position sizes you can afford to lose. Then set them lower.
4. Do not run with funds borrowed, leveraged, or earmarked for anything
   else in your life.
5. Monitor the bot. Do not let it run unattended for long periods.
6. Have a kill switch ready (`Ctrl+C` works; closing positions cleanly is
   harder, plan for it).

## TL;DR

This bot lost money for the person who wrote it. It will probably lose
money for you. If you run it with real funds and it loses your money,
that is your responsibility, not theirs. There is no recourse.
