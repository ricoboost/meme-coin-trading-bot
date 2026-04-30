"""Fallback polling-based tracked-wallet activity monitor."""

from __future__ import annotations

import logging
import os
from typing import Iterable

import pandas as pd

from src.bot.config import BotConfig
from src.bot.models import CandidateEvent
from src.clients.helius_client import HeliusClient
from src.monitoring.parsing import trade_from_tx
from src.utils.io import read_parquet

_POOL_LOGGER = logging.getLogger("wallet_pool_filter")


def _drop_never_ranked_bagholders(df: pd.DataFrame) -> pd.DataFrame:
    """Drop wallets that look like accumulators, not snipers.

    Criteria (all must be true):
      - never appeared on any provider_a leaderboard (daily/weekly/monthly = 0)
      - currently sitting on ≥10 active positions (7d)
      - closed ≤2 trades in 30d (near-zero realized turnover)

    Rationale: the wallet lane's cluster signal is meant to capture fresh-pump
    snipers. Helius/PROVIDER_B-discovered bagholders dilute that signal by repeatedly
    buying into older tokens they already hold, making it look like a wallet
    touched a token while contributing no launch-timing edge.
    """
    required = {
        "appears_daily",
        "appears_weekly",
        "appears_monthly",
        "active_trades_7d",
        "n_closed_30d",
    }
    if not required.issubset(df.columns):
        return df
    ranked = (
        df["appears_daily"].fillna(0).astype(float)
        + df["appears_weekly"].fillna(0).astype(float)
        + df["appears_monthly"].fillna(0).astype(float)
    ) > 0
    is_bag = (df["active_trades_7d"].fillna(0).astype(float) >= 10) & (
        df["n_closed_30d"].fillna(0).astype(float) <= 2
    )
    drop_mask = (~ranked) & is_bag
    dropped = int(drop_mask.sum())
    if dropped:
        _POOL_LOGGER.info(
            "wallet_pool filter: dropped %d never-ranked bagholders (of %d)",
            dropped,
            len(df),
        )
    return df.loc[~drop_mask].reset_index(drop=True)


class WalletActivityStream:
    """Poll recent tracked-wallet history and emit unseen events."""

    def __init__(self, config: BotConfig, limit_wallets: int | None = None) -> None:
        self.config = config
        self.wallet_df = self._load_wallet_df(limit_wallets=limit_wallets)
        self.wallet_scores = {
            row["wallet"]: float(row.get("score", 0.0))
            for row in self.wallet_df.to_dict(orient="records")
            if row.get("wallet")
        }
        self.client = HeliusClient(
            api_key=config.helius_api_key,
            base_url=config.helius_base_url,
            timeout_sec=30,
            page_size=20,
            max_pages=1,
        )

    def _load_wallet_df(self, limit_wallets: int | None = None) -> pd.DataFrame:
        """Load tracked wallet pool.

        Historically pair-first mode skipped this load because wallets weren't
        used as a signal. The wallet lane re-introduces that need: even in
        pair-first discovery, helius_ws cross-references tx signers against
        this set to populate `tracked_wallets` on events. Load the pool when
        either the wallet lane or tracked-wallet features are enabled.
        """
        needs_pool = (
            self.config.discovery_mode != "pair_first"
            or bool(getattr(self.config, "enable_wallet_strategy", False))
            or bool(getattr(self.config, "tracked_wallet_features_enabled", False))
        )
        if not needs_pool:
            return pd.DataFrame(columns=["wallet", "score", "best_rank"])

        if not self.config.tracked_wallets_path.exists():
            raise ValueError(f"Tracked wallets file missing: {self.config.tracked_wallets_path}")

        wallet_df = read_parquet(self.config.tracked_wallets_path)
        if wallet_df.empty:
            raise ValueError(f"Tracked wallets file is empty: {self.config.tracked_wallets_path}")

        if os.getenv("WALLET_POOL_EXCLUDE_BAGHOLDERS", "true").lower() in (
            "1",
            "true",
            "yes",
        ):
            wallet_df = _drop_never_ranked_bagholders(wallet_df)
            if wallet_df.empty:
                raise ValueError(
                    "wallet_pool is empty after bagholder filter — disable "
                    "WALLET_POOL_EXCLUDE_BAGHOLDERS or re-scrape the pool"
                )

        sort_cols = [column for column in ("score", "best_rank") if column in wallet_df.columns]
        if sort_cols:
            ascending = [False if column == "score" else True for column in sort_cols]
            wallet_df = wallet_df.sort_values(sort_cols, ascending=ascending)
        if limit_wallets:
            wallet_df = wallet_df.head(limit_wallets)
        return wallet_df

    def iter_wallets(self) -> Iterable[str]:
        return self.wallet_df["wallet"].astype(str).tolist()

    def poll(self, last_seen_signatures: dict[str, str]) -> list[CandidateEvent]:
        """Fetch newest transactions and return unseen token events."""
        events: list[CandidateEvent] = []
        for wallet in self.iter_wallets():
            rows = self.client._get(f"/v0/addresses/{wallet}/transactions", params={"limit": 20})
            if not rows:
                continue
            wallet_events: list[CandidateEvent] = []
            seen_signature = last_seen_signatures.get(wallet)
            for tx in rows:
                if seen_signature and tx.get("signature") == seen_signature:
                    break
                event = trade_from_tx(wallet, tx)
                if event is not None and event.token_mint:
                    wallet_events.append(event)
            if rows and rows[0].get("signature"):
                last_seen_signatures[wallet] = rows[0]["signature"]
            events.extend(reversed(wallet_events))
        events.sort(key=lambda event: event.block_time)
        return events
