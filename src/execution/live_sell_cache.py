"""Background-refreshed pre-built Jupiter sell TXs for open live positions.

Rebuilds a ready-to-sign Jupiter swap TX (token → SOL) for each registered
position on a staggered schedule. When an exit fires, the TX is already built —
execution collapses to sign (~1ms) + broadcast (~50-150ms).

Refresh strategy
----------------
Solana blockhash expires after ~150 blocks (~60 s). We refresh every
REFRESH_INTERVAL_SEC (default 30 s) — well within expiry, and at most
2 Jupiter calls per position per minute vs 60 with a 2 s interval.

On consecutive Jupiter errors the interval backs off up to MAX_BACKOFF_SEC
to avoid hammering a degraded API. On the next success the interval resets.

Only used in live mode. For paper trading this module is never instantiated.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.execution.trade_executor_live import LiveTradeExecutor, SwapTransaction

logger = logging.getLogger(__name__)

# Refresh every 12s — TX expires at 55s, so 12s gives 43s of guaranteed freshness.
# Reduced from 30s: a position registered just after a 30s refresh could wait 30s
# for its first TX build. For sniper trades (75-120s hold), TX could expire before
# ever being used. At 12s we get 3+ refreshes within the sniper hold window.
REFRESH_INTERVAL_SEC: float = 12.0
# Maximum backoff on consecutive errors (caps at 5 minutes)
MAX_BACKOFF_SEC: float = 300.0
# Treat cached TX as expired if older than this — force fresh Jupiter call at exit
TX_EXPIRY_SEC: float = 55.0
# After this many consecutive per-mint build failures, auto-unregister the mint.
# Keeps the cache from spinning forever on a dust/drained/dead position when the
# exit engine hasn't fired yet. ~5 refreshes ≈ 60s of continuous failure.
PER_MINT_FAIL_THRESHOLD: int = 5


class LiveSellCache:
    """Pre-builds and continuously refreshes Jupiter sell TXs for open positions.

    Usage
    -----
    1. Call ``register(token_mint, token_amount)`` when a position opens.
    2. Call ``get(token_mint)`` at exit time to retrieve the latest pre-built TX.
       Returns None if no TX cached or if the cached TX is expired (>55 s old).
    3. Call ``unregister(token_mint)`` after the position closes.

    Rate limiting
    -------------
    Refreshes on a 30 s interval so each position generates at most 2 Jupiter
    calls per minute (quote + swap).  Consecutive errors trigger exponential
    backoff up to MAX_BACKOFF_SEC to protect against a degraded Jupiter API.
    Positions are refreshed one at a time with a 1 s gap between them to
    avoid burst requests when many positions are open simultaneously.
    """

    def __init__(self, live_executor: "LiveTradeExecutor") -> None:
        self._executor = live_executor
        self._positions: dict[str, int] = {}  # mint → token_amount
        self._strategies: dict[str, str] = {}  # mint → strategy_id ("sniper"/"main")
        self._sources: dict[str, str] = {}  # mint → source_program ("PUMP_FUN", etc.)
        self._cache: dict[str, tuple[float, "SwapTransaction"]] = {}  # mint → (ts, tx)
        self._per_mint_fails: dict[str, int] = {}  # mint → consecutive build failures
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._refresh_loop, name="live-sell-cache", daemon=True
        )
        self._consecutive_errors: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread.start()
        logger.info(
            "LiveSellCache started (refresh=%.0fs expiry=%.0fs)",
            REFRESH_INTERVAL_SEC,
            TX_EXPIRY_SEC,
        )

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        token_mint: str,
        token_amount: int,
        strategy: str | None = None,
        source_program: str | None = None,
    ) -> None:
        """Register a new open position."""
        with self._lock:
            self._positions[token_mint] = token_amount
            if strategy:
                self._strategies[token_mint] = strategy
            if source_program:
                self._sources[token_mint] = source_program
            self._per_mint_fails.pop(token_mint, None)
        logger.debug(
            "LiveSellCache.register: %s amount=%d strategy=%s source=%s",
            token_mint[:8],
            token_amount,
            strategy or "default",
            source_program or "unknown",
        )

    def update_amount(self, token_mint: str, token_amount: int) -> None:
        """Update the sell amount after a partial exit. Invalidates cached TX."""
        with self._lock:
            if token_mint in self._positions:
                self._positions[token_mint] = token_amount
                self._cache.pop(token_mint, None)
                self._per_mint_fails.pop(token_mint, None)

    def unregister(self, token_mint: str) -> None:
        """Remove a closed position from the cache."""
        with self._lock:
            self._positions.pop(token_mint, None)
            self._strategies.pop(token_mint, None)
            self._sources.pop(token_mint, None)
            self._cache.pop(token_mint, None)
            self._per_mint_fails.pop(token_mint, None)
        logger.debug("LiveSellCache.unregister: %s", token_mint[:8])

    def get_strategy(self, token_mint: str) -> str | None:
        """Return the registered strategy id for a mint, or None if unknown."""
        with self._lock:
            return self._strategies.get(token_mint)

    def get_source_program(self, token_mint: str) -> str | None:
        """Return the source_program recorded at entry, or None if unknown."""
        with self._lock:
            return self._sources.get(token_mint)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, token_mint: str) -> "SwapTransaction | None":
        """Return the latest pre-built sell TX, or None if missing or expired."""
        with self._lock:
            entry = self._cache.get(token_mint)
        if entry is None:
            return None
        ts, tx = entry
        if time.monotonic() - ts > TX_EXPIRY_SEC:
            return None  # expired — caller falls back to fresh Jupiter
        return tx

    # ------------------------------------------------------------------
    # Background refresh loop
    # ------------------------------------------------------------------

    def _refresh_loop(self) -> None:
        # Small initial delay so the first refresh doesn't pile on with startup.
        self._stop.wait(5.0)

        while not self._stop.is_set():
            with self._lock:
                snapshot = dict(self._positions)

            error_this_cycle = False
            for mint, amount in snapshot.items():
                if self._stop.is_set():
                    break

                # Skip if the cached TX is still fresh enough
                with self._lock:
                    entry = self._cache.get(mint)
                if entry is not None:
                    age = time.monotonic() - entry[0]
                    if age < REFRESH_INTERVAL_SEC:
                        continue  # not due for refresh yet

                with self._lock:
                    mint_strategy = self._strategies.get(mint)
                    mint_source = self._sources.get(mint)
                prefer_jupiter = mint_strategy == "main"
                try:
                    tx = self._executor.prefetch_sell_tx(
                        token_mint=mint,
                        token_amount=amount,
                        prefer_jupiter=prefer_jupiter,
                        strategy=mint_strategy,
                        source_program=mint_source,
                    )
                    if tx is not None:
                        with self._lock:
                            if mint in self._positions:
                                self._cache[mint] = (time.monotonic(), tx)
                                self._per_mint_fails.pop(mint, None)
                        logger.debug(
                            "LiveSellCache refreshed %s last_valid=%d",
                            mint[:8],
                            tx.last_valid_block_height,
                        )
                    else:
                        # prefetch swallowed an exception and returned None —
                        # count it as a build failure for this mint.
                        with self._lock:
                            fails = self._per_mint_fails.get(mint, 0) + 1
                            self._per_mint_fails[mint] = fails
                        if fails >= PER_MINT_FAIL_THRESHOLD:
                            logger.warning(
                                "LiveSellCache auto-unregister %s after %d consecutive build failures",
                                mint[:8],
                                fails,
                            )
                            self.unregister(mint)
                        error_this_cycle = True
                except Exception as exc:  # noqa: BLE001
                    logger.debug("LiveSellCache refresh failed for %s: %s", mint[:8], exc)
                    with self._lock:
                        fails = self._per_mint_fails.get(mint, 0) + 1
                        self._per_mint_fails[mint] = fails
                    if fails >= PER_MINT_FAIL_THRESHOLD:
                        logger.warning(
                            "LiveSellCache auto-unregister %s after %d consecutive build failures",
                            mint[:8],
                            fails,
                        )
                        self.unregister(mint)
                    error_this_cycle = True

                # 1 s gap between mints to avoid bursting Jupiter
                if not self._stop.wait(1.0):
                    pass  # keep going

            # Adjust backoff based on error state
            if error_this_cycle:
                self._consecutive_errors += 1
                backoff = min(
                    REFRESH_INTERVAL_SEC * (2**self._consecutive_errors),
                    MAX_BACKOFF_SEC,
                )
                logger.warning(
                    "LiveSellCache: %d consecutive error(s), backing off %.0fs",
                    self._consecutive_errors,
                    backoff,
                )
                self._stop.wait(backoff)
            else:
                self._consecutive_errors = 0
                self._stop.wait(REFRESH_INTERVAL_SEC)
