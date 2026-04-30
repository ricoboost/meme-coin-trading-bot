"""Background-refreshed Jupiter sell quotes for open positions.

Pre-fetches exit quotes every REFRESH_INTERVAL_SEC so that sniper TP/stop
verification can use a cached result instead of blocking the event loop on
a live Jupiter HTTP call. Cache entries expire after TTL_SEC; on a miss the
caller falls back to a fresh quote.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.execution.trade_executor import TradeExecutor

logger = logging.getLogger(__name__)

# Default TTL for cached quotes. A quote older than this is treated as a miss.
_DEFAULT_TTL_SEC: float = 5.0
# How often the background thread refreshes quotes for all open positions.
_DEFAULT_REFRESH_INTERVAL_SEC: float = 2.0


class PositionQuoteCache:
    """Thread-safe LRU-style cache of Jupiter sell quotes for open positions.

    Usage:
        cache = PositionQuoteCache(trade_executor)
        cache.start()
        # on position open:
        cache.register(token_mint, token_amount_raw)
        # in exit logic:
        est = cache.get(token_mint)
        if est is None:
            est = trade_executor.simulate_paper_sell(...)  # fallback
        # on position close:
        cache.unregister(token_mint)
        # on bot shutdown:
        cache.stop()
    """

    def __init__(
        self,
        trade_executor: TradeExecutor,
        ttl_sec: float = _DEFAULT_TTL_SEC,
        refresh_interval_sec: float = _DEFAULT_REFRESH_INTERVAL_SEC,
    ) -> None:
        self._executor = trade_executor
        self._ttl_sec = ttl_sec
        self._refresh_interval_sec = refresh_interval_sec
        # token_mint → (monotonic_timestamp, PaperTradeEstimate)
        self._cache: dict[str, tuple[float, Any]] = {}
        # token_mint → raw token amount to use for sell quotes
        self._positions: dict[str, int] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, token_mint: str, token_amount_raw: int) -> None:
        """Register an open position so its sell quote is kept warm."""
        if token_amount_raw <= 0:
            return
        with self._lock:
            self._positions[token_mint] = token_amount_raw
        logger.debug("quote_cache: registered %s (amount=%d)", token_mint[:8], token_amount_raw)

    def update_token_amount(self, token_mint: str, new_amount: int) -> None:
        """Update the token amount after a partial sell and invalidate cached quote."""
        with self._lock:
            if token_mint not in self._positions:
                return
            self._positions[token_mint] = new_amount
            self._cache.pop(token_mint, None)

    def unregister(self, token_mint: str) -> None:
        """Remove a closed position from the cache."""
        with self._lock:
            self._positions.pop(token_mint, None)
            self._cache.pop(token_mint, None)
        logger.debug("quote_cache: unregistered %s", token_mint[:8])

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, token_mint: str) -> Any | None:
        """Return the cached ``PaperTradeEstimate`` if fresh, else ``None``."""
        with self._lock:
            entry = self._cache.get(token_mint)
        if entry is None:
            self._misses += 1
            return None
        ts, result = entry
        if time.monotonic() - ts > self._ttl_sec:
            self._misses += 1
            return None
        self._hits += 1
        return result

    # ------------------------------------------------------------------
    # Background refresh
    # ------------------------------------------------------------------

    def _fetch_one(self, mint: str, amount: int) -> tuple[str, Any]:
        """Fetch a single sell quote; returns (mint, result) for the caller."""
        return mint, self._executor.simulate_paper_sell(token_mint=mint, token_amount=amount)

    def _refresh_loop(self) -> None:
        """Background thread: refresh sell quotes for all registered positions."""
        logger.info(
            "quote_cache: refresh thread started (interval=%.1fs, ttl=%.1fs)",
            self._refresh_interval_sec,
            self._ttl_sec,
        )
        while not self._stop_event.wait(self._refresh_interval_sec):
            with self._lock:
                items = list(self._positions.items())

            # Parallel refresh: fetch positions in small batches to avoid rate-limiting.
            # Max 2 concurrent workers to stay under Jupiter API rate limits; a 0.15s
            # inter-batch sleep spreads the load further. With 14 positions this produces
            # ~7 batches × 0.15s = ~1.05s of spread across the 2s interval.
            active_items = [(m, a) for m, a in items if a > 0]
            if not active_items:
                continue

            _BATCH_SIZE = 2
            for i in range(0, len(active_items), _BATCH_SIZE):
                if self._stop_event.is_set():
                    break
                batch = active_items[i : i + _BATCH_SIZE]
                with ThreadPoolExecutor(max_workers=_BATCH_SIZE) as pool:
                    futures = {
                        pool.submit(self._fetch_one, mint, amount): mint
                        for mint, amount in batch
                        if not self._stop_event.is_set()
                    }
                    try:
                        # Timeout must exceed Jupiter's own HTTP timeout (10s) so we
                        # don't kill the iterator before the future resolves.
                        for fut in as_completed(futures, timeout=12.0):
                            if self._stop_event.is_set():
                                break
                            try:
                                mint, result = fut.result()
                                with self._lock:
                                    if mint in self._positions:
                                        self._cache[mint] = (time.monotonic(), result)
                            except Exception as exc:  # noqa: BLE001
                                mint = futures[fut]
                                logger.debug(
                                    "quote_cache: refresh failed for %s: %s",
                                    mint[:8],
                                    exc,
                                )
                    except TimeoutError:
                        # One or more futures exceeded the timeout — skip this batch,
                        # the thread survives and retries on the next refresh cycle.
                        logger.debug(
                            "quote_cache: batch timed out, skipping %d pending futures",
                            len(futures),
                        )
                if i + _BATCH_SIZE < len(active_items) and not self._stop_event.is_set():
                    time.sleep(0.15)

        logger.info(
            "quote_cache: refresh thread stopped (hits=%d misses=%d)",
            self._hits,
            self._misses,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background refresh thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._refresh_loop,
            name="quote-cache-refresh",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background refresh thread and clear state."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        with self._lock:
            self._cache.clear()
            self._positions.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "open_positions": len(self._positions),
                "cached_quotes": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (
                    self._hits / (self._hits + self._misses)
                    if (self._hits + self._misses) > 0
                    else 0.0
                ),
            }
