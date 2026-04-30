"""Market regime monitor — detects hot/normal/cold market conditions.

Tracks three independent signals:
  1. Rolling bot win rate (last N closed positions) — lags by 30-60 min but
     is the most reliable indicator of actual market quality.
  2. Candidate discovery rate (tokens seen per 5 min via gRPC stream) —
     real-time proxy for Pump.fun ecosystem activity level.
  3. SOL price 1h change (optional, from CoinGecko) — leading macro signal;
     SOL selling pressure kills memecoin momentum 30-60 min later.

When any enabled gate triggers, entries are paused until conditions recover.
Hysteresis prevents flip-flopping: pause threshold < resume threshold,
and a minimum cooldown enforces a floor on pause duration.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    """Snapshot of current market regime."""

    score: float  # 0.0 (dead) to 1.0 (hot), composite
    label: str  # "hot" / "normal" / "cold" / "dead"
    favorable: bool  # True = trade, False = pause entries
    win_rate: float | None  # rolling win rate (None if not bootstrapped)
    n_positions: int  # positions in win rate window
    candidate_rate_5min: int  # candidates seen in last 5 minutes
    sol_change_1h: float | None  # 1h SOL/USD price change (None if disabled)
    pause_reason: str | None  # which signal triggered current pause


class MarketRegimeMonitor:
    """
    Monitors market conditions and gates entry execution.

    Parameters
    ----------
    enabled:
        If False, is_favorable() always returns True (monitor is off).
    win_rate_window:
        Number of recent closed positions used for rolling win rate.
    min_win_rate:
        Win rate below this triggers a pause (e.g. 0.25 = 25%).
    bootstrap_positions:
        Number of closed positions required before win rate gating is active.
    min_candidates_5min:
        Minimum candidates expected in any 5-minute window.
        Set to 0 (default) to disable candidate rate gating.
    sol_enabled:
        If True, polls CoinGecko for SOL/USD price every 10 minutes.
    sol_drop_threshold:
        SOL 1h change below this triggers a pause (e.g. -0.05 = -5%).
    pause_cooldown_sec:
        Minimum pause duration in seconds before re-evaluating resume.
    event_log:
        Optional EventLogger for market_regime_pause / resume events.
    """

    def __init__(
        self,
        enabled: bool = True,
        win_rate_window: int = 15,
        min_win_rate: float = 0.25,
        bootstrap_positions: int = 5,
        min_candidates_5min: int = 0,
        sol_enabled: bool = False,
        sol_drop_threshold: float = -0.05,
        pause_cooldown_sec: int = 300,
        event_log: Any = None,
    ) -> None:
        self._enabled = enabled
        self._win_rate_window = win_rate_window
        self._min_win_rate = min_win_rate
        self._bootstrap_positions = bootstrap_positions
        self._min_candidates_5min = min_candidates_5min
        self._sol_enabled = sol_enabled
        self._sol_drop_threshold = sol_drop_threshold
        self._pause_cooldown_sec = pause_cooldown_sec
        self._event_log = event_log

        # Rolling outcome deque (stores pnl_sol per closed position)
        self._outcome_window: deque[float] = deque(maxlen=win_rate_window)
        self._bootstrapped: bool = False

        # Unique token mints seen in the rolling 5-min window.
        # Maps token_mint → first-seen datetime; entries expire after 5 min.
        # Using unique mints (not raw event count) gives a meaningful "new launches
        # per 5 min" signal that distinguishes hot from cold markets.
        self._candidate_mint_window: dict[str, datetime] = {}
        self._started_at: datetime | None = None

        # SOL price history (maxlen=12 → 2h at 10-min intervals)
        self._sol_prices: deque[tuple[datetime, float]] = deque(maxlen=12)

        # Pause state
        self._paused_since: datetime | None = None
        self._pause_reason: str | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def record_candidate_seen(self, token_mint: str, now: datetime) -> None:
        """Record a unique token seen in the gRPC stream.

        Counts distinct token mints in the last 5 minutes — a real-time proxy
        for new-launch activity. A single token generates many BUY/SELL events;
        counting unique mints gives a meaningful hot/cold market signal.
        """
        if self._started_at is None:
            self._started_at = now
        cutoff = now - timedelta(seconds=300)
        # Prune expired mints from the window
        expired = [m for m, ts in self._candidate_mint_window.items() if ts < cutoff]
        for m in expired:
            del self._candidate_mint_window[m]
        # Record first occurrence of this mint in the current window
        if token_mint not in self._candidate_mint_window:
            self._candidate_mint_window[token_mint] = now

    def record_position_closed(self, pnl_sol: float) -> None:
        """Call whenever any main-strategy position closes with its realized PnL in SOL."""
        self._outcome_window.append(pnl_sol)
        if not self._bootstrapped and len(self._outcome_window) >= self._bootstrap_positions:
            self._bootstrapped = True
            wins = sum(1 for p in self._outcome_window if p > 0)
            win_rate = wins / len(self._outcome_window)
            logger.info(
                "market_regime: bootstrap complete (%d positions, win_rate=%.1f%%), win-rate gating now active",
                len(self._outcome_window),
                win_rate * 100,
            )

    def update_sol_price(self, price: float, ts: datetime) -> None:
        """Update SOL/USD price reading. Called by background polling task."""
        self._sol_prices.append((ts, price))
        change = self._get_sol_change_1h()
        logger.debug(
            "market_regime: SOL price=%.2f, 1h_change=%s",
            price,
            f"{change:.3%}" if change is not None else "N/A",
        )

    def is_favorable(self) -> bool:
        """Returns True if market conditions support new entries, False if paused."""
        if not self._enabled:
            return True
        now = datetime.now(tz=timezone.utc)

        if self._paused_since is not None:
            # In pause: enforce minimum cooldown
            elapsed = (now - self._paused_since).total_seconds()
            if elapsed < self._pause_cooldown_sec:
                return False
            # Cooldown elapsed: check if conditions have recovered (uses resume thresholds)
            still_bad, _ = self._check_triggers(now, resuming=True)
            if still_bad:
                return False
            # Conditions recovered — resume
            self._log_resume(now)
            self._paused_since = None
            self._pause_reason = None
            return True

        # Not paused: check if we should pause (uses pause thresholds)
        should_pause, reason = self._check_triggers(now, resuming=False)
        if should_pause:
            self._paused_since = now
            self._pause_reason = reason
            self._log_pause(reason, now)
            return False

        return True

    def get_state(self) -> RegimeState:
        """Return a full snapshot of current regime state (read-only, no side effects)."""
        now = datetime.now(tz=timezone.utc)
        win_rate, n_pos = self._current_win_rate()
        candidate_rate = len(self._candidate_mint_window)
        sol_change = self._get_sol_change_1h() if self._sol_enabled else None
        score = self._compute_score(win_rate, candidate_rate, sol_change)
        # Compute favorable without triggering transitions (is_favorable() manages those)
        if not self._enabled:
            favorable = True
        elif self._paused_since is not None:
            favorable = False
        else:
            _, bad_reason = self._check_triggers(now, resuming=False)
            favorable = bad_reason is None

        if score >= 0.65:
            label = "hot"
        elif score >= 0.45:
            label = "normal"
        elif score >= 0.25:
            label = "cold"
        else:
            label = "dead"

        return RegimeState(
            score=round(score, 4),
            label=label,
            favorable=favorable,
            win_rate=round(win_rate, 4) if win_rate is not None else None,
            n_positions=n_pos,
            candidate_rate_5min=candidate_rate,
            sol_change_1h=round(sol_change, 4) if sol_change is not None else None,
            pause_reason=self._pause_reason,
        )

    async def start_sol_price_polling(self) -> None:
        """Background task: polls CoinGecko for SOL/USD price every 10 minutes."""
        import httpx

        while True:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(
                        "https://api.coingecko.com/api/v3/simple/price",
                        params={"ids": "solana", "vs_currencies": "usd"},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    price = float(data["solana"]["usd"])
                    self.update_sol_price(price, datetime.now(tz=timezone.utc))
            except Exception as exc:
                logger.debug("market_regime: SOL price poll failed: %s", exc)
            await asyncio.sleep(600)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _current_win_rate(self) -> tuple[float | None, int]:
        n = len(self._outcome_window)
        if n == 0 or not self._bootstrapped:
            return None, n
        wins = sum(1 for p in self._outcome_window if p > 0)
        return wins / n, n

    def _get_sol_change_1h(self) -> float | None:
        """Compute 1h SOL price change from buffered readings."""
        if len(self._sol_prices) < 2:
            return None
        now_ts, now_price = self._sol_prices[-1]
        target_ts = now_ts - timedelta(hours=1)
        best_ts: datetime | None = None
        best_price: float | None = None
        for ts, price in self._sol_prices:
            if ts >= now_ts:
                continue
            if best_ts is None or abs((ts - target_ts).total_seconds()) < abs(
                (best_ts - target_ts).total_seconds()
            ):
                best_ts, best_price = ts, price
        if best_price is None or best_price <= 0:
            return None
        age_min = (now_ts - best_ts).total_seconds() / 60  # type: ignore[operator]
        if age_min < 30:
            return None  # not enough history yet (< 30 min of readings)
        return (now_price - best_price) / best_price

    def _check_triggers(self, now: datetime, *, resuming: bool) -> tuple[bool, str | None]:
        """
        Check all active gates.
        resuming=True uses tighter (resume) thresholds to enforce hysteresis.
        Returns (should_pause, reason_string).
        """
        # Win rate gate
        win_rate, n_pos = self._current_win_rate()
        if win_rate is not None:
            threshold = (self._min_win_rate + 0.10) if resuming else self._min_win_rate
            if win_rate < threshold:
                return True, f"win_rate={win_rate:.2f}<{threshold:.2f} (n={n_pos})"

        # Candidate rate gate (only after 5-min startup grace period)
        if self._min_candidates_5min > 0 and self._started_at is not None:
            elapsed = (now - self._started_at).total_seconds()
            if elapsed >= 300:
                rate = len(self._candidate_mint_window)
                # Resume requires 20% more than pause threshold (hysteresis)
                threshold = (
                    int(self._min_candidates_5min * 1.2) if resuming else self._min_candidates_5min
                )
                if rate < threshold:
                    return True, f"candidate_rate={rate}/5min<{threshold}"

        # SOL price gate
        if self._sol_enabled:
            sol_change = self._get_sol_change_1h()
            if sol_change is not None:
                # Resume requires 2pp above drop threshold (hysteresis)
                threshold = (
                    (self._sol_drop_threshold + 0.02) if resuming else self._sol_drop_threshold
                )
                if sol_change < threshold:
                    return True, f"sol_1h={sol_change:.3%}<{threshold:.3%}"

        return False, None

    def _compute_score(
        self,
        win_rate: float | None,
        candidate_rate: int,
        sol_change: float | None,
    ) -> float:
        """Composite score 0.0–1.0 for dashboard/logging."""
        # Win rate component (weight 0.5): 0.0 at 25% win rate, 1.0 at 70%
        if win_rate is not None:
            win_score = max(0.0, min(1.0, (win_rate - 0.25) / 0.45))
        else:
            win_score = 0.5  # neutral during bootstrap

        # Candidate rate component (weight 0.3): scales against 2× min threshold
        if self._min_candidates_5min > 0:
            baseline = self._min_candidates_5min * 2  # "hot" = 2× the floor
            cand_score = max(0.0, min(1.0, candidate_rate / max(baseline, 1)))
        else:
            cand_score = 0.5  # gate disabled, neutral

        # SOL component (weight 0.2): 0.0 at drop_threshold, 1.0 at +abs(threshold)
        if self._sol_enabled and sol_change is not None:
            spread = abs(self._sol_drop_threshold) * 2
            sol_score = max(0.0, min(1.0, (sol_change - self._sol_drop_threshold) / spread))
        else:
            sol_score = 0.5  # disabled/unavailable, neutral

        return win_score * 0.5 + cand_score * 0.3 + sol_score * 0.2

    def _log_pause(self, reason: str | None, now: datetime) -> None:
        msg = f"market_regime: PAUSED — {reason}"
        logger.warning(msg)
        if self._event_log is not None:
            try:
                state = self._snapshot_for_log(now)
                state["pause_reason"] = reason
                self._event_log.log("market_regime_pause", state)
            except Exception:
                pass

    def _log_resume(self, now: datetime) -> None:
        logger.info("market_regime: RESUMED — conditions recovered")
        if self._event_log is not None:
            try:
                state = self._snapshot_for_log(now)
                self._event_log.log("market_regime_resume", state)
            except Exception:
                pass

    def _snapshot_for_log(self, now: datetime) -> dict:
        win_rate, n_pos = self._current_win_rate()
        return {
            "win_rate": round(win_rate, 4) if win_rate is not None else None,
            "n_positions": n_pos,
            "candidate_rate_5min": len(self._candidate_mint_window),
            "sol_change_1h": round(self._get_sol_change_1h(), 4)
            if self._sol_enabled and self._get_sol_change_1h() is not None
            else None,
            "bootstrapped": self._bootstrapped,
        }
