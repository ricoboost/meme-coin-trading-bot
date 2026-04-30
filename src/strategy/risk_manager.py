"""Entry risk controls for paper trading."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

from src.bot.config import BotConfig
from src.storage.bot_db import BotDB


class RiskManager:
    """Deterministic risk controls."""

    def __init__(self, config: BotConfig, db: BotDB) -> None:
        self.config = config
        self.db = db
        self._lock = threading.RLock()
        self._daily_counter_date: str | None = None
        self._daily_row_cache: dict[str, float] = {
            "daily_loss_sol": 0.0,
            "net_realized_pnl_sol": 0.0,
            "session_loss_sol": 0.0,
        }
        self._cooldowns: dict[str, datetime] = {}
        # Session-scoped blacklist: any token we closed at a loss this session.
        # Never re-enter the same token after a losing trade — prevents bleeding
        # on dying tokens whose post-close 30-min cooldown has expired.
        self._burned_tokens: set[str] = set()
        self._load_cooldowns()
        self._load_daily_row(datetime.now(tz=timezone.utc).date().isoformat())

    @staticmethod
    def _parse_dt(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _load_cooldowns(self) -> None:
        rows = self.db.fetchall("SELECT token_mint, cooldown_until FROM token_cooldowns")
        now = datetime.now(tz=timezone.utc)
        cooldowns: dict[str, datetime] = {}
        for row in rows:
            token = str(row["token_mint"] or "")
            cooldown_until = self._parse_dt(row["cooldown_until"])
            if token and cooldown_until is not None and cooldown_until > now:
                cooldowns[token] = cooldown_until
        with self._lock:
            self._cooldowns = cooldowns

    def _load_daily_row(self, counter_date: str) -> None:
        row = self.db.fetchone(
            """
            SELECT daily_loss_sol, net_realized_pnl_sol, session_loss_sol
            FROM risk_counters
            WHERE counter_date = ?
            """,
            (counter_date,),
        )
        if row is None:
            daily_row = {
                "daily_loss_sol": 0.0,
                "net_realized_pnl_sol": 0.0,
                "session_loss_sol": 0.0,
            }
        else:
            daily_loss_sol = float(row["daily_loss_sol"] or 0.0)
            net_realized_pnl_sol = float(row["net_realized_pnl_sol"] or 0.0)
            session_loss_sol = float(row["session_loss_sol"] or 0.0)
            if (
                abs(net_realized_pnl_sol) <= 1e-12
                and abs(session_loss_sol) <= 1e-12
                and abs(daily_loss_sol) > 1e-12
            ):
                net_realized_pnl_sol = daily_loss_sol
                session_loss_sol = min(0.0, daily_loss_sol)
            daily_row = {
                "daily_loss_sol": daily_loss_sol,
                "net_realized_pnl_sol": net_realized_pnl_sol,
                "session_loss_sol": session_loss_sol,
            }
        with self._lock:
            self._daily_counter_date = str(counter_date)
            self._daily_row_cache = daily_row

    def _ensure_current_day_locked(self) -> str:
        today = datetime.now(tz=timezone.utc).date().isoformat()
        if self._daily_counter_date != today:
            self._load_daily_row(today)
        return today

    def _prune_cooldowns_locked(self, now: datetime) -> None:
        expired = [token for token, until in self._cooldowns.items() if until <= now]
        for token in expired:
            self._cooldowns.pop(token, None)

    def _daily_row(self) -> dict[str, float]:
        with self._lock:
            self._ensure_current_day_locked()
            return dict(self._daily_row_cache)

    def _effective_daily_loss(self) -> float:
        net_pnl = self._daily_row()["net_realized_pnl_sol"]
        return max(0.0, -net_pnl)

    def record_daily_loss(self, pnl_sol: float, realized_at: datetime | None = None) -> None:
        """Record realized leg PnL.

        Despite the legacy method name, this stores both:
        - net realized PnL for risk halting
        - cumulative session loss (negative-only) for dashboard visibility
        """
        if self.db.has_trade_legs():
            ts = realized_at or datetime.now(tz=timezone.utc)
            self.db.sync_risk_counter(ts.date().isoformat())
            self._load_daily_row(datetime.now(tz=timezone.utc).date().isoformat())
            return
        today = datetime.now(tz=timezone.utc).date().isoformat()
        self.db.execute(
            """
            INSERT INTO risk_counters (counter_date, daily_loss_sol, net_realized_pnl_sol, session_loss_sol)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(counter_date) DO UPDATE SET
                net_realized_pnl_sol = risk_counters.net_realized_pnl_sol + excluded.net_realized_pnl_sol,
                session_loss_sol = risk_counters.session_loss_sol + excluded.session_loss_sol,
                daily_loss_sol = CASE
                    WHEN (risk_counters.net_realized_pnl_sol + excluded.net_realized_pnl_sol) < 0
                    THEN (risk_counters.net_realized_pnl_sol + excluded.net_realized_pnl_sol)
                    ELSE 0
                END
            """,
            (today, min(0.0, pnl_sol), pnl_sol, min(0.0, pnl_sol)),
        )
        self._load_daily_row(today)

    def set_cooldown(self, token_mint: str, minutes: int, reason: str) -> None:
        cooldown_until_dt = datetime.now(tz=timezone.utc) + timedelta(minutes=minutes)
        cooldown_until = cooldown_until_dt.isoformat()
        self.db.execute(
            """
            INSERT INTO token_cooldowns (token_mint, cooldown_until, reason)
            VALUES (?, ?, ?)
            ON CONFLICT(token_mint) DO UPDATE SET cooldown_until = excluded.cooldown_until, reason = excluded.reason
            """,
            (token_mint, cooldown_until, reason),
        )
        with self._lock:
            self._cooldowns[str(token_mint or "")] = cooldown_until_dt

    def in_cooldown(self, token_mint: str) -> bool:
        now = datetime.now(tz=timezone.utc)
        with self._lock:
            self._prune_cooldowns_locked(now)
            cooldown_until = self._cooldowns.get(str(token_mint or ""))
            return cooldown_until is not None and cooldown_until > now

    def mark_burned(self, token_mint: str, reason: str | None = None) -> None:
        """Permanently blacklist a token for the current session."""
        mint = str(token_mint or "")
        if not mint:
            return
        with self._lock:
            self._burned_tokens.add(mint)

    def is_burned(self, token_mint: str) -> bool:
        with self._lock:
            return str(token_mint or "") in self._burned_tokens

    def can_open(
        self,
        token_mint: str,
        proposed_size_sol: float,
        open_position_count: int,
        total_exposure_sol: float,
    ) -> tuple[bool, str | None]:
        """Return whether a new entry is allowed."""
        now = datetime.now(tz=timezone.utc)
        mint = str(token_mint or "")
        with self._lock:
            self._ensure_current_day_locked()
            if mint in self._burned_tokens:
                return False, "token_burned_session"
            self._prune_cooldowns_locked(now)
            cooldown_until = self._cooldowns.get(mint)
            if cooldown_until is not None and cooldown_until > now:
                return False, "token_cooldown"
            if proposed_size_sol > self.config.max_position_sol:
                return False, "max_position_size"
            if total_exposure_sol + proposed_size_sol > self.config.max_total_exposure_sol:
                return False, "max_total_exposure"
            if open_position_count >= self.config.max_open_positions:
                return False, "max_open_positions"
            net_pnl = float(self._daily_row_cache.get("net_realized_pnl_sol", 0.0) or 0.0)
            if max(0.0, -net_pnl) >= self.config.max_daily_loss_sol:
                return False, "max_daily_loss"
            return True, None
