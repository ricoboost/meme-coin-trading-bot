"""SQLite persistence for the Phase 2 paper bot."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.io import dumps_json_safe


class BotDB:
    """Small SQLite wrapper for bot state."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: required because exit_engine runs in a thread-pool
        # executor (run_in_executor) while the entry path runs on the asyncio event loop.
        # All public methods serialise access through self._lock (RLock) so the connection
        # is never used from two threads simultaneously.
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # WAL mode: readers don't block writers and vice versa.
        # NORMAL sync is safe with WAL (journal survives a crash; data file is recovered
        # from the WAL on next open) and eliminates the full fsync on every commit.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        # Serialises all access from multiple threads (entry loop + exit executor).
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        cursor = self.conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_mint TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_price_sol REAL NOT NULL,
                size_sol REAL NOT NULL,
                amount_received REAL NOT NULL,
                strategy_id TEXT NOT NULL DEFAULT 'main',
                selected_rule_id TEXT NOT NULL,
                selected_regime TEXT NOT NULL,
                matched_rule_ids TEXT NOT NULL,
                triggering_wallet TEXT NOT NULL,
                triggering_wallet_score REAL NOT NULL,
                status TEXT NOT NULL,
                realized_pnl_sol REAL NOT NULL DEFAULT 0,
                unrealized_pnl_sol REAL NOT NULL DEFAULT 0,
                exit_stage INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_mint TEXT NOT NULL,
                action TEXT NOT NULL,
                mode TEXT NOT NULL,
                strategy_id TEXT NOT NULL DEFAULT 'main',
                size_sol REAL NOT NULL,
                price_sol REAL,
                tx_signature TEXT,
                status TEXT NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rule_performance (
                rule_id TEXT PRIMARY KEY,
                regime TEXT NOT NULL,
                entries INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                stop_outs INTEGER NOT NULL DEFAULT 0,
                hit_2x INTEGER NOT NULL DEFAULT 0,
                hit_5x INTEGER NOT NULL DEFAULT 0,
                average_pnl REAL NOT NULL DEFAULT 0,
                realized_pnl REAL NOT NULL DEFAULT 0,
                recent_pnl REAL NOT NULL DEFAULT 0,
                active_positions INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS token_cooldowns (
                token_mint TEXT PRIMARY KEY,
                cooldown_until TEXT NOT NULL,
                reason TEXT
            );
            CREATE TABLE IF NOT EXISTS risk_counters (
                counter_date TEXT PRIMARY KEY,
                daily_loss_sol REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                label TEXT,
                is_active INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS trade_legs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id INTEGER,
                token_mint TEXT NOT NULL,
                action TEXT NOT NULL,
                mode TEXT NOT NULL,
                strategy_id TEXT NOT NULL DEFAULT 'main',
                selected_rule_id TEXT,
                selected_regime TEXT,
                close_position INTEGER NOT NULL DEFAULT 0,
                stop_out INTEGER NOT NULL DEFAULT 0,
                hit_2x_achieved INTEGER NOT NULL DEFAULT 0,
                hit_5x_achieved INTEGER NOT NULL DEFAULT 0,
                quote_used INTEGER NOT NULL DEFAULT 0,
                quote_source TEXT,
                quote_error TEXT,
                observed_price_sol REAL,
                observed_price_raw_sol REAL,
                observed_price_reliable_sol REAL,
                observed_pnl_multiple REAL,
                executed_pnl_multiple REAL,
                cost_basis_sol REAL NOT NULL DEFAULT 0,
                leg_size_sol REAL NOT NULL DEFAULT 0,
                token_amount_raw REAL NOT NULL DEFAULT 0,
                gross_sol REAL NOT NULL DEFAULT 0,
                net_sol REAL NOT NULL DEFAULT 0,
                fee_sol REAL NOT NULL DEFAULT 0,
                realized_leg_pnl_sol REAL NOT NULL DEFAULT 0,
                realized_total_pnl_sol REAL NOT NULL DEFAULT 0,
                reason TEXT,
                tx_signature TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS token_first_seen (
                token_mint TEXT PRIMARY KEY,
                first_seen_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS token_source_first_seen (
                token_mint TEXT NOT NULL,
                source_program TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                PRIMARY KEY (token_mint, source_program)
            );
            CREATE TABLE IF NOT EXISTS token_launchers (
                token_mint TEXT PRIMARY KEY,
                launcher_wallet TEXT NOT NULL,
                first_source TEXT,
                recorded_at TEXT NOT NULL,
                graduated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS launcher_stats (
                launcher_wallet TEXT PRIMARY KEY,
                launches INTEGER NOT NULL DEFAULT 0,
                graduations INTEGER NOT NULL DEFAULT 0,
                first_seen_at TEXT NOT NULL,
                last_updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS token_ohlcv (
                token_mint TEXT NOT NULL,
                interval TEXT NOT NULL,
                ts INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume_sol REAL NOT NULL DEFAULT 0,
                tick_count INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL,
                PRIMARY KEY (token_mint, interval, ts)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS idx_token_launchers_wallet ON token_launchers(launcher_wallet);
            CREATE INDEX IF NOT EXISTS idx_trade_legs_position_id ON trade_legs(position_id);
            CREATE INDEX IF NOT EXISTS idx_trade_legs_rule_id ON trade_legs(selected_rule_id);
            CREATE INDEX IF NOT EXISTS idx_trade_legs_created_at ON trade_legs(created_at);
            CREATE INDEX IF NOT EXISTS idx_trade_legs_action_date ON trade_legs(action, created_at);
            CREATE INDEX IF NOT EXISTS idx_ohlcv_mint_ts ON token_ohlcv(token_mint, ts DESC);
            """
        )
        self._ensure_column("positions", "strategy_id", "TEXT NOT NULL DEFAULT 'main'")
        self._ensure_column("executions", "strategy_id", "TEXT NOT NULL DEFAULT 'main'")
        self._ensure_column("risk_counters", "net_realized_pnl_sol", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("risk_counters", "session_loss_sol", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("sessions", "ended_at", "TEXT")
        self._ensure_column("sessions", "label", "TEXT")
        self._ensure_column("sessions", "is_active", "INTEGER NOT NULL DEFAULT 0")
        self._bootstrap_sessions()
        self._rebuild_derived_state_from_history()
        self._backfill_position_regimes_from_rule_performance()
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        existing = {
            str(row["name"]) for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column in existing:
            return
        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _bootstrap_sessions(self) -> None:
        """Ensure at least one active session exists."""
        row = self.conn.execute(
            """
            SELECT id
            FROM sessions
            WHERE is_active = 1
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        if row is not None:
            return
        now = datetime.now(tz=timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO sessions (started_at, ended_at, label, is_active)
            VALUES (?, NULL, ?, 1)
            """,
            (now, "session_1"),
        )

    def _backfill_position_regimes_from_rule_performance(self) -> None:
        """Normalize stored position regimes using canonical rule performance regime."""
        self.conn.execute(
            """
            UPDATE positions
            SET selected_regime = (
                SELECT rp.regime
                FROM rule_performance rp
                WHERE rp.rule_id = positions.selected_rule_id
            )
            WHERE selected_rule_id IN (SELECT rule_id FROM rule_performance)
            """
        )

    @staticmethod
    def _load_json(raw: Any) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            payload = json.loads(str(raw))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _exit_timestamp_from_position(row: sqlite3.Row) -> str:
        metadata = BotDB._load_json(row["metadata_json"])
        exit_at = str(metadata.get("last_exit_at") or row["entry_time"] or "")
        return exit_at

    def _rebuild_derived_state_from_history(self) -> None:
        """Rebuild drift-prone summary tables from canonical history."""
        self.sync_all_rule_performance()
        self.sync_all_risk_counters()

    def _legacy_positions_without_buy_leg_locked(
        self, rule_id: str | None = None
    ) -> list[sqlite3.Row]:
        if rule_id is None:
            query = """
                SELECT *
                FROM positions p
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM trade_legs tl
                    WHERE tl.position_id = p.id
                      AND tl.action = 'BUY'
                )
            """
            params: tuple[Any, ...] = ()
        else:
            query = """
                SELECT *
                FROM positions p
                WHERE p.selected_rule_id = ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM trade_legs tl
                      WHERE tl.position_id = p.id
                        AND tl.action = 'BUY'
                  )
            """
            params = (rule_id,)
        return list(self.conn.execute(query, params).fetchall())

    def has_trade_legs(self) -> bool:
        """Return whether at least one canonical trade leg is present."""
        with self._lock:
            row = self.conn.execute("SELECT 1 FROM trade_legs LIMIT 1").fetchone()
            return row is not None

    def sync_rule_performance(self, rule_id: str, regime: str | None = None) -> None:
        """Rebuild one rule row from canonical trade legs plus legacy positions."""
        default_regime = str(regime or "unknown")
        with self._lock:
            entries = 0
            wins = 0
            losses = 0
            stop_outs = 0
            hit_2x = 0
            hit_5x = 0
            realized_pnl = 0.0
            closed_positions = 0
            recent_pnl = 0.0
            recent_at = ""
            derived_regime = default_regime

            leg_rows = self.conn.execute(
                """
                SELECT action, close_position, stop_out, hit_2x_achieved, hit_5x_achieved,
                       realized_total_pnl_sol, realized_leg_pnl_sol, created_at, selected_regime
                FROM trade_legs
                WHERE selected_rule_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (rule_id,),
            ).fetchall()
            for row in leg_rows:
                if row["selected_regime"]:
                    derived_regime = str(row["selected_regime"])
                action = str(row["action"] or "")
                if action == "BUY":
                    entries += 1
                    continue
                if action == "BUY_FEE":
                    entries += 1
                    closed_positions += 1
                    pnl = float(row["realized_leg_pnl_sol"] or row["realized_total_pnl_sol"] or 0.0)
                    realized_pnl += pnl
                    wins += int(pnl > 0.0)
                    losses += int(pnl <= 0.0)
                    created_at = str(row["created_at"] or "")
                    if created_at >= recent_at:
                        recent_at = created_at
                        recent_pnl = pnl
                    continue
                if action == "SELL_FEE":
                    pnl = float(row["realized_leg_pnl_sol"] or 0.0)
                    realized_pnl += pnl
                    created_at = str(row["created_at"] or "")
                    if created_at >= recent_at:
                        recent_at = created_at
                        recent_pnl = pnl
                    continue
                if action != "SELL" or not bool(row["close_position"]):
                    continue
                closed_positions += 1
                pnl = float(row["realized_total_pnl_sol"] or 0.0)
                realized_pnl += pnl
                wins += int(pnl > 0.0)
                losses += int(pnl <= 0.0)
                stop_outs += int(bool(row["stop_out"]))
                hit_2x += int(bool(row["hit_2x_achieved"]))
                hit_5x += int(bool(row["hit_5x_achieved"]))
                created_at = str(row["created_at"] or "")
                if created_at >= recent_at:
                    recent_at = created_at
                    recent_pnl = pnl

            legacy_rows = self._legacy_positions_without_buy_leg_locked(rule_id=rule_id)
            for row in legacy_rows:
                entries += 1
                if row["selected_regime"]:
                    derived_regime = str(row["selected_regime"])
                status = str(row["status"] or "")
                if status == "OPEN":
                    continue
                closed_positions += 1
                pnl = float(row["realized_pnl_sol"] or 0.0)
                realized_pnl += pnl
                wins += int(pnl > 0.0)
                losses += int(pnl <= 0.0)
                metadata = self._load_json(row["metadata_json"])
                last_reason = str(metadata.get("last_exit_reason") or "")
                stop_outs += int("stop_out" in last_reason)
                hit_2x += int(bool(metadata.get("hit_2x_achieved", False)))
                hit_5x += int(bool(metadata.get("hit_5x_achieved", False)))
                exit_at = self._exit_timestamp_from_position(row)
                if exit_at >= recent_at:
                    recent_at = exit_at
                    recent_pnl = pnl

            if entries <= 0:
                self.conn.execute("DELETE FROM rule_performance WHERE rule_id = ?", (rule_id,))
                self.conn.commit()
                return

            active_positions = max(0, entries - closed_positions)
            average_pnl = realized_pnl / float(entries) if entries > 0 else 0.0
            self.conn.execute(
                """
                INSERT INTO rule_performance (
                    rule_id, regime, entries, wins, losses, stop_outs, hit_2x, hit_5x,
                    average_pnl, realized_pnl, recent_pnl, active_positions
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(rule_id) DO UPDATE SET
                    regime = excluded.regime,
                    entries = excluded.entries,
                    wins = excluded.wins,
                    losses = excluded.losses,
                    stop_outs = excluded.stop_outs,
                    hit_2x = excluded.hit_2x,
                    hit_5x = excluded.hit_5x,
                    average_pnl = excluded.average_pnl,
                    realized_pnl = excluded.realized_pnl,
                    recent_pnl = excluded.recent_pnl,
                    active_positions = excluded.active_positions
                """,
                (
                    rule_id,
                    derived_regime or default_regime,
                    entries,
                    wins,
                    losses,
                    stop_outs,
                    hit_2x,
                    hit_5x,
                    average_pnl,
                    realized_pnl,
                    recent_pnl,
                    active_positions,
                ),
            )
            self.conn.commit()

    def sync_all_rule_performance(self) -> None:
        """Rebuild the full rule_performance table."""
        with self._lock:
            rules = {
                str(row["rule_id"])
                for row in self.conn.execute(
                    """
                    SELECT DISTINCT selected_rule_id AS rule_id
                    FROM trade_legs
                    WHERE selected_rule_id IS NOT NULL
                      AND selected_rule_id != ''
                    UNION
                    SELECT DISTINCT selected_rule_id AS rule_id
                    FROM positions
                    WHERE selected_rule_id IS NOT NULL
                      AND selected_rule_id != ''
                    """
                ).fetchall()
            }
            self.conn.execute("DELETE FROM rule_performance")
            self.conn.commit()
        for rule_id in sorted(rules):
            self.sync_rule_performance(rule_id)

    def sync_risk_counter(self, counter_date: str) -> None:
        """Rebuild one day of realized risk counters from canonical realized legs."""
        with self._lock:
            leg_row = self.conn.execute(
                """
                SELECT
                    COALESCE(SUM(realized_leg_pnl_sol), 0) AS net_realized_pnl_sol,
                    COALESCE(SUM(CASE WHEN realized_leg_pnl_sol < 0 THEN realized_leg_pnl_sol ELSE 0 END), 0) AS session_loss_sol
                FROM trade_legs
                WHERE action IN ('SELL', 'BUY_FEE', 'SELL_FEE')
                  AND substr(created_at, 1, 10) = ?
                """,
                (counter_date,),
            ).fetchone()
            net_realized = float(leg_row["net_realized_pnl_sol"] or 0.0)
            session_loss = float(leg_row["session_loss_sol"] or 0.0)

            legacy_rows = self.conn.execute(
                """
                SELECT p.entry_time, p.realized_pnl_sol, p.metadata_json
                FROM positions p
                WHERE p.status = 'CLOSED'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM trade_legs tl
                      WHERE tl.position_id = p.id
                        AND tl.action = 'SELL'
                        AND tl.close_position = 1
                  )
                """
            ).fetchall()
            for row in legacy_rows:
                exit_at = self._exit_timestamp_from_position(row)
                if not exit_at.startswith(counter_date):
                    continue
                pnl = float(row["realized_pnl_sol"] or 0.0)
                net_realized += pnl
                if pnl < 0:
                    session_loss += pnl

            daily_loss = min(0.0, net_realized)
            self.conn.execute(
                """
                INSERT INTO risk_counters (counter_date, daily_loss_sol, net_realized_pnl_sol, session_loss_sol)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(counter_date) DO UPDATE SET
                    daily_loss_sol = excluded.daily_loss_sol,
                    net_realized_pnl_sol = excluded.net_realized_pnl_sol,
                    session_loss_sol = excluded.session_loss_sol
                """,
                (counter_date, daily_loss, net_realized, session_loss),
            )
            self.conn.commit()

    def sync_all_risk_counters(self) -> None:
        """Rebuild all daily risk counters from canonical history."""
        with self._lock:
            dates = {
                str(row["counter_date"])
                for row in self.conn.execute(
                    """
                    SELECT DISTINCT substr(created_at, 1, 10) AS counter_date
                    FROM trade_legs
                    WHERE action IN ('SELL', 'BUY_FEE', 'SELL_FEE')
                    """
                ).fetchall()
                if row["counter_date"]
            }
            for row in self.conn.execute(
                """
                SELECT entry_time, metadata_json
                FROM positions
                WHERE status = 'CLOSED'
                """
            ).fetchall():
                exit_at = self._exit_timestamp_from_position(row)
                if exit_at:
                    dates.add(str(exit_at[:10]))
            self.conn.execute("DELETE FROM risk_counters")
            self.conn.commit()
        for counter_date in sorted(dates):
            self.sync_risk_counter(counter_date)

    def record_trade_leg(
        self,
        *,
        position_id: int | None,
        token_mint: str,
        action: str,
        mode: str,
        strategy_id: str = "main",
        selected_rule_id: str | None = None,
        selected_regime: str | None = None,
        close_position: bool = False,
        stop_out: bool = False,
        hit_2x_achieved: bool = False,
        hit_5x_achieved: bool = False,
        quote_used: bool = False,
        quote_source: str | None = None,
        quote_error: str | None = None,
        observed_price_sol: float | None = None,
        observed_price_raw_sol: float | None = None,
        observed_price_reliable_sol: float | None = None,
        observed_pnl_multiple: float | None = None,
        executed_pnl_multiple: float | None = None,
        cost_basis_sol: float = 0.0,
        leg_size_sol: float = 0.0,
        token_amount_raw: float = 0.0,
        gross_sol: float = 0.0,
        net_sol: float = 0.0,
        fee_sol: float = 0.0,
        realized_leg_pnl_sol: float = 0.0,
        realized_total_pnl_sol: float = 0.0,
        reason: str | None = None,
        tx_signature: str | None = None,
        created_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Persist one canonical trade leg."""
        created_at_value = created_at or datetime.now(tz=timezone.utc).isoformat()
        cursor = self.execute(
            """
            INSERT INTO trade_legs (
                position_id, token_mint, action, mode, strategy_id, selected_rule_id, selected_regime,
                close_position, stop_out, hit_2x_achieved, hit_5x_achieved, quote_used,
                quote_source, quote_error, observed_price_sol, observed_price_raw_sol,
                observed_price_reliable_sol, observed_pnl_multiple, executed_pnl_multiple,
                cost_basis_sol, leg_size_sol, token_amount_raw, gross_sol, net_sol, fee_sol,
                realized_leg_pnl_sol, realized_total_pnl_sol, reason, tx_signature, created_at,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position_id,
                token_mint,
                action,
                mode,
                strategy_id,
                selected_rule_id,
                selected_regime,
                int(close_position),
                int(stop_out),
                int(hit_2x_achieved),
                int(hit_5x_achieved),
                int(quote_used),
                quote_source,
                quote_error,
                observed_price_sol,
                observed_price_raw_sol,
                observed_price_reliable_sol,
                observed_pnl_multiple,
                executed_pnl_multiple,
                cost_basis_sol,
                leg_size_sol,
                token_amount_raw,
                gross_sol,
                net_sol,
                fee_sol,
                realized_leg_pnl_sol,
                realized_total_pnl_sol,
                reason,
                tx_signature,
                created_at_value,
                self.dumps_json(metadata or {}),
            ),
        )
        return int(cursor.lastrowid or 0)

    def record_token_observation(
        self,
        token_mint: str,
        observed_at: datetime,
        source_program: str | None = None,
    ) -> tuple[str, str | None]:
        """Persist earliest token/source observation and return canonical first-seen times."""
        observed_iso = observed_at.astimezone(timezone.utc).isoformat()
        source_first_seen: str | None = None
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO token_first_seen (token_mint, first_seen_at)
                VALUES (?, ?)
                ON CONFLICT(token_mint) DO UPDATE SET
                    first_seen_at = CASE
                        WHEN excluded.first_seen_at < token_first_seen.first_seen_at
                        THEN excluded.first_seen_at
                        ELSE token_first_seen.first_seen_at
                    END
                """,
                (token_mint, observed_iso),
            )
            token_row = self.conn.execute(
                "SELECT first_seen_at FROM token_first_seen WHERE token_mint = ?",
                (token_mint,),
            ).fetchone()
            if source_program:
                self.conn.execute(
                    """
                    INSERT INTO token_source_first_seen (token_mint, source_program, first_seen_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(token_mint, source_program) DO UPDATE SET
                        first_seen_at = CASE
                            WHEN excluded.first_seen_at < token_source_first_seen.first_seen_at
                            THEN excluded.first_seen_at
                            ELSE token_source_first_seen.first_seen_at
                        END
                    """,
                    (token_mint, source_program, observed_iso),
                )
                source_row = self.conn.execute(
                    """
                    SELECT first_seen_at
                    FROM token_source_first_seen
                    WHERE token_mint = ? AND source_program = ?
                    """,
                    (token_mint, source_program),
                ).fetchone()
                if source_row is not None:
                    source_first_seen = str(source_row["first_seen_at"])
            self.conn.commit()
        assert token_row is not None
        return str(token_row["first_seen_at"]), source_first_seen

    def record_token_launcher(
        self,
        token_mint: str,
        launcher_wallet: str,
        first_source: str | None,
        observed_at: datetime,
    ) -> None:
        """Record a token's proxy-launcher wallet and bump the launcher's launch counter.

        The "launcher" is the triggering_wallet of the first swap we observed for
        this token_mint — imperfect but inexpensive. Idempotent on token_mint: only
        the first observation counts toward the launcher's launches.
        """
        if not token_mint or not launcher_wallet:
            return
        observed_iso = observed_at.astimezone(timezone.utc).isoformat()
        with self._lock:
            cursor = self.conn.execute(
                """
                INSERT INTO token_launchers (token_mint, launcher_wallet, first_source, recorded_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(token_mint) DO NOTHING
                """,
                (token_mint, launcher_wallet, first_source, observed_iso),
            )
            inserted = cursor.rowcount > 0
            if inserted:
                self.conn.execute(
                    """
                    INSERT INTO launcher_stats (launcher_wallet, launches, graduations, first_seen_at, last_updated_at)
                    VALUES (?, 1, 0, ?, ?)
                    ON CONFLICT(launcher_wallet) DO UPDATE SET
                        launches = launcher_stats.launches + 1,
                        last_updated_at = excluded.last_updated_at
                    """,
                    (launcher_wallet, observed_iso, observed_iso),
                )
            self.conn.commit()

    def record_token_graduation(self, token_mint: str, observed_at: datetime) -> None:
        """Mark a token graduated and increment the launcher's graduation counter.

        Idempotent: only the first graduation observation counts. Safe to call on
        every PUMP_AMM observation — we early-return if graduated_at is already set.
        """
        if not token_mint:
            return
        observed_iso = observed_at.astimezone(timezone.utc).isoformat()
        with self._lock:
            row = self.conn.execute(
                "SELECT launcher_wallet, graduated_at FROM token_launchers WHERE token_mint = ?",
                (token_mint,),
            ).fetchone()
            if row is None or row["graduated_at"] is not None:
                return
            launcher_wallet = str(row["launcher_wallet"])
            self.conn.execute(
                "UPDATE token_launchers SET graduated_at = ? WHERE token_mint = ? AND graduated_at IS NULL",
                (observed_iso, token_mint),
            )
            self.conn.execute(
                """
                UPDATE launcher_stats
                SET graduations = graduations + 1,
                    last_updated_at = ?
                WHERE launcher_wallet = ?
                """,
                (observed_iso, launcher_wallet),
            )
            self.conn.commit()

    def get_launcher_stats(self, launcher_wallet: str) -> dict | None:
        """Return {launches, graduations, first_seen_at} or None if wallet unknown."""
        if not launcher_wallet:
            return None
        with self._lock:
            row = self.conn.execute(
                "SELECT launches, graduations, first_seen_at FROM launcher_stats WHERE launcher_wallet = ?",
                (launcher_wallet,),
            ).fetchone()
        if row is None:
            return None
        return {
            "launches": int(row["launches"]),
            "graduations": int(row["graduations"]),
            "first_seen_at": str(row["first_seen_at"]),
        }

    def active_session(self) -> sqlite3.Row | None:
        """Return the active session row (or latest session if none active)."""
        with self._lock:
            row = self.conn.execute(
                """
                SELECT id, started_at, ended_at, label, is_active
                FROM sessions
                WHERE is_active = 1
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
            if row is not None:
                return row
            return self.conn.execute(
                """
                SELECT id, started_at, ended_at, label, is_active
                FROM sessions
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()

    def start_new_session(self, label: str | None = None) -> sqlite3.Row:
        """Create and activate a new dashboard session."""
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._lock:
            self.conn.execute(
                """
                UPDATE sessions
                SET is_active = 0,
                    ended_at = COALESCE(ended_at, ?)
                WHERE is_active = 1
                """,
                (now,),
            )
            cursor = self.conn.execute(
                """
                INSERT INTO sessions (started_at, ended_at, label, is_active)
                VALUES (?, NULL, ?, 1)
                """,
                (now, label),
            )
            self.conn.commit()
            session_id = int(cursor.lastrowid)
            row = self.conn.execute(
                """
                SELECT id, started_at, ended_at, label, is_active
                FROM sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
            assert row is not None
            return row

    def end_active_session(self, ended_at: str | None = None) -> sqlite3.Row | None:
        """Mark the current active session ended and return the latest session row."""
        now = ended_at or datetime.now(tz=timezone.utc).isoformat()
        with self._lock:
            self.conn.execute(
                """
                UPDATE sessions
                SET is_active = 0,
                    ended_at = COALESCE(ended_at, ?)
                WHERE is_active = 1
                """,
                (now,),
            )
            self.conn.commit()
            return self.conn.execute(
                """
                SELECT id, started_at, ended_at, label, is_active
                FROM sessions
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            self.conn.commit()
            return cursor

    def executemany(self, query: str, params: list[tuple[Any, ...]]) -> None:
        with self._lock:
            cursor = self.conn.cursor()
            cursor.executemany(query, params)
            self.conn.commit()

    def fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(query, params).fetchall())

    def fetchone(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(query, params).fetchone()

    @staticmethod
    def dumps_json(value: Any) -> str:
        return dumps_json_safe(value)
