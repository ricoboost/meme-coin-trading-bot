"""Read-only dashboard queries over bot state."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.dashboard.metrics import compute_hot_path_metrics
from src.utils.io import read_jsonl


@dataclass(frozen=True)
class DashboardPaths:
    """Paths consumed by the dashboard."""

    db_path: Path
    event_log_path: Path
    status_path: Path


class DashboardDataStore:
    """Serve read-only snapshots of bot activity."""

    def __init__(self, paths: DashboardPaths) -> None:
        self.paths = paths

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.paths.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        if not self.paths.db_path.exists():
            return []
        try:
            with self._connect() as conn:
                return [dict(row) for row in conn.execute(query, params).fetchall()]
        except sqlite3.OperationalError:
            # Allow dashboard startup before DB schema is initialized.
            return []

    def _read_recent_event_rows(self, limit: int) -> list[dict[str, Any]]:
        """Read only the most recent JSONL rows without scanning the full file."""
        if limit <= 0 or not self.paths.event_log_path.exists():
            return []

        # Keep a modest memory footprint and avoid full-file scans on every poll.
        max_lines = max(limit, 100)
        try:
            with self.paths.event_log_path.open("rb") as handle:
                handle.seek(0, 2)
                file_size = handle.tell()
                chunk_size = 256 * 1024
                max_bytes = 8 * 1024 * 1024
                read_bytes = 0
                buffer = b""

                while read_bytes < max_bytes and len(buffer.splitlines()) <= (max_lines + 1):
                    if file_size <= 0:
                        break
                    step = min(chunk_size, file_size)
                    file_size -= step
                    handle.seek(file_size)
                    chunk = handle.read(step)
                    buffer = chunk + buffer
                    read_bytes += step
                    if file_size == 0:
                        break

            lines = buffer.splitlines()[-max_lines:]
            records: list[dict[str, Any]] = []
            for raw in lines:
                if not raw:
                    continue
                try:
                    records.append(json.loads(raw.decode("utf-8")))
                except Exception:  # noqa: BLE001
                    continue
            return records
        except Exception:  # noqa: BLE001
            # Fallback to existing full-file reader when tail-read fails.
            rows = read_jsonl(self.paths.event_log_path)
            return rows[-max_lines:]

    def _read_events_since(
        self,
        window_start: datetime,
        *,
        hard_cap_bytes: int = 64 * 1024 * 1024,
        hard_cap_rows: int = 200_000,
    ) -> list[dict[str, Any]]:
        """Read JSONL rows backwards until one is older than ``window_start``.

        Used for time-window analytics (e.g. wallet-lane last-15-min) where the
        row-capped tail reader would truncate high-volume sessions and produce
        regressive counters. Byte / row caps are safety nets only — under normal
        event rates we stop when the boundary row is hit.
        """
        if not self.paths.event_log_path.exists():
            return []
        try:
            with self.paths.event_log_path.open("rb") as handle:
                handle.seek(0, 2)
                file_size = handle.tell()
                chunk_size = 256 * 1024
                read_bytes = 0
                buffer = b""
                boundary_hit = False

                while read_bytes < hard_cap_bytes and not boundary_hit:
                    if file_size <= 0:
                        break
                    step = min(chunk_size, file_size)
                    file_size -= step
                    handle.seek(file_size)
                    chunk = handle.read(step)
                    buffer = chunk + buffer
                    read_bytes += step

                    # Peek at the oldest fully-decoded line in the current buffer;
                    # if its timestamp is already past `window_start` we can stop
                    # reading more history.
                    newline_idx = buffer.find(b"\n")
                    if newline_idx == -1:
                        continue
                    head = buffer[:newline_idx]
                    try:
                        head_row = json.loads(head.decode("utf-8"))
                    except Exception:  # noqa: BLE001
                        continue
                    head_ts = self._parse_dt(head_row.get("logged_at"))
                    if head_ts is not None and head_ts < window_start:
                        boundary_hit = True

            lines = buffer.splitlines()
            records: list[dict[str, Any]] = []
            for raw in lines:
                if not raw:
                    continue
                try:
                    row = json.loads(raw.decode("utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                ts = self._parse_dt(row.get("logged_at"))
                if ts is not None and ts < window_start:
                    continue
                records.append(row)
                if len(records) >= hard_cap_rows:
                    break
            return records
        except Exception:  # noqa: BLE001
            return []

    def _fetchone(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        rows = self._fetchall(query, params)
        return rows[0] if rows else None

    def _active_session(self) -> dict[str, Any] | None:
        """Return active session row if available."""
        row = self._fetchone(
            """
            SELECT id, started_at, ended_at, label, is_active
            FROM sessions
            WHERE is_active = 1
            ORDER BY id DESC
            LIMIT 1
            """
        )
        if row is not None:
            return row
        return self._fetchone(
            """
            SELECT id, started_at, ended_at, label, is_active
            FROM sessions
            ORDER BY id DESC
            LIMIT 1
            """
        )

    def _session_clause(
        self,
        *,
        column: str,
        session: dict[str, Any] | None,
        has_where: bool,
    ) -> tuple[str, tuple[Any, ...]]:
        """Build SQL clause limiting rows to active session window."""
        if not session:
            return "", ()
        started_at = session.get("started_at")
        ended_at = session.get("ended_at")
        if not started_at:
            return "", ()

        conditions = [f"{column} >= ?"]
        params: list[Any] = [started_at]
        if ended_at:
            conditions.append(f"{column} < ?")
            params.append(ended_at)
        prefix = "WHERE" if not has_where else "AND"
        return f" {prefix} " + " AND ".join(conditions), tuple(params)

    @staticmethod
    def _parse_dt(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:  # noqa: BLE001
            return None

    def _within_session(self, row_time: Any, session: dict[str, Any] | None) -> bool:
        """Return True if timestamp is inside active session window."""
        if not session:
            return True
        started = self._parse_dt(session.get("started_at"))
        ended = self._parse_dt(session.get("ended_at"))
        ts = self._parse_dt(row_time)
        if ts is None:
            return False
        if started and ts < started:
            return False
        if ended and ts >= ended:
            return False
        return True

    def summary(self) -> dict[str, Any]:
        """Return a top-level monitoring summary."""
        session = self._active_session()

        position_clause, position_params = self._session_clause(
            column="entry_time",
            session=session,
            has_where=True,
        )
        execution_clause, execution_params = self._session_clause(
            column="created_at",
            session=session,
            has_where=False,
        )

        open_row = (
            self._fetchone(
                f"""
            SELECT COUNT(*) AS open_positions,
                   COALESCE(SUM(size_sol), 0) AS open_exposure_sol,
                   COALESCE(SUM(unrealized_pnl_sol), 0) AS unrealized_pnl_sol
            FROM positions
            WHERE status = 'OPEN'{position_clause}
            """,
                position_params,
            )
            or {}
        )
        closed_row = (
            self._fetchone(
                f"""
            SELECT COUNT(*) AS closed_positions,
                   COALESCE(SUM(realized_pnl_sol), 0) AS realized_pnl_sol
            FROM positions
            WHERE status != 'OPEN'{position_clause}
            """,
                position_params,
            )
            or {}
        )
        sniper_open_row = (
            self._fetchone(
                f"""
            SELECT COUNT(*) AS sniper_open_positions,
                   COALESCE(SUM(size_sol), 0) AS sniper_open_exposure_sol
            FROM positions
            WHERE status = 'OPEN'
              AND COALESCE(strategy_id, json_extract(metadata_json, '$.strategy_id'), 'main') = 'sniper'{position_clause}
            """,
                position_params,
            )
            or {}
        )
        sniper_closed_row = (
            self._fetchone(
                f"""
            SELECT COUNT(*) AS sniper_closed_positions,
                   COALESCE(SUM(realized_pnl_sol), 0) AS sniper_realized_pnl_sol
            FROM positions
            WHERE status != 'OPEN'
              AND COALESCE(strategy_id, json_extract(metadata_json, '$.strategy_id'), 'main') = 'sniper'{position_clause}
            """,
                position_params,
            )
            or {}
        )
        execution_row = (
            self._fetchone(
                f"""
            SELECT COUNT(*) AS total_executions,
                   COALESCE(SUM(CASE WHEN action = 'BUY'  AND status != 'FAILED' THEN 1 ELSE 0 END), 0) AS buy_count,
                   COALESCE(SUM(CASE WHEN action = 'SELL' AND status != 'FAILED' THEN 1 ELSE 0 END), 0) AS sell_count,
                   COALESCE(SUM(CASE WHEN action = 'BUY'  AND status  = 'FAILED' THEN 1 ELSE 0 END), 0) AS buy_failed_count,
                   COALESCE(SUM(CASE WHEN action = 'SELL' AND status  = 'FAILED' THEN 1 ELSE 0 END), 0) AS sell_failed_count,
                   COALESCE(SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END), 0) AS failed_count,
                   MAX(created_at) AS last_execution_at,
                   MAX(mode) AS latest_mode
            FROM executions{execution_clause}
            """,
                execution_params,
            )
            or {}
        )
        sniper_execution_clause, sniper_execution_params = self._session_clause(
            column="created_at",
            session=session,
            has_where=True,
        )
        sniper_execution_row = (
            self._fetchone(
                f"""
            SELECT COUNT(*) AS sniper_total_executions,
                   COALESCE(SUM(CASE WHEN action = 'BUY' THEN 1 ELSE 0 END), 0) AS sniper_buy_count,
                   COALESCE(SUM(CASE WHEN action = 'SELL' THEN 1 ELSE 0 END), 0) AS sniper_sell_count
            FROM executions
            WHERE COALESCE(strategy_id, 'main') = 'sniper'{sniper_execution_clause}
            """,
                sniper_execution_params,
            )
            or {}
        )

        leg_clause, leg_params = self._session_clause(
            column="created_at",
            session=session,
            has_where=False,
        )
        leg_row = (
            self._fetchone(
                f"""
            SELECT COUNT(*) AS leg_count,
                   COALESCE(SUM(CASE WHEN action IN ('SELL', 'BUY_FEE', 'SELL_FEE') THEN realized_leg_pnl_sol ELSE 0 END), 0) AS realized_pnl_sol,
                   COALESCE(SUM(CASE WHEN action IN ('SELL', 'BUY_FEE', 'SELL_FEE')
                                      AND COALESCE(strategy_id, 'main') = 'sniper'
                                     THEN realized_leg_pnl_sol ELSE 0 END), 0) AS sniper_realized_pnl_sol,
                   COALESCE(SUM(CASE WHEN action IN ('BUY', 'SELL', 'BUY_FEE', 'SELL_FEE') THEN fee_sol ELSE 0 END), 0) AS total_fee_sol
            FROM trade_legs{leg_clause}
            """,
                leg_params,
            )
            or {}
        )
        has_canonical_legs = int(leg_row.get("leg_count", 0) or 0) > 0

        realized_session = (
            float(leg_row.get("realized_pnl_sol", 0.0) or 0.0)
            if has_canonical_legs
            else float(closed_row.get("realized_pnl_sol", 0.0) or 0.0)
        )
        net_realized_pnl_today = realized_session
        current_session_loss = min(0.0, realized_session)
        effective_daily_loss = max(0.0, -realized_session)

        perf_row = (
            self._fetchone(
                f"""
            SELECT COUNT(DISTINCT selected_rule_id) AS tracked_rules,
                   COALESCE(SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END), 0) AS active_rule_positions
            FROM positions
            WHERE selected_rule_id IS NOT NULL
              AND selected_rule_id != ''{position_clause}
            """,
                position_params,
            )
            or {}
        )
        fees_clause, fees_params = self._session_clause(
            column="entry_time",
            session=session,
            has_where=False,
        )
        fees_row = (
            self._fetchone(
                f"""
            SELECT COALESCE(SUM(COALESCE(json_extract(metadata_json, '$.paper_cumulative_fees_sol'), 0)), 0) AS paper_fees_sol
            FROM positions{fees_clause}
            """,
                fees_params,
            )
            or {}
        )
        pnl_source_row = (
            self._fetchone(
                f"""
            SELECT
              COALESCE(SUM(CASE WHEN json_extract(metadata_json, '$.pnl_source') = 'reconciled' THEN 1 ELSE 0 END), 0) AS pnl_reconciled_count,
              COALESCE(SUM(CASE WHEN json_extract(metadata_json, '$.pnl_source') = 'mark_price_fallback' THEN 1 ELSE 0 END), 0) AS pnl_fallback_count
            FROM positions
            WHERE status = 'CLOSED'{position_clause}
            """,
                position_params,
            )
            or {}
        )
        latest_event = self.recent_events(limit=1)
        status = self.status()
        realized_net_pnl = realized_session
        total_fees_sol = (
            float(leg_row.get("total_fee_sol", 0.0) or 0.0)
            if has_canonical_legs
            else float(fees_row.get("paper_fees_sol", 0.0) or 0.0)
        )
        sniper_realized_pnl_sol = (
            float(leg_row.get("sniper_realized_pnl_sol", 0.0) or 0.0)
            if has_canonical_legs
            else float(sniper_closed_row.get("sniper_realized_pnl_sol", 0.0) or 0.0)
        )
        return {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "db_exists": self.paths.db_path.exists(),
            "event_log_exists": self.paths.event_log_path.exists(),
            "open_positions": int(open_row.get("open_positions", 0) or 0),
            "closed_positions": int(closed_row.get("closed_positions", 0) or 0),
            "open_exposure_sol": float(open_row.get("open_exposure_sol", 0.0) or 0.0),
            "unrealized_pnl_sol": float(open_row.get("unrealized_pnl_sol", 0.0) or 0.0),
            "realized_pnl_sol": realized_net_pnl,
            "realized_net_pnl_sol": realized_net_pnl,
            "sniper_open_positions": int(sniper_open_row.get("sniper_open_positions", 0) or 0),
            "sniper_closed_positions": int(
                sniper_closed_row.get("sniper_closed_positions", 0) or 0
            ),
            "sniper_open_exposure_sol": float(
                sniper_open_row.get("sniper_open_exposure_sol", 0.0) or 0.0
            ),
            "sniper_realized_pnl_sol": sniper_realized_pnl_sol,
            "paper_fees_sol": total_fees_sol,
            "total_executions": int(execution_row.get("total_executions", 0) or 0),
            "buy_count": int(execution_row.get("buy_count", 0) or 0),
            "sell_count": int(execution_row.get("sell_count", 0) or 0),
            "buy_failed_count": int(execution_row.get("buy_failed_count", 0) or 0),
            "sell_failed_count": int(execution_row.get("sell_failed_count", 0) or 0),
            "failed_count": int(execution_row.get("failed_count", 0) or 0),
            "pnl_reconciled_count": int(pnl_source_row.get("pnl_reconciled_count", 0) or 0),
            "pnl_fallback_count": int(pnl_source_row.get("pnl_fallback_count", 0) or 0),
            "avoided_entry_count": self._count_avoided_entries(),
            "last_execution_at": execution_row.get("last_execution_at"),
            "sniper_total_executions": int(
                sniper_execution_row.get("sniper_total_executions", 0) or 0
            ),
            "sniper_buy_count": int(sniper_execution_row.get("sniper_buy_count", 0) or 0),
            "sniper_sell_count": int(sniper_execution_row.get("sniper_sell_count", 0) or 0),
            "mode_hint": str(
                status.get("mode") or execution_row.get("latest_mode") or "paper"
            ).upper(),
            "daily_loss_date": session.get("started_at") if session else None,
            "daily_loss_sol": effective_daily_loss,
            "net_realized_pnl_today_sol": net_realized_pnl_today,
            "current_session_loss_sol": current_session_loss,
            "tracked_rules": int(perf_row.get("tracked_rules", 0) or 0),
            "active_rule_positions": int(perf_row.get("active_rule_positions", 0) or 0),
            "active_session_id": int(session["id"]) if session is not None else None,
            "active_session_started_at": session.get("started_at") if session else None,
            "active_session_label": session.get("label") if session else None,
            "latest_event_type": latest_event[0]["event_type"] if latest_event else None,
            "latest_event_at": latest_event[0]["logged_at"] if latest_event else None,
            "monitoring_mode": status.get("monitoring_mode"),
            "discovery_mode": status.get("discovery_mode"),
            "bot_status": status.get("status"),
            "processed_events": int(status.get("processed_events", 0) or 0),
            "websocket_subscribed_count": int(status.get("websocket_subscribed_count", 0) or 0),
            "websocket_subscription_total": int(status.get("websocket_subscription_total", 0) or 0),
            "websocket_ready": bool(status.get("websocket_ready", False)),
            "websocket_notification_count": int(status.get("websocket_notification_count", 0) or 0),
            "websocket_candidate_event_count": int(
                status.get("websocket_candidate_event_count", 0) or 0
            ),
            "websocket_dropped_notification_count": int(
                status.get("websocket_dropped_notification_count", 0) or 0
            ),
            "websocket_last_drop_reason": status.get("websocket_last_drop_reason"),
            "websocket_drop_reason_counts": status.get("websocket_drop_reason_counts") or {},
            "websocket_parse_request_count": int(
                status.get("websocket_parse_request_count", 0) or 0
            ),
            "websocket_parse_batch_count": int(status.get("websocket_parse_batch_count", 0) or 0),
            "websocket_parsed_signature_count": int(
                status.get("websocket_parsed_signature_count", 0) or 0
            ),
            "websocket_avg_batch_size": float(status.get("websocket_avg_batch_size", 0.0) or 0.0),
            "pending_candidate_count": int(status.get("pending_candidate_count", 0) or 0),
            "entries_paused": bool(status.get("entries_paused", False)),
            "session_mode": status.get("session_mode") or "active",
            "end_session_requested_at": status.get("end_session_requested_at"),
            "end_session_applied_at": status.get("end_session_applied_at"),
            "end_session_close_attempts": int(status.get("end_session_close_attempts", 0) or 0),
            "end_session_closed": int(status.get("end_session_closed", 0) or 0),
            "end_session_failed": int(status.get("end_session_failed", 0) or 0),
        }

    def open_positions(
        self,
        token: str | None = None,
        rule_id: str | None = None,
        regime: str | None = None,
    ) -> list[dict[str, Any]]:
        session = self._active_session()
        clause, params = self._session_clause(column="entry_time", session=session, has_where=True)
        rows = self._fetchall(
            f"""
            SELECT *
            FROM positions
            WHERE status = 'OPEN'{clause}
            ORDER BY entry_time DESC
            """,
            params,
        )
        return self._filter_positions(rows, token=token, rule_id=rule_id, regime=regime)

    def recent_positions(
        self,
        limit: int = 25,
        token: str | None = None,
        rule_id: str | None = None,
        regime: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        session = self._active_session()
        clause, params = self._session_clause(column="entry_time", session=session, has_where=False)
        rows = self._fetchall(
            f"""
            SELECT *
            FROM positions{clause}
            ORDER BY entry_time DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        return self._filter_positions(
            rows, token=token, rule_id=rule_id, regime=regime, status=status
        )

    def recent_executions(
        self,
        limit: int = 50,
        token: str | None = None,
        action: str | None = None,
        mode: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        session = self._active_session()
        clause, params = self._session_clause(column="created_at", session=session, has_where=False)
        rows = self._fetchall(
            f"""
            SELECT *
            FROM executions{clause}
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        filtered: list[dict[str, Any]] = []
        for row in rows:
            if token and token.lower() not in str(row.get("token_mint", "")).lower():
                continue
            if action and action.lower() != str(row.get("action", "")).lower():
                continue
            if mode and mode.lower() != str(row.get("mode", "")).lower():
                continue
            if status and status.lower() != str(row.get("status", "")).lower():
                continue
            filtered.append(row)
        return filtered

    def rule_performance(
        self, limit: int = 25, rule_id: str | None = None, regime: str | None = None
    ) -> list[dict[str, Any]]:
        session = self._active_session()
        clause, params = self._session_clause(column="created_at", session=session, has_where=True)
        rows = self._fetchall(
            f"""
            SELECT
                selected_rule_id AS rule_id,
                COALESCE(NULLIF(selected_regime, ''), 'unknown') AS regime,
                COALESCE(SUM(CASE WHEN action IN ('BUY', 'BUY_FEE') THEN 1 ELSE 0 END), 0) AS entries,
                COALESCE(SUM(CASE
                    WHEN action = 'SELL' AND close_position = 1 AND realized_total_pnl_sol > 0 THEN 1
                    WHEN action = 'BUY_FEE' AND realized_leg_pnl_sol > 0 THEN 1
                    ELSE 0
                END), 0) AS wins,
                COALESCE(SUM(CASE
                    WHEN action = 'SELL' AND close_position = 1 AND realized_total_pnl_sol <= 0 THEN 1
                    WHEN action = 'BUY_FEE' AND realized_leg_pnl_sol <= 0 THEN 1
                    ELSE 0
                END), 0) AS losses,
                COALESCE(SUM(CASE WHEN action = 'SELL' AND close_position = 1 AND stop_out = 1 THEN 1 ELSE 0 END), 0) AS stop_outs,
                COALESCE(SUM(CASE WHEN action = 'SELL' AND close_position = 1 AND hit_2x_achieved = 1 THEN 1 ELSE 0 END), 0) AS hit_2x,
                COALESCE(SUM(CASE WHEN action = 'SELL' AND close_position = 1 AND hit_5x_achieved = 1 THEN 1 ELSE 0 END), 0) AS hit_5x,
                COALESCE(SUM(CASE WHEN action IN ('SELL', 'BUY_FEE', 'SELL_FEE') THEN realized_leg_pnl_sol ELSE 0 END), 0) AS realized_pnl,
                COALESCE(SUM(CASE WHEN action = 'BUY' THEN 1 ELSE 0 END), 0)
                  - COALESCE(SUM(CASE WHEN action = 'SELL' AND close_position = 1 THEN 1 ELSE 0 END), 0) AS active_positions
            FROM trade_legs
            WHERE selected_rule_id IS NOT NULL
              AND selected_rule_id != ''{clause}
            GROUP BY selected_rule_id, COALESCE(NULLIF(selected_regime, ''), 'unknown')
            ORDER BY realized_pnl DESC, entries DESC, rule_id ASC
            LIMIT ?
            """,
            (*params, limit),
        )
        filtered: list[dict[str, Any]] = []
        for row in rows:
            if rule_id and rule_id.lower() not in str(row.get("rule_id", "")).lower():
                continue
            if regime and regime.lower() not in str(row.get("regime", "")).lower():
                continue
            entries = int(row.get("entries", 0) or 0)
            realized_pnl = float(row.get("realized_pnl", 0.0) or 0.0)
            row["average_pnl"] = realized_pnl / float(entries) if entries > 0 else 0.0
            row["recent_pnl"] = realized_pnl
            filtered.append(row)
        return filtered

    def recent_events(
        self,
        limit: int = 100,
        token: str | None = None,
        rule_id: str | None = None,
        regime: str | None = None,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        session = self._active_session()
        # Scan a larger recent window so filtered queries still have enough rows.
        scan_limit = max(limit * 30, 1000)
        rows = list(reversed(self._read_recent_event_rows(scan_limit)))
        filtered: list[dict[str, Any]] = []
        for row in rows:
            if not self._within_session(row.get("logged_at"), session):
                continue
            if token and token.lower() not in json.dumps(row).lower():
                continue
            if rule_id and rule_id.lower() not in json.dumps(row).lower():
                continue
            if regime and regime.lower() not in json.dumps(row).lower():
                continue
            if event_type and event_type.lower() != str(row.get("event_type", "")).lower():
                continue
            filtered.append(row)
            if len(filtered) >= limit:
                break
        return filtered

    _AVOIDED_ENTRY_EVENT_TYPES = frozenset(
        {
            "entry_rejected",
            "sniper_entry_rejected",
            "main_entry_rejected",
            "live_entry_failed",
            "live_entry_failed_fee_burn",
            "live_entry_lp_guard",
        }
    )

    def _count_avoided_entries(self) -> int:
        """Count entry-avoidance events in the active session."""
        session = self._active_session()
        rows = self._read_recent_event_rows(2000)
        count = 0
        for row in rows:
            if not self._within_session(row.get("logged_at"), session):
                continue
            if str(row.get("event_type", "")) in self._AVOIDED_ENTRY_EVENT_TYPES:
                count += 1
        return count

    def status(self) -> dict[str, Any]:
        """Return bot runtime status JSON if present."""
        if not self.paths.status_path.exists():
            return {}
        try:
            return json.loads(self.paths.status_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def subscribed_wallets(self) -> dict[str, Any]:
        """Return websocket subscription wallet details from runtime status."""
        status = self.status()
        wallets = list(status.get("websocket_subscribed_wallets") or [])
        tracked = list(status.get("tracked_wallet_list") or [])
        subscribed = set(wallets)
        pending = [wallet for wallet in tracked if wallet not in subscribed]
        return {
            "websocket_connected": bool(status.get("websocket_connected", False)),
            "websocket_ready": bool(status.get("websocket_ready", False)),
            "subscribed_count": int(status.get("websocket_subscribed_count", len(wallets)) or 0),
            "subscription_total": int(
                status.get("websocket_subscription_total", len(tracked)) or 0
            ),
            "latest_wallet": status.get("websocket_latest_wallet"),
            "subscribed_wallets": wallets,
            "pending_wallets": pending,
            "candidate_event_count": int(status.get("websocket_candidate_event_count", 0) or 0),
            "dropped_notification_count": int(
                status.get("websocket_dropped_notification_count", 0) or 0
            ),
            "drop_reason_counts": status.get("websocket_drop_reason_counts") or {},
            "parse_request_count": int(status.get("websocket_parse_request_count", 0) or 0),
            "parse_batch_count": int(status.get("websocket_parse_batch_count", 0) or 0),
            "parsed_signature_count": int(status.get("websocket_parsed_signature_count", 0) or 0),
            "avg_batch_size": float(status.get("websocket_avg_batch_size", 0.0) or 0.0),
            "parse_sample_rate": status.get("websocket_parse_sample_rate"),
            "parse_budget_per_min": status.get("websocket_parse_budget_per_min"),
            "parse_budget_per_day": status.get("websocket_parse_budget_per_day"),
            "parse_budget_remaining_min": status.get("websocket_parse_budget_remaining_min"),
            "parse_budget_remaining_day": status.get("websocket_parse_budget_remaining_day"),
            "pending_parse_queue": int(status.get("websocket_pending_parse_queue", 0) or 0),
            "parse_estimated_credits": int(status.get("websocket_parse_estimated_credits", 0) or 0),
            "subscription_target_type": status.get("websocket_subscription_target_type"),
            "subscription_targets": status.get("websocket_subscription_targets") or [],
        }

    def pnl_series(
        self, limit: int = 200, rule_id: str | None = None, regime: str | None = None
    ) -> list[dict[str, Any]]:
        """Return cumulative realized PnL points from canonical realized legs."""
        session = self._active_session()
        clause, params = self._session_clause(column="created_at", session=session, has_where=True)
        rows = self._fetchall(
            f"""
            SELECT created_at, realized_leg_pnl_sol, selected_rule_id, selected_regime, token_mint, action
            FROM trade_legs
            WHERE action IN ('SELL', 'BUY_FEE', 'SELL_FEE'){clause}
            ORDER BY created_at ASC, id ASC
            """,
            params,
        )
        running = 0.0
        points: list[dict[str, Any]] = []
        for row in rows:
            if rule_id and rule_id.lower() not in str(row.get("selected_rule_id", "")).lower():
                continue
            if regime and regime.lower() not in str(row.get("selected_regime", "")).lower():
                continue
            realized = float(row.get("realized_leg_pnl_sol", 0.0) or 0.0)
            running += realized
            points.append(
                {
                    "time": row.get("created_at"),
                    "token_mint": row.get("token_mint"),
                    "rule_id": row.get("selected_rule_id"),
                    "regime": row.get("selected_regime"),
                    "action": row.get("action"),
                    "realized_pnl_sol": realized,
                    "cumulative_realized_pnl_sol": running,
                }
            )
        return points[-limit:]

    def rule_pnl_series(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return cumulative realized PnL per rule from canonical realized legs."""
        session = self._active_session()
        clause, params = self._session_clause(column="created_at", session=session, has_where=True)
        rows = self._fetchall(
            f"""
            SELECT created_at, realized_leg_pnl_sol, selected_rule_id
            FROM trade_legs
            WHERE action IN ('SELL', 'BUY_FEE', 'SELL_FEE'){clause}
            ORDER BY created_at ASC, id ASC
            """,
            params,
        )
        running_by_rule: dict[str, float] = {}
        points: list[dict[str, Any]] = []
        for row in rows:
            rule_id = str(row.get("selected_rule_id") or "unknown")
            running_by_rule[rule_id] = running_by_rule.get(rule_id, 0.0) + float(
                row.get("realized_leg_pnl_sol", 0.0) or 0.0
            )
            points.append(
                {
                    "time": row.get("created_at"),
                    "rule_id": rule_id,
                    "cumulative_realized_pnl_sol": running_by_rule[rule_id],
                }
            )
        return points[-limit:]

    def activity_series(self, limit: int = 60) -> list[dict[str, Any]]:
        """Return recent per-minute activity buckets from executions and rejections."""
        executions = self.recent_executions(limit=500)
        rejections = self.rejected_trades(limit=500)
        buckets: dict[str, dict[str, Any]] = {}

        def _bucket(ts: str | None) -> str | None:
            if not ts:
                return None
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except ValueError:
                return None
            return dt.replace(second=0, microsecond=0).isoformat()

        for row in executions:
            key = _bucket(row.get("created_at"))
            if not key:
                continue
            item = buckets.setdefault(key, {"time": key, "executions": 0, "rejections": 0})
            item["executions"] += 1

        for row in rejections:
            key = _bucket(row.get("logged_at"))
            if not key:
                continue
            item = buckets.setdefault(key, {"time": key, "executions": 0, "rejections": 0})
            item["rejections"] += 1

        ordered = [buckets[key] for key in sorted(buckets)]
        return ordered[-limit:]

    def rejected_trades(
        self,
        limit: int = 50,
        token: str | None = None,
        rule_id: str | None = None,
        regime: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return recent entry rejections from the event log."""
        rows = self.recent_events(
            limit=max(limit * 4, 100),
            token=token,
            rule_id=rule_id,
            regime=regime,
            event_type="entry_rejected",
        )
        return rows[:limit]

    def rejection_summary(self, limit: int = 250) -> dict[str, Any]:
        """Return compact reason/token counts for recent entry rejections."""
        rows = self.rejected_trades(limit=limit)
        by_reason: dict[str, int] = {}
        by_token: dict[str, int] = {}
        for row in rows:
            reason = str(row.get("reason") or "unknown")
            token = str(row.get("token_mint") or "unknown")
            by_reason[reason] = by_reason.get(reason, 0) + 1
            by_token[token] = by_token.get(token, 0) + 1
        top_reasons = [
            {"reason": reason, "count": count}
            for reason, count in sorted(by_reason.items(), key=lambda item: (-item[1], item[0]))[:8]
        ]
        top_tokens = [
            {"token_mint": token, "count": count}
            for token, count in sorted(by_token.items(), key=lambda item: (-item[1], item[0]))[:8]
        ]
        return {"top_reasons": top_reasons, "top_tokens": top_tokens}

    def sessions_list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Return a list of sessions with per-session aggregates (trades, pnl)."""
        sessions = self._fetchall(
            """
            SELECT id, started_at, ended_at, label, is_active
            FROM sessions
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(max(1, limit)),),
        )
        if not sessions:
            return []

        # Per-session aggregates from trade_legs (close legs only).
        aggregates: dict[int, dict[str, Any]] = {}
        for session in sessions:
            sid = int(session["id"])
            clause, params = self._session_clause(
                column="created_at", session=session, has_where=True
            )
            rows = self._fetchall(
                f"""
                SELECT
                    COALESCE(realized_total_pnl_sol, 0) AS pnl,
                    stop_out,
                    hit_2x_achieved,
                    hit_5x_achieved
                FROM trade_legs
                WHERE action = 'SELL' AND close_position = 1{clause}
                """,
                params,
            )
            trades = len(rows)
            wins = losses = be = stop_outs = hits_2x = hits_5x = 0
            pnl_sum = 0.0
            for row in rows:
                pnl = float(row.get("pnl") or 0.0)
                pnl_sum += pnl
                if pnl > 1e-9:
                    wins += 1
                elif pnl < -1e-9:
                    losses += 1
                else:
                    be += 1
                if int(row.get("stop_out") or 0):
                    stop_outs += 1
                if int(row.get("hit_2x_achieved") or 0):
                    hits_2x += 1
                if int(row.get("hit_5x_achieved") or 0):
                    hits_5x += 1
            decisive = wins + losses
            aggregates[sid] = {
                "trades": trades,
                "wins": wins,
                "losses": losses,
                "breakevens": be,
                "stop_outs": stop_outs,
                "hit_2x": hits_2x,
                "hit_5x": hits_5x,
                "realized_pnl_sol": pnl_sum,
                "win_rate": (float(wins) / float(decisive)) if decisive else None,
                "avg_pnl_sol": (pnl_sum / float(trades)) if trades else None,
            }

        out: list[dict[str, Any]] = []
        for session in sessions:
            sid = int(session["id"])
            row = dict(session)
            row["id"] = sid
            row["is_active"] = bool(int(session.get("is_active") or 0))
            row.update(aggregates.get(sid, {}))
            out.append(row)
        return out

    def session_scoreboard(self) -> dict[str, Any]:
        """Aggregate session-level win/loss stats, per-strategy, and funnel counts."""
        session = self._active_session()
        leg_clause, leg_params = self._session_clause(
            column="created_at", session=session, has_where=True
        )
        rows = self._fetchall(
            f"""
            SELECT
                COALESCE(strategy_id, 'main') AS strategy_id,
                COALESCE(realized_total_pnl_sol, 0) AS pnl,
                token_mint,
                selected_rule_id,
                created_at,
                stop_out,
                hit_2x_achieved,
                hit_5x_achieved
            FROM trade_legs
            WHERE action = 'SELL' AND close_position = 1{leg_clause}
            ORDER BY created_at ASC, id ASC
            """,
            leg_params,
        )

        total_wins = 0
        total_losses = 0
        total_be = 0
        total_pnl_sum = 0.0
        best: dict[str, Any] | None = None
        worst: dict[str, Any] | None = None
        stop_outs = 0
        hits_2x = 0
        hits_5x = 0
        per_strategy: dict[str, dict[str, Any]] = {}
        for row in rows:
            pnl = float(row.get("pnl") or 0.0)
            strat = str(row.get("strategy_id") or "main")
            rec = per_strategy.setdefault(
                strat,
                {
                    "strategy_id": strat,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "breakevens": 0,
                    "realized_pnl_sol": 0.0,
                },
            )
            rec["trades"] += 1
            rec["realized_pnl_sol"] += pnl
            if pnl > 1e-9:
                rec["wins"] += 1
                total_wins += 1
            elif pnl < -1e-9:
                rec["losses"] += 1
                total_losses += 1
            else:
                rec["breakevens"] += 1
                total_be += 1
            total_pnl_sum += pnl
            if int(row.get("stop_out") or 0):
                stop_outs += 1
            if int(row.get("hit_2x_achieved") or 0):
                hits_2x += 1
            if int(row.get("hit_5x_achieved") or 0):
                hits_5x += 1
            candidate = {
                "pnl_sol": pnl,
                "token_mint": row.get("token_mint"),
                "rule_id": row.get("selected_rule_id"),
                "closed_at": row.get("created_at"),
                "strategy_id": strat,
            }
            if best is None or pnl > float(best["pnl_sol"]):
                best = candidate
            if worst is None or pnl < float(worst["pnl_sol"]):
                worst = candidate

        total_closed = total_wins + total_losses + total_be
        decisive = total_wins + total_losses
        win_rate = float(total_wins) / float(decisive) if decisive else 0.0
        avg_pnl = total_pnl_sum / float(total_closed) if total_closed else 0.0
        for rec in per_strategy.values():
            rec_decisive = rec["wins"] + rec["losses"]
            rec["win_rate"] = float(rec["wins"]) / float(rec_decisive) if rec_decisive else 0.0
            rec["avg_pnl_sol"] = (
                float(rec["realized_pnl_sol"]) / float(rec["trades"]) if rec["trades"] else 0.0
            )

        status = self.status()
        rejections = self.rejection_summary(limit=500)
        notifs = int(status.get("websocket_notification_count") or 0)
        candidates = int(status.get("websocket_candidate_event_count") or 0)
        dropped = int(status.get("websocket_dropped_notification_count") or 0)
        summary = self.summary()
        entries = int(summary.get("buy_count") or 0)
        open_positions = int(summary.get("open_positions") or 0)
        rejected_total = sum(
            int(item.get("count") or 0) for item in rejections.get("top_reasons", [])
        )
        funnel = {
            "notifications": notifs,
            "dropped": dropped,
            "candidates": candidates,
            "rejected": rejected_total,
            "entries": entries,
            "open": open_positions,
            "closed": total_closed,
            "wins": total_wins,
            "losses": total_losses,
            "top_reject_reasons": rejections.get("top_reasons", []),
        }

        return {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "active_session_id": summary.get("active_session_id"),
            "closed_positions": total_closed,
            "wins": total_wins,
            "losses": total_losses,
            "breakevens": total_be,
            "decisive": decisive,
            "win_rate": win_rate,
            "realized_pnl_sol": total_pnl_sum,
            "avg_pnl_sol": avg_pnl,
            "stop_outs": stop_outs,
            "hit_2x": hits_2x,
            "hit_5x": hits_5x,
            "best_trade": best,
            "worst_trade": worst,
            "per_strategy": sorted(
                per_strategy.values(),
                key=lambda rec: (
                    -float(rec.get("realized_pnl_sol") or 0.0),
                    str(rec.get("strategy_id")),
                ),
            ),
            "funnel": funnel,
        }

    def token_detail(self, token_mint: str) -> dict[str, Any]:
        """Return a focused token drill-down."""
        positions = self.recent_positions(limit=200, token=token_mint)
        executions = self.recent_executions(limit=200, token=token_mint)
        events = self.recent_events(limit=200, token=token_mint)
        session = self._active_session()
        leg_clause, leg_params = self._session_clause(
            column="created_at", session=session, has_where=True
        )
        leg_row = (
            self._fetchone(
                f"""
            SELECT
                COALESCE(SUM(CASE WHEN action IN ('SELL', 'BUY_FEE', 'SELL_FEE') THEN realized_leg_pnl_sol ELSE 0 END), 0) AS realized_pnl_sol,
                COALESCE(SUM(CASE WHEN action IN ('BUY', 'SELL', 'BUY_FEE', 'SELL_FEE') THEN fee_sol ELSE 0 END), 0) AS fee_sol
            FROM trade_legs
            WHERE token_mint = ?{leg_clause}
            """,
                (token_mint, *leg_params),
            )
            or {}
        )
        open_positions = [row for row in positions if str(row.get("status")) == "OPEN"]
        realized = float(leg_row.get("realized_pnl_sol", 0.0) or 0.0)
        fees_deducted = float(leg_row.get("fee_sol", 0.0) or 0.0)
        rules = sorted(
            {str(row.get("selected_rule_id")) for row in positions if row.get("selected_rule_id")}
        )
        regimes = sorted(
            {str(row.get("selected_regime")) for row in positions if row.get("selected_regime")}
        )
        return {
            "token_mint": token_mint,
            "position_count": len(positions),
            "open_positions": len(open_positions),
            "execution_count": len(executions),
            "event_count": len(events),
            "realized_pnl_sol": realized,
            "fees_deducted_sol": fees_deducted,
            "rules": rules,
            "regimes": regimes,
            "latest_position_at": positions[0]["entry_time"] if positions else None,
            "latest_execution_at": executions[0]["created_at"] if executions else None,
            "positions": positions[:25],
            "executions": executions[:25],
            "events": events[:25],
        }

    def rule_detail(self, rule_id: str) -> dict[str, Any]:
        """Return a focused rule drill-down."""
        perf = self.rule_performance(limit=500, rule_id=rule_id)
        positions = self.recent_positions(limit=200, rule_id=rule_id)
        executions = self.recent_executions(limit=200)
        events = self.recent_events(limit=200, rule_id=rule_id)
        regimes = sorted(
            {str(row.get("selected_regime")) for row in positions if row.get("selected_regime")}
        )
        tokens = sorted({str(row.get("token_mint")) for row in positions if row.get("token_mint")})
        linked_executions: list[dict[str, Any]] = []
        token_set = set(tokens)
        for row in executions:
            if str(row.get("token_mint")) in token_set:
                linked_executions.append(row)
        return {
            "rule_id": rule_id,
            "performance": perf[0] if perf else None,
            "position_count": len(positions),
            "execution_count": len(linked_executions),
            "event_count": len(events),
            "regimes": regimes,
            "tokens": tokens[:50],
            "positions": positions[:25],
            "executions": linked_executions[:25],
            "events": events[:25],
        }

    def health(self) -> dict[str, Any]:
        summary = self.summary()
        status = self.status()
        return {
            "generated_at": summary["generated_at"],
            "db_exists": summary["db_exists"],
            "event_log_exists": summary["event_log_exists"],
            "mode_hint": summary["mode_hint"],
            "bot_status": status.get("status"),
            "monitoring_mode": status.get("monitoring_mode"),
            "tracked_wallets": status.get("tracked_wallets"),
            "active_rules": status.get("active_rules"),
            "processed_events": status.get("processed_events"),
            "last_cycle": status.get("last_cycle"),
            "last_cycle_processed": status.get("last_cycle_processed"),
            "open_positions": summary["open_positions"],
            "failed_count": summary["failed_count"],
            "last_execution_at": summary["last_execution_at"],
            "latest_event_type": summary["latest_event_type"],
            "latest_event_at": summary["latest_event_at"],
            "last_seen_event_at": status.get("last_seen_event_at"),
            "last_seen_token": status.get("last_seen_token"),
            "last_seen_wallet": status.get("last_seen_wallet"),
            "websocket_unavailable_reason": status.get("websocket_unavailable_reason"),
            "websocket_failure": status.get("websocket_failure"),
            "websocket_subscribed_count": status.get("websocket_subscribed_count"),
            "websocket_subscription_total": status.get("websocket_subscription_total"),
            "websocket_ready": status.get("websocket_ready"),
            "websocket_notification_count": status.get("websocket_notification_count"),
            "websocket_candidate_event_count": status.get("websocket_candidate_event_count"),
            "websocket_dropped_notification_count": status.get(
                "websocket_dropped_notification_count"
            ),
            "websocket_filter_mode": status.get("websocket_filter_mode"),
            "websocket_last_drop_reason": status.get("websocket_last_drop_reason"),
            "websocket_parse_request_count": status.get("websocket_parse_request_count"),
            "websocket_parse_batch_count": status.get("websocket_parse_batch_count"),
            "websocket_parsed_signature_count": status.get("websocket_parsed_signature_count"),
            "websocket_avg_batch_size": status.get("websocket_avg_batch_size"),
            "websocket_parse_sample_rate": status.get("websocket_parse_sample_rate"),
            "websocket_parse_budget_per_min": status.get("websocket_parse_budget_per_min"),
            "websocket_parse_budget_per_day": status.get("websocket_parse_budget_per_day"),
            "websocket_parse_budget_remaining_min": status.get(
                "websocket_parse_budget_remaining_min"
            ),
            "websocket_parse_budget_remaining_day": status.get(
                "websocket_parse_budget_remaining_day"
            ),
            "websocket_pending_parse_queue": status.get("websocket_pending_parse_queue"),
            "websocket_parse_estimated_credits": status.get("websocket_parse_estimated_credits"),
            "pending_candidate_count": status.get("pending_candidate_count"),
            "entries_paused": bool(status.get("entries_paused", False)),
            "session_mode": status.get("session_mode"),
            "active_session_id": summary.get("active_session_id"),
            "active_session_started_at": summary.get("active_session_started_at"),
            "active_session_label": summary.get("active_session_label"),
            "new_session_requested_at": status.get("new_session_requested_at"),
            "end_session_requested_at": status.get("end_session_requested_at"),
            "end_session_applied_at": status.get("end_session_applied_at"),
            "end_session_close_attempts": status.get("end_session_close_attempts"),
            "end_session_closed": status.get("end_session_closed"),
            "end_session_failed": status.get("end_session_failed"),
        }

    def hot_path_metrics(self, *, window_sec: float = 300.0) -> dict[str, Any]:
        """Latency percentiles per hot-path stage, derived from event log."""
        return compute_hot_path_metrics(self.paths.event_log_path, window_sec=window_sec)

    def wallet_panel(self, *, window_min: int = 15) -> dict[str, Any]:
        """Wallet-lane monitoring: pool health + cluster activity + lane funnel."""
        status = self.status()
        session = self._active_session()
        now = datetime.now(tz=timezone.utc)
        window_start = now - timedelta(minutes=window_min)

        tracked_count = int(status.get("tracked_wallets", 0) or 0)
        pool_path_str = os.getenv("TRACKED_WALLETS_PATH", "data/bronze/wallet_pool.parquet")
        pool_path = Path(pool_path_str)
        if not pool_path.is_absolute():
            pool_path = Path.cwd() / pool_path
        pool_refresh_age_sec: float | None = None
        pool_mtime_iso: str | None = None
        if pool_path.exists():
            mtime = pool_path.stat().st_mtime
            pool_refresh_age_sec = max(0.0, now.timestamp() - mtime)
            pool_mtime_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

        rows = self._read_events_since(window_start)
        total_events = 0
        events_with_tracked = 0
        biggest_cluster = 0
        biggest_cluster_token: str | None = None
        biggest_cluster_at: str | None = None
        wallet_appearances: dict[str, int] = {}
        last_tracked_buy: dict[str, Any] | None = None

        wallet_candidates = 0
        wallet_rejections_by_reason: dict[str, int] = {}

        for row in rows:
            logged_at = row.get("logged_at")
            if not self._within_session(logged_at, session):
                continue
            ts = self._parse_dt(logged_at)
            if ts is None or ts < window_start:
                continue

            event_type = str(row.get("event_type", ""))
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else row
            feature_snapshot = (
                payload.get("feature_snapshot")
                if isinstance(payload.get("feature_snapshot"), dict)
                else {}
            )

            # Cluster size: prefer payload-level field (populated by wallet-lane
            # rejections even when feature_snapshot is stripped), fall back to
            # snapshot field for generic candidate events.
            # Report 300s cluster (matches current wallet-lane gate window);
            # keep 30s as a fallback for legacy events.
            cluster_300s = int(
                payload.get("tracked_wallet_cluster_300s")
                or feature_snapshot.get("tracked_wallet_cluster_300s")
                or 0
            )
            cluster_30s = int(
                payload.get("tracked_wallet_cluster_30s")
                or feature_snapshot.get("tracked_wallet_cluster_30s")
                or 0
            )
            cluster_size = max(cluster_300s, cluster_30s)
            tracked_present = bool(
                feature_snapshot.get("tracked_wallet_present_60s")
                or feature_snapshot.get("tracked_wallet_count_60s")
                or 0
            )

            is_wallet_lane_event = event_type in (
                "wallet_candidate_selected",
                "wallet_entry_rejected",
                "wallet_copy_candidate_selected",
                "wallet_copy_entry_rejected",
            )

            # Activity counter: any event that either carries a feature snapshot
            # or is a wallet-lane event proves one transaction was evaluated.
            if feature_snapshot or is_wallet_lane_event:
                total_events += 1
                if tracked_present or cluster_size > 0 or is_wallet_lane_event:
                    events_with_tracked += 1

            if cluster_size > biggest_cluster:
                biggest_cluster = cluster_size
                biggest_cluster_token = payload.get("token_mint") or feature_snapshot.get(
                    "token_mint"
                )
                biggest_cluster_at = logged_at

            matched = payload.get("matched_tracked_wallets")
            if isinstance(matched, list) and matched:
                for wallet in matched:
                    if not wallet:
                        continue
                    wallet_appearances[str(wallet)] = wallet_appearances.get(str(wallet), 0) + 1
                if is_wallet_lane_event and (
                    last_tracked_buy is None
                    or (self._parse_dt(last_tracked_buy.get("ts")) or window_start) < ts
                ):
                    last_tracked_buy = {
                        "ts": logged_at,
                        "token_mint": payload.get("token_mint"),
                        "wallets": [str(w) for w in matched if w],
                    }

            if event_type in ("wallet_entry_rejected", "wallet_copy_entry_rejected"):
                reason = str(payload.get("reason") or "unknown")
                # Unpack engine-gate failures into per-check categories so
                # the funnel shows which 300s-scheme gate is biting.
                if reason in ("wallet_engine_gate", "wallet_copy_gate"):
                    failures = payload.get("failures") or []
                    if isinstance(failures, list) and failures:
                        for fail in failures:
                            key = str(fail).split("<", 1)[0].split(">", 1)[0].strip() or reason
                            wallet_rejections_by_reason[key] = (
                                wallet_rejections_by_reason.get(key, 0) + 1
                            )
                    else:
                        wallet_rejections_by_reason[reason] = (
                            wallet_rejections_by_reason.get(reason, 0) + 1
                        )
                else:
                    wallet_rejections_by_reason[reason] = (
                        wallet_rejections_by_reason.get(reason, 0) + 1
                    )
            elif event_type in (
                "wallet_candidate_selected",
                "wallet_copy_candidate_selected",
            ):
                wallet_candidates += 1

        top_wallets = sorted(
            ({"wallet": w, "appearances": n} for w, n in wallet_appearances.items()),
            key=lambda r: r["appearances"],
            reverse=True,
        )[:5]
        top_rejections = sorted(
            ({"reason": r, "count": n} for r, n in wallet_rejections_by_reason.items()),
            key=lambda r: r["count"],
            reverse=True,
        )[:6]

        exec_clause, exec_params = self._session_clause(
            column="created_at", session=session, has_where=True
        )
        exec_row = (
            self._fetchone(
                f"""
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(CASE WHEN action = 'BUY' AND status != 'FAILED' THEN 1 ELSE 0 END), 0) AS entries,
                   COALESCE(SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END), 0) AS failures
            FROM executions
            WHERE COALESCE(strategy_id, 'main') = 'wallet'{exec_clause}
            """,
                exec_params,
            )
            or {}
        )

        pos_clause, pos_params = self._session_clause(
            column="entry_time", session=session, has_where=True
        )
        pos_row = (
            self._fetchone(
                f"""
            SELECT COUNT(*) AS closed,
                   COALESCE(SUM(CASE WHEN realized_pnl_sol > 0 THEN 1 ELSE 0 END), 0) AS wins,
                   COALESCE(SUM(CASE WHEN realized_pnl_sol <= 0 THEN 1 ELSE 0 END), 0) AS losses,
                   COALESCE(SUM(realized_pnl_sol), 0) AS realized_pnl_sol
            FROM positions
            WHERE status != 'OPEN'
              AND COALESCE(strategy_id, json_extract(metadata_json, '$.strategy_id'), 'main') = 'wallet'{pos_clause}
            """,
                pos_params,
            )
            or {}
        )

        closed_count = int(pos_row.get("closed", 0) or 0)
        wins = int(pos_row.get("wins", 0) or 0)
        win_rate = (wins / closed_count) if closed_count > 0 else None

        return {
            "generated_at": now.isoformat(),
            "window_min": window_min,
            "pool": {
                "tracked_wallets": tracked_count,
                "features_enabled": bool(status.get("tracked_wallet_features_enabled", False)),
                "wallet_enabled": bool(status.get("wallet_enabled", False)),
                "wallet_open_positions": int(status.get("wallet_open_positions", 0) or 0),
                "wallet_exposure_sol": float(status.get("wallet_exposure_sol", 0.0) or 0.0),
                "pool_path": str(pool_path),
                "pool_refresh_age_sec": pool_refresh_age_sec,
                "pool_last_refreshed_at": pool_mtime_iso,
                "last_seen_event_at": status.get("last_seen_event_at"),
                "last_seen_wallet": status.get("last_seen_wallet"),
                "last_seen_token": status.get("last_seen_token"),
            },
            "activity": {
                "events_seen": total_events,
                "events_with_tracked_wallet": events_with_tracked,
                "tracked_wallet_share": (events_with_tracked / total_events)
                if total_events
                else 0.0,
                "biggest_cluster": biggest_cluster,
                "biggest_cluster_token": biggest_cluster_token,
                "biggest_cluster_at": biggest_cluster_at,
                "top_wallets": top_wallets,
                "last_tracked_buy": last_tracked_buy,
            },
            "funnel": {
                "candidates_selected": wallet_candidates,
                "rejections_total": sum(wallet_rejections_by_reason.values()),
                "top_rejections": top_rejections,
                "entries_executed": int(exec_row.get("entries", 0) or 0),
                "executions_total": int(exec_row.get("total", 0) or 0),
                "failures": int(exec_row.get("failures", 0) or 0),
                "closed_trades": closed_count,
                "wins": wins,
                "losses": int(pos_row.get("losses", 0) or 0),
                "win_rate": win_rate,
                "realized_pnl_sol": float(pos_row.get("realized_pnl_sol", 0.0) or 0.0),
            },
        }

    def live_tick(self) -> dict[str, Any]:
        """Compact live snapshot for the SSE stream.

        Small enough to push at ~2 Hz — summary KPIs, open positions with
        unrealized PnL, hot-path metrics, and a concise health slice.
        """
        summary = self.summary()
        health = self.health()
        positions = [self._expand_json_columns(row) for row in self.open_positions()]
        metrics = self.hot_path_metrics()
        return {
            "generated_at": summary.get("generated_at"),
            "summary": {
                "open_positions": summary.get("open_positions"),
                "open_exposure_sol": summary.get("open_exposure_sol"),
                "unrealized_pnl_sol": summary.get("unrealized_pnl_sol"),
                "realized_pnl_sol": summary.get("realized_pnl_sol"),
                "closed_positions": summary.get("closed_positions"),
                "buy_count": summary.get("buy_count"),
                "sell_count": summary.get("sell_count"),
                "buy_failed_count": summary.get("buy_failed_count"),
                "sell_failed_count": summary.get("sell_failed_count"),
                "failed_count": summary.get("failed_count"),
                "avoided_entry_count": summary.get("avoided_entry_count"),
                "last_execution_at": summary.get("last_execution_at"),
                "latest_event_at": summary.get("latest_event_at"),
                "latest_event_type": summary.get("latest_event_type"),
                "active_session_id": summary.get("active_session_id"),
                "active_session_started_at": summary.get("active_session_started_at"),
                "active_session_label": summary.get("active_session_label"),
            },
            "positions": positions,
            "metrics": metrics,
            "health": {
                "bot_status": health.get("bot_status"),
                "monitoring_mode": health.get("monitoring_mode"),
                "entries_paused": health.get("entries_paused"),
                "processed_events": health.get("processed_events"),
                "tracked_wallets": health.get("tracked_wallets"),
                "active_rules": health.get("active_rules"),
                "pending_candidate_count": health.get("pending_candidate_count"),
                "websocket_ready": health.get("websocket_ready"),
                "websocket_subscribed_count": health.get("websocket_subscribed_count"),
                "websocket_subscription_total": health.get("websocket_subscription_total"),
                "websocket_notification_count": health.get("websocket_notification_count"),
                "websocket_candidate_event_count": health.get("websocket_candidate_event_count"),
                "websocket_dropped_notification_count": health.get(
                    "websocket_dropped_notification_count"
                ),
                "websocket_pending_parse_queue": health.get("websocket_pending_parse_queue"),
                "last_cycle": health.get("last_cycle"),
                "last_cycle_processed": health.get("last_cycle_processed"),
            },
        }

    def _expand_json_columns(self, row: dict[str, Any]) -> dict[str, Any]:
        record = dict(row)
        for key in ("matched_rule_ids", "metadata_json"):
            value = record.get(key)
            if not value:
                record[key] = None
                continue
            try:
                record[key] = json.loads(str(value))
            except Exception:  # noqa: BLE001
                pass
        return record

    def _filter_positions(
        self,
        rows: list[dict[str, Any]],
        token: str | None = None,
        rule_id: str | None = None,
        regime: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for raw in rows:
            row = self._expand_json_columns(raw)
            if token and token.lower() not in str(row.get("token_mint", "")).lower():
                continue
            if rule_id and rule_id.lower() not in str(row.get("selected_rule_id", "")).lower():
                continue
            if regime and regime.lower() not in str(row.get("selected_regime", "")).lower():
                continue
            if status and status.lower() != str(row.get("status", "")).lower():
                continue
            filtered.append(row)
        return filtered
