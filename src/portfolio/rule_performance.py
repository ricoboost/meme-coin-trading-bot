"""Per-rule performance tracking."""

from __future__ import annotations

from src.storage.bot_db import BotDB


class RulePerformanceTracker:
    """Persist simple per-rule paper-trading performance metrics."""

    def __init__(self, db: BotDB) -> None:
        self.db = db

    def ensure_rule(self, rule_id: str, regime: str) -> None:
        self.db.execute(
            """
            INSERT INTO rule_performance (rule_id, regime)
            VALUES (?, ?)
            ON CONFLICT(rule_id) DO NOTHING
            """,
            (rule_id, regime),
        )

    def record_entry(self, rule_id: str, regime: str) -> None:
        self.ensure_rule(rule_id, regime)
        if self.db.has_trade_legs():
            self.db.sync_rule_performance(rule_id, regime)
            return
        self.db.execute(
            "UPDATE rule_performance SET entries = entries + 1, active_positions = active_positions + 1 WHERE rule_id = ?",
            (rule_id,),
        )

    def record_exit(
        self,
        rule_id: str,
        pnl_sol: float,
        hit_2x: bool,
        hit_5x: bool,
        stop_out: bool,
        close_position: bool = True,
    ) -> None:
        if self.db.has_trade_legs():
            self.db.sync_rule_performance(rule_id)
            return
        row = self.db.fetchone(
            "SELECT entries, realized_pnl FROM rule_performance WHERE rule_id = ?",
            (rule_id,),
        )
        if row is None:
            return
        entries = int(row["entries"]) or 1
        wins = 1 if close_position and pnl_sol > 0 else 0
        losses = 1 if close_position and pnl_sol <= 0 else 0
        realized = float(row["realized_pnl"]) + pnl_sol
        avg_pnl = realized / entries
        self.db.execute(
            """
            UPDATE rule_performance
            SET wins = wins + ?,
                losses = losses + ?,
                stop_outs = stop_outs + ?,
                hit_2x = hit_2x + ?,
                hit_5x = hit_5x + ?,
                realized_pnl = ?,
                recent_pnl = ?,
                average_pnl = ?,
                active_positions = CASE WHEN ? = 1 AND active_positions > 0 THEN active_positions - 1 ELSE active_positions END
            WHERE rule_id = ?
            """,
            (
                wins,
                losses,
                int(stop_out and close_position),
                int(hit_2x and close_position),
                int(hit_5x and close_position),
                realized,
                pnl_sol,
                avg_pnl,
                int(close_position),
                rule_id,
            ),
        )
