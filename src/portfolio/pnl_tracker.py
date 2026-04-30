"""Portfolio-level PnL helpers."""

from __future__ import annotations

from src.storage.bot_db import BotDB


class PnLTracker:
    """Track realized portfolio PnL from stored positions."""

    def __init__(self, db: BotDB) -> None:
        self.db = db

    def total_realized_pnl(self) -> float:
        row = self.db.fetchone("SELECT COALESCE(SUM(realized_pnl_sol), 0) AS total FROM positions")
        return float(row["total"]) if row else 0.0
