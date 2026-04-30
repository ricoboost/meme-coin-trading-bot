"""Open and closed position persistence."""

from __future__ import annotations

import threading
from typing import Any

from src.bot.models import PositionRecord
from src.storage.bot_db import BotDB


class PositionManager:
    """Persist and query simulated positions."""

    def __init__(self, db: BotDB) -> None:
        self.db = db
        self._lock = threading.RLock()
        self._open_positions_by_id: dict[int, dict[str, Any]] = {}
        self._open_position_ids_by_token: dict[str, set[int]] = {}
        self._reload_open_positions()

    def _reload_open_positions(self) -> None:
        rows = self.db.fetchall("SELECT * FROM positions WHERE status = 'OPEN' ORDER BY entry_time")
        with self._lock:
            self._open_positions_by_id = {int(row["id"]): dict(row) for row in rows}
            token_index: dict[str, set[int]] = {}
            for position_id, row in self._open_positions_by_id.items():
                token = str(row.get("token_mint") or "")
                if not token:
                    continue
                token_index.setdefault(token, set()).add(position_id)
            self._open_position_ids_by_token = token_index

    def _index_open_position_locked(self, row: dict[str, Any]) -> None:
        position_id = int(row["id"])
        token = str(row.get("token_mint") or "")
        self._open_positions_by_id[position_id] = dict(row)
        if token:
            self._open_position_ids_by_token.setdefault(token, set()).add(position_id)

    def _remove_open_position_locked(self, position_id: int) -> None:
        existing = self._open_positions_by_id.pop(position_id, None)
        if existing is None:
            return
        token = str(existing.get("token_mint") or "")
        if not token:
            return
        token_ids = self._open_position_ids_by_token.get(token)
        if not token_ids:
            return
        token_ids.discard(position_id)
        if not token_ids:
            self._open_position_ids_by_token.pop(token, None)

    def _update_open_position_locked(self, position_id: int, **fields: Any) -> None:
        row = self._open_positions_by_id.get(position_id)
        if row is None:
            return
        row.update(fields)

    def has_open_position(self, token_mint: str) -> bool:
        with self._lock:
            return bool(self._open_position_ids_by_token.get(str(token_mint or "")))

    def list_open_positions_for_token(self, token_mint: str) -> list[dict[str, Any]]:
        with self._lock:
            position_ids = sorted(
                self._open_position_ids_by_token.get(str(token_mint or ""), set())
            )
            positions = [
                dict(self._open_positions_by_id[position_id])
                for position_id in position_ids
                if position_id in self._open_positions_by_id
            ]
        positions.sort(key=lambda row: str(row.get("entry_time") or ""))
        return positions

    def open_position_count(self) -> int:
        with self._lock:
            return len(self._open_positions_by_id)

    def total_open_exposure(self) -> float:
        with self._lock:
            return sum(
                float(row.get("size_sol", 0.0) or 0.0)
                for row in self._open_positions_by_id.values()
            )

    def open_position(self, position: PositionRecord) -> int:
        cursor = self.db.execute(
            """
            INSERT INTO positions (
                token_mint, entry_time, entry_price_sol, size_sol, amount_received,
                strategy_id,
                selected_rule_id, selected_regime, matched_rule_ids, triggering_wallet,
                triggering_wallet_score, status, realized_pnl_sol, unrealized_pnl_sol,
                exit_stage, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position.token_mint,
                position.entry_time.isoformat(),
                position.entry_price_sol,
                position.size_sol,
                position.amount_received,
                position.strategy_id,
                position.selected_rule_id,
                position.selected_regime,
                self.db.dumps_json(position.matched_rule_ids),
                position.triggering_wallet,
                position.triggering_wallet_score,
                position.status,
                position.realized_pnl_sol,
                position.unrealized_pnl_sol,
                position.exit_stage,
                self.db.dumps_json(position.metadata),
            ),
        )
        position_id = int(cursor.lastrowid or 0)
        if position_id > 0:
            row = {
                "id": position_id,
                "token_mint": position.token_mint,
                "entry_time": position.entry_time.isoformat(),
                "entry_price_sol": position.entry_price_sol,
                "size_sol": position.size_sol,
                "amount_received": position.amount_received,
                "strategy_id": position.strategy_id,
                "selected_rule_id": position.selected_rule_id,
                "selected_regime": position.selected_regime,
                "matched_rule_ids": self.db.dumps_json(position.matched_rule_ids),
                "triggering_wallet": position.triggering_wallet,
                "triggering_wallet_score": position.triggering_wallet_score,
                "status": position.status,
                "realized_pnl_sol": position.realized_pnl_sol,
                "unrealized_pnl_sol": position.unrealized_pnl_sol,
                "exit_stage": position.exit_stage,
                "metadata_json": self.db.dumps_json(position.metadata),
            }
            with self._lock:
                self._index_open_position_locked(row)
        return position_id

    def list_open_positions(self) -> list[dict]:
        with self._lock:
            positions = [dict(row) for row in self._open_positions_by_id.values()]
        positions.sort(key=lambda row: str(row.get("entry_time") or ""))
        return positions

    def update_position_stage(
        self, position_id: int, exit_stage: int, realized_pnl_sol: float, status: str
    ) -> None:
        self.db.execute(
            "UPDATE positions SET exit_stage = ?, realized_pnl_sol = ?, status = ? WHERE id = ?",
            (exit_stage, realized_pnl_sol, status, position_id),
        )
        with self._lock:
            if str(status or "").upper() == "OPEN":
                self._update_open_position_locked(
                    int(position_id),
                    exit_stage=exit_stage,
                    realized_pnl_sol=realized_pnl_sol,
                    status=status,
                )
            else:
                self._remove_open_position_locked(int(position_id))

    def update_position_after_exit(
        self,
        position_id: int,
        exit_stage: int,
        realized_pnl_sol: float,
        status: str,
        remaining_size_sol: float,
        remaining_amount_received: float,
        metadata: dict[str, Any],
        unrealized_pnl_sol: float | None = None,
    ) -> None:
        """Persist one staged-exit update with remaining position state."""
        # For closed positions unrealized_pnl_sol must be zeroed.
        # For partial exits, pass the remaining mark-to-market estimate.
        _unrealized = 0.0 if status == "CLOSED" else (unrealized_pnl_sol or 0.0)
        self.db.execute(
            """
            UPDATE positions
            SET exit_stage = ?,
                realized_pnl_sol = ?,
                unrealized_pnl_sol = ?,
                status = ?,
                size_sol = ?,
                amount_received = ?,
                metadata_json = ?
            WHERE id = ?
            """,
            (
                exit_stage,
                realized_pnl_sol,
                _unrealized,
                status,
                remaining_size_sol,
                remaining_amount_received,
                self.db.dumps_json(metadata),
                position_id,
            ),
        )
        with self._lock:
            if str(status or "").upper() == "OPEN":
                self._update_open_position_locked(
                    int(position_id),
                    exit_stage=exit_stage,
                    realized_pnl_sol=realized_pnl_sol,
                    unrealized_pnl_sol=_unrealized,
                    status=status,
                    size_sol=remaining_size_sol,
                    amount_received=remaining_amount_received,
                    metadata_json=self.db.dumps_json(metadata),
                )
            else:
                self._remove_open_position_locked(int(position_id))

    def set_unrealized_pnl(self, position_id: int, unrealized_pnl_sol: float) -> None:
        """Lightweight update of the mark-to-market unrealized PnL column only."""
        self.db.execute(
            "UPDATE positions SET unrealized_pnl_sol = ? WHERE id = ? AND status = 'OPEN'",
            (unrealized_pnl_sol, position_id),
        )
        with self._lock:
            self._update_open_position_locked(
                int(position_id), unrealized_pnl_sol=unrealized_pnl_sol
            )

    def update_metadata(self, position_id: int, metadata: dict[str, Any]) -> None:
        """Persist metadata for one open position without changing stage/size."""
        self.db.execute(
            "UPDATE positions SET metadata_json = ? WHERE id = ?",
            (self.db.dumps_json(metadata), position_id),
        )
        with self._lock:
            self._update_open_position_locked(
                int(position_id), metadata_json=self.db.dumps_json(metadata)
            )
