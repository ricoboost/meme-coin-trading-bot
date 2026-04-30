"""Periodic wallet-vs-DB reconciliation for live mode.

Compares on-chain token balances (via ``getTokenAccountsByOwner``) to the
``positions`` table for every OPEN live position. Drift between the two —
either the wallet holds materially less than the DB claims (external close,
partial sell reconciliation failure) or materially more (external transfer,
buy reconciliation mis-attribution) — emits a ``live_reconciler_drift``
event so an operator can investigate.

The reconciler does **not** mutate the DB. Auto-healing drift would mask
real bugs (e.g. a reconciliation routine silently dropping 5% per sell).
This is a read-only alerting surface; corrective action is manual.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from src.execution.broadcaster import Broadcaster
    from src.storage.bot_db import BotDB
    from src.storage.event_log import EventLogger

logger = logging.getLogger(__name__)


# Callback invoked when we observe drift ≥ force_close_threshold:
#   (position_row, reason, last_error) -> bool (True if closed)
ForceCloseCallback = Callable[[dict[str, Any], str, str], bool]


class LiveReconciler:
    """Background thread comparing wallet token balances to DB position amounts."""

    def __init__(
        self,
        *,
        db: "BotDB",
        broadcaster: "Broadcaster",
        event_log: "EventLogger",
        wallet_pubkey: str,
        interval_sec: float = 60.0,
        drift_threshold_pct: float = 0.10,
        force_close_threshold_pct: float = 0.90,
    ) -> None:
        self._db = db
        self._broadcaster = broadcaster
        self._event_log = event_log
        self._wallet = str(wallet_pubkey)
        self._interval_sec = max(10.0, float(interval_sec))
        self._drift_threshold = max(0.01, float(drift_threshold_pct))
        self._force_close_threshold = max(self._drift_threshold, float(force_close_threshold_pct))
        self._force_close_cb: ForceCloseCallback | None = None
        self._force_closed_ids: set[int] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-reconciler", daemon=True)

    def set_force_close_callback(self, cb: ForceCloseCallback | None) -> None:
        """Install a callback that receives (position, reason, last_error) when
        drift crosses the force-close threshold. Wired after exit_engine
        construction in the runner since the reconciler is built earlier."""
        self._force_close_cb = cb

    def start(self) -> None:
        self._thread.start()
        logger.info(
            "LiveReconciler started (interval=%.0fs drift_threshold=%.2f%%)",
            self._interval_sec,
            self._drift_threshold * 100,
        )

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        # Initial delay so the reconciler doesn't race bot startup.
        self._stop.wait(self._interval_sec)
        while not self._stop.is_set():
            try:
                self._reconcile_once()
            except Exception as exc:  # noqa: BLE001
                logger.warning("LiveReconciler cycle failed: %s", exc)
            self._stop.wait(self._interval_sec)

    def _fetch_open_positions(self) -> list[dict[str, Any]]:
        rows = self._db.fetchall(
            """
            SELECT id, token_mint, amount_received, size_sol, status
            FROM positions
            WHERE status = 'OPEN'
            """,
        )
        return [dict(r) for r in rows] if rows else []

    def _reconcile_once(self) -> None:
        positions = self._fetch_open_positions()
        if not positions:
            return

        for row in positions:
            if self._stop.is_set():
                return
            mint = str(row.get("token_mint") or "")
            if not mint:
                continue
            db_amount = float(row.get("amount_received") or 0.0)
            try:
                on_chain_raw, _decimals = self._broadcaster.get_owner_token_balance_raw(
                    self._wallet, mint, commitment="confirmed"
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("reconciler balance fetch failed for %s: %s", mint[:8], exc)
                continue

            on_chain = float(on_chain_raw)
            # Normalize to the same scale as DB. ``amount_received`` is stored
            # as raw token units (no decimal scaling) per buy reconciliation.
            if db_amount <= 0 and on_chain <= 0:
                continue

            # Expected: on_chain ≈ db_amount. Compute relative drift against
            # the larger of the two so a 0-vs-anything case reads as 100%.
            denom = max(db_amount, on_chain, 1.0)
            drift = abs(on_chain - db_amount) / denom
            if drift < self._drift_threshold:
                continue

            position_id = int(row.get("id") or 0)
            self._event_log.log(
                "live_reconciler_drift",
                {
                    "token_mint": mint,
                    "position_id": position_id,
                    "db_amount": db_amount,
                    "on_chain_amount": on_chain,
                    "drift_ratio": round(drift, 6),
                    "threshold": self._drift_threshold,
                    "force_close_threshold": self._force_close_threshold,
                },
            )
            logger.warning(
                "🟡 reconciler drift %s: db=%.0f on_chain=%.0f drift=%.2f%%",
                mint[:12],
                db_amount,
                on_chain,
                drift * 100,
            )

            # When the wallet has essentially no tokens but the DB still tracks
            # a position, every sell attempt will fail on preflight (no tokens
            # to move). Invoke the force-close callback to mark the slot CLOSED
            # via the stuck-rug path and stop the sell loop. Only fires when
            # we hold < 10% of what the DB claims AND the callback is wired.
            if (
                self._force_close_cb is not None
                and drift >= self._force_close_threshold
                and db_amount > 0
                and position_id > 0
                and position_id not in self._force_closed_ids
            ):
                reason = "reconciler_drift_force_close"
                last_error = (
                    f"on_chain={on_chain:.0f} db={db_amount:.0f} drift={drift:.2f} "
                    f"(external_close_or_rug_freeze)"
                )
                try:
                    closed = bool(self._force_close_cb(row, reason, last_error))
                except Exception as exc:  # noqa: BLE001
                    closed = False
                    logger.exception(
                        "reconciler force-close callback failed for position %d: %s",
                        position_id,
                        exc,
                    )
                if closed:
                    self._force_closed_ids.add(position_id)
                    self._event_log.log(
                        "live_reconciler_force_close",
                        {
                            "token_mint": mint,
                            "position_id": position_id,
                            "reason": reason,
                            "drift_ratio": round(drift, 6),
                        },
                    )
                    logger.error(
                        "🔴 reconciler FORCE-CLOSED position %d %s (drift=%.2f%%, on_chain=%.0f)",
                        position_id,
                        mint[:12],
                        drift * 100,
                        on_chain,
                    )
