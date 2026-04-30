"""Append-only JSONL event logging."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.io import dumps_json_safe


_REJECT_EVENT_TYPES = frozenset(
    {
        "entry_rejected",
        "sniper_entry_rejected",
        "main_entry_rejected",
    }
)

# Reason → gate classification. Dashboards + offline analysis can split the
# funnel into rule-miss (strategy thesis), safety-block (rug/execution surface),
# and capacity (slot/dedupe/cooldown) without re-deriving the mapping each
# time. Unknown reasons fall through to "unknown" so new reject sites still
# appear in the funnel — they just don't count toward any bucket until mapped.
_RULE_REASONS = frozenset(
    {
        "candidate_score_below_threshold",
        "sniper_no_matching_runtime_rule",
        "sniper_no_runtime_rules",
        "no_matching_rules",
        "ml_gate_rejected",
        "main_token_age_above_max",
        "awaiting_cluster_confirmation",
        "entry_lane_gate",
        "recovery_confirmation_gate",
    }
)
_SAFETY_REASONS = frozenset(
    {
        "live_entry_pool_abandoned",
        "live_entry_pool_low_liquidity",
        "live_entry_token_2022_risky_extension",
        "live_entry_token_program_blocked",
        "live_entry_freeze_authority_live",
        "live_entry_mint_authority_live",
        "live_entry_thin_crowd",
        "pure_buy_flow_young_token",
        "live_entry_holder_concentration",
        "live_entry_holder_concentration_top5",
        "guard_rpc_unavailable_honeypot",
        "live_entry_honeypot_detected",
        "honeypot_sim_rpc_failed",
        "live_entry_lp_guard",
        "paper_entry_liquidity_guard",
        "entry_quality_gate",
        "guard_rpc_unavailable_token_program",
    }
)
_CAPACITY_REASONS = frozenset(
    {
        "sniper_max_open_positions",
        "sniper_max_exposure",
        "existing_open_position",
        "open_positions_exist",
        "token_candidate_cooldown",
    }
)


def gate_for_reason(reason: str | None) -> str:
    if not reason:
        return "unknown"
    if reason in _SAFETY_REASONS:
        return "safety"
    if reason in _RULE_REASONS:
        return "rule"
    if reason in _CAPACITY_REASONS:
        return "capacity"
    return "unknown"


class EventLogger:
    """Write structured bot events to JSONL."""

    _DEFAULT_THROTTLED_EVENT_TYPES = frozenset(
        {
            "entry_rejected",
            "sniper_entry_rejected",
            "candidate_deferred",
        }
    )

    def __init__(
        self,
        path: Path,
        *,
        throttle_window_sec: float = 15.0,
        throttled_event_types: tuple[str, ...] = (),
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        self._throttle_window_sec = max(0.0, float(throttle_window_sec))
        types = tuple(item for item in throttled_event_types if item)
        self._throttled_event_types = (
            frozenset(types) if types else self._DEFAULT_THROTTLED_EVENT_TYPES
        )
        self._throttle_state: dict[tuple[str, str], dict[str, float | int]] = {}
        # Context fields merged into every emitted event. Let the runner set
        # mode/session_id once at startup so every downstream log line is
        # filterable without touching individual call sites.
        self._context: dict[str, Any] = {}

    def set_context(self, **fields: Any) -> None:
        """Merge constant fields into every future event (mode, session_id, ...)."""
        with self._lock:
            for key, value in fields.items():
                if value is None:
                    self._context.pop(key, None)
                else:
                    self._context[key] = value

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        with self._lock:
            if self._handle.closed:
                return
            self._handle.flush()
            self._handle.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _throttle_key(self, event_type: str, payload: dict[str, Any]) -> tuple[str, str] | None:
        if event_type not in self._throttled_event_types:
            return None
        key_parts = [
            str(event_type),
            str(payload.get("reason") or ""),
            str(payload.get("guard_reason") or ""),
            str(payload.get("strategy_id") or ""),
            str(payload.get("rule_id") or payload.get("selected_rule_id") or ""),
            str(payload.get("entry_lane") or ""),
        ]
        return event_type, "|".join(key_parts)

    def should_emit(self, event_type: str, payload: dict[str, Any]) -> bool:
        """Return whether one event would be written right now after throttling."""
        throttle_key = self._throttle_key(event_type, payload)
        if throttle_key is None or self._throttle_window_sec <= 0:
            return True
        with self._lock:
            state = self._throttle_state.get(throttle_key)
            if state is None:
                return True
            now_ts = time.monotonic()
            last_logged_at = float(state.get("last_logged_at", 0.0) or 0.0)
            return (now_ts - last_logged_at) >= self._throttle_window_sec

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append one event row."""
        row = {
            "event_type": event_type,
            "logged_at": datetime.now(tz=timezone.utc).isoformat(),
            **self._context,
            **payload,
        }
        # Auto-tag reject events with a gate category so downstream funnels
        # can split rule-miss vs safety-block vs capacity without re-deriving
        # the mapping. Preserve an explicit gate override if the caller set one.
        if event_type in _REJECT_EVENT_TYPES and "gate" not in row:
            row["gate"] = gate_for_reason(row.get("reason"))
        throttle_key = self._throttle_key(event_type, payload)
        with self._lock:
            if throttle_key is not None and self._throttle_window_sec > 0:
                now_ts = time.monotonic()
                state = self._throttle_state.get(throttle_key)
                if state is not None:
                    last_logged_at = float(state.get("last_logged_at", 0.0) or 0.0)
                    if (now_ts - last_logged_at) < self._throttle_window_sec:
                        state["suppressed_count"] = int(state.get("suppressed_count", 0) or 0) + 1
                        return
                    suppressed_count = int(state.get("suppressed_count", 0) or 0)
                else:
                    suppressed_count = 0
                self._throttle_state[throttle_key] = {
                    "last_logged_at": now_ts,
                    "suppressed_count": 0,
                }
                if suppressed_count > 0:
                    row["suppressed_count_since_last_log"] = suppressed_count

            self._handle.write(dumps_json_safe(row) + "\n")
