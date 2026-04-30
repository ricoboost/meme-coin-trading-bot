"""Simple runtime state and status helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from src.utils.io import write_json


@dataclass
class BotRuntimeState:
    """Mutable runtime state for the bot loop."""

    last_seen_signatures: dict[str, str] = field(default_factory=dict)
    token_first_seen: dict[str, object] = field(default_factory=dict)
    daily_loss_date: date | None = None


class BotStatusWriter:
    """Persist lightweight bot runtime status for the dashboard."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.state: dict[str, Any] = {}

    def update(self, **fields: Any) -> None:
        """Merge fields into the current runtime status and write to disk."""
        self.state.update(fields)
        write_json(self.path, self.state)
