"""Time helpers for UTC-aware processing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd


UTC = timezone.utc


def utcnow() -> datetime:
    """Return the current UTC time."""
    return datetime.now(tz=UTC)


def lookback_cutoff(days: int) -> datetime:
    """Return the UTC cutoff timestamp for the given lookback window."""
    return utcnow() - timedelta(days=days)


def to_datetime(value: object, unit: Optional[str] = None) -> pd.Timestamp:
    """Convert a value to a UTC pandas timestamp."""
    return pd.to_datetime(value, unit=unit, utc=True)
