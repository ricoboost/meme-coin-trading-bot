"""Hot-path metrics derived from the event log.

Reads the last N structured events from ``events.jsonl`` and extracts
latency fields into per-stage samples. Returns a snapshot containing
p50 / p95 / p99 / last / count-over-window for each stage so the
dashboard can render live charts without any engine instrumentation.

The event payloads already carry ``latency_trace``,
``pipeline_latency_trace`` and ``execution_latency_trace`` dictionaries
— this module is purely a read-side aggregator.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Stage -> ordered list of (source_trace, field, predicate). First matching
# trace that has the field wins. A predicate (callable) may reject rows,
# e.g. buy-vs-sell separation via ``execution_latency_trace.path``.
_STAGE_EXTRACTORS: dict[
    str,
    list[tuple[str, str, Any]],
] = {
    "event_to_runner_ms": [
        ("latency_trace", "event_effective_to_runner_arrival_ms", None),
        ("pipeline_latency_trace", "event_effective_to_runner_arrival_ms", None),
    ],
    "feature_build_ms": [
        ("pipeline_latency_trace", "feature_build_ms", None),
    ],
    "arrival_to_dispatch_ms": [
        ("pipeline_latency_trace", "arrival_to_entry_dispatch_ms", None),
    ],
    "buy_total_ms": [
        (
            "execution_latency_trace",
            "total_execution_ms",
            lambda trace: str(trace.get("path", "")).startswith("live_buy"),
        ),
    ],
    "sell_total_ms": [
        (
            "execution_latency_trace",
            "total_execution_ms",
            lambda trace: str(trace.get("path", "")).startswith("live_sell"),
        ),
    ],
    "broadcast_send_ms": [
        ("execution_latency_trace", "broadcast_send_ms", None),
    ],
    "broadcast_confirm_ms": [
        ("execution_latency_trace", "broadcast_confirm_ms", None),
    ],
    "jupiter_order_ms": [
        ("execution_latency_trace", "jupiter_order_ms", None),
    ],
    "preflight_ms": [
        ("execution_latency_trace", "preflight_ms", None),
    ],
    "reconcile_ms": [
        ("execution_latency_trace", "reconcile_ms", None),
    ],
}


def _percentile(samples: list[float], pct: float) -> float | None:
    if not samples:
        return None
    ordered = sorted(samples)
    if pct <= 0:
        return ordered[0]
    if pct >= 100:
        return ordered[-1]
    idx = (len(ordered) - 1) * (pct / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _parse_ts(value: Any) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except Exception:  # noqa: BLE001
        return None


def _extract_sample(event: dict[str, Any], stage: str) -> tuple[float, float] | None:
    for trace_key, field, predicate in _STAGE_EXTRACTORS.get(stage, ()):
        trace = event.get(trace_key)
        if not isinstance(trace, dict):
            continue
        if predicate is not None and not predicate(trace):
            continue
        raw = trace.get(field)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value < 0 or value != value:  # reject negatives + NaN
            continue
        ts = (
            _parse_ts(event.get("timestamp"))
            or _parse_ts(trace.get("event_effective_time"))
            or time.time()
        )
        return value, ts
    return None


def _tail_events(path: Path, *, max_lines: int = 2000) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            file_size = handle.tell()
            chunk = 256 * 1024
            max_bytes = 4 * 1024 * 1024
            read = 0
            buf = b""
            while read < max_bytes and len(buf.splitlines()) <= (max_lines + 1):
                if file_size <= 0:
                    break
                step = min(chunk, file_size)
                file_size -= step
                handle.seek(file_size)
                buf = handle.read(step) + buf
                read += step
                if file_size == 0:
                    break
        lines = buf.splitlines()[-max_lines:]
    except Exception:  # noqa: BLE001
        return []
    rows: list[dict[str, Any]] = []
    for raw in lines:
        if not raw:
            continue
        try:
            rows.append(json.loads(raw.decode("utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return rows


def compute_hot_path_metrics(
    event_log_path: Path,
    *,
    max_events: int = 2000,
    window_sec: float = 300.0,
) -> dict[str, Any]:
    """Return a per-stage latency snapshot derived from recent events.

    The result shape is::

        {
          "generated_at": iso8601,
          "window_sec": 300,
          "stages": {
            "buy_total_ms": {
              "count":        int,
              "last":         float | None,
              "p50":          float | None,
              "p95":          float | None,
              "p99":          float | None,
              "mean":         float | None,
              "samples":      [ [ts, ms], ... last 60 ],
            },
            ...
          }
        }
    """
    rows = _tail_events(event_log_path, max_lines=max_events)
    cutoff = time.time() - max(0.0, float(window_sec))

    per_stage: dict[str, list[tuple[float, float]]] = {s: [] for s in _STAGE_EXTRACTORS}
    for row in rows:
        for stage in _STAGE_EXTRACTORS:
            sample = _extract_sample(row, stage)
            if sample is None:
                continue
            value, ts = sample
            if ts < cutoff:
                continue
            per_stage[stage].append((ts, value))

    stages: dict[str, Any] = {}
    for stage, samples in per_stage.items():
        samples.sort(key=lambda item: item[0])
        values = [v for _, v in samples]
        stages[stage] = {
            "count": len(values),
            "last": values[-1] if values else None,
            "p50": _percentile(values, 50),
            "p95": _percentile(values, 95),
            "p99": _percentile(values, 99),
            "mean": (sum(values) / len(values)) if values else None,
            "samples": [[round(ts, 3), round(v, 3)] for ts, v in samples[-60:]],
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_sec": float(window_sec),
        "stages": stages,
    }
