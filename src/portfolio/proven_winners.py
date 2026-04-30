"""Save winning trades' triggering wallets to manual_wallets.csv.

When a position closes in profit, the tracked wallet that fired the entry is
worth remembering — it's a live, proven signal, not a scraped guess. This
module appends such wallets to `data/bronze/manual_wallets.csv` (same schema
as the rest of the manual list: wallet,source,note) so the next
`refresh_wallet_pool` run picks them up automatically via the existing
collect_manual_wallets → union pipeline.

Dedupe rule: skip if the wallet already appears with source=proven_winner.
A wallet already present under a different source (e.g. provider_b) still gets
appended as proven_winner — the union groups by wallet and collapses sources
into a list, so both tags survive and build_scores boosts for it.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Optional

from src.utils.io import project_root

_SOURCE_TAG = "proven_winner"
_CSV_HEADER = ("wallet", "source", "note")
_LOCK = threading.Lock()


def _csv_path() -> Path:
    return project_root() / "data" / "bronze" / "manual_wallets.csv"


def _already_recorded(path: Path, wallet: str) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if (row.get("wallet") or "").strip() == wallet and (
                    row.get("source") or ""
                ).strip() == _SOURCE_TAG:
                    return True
    except (OSError, csv.Error):
        return False
    return False


def append_winner(
    wallet: str,
    *,
    position_id: int,
    realized_pnl_pct: float,
    realized_pnl_sol: Optional[float] = None,
) -> bool:
    """Append a winning triggering wallet to manual_wallets.csv.

    Returns True if a new row was written, False if deduped or invalid.
    Safe to call from any thread — guarded by a module-level lock.
    """
    wallet = (wallet or "").strip()
    if not wallet:
        return False

    path = _csv_path()
    with _LOCK:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False

        if _already_recorded(path, wallet):
            return False

        pct_tag = f"{realized_pnl_pct * 100:+.2f}%"
        sol_tag = f"_{realized_pnl_sol:+.4f}sol" if realized_pnl_sol is not None else ""
        note = f"pos_{position_id}_{pct_tag}{sol_tag}"

        write_header = not path.exists() or path.stat().st_size == 0
        try:
            with path.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                if write_header:
                    writer.writerow(_CSV_HEADER)
                writer.writerow([wallet, _SOURCE_TAG, note])
        except OSError:
            return False

    return True
