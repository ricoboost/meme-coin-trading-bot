"""Path, config, and data IO helpers."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    """Resolved application settings and paths."""

    root_dir: Path
    settings: dict[str, Any]
    wallets: dict[str, Any]
    env: dict[str, Any]

    @property
    def paths(self) -> dict[str, Path]:
        """Return configured project paths as absolute Path objects."""
        configured = self.settings.get("paths", {})
        return {key: self.root_dir / value for key, value in configured.items()}


def project_root() -> Path:
    """Return the repository root based on this file location."""
    return Path(__file__).resolve().parents[2]


def load_app_config() -> AppConfig:
    """Load environment variables and YAML config files."""
    root = project_root()
    load_dotenv(root / ".env")
    with (root / "config" / "settings.yaml").open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}
    with (root / "config" / "wallets.yaml").open("r", encoding="utf-8") as handle:
        wallets = yaml.safe_load(handle) or {}

    env = {
        "HELIUS_API_KEY": os.getenv("HELIUS_API_KEY", ""),
        "HELIUS_BASE_URL": os.getenv(
            "HELIUS_BASE_URL",
            str(settings.get("helius", {}).get("base_url", "https://api-mainnet.helius-rpc.com")),
        ),
        "LOOKBACK_DAYS": int(os.getenv("LOOKBACK_DAYS", "100")),
        "PROVIDER_A_TOP_N": int(
            os.getenv(
                "PROVIDER_A_TOP_N",
                str(settings.get("provider_a", {}).get("top_n_default", 50)),
            )
        ),
    }
    config = AppConfig(root_dir=root, settings=settings, wallets=wallets, env=env)
    ensure_directories(config)
    return config


def ensure_directories(config: AppConfig) -> None:
    """Create configured directories if they do not exist."""
    for path in config.paths.values():
        path.mkdir(parents=True, exist_ok=True)


def sanitize_for_json(value: Any) -> Any:
    """Recursively replace non-JSON-safe values with serializable ones."""
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def dumps_json_safe(value: Any, **kwargs: Any) -> str:
    """Serialize JSON with NaN/Infinity normalized to null."""
    return json.dumps(sanitize_for_json(value), default=str, allow_nan=False, **kwargs)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read parquet if present, otherwise return an empty dataframe."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique tmp per-call so concurrent writers to the same path don't stomp
    # each other's tmp (Thread A's replace would otherwise consume Thread B's
    # tmp, leaving B to crash on a missing file).
    unique = f".{os.getpid()}.{os.urandom(4).hex()}.tmp"
    tmp_path = path.with_suffix(path.suffix + unique)
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(dumps_json_safe(payload, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def read_json(path: Path) -> Any:
    """Read JSON content from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Append JSON rows and return the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(dumps_json_safe(row) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # Ignore torn/partial lines while another process is writing.
                    continue
    return records


def dataset_path(config: AppConfig, tier: str, name: str) -> Path:
    """Resolve a dataset parquet path from its storage tier and file name."""
    return config.paths[f"{tier}_dir"] / name
