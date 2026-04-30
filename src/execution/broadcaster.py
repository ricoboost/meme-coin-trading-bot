"""Helius transaction broadcaster with explicit send/rebroadcast control.

Supports three landing paths:

- ``staked_rpc``: standard Helius ``sendTransaction`` over the RPC URL
- ``helius_sender`` / ``helius_sender_swqos``: Helius Sender low-latency path
- ``helius_bundle``: bundle submission via ``sendBundle`` to a configured URL

For all transports we keep Solana RPC ``maxRetries`` at ``0`` and run our own
rebroadcast + confirmation loop, which is the recommended pattern for
latency-sensitive trading flows.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

logger = logging.getLogger(__name__)

_MAX_CONFIRM_POLLS = 150
_DEFAULT_CONFIRM_POLL_INTERVAL_SEC = 0.2
_DEFAULT_REBROADCAST_INTERVAL_SEC = 0.25
_SYSTEM_PROGRAM_ID = "11111111111111111111111111111111"
_COMPUTE_BUDGET_PROGRAM_ID = "ComputeBudget111111111111111111111111111111"


class BroadcastError(Exception):
    """Raised when transaction broadcast or confirmation fails."""


@dataclass(frozen=True)
class BroadcastResult:
    """Result of broadcasting a signed transaction."""

    signature: str
    confirmed: bool
    slot: int | None = None
    error: str | None = None
    send_latency_ms: float = 0.0
    confirm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    sent_at: str | None = None
    confirmed_at: str | None = None
    transport: str = "staked_rpc"
    bundle_id: str | None = None
    send_attempts: int = 0
    validated_tip_account: str | None = None
    validated_tip_lamports: int = 0
    validated_has_compute_unit_price: bool = False


def _append_query_param(url: str, name: str, value: str) -> str:
    """Return ``url`` with a query parameter merged in."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query[name] = value
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


class Broadcaster:
    """Send signed transactions to Solana via Helius RPC/Sender/bundle paths."""

    _FEE_CACHE_TTL_SEC: float = 10.0

    def __init__(
        self,
        rpc_url: str,
        timeout_sec: int = 30,
        fee_cache_ttl_sec: float = 10.0,
        *,
        sender_url: str = "",
        bundle_url: str = "",
        broadcast_mode: str = "staked_rpc",
        jito_tip_accounts: tuple[str, ...] = (),
        confirm_poll_interval_sec: float = _DEFAULT_CONFIRM_POLL_INTERVAL_SEC,
        rebroadcast_interval_sec: float = _DEFAULT_REBROADCAST_INTERVAL_SEC,
        max_rebroadcast_attempts: int = 8,
        sender_idle_ping_sec: float = 45.0,
        sender_active_warm: bool = True,
        sender_warm_interval_sec: float = 1.0,
    ) -> None:
        if not rpc_url:
            raise BroadcastError("HELIUS_RPC_URL is required for transaction broadcasting")
        if broadcast_mode not in {
            "staked_rpc",
            "helius_sender",
            "helius_sender_swqos",
            "helius_bundle",
        }:
            raise BroadcastError(f"Unsupported broadcast_mode={broadcast_mode}")

        self.rpc_url = rpc_url
        self.sender_url = sender_url.strip()
        self.bundle_url = bundle_url.strip() or self.sender_url or self.rpc_url
        self.broadcast_mode = broadcast_mode
        self.jito_tip_accounts = tuple(
            str(item).strip() for item in jito_tip_accounts if str(item).strip()
        )
        self.timeout_sec = timeout_sec
        self._fee_cache_ttl_sec = fee_cache_ttl_sec
        self._fee_cache: dict[str, Any] | None = None
        self._fee_cache_ts: float = 0.0
        self._confirm_poll_interval_sec = max(float(confirm_poll_interval_sec), 0.05)
        self._rebroadcast_interval_sec = max(float(rebroadcast_interval_sec), 0.05)
        self._max_rebroadcast_attempts = max(int(max_rebroadcast_attempts), 1)
        self._sender_idle_ping_sec = max(float(sender_idle_ping_sec), 0.0)
        self._sender_active_warm = bool(sender_active_warm)
        self._sender_warm_interval_sec = max(float(sender_warm_interval_sec), 0.0)
        self._sender_last_activity_ts: float = 0.0
        self._client = httpx.Client(timeout=self.timeout_sec)
        self._async_client = httpx.AsyncClient(timeout=self.timeout_sec)
        self._alt_cache: dict[str, Any] = {}
        # (mint → (monotonic_expiry_ts, result)) for top-holder pct lookups
        # used by the entry guard. Cheap TTL cache so back-to-back evaluations
        # on the same candidate don't spam Helius.
        self._top_holder_cache: dict[str, tuple[float, float]] = {}
        self._top_holder_cache_lock = threading.Lock()
        # (lp_mint → (expiry_ts, burn_fraction_or_None)). TTL cache for the
        # LP-burn guard; matches the top-holder cache pattern above.
        self._lp_burn_cache: dict[str, tuple[float, float | None]] = {}
        self._lp_burn_cache_lock = threading.Lock()
        # (pool_pubkey → (expiry_ts, lp_mint_or_None)). Pool account data only
        # changes on program upgrades, so caching the decoded LP mint avoids a
        # getAccountInfo per entry candidate.
        self._pool_lp_mint_cache: dict[str, tuple[float, str | None]] = {}
        self._pool_lp_mint_cache_lock = threading.Lock()
        # (mint → (mint_authority|None, freeze_authority|None)). Mint authorities
        # are immutable once revoked but mutable while set — cache for
        # ``_mint_authority_cache_ttl_sec`` so re-evaluating the same candidate
        # within a minute doesn't hit Helius again. ``None`` = authority revoked.
        self._mint_authority_cache: dict[
            str, tuple[float, tuple[str | None, str | None] | None]
        ] = {}
        self._mint_authority_cache_lock = threading.Lock()
        # (mint → (expiry_ts, creator_pubkey_or_None)). Decoded from pump.fun
        # bonding curve PDA; creator is immutable for the token's lifetime.
        self._pump_creator_cache: dict[str, tuple[float, str | None]] = {}
        self._pump_creator_cache_lock = threading.Lock()
        # (creator → (expiry_ts, token_count_or_None)). Helius DAS
        # getAssetsByCreator cache; count can grow quickly on serial creators
        # so TTL is short.
        self._creator_token_count_cache: dict[str, tuple[float, int | None]] = {}
        self._creator_token_count_cache_lock = threading.Lock()
        self._sender_warm_stop = threading.Event()
        self._sender_warm_thread: threading.Thread | None = None
        if (
            self.broadcast_mode in {"helius_sender", "helius_sender_swqos", "helius_bundle"}
            and self._sender_active_warm
            and self._sender_warm_interval_sec > 0.0
        ):
            self._sender_warm_thread = threading.Thread(
                target=self._sender_warm_loop,
                name="sender-warm",
                daemon=True,
            )
            self._sender_warm_thread.start()

    def close(self) -> None:
        """Close the persistent sync HTTP connection pool."""
        self._sender_warm_stop.set()
        if self._sender_warm_thread is not None and self._sender_warm_thread.is_alive():
            self._sender_warm_thread.join(timeout=1.0)
        self._client.close()

    # ------------------------------------------------------------------
    # Internal JSON-RPC helpers (sync + async)
    # ------------------------------------------------------------------

    @staticmethod
    def _rpc_body(method: str, params: list[Any]) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }

    async def _rpc_call_async(
        self, method: str, params: list[Any], *, url: str | None = None
    ) -> dict[str, Any]:
        """Execute a single JSON-RPC call without blocking the event loop."""
        resp = await self._async_client.post(
            url or self.rpc_url, json=self._rpc_body(method, params)
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        if "error" in data and data["error"]:
            raise BroadcastError(f"RPC error ({method}): {data['error']}")
        return data

    def _rpc_call(
        self, method: str, params: list[Any], *, url: str | None = None
    ) -> dict[str, Any]:
        """Execute a single JSON-RPC call."""
        resp = self._client.post(url or self.rpc_url, json=self._rpc_body(method, params))
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        if "error" in data and data["error"]:
            raise BroadcastError(f"RPC error ({method}): {data['error']}")
        return data

    def _sender_send_url(self) -> str:
        if not self.sender_url:
            raise BroadcastError("HELIUS_SENDER_URL is required for sender broadcast mode")
        if self.broadcast_mode == "helius_sender_swqos":
            return _append_query_param(self.sender_url, "swqos_only", "true")
        return self.sender_url

    def _sender_ping_url(self) -> str | None:
        target = self.sender_url or self.bundle_url
        if not target:
            return None
        parts = urlsplit(target)
        if not parts.scheme or not parts.netloc:
            return None
        return urlunsplit((parts.scheme, parts.netloc, "/ping", "", ""))

    def _mark_sender_activity(self) -> None:
        self._sender_last_activity_ts = time.monotonic()

    def _warm_health_once(self) -> None:
        try:
            self._rpc_call("getHealth", [], url=self.rpc_url)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Sender active warm getHealth failed: %s", exc)

    def _sender_warm_loop(self) -> None:
        while not self._sender_warm_stop.is_set():
            self._warm_health_once()
            if self._sender_warm_stop.wait(self._sender_warm_interval_sec):
                break

    def _maybe_warm_sender(self) -> None:
        if self.broadcast_mode not in {
            "helius_sender",
            "helius_sender_swqos",
            "helius_bundle",
        }:
            return
        if self._sender_idle_ping_sec <= 0:
            return
        now = time.monotonic()
        if (
            self._sender_last_activity_ts > 0
            and now - self._sender_last_activity_ts < self._sender_idle_ping_sec
        ):
            return
        ping_url = self._sender_ping_url()
        if not ping_url:
            return
        try:
            resp = self._client.get(ping_url)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Sender warm ping failed: %s", exc)
        self._mark_sender_activity()

    async def _maybe_warm_sender_async(self) -> None:
        if self.broadcast_mode not in {
            "helius_sender",
            "helius_sender_swqos",
            "helius_bundle",
        }:
            return
        if self._sender_idle_ping_sec <= 0:
            return
        now = time.monotonic()
        if (
            self._sender_last_activity_ts > 0
            and now - self._sender_last_activity_ts < self._sender_idle_ping_sec
        ):
            return
        ping_url = self._sender_ping_url()
        if not ping_url:
            return
        try:
            resp = await self._async_client.get(ping_url)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Async sender warm ping failed: %s", exc)
        self._mark_sender_activity()

    def get_transaction(self, signature: str) -> dict[str, Any] | None:
        """Fetch confirmed transaction metadata via JSON-RPC."""
        data = self._rpc_call(
            "getTransaction",
            [
                signature,
                {
                    "encoding": "jsonParsed",
                    "commitment": "confirmed",
                    "maxSupportedTransactionVersion": 0,
                },
            ],
        )
        result = data.get("result")
        return result if isinstance(result, dict) else None

    async def get_transaction_async(self, signature: str) -> dict[str, Any] | None:
        """Async variant of :meth:`get_transaction`."""
        data = await self._rpc_call_async(
            "getTransaction",
            [
                signature,
                {
                    "encoding": "jsonParsed",
                    "commitment": "confirmed",
                    "maxSupportedTransactionVersion": 0,
                },
            ],
        )
        result = data.get("result")
        return result if isinstance(result, dict) else None

    @staticmethod
    def _sum_token_accounts_raw(entries: list[Any]) -> tuple[int, int]:
        total_raw = 0
        decimals = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            account = entry.get("account") or {}
            data = account.get("data") or {}
            parsed = data.get("parsed") or {}
            info = parsed.get("info") or {}
            token_amount = info.get("tokenAmount") or {}
            try:
                total_raw += int(token_amount.get("amount") or 0)
            except (TypeError, ValueError):
                continue
            try:
                decimals = int(token_amount.get("decimals") or decimals or 0)
            except (TypeError, ValueError):
                pass
        return total_raw, decimals

    def get_balance(self, pubkey: str, *, commitment: str = "processed") -> int:
        """Fetch wallet SOL balance in lamports."""
        data = self._rpc_call(
            "getBalance",
            [
                str(pubkey),
                {
                    "commitment": str(commitment or "processed"),
                },
            ],
        )
        value = (data.get("result") or {}).get("value")
        return int(value or 0)

    def get_account_owner(self, pubkey: str, *, commitment: str = "confirmed") -> str | None:
        """Return the program that owns `pubkey` (e.g. the token program for a mint)."""
        data = self._rpc_call(
            "getAccountInfo",
            [
                str(pubkey),
                {"encoding": "base64", "commitment": str(commitment or "confirmed")},
            ],
        )
        value = ((data.get("result") or {}).get("value")) or {}
        owner = value.get("owner")
        return str(owner) if owner else None

    async def get_account_owner_async(
        self, pubkey: str, *, commitment: str = "confirmed"
    ) -> str | None:
        data = await self._rpc_call_async(
            "getAccountInfo",
            [
                str(pubkey),
                {"encoding": "base64", "commitment": str(commitment or "confirmed")},
            ],
        )
        value = ((data.get("result") or {}).get("value")) or {}
        owner = value.get("owner")
        return str(owner) if owner else None

    def get_mint_extensions(
        self,
        mint: str,
        *,
        commitment: str = "confirmed",
    ) -> tuple[str, list[str]] | None:
        """Return ``(owner_program, extensions)`` for a mint via jsonParsed.

        ``extensions`` is the list of Token-2022 extension names attached to
        the mint (empty for SPL Token mints and for plain Token-2022 mints
        without extensions). Used by entry guards to accept metadata-only
        Token-2022 mints while rejecting ones with risky extensions
        (transferFee, permanentDelegate, transferHook, etc.).

        Returns ``None`` on RPC failure so callers can fail-closed.
        """
        data = self._rpc_call(
            "getAccountInfo",
            [
                str(mint),
                {
                    "encoding": "jsonParsed",
                    "commitment": str(commitment or "confirmed"),
                },
            ],
        )
        value = ((data.get("result") or {}).get("value")) or {}
        owner = value.get("owner")
        if not owner:
            return None
        parsed = ((value.get("data") or {}).get("parsed")) or {}
        info = parsed.get("info") or {}
        raw_exts = info.get("extensions") or []
        extensions: list[str] = []
        for ext in raw_exts:
            name = ext.get("extension") if isinstance(ext, dict) else None
            if name:
                extensions.append(str(name))
        return str(owner), extensions

    def get_mint_authorities(
        self,
        mint: str,
        *,
        cache_ttl_sec: float = 60.0,
        commitment: str = "confirmed",
    ) -> tuple[str | None, str | None] | None:
        """Return ``(mint_authority, freeze_authority)`` for an SPL mint.

        Each element is the authority pubkey as a string, or ``None`` when that
        authority has been revoked (the safe state for a token intended for
        trading).

        Returns ``None`` when we couldn't decide (RPC failure, unparseable
        data, Token-2022 with exotic extensions we can't decode). Callers
        should treat ``None`` as "unknown" and apply their own policy —
        hard-rejecting on unknown is generally too strict because RPC hiccups
        would block all entries.

        Cached for ``cache_ttl_sec`` seconds. Authorities are semi-immutable
        (they can be revoked but not reinstated) so a short TTL is fine.
        """
        now = time.monotonic()
        with self._mint_authority_cache_lock:
            entry = self._mint_authority_cache.get(mint)
            if entry is not None and entry[0] > now:
                return entry[1]
        try:
            data = self._rpc_call(
                "getAccountInfo",
                [
                    str(mint),
                    {
                        "encoding": "jsonParsed",
                        "commitment": str(commitment or "confirmed"),
                    },
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("getAccountInfo(jsonParsed) failed for mint %s: %s", str(mint)[:12], exc)
            with self._mint_authority_cache_lock:
                self._mint_authority_cache[mint] = (now + cache_ttl_sec, None)
            return None
        value = ((data.get("result") or {}).get("value")) or {}
        parsed = ((value.get("data") or {}).get("parsed")) or {}
        info = parsed.get("info") or {}
        # jsonParsed returns ``null`` for revoked authorities and a pubkey string
        # when set. Missing keys (some Token-2022 variants) ⇒ unknown.
        if "mintAuthority" not in info and "freezeAuthority" not in info:
            with self._mint_authority_cache_lock:
                self._mint_authority_cache[mint] = (now + cache_ttl_sec, None)
            return None
        mint_auth_raw = info.get("mintAuthority")
        freeze_auth_raw = info.get("freezeAuthority")
        mint_auth = str(mint_auth_raw) if mint_auth_raw else None
        freeze_auth = str(freeze_auth_raw) if freeze_auth_raw else None
        result = (mint_auth, freeze_auth)
        with self._mint_authority_cache_lock:
            self._mint_authority_cache[mint] = (now + cache_ttl_sec, result)
        return result

    # ------------------------------------------------------------------
    # Token distribution helpers (holder-concentration / LP-burn checks)
    # ------------------------------------------------------------------
    # These are used by the live entry guards. All methods are best-effort
    # and return None / {} on failure — the caller decides whether to reject
    # or let the trade through when data is unavailable.

    _BURN_LIKE_OWNERS = frozenset(
        {
            "1nc1nerator11111111111111111111111111111111",
            "11111111111111111111111111111111",
        }
    )

    def get_token_supply_raw(
        self,
        mint: str,
        *,
        commitment: str = "confirmed",
    ) -> tuple[int, int]:
        """Return (raw_supply, decimals) for a mint. (0, 0) on failure."""
        try:
            data = self._rpc_call(
                "getTokenSupply",
                [str(mint), {"commitment": str(commitment or "confirmed")}],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("getTokenSupply failed for %s: %s", str(mint)[:12], exc)
            return 0, 0
        value = ((data.get("result") or {}).get("value")) or {}
        try:
            amount = int(value.get("amount") or 0)
        except (TypeError, ValueError):
            amount = 0
        try:
            decimals = int(value.get("decimals") or 0)
        except (TypeError, ValueError):
            decimals = 0
        return amount, decimals

    def get_token_largest_accounts(
        self,
        mint: str,
        *,
        commitment: str = "confirmed",
    ) -> list[tuple[str, int]]:
        """Return [(token_account_pubkey, raw_amount), ...] sorted descending.

        Returns an empty list on RPC failure.
        """
        try:
            data = self._rpc_call(
                "getTokenLargestAccounts",
                [str(mint), {"commitment": str(commitment or "confirmed")}],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("getTokenLargestAccounts failed for %s: %s", str(mint)[:12], exc)
            return []
        rows = list(((data.get("result") or {}).get("value")) or [])
        result: list[tuple[str, int]] = []
        for row in rows:
            try:
                amount = int(row.get("amount") or 0)
            except (TypeError, ValueError):
                continue
            pubkey = str(row.get("address") or "")
            if pubkey:
                result.append((pubkey, amount))
        result.sort(key=lambda item: item[1], reverse=True)
        return result

    def get_token_account_owners(
        self,
        token_accounts: list[str],
        *,
        commitment: str = "confirmed",
    ) -> dict[str, str]:
        """Bulk-resolve token account pubkeys → owner pubkeys via jsonParsed."""
        if not token_accounts:
            return {}
        try:
            data = self._rpc_call(
                "getMultipleAccounts",
                [
                    list(token_accounts),
                    {
                        "encoding": "jsonParsed",
                        "commitment": str(commitment or "confirmed"),
                    },
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "getMultipleAccounts failed for %d accounts: %s",
                len(token_accounts),
                exc,
            )
            return {}
        values = list(((data.get("result") or {}).get("value")) or [])
        owners: dict[str, str] = {}
        for pubkey, entry in zip(token_accounts, values):
            if not entry:
                continue
            parsed = (((entry.get("data") or {}).get("parsed")) or {}).get("info") or {}
            owner = parsed.get("owner")
            if owner:
                owners[str(pubkey)] = str(owner)
        return owners

    def get_top_n_non_pool_holder_sum_pct(
        self,
        mint: str,
        *,
        n: int = 5,
        exclude_owners: tuple[str, ...] = (),
        skip_top_holder: bool = True,
    ) -> float | None:
        """Return summed supply fraction held by the top-N non-pool wallets.

        Like :meth:`get_top_non_pool_holder_pct` but sums across N holders so
        bundled snipers who split dev allocation across multiple wallets can't
        slip under a top-1 threshold. Uses the same exclude/skip_top rules.

        Returns ``None`` when data is insufficient — callers should treat that
        as "unknown" and let the trade through.
        """
        if n <= 0:
            return 0.0
        supply_raw, _decimals = self.get_token_supply_raw(mint)
        if supply_raw <= 0:
            return None
        largest = self.get_token_largest_accounts(mint)
        if not largest:
            return None
        top_pubkeys = [pubkey for pubkey, _ in largest]
        owners_map = self.get_token_account_owners(top_pubkeys)
        excluded: set[str] = set(exclude_owners) | set(self._BURN_LIKE_OWNERS)
        picked = 0
        total = 0.0
        for idx, (pubkey, amount) in enumerate(largest):
            if skip_top_holder and idx == 0:
                continue
            owner = owners_map.get(pubkey, "")
            if owner and owner in excluded:
                continue
            total += float(amount) / float(supply_raw)
            picked += 1
            if picked >= n:
                break
        return total

    def get_top_non_pool_holder_pct(
        self,
        mint: str,
        *,
        exclude_owners: tuple[str, ...] = (),
        skip_top_holder: bool = True,
        cache_ttl_sec: float = 60.0,
    ) -> float | None:
        """Return the largest supply fraction held by a non-excluded wallet.

        Assumptions:
        - The single biggest holder of a fresh Pump/Pump-AMM token is the
          pool/bonding-curve itself. We skip it by default (``skip_top_holder``).
        - Known burn addresses are filtered out.
        - ``exclude_owners`` lets callers pass additional pool/router pubkeys.

        Returns None if we couldn't gather enough data to decide (supply=0,
        no largest accounts, etc.) — callers should treat None as "unknown"
        and allow the trade through rather than hard-fail.
        """
        now = time.monotonic()
        with self._top_holder_cache_lock:
            entry = self._top_holder_cache.get(mint)
            if entry is not None and entry[0] > now:
                return entry[1] if entry[1] >= 0 else None

        supply_raw, _decimals = self.get_token_supply_raw(mint)
        if supply_raw <= 0:
            with self._top_holder_cache_lock:
                self._top_holder_cache[mint] = (now + cache_ttl_sec, -1.0)
            return None

        largest = self.get_token_largest_accounts(mint)
        if not largest:
            with self._top_holder_cache_lock:
                self._top_holder_cache[mint] = (now + cache_ttl_sec, -1.0)
            return None

        top_pubkeys = [pubkey for pubkey, _ in largest]
        owners_map = self.get_token_account_owners(top_pubkeys)

        excluded: set[str] = set(exclude_owners) | set(self._BURN_LIKE_OWNERS)
        max_pct = 0.0
        for idx, (pubkey, amount) in enumerate(largest):
            if skip_top_holder and idx == 0:
                continue
            owner = owners_map.get(pubkey, "")
            if owner and owner in excluded:
                continue
            pct = float(amount) / float(supply_raw)
            if pct > max_pct:
                max_pct = pct

        with self._top_holder_cache_lock:
            self._top_holder_cache[mint] = (now + cache_ttl_sec, max_pct)
        return max_pct

    def get_pump_amm_lp_mint(self, pool_pubkey: str) -> str | None:
        """Decode the LP mint address from a Pump-AMM pool state account.

        Pool account layout (confirmed, 301 bytes):
            [0:8]     discriminator = f19a6d0411b16dbc
            [107:139] lp_mint

        Returns None when the account is missing, too small, or the fetch
        fails — callers should treat that as "unknown" (LP guard no-op).
        """
        try:
            from solders.pubkey import Pubkey  # type: ignore[import-untyped]
        except Exception:  # noqa: BLE001
            return None
        try:
            data = self._rpc_call(
                "getAccountInfo",
                [
                    str(pool_pubkey),
                    {"encoding": "base64", "commitment": "confirmed"},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "pump-amm pool account fetch failed for %s: %s",
                str(pool_pubkey)[:12],
                exc,
            )
            return None
        value = ((data.get("result") or {}).get("value")) or {}
        raw = value.get("data") or []
        if not raw or not isinstance(raw, list) or not raw[0]:
            return None
        try:
            account_bytes = base64.b64decode(raw[0])
        except Exception as exc:  # noqa: BLE001
            logger.debug("pump-amm pool data decode failed: %s", exc)
            return None
        if len(account_bytes) < 139:
            return None
        lp_mint_bytes = account_bytes[107:139]
        try:
            return str(Pubkey.from_bytes(lp_mint_bytes))
        except Exception as exc:  # noqa: BLE001
            logger.debug("pump-amm lp_mint decode failed: %s", exc)
            return None

    def get_raydium_v4_lp_mint(self, pool_pubkey: str) -> str | None:
        """Decode the LP mint address from a Raydium V4 AMM pool state account.

        ``LiquidityStateV4`` (752 bytes) layout (subset relevant here):
            [336:368]  pool_coin_token_account
            [368:400]  pool_pc_token_account
            [400:432]  coin_mint_address
            [432:464]  pc_mint_address
            [464:496]  lp_mint_address

        Returns None when the account is missing, too small, or the fetch
        fails — callers should treat that as "unknown" (fail-closed at the
        guard layer).
        """
        try:
            from solders.pubkey import Pubkey  # type: ignore[import-untyped]
        except Exception:  # noqa: BLE001
            return None
        try:
            data = self._rpc_call(
                "getAccountInfo",
                [
                    str(pool_pubkey),
                    {"encoding": "base64", "commitment": "confirmed"},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("raydium-v4 pool fetch failed for %s: %s", str(pool_pubkey)[:12], exc)
            return None
        value = ((data.get("result") or {}).get("value")) or {}
        raw = value.get("data") or []
        if not raw or not isinstance(raw, list) or not raw[0]:
            return None
        try:
            account_bytes = base64.b64decode(raw[0])
        except Exception as exc:  # noqa: BLE001
            logger.debug("raydium-v4 pool decode failed: %s", exc)
            return None
        if len(account_bytes) < 496:
            return None
        lp_mint_bytes = account_bytes[464:496]
        try:
            return str(Pubkey.from_bytes(lp_mint_bytes))
        except Exception as exc:  # noqa: BLE001
            logger.debug("raydium-v4 lp_mint decode failed: %s", exc)
            return None

    def get_pump_fun_creator(
        self,
        mint: str,
        *,
        cache_ttl_sec: float = 3600.0,
    ) -> str | None:
        """Return the creator wallet for a pump.fun bonding-curve token.

        Decodes the BondingCurve PDA at seeds ``(b"bonding-curve", mint)`` under
        pump.fun program ``6EF8rr…1Mp8``. Layout (Anchor):
            [0:8]    discriminator
            [8:16]   virtual_token_reserves (u64)
            [16:24]  virtual_sol_reserves (u64)
            [24:32]  real_token_reserves (u64)
            [32:40]  real_sol_reserves (u64)
            [40:48]  token_total_supply (u64)
            [48:49]  complete (bool)
            [49:81]  creator (Pubkey)

        Returns the creator as a base58 pubkey string, or ``None`` when the
        PDA is missing (non-pump token or graduated past curve deletion), the
        account is too small, or the RPC call fails. Cached per-mint since
        creator is immutable.
        """
        now = time.monotonic()
        with self._pump_creator_cache_lock:
            entry = self._pump_creator_cache.get(mint)
            if entry is not None and entry[0] > now:
                return entry[1]
        try:
            from solders.pubkey import Pubkey  # type: ignore[import-untyped]
        except Exception:  # noqa: BLE001
            return None
        try:
            pump_program = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
            mint_pk = Pubkey.from_string(str(mint))
            pda, _ = Pubkey.find_program_address([b"bonding-curve", bytes(mint_pk)], pump_program)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "pump.fun bonding-curve PDA derive failed for %s: %s",
                str(mint)[:12],
                exc,
            )
            with self._pump_creator_cache_lock:
                self._pump_creator_cache[mint] = (now + cache_ttl_sec, None)
            return None
        try:
            data = self._rpc_call(
                "getAccountInfo",
                [
                    str(pda),
                    {"encoding": "base64", "commitment": "confirmed"},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("pump.fun bonding-curve fetch failed for %s: %s", str(mint)[:12], exc)
            with self._pump_creator_cache_lock:
                self._pump_creator_cache[mint] = (now + cache_ttl_sec, None)
            return None
        value = ((data.get("result") or {}).get("value")) or {}
        raw = value.get("data") or []
        if not raw or not isinstance(raw, list) or not raw[0]:
            with self._pump_creator_cache_lock:
                self._pump_creator_cache[mint] = (now + cache_ttl_sec, None)
            return None
        try:
            account_bytes = base64.b64decode(raw[0])
        except Exception as exc:  # noqa: BLE001
            logger.debug("pump.fun bonding-curve decode failed: %s", exc)
            with self._pump_creator_cache_lock:
                self._pump_creator_cache[mint] = (now + cache_ttl_sec, None)
            return None
        if len(account_bytes) < 81:
            with self._pump_creator_cache_lock:
                self._pump_creator_cache[mint] = (now + cache_ttl_sec, None)
            return None
        try:
            creator = str(Pubkey.from_bytes(account_bytes[49:81]))
        except Exception as exc:  # noqa: BLE001
            logger.debug("pump.fun creator decode failed: %s", exc)
            with self._pump_creator_cache_lock:
                self._pump_creator_cache[mint] = (now + cache_ttl_sec, None)
            return None
        with self._pump_creator_cache_lock:
            self._pump_creator_cache[mint] = (now + cache_ttl_sec, creator)
        return creator

    def get_creator_recent_token_count(
        self,
        creator: str,
        *,
        limit: int = 25,
        cache_ttl_sec: float = 300.0,
    ) -> int | None:
        """Count tokens created by ``creator`` via Helius DAS getAssetsByCreator.

        Returns the number of assets where ``creator`` appears as a creator,
        capped at ``limit`` (most-recent first). Returns ``None`` on RPC
        failure — callers should treat ``None`` as "unknown" and apply their
        own fail-open/closed policy.

        Cached per-creator with a short TTL since serial creators can spin up
        new mints within minutes.
        """
        now = time.monotonic()
        with self._creator_token_count_cache_lock:
            entry = self._creator_token_count_cache.get(creator)
            if entry is not None and entry[0] > now:
                return entry[1]
        try:
            data = self._rpc_call(
                "getAssetsByCreator",
                [
                    {
                        "creatorAddress": str(creator),
                        "onlyVerified": False,
                        "page": 1,
                        "limit": int(limit),
                        "sortBy": {"sortBy": "created", "sortDirection": "desc"},
                    },
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("getAssetsByCreator failed for %s: %s", str(creator)[:12], exc)
            with self._creator_token_count_cache_lock:
                self._creator_token_count_cache[creator] = (now + cache_ttl_sec, None)
            return None
        result = data.get("result") or {}
        items = result.get("items") or []
        count = int(len(items))
        with self._creator_token_count_cache_lock:
            self._creator_token_count_cache[creator] = (now + cache_ttl_sec, count)
        return count

    def get_pool_lp_mint_cached(
        self,
        pool_pubkey: str,
        source: str,
        *,
        cache_ttl_sec: float = 900.0,
    ) -> str | None:
        """TTL-cached dispatch to the per-source LP-mint decoder.

        ``source`` is the upstream source label (``"PUMP_AMM"``,
        ``"RAYDIUM"``). Unknown sources return None.
        """
        key = f"{source}:{pool_pubkey}"
        now = time.monotonic()
        with self._pool_lp_mint_cache_lock:
            hit = self._pool_lp_mint_cache.get(key)
            if hit and hit[0] > now:
                return hit[1]
        if source == "PUMP_AMM":
            lp_mint = self.get_pump_amm_lp_mint(pool_pubkey)
        elif source == "RAYDIUM":
            lp_mint = self.get_raydium_v4_lp_mint(pool_pubkey)
        else:
            lp_mint = None
        with self._pool_lp_mint_cache_lock:
            self._pool_lp_mint_cache[key] = (now + float(cache_ttl_sec), lp_mint)
        return lp_mint

    def get_lp_burn_fraction_cached(
        self,
        lp_mint: str,
        *,
        cache_ttl_sec: float = 300.0,
    ) -> float | None:
        """TTL-cached wrapper over :meth:`get_lp_burn_fraction`.

        Exists so the LP-burn guard can re-evaluate the same candidate on
        back-to-back tick events without re-hammering
        ``getTokenLargestAccounts``.
        """
        now = time.monotonic()
        with self._lp_burn_cache_lock:
            hit = self._lp_burn_cache.get(lp_mint)
            if hit and hit[0] > now:
                return hit[1]
        fraction = self.get_lp_burn_fraction(lp_mint)
        with self._lp_burn_cache_lock:
            self._lp_burn_cache[lp_mint] = (now + float(cache_ttl_sec), fraction)
        return fraction

    def get_lp_burn_fraction(
        self,
        lp_mint: str,
        *,
        cache_ttl_sec: float = 300.0,
    ) -> float | None:
        """Return fraction of LP supply held at burn-like addresses (0..1).

        Used by the LP-burned entry guard. Returns None when we can't decide
        (zero supply, RPC failure). LP mints for Pump-AMM pools live at bytes
        [107:139] of the pool state account; resolving them is the caller's
        job — this helper just scores once you have a candidate mint.
        """
        supply_raw, _decimals = self.get_token_supply_raw(lp_mint)
        if supply_raw <= 0:
            # Zero supply = LP fully burned via mint shutdown → treat as safe.
            return 1.0
        largest = self.get_token_largest_accounts(lp_mint)
        if not largest:
            return None
        top_pubkeys = [pubkey for pubkey, _ in largest]
        owners_map = self.get_token_account_owners(top_pubkeys)
        burned_raw = 0
        for pubkey, amount in largest:
            owner = owners_map.get(pubkey, "")
            if owner and owner in self._BURN_LIKE_OWNERS:
                burned_raw += amount
        return float(burned_raw) / float(supply_raw) if supply_raw > 0 else None

    def get_owner_token_balance_raw(
        self,
        owner_pubkey: str,
        token_mint: str,
        *,
        commitment: str = "processed",
    ) -> tuple[int, int]:
        data = self._rpc_call(
            "getTokenAccountsByOwner",
            [
                str(owner_pubkey),
                {"mint": str(token_mint)},
                {
                    "encoding": "jsonParsed",
                    "commitment": str(commitment or "processed"),
                },
            ],
        )
        entries = list(((data.get("result") or {}).get("value")) or [])
        return self._sum_token_accounts_raw(entries)

    async def get_balance_async(self, pubkey: str, *, commitment: str = "processed") -> int:
        """Async variant of :meth:`get_balance`."""
        data = await self._rpc_call_async(
            "getBalance",
            [
                str(pubkey),
                {
                    "commitment": str(commitment or "processed"),
                },
            ],
        )
        value = (data.get("result") or {}).get("value")
        return int(value or 0)

    async def get_owner_token_balance_raw_async(
        self,
        owner_pubkey: str,
        token_mint: str,
        *,
        commitment: str = "processed",
    ) -> tuple[int, int]:
        data = await self._rpc_call_async(
            "getTokenAccountsByOwner",
            [
                str(owner_pubkey),
                {"mint": str(token_mint)},
                {
                    "encoding": "jsonParsed",
                    "commitment": str(commitment or "processed"),
                },
            ],
        )
        entries = list(((data.get("result") or {}).get("value")) or [])
        return self._sum_token_accounts_raw(entries)

    def get_latest_blockhash(self) -> tuple[str, int]:
        """Fetch a recent blockhash with its last valid block height."""
        data = self._rpc_call("getLatestBlockhash", [{"commitment": "confirmed"}])
        value = (data.get("result") or {}).get("value") or {}
        blockhash = str(value.get("blockhash") or "")
        last_valid = int(value.get("lastValidBlockHeight") or 0)
        if not blockhash:
            raise BroadcastError(f"getLatestBlockhash returned no blockhash: {data}")
        return blockhash, last_valid

    async def get_latest_blockhash_async(self) -> tuple[str, int]:
        """Async variant of :meth:`get_latest_blockhash`."""
        data = await self._rpc_call_async("getLatestBlockhash", [{"commitment": "confirmed"}])
        value = (data.get("result") or {}).get("value") or {}
        blockhash = str(value.get("blockhash") or "")
        last_valid = int(value.get("lastValidBlockHeight") or 0)
        if not blockhash:
            raise BroadcastError(f"getLatestBlockhash returned no blockhash: {data}")
        return blockhash, last_valid

    @staticmethod
    def _decode_account_data_base64(account_info: dict[str, Any]) -> bytes:
        data_field = account_info.get("data")
        if isinstance(data_field, list) and data_field:
            return base64.b64decode(str(data_field[0]))
        if isinstance(data_field, str):
            return base64.b64decode(data_field)
        raise BroadcastError(f"unsupported account data format: {account_info}")

    @staticmethod
    def _alt_from_account_info(address: str, account_info: dict[str, Any]):
        from solders.address_lookup_table_account import (
            AddressLookupTable,
            AddressLookupTableAccount,
        )  # type: ignore[import-untyped]
        from solders.pubkey import Pubkey  # type: ignore[import-untyped]

        raw_data = Broadcaster._decode_account_data_base64(account_info)
        try:
            # RPC returns raw ALT account data. solders expects that payload to be
            # decoded with `deserialize`, not `from_bytes`.
            table = AddressLookupTable.deserialize(raw_data)
        except Exception as exc:  # noqa: BLE001
            raise BroadcastError(
                f"failed to decode ALT account {address}: {exc} (raw_len={len(raw_data)})"
            ) from exc
        return AddressLookupTableAccount(Pubkey.from_string(address), list(table.addresses))

    def get_address_lookup_table_accounts(
        self, addresses: list[str] | tuple[str, ...]
    ) -> list[Any]:
        """Fetch and cache ALT accounts for VersionedTransaction compilation."""
        keys = [str(item).strip() for item in addresses if str(item).strip()]
        if not keys:
            return []

        cached: list[Any] = []
        missing: list[str] = []
        for key in keys:
            cached_alt = self._alt_cache.get(key)
            if cached_alt is None:
                missing.append(key)
            else:
                cached.append(cached_alt)

        if missing:
            data = self._rpc_call(
                "getMultipleAccounts",
                [missing, {"encoding": "base64", "commitment": "confirmed"}],
            )
            values = list(((data.get("result") or {}).get("value") or []))
            if len(values) != len(missing):
                raise BroadcastError(
                    f"getMultipleAccounts ALT count mismatch: expected={len(missing)} got={len(values)}"
                )
            for address, account_info in zip(missing, values):
                if not isinstance(account_info, dict):
                    raise BroadcastError(f"ALT account missing for {address}")
                alt_account = self._alt_from_account_info(address, account_info)
                self._alt_cache[address] = alt_account
                cached.append(alt_account)

        return [self._alt_cache[key] for key in keys]

    async def get_address_lookup_table_accounts_async(
        self, addresses: list[str] | tuple[str, ...]
    ) -> list[Any]:
        """Async variant of :meth:`get_address_lookup_table_accounts`."""
        keys = [str(item).strip() for item in addresses if str(item).strip()]
        if not keys:
            return []

        missing = [key for key in keys if key not in self._alt_cache]
        if missing:
            data = await self._rpc_call_async(
                "getMultipleAccounts",
                [missing, {"encoding": "base64", "commitment": "confirmed"}],
            )
            values = list(((data.get("result") or {}).get("value") or []))
            if len(values) != len(missing):
                raise BroadcastError(
                    f"getMultipleAccounts ALT count mismatch: expected={len(missing)} got={len(values)}"
                )
            for address, account_info in zip(missing, values):
                if not isinstance(account_info, dict):
                    raise BroadcastError(f"ALT account missing for {address}")
                self._alt_cache[address] = self._alt_from_account_info(address, account_info)

        return [self._alt_cache[key] for key in keys]

    # ------------------------------------------------------------------
    # Priority fee estimate
    # ------------------------------------------------------------------

    def get_priority_fee_estimate(self, account_keys: list[str] | None = None) -> dict[str, Any]:
        """Estimate priority/Jito fees for the current network conditions."""
        scoped_account_keys = [str(key) for key in (account_keys or []) if str(key)]
        use_cache = not scoped_account_keys
        now = time.monotonic()
        if (
            use_cache
            and self._fee_cache is not None
            and now - self._fee_cache_ts < self._fee_cache_ttl_sec
        ):
            return self._fee_cache

        params: list[Any] = [
            {
                "accountKeys": scoped_account_keys,
                "options": {"recommended": True},
            }
        ]
        data = self._rpc_call("getPriorityFeeEstimate", params)
        result = data.get("result")

        priority_fee = 0
        jito_tip = 0
        if isinstance(result, dict):
            for key in (
                "priorityFeeEstimate",
                "priority_fee_estimate",
                "recommended",
                "recommendedPriorityFee",
                "priorityFee",
            ):
                value = result.get(key)
                if isinstance(value, (int, float)):
                    priority_fee = int(value)
                    break

            for key in (
                "jitoTipEstimate",
                "recommendedJitoTipLamports",
                "jito_tip_lamports",
                "jitoTipLamports",
            ):
                value = result.get(key)
                if isinstance(value, (int, float)):
                    jito_tip = int(value)
                    break
        elif isinstance(result, (int, float)):
            priority_fee = int(result)

        fee_result = {
            "priority_fee_lamports": max(priority_fee, 0),
            "jito_tip_lamports": max(jito_tip, 0),
            "raw": result,
        }
        if use_cache:
            self._fee_cache = fee_result
            self._fee_cache_ts = time.monotonic()
        return fee_result

    async def get_priority_fee_estimate_async(
        self, account_keys: list[str] | None = None
    ) -> dict[str, Any]:
        """Async variant of :meth:`get_priority_fee_estimate`."""
        scoped_account_keys = [str(key) for key in (account_keys or []) if str(key)]
        use_cache = not scoped_account_keys
        now = time.monotonic()
        if (
            use_cache
            and self._fee_cache is not None
            and now - self._fee_cache_ts < self._fee_cache_ttl_sec
        ):
            return self._fee_cache

        params: list[Any] = [
            {
                "accountKeys": scoped_account_keys,
                "options": {"recommended": True},
            }
        ]
        data = await self._rpc_call_async("getPriorityFeeEstimate", params)
        result = data.get("result")

        priority_fee = 0
        jito_tip = 0
        if isinstance(result, dict):
            for key in (
                "priorityFeeEstimate",
                "priority_fee_estimate",
                "recommended",
                "recommendedPriorityFee",
                "priorityFee",
            ):
                value = result.get(key)
                if isinstance(value, (int, float)):
                    priority_fee = int(value)
                    break

            for key in (
                "jitoTipEstimate",
                "recommendedJitoTipLamports",
                "jito_tip_lamports",
                "jitoTipLamports",
            ):
                value = result.get(key)
                if isinstance(value, (int, float)):
                    jito_tip = int(value)
                    break
        elif isinstance(result, (int, float)):
            priority_fee = int(result)

        fee_result = {
            "priority_fee_lamports": max(priority_fee, 0),
            "jito_tip_lamports": max(jito_tip, 0),
            "raw": result,
        }
        if use_cache:
            self._fee_cache = fee_result
            self._fee_cache_ts = time.monotonic()
        return fee_result

    # ------------------------------------------------------------------
    # Signature / status helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_signature(signed_tx: bytes) -> str:
        try:
            from solders.transaction import VersionedTransaction  # type: ignore[import-untyped]

            tx = VersionedTransaction.from_bytes(signed_tx)
            signatures = list(getattr(tx, "signatures", []) or [])
            if not signatures:
                raise BroadcastError("signed transaction has no signatures")
            return str(signatures[0])
        except BroadcastError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BroadcastError(f"failed to extract transaction signature: {exc}") from exc

    def _validate_low_latency_transaction(self, signed_tx: bytes) -> dict[str, Any]:
        """Ensure Sender/bundle transactions contain both tip and CU price instructions."""
        try:
            from solders.transaction import VersionedTransaction  # type: ignore[import-untyped]

            tx = VersionedTransaction.from_bytes(signed_tx)
            message = tx.message
            account_keys = [str(key) for key in list(message.account_keys)]
            has_compute_unit_price = False
            tip_account: str | None = None
            tip_lamports = 0
            for ci in list(message.instructions):
                try:
                    program_id = account_keys[int(ci.program_id_index)]
                except Exception:  # noqa: BLE001
                    continue
                data = bytes(ci.data)
                if program_id == _COMPUTE_BUDGET_PROGRAM_ID and data[:1] == b"\x03":
                    has_compute_unit_price = True
                    continue
                if program_id != _SYSTEM_PROGRAM_ID:
                    continue
                if len(data) < 12 or data[:4] != b"\x02\x00\x00\x00":
                    continue
                account_indexes = list(bytes(ci.accounts))
                if len(account_indexes) < 2:
                    continue
                try:
                    dest = account_keys[account_indexes[1]]
                except Exception:  # noqa: BLE001
                    continue
                if self.jito_tip_accounts and dest not in self.jito_tip_accounts:
                    continue
                lamports = int.from_bytes(data[4:12], "little", signed=False)
                if lamports > tip_lamports:
                    tip_lamports = lamports
                    tip_account = dest
            min_tip_lamports = 200_000
            if self.broadcast_mode == "helius_sender_swqos":
                min_tip_lamports = 5_000
            if not has_compute_unit_price:
                raise BroadcastError(
                    "sender transaction missing ComputeBudget setComputeUnitPrice instruction"
                )
            if tip_account is None or tip_lamports < min_tip_lamports:
                raise BroadcastError(
                    f"sender transaction missing valid Jito tip transfer >= {min_tip_lamports} lamports"
                )
            return {
                "tip_account": tip_account,
                "tip_lamports": tip_lamports,
                "has_compute_unit_price": has_compute_unit_price,
            }
        except BroadcastError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BroadcastError(f"failed to validate sender transaction: {exc}") from exc

    def _check_signature_status_once(self, signature: str) -> dict[str, Any] | None:
        data = self._rpc_call(
            "getSignatureStatuses",
            [[signature], {"searchTransactionHistory": True}],
        )
        statuses = data.get("result", {}).get("value", [])
        if statuses and statuses[0] is not None:
            return statuses[0]
        return None

    async def _check_signature_status_once_async(self, signature: str) -> dict[str, Any] | None:
        data = await self._rpc_call_async(
            "getSignatureStatuses",
            [[signature], {"searchTransactionHistory": True}],
        )
        statuses = data.get("result", {}).get("value", [])
        if statuses and statuses[0] is not None:
            return statuses[0]
        return None

    def _current_block_height(self) -> int:
        data = self._rpc_call("getBlockHeight", [])
        return int(data.get("result", 0) or 0)

    async def _current_block_height_async(self) -> int:
        data = await self._rpc_call_async("getBlockHeight", [])
        return int(data.get("result", 0) or 0)

    def simulate_transaction(
        self,
        signed_tx: bytes,
        *,
        commitment: str = "confirmed",
    ) -> dict[str, Any]:
        """Simulate a fully-signed TX and return the RPC value block.

        Used as an optional preflight before a broadcast so the caller can
        decide whether to pay the priority fee + jito tip on a TX that will
        definitely revert. Callers should inspect ``err`` (None = success) and
        ``logs`` for the underlying reason.
        """
        encoded = base64.b64encode(signed_tx).decode("ascii")
        params: list[Any] = [
            encoded,
            {
                "encoding": "base64",
                "commitment": str(commitment or "confirmed"),
                "sigVerify": False,
                "replaceRecentBlockhash": False,
            },
        ]
        data = self._rpc_call("simulateTransaction", params, url=self.rpc_url)
        value = ((data.get("result") or {}).get("value")) or {}
        return value

    async def simulate_transaction_async(
        self,
        signed_tx: bytes,
        *,
        commitment: str = "confirmed",
    ) -> dict[str, Any]:
        encoded = base64.b64encode(signed_tx).decode("ascii")
        params: list[Any] = [
            encoded,
            {
                "encoding": "base64",
                "commitment": str(commitment or "confirmed"),
                "sigVerify": False,
                "replaceRecentBlockhash": False,
            },
        ]
        data = await self._rpc_call_async("simulateTransaction", params, url=self.rpc_url)
        value = ((data.get("result") or {}).get("value")) or {}
        return value

    def simulate_bundle(
        self,
        signed_txs: list[bytes],
        *,
        pre_execution_accounts: list[list[str]] | None = None,
    ) -> dict[str, Any]:
        """Simulate an ordered list of TXs as an atomic Jito-style bundle.

        Used by the honeypot entry guard to verify that a buy followed by an
        immediate sell would succeed end-to-end on-chain — catches rugs that
        let you buy but reject the sell.

        Returns the RPC value block verbatim. A successful simulation has
        ``summary == "succeeded"`` (or no ``transactionResults[i].err``
        entries). Callers should inspect both the overall summary and every
        per-tx result.

        Not all RPC providers expose ``simulateBundle``. When it's not
        supported the RPC returns a method-not-found error — surfaced as an
        exception. Callers should catch and fail open (log + allow) rather
        than treating an RPC limitation as a rug signal.
        """
        encoded = [base64.b64encode(tx).decode("ascii") for tx in signed_txs]
        bundle: dict[str, Any] = {
            "encodedTransactions": encoded,
        }
        cfg: dict[str, Any] = {
            "transactionEncoding": "base64",
            "skipSigVerify": True,
            "replaceRecentBlockhash": True,
        }
        if pre_execution_accounts:
            cfg["preExecutionAccountsConfigs"] = [
                {"addresses": accs, "encoding": "base64"} for accs in pre_execution_accounts
            ]
        # Jito/Helius simulateBundle expects a single struct arg with
        # {encodedTransactions, ...config}. Sending [encoded, cfg] yields
        # -32602 "invalid type: string" because the deserializer tries to
        # decode each base64 tx as a struct.
        params: list[Any] = [{**bundle, **cfg}]
        data = self._rpc_call("simulateBundle", params, url=self.rpc_url)
        value = ((data.get("result") or {}).get("value")) or {}
        return value

    # ------------------------------------------------------------------
    # Low-level send-once helpers
    # ------------------------------------------------------------------

    def _send_transaction_rpc_once(self, signed_tx: bytes, *, skip_preflight: bool = True) -> str:
        encoded = base64.b64encode(signed_tx).decode("ascii")
        params: list[Any] = [
            encoded,
            {
                "encoding": "base64",
                "skipPreflight": skip_preflight,
                "preflightCommitment": "processed",
                "maxRetries": 0,
            },
        ]
        data = self._rpc_call("sendTransaction", params, url=self.rpc_url)
        signature = data.get("result")
        if not signature:
            raise BroadcastError(f"sendTransaction returned no signature: {data}")
        logger.info("Transaction sent via staked_rpc: %s", signature)
        return str(signature)

    async def _send_transaction_rpc_once_async(
        self, signed_tx: bytes, *, skip_preflight: bool = True
    ) -> str:
        encoded = base64.b64encode(signed_tx).decode("ascii")
        params: list[Any] = [
            encoded,
            {
                "encoding": "base64",
                "skipPreflight": skip_preflight,
                "preflightCommitment": "processed",
                "maxRetries": 0,
            },
        ]
        data = await self._rpc_call_async("sendTransaction", params, url=self.rpc_url)
        signature = data.get("result")
        if not signature:
            raise BroadcastError(f"sendTransaction returned no signature: {data}")
        logger.info("Transaction sent via staked_rpc: %s", signature)
        return str(signature)

    def _send_transaction_sender_once(self, signed_tx: bytes) -> str:
        encoded = base64.b64encode(signed_tx).decode("ascii")
        params: list[Any] = [
            encoded,
            {
                "encoding": "base64",
                "skipPreflight": True,
                "maxRetries": 0,
            },
        ]
        data = self._rpc_call("sendTransaction", params, url=self._sender_send_url())
        signature = data.get("result")
        if not signature:
            raise BroadcastError(f"Sender sendTransaction returned no signature: {data}")
        logger.info("Transaction sent via helius_sender: %s", signature)
        return str(signature)

    async def _send_transaction_sender_once_async(self, signed_tx: bytes) -> str:
        encoded = base64.b64encode(signed_tx).decode("ascii")
        params: list[Any] = [
            encoded,
            {
                "encoding": "base64",
                "skipPreflight": True,
                "maxRetries": 0,
            },
        ]
        data = await self._rpc_call_async("sendTransaction", params, url=self._sender_send_url())
        signature = data.get("result")
        if not signature:
            raise BroadcastError(f"Sender sendTransaction returned no signature: {data}")
        logger.info("Transaction sent via helius_sender: %s", signature)
        return str(signature)

    def _send_bundle_once(self, signed_tx: bytes) -> str:
        encoded = base64.b64encode(signed_tx).decode("ascii")
        params: list[Any] = [
            [encoded],
            {
                "encoding": "base64",
            },
        ]
        data = self._rpc_call("sendBundle", params, url=self.bundle_url)
        bundle_id = data.get("result")
        if not bundle_id:
            raise BroadcastError(f"sendBundle returned no bundle id: {data}")
        logger.info("Bundle sent via helius_bundle: %s", bundle_id)
        return str(bundle_id)

    async def _send_bundle_once_async(self, signed_tx: bytes) -> str:
        encoded = base64.b64encode(signed_tx).decode("ascii")
        params: list[Any] = [
            [encoded],
            {
                "encoding": "base64",
            },
        ]
        data = await self._rpc_call_async("sendBundle", params, url=self.bundle_url)
        bundle_id = data.get("result")
        if not bundle_id:
            raise BroadcastError(f"sendBundle returned no bundle id: {data}")
        logger.info("Bundle sent via helius_bundle: %s", bundle_id)
        return str(bundle_id)

    # ------------------------------------------------------------------
    # Confirm-only polling helpers
    # ------------------------------------------------------------------

    def confirm_transaction(
        self,
        signature: str,
        last_valid_block_height: int = 0,
        max_polls: int = _MAX_CONFIRM_POLLS,
    ) -> BroadcastResult:
        """Poll for transaction confirmation."""
        for attempt in range(max_polls):
            try:
                status = self._check_signature_status_once(signature)
                if status is not None:
                    err = status.get("err")
                    if err:
                        return BroadcastResult(
                            signature=signature,
                            confirmed=False,
                            slot=status.get("slot"),
                            error=str(err),
                        )
                    commitment = status.get("confirmationStatus", "")
                    if commitment in ("confirmed", "finalized"):
                        return BroadcastResult(
                            signature=signature, confirmed=True, slot=status.get("slot")
                        )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Confirmation poll %s failed: %s", attempt + 1, exc)

            if last_valid_block_height > 0 and attempt % 5 == 0:
                try:
                    current_bh = self._current_block_height()
                    if current_bh > last_valid_block_height:
                        return BroadcastResult(
                            signature=signature,
                            confirmed=False,
                            error="block_height_exceeded",
                        )
                except Exception:  # noqa: BLE001
                    pass

            time.sleep(self._confirm_poll_interval_sec)

        return BroadcastResult(signature=signature, confirmed=False, error="timeout")

    async def confirm_transaction_async(
        self,
        signature: str,
        last_valid_block_height: int = 0,
        max_polls: int = _MAX_CONFIRM_POLLS,
    ) -> BroadcastResult:
        """Poll for transaction confirmation without blocking the event loop."""
        for attempt in range(max_polls):
            try:
                status = await self._check_signature_status_once_async(signature)
                if status is not None:
                    err = status.get("err")
                    if err:
                        return BroadcastResult(
                            signature=signature,
                            confirmed=False,
                            slot=status.get("slot"),
                            error=str(err),
                        )
                    commitment = status.get("confirmationStatus", "")
                    if commitment in ("confirmed", "finalized"):
                        return BroadcastResult(
                            signature=signature, confirmed=True, slot=status.get("slot")
                        )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Async confirmation poll %s failed: %s", attempt + 1, exc)

            if last_valid_block_height > 0 and attempt % 5 == 0:
                try:
                    current_bh = await self._current_block_height_async()
                    if current_bh > last_valid_block_height:
                        return BroadcastResult(
                            signature=signature,
                            confirmed=False,
                            error="block_height_exceeded",
                        )
                except Exception:  # noqa: BLE001
                    pass

            await asyncio.sleep(self._confirm_poll_interval_sec)

        return BroadcastResult(signature=signature, confirmed=False, error="timeout")

    # ------------------------------------------------------------------
    # Combined send + confirm with explicit rebroadcast
    # ------------------------------------------------------------------

    def _send_once(self, signed_tx: bytes) -> tuple[str, str | None]:
        if self.broadcast_mode == "staked_rpc":
            return self._send_transaction_rpc_once(signed_tx), None
        if self.broadcast_mode in {"helius_sender", "helius_sender_swqos"}:
            self._maybe_warm_sender()
            return self._send_transaction_sender_once(signed_tx), None
        if self.broadcast_mode == "helius_bundle":
            self._maybe_warm_sender()
            signature = self._extract_signature(signed_tx)
            bundle_id = self._send_bundle_once(signed_tx)
            return signature, bundle_id
        raise BroadcastError(f"Unsupported broadcast_mode={self.broadcast_mode}")

    async def _send_once_async(self, signed_tx: bytes) -> tuple[str, str | None]:
        if self.broadcast_mode == "staked_rpc":
            return await self._send_transaction_rpc_once_async(signed_tx), None
        if self.broadcast_mode in {"helius_sender", "helius_sender_swqos"}:
            await self._maybe_warm_sender_async()
            return await self._send_transaction_sender_once_async(signed_tx), None
        if self.broadcast_mode == "helius_bundle":
            await self._maybe_warm_sender_async()
            signature = self._extract_signature(signed_tx)
            bundle_id = await self._send_bundle_once_async(signed_tx)
            return signature, bundle_id
        raise BroadcastError(f"Unsupported broadcast_mode={self.broadcast_mode}")

    def broadcast(self, signed_tx: bytes, last_valid_block_height: int = 0) -> BroadcastResult:
        """Send a transaction and wait for confirmation with explicit rebroadcast."""
        total_started = time.monotonic()
        send_latency_ms = 0.0
        sent_at: str | None = None
        confirmed_at: str | None = None
        signature = ""
        bundle_id: str | None = None
        attempts = 0
        last_error: str | None = None
        next_send_monotonic = total_started
        confirm_started: float | None = None
        tx_validation: dict[str, Any] | None = None
        if self.broadcast_mode in {
            "helius_sender",
            "helius_sender_swqos",
            "helius_bundle",
        }:
            tx_validation = self._validate_low_latency_transaction(signed_tx)
            logger.info(
                "Low-latency tx validated: mode=%s tip_account=%s tip_lamports=%d cu_price=%s",
                self.broadcast_mode,
                tx_validation.get("tip_account"),
                int(tx_validation.get("tip_lamports", 0) or 0),
                bool(tx_validation.get("has_compute_unit_price", False)),
            )
        validation_kwargs = {
            "validated_tip_account": str(tx_validation.get("tip_account"))
            if tx_validation
            else None,
            "validated_tip_lamports": int(tx_validation.get("tip_lamports", 0) or 0)
            if tx_validation
            else 0,
            "validated_has_compute_unit_price": bool(
                tx_validation.get("has_compute_unit_price", False)
            )
            if tx_validation
            else False,
        }

        for poll_count in range(_MAX_CONFIRM_POLLS):
            now = time.monotonic()
            should_send = attempts == 0 or (
                attempts < self._max_rebroadcast_attempts and now >= next_send_monotonic
            )
            if should_send:
                attempts += 1
                attempt_started = time.monotonic()
                try:
                    signature, maybe_bundle_id = self._send_once(signed_tx)
                    if maybe_bundle_id:
                        bundle_id = maybe_bundle_id
                    self._mark_sender_activity()
                    if sent_at is None:
                        sent_at = datetime.now(tz=timezone.utc).isoformat()
                        send_latency_ms = (time.monotonic() - attempt_started) * 1000.0
                        confirm_started = time.monotonic()
                    next_send_monotonic = time.monotonic() + self._rebroadcast_interval_sec
                except BroadcastError as exc:
                    last_error = str(exc)
                    if sent_at is None:
                        sent_at = datetime.now(tz=timezone.utc).isoformat()
                        send_latency_ms = (time.monotonic() - attempt_started) * 1000.0
                        confirm_started = time.monotonic()
                    next_send_monotonic = time.monotonic() + self._rebroadcast_interval_sec
                    logger.debug("Broadcast attempt %s failed (%s)", attempts, exc)

            if signature:
                try:
                    status = self._check_signature_status_once(signature)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Status check failed for %s: %s", signature, exc)
                    status = None
                if status is not None:
                    err = status.get("err")
                    if err:
                        total_latency_ms = (time.monotonic() - total_started) * 1000.0
                        confirm_latency_ms = (
                            (time.monotonic() - confirm_started) * 1000.0
                            if confirm_started is not None
                            else 0.0
                        )
                        return BroadcastResult(
                            signature=signature,
                            confirmed=False,
                            slot=status.get("slot"),
                            error=str(err),
                            send_latency_ms=send_latency_ms,
                            confirm_latency_ms=confirm_latency_ms,
                            total_latency_ms=total_latency_ms,
                            sent_at=sent_at,
                            confirmed_at=None,
                            transport=self.broadcast_mode,
                            bundle_id=bundle_id,
                            send_attempts=attempts,
                            **validation_kwargs,
                        )
                    if status.get("confirmationStatus", "") in (
                        "confirmed",
                        "finalized",
                    ):
                        confirmed_at = datetime.now(tz=timezone.utc).isoformat()
                        total_latency_ms = (time.monotonic() - total_started) * 1000.0
                        confirm_latency_ms = (
                            (time.monotonic() - confirm_started) * 1000.0
                            if confirm_started is not None
                            else 0.0
                        )
                        return BroadcastResult(
                            signature=signature,
                            confirmed=True,
                            slot=status.get("slot"),
                            error=None,
                            send_latency_ms=send_latency_ms,
                            confirm_latency_ms=confirm_latency_ms,
                            total_latency_ms=total_latency_ms,
                            sent_at=sent_at,
                            confirmed_at=confirmed_at,
                            transport=self.broadcast_mode,
                            bundle_id=bundle_id,
                            send_attempts=attempts,
                            **validation_kwargs,
                        )

            if last_valid_block_height > 0 and poll_count % 5 == 0:
                try:
                    current_bh = self._current_block_height()
                    if current_bh > last_valid_block_height:
                        total_latency_ms = (time.monotonic() - total_started) * 1000.0
                        confirm_latency_ms = (
                            (time.monotonic() - confirm_started) * 1000.0
                            if confirm_started is not None
                            else 0.0
                        )
                        return BroadcastResult(
                            signature=signature,
                            confirmed=False,
                            error="block_height_exceeded",
                            send_latency_ms=send_latency_ms,
                            confirm_latency_ms=confirm_latency_ms,
                            total_latency_ms=total_latency_ms,
                            sent_at=sent_at,
                            confirmed_at=None,
                            transport=self.broadcast_mode,
                            bundle_id=bundle_id,
                            send_attempts=attempts,
                            **validation_kwargs,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Block-height check failed: %s", exc)

            time.sleep(self._confirm_poll_interval_sec)

        total_latency_ms = (time.monotonic() - total_started) * 1000.0
        confirm_latency_ms = (
            (time.monotonic() - confirm_started) * 1000.0 if confirm_started is not None else 0.0
        )
        return BroadcastResult(
            signature=signature,
            confirmed=False,
            error=last_error or "timeout",
            send_latency_ms=send_latency_ms,
            confirm_latency_ms=confirm_latency_ms,
            total_latency_ms=total_latency_ms,
            sent_at=sent_at,
            confirmed_at=None,
            transport=self.broadcast_mode,
            bundle_id=bundle_id,
            send_attempts=attempts,
            **validation_kwargs,
        )

    async def broadcast_async(
        self,
        signed_tx: bytes,
        last_valid_block_height: int = 0,
    ) -> BroadcastResult:
        """Send and confirm a transaction without blocking the event loop."""
        total_started = time.monotonic()
        send_latency_ms = 0.0
        sent_at: str | None = None
        confirmed_at: str | None = None
        signature = ""
        bundle_id: str | None = None
        attempts = 0
        last_error: str | None = None
        next_send_monotonic = total_started
        confirm_started: float | None = None
        tx_validation: dict[str, Any] | None = None
        if self.broadcast_mode in {
            "helius_sender",
            "helius_sender_swqos",
            "helius_bundle",
        }:
            tx_validation = self._validate_low_latency_transaction(signed_tx)
            logger.info(
                "Low-latency tx validated: mode=%s tip_account=%s tip_lamports=%d cu_price=%s",
                self.broadcast_mode,
                tx_validation.get("tip_account"),
                int(tx_validation.get("tip_lamports", 0) or 0),
                bool(tx_validation.get("has_compute_unit_price", False)),
            )
        validation_kwargs = {
            "validated_tip_account": str(tx_validation.get("tip_account"))
            if tx_validation
            else None,
            "validated_tip_lamports": int(tx_validation.get("tip_lamports", 0) or 0)
            if tx_validation
            else 0,
            "validated_has_compute_unit_price": bool(
                tx_validation.get("has_compute_unit_price", False)
            )
            if tx_validation
            else False,
        }

        for poll_count in range(_MAX_CONFIRM_POLLS):
            now = time.monotonic()
            should_send = attempts == 0 or (
                attempts < self._max_rebroadcast_attempts and now >= next_send_monotonic
            )
            if should_send:
                attempts += 1
                attempt_started = time.monotonic()
                try:
                    signature, maybe_bundle_id = await self._send_once_async(signed_tx)
                    if maybe_bundle_id:
                        bundle_id = maybe_bundle_id
                    self._mark_sender_activity()
                    if sent_at is None:
                        sent_at = datetime.now(tz=timezone.utc).isoformat()
                        send_latency_ms = (time.monotonic() - attempt_started) * 1000.0
                        confirm_started = time.monotonic()
                    next_send_monotonic = time.monotonic() + self._rebroadcast_interval_sec
                except BroadcastError as exc:
                    last_error = str(exc)
                    if sent_at is None:
                        sent_at = datetime.now(tz=timezone.utc).isoformat()
                        send_latency_ms = (time.monotonic() - attempt_started) * 1000.0
                        confirm_started = time.monotonic()
                    next_send_monotonic = time.monotonic() + self._rebroadcast_interval_sec
                    logger.debug("Async broadcast attempt %s failed (%s)", attempts, exc)

            if signature:
                try:
                    status = await self._check_signature_status_once_async(signature)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Async status check failed for %s: %s", signature, exc)
                    status = None
                if status is not None:
                    err = status.get("err")
                    if err:
                        total_latency_ms = (time.monotonic() - total_started) * 1000.0
                        confirm_latency_ms = (
                            (time.monotonic() - confirm_started) * 1000.0
                            if confirm_started is not None
                            else 0.0
                        )
                        return BroadcastResult(
                            signature=signature,
                            confirmed=False,
                            slot=status.get("slot"),
                            error=str(err),
                            send_latency_ms=send_latency_ms,
                            confirm_latency_ms=confirm_latency_ms,
                            total_latency_ms=total_latency_ms,
                            sent_at=sent_at,
                            confirmed_at=None,
                            transport=self.broadcast_mode,
                            bundle_id=bundle_id,
                            send_attempts=attempts,
                            **validation_kwargs,
                        )
                    if status.get("confirmationStatus", "") in (
                        "confirmed",
                        "finalized",
                    ):
                        confirmed_at = datetime.now(tz=timezone.utc).isoformat()
                        total_latency_ms = (time.monotonic() - total_started) * 1000.0
                        confirm_latency_ms = (
                            (time.monotonic() - confirm_started) * 1000.0
                            if confirm_started is not None
                            else 0.0
                        )
                        return BroadcastResult(
                            signature=signature,
                            confirmed=True,
                            slot=status.get("slot"),
                            error=None,
                            send_latency_ms=send_latency_ms,
                            confirm_latency_ms=confirm_latency_ms,
                            total_latency_ms=total_latency_ms,
                            sent_at=sent_at,
                            confirmed_at=confirmed_at,
                            transport=self.broadcast_mode,
                            bundle_id=bundle_id,
                            send_attempts=attempts,
                            **validation_kwargs,
                        )

            if last_valid_block_height > 0 and poll_count % 5 == 0:
                try:
                    current_bh = await self._current_block_height_async()
                    if current_bh > last_valid_block_height:
                        total_latency_ms = (time.monotonic() - total_started) * 1000.0
                        confirm_latency_ms = (
                            (time.monotonic() - confirm_started) * 1000.0
                            if confirm_started is not None
                            else 0.0
                        )
                        return BroadcastResult(
                            signature=signature,
                            confirmed=False,
                            error="block_height_exceeded",
                            send_latency_ms=send_latency_ms,
                            confirm_latency_ms=confirm_latency_ms,
                            total_latency_ms=total_latency_ms,
                            sent_at=sent_at,
                            confirmed_at=None,
                            transport=self.broadcast_mode,
                            bundle_id=bundle_id,
                            send_attempts=attempts,
                            **validation_kwargs,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Async block-height check failed: %s", exc)

            await asyncio.sleep(self._confirm_poll_interval_sec)

        total_latency_ms = (time.monotonic() - total_started) * 1000.0
        confirm_latency_ms = (
            (time.monotonic() - confirm_started) * 1000.0 if confirm_started is not None else 0.0
        )
        return BroadcastResult(
            signature=signature,
            confirmed=False,
            error=last_error or "timeout",
            send_latency_ms=send_latency_ms,
            confirm_latency_ms=confirm_latency_ms,
            total_latency_ms=total_latency_ms,
            sent_at=sent_at,
            confirmed_at=None,
            transport=self.broadcast_mode,
            bundle_id=bundle_id,
            send_attempts=attempts,
            **validation_kwargs,
        )
