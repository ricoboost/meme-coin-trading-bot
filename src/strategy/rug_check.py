"""Async rug-check helper for Solana tokens.

State-free: all checks are read-only Helius RPC + Jupiter quote calls.
Callable from any asyncio context; no executor, signer, or transaction
state required.

Checks performed (all best-effort — returns structured verdict rather than
raising):

- ``mint_renounced``     mint authority revoked (None)
- ``freeze_renounced``   freeze authority revoked (None)
- ``lp_burned``          best-effort; True when unverifiable on pump-style
                         tokens (no standard LP mint); flags top-1 / top-5
                         non-pool holder concentration for transparency
- ``serial_creator_risk``   always False for now — creator history tracking
                            is out of scope day 1
- ``jupiter_quote_ok``   Jupiter v6 quote succeeds with reasonable price
                         impact for a small probe buy
- ``preflight_sim_ok``   always True for now — atomic buy+sell simulateBundle
                         is signer-coupled and not run by this helper

The ``flags`` list on the result accumulates any quantitative warnings
(e.g. ``"top_holder_pct:34.5"``, ``"freeze_authority_set"``, ``"price_impact:12.3"``).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_SOL_MINT = "So11111111111111111111111111111111111111112"
_BURN_LIKE_OWNERS: frozenset[str] = frozenset(
    {
        "1nc1nerator11111111111111111111111111111111",
        "11111111111111111111111111111111",
    }
)


class RugCheckResult(BaseModel):
    lp_burned: bool
    mint_renounced: bool
    freeze_renounced: bool
    serial_creator_risk: bool
    jupiter_quote_ok: bool
    preflight_sim_ok: bool
    flags: list[str] = []

    @property
    def all_passed(self) -> bool:
        return (
            self.mint_renounced
            and self.freeze_renounced
            and self.lp_burned
            and not self.serial_creator_risk
            and self.jupiter_quote_ok
            and self.preflight_sim_ok
        )


@dataclass
class _CacheEntry:
    expires_at: float
    value: Any


class RugChecker:
    """Async rug-check runner against Helius RPC + Jupiter quote."""

    def __init__(
        self,
        *,
        rpc_url: str,
        jupiter_quote_url: str = "https://lite-api.jup.ag/swap/v1/quote",
        probe_sol_lamports: int = 100_000_000,
        max_top_holder_pct: float = 0.25,
        max_top5_holder_pct: float = 0.40,
        max_price_impact_pct: float = 15.0,
        cache_ttl_sec: float = 60.0,
        http_timeout_sec: float = 6.0,
    ) -> None:
        self._rpc_url = rpc_url
        self._jupiter_quote_url = jupiter_quote_url
        self._probe_sol_lamports = probe_sol_lamports
        self._max_top_holder_pct = max_top_holder_pct
        self._max_top5_holder_pct = max_top5_holder_pct
        self._max_price_impact_pct = max_price_impact_pct
        self._cache_ttl_sec = cache_ttl_sec
        self._client = httpx.AsyncClient(timeout=http_timeout_sec)
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "RugChecker":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def check(
        self,
        mint: str,
        *,
        pool_exclude: tuple[str, ...] = (),
    ) -> RugCheckResult:
        """Run every check concurrently; return the combined verdict."""
        flags: list[str] = []
        authorities_task = asyncio.create_task(self._get_mint_authorities(mint))
        holders_task = asyncio.create_task(
            self._get_holder_concentration(mint, pool_exclude=pool_exclude)
        )
        jupiter_task = asyncio.create_task(self._check_jupiter_quote(mint))
        authorities = await authorities_task
        holders = await holders_task
        jupiter_verdict, jupiter_flags = await jupiter_task

        if authorities is None:
            mint_renounced = False
            freeze_renounced = False
            flags.append("mint_authority_unknown")
        else:
            mint_auth, freeze_auth = authorities
            mint_renounced = mint_auth is None
            freeze_renounced = freeze_auth is None
            if mint_auth is not None:
                flags.append("mint_authority_set")
            if freeze_auth is not None:
                flags.append("freeze_authority_set")

        if holders is None:
            lp_burned = True
            flags.append("holder_data_unknown")
        else:
            top1_pct, top5_pct = holders
            flags.append(f"top_holder_pct:{top1_pct * 100:.1f}")
            flags.append(f"top5_holder_pct:{top5_pct * 100:.1f}")
            lp_burned = (
                top1_pct <= self._max_top_holder_pct and top5_pct <= self._max_top5_holder_pct
            )

        flags.extend(jupiter_flags)
        return RugCheckResult(
            lp_burned=lp_burned,
            mint_renounced=mint_renounced,
            freeze_renounced=freeze_renounced,
            serial_creator_risk=False,
            jupiter_quote_ok=jupiter_verdict,
            preflight_sim_ok=True,
            flags=flags,
        )

    async def _rpc_call(self, method: str, params: list[Any]) -> dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        resp = await self._client.post(self._rpc_url, json=payload)
        resp.raise_for_status()
        return resp.json()

    async def _cached(self, key: str, factory) -> Any:
        now = time.monotonic()
        async with self._lock:
            entry = self._cache.get(key)
            if entry is not None and entry.expires_at > now:
                return entry.value
        value = await factory()
        async with self._lock:
            self._cache[key] = _CacheEntry(now + self._cache_ttl_sec, value)
        return value

    async def _get_mint_authorities(self, mint: str) -> tuple[str | None, str | None] | None:
        async def factory() -> tuple[str | None, str | None] | None:
            try:
                data = await self._rpc_call(
                    "getAccountInfo",
                    [str(mint), {"encoding": "jsonParsed", "commitment": "confirmed"}],
                )
            except Exception as exc:
                logger.debug("getAccountInfo failed for %s: %s", mint[:12], exc)
                return None
            value = ((data.get("result") or {}).get("value")) or {}
            parsed = ((value.get("data") or {}).get("parsed")) or {}
            info = parsed.get("info") or {}
            if "mintAuthority" not in info and "freezeAuthority" not in info:
                return None
            mint_auth = info.get("mintAuthority")
            freeze_auth = info.get("freezeAuthority")
            return (
                str(mint_auth) if mint_auth else None,
                str(freeze_auth) if freeze_auth else None,
            )

        return await self._cached(f"auth:{mint}", factory)

    async def _get_token_supply_raw(self, mint: str) -> tuple[int, int]:
        try:
            data = await self._rpc_call(
                "getTokenSupply",
                [str(mint), {"commitment": "confirmed"}],
            )
        except Exception as exc:
            logger.debug("getTokenSupply failed for %s: %s", mint[:12], exc)
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

    async def _get_token_largest_accounts(self, mint: str) -> list[tuple[str, int]]:
        try:
            data = await self._rpc_call(
                "getTokenLargestAccounts",
                [str(mint), {"commitment": "confirmed"}],
            )
        except Exception as exc:
            logger.debug("getTokenLargestAccounts failed for %s: %s", mint[:12], exc)
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

    async def _get_token_account_owners(self, token_accounts: list[str]) -> dict[str, str]:
        if not token_accounts:
            return {}
        try:
            data = await self._rpc_call(
                "getMultipleAccounts",
                [
                    list(token_accounts),
                    {"encoding": "jsonParsed", "commitment": "confirmed"},
                ],
            )
        except Exception as exc:
            logger.debug(
                "getMultipleAccounts failed for %d accts: %s",
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

    async def _get_holder_concentration(
        self,
        mint: str,
        *,
        pool_exclude: tuple[str, ...] = (),
    ) -> tuple[float, float] | None:
        """(top1_non_pool_pct, top5_non_pool_sum_pct) as fractions 0..1."""

        async def factory() -> tuple[float, float] | None:
            supply_raw, _ = await self._get_token_supply_raw(mint)
            if supply_raw <= 0:
                return None
            largest = await self._get_token_largest_accounts(mint)
            if not largest:
                return None
            owners_map = await self._get_token_account_owners([pubkey for pubkey, _ in largest])
            excluded = set(pool_exclude) | set(_BURN_LIKE_OWNERS)
            top1_pct = 0.0
            top5_sum = 0.0
            picked = 0
            for idx, (pubkey, amount) in enumerate(largest):
                if idx == 0:
                    continue
                owner = owners_map.get(pubkey, "")
                if owner and owner in excluded:
                    continue
                pct = float(amount) / float(supply_raw)
                if pct > top1_pct:
                    top1_pct = pct
                if picked < 5:
                    top5_sum += pct
                    picked += 1
            return (top1_pct, top5_sum)

        return await self._cached(f"holders:{mint}", factory)

    async def _check_jupiter_quote(self, mint: str) -> tuple[bool, list[str]]:
        flags: list[str] = []
        params = {
            "inputMint": _SOL_MINT,
            "outputMint": str(mint),
            "amount": str(int(self._probe_sol_lamports)),
            "slippageBps": "1000",
            "onlyDirectRoutes": "false",
            "swapMode": "ExactIn",
        }
        try:
            resp = await self._client.get(self._jupiter_quote_url, params=params)
        except Exception as exc:
            flags.append(f"jupiter_quote_error:{type(exc).__name__}")
            return False, flags
        if resp.status_code != 200:
            flags.append(f"jupiter_quote_status:{resp.status_code}")
            return False, flags
        try:
            body = resp.json()
        except Exception:
            flags.append("jupiter_quote_unparseable")
            return False, flags
        if not body.get("outAmount"):
            flags.append("jupiter_no_route")
            return False, flags
        try:
            price_impact_pct = float(body.get("priceImpactPct") or 0.0) * 100.0
        except (TypeError, ValueError):
            price_impact_pct = 0.0
        flags.append(f"price_impact:{price_impact_pct:.2f}")
        if price_impact_pct < 0 or price_impact_pct > self._max_price_impact_pct:
            return False, flags
        return True, flags
