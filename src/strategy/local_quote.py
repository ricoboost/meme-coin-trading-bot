"""Local PUMP_AMM constant-product quote engine.

Computes sell/buy quotes directly from cached pool vault reserves, bypassing
Jupiter entirely.  Reserves are updated from the transaction metadata of every
PUMP_AMM swap that passes through the Yellowstone gRPC stream — zero extra RPC
calls after the first swap is observed for a token.

Formula (fee-on-input, same as on-chain):
    in_after_fee = amount_in * (10_000 - fee_bps) // 10_000
    out = in_after_fee * reserve_out // (reserve_in + in_after_fee)

Latency: ~0 µs (pure integer arithmetic on cached values).

Usage:
    engine = PumpAMMQuoteEngine()

    # Called from helius_ws.py on every PUMP_AMM swap event:
    engine.update_from_swap_meta(token_mint, post_token_balances, account_keys)

    # Called from exit_engine.py Layer-1 check:
    lamports_out = engine.quote_sell(token_mint, raw_token_amount)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# WSOL mint address (canonical)
_WSOL_MINT = "So11111111111111111111111111111111111111112"

# Fee schedule per source program (basis points, fee-on-input)
_FEE_BPS_BY_SOURCE: dict[str, int] = {
    "PUMP_AMM": 30,  # 25 LP + 5 protocol
    "RAYDIUM_LAUNCHLAB": 25,  # Raydium LaunchLab standard fee
}
_DEFAULT_FEE_BPS: int = 30

# Reserves older than this are treated as stale.
# A single 30-second window covers typical sniper hold times.
_STALE_TTL_SEC: float = 60.0
_PUMP_AMM_PROGRAM_IDS = {
    "term9YPb9mzAsABaqN71A4xdbxHmpBNZavpBiQKZzN3",
    "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA",
}
_PUMP_AMM_BUY_DISC = bytes([102, 6, 61, 18, 1, 218, 235, 234])
_PUMP_AMM_SELL_DISC = bytes([51, 230, 133, 164, 1, 127, 131, 173])


@dataclass(frozen=True)
class PoolReserves:
    """Snapshot of pool reserves extracted from a swap transaction."""

    token_mint: str
    sol_reserve: int  # lamports in the WSOL vault
    token_reserve: int  # raw token units in the token vault
    ts: float  # monotonic timestamp of the last update
    fee_bps: int = _DEFAULT_FEE_BPS  # fee basis points for this pool's source program
    token_decimals: int = 0
    # "vault_match" when pinned via known vault addresses (reliable on any
    # pool shape) or "heuristic_fallback" when derived from the largest-
    # balance heuristic (fragile on fresh post-migration Pump-AMM pools).
    source: str = "heuristic_fallback"


@dataclass(frozen=True)
class PumpAMMNativePoolState:
    """Cached native Pump AMM account map captured from observed swap instructions."""

    token_mint: str
    program_id: str
    pool: str
    global_config: str
    base_mint: str
    quote_mint: str
    pool_base_token_account: str
    pool_quote_token_account: str
    protocol_fee_recipient: str
    protocol_fee_recipient_token_account: str
    base_token_program: str
    quote_token_program: str
    event_authority: str
    coin_creator_vault_ata: str
    coin_creator_vault_authority: str
    fee_config: str
    fee_program: str
    global_volume_accumulator: str | None = None
    ts: float = 0.0
    last_signature: str | None = None
    last_side: str | None = None

    @property
    def token_is_base_quote_is_wsol(self) -> bool:
        return self.base_mint == self.token_mint and self.quote_mint == _WSOL_MINT


class PumpAMMQuoteEngine:
    """Thread-safe, zero-latency quote engine backed by cached pool reserves.

    Reserves are kept fresh by calling ``update_from_swap_meta()`` inside the
    Yellowstone gRPC event loop whenever a PUMP_AMM swap is observed.  For a
    position that is actively traded, the cache is typically <2 s old.

    Pool state account layout (confirmed, 301 bytes):
        [0:8]   discriminator = f19a6d0411b16dbc  (account:Pool)
        [8]     bump (u8)
        [9:11]  index (u16)
        [11:43] creator
        [43:75] base_mint  = token
        [75:107] quote_mint = WSOL
        [107:139] lp_mint
        [139:171] base_vault  = token vault address ← used for reserve lookup
        [171:203] quote_vault = WSOL vault address  ← used for reserve lookup

    Reserve extraction (no pool state read required):
        On each swap transaction, the ``post_token_balances`` metadata contains
        the post-swap balance for every touched account including pool vaults.
        We identify the pool vault as the non-user token account with the
        *largest* balance for the respective mint.
    """

    POOL_DISC = bytes.fromhex("f19a6d0411b16dbc")

    def __init__(self, fee_bps: int = _DEFAULT_FEE_BPS) -> None:
        self._default_fee_bps = fee_bps
        # token_mint → PoolReserves
        self._reserves: dict[str, PoolReserves] = {}
        self._native_pool_state: dict[str, PumpAMMNativePoolState] = {}
        self._lock = threading.Lock()
        self._updates: int = 0
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Reserve update (called from gRPC event loop)
    # ------------------------------------------------------------------

    def update_from_swap_meta(
        self,
        token_mint: str,
        post_token_balances: list,
        triggering_wallet: str,
        source_program: str = "PUMP_AMM",
        account_keys: list[str] | None = None,
        message_instructions: list[Any] | None = None,
        signature: str | None = None,
    ) -> bool:
        """Extract pool reserves from Yellowstone post_token_balances and cache them.

        Parameters
        ----------
        token_mint:
            Mint of the token being swapped (not WSOL).
        post_token_balances:
            The ``meta.post_token_balances`` protobuf list from the transaction.
        triggering_wallet:
            The wallet that initiated the swap — excluded when searching for
            pool vault candidates.
        source_program:
            Source program label (e.g. ``"PUMP_AMM"``, ``"RAYDIUM_LAUNCHLAB"``).
            Used to select the correct fee_bps for the pool.

        Returns
        -------
        bool
            True if reserves were updated successfully.
        """
        now_ts = time.monotonic()
        sol_reserve: int | None = None
        token_reserve: int | None = None
        token_decimals = 0

        # Parse the swap instruction first so we know the authoritative vault
        # account addresses.  This lets us pin reserves to the real pool vaults
        # instead of guessing via "largest balance", which breaks on fresh
        # post-migration Pump-AMM pools where bonding-curve escrow accounts or
        # aggregator intermediates can hold larger balances than the new AMM
        # vaults (observed on token 4SXJJw…pump, session 20260419T103007Z:
        # local quote read 2.3 SOL vs true vault 0.048 SOL — 48x inflation).
        native_state = self._extract_native_pool_state(
            token_mint=token_mint,
            source_program=source_program,
            account_keys=account_keys,
            message_instructions=message_instructions,
            now_ts=now_ts,
            signature=signature,
        )

        # Path A: deterministic vault-address match (preferred when we have
        # native pool state + account_keys).  Looks up each post_token_balance
        # row's account via `account_index → account_keys[idx]` and matches
        # against the pool's known base/quote vault addresses.
        matched_via_vault = False
        if native_state is not None and native_state.token_is_base_quote_is_wsol and account_keys:
            base_vault = native_state.pool_base_token_account
            quote_vault = native_state.pool_quote_token_account
            for row in post_token_balances:
                try:
                    idx = int(getattr(row, "account_index", -1))
                except (TypeError, ValueError):
                    continue
                if idx < 0 or idx >= len(account_keys):
                    continue
                account_address = str(account_keys[idx] or "")
                if account_address not in (base_vault, quote_vault):
                    continue
                ui = getattr(row, "ui_token_amount", None)
                if ui is None:
                    continue
                try:
                    raw_balance = int(getattr(ui, "amount", "") or 0)
                except (ValueError, TypeError):
                    continue
                if raw_balance <= 0:
                    continue
                mint = getattr(row, "mint", "")
                if account_address == quote_vault and mint == _WSOL_MINT:
                    sol_reserve = raw_balance
                elif account_address == base_vault and mint == token_mint:
                    token_reserve = raw_balance
                    try:
                        token_decimals = max(0, int(getattr(ui, "decimals", 0) or 0))
                    except (TypeError, ValueError):
                        token_decimals = 0
            matched_via_vault = sol_reserve is not None and token_reserve is not None

        # Path B: fallback heuristic (largest balance excluding triggering
        # wallet).  Only used when native state is unavailable or vault match
        # failed — typically non-Pump-AMM paths or the very first swap before
        # the native state has been cached.
        if not matched_via_vault:
            wsol_candidates: list[tuple[int, str]] = []
            token_candidates: list[tuple[int, str, int]] = []
            for row in post_token_balances:
                mint = getattr(row, "mint", "")
                owner = getattr(row, "owner", "")
                if not mint or not owner:
                    continue
                if owner == triggering_wallet:
                    continue
                ui = getattr(row, "ui_token_amount", None)
                if ui is None:
                    continue
                try:
                    raw_balance = int(getattr(ui, "amount", "") or 0)
                except (ValueError, TypeError):
                    continue
                if raw_balance <= 0:
                    continue
                if mint == _WSOL_MINT:
                    wsol_candidates.append((raw_balance, owner))
                elif mint == token_mint:
                    try:
                        decimals_int = int(getattr(ui, "decimals", 0) or 0)
                    except (TypeError, ValueError):
                        decimals_int = 0
                    token_candidates.append((raw_balance, owner, max(0, decimals_int)))
            if wsol_candidates and sol_reserve is None:
                sol_reserve = max(wsol_candidates, key=lambda x: x[0])[0]
            if token_candidates and token_reserve is None:
                token_reserve, _, token_decimals = max(token_candidates, key=lambda x: x[0])

        if sol_reserve is None or token_reserve is None:
            return False

        fee_bps = _FEE_BPS_BY_SOURCE.get(source_program, self._default_fee_bps)
        reserves = PoolReserves(
            token_mint=token_mint,
            sol_reserve=sol_reserve,
            token_reserve=token_reserve,
            ts=now_ts,
            fee_bps=fee_bps,
            token_decimals=token_decimals,
            source="vault_match" if matched_via_vault else "heuristic_fallback",
        )
        with self._lock:
            self._reserves[token_mint] = reserves
            if native_state is not None:
                self._native_pool_state[token_mint] = native_state
            self._updates += 1

        logger.debug(
            "local_quote: updated %s sol_reserve=%.4f token_reserve=%d",
            token_mint[:8],
            sol_reserve / 1e9,
            token_reserve,
        )
        return True

    def _extract_native_pool_state(
        self,
        *,
        token_mint: str,
        source_program: str,
        account_keys: list[str] | None,
        message_instructions: list[Any] | None,
        now_ts: float,
        signature: str | None,
    ) -> PumpAMMNativePoolState | None:
        if source_program != "PUMP_AMM":
            return None
        if not account_keys or not message_instructions:
            return None
        for compiled_instruction in message_instructions:
            try:
                program_id_index = int(getattr(compiled_instruction, "program_id_index", -1))
            except (TypeError, ValueError):
                continue
            if program_id_index < 0 or program_id_index >= len(account_keys):
                continue
            program_id = str(account_keys[program_id_index] or "")
            if program_id not in _PUMP_AMM_PROGRAM_IDS:
                continue
            raw_data = bytes(getattr(compiled_instruction, "data", b"") or b"")
            if len(raw_data) < 8:
                continue
            if raw_data[:8] == _PUMP_AMM_BUY_DISC:
                side = "buy"
                required_accounts = 23
            elif raw_data[:8] == _PUMP_AMM_SELL_DISC:
                side = "sell"
                required_accounts = 21
            else:
                continue
            indexes = list(bytes(getattr(compiled_instruction, "accounts", b"") or b""))
            if len(indexes) < required_accounts:
                continue
            ordered_accounts: list[str] = []
            valid = True
            for index in indexes:
                if index < 0 or index >= len(account_keys):
                    valid = False
                    break
                ordered_accounts.append(str(account_keys[index] or ""))
            if not valid or len(ordered_accounts) < required_accounts:
                continue
            base_mint = ordered_accounts[3]
            quote_mint = ordered_accounts[4]
            if token_mint not in {base_mint, quote_mint}:
                continue
            global_volume_accumulator = (
                ordered_accounts[19] if side == "buy" and len(ordered_accounts) >= 20 else None
            )
            return PumpAMMNativePoolState(
                token_mint=token_mint,
                program_id=program_id,
                pool=ordered_accounts[0],
                global_config=ordered_accounts[2],
                base_mint=base_mint,
                quote_mint=quote_mint,
                pool_base_token_account=ordered_accounts[7],
                pool_quote_token_account=ordered_accounts[8],
                protocol_fee_recipient=ordered_accounts[9],
                protocol_fee_recipient_token_account=ordered_accounts[10],
                base_token_program=ordered_accounts[11],
                quote_token_program=ordered_accounts[12],
                event_authority=ordered_accounts[15],
                coin_creator_vault_ata=ordered_accounts[17],
                coin_creator_vault_authority=ordered_accounts[18],
                fee_config=ordered_accounts[21] if side == "buy" else ordered_accounts[19],
                fee_program=ordered_accounts[22] if side == "buy" else ordered_accounts[20],
                global_volume_accumulator=global_volume_accumulator,
                ts=now_ts,
                last_signature=signature,
                last_side=side,
            )
        return None

    def update_reserves_direct(
        self,
        token_mint: str,
        sol_reserve: int,
        token_reserve: int,
        fee_bps: int | None = None,
        token_decimals: int = 0,
    ) -> None:
        """Directly set reserves (for testing or manual override)."""
        reserves = PoolReserves(
            token_mint=token_mint,
            sol_reserve=sol_reserve,
            token_reserve=token_reserve,
            ts=time.monotonic(),
            fee_bps=fee_bps if fee_bps is not None else self._default_fee_bps,
            token_decimals=max(0, int(token_decimals)),
        )
        with self._lock:
            self._reserves[token_mint] = reserves
            self._updates += 1

    # ------------------------------------------------------------------
    # Quote computation
    # ------------------------------------------------------------------

    def quote_sell(
        self,
        token_mint: str,
        token_amount_raw: int,
        stale_ttl_sec: float = _STALE_TTL_SEC,
    ) -> int | None:
        """Compute lamports returned for selling ``token_amount_raw`` tokens.

        Returns ``None`` if no reserves are cached or the cache is stale.
        """
        reserves = self._get_fresh(token_mint, stale_ttl_sec)
        if reserves is None:
            self._misses += 1
            return None
        result = self._cp_out(
            amount_in=token_amount_raw,
            reserve_in=reserves.token_reserve,
            reserve_out=reserves.sol_reserve,
            fee_bps=reserves.fee_bps,
        )
        self._hits += 1
        return result

    def quote_buy(
        self,
        token_mint: str,
        sol_amount_lamports: int,
        stale_ttl_sec: float = _STALE_TTL_SEC,
    ) -> int | None:
        """Compute raw tokens returned for buying with ``sol_amount_lamports``.

        Returns ``None`` if no reserves are cached or the cache is stale.
        """
        reserves = self._get_fresh(token_mint, stale_ttl_sec)
        if reserves is None:
            self._misses += 1
            return None
        result = self._cp_out(
            amount_in=sol_amount_lamports,
            reserve_in=reserves.sol_reserve,
            reserve_out=reserves.token_reserve,
            fee_bps=reserves.fee_bps,
        )
        self._hits += 1
        return result

    def mark_price_sol(
        self, token_mint: str, stale_ttl_sec: float = _STALE_TTL_SEC
    ) -> float | None:
        """Return reserve-implied mid price in SOL per UI token."""
        reserves = self._get_fresh(token_mint, stale_ttl_sec)
        if reserves is None or reserves.token_reserve <= 0:
            return None
        token_units = float(reserves.token_reserve) / float(10 ** max(0, reserves.token_decimals))
        if token_units <= 0:
            return None
        return (float(reserves.sol_reserve) / 1_000_000_000.0) / token_units

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cp_out(
        self,
        amount_in: int,
        reserve_in: int,
        reserve_out: int,
        fee_bps: int | None = None,
    ) -> int:
        """Constant-product AMM output with fee-on-input.

        in_after_fee = amount_in × (10_000 - fee_bps) / 10_000
        out = in_after_fee × reserve_out / (reserve_in + in_after_fee)
        """
        if reserve_in <= 0 or reserve_out <= 0 or amount_in <= 0:
            return 0
        _fee = fee_bps if fee_bps is not None else self._default_fee_bps
        in_after_fee = amount_in * (10_000 - _fee) // 10_000
        return (in_after_fee * reserve_out) // (reserve_in + in_after_fee)

    def _get_fresh(self, token_mint: str, stale_ttl_sec: float) -> PoolReserves | None:
        with self._lock:
            reserves = self._reserves.get(token_mint)
        if reserves is None:
            return None
        if time.monotonic() - reserves.ts > stale_ttl_sec:
            return None
        return reserves

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def has_reserves(self, token_mint: str) -> bool:
        """Return True if fresh reserves are available for this mint."""
        return self._get_fresh(token_mint, _STALE_TTL_SEC) is not None

    def get_reserves(self, token_mint: str) -> PoolReserves | None:
        """Return the cached reserves (or None if missing/stale)."""
        return self._get_fresh(token_mint, _STALE_TTL_SEC)

    def get_native_pool_state(
        self,
        token_mint: str,
        stale_ttl_sec: float = _STALE_TTL_SEC,
    ) -> PumpAMMNativePoolState | None:
        """Return cached native Pump AMM account state for this mint if still fresh."""
        with self._lock:
            state = self._native_pool_state.get(token_mint)
        if state is None:
            return None
        if time.monotonic() - float(state.ts or 0.0) > stale_ttl_sec:
            return None
        return state

    def has_native_pool_state(self, token_mint: str) -> bool:
        """Return whether native Pump AMM account state is cached and fresh."""
        return self.get_native_pool_state(token_mint, _STALE_TTL_SEC) is not None

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "tracked_mints": len(self._reserves),
                "native_tracked_mints": len(self._native_pool_state),
                "updates": self._updates,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }
