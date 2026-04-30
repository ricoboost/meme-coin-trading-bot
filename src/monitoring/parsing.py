"""Shared parsing helpers for wallet-first and pair-first transaction events."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Iterable

import base58
import pandas as pd

from src.bot.models import CandidateEvent

EXCLUDED_MINTS = {
    "So11111111111111111111111111111111111111112",  # Wrapped SOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2tQeMgsQUK91t6bp1KUKf5Qd",  # USDT
}

# Program -> normalized source label used by discovery filters.
PROGRAM_SOURCE_BY_ID: dict[str, str] = {
    # Pump.fun / Pump AMM
    "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P": "PUMP_FUN",
    "term9YPb9mzAsABaqN71A4xdbxHmpBNZavpBiQKZzN3": "PUMP_AMM",
    "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA": "PUMP_AMM",
    # Raydium families
    "LanMV9sAd7wArD4vJFi88GyFnpT5z6YGeFCYcRpEPRQ": "RAYDIUM_LAUNCHLAB",
    "LanMV9sAd7wArD4vJFi2qDdfnVhFxYSUg6eADduJ3uj": "RAYDIUM_LAUNCHLAB",
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "RAYDIUM",
    "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK": "RAYDIUM",
    "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C": "RAYDIUM",
}

# Anchor instruction names used for first-8-byte discriminator matching.
PUMP_ANCHOR_METHODS = {
    "create",
    "create_v2",
    "buy",
    "sell",
}
PUMP_AMM_ANCHOR_METHODS = {
    "swap",
    "buy",
    "sell",
}
RAYDIUM_LAUNCHLAB_ANCHOR_METHODS = {
    "initialize",
    "initialize2",
    "swap",
    "swap_base_input",
    "swap_base_output",
    "buy",
    "sell",
}


def _anchor_discriminator(method: str) -> bytes:
    """Compute Anchor instruction discriminator for `global:<method>`."""
    return hashlib.sha256(f"global:{method}".encode("utf-8")).digest()[:8]


ANCHOR_DISCRIMINATORS: dict[str, dict[bytes, str]] = {
    "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P": {
        _anchor_discriminator(method): method for method in PUMP_ANCHOR_METHODS
    },
    "term9YPb9mzAsABaqN71A4xdbxHmpBNZavpBiQKZzN3": {
        _anchor_discriminator(method): method for method in PUMP_AMM_ANCHOR_METHODS
    },
    "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA": {
        _anchor_discriminator(method): method for method in PUMP_AMM_ANCHOR_METHODS
    },
    "LanMV9sAd7wArD4vJFi88GyFnpT5z6YGeFCYcRpEPRQ": {
        _anchor_discriminator(method): method for method in RAYDIUM_LAUNCHLAB_ANCHOR_METHODS
    },
    "LanMV9sAd7wArD4vJFi2qDdfnVhFxYSUg6eADduJ3uj": {
        _anchor_discriminator(method): method for method in RAYDIUM_LAUNCHLAB_ANCHOR_METHODS
    },
}

SWAP_METHOD_HINTS = {
    "swap",
    "buy",
    "sell",
    "swap_base_input",
    "swap_base_output",
}
LAUNCH_METHOD_HINTS = {
    "create",
    "create_v2",
    "initialize",
    "initialize2",
}


@dataclass
class InstructionSignals:
    """Decoded instruction and log signals for one streamed transaction."""

    sources: set[str] = field(default_factory=set)
    has_swap_signal: bool = False
    has_launch_signal: bool = False
    methods: list[str] = field(default_factory=list)


def _account_key_pubkey(entry: object) -> str | None:
    """Extract pubkey from a jsonParsed account key entry."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        pubkey = entry.get("pubkey")
        if isinstance(pubkey, str) and pubkey:
            return pubkey
    return None


def _account_key_is_signer(entry: object) -> bool:
    """Check whether an account key entry is marked signer."""
    return bool(isinstance(entry, dict) and entry.get("signer") is True)


def _parse_ui_token_amount(row: dict) -> float | None:
    """Extract normalized ui token amount from token balance row."""
    ui_amount = row.get("uiTokenAmount") if isinstance(row, dict) else None
    if isinstance(ui_amount, dict):
        amount_str = ui_amount.get("uiAmountString")
        if amount_str is not None:
            parsed = _to_float(amount_str)
            if parsed is not None:
                return parsed
        parsed = _to_float(ui_amount.get("uiAmount"))
        if parsed is not None:
            return parsed
        raw = _to_float(ui_amount.get("amount"))
        decimals_raw = ui_amount.get("decimals")
        try:
            decimals = int(decimals_raw)
        except (TypeError, ValueError):
            decimals = None
        if raw is not None and decimals is not None and decimals >= 0:
            try:
                return raw / (10**decimals)
            except (OverflowError, ValueError):
                return None
    return _to_float(row.get("tokenAmount") if isinstance(row, dict) else None)


def _token_balances_by_owner(rows: list[dict], owner: str) -> dict[str, float]:
    """Aggregate token balances by mint for one owner."""
    balances: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("owner") != owner:
            continue
        mint = row.get("mint")
        if not isinstance(mint, str) or not mint:
            continue
        amount = _parse_ui_token_amount(row)
        if amount is None:
            continue
        balances[mint] = balances.get(mint, 0.0) + float(amount)
    return balances


def _resolve_source_from_programs(
    program_ids: set[str],
    allowed_sources: set[str] | None = None,
) -> tuple[str | None, bool]:
    """Resolve normalized source label from transaction program IDs."""
    matched = [PROGRAM_SOURCE_BY_ID[pid] for pid in program_ids if pid in PROGRAM_SOURCE_BY_ID]
    if not matched:
        return None, not bool(allowed_sources)
    # Keep deterministic order for mixed-program txs.
    matched = sorted(set(matched))
    if allowed_sources:
        allowed_matched = [source for source in matched if source in allowed_sources]
        if not allowed_matched:
            return None, False
        return allowed_matched[0], True
    return matched[0], True


def _resolve_source_from_labels(
    labels: set[str],
    allowed_sources: set[str] | None = None,
) -> tuple[str | None, bool]:
    """Resolve normalized source label from already-normalized source labels."""
    if not labels:
        return None, not bool(allowed_sources)
    matched = sorted(labels)
    if allowed_sources:
        allowed_matched = [label for label in matched if label in allowed_sources]
        if not allowed_matched:
            return None, False
        return allowed_matched[0], True
    return matched[0], True


def _instruction_program_id(ix: dict, account_keys: list[str]) -> str | None:
    """Resolve program id from one instruction payload."""
    if not isinstance(ix, dict):
        return None
    program_id = ix.get("programId")
    if isinstance(program_id, str) and program_id:
        return program_id
    index = ix.get("programIdIndex")
    if isinstance(index, int) and 0 <= index < len(account_keys):
        return account_keys[index]
    return None


def _extract_message_instructions(message: dict) -> list[dict]:
    """Extract top-level and inner instructions from jsonParsed transaction message/meta."""
    instructions: list[dict] = []
    top = message.get("instructions")
    if isinstance(top, list):
        instructions.extend(ix for ix in top if isinstance(ix, dict))
    return instructions


def _extract_inner_instructions(meta: dict) -> list[dict]:
    """Extract inner instruction payloads from meta.innerInstructions."""
    instructions: list[dict] = []
    inner = meta.get("innerInstructions")
    if not isinstance(inner, list):
        return instructions
    for row in inner:
        if not isinstance(row, dict):
            continue
        nested = row.get("instructions")
        if not isinstance(nested, list):
            continue
        instructions.extend(ix for ix in nested if isinstance(ix, dict))
    return instructions


def _decode_anchor_method(program_id: str, raw_data: object) -> str | None:
    """Decode Anchor method by discriminator when instruction data is base58 bytes."""
    if not isinstance(raw_data, str) or not raw_data:
        return None
    discriminator_map = ANCHOR_DISCRIMINATORS.get(program_id)
    if not discriminator_map:
        return None
    try:
        raw = base58.b58decode(raw_data)
    except Exception:  # noqa: BLE001
        return None
    if len(raw) < 8:
        return None
    return discriminator_map.get(raw[:8])


def _collect_instruction_signals(
    ws_result: dict,
    account_keys: list[str],
) -> InstructionSignals:
    """Collect swap/create signals from instruction payloads and log messages."""
    signals = InstructionSignals()
    tx_outer = ws_result.get("transaction")
    if not isinstance(tx_outer, dict):
        return signals
    meta = tx_outer.get("meta")
    tx_inner = tx_outer.get("transaction")
    if not isinstance(meta, dict) or not isinstance(tx_inner, dict):
        return signals
    message = tx_inner.get("message")
    if not isinstance(message, dict):
        return signals

    instructions = _extract_message_instructions(message) + _extract_inner_instructions(meta)
    for ix in instructions:
        program_id = _instruction_program_id(ix, account_keys)
        if not program_id:
            continue
        source = PROGRAM_SOURCE_BY_ID.get(program_id)
        if source:
            signals.sources.add(source)

        method: str | None = None
        data = ix.get("data")
        method = _decode_anchor_method(program_id, data)
        if not method:
            parsed = ix.get("parsed")
            if isinstance(parsed, dict):
                parsed_type = parsed.get("type")
                if isinstance(parsed_type, str) and parsed_type:
                    method = parsed_type.lower()
            elif isinstance(parsed, str):
                method = parsed.lower()

        if method:
            normalized = method.lower()
            signals.methods.append(normalized)
            if any(hint in normalized for hint in SWAP_METHOD_HINTS):
                signals.has_swap_signal = True
            if any(hint in normalized for hint in LAUNCH_METHOD_HINTS):
                signals.has_launch_signal = True

    log_messages = meta.get("logMessages") or []
    if isinstance(log_messages, list):
        for entry in log_messages:
            if not isinstance(entry, str):
                continue
            lower = entry.lower()
            if "instruction:" not in lower:
                continue
            if "create_v2" in lower or "instruction: create" in lower:
                signals.has_launch_signal = True
                signals.methods.append("log:create")
            if (
                "instruction: buy" in lower
                or "instruction: sell" in lower
                or "instruction: swap" in lower
            ):
                signals.has_swap_signal = True
                signals.methods.append("log:swap")

    return signals


def is_memecoin_candidate_mint(mint: str | None) -> bool:
    """Return whether a mint looks like a tradeable memecoin candidate."""
    if not mint:
        return False
    return mint not in EXCLUDED_MINTS


def resolve_tracked_wallet(tx: dict, tracked_wallets: Iterable[str]) -> str | None:
    """Resolve the most likely tracked wallet associated with a transaction."""
    wallet_set = set(tracked_wallets)
    fee_payer = tx.get("feePayer")
    if fee_payer in wallet_set:
        return str(fee_payer)

    candidates: list[str] = []
    for row in tx.get("tokenTransfers") or []:
        for key in ("fromUserAccount", "toUserAccount"):
            value = row.get(key)
            if value in wallet_set:
                candidates.append(str(value))
    for row in tx.get("nativeTransfers") or []:
        for key in ("fromUserAccount", "toUserAccount"):
            value = row.get(key)
            if value in wallet_set:
                candidates.append(str(value))
    if not candidates:
        return None
    return candidates[0]


def resolve_primary_wallet(tx: dict) -> str | None:
    """Resolve the likely initiating wallet for a parsed swap transaction."""
    fee_payer = tx.get("feePayer")
    if isinstance(fee_payer, str) and fee_payer:
        return fee_payer

    for row in tx.get("nativeTransfers") or []:
        from_account = row.get("fromUserAccount")
        if isinstance(from_account, str) and from_account:
            return from_account

    for row in tx.get("tokenTransfers") or []:
        from_account = row.get("fromUserAccount")
        if isinstance(from_account, str) and from_account:
            return from_account
        to_account = row.get("toUserAccount")
        if isinstance(to_account, str) and to_account:
            return to_account
    return None


def find_tracked_wallets(tx: dict, tracked_wallets: Iterable[str]) -> tuple[str, ...]:
    """Return tracked wallets that appear in the parsed transaction payload."""
    wallet_set = set(tracked_wallets)
    matched: set[str] = set()
    fee_payer = tx.get("feePayer")
    if fee_payer in wallet_set:
        matched.add(str(fee_payer))

    for row in tx.get("tokenTransfers") or []:
        for key in ("fromUserAccount", "toUserAccount"):
            value = row.get(key)
            if value in wallet_set:
                matched.add(str(value))
    for row in tx.get("nativeTransfers") or []:
        for key in ("fromUserAccount", "toUserAccount"):
            value = row.get(key)
            if value in wallet_set:
                matched.add(str(value))
    return tuple(sorted(matched))


def _to_float(value: object) -> float | None:
    """Best-effort conversion to float."""
    try:
        converted = float(pd.to_numeric(value, errors="coerce"))
    except (TypeError, ValueError):
        return None
    if pd.isna(converted):
        return None
    return converted


def _token_amount_from_leg(leg: dict) -> float | None:
    """Extract normalized token amount from a swap leg."""
    raw_amount = leg.get("rawTokenAmount")
    if isinstance(raw_amount, dict):
        token_amount = _to_float(raw_amount.get("tokenAmount"))
        decimals_raw = raw_amount.get("decimals")
        try:
            decimals = int(decimals_raw)
        except (TypeError, ValueError):
            decimals = None
        if token_amount is not None and token_amount > 0:
            if decimals is not None and decimals >= 0:
                try:
                    return token_amount / (10**decimals)
                except (OverflowError, ValueError):
                    return None
            return token_amount

    token_amount = _to_float(leg.get("tokenAmount"))
    if token_amount is None or token_amount <= 0:
        return None
    return token_amount


def extract_swap_buy_from_events(
    tx: dict,
    wallet: str,
    require_pump_suffix: bool = False,
) -> tuple[str, float, float] | None:
    """Extract SOL->token BUY legs directly from `events.swap` when present."""
    events = tx.get("events") or {}
    swap = events.get("swap") if isinstance(events, dict) else None
    if not isinstance(swap, dict):
        return None

    native_input = swap.get("nativeInput")
    native_output = swap.get("nativeOutput")
    native_input_lamports = _to_float(
        native_input.get("amount") if isinstance(native_input, dict) else native_input
    )
    native_output_lamports = _to_float(
        native_output.get("amount") if isinstance(native_output, dict) else native_output
    )
    if native_input_lamports is None or native_input_lamports <= 0:
        return None
    # Ambiguous native in+out payloads are not trusted for entry pricing.
    if native_output_lamports is not None and native_output_lamports > 0:
        return None

    token_outputs = swap.get("tokenOutputs") or []
    if not isinstance(token_outputs, list) or not token_outputs:
        return None

    candidates: list[tuple[dict, float]] = []
    for leg in token_outputs:
        if not isinstance(leg, dict):
            continue
        mint = leg.get("mint")
        if not mint or not is_memecoin_candidate_mint(str(mint)):
            continue
        mint_str = str(mint)
        if require_pump_suffix and not mint_str.lower().endswith("pump"):
            continue
        token_amount = _token_amount_from_leg(leg)
        if token_amount is None or token_amount <= 0:
            continue
        candidates.append((leg, token_amount))
    if not candidates:
        return None

    if not require_pump_suffix:
        pump_candidates = [
            item for item in candidates if str(item[0].get("mint") or "").lower().endswith("pump")
        ]
        if pump_candidates:
            candidates = pump_candidates

    wallet_candidates = [item for item in candidates if item[0].get("userAccount") == wallet]
    if wallet_candidates:
        candidates = wallet_candidates

    chosen_leg, token_amount = max(candidates, key=lambda item: item[1])
    sol_amount = native_input_lamports / 1_000_000_000
    if sol_amount <= 0 or token_amount <= 0:
        return None

    return str(chosen_leg.get("mint") or ""), float(sol_amount), float(token_amount)


def trade_from_tx(wallet: str, tx: dict) -> CandidateEvent | None:
    """Convert a parsed transaction payload into a candidate event when possible."""
    event, _ = classify_trade_from_tx(wallet, tx)
    return event


def classify_trade_from_tx(wallet: str, tx: dict) -> tuple[CandidateEvent | None, str]:
    """Convert a parsed transaction payload into a candidate event and return a drop reason when rejected."""
    tx_type = str(tx.get("type") or "").upper()
    if tx_type != "SWAP":
        return None, "not_swap"

    timestamp = tx.get("timestamp")
    if timestamp is None:
        return None, "missing_timestamp"

    swap_event_buy = extract_swap_buy_from_events(tx=tx, wallet=wallet, require_pump_suffix=False)
    if swap_event_buy is not None:
        token_mint, sol_amount, token_amount = swap_event_buy
        return CandidateEvent(
            token_mint=token_mint,
            signature=tx.get("signature"),
            block_time=datetime.fromtimestamp(int(timestamp), tz=timezone.utc),
            triggering_wallet=wallet,
            side="BUY",
            sol_amount=sol_amount,
            token_amount=token_amount,
            source_program=tx.get("source") or tx.get("type"),
            tracked_wallets=(wallet,),
            discovery_source="wallet_first",
            event_time_source="solana_block_time",
        ), "accepted"

    token_transfers = tx.get("tokenTransfers") or []
    native_transfers = tx.get("nativeTransfers") or []
    if not token_transfers or not native_transfers:
        return None, "missing_transfers"

    token_transfer = select_wallet_token_transfer(wallet, token_transfers)
    if token_transfer is None:
        if any(row.get("mint") in EXCLUDED_MINTS for row in token_transfers if row.get("mint")):
            return None, "excluded_base_mint"
        return None, "no_wallet_token_transfer"

    side = infer_swap_side(wallet, token_transfer, native_transfers)
    if side != "BUY":
        return None, "not_buy"

    native_transfer = select_wallet_native_transfer_for_side(wallet, native_transfers, side)
    if native_transfer is None:
        return None, "no_wallet_native_transfer"
    token_amount = pd.to_numeric(token_transfer.get("tokenAmount"), errors="coerce")
    sol_amount = pd.to_numeric(native_transfer.get("amount"), errors="coerce")
    if pd.isna(token_amount) or pd.isna(sol_amount) or token_amount <= 0 or sol_amount <= 0:
        return None, "invalid_amounts"

    return CandidateEvent(
        token_mint=token_transfer.get("mint"),
        signature=tx.get("signature"),
        block_time=datetime.fromtimestamp(int(timestamp), tz=timezone.utc),
        triggering_wallet=wallet,
        side=side,
        sol_amount=float(sol_amount) / 1_000_000_000,
        token_amount=float(token_amount),
        source_program=tx.get("source") or tx.get("type"),
        tracked_wallets=(wallet,),
        discovery_source="wallet_first",
        event_time_source="solana_block_time",
    ), "accepted"


def select_wallet_token_transfer(wallet: str, token_transfers: list[dict]) -> dict | None:
    """Pick the most wallet-relevant non-base token transfer."""
    candidates = [
        row
        for row in token_transfers
        if row.get("mint")
        and is_memecoin_candidate_mint(row.get("mint"))
        and (row.get("fromUserAccount") == wallet or row.get("toUserAccount") == wallet)
    ]
    if not candidates:
        return None

    # Prefer transfers where the tracked wallet is the recipient, since the bot
    # is only interested in new entries.
    buy_candidates = [row for row in candidates if row.get("toUserAccount") == wallet]
    if buy_candidates:
        return max(
            buy_candidates,
            key=lambda row: float(pd.to_numeric(row.get("tokenAmount"), errors="coerce") or 0.0),
        )
    return max(
        candidates,
        key=lambda row: float(pd.to_numeric(row.get("tokenAmount"), errors="coerce") or 0.0),
    )


def select_primary_token_transfer(
    token_transfers: list[dict],
    require_pump_suffix: bool = False,
) -> dict | None:
    """Pick the most meaningful non-base token transfer for pair-first parsing."""
    candidates: list[dict] = []
    for row in token_transfers:
        mint = row.get("mint")
        if not mint or not is_memecoin_candidate_mint(mint):
            continue
        if require_pump_suffix and not str(mint).lower().endswith("pump"):
            continue
        amount = pd.to_numeric(row.get("tokenAmount"), errors="coerce")
        if pd.isna(amount) or amount <= 0:
            continue
        candidates.append(row)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: float(pd.to_numeric(row.get("tokenAmount"), errors="coerce") or 0.0),
    )


def select_pair_first_token_transfer(
    wallet: str,
    token_transfers: list[dict],
    require_pump_suffix: bool = False,
) -> dict | None:
    """Pick a likely traded memecoin leg for pair-first parsing."""
    candidates: list[dict] = []
    for row in token_transfers:
        mint = row.get("mint")
        if not mint or not is_memecoin_candidate_mint(mint):
            continue
        amount = pd.to_numeric(row.get("tokenAmount"), errors="coerce")
        if pd.isna(amount) or amount <= 0:
            continue
        candidates.append(row)
    if not candidates:
        return None

    if require_pump_suffix:
        candidates = [
            row for row in candidates if str(row.get("mint") or "").lower().endswith("pump")
        ]
        if not candidates:
            return None
    else:
        pump_candidates = [
            row for row in candidates if str(row.get("mint") or "").lower().endswith("pump")
        ]
        if pump_candidates:
            candidates = pump_candidates

    wallet_candidates = [
        row
        for row in candidates
        if row.get("fromUserAccount") == wallet or row.get("toUserAccount") == wallet
    ]
    if wallet_candidates:
        candidates = wallet_candidates
    buy_candidates = [row for row in candidates if row.get("toUserAccount") == wallet]
    if buy_candidates:
        candidates = buy_candidates

    return max(
        candidates,
        key=lambda row: float(pd.to_numeric(row.get("tokenAmount"), errors="coerce") or 0.0),
    )


def select_wallet_native_transfer(wallet: str, native_transfers: list[dict]) -> dict | None:
    """Pick the most wallet-relevant native transfer for swap-side inference."""
    candidates = [
        row
        for row in native_transfers
        if row.get("fromUserAccount") == wallet or row.get("toUserAccount") == wallet
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: float(pd.to_numeric(row.get("amount"), errors="coerce") or 0.0),
    )


def select_wallet_native_transfer_for_side(
    wallet: str, native_transfers: list[dict], side: str
) -> dict | None:
    """Pick the SOL leg aligned with inferred side to avoid mismatched transfer pairing."""
    if side.upper() == "BUY":
        preferred = [row for row in native_transfers if row.get("fromUserAccount") == wallet]
    elif side.upper() == "SELL":
        preferred = [row for row in native_transfers if row.get("toUserAccount") == wallet]
    else:
        preferred = []

    if preferred:
        return max(
            preferred,
            key=lambda row: float(pd.to_numeric(row.get("amount"), errors="coerce") or 0.0),
        )
    return select_wallet_native_transfer(wallet=wallet, native_transfers=native_transfers)


def infer_swap_side(wallet: str, token_transfer: dict, native_transfers: list[dict]) -> str:
    """Infer BUY/SELL for routed swaps more robustly than a strict toUserAccount check."""
    if token_transfer.get("toUserAccount") == wallet:
        return "BUY"
    if token_transfer.get("fromUserAccount") == wallet:
        return "SELL"

    # Routed swaps often debit SOL from the tracked wallet while crediting tokens
    # through an intermediate token account. Treat outbound SOL plus wallet-linked
    # token activity as a BUY.
    wallet_sent_sol = any(row.get("fromUserAccount") == wallet for row in native_transfers)
    wallet_received_sol = any(row.get("toUserAccount") == wallet for row in native_transfers)
    if wallet_sent_sol and not wallet_received_sol:
        return "BUY"
    if wallet_received_sol and not wallet_sent_sol:
        return "SELL"

    # Ambiguous routed transfers are not trusted for entry triggers.
    # Force explicit evidence for BUY/SELL and otherwise drop.
    return "UNKNOWN"


def trade_from_tx_any(tracked_wallets: Iterable[str], tx: dict) -> CandidateEvent | None:
    """Convert a transaction into a candidate event for any tracked wallet it likely belongs to."""
    event, _ = classify_trade_from_tx_any(tracked_wallets, tx)
    return event


def classify_trade_from_tx_any(
    tracked_wallets: Iterable[str], tx: dict
) -> tuple[CandidateEvent | None, str]:
    """Convert a transaction into a candidate event for any tracked wallet and return a drop reason when rejected."""
    wallet = resolve_tracked_wallet(tx, tracked_wallets)
    if wallet is None:
        return None, "no_tracked_wallet"
    return classify_trade_from_tx(wallet, tx)


def classify_trade_from_tx_pair_first(
    tracked_wallets: Iterable[str],
    tx: dict,
    allowed_sources: set[str] | None = None,
    require_pump_suffix: bool = False,
) -> tuple[CandidateEvent | None, str]:
    """Parse a pair-first swap event and annotate tracked-wallet presence."""
    tx_type = str(tx.get("type") or "").upper()
    if tx_type != "SWAP":
        return None, "not_swap"

    source = str(tx.get("source") or "").upper()
    if allowed_sources and source and source not in allowed_sources:
        return None, "source_not_allowed"

    timestamp = tx.get("timestamp")
    if timestamp is None:
        return None, "missing_timestamp"

    wallet = resolve_primary_wallet(tx)
    if wallet is None:
        return None, "missing_wallet_context"

    swap_event_buy = extract_swap_buy_from_events(
        tx=tx,
        wallet=wallet,
        require_pump_suffix=require_pump_suffix,
    )
    if swap_event_buy is not None:
        token_mint, sol_amount, token_amount = swap_event_buy
        tracked_involved = find_tracked_wallets(tx, tracked_wallets)
        return CandidateEvent(
            token_mint=token_mint,
            signature=tx.get("signature"),
            block_time=datetime.fromtimestamp(int(timestamp), tz=timezone.utc),
            triggering_wallet=wallet,
            side="BUY",
            sol_amount=sol_amount,
            token_amount=token_amount,
            source_program=tx.get("source") or tx.get("type"),
            tracked_wallets=tracked_involved,
            discovery_source="pair_first",
            event_time_source="solana_block_time",
        ), "accepted"

    token_transfers = tx.get("tokenTransfers") or []
    native_transfers = tx.get("nativeTransfers") or []
    if not token_transfers or not native_transfers:
        return None, "missing_transfers"

    token_transfer = select_pair_first_token_transfer(
        wallet=wallet,
        token_transfers=token_transfers,
        require_pump_suffix=require_pump_suffix,
    )
    if token_transfer is None and not require_pump_suffix:
        token_transfer = select_primary_token_transfer(
            token_transfers=token_transfers, require_pump_suffix=False
        )
    if token_transfer is None:
        if any(row.get("mint") in EXCLUDED_MINTS for row in token_transfers if row.get("mint")):
            return None, "excluded_base_mint"
        return None, "no_candidate_token_transfer"

    side = infer_swap_side(wallet, token_transfer, native_transfers)
    if side != "BUY":
        return None, "not_buy"

    native_transfer = select_wallet_native_transfer_for_side(wallet, native_transfers, side)
    if native_transfer is None:
        return None, "no_wallet_native_transfer"

    token_amount = pd.to_numeric(token_transfer.get("tokenAmount"), errors="coerce")
    sol_amount = pd.to_numeric(native_transfer.get("amount"), errors="coerce")
    if pd.isna(token_amount) or pd.isna(sol_amount) or token_amount <= 0 or sol_amount <= 0:
        return None, "invalid_amounts"

    tracked_involved = find_tracked_wallets(tx, tracked_wallets)
    return CandidateEvent(
        token_mint=str(token_transfer.get("mint") or ""),
        signature=tx.get("signature"),
        block_time=datetime.fromtimestamp(int(timestamp), tz=timezone.utc),
        triggering_wallet=wallet,
        side=side,
        sol_amount=float(sol_amount) / 1_000_000_000,
        token_amount=float(token_amount),
        source_program=tx.get("source") or tx.get("type"),
        tracked_wallets=tracked_involved,
        discovery_source="pair_first",
        event_time_source="solana_block_time",
    ), "accepted"


def classify_trade_from_ws_result_pair_first(
    tracked_wallets: Iterable[str],
    ws_result: dict,
    allowed_sources: set[str] | None = None,
    require_pump_suffix: bool = False,
) -> tuple[CandidateEvent | None, str]:
    """Parse `transactionSubscribe` full/jsonParsed notification into pair-first event."""
    tx_outer = ws_result.get("transaction")
    if not isinstance(tx_outer, dict):
        return None, "missing_transaction_payload"

    meta = tx_outer.get("meta")
    tx_inner = tx_outer.get("transaction")
    if not isinstance(meta, dict) or not isinstance(tx_inner, dict):
        return None, "missing_transaction_payload"
    if meta.get("err") is not None:
        return None, "failed_tx"

    message = tx_inner.get("message")
    if not isinstance(message, dict):
        return None, "missing_message"

    account_keys_raw = message.get("accountKeys")
    if not isinstance(account_keys_raw, list) or not account_keys_raw:
        return None, "missing_account_keys"
    account_keys = [
        key for key in (_account_key_pubkey(entry) for entry in account_keys_raw) if key
    ]
    if not account_keys:
        return None, "missing_account_keys"
    account_key_set = set(account_keys)

    instruction_signals = _collect_instruction_signals(
        ws_result=ws_result, account_keys=account_keys
    )

    source_program, source_ok = _resolve_source_from_programs(
        account_key_set, allowed_sources=allowed_sources
    )
    if instruction_signals.sources:
        source_program, source_ok = _resolve_source_from_labels(
            set(instruction_signals.sources),
            allowed_sources=allowed_sources,
        )
    if not source_ok:
        return None, "source_not_allowed"

    triggering_wallet: str | None = None
    for entry in account_keys_raw:
        if _account_key_is_signer(entry):
            triggering_wallet = _account_key_pubkey(entry)
            if triggering_wallet:
                break
    if not triggering_wallet:
        triggering_wallet = account_keys[0]

    tracked_wallet_set = set(tracked_wallets)
    tracked_involved = tuple(sorted(account_key_set & tracked_wallet_set))

    pre_token_rows = meta.get("preTokenBalances") or []
    post_token_rows = meta.get("postTokenBalances") or []
    if not isinstance(pre_token_rows, list) or not isinstance(post_token_rows, list):
        return None, "missing_transfers"

    pre_token = _token_balances_by_owner(pre_token_rows, triggering_wallet)
    post_token = _token_balances_by_owner(post_token_rows, triggering_wallet)

    token_candidates: list[tuple[str, float]] = []
    for mint in set(pre_token) | set(post_token):
        if not is_memecoin_candidate_mint(mint):
            continue
        if require_pump_suffix and not mint.lower().endswith("pump"):
            continue
        delta = post_token.get(mint, 0.0) - pre_token.get(mint, 0.0)
        if abs(delta) > 0:
            token_candidates.append((mint, delta))
    if not token_candidates:
        return None, "no_token_delta"

    token_mint, token_delta = max(token_candidates, key=lambda item: abs(item[1]))
    token_amount = abs(float(token_delta))
    if token_amount <= 0:
        return None, "invalid_amounts"

    wallet_index = account_keys.index(triggering_wallet) if triggering_wallet in account_keys else 0
    pre_balances = meta.get("preBalances") or []
    post_balances = meta.get("postBalances") or []
    if (
        not isinstance(pre_balances, list)
        or not isinstance(post_balances, list)
        or wallet_index >= len(pre_balances)
        or wallet_index >= len(post_balances)
    ):
        return None, "missing_native_balances"
    pre_lamports = _to_float(pre_balances[wallet_index])
    post_lamports = _to_float(post_balances[wallet_index])
    fee_lamports = _to_float(meta.get("fee")) or 0.0
    if pre_lamports is None or post_lamports is None:
        return None, "missing_native_balances"
    delta_lamports = pre_lamports - post_lamports
    side = "BUY" if token_delta > 0 else "SELL"
    if side == "BUY":
        sol_lamports = delta_lamports - fee_lamports
        if sol_lamports <= 0:
            sol_lamports = delta_lamports
        if sol_lamports <= 0:
            return None, "no_sol_spent"
        sol_amount = sol_lamports / 1_000_000_000
        if sol_amount <= 0:
            return None, "no_sol_spent"
    else:
        sol_lamports = (-delta_lamports) + fee_lamports
        if sol_lamports <= 0:
            sol_lamports = -delta_lamports
        if sol_lamports <= 0:
            return None, "no_sol_received"
        sol_amount = sol_lamports / 1_000_000_000
        if sol_amount <= 0:
            return None, "no_sol_received"

    # Prefer explicit instruction/log swap signals and exclude pure launch/create txs.
    if instruction_signals.has_launch_signal and not instruction_signals.has_swap_signal:
        return None, "launch_instruction"
    if not instruction_signals.has_swap_signal and source_program in {
        "PUMP_FUN",
        "PUMP_AMM",
        "RAYDIUM",
        "RAYDIUM_LAUNCHLAB",
    }:
        # Fallback heuristic for non-decoded variants: keep only balance-delta-confirmed swaps.
        # This preserves throughput while still rejecting known create/init txs above.
        pass

    ts = ws_result.get("blockTime") or ws_result.get("timestamp")
    if ts is None:
        block_time = datetime.now(tz=timezone.utc)
    else:
        try:
            block_time = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            block_time = datetime.now(tz=timezone.utc)

    signature = ws_result.get("signature")
    if not isinstance(signature, str) or not signature:
        signatures = tx_inner.get("signatures") or []
        if isinstance(signatures, list) and signatures and isinstance(signatures[0], str):
            signature = signatures[0]
    if not isinstance(signature, str) or not signature:
        return None, "missing_signature"

    return CandidateEvent(
        token_mint=str(token_mint),
        signature=signature,
        block_time=block_time,
        triggering_wallet=str(triggering_wallet),
        side=side,
        sol_amount=float(sol_amount),
        token_amount=float(token_amount),
        source_program=source_program or "UNKNOWN",
        tracked_wallets=tracked_involved,
        discovery_source="pair_first",
        event_time_source="solana_block_time",
    ), "accepted"


def classify_trade_from_ws_result_any(
    tracked_wallets: Iterable[str],
    ws_result: dict,
    allowed_sources: set[str] | None = None,
    require_pump_suffix: bool = False,
) -> tuple[CandidateEvent | None, str]:
    """Parse websocket notification for wallet-first mode."""
    event, reason = classify_trade_from_ws_result_pair_first(
        tracked_wallets=tracked_wallets,
        ws_result=ws_result,
        allowed_sources=allowed_sources,
        require_pump_suffix=require_pump_suffix,
    )
    if event is None:
        return None, reason
    wallet_set = set(tracked_wallets)
    if event.triggering_wallet in wallet_set:
        event.discovery_source = "wallet_first"
        event.tracked_wallets = (event.triggering_wallet,)
        return event, "accepted"
    if event.tracked_wallets:
        event.discovery_source = "wallet_first"
        event.triggering_wallet = event.tracked_wallets[0]
        event.tracked_wallets = (event.tracked_wallets[0],)
        return event, "accepted"
    return None, "no_tracked_wallet"
