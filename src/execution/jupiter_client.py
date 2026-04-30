"""Jupiter v2 API client for swap quote and transaction building.

Uses the Jupiter Aggregator v2 API (api.jup.ag/swap/v2) to obtain a quote
and build an unsigned swap transaction in a single /order call.  The unsigned
transaction is signed locally and broadcast via Helius RPC.
"""

from __future__ import annotations

import base64
import logging
import json
from dataclasses import dataclass
from typing import Any

import httpx

from src.utils.retry import default_retry

logger = logging.getLogger(__name__)

# Wrapped SOL mint used by Jupiter
SOL_MINT = "So11111111111111111111111111111111111111112"

# 1 SOL = 1_000_000_000 lamports
LAMPORTS_PER_SOL = 1_000_000_000


class JupiterError(Exception):
    """Raised when a Jupiter API call fails or returns an invalid response."""


@dataclass(frozen=True)
class SwapOrder:
    """Combined quote + unsigned transaction from Jupiter v2 /order endpoint.

    A single GET /order call returns both the routing quote and an assembled
    unsigned transaction, saving one round-trip vs the old v1 two-step flow.
    """

    input_mint: str
    output_mint: str
    in_amount: int  # lamports / smallest token unit
    out_amount: int  # lamports / smallest token unit
    price_impact_pct: float
    slippage_bps: int
    last_valid_block_height: int
    raw_transaction: bytes  # raw unsigned transaction bytes
    raw: dict[str, Any]  # full /order response
    priority_fee_lamports: int = 0
    jito_tip_lamports: int = 0
    broadcast_fee_type: str | None = None


# Kept for backward compatibility with paper-trading callers that only
# need quote data and don't use the transaction.
@dataclass(frozen=True)
class SwapQuote:
    """Parsed Jupiter quote (no transaction).  Populated from SwapOrder."""

    input_mint: str
    output_mint: str
    in_amount: int
    out_amount: int
    price_impact_pct: float
    slippage_bps: int
    raw: dict[str, Any]


# Kept for backward compatibility with callers that return SwapTransaction.
@dataclass(frozen=True)
class SwapTransaction:
    """Unsigned swap transaction.  Populated from SwapOrder."""

    raw_transaction: bytes
    last_valid_block_height: int
    in_amount: int = 0
    out_amount: int = 0
    built_at: str | None = None


class JupiterClient:
    """Thin wrapper over the Jupiter v2 REST API (api.jup.ag/swap/v2).

    Parameters
    ----------
    base_url:
        API base URL.  Paid endpoint: ``https://api.jup.ag/swap/v2``.
        Public endpoint: ``https://lite-api.jup.ag/swap/v2``.
    api_key:
        Jupiter API key.  Sent as ``x-api-key`` request header.
    user_public_key:
        Default wallet public key used as the ``taker`` parameter.
        Can be overridden per-call in :meth:`get_order`.
    timeout_sec:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        user_public_key: str = "",
        timeout_sec: int = 30,
    ) -> None:
        if not base_url:
            raise JupiterError("JUPITER_BASE_URL is required for live trading")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.user_public_key = user_public_key.strip()
        self.timeout_sec = timeout_sec
        self._headers: dict[str, str] = {}
        if self.api_key:
            # v2 uses x-api-key instead of Authorization: Bearer
            self._headers["x-api-key"] = self.api_key
            logger.info("JupiterClient: using authenticated endpoint (%s)", self.base_url)
        else:
            logger.info("JupiterClient: using public endpoint (%s)", self.base_url)
        # Persistent connection pools — reuses TCP+TLS across calls, saving 50-90ms per request.
        self._client = httpx.Client(timeout=self.timeout_sec, headers=self._headers)
        # Async client for get_order_async() — lets the event loop stay unblocked
        # on the Jupiter call when no pre-built TX is available at fire time.
        self._async_client = httpx.AsyncClient(timeout=self.timeout_sec, headers=self._headers)

    def close(self) -> None:
        """Close the persistent sync HTTP connection pool.

        The async client requires an async context to close gracefully; it will
        be cleaned up at process exit.
        """
        self._client.close()

    @staticmethod
    def _response_error_detail(resp: httpx.Response) -> str:
        try:
            payload = resp.json()
        except Exception:  # noqa: BLE001
            text = (resp.text or "").strip()
            return text[:500] if text else f"HTTP {resp.status_code}"
        if isinstance(payload, dict):
            error = payload.get("error")
            if error:
                return str(error)
            message = payload.get("message")
            if message:
                return str(message)
        try:
            rendered = json.dumps(payload, ensure_ascii=True)
        except Exception:  # noqa: BLE001
            rendered = str(payload)
        return rendered[:500]

    def _raise_for_status(self, resp: httpx.Response, *, context: str) -> None:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = self._response_error_detail(resp)
            raise JupiterError(f"{context} HTTP {resp.status_code}: {detail}") from exc
        except httpx.HTTPError as exc:
            raise JupiterError(f"{context} transport error: {exc}") from exc

    def _v1_base_url(self) -> str:
        """Return the Metis v1 base URL derived from the configured v2 base."""
        if "/swap/v2" in self.base_url:
            return self.base_url.replace("/swap/v2", "/swap/v1")
        if self.base_url.endswith("/swap"):
            return f"{self.base_url}/v1"
        if self.base_url.endswith("/v2"):
            return self.base_url[:-3] + "/v1"
        return "https://api.jup.ag/swap/v1"

    # ------------------------------------------------------------------
    # Primary v2 method: quote + unsigned TX in one call
    # ------------------------------------------------------------------

    @default_retry()
    def get_order(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int | None = 150,
        user_public_key: str = "",
        priority_fee_lamports: int | None = None,
        jito_tip_lamports: int | None = None,
        broadcast_fee_type: str | None = None,
    ) -> SwapOrder:
        """Fetch a quote AND build the unsigned swap transaction in one call.

        Jupiter v2 /order combines the old /quote + /swap two-step flow into
        a single GET request, reducing latency by one round-trip.

        Parameters
        ----------
        input_mint:
            Mint address of the input token (e.g. SOL_MINT for buying).
        output_mint:
            Mint address of the output token.
        amount:
            Amount in the smallest unit of the *input* token.
        slippage_bps:
            Maximum allowed slippage in basis points.
        user_public_key:
            Wallet that will sign the transaction (``taker``).
            Falls back to the ``user_public_key`` passed at construction.
        priority_fee_lamports:
            Priority fee to embed in the transaction.

        Returns
        -------
        SwapOrder
        """
        taker = (user_public_key.strip() or self.user_public_key).strip()
        if not taker:
            raise JupiterError("user_public_key is required for Jupiter v2 /order (set taker)")

        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "taker": taker,
        }
        if slippage_bps is not None:
            params["slippageBps"] = str(slippage_bps)
        if priority_fee_lamports is not None:
            params["priorityFeeLamports"] = str(priority_fee_lamports)
        if jito_tip_lamports is not None:
            params["jitoTipLamports"] = str(jito_tip_lamports)
        if broadcast_fee_type:
            params["broadcastFeeType"] = str(broadcast_fee_type)
        url = f"{self.base_url}/order"
        logger.debug("Jupiter /order request: %s params=%s", url, params)

        resp = self._client.get(url, params=params)
        self._raise_for_status(resp, context="Jupiter order")
        data: dict[str, Any] = resp.json()

        if "error" in data:
            raise JupiterError(f"Jupiter order error: {data['error']}")
        if "outAmount" not in data:
            raise JupiterError(f"Jupiter order missing outAmount: {data}")
        tx_b64 = data.get("transaction")
        if not tx_b64:
            raise JupiterError(f"Jupiter order missing transaction: {list(data.keys())}")

        raw_tx = base64.b64decode(tx_b64)
        last_valid = int(data.get("lastValidBlockHeight", 0))
        # v2 uses "priceImpact" (v1 used "priceImpactPct")
        price_impact = float(data.get("priceImpact", data.get("priceImpactPct", 0.0)))

        order = SwapOrder(
            input_mint=data.get("inputMint", input_mint),
            output_mint=data.get("outputMint", output_mint),
            in_amount=int(data.get("inAmount", amount)),
            out_amount=int(data["outAmount"]),
            price_impact_pct=price_impact,
            slippage_bps=int(data.get("slippageBps", slippage_bps or 0) or 0),
            priority_fee_lamports=int(
                data.get("prioritizationFeeLamports", priority_fee_lamports or 0) or 0
            ),
            jito_tip_lamports=int(jito_tip_lamports or 0),
            broadcast_fee_type=broadcast_fee_type,
            last_valid_block_height=last_valid,
            raw_transaction=raw_tx,
            raw=data,
        )
        logger.info(
            "Jupiter order: %s → %s | in=%s out=%s impact=%.4f%% last_valid=%d",
            input_mint[:8],
            output_mint[:8],
            order.in_amount,
            order.out_amount,
            order.price_impact_pct * 100,
            order.last_valid_block_height,
        )
        return order

    async def get_order_async(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int | None = 150,
        user_public_key: str = "",
        priority_fee_lamports: int | None = None,
        jito_tip_lamports: int | None = None,
        broadcast_fee_type: str | None = None,
    ) -> SwapOrder:
        """Async version of :meth:`get_order` — does not block the event loop.

        Uses the persistent ``_async_client`` (httpx.AsyncClient) so the event
        loop can continue ingesting gRPC events while Jupiter responds.
        Identical semantics to the sync version; use this as the fallback path
        when no pre-built TX is available at fire time.
        """
        taker = (user_public_key.strip() or self.user_public_key).strip()
        if not taker:
            raise JupiterError("user_public_key is required for Jupiter v2 /order (set taker)")

        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "taker": taker,
        }
        if slippage_bps is not None:
            params["slippageBps"] = str(slippage_bps)
        if priority_fee_lamports is not None:
            params["priorityFeeLamports"] = str(priority_fee_lamports)
        if jito_tip_lamports is not None:
            params["jitoTipLamports"] = str(jito_tip_lamports)
        if broadcast_fee_type:
            params["broadcastFeeType"] = str(broadcast_fee_type)
        url = f"{self.base_url}/order"
        logger.debug("Jupiter /order async request: %s params=%s", url, params)

        resp = await self._async_client.get(url, params=params)
        self._raise_for_status(resp, context="Jupiter order")
        data: dict[str, Any] = resp.json()

        if "error" in data:
            raise JupiterError(f"Jupiter order error: {data['error']}")
        if "outAmount" not in data:
            raise JupiterError(f"Jupiter order missing outAmount: {data}")
        tx_b64 = data.get("transaction")
        if not tx_b64:
            raise JupiterError(f"Jupiter order missing transaction: {list(data.keys())}")

        raw_tx = base64.b64decode(tx_b64)
        last_valid = int(data.get("lastValidBlockHeight", 0))
        price_impact = float(data.get("priceImpact", data.get("priceImpactPct", 0.0)))

        order = SwapOrder(
            input_mint=data.get("inputMint", input_mint),
            output_mint=data.get("outputMint", output_mint),
            in_amount=int(data.get("inAmount", amount)),
            out_amount=int(data["outAmount"]),
            price_impact_pct=price_impact,
            slippage_bps=int(data.get("slippageBps", slippage_bps or 0) or 0),
            priority_fee_lamports=int(
                data.get("prioritizationFeeLamports", priority_fee_lamports or 0) or 0
            ),
            jito_tip_lamports=int(jito_tip_lamports or 0),
            broadcast_fee_type=broadcast_fee_type,
            last_valid_block_height=last_valid,
            raw_transaction=raw_tx,
            raw=data,
        )
        logger.info(
            "Jupiter order (async): %s → %s | in=%s out=%s impact=%.4f%% last_valid=%d",
            input_mint[:8],
            output_mint[:8],
            order.in_amount,
            order.out_amount,
            order.price_impact_pct * 100,
            order.last_valid_block_height,
        )
        return order

    # ------------------------------------------------------------------
    # Backward-compat shims (paper trading / legacy callers)
    # ------------------------------------------------------------------

    def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 150,
        priority_fee_lamports: int = 50_000,
    ) -> SwapQuote:
        """Return just the quote fields (no TX).  Calls :meth:`get_order` internally.

        Used by paper-trading callers that only need in/out amounts and
        price impact — the embedded transaction is discarded.
        """
        order = self.get_order(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=amount,
            slippage_bps=slippage_bps,
            priority_fee_lamports=priority_fee_lamports,
        )
        return SwapQuote(
            input_mint=order.input_mint,
            output_mint=order.output_mint,
            in_amount=order.in_amount,
            out_amount=order.out_amount,
            price_impact_pct=order.price_impact_pct,
            slippage_bps=order.slippage_bps,
            raw=order.raw,
        )

    @default_retry()
    def get_metis_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int,
        *,
        swap_mode: str = "ExactIn",
        for_jito_bundle: bool = False,
        as_legacy_transaction: bool = False,
    ) -> dict[str, Any]:
        """Fetch a Metis v1 quote for custom transaction assembly."""
        params: dict[str, str] = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": str(slippage_bps),
            "swapMode": str(swap_mode),
            "restrictIntermediateTokens": "true",
            "instructionVersion": "V2",
            "asLegacyTransaction": "true" if as_legacy_transaction else "false",
        }
        if for_jito_bundle:
            params["forJitoBundle"] = "true"
        url = f"{self._v1_base_url()}/quote"
        resp = self._client.get(url, params=params)
        self._raise_for_status(resp, context="Jupiter Metis quote")
        data: dict[str, Any] = resp.json()
        if "error" in data and data["error"]:
            raise JupiterError(f"Jupiter Metis quote error: {data['error']}")
        if "outAmount" not in data or "routePlan" not in data:
            raise JupiterError(f"Jupiter Metis quote missing fields: {data}")
        logger.info(
            "Jupiter Metis quote: %s → %s | in=%s out=%s impact=%s",
            input_mint[:8],
            output_mint[:8],
            data.get("inAmount"),
            data.get("outAmount"),
            data.get("priceImpactPct"),
        )
        return data

    async def get_metis_quote_async(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int,
        *,
        swap_mode: str = "ExactIn",
        for_jito_bundle: bool = False,
        as_legacy_transaction: bool = False,
    ) -> dict[str, Any]:
        """Async variant of :meth:`get_metis_quote`."""
        params: dict[str, str] = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": str(slippage_bps),
            "swapMode": str(swap_mode),
            "restrictIntermediateTokens": "true",
            "instructionVersion": "V2",
            "asLegacyTransaction": "true" if as_legacy_transaction else "false",
        }
        if for_jito_bundle:
            params["forJitoBundle"] = "true"
        url = f"{self._v1_base_url()}/quote"
        resp = await self._async_client.get(url, params=params)
        self._raise_for_status(resp, context="Jupiter Metis quote")
        data: dict[str, Any] = resp.json()
        if "error" in data and data["error"]:
            raise JupiterError(f"Jupiter Metis quote error: {data['error']}")
        if "outAmount" not in data or "routePlan" not in data:
            raise JupiterError(f"Jupiter Metis quote missing fields: {data}")
        logger.info(
            "Jupiter Metis quote (async): %s → %s | in=%s out=%s impact=%s",
            input_mint[:8],
            output_mint[:8],
            data.get("inAmount"),
            data.get("outAmount"),
            data.get("priceImpactPct"),
        )
        return data

    @default_retry()
    def get_swap_instructions(
        self,
        *,
        quote_response: dict[str, Any],
        user_public_key: str,
        dynamic_compute_unit_limit: bool = True,
        wrap_and_unwrap_sol: bool = True,
        as_legacy_transaction: bool = False,
        blockhash_slots_to_expiry: int | None = None,
        use_shared_accounts: bool | None = None,
        skip_user_accounts_rpc_calls: bool | None = None,
    ) -> dict[str, Any]:
        """Fetch raw Metis swap instructions for custom transaction assembly."""
        if not user_public_key:
            raise JupiterError("user_public_key is required for swap-instructions")
        payload: dict[str, Any] = {
            "userPublicKey": user_public_key,
            "quoteResponse": quote_response,
            "wrapAndUnwrapSol": bool(wrap_and_unwrap_sol),
            "asLegacyTransaction": bool(as_legacy_transaction),
            "dynamicComputeUnitLimit": bool(dynamic_compute_unit_limit),
        }
        if blockhash_slots_to_expiry is not None:
            payload["blockhashSlotsToExpiry"] = int(blockhash_slots_to_expiry)
        if use_shared_accounts is not None:
            payload["useSharedAccounts"] = bool(use_shared_accounts)
        if skip_user_accounts_rpc_calls is not None:
            payload["skipUserAccountsRpcCalls"] = bool(skip_user_accounts_rpc_calls)
        url = f"{self._v1_base_url()}/swap-instructions"
        resp = self._client.post(url, json=payload)
        self._raise_for_status(resp, context="Jupiter swap-instructions")
        data: dict[str, Any] = resp.json()
        if "error" in data and data["error"]:
            raise JupiterError(f"Jupiter swap-instructions error: {data['error']}")
        if "swapInstruction" not in data:
            raise JupiterError(f"Jupiter swap-instructions missing swapInstruction: {data}")
        logger.info(
            "Jupiter swap-instructions: compute=%d setup=%d other=%d alts=%d",
            len(list(data.get("computeBudgetInstructions") or [])),
            len(list(data.get("setupInstructions") or [])),
            len(list(data.get("otherInstructions") or [])),
            len(list(data.get("addressLookupTableAddresses") or [])),
        )
        return data

    async def get_swap_instructions_async(
        self,
        *,
        quote_response: dict[str, Any],
        user_public_key: str,
        dynamic_compute_unit_limit: bool = True,
        wrap_and_unwrap_sol: bool = True,
        as_legacy_transaction: bool = False,
        blockhash_slots_to_expiry: int | None = None,
        use_shared_accounts: bool | None = None,
        skip_user_accounts_rpc_calls: bool | None = None,
    ) -> dict[str, Any]:
        """Async variant of :meth:`get_swap_instructions`."""
        if not user_public_key:
            raise JupiterError("user_public_key is required for swap-instructions")
        payload: dict[str, Any] = {
            "userPublicKey": user_public_key,
            "quoteResponse": quote_response,
            "wrapAndUnwrapSol": bool(wrap_and_unwrap_sol),
            "asLegacyTransaction": bool(as_legacy_transaction),
            "dynamicComputeUnitLimit": bool(dynamic_compute_unit_limit),
        }
        if blockhash_slots_to_expiry is not None:
            payload["blockhashSlotsToExpiry"] = int(blockhash_slots_to_expiry)
        if use_shared_accounts is not None:
            payload["useSharedAccounts"] = bool(use_shared_accounts)
        if skip_user_accounts_rpc_calls is not None:
            payload["skipUserAccountsRpcCalls"] = bool(skip_user_accounts_rpc_calls)
        url = f"{self._v1_base_url()}/swap-instructions"
        resp = await self._async_client.post(url, json=payload)
        self._raise_for_status(resp, context="Jupiter swap-instructions")
        data: dict[str, Any] = resp.json()
        if "error" in data and data["error"]:
            raise JupiterError(f"Jupiter swap-instructions error: {data['error']}")
        if "swapInstruction" not in data:
            raise JupiterError(f"Jupiter swap-instructions missing swapInstruction: {data}")
        logger.info(
            "Jupiter swap-instructions (async): compute=%d setup=%d other=%d alts=%d",
            len(list(data.get("computeBudgetInstructions") or [])),
            len(list(data.get("setupInstructions") or [])),
            len(list(data.get("otherInstructions") or [])),
            len(list(data.get("addressLookupTableAddresses") or [])),
        )
        return data
