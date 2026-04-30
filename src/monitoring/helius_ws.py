"""Chainstack Yellowstone gRPC monitor (transport replacement for websocket monitor)."""

from __future__ import annotations

import asyncio
import logging
import random
import threading
from collections import deque
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from time import time
from typing import Any, Callable, Iterable

import base58
import grpc
import pandas as pd

from src.bot.config import BotConfig
from src.bot.models import CandidateEvent
from src.monitoring.parsing import EXCLUDED_MINTS, PROGRAM_SOURCE_BY_ID
from src.monitoring.yellowstone_proto.generated import geyser_pb2, geyser_pb2_grpc
from src.strategy.local_quote import PumpAMMQuoteEngine


@dataclass
class WebsocketUnavailableError(RuntimeError):
    """Raised when realtime monitoring transport is unavailable."""

    reason: str

    def __str__(self) -> str:
        return self.reason


class HeliusWebsocketMonitor:
    """Runtime monitor using Chainstack Yellowstone gRPC transport."""

    def __init__(
        self,
        config: BotConfig,
        wallets: list[str],
        status_callback: Callable[..., None] | None = None,
        local_quote_engine: PumpAMMQuoteEngine | None = None,
    ) -> None:
        self.config = config
        self.wallets = wallets
        self._local_quote_engine = local_quote_engine
        self.discovery_mode = str(config.discovery_mode or "pair_first").lower()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._status_callback = status_callback
        self._notification_count = 0
        self._candidate_event_count = 0
        self._candidate_tracked_wallet_hit_count = 0
        self._dropped_notification_count = 0
        self._drop_reason_counts: dict[str, int] = {}
        self._recent_signatures: deque[str] = deque(maxlen=10000)
        self._recent_signature_set: set[str] = set()
        # Lock protecting _recent_signatures/_recent_signature_set when
        # _candidate_from_update runs in the parse thread pool.
        self._sig_lock = threading.Lock()
        # Thread pool for offloading protobuf parsing off the async event loop.
        # max_workers=2: one parses while the next waits for gRPC, giving pipelining
        # without introducing parallelism that would require broader locking.
        self._parse_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="grpc-parse")
        self._allowed_sources = {source.upper() for source in config.discovery_allowed_sources}
        self._tracked_wallet_set = set(wallets)
        self._subscription_targets = list(dict.fromkeys(config.discovery_account_include))
        self._subscription_target_type = "program"
        self._filter_name_to_program: dict[str, str] = {}
        self._channel: grpc.aio.Channel | None = None
        self._max_retries = max(0, int(config.chainstack_reconnect_max_retries))
        self._backoff_initial_sec = max(1, int(config.chainstack_reconnect_backoff_initial_sec))
        self._backoff_max_sec = max(
            self._backoff_initial_sec, int(config.chainstack_reconnect_backoff_max_sec)
        )

    @property
    def grpc_endpoint(self) -> str:
        """Normalized Chainstack gRPC endpoint host:port (without scheme)."""
        endpoint = (self.config.chainstack_grpc_endpoint or "").strip()
        endpoint = endpoint.removeprefix("https://").removeprefix("http://")
        return endpoint.rstrip("/")

    def _record_drop(self, reason: str, count: int = 1, emit_status: bool = True) -> None:
        """Record dropped notifications and optionally emit status."""
        if count <= 0:
            return
        self._dropped_notification_count += count
        self._drop_reason_counts[reason] = self._drop_reason_counts.get(reason, 0) + count
        if emit_status:
            self._emit_drop_status(last_reason=reason)

    def _emit_drop_status(self, last_reason: str | None = None) -> None:
        """Push current drop counters to status writer."""
        if self._status_callback is None:
            return
        resolved_reason = last_reason
        if resolved_reason is None and self._drop_reason_counts:
            resolved_reason = next(reversed(self._drop_reason_counts))
        self._status_callback(
            websocket_dropped_notification_count=self._dropped_notification_count,
            websocket_drop_reason_counts=self._drop_reason_counts,
            websocket_last_drop_reason=resolved_reason,
        )

    def _remember_signature(self, signature: str) -> None:
        """Keep a bounded dedupe cache for stream signatures (thread-safe via _sig_lock)."""
        with self._sig_lock:
            if len(self._recent_signatures) == self._recent_signatures.maxlen:
                oldest = self._recent_signatures.popleft()
                self._recent_signature_set.discard(oldest)
            self._recent_signatures.append(signature)
            self._recent_signature_set.add(signature)

    def _create_grpc_channel(self) -> grpc.aio.Channel:
        """Create authenticated secure gRPC channel."""
        endpoint = self.grpc_endpoint
        token = (self.config.chainstack_grpc_token or "").strip()
        if not endpoint:
            raise WebsocketUnavailableError("CHAINSTACK_GRPC_ENDPOINT is missing.")
        if not token:
            raise WebsocketUnavailableError("CHAINSTACK_GRPC_TOKEN is missing.")

        auth = grpc.metadata_call_credentials(
            lambda _, callback: callback((("x-token", token),), None)
        )
        channel_creds = grpc.composite_channel_credentials(grpc.ssl_channel_credentials(), auth)
        options = [
            ("grpc.keepalive_time_ms", 30_000),
            ("grpc.keepalive_timeout_ms", 10_000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.min_time_between_pings_ms", 10_000),
            ("grpc.max_receive_message_length", 32 * 1024 * 1024),
        ]
        return grpc.aio.secure_channel(endpoint, channel_creds, options=options)

    @staticmethod
    def _error_label(exc: Exception) -> str:
        """Return concise error label for logs and status."""
        if isinstance(exc, grpc.aio.AioRpcError):
            code = exc.code()
            details = exc.details() or ""
            return f"{code.name}: {details}".strip()
        return str(exc)

    def _build_subscription_request(self) -> geyser_pb2.SubscribeRequest:
        """Build one SubscribeRequest with one named transaction filter per program."""
        request = geyser_pb2.SubscribeRequest()
        request.commitment = geyser_pb2.CommitmentLevel.PROCESSED
        self._filter_name_to_program.clear()

        for idx, program_id in enumerate(self._subscription_targets):
            source = PROGRAM_SOURCE_BY_ID.get(program_id, "PROGRAM")
            filter_name = f"{source.lower()}_{idx}"
            self._filter_name_to_program[filter_name] = program_id
            tx_filter = request.transactions[filter_name]
            tx_filter.account_include.append(program_id)
            tx_filter.failed = False
            tx_filter.vote = False
        return request

    async def _request_iterator(self, request: geyser_pb2.SubscribeRequest):
        """Send initial subscription and keep stream alive with ping frames."""
        yield request
        while True:
            await asyncio.sleep(30)
            ping = geyser_pb2.SubscribeRequest()
            ping.ping.id = int(time())
            yield ping

    @staticmethod
    def _b58(value: Any) -> str | None:
        """Convert bytes-like value to base58 string."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, bytearray):
            value = bytes(value)
        if isinstance(value, bytes):
            return base58.b58encode(value).decode("utf-8")
        return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        """Best-effort numeric conversion."""
        try:
            converted = float(pd.to_numeric(value, errors="coerce"))
        except (TypeError, ValueError):
            return None
        if pd.isna(converted):
            return None
        return converted

    @staticmethod
    def _proto_timestamp_to_datetime(value: Any) -> datetime | None:
        """Convert one protobuf Timestamp-like object into UTC datetime."""
        if value is None or not hasattr(value, "seconds"):
            return None
        try:
            seconds = int(getattr(value, "seconds", 0))
            nanos = int(getattr(value, "nanos", 0))
        except (TypeError, ValueError):
            return None
        return datetime.fromtimestamp(seconds + (nanos / 1_000_000_000), tz=timezone.utc)

    @staticmethod
    def _is_memecoin_candidate_mint(mint: str | None, require_pump_suffix: bool) -> bool:
        """Return whether mint passes non-base and optional pump suffix filters."""
        if not mint or mint in EXCLUDED_MINTS:
            return False
        if require_pump_suffix and not mint.lower().endswith("pump"):
            return False
        return True

    def _logs_has_swap_signal(self, logs: list[str], source_program: str | None = None) -> bool:
        """Quickly detect buy/sell/swap tx and skip create/init-only logs."""
        if not logs:
            return False
        lowered = [entry.lower() for entry in logs if isinstance(entry, str)]
        has_swap = any(
            ("instruction: buy" in entry)
            or ("instruction: sell" in entry)
            or ("instruction: swap" in entry)
            for entry in lowered
        )
        # Raydium V4 (AMM V4, 675kPX9M…) is not Anchor: it doesn't emit
        # "Program log: Instruction: <name>", it emits "Program log: ray_log: <b64>".
        # Treat any ray_log line as a swap signal and let the downstream
        # balance-delta gate reject liquidity ops (they have no SOL↔token delta).
        if not has_swap and source_program == "RAYDIUM":
            if any("ray_log:" in entry for entry in lowered):
                has_swap = True
        has_launch_only = (
            any(
                ("instruction: create" in entry)
                or ("instruction: create_v2" in entry)
                or ("instruction: initialize" in entry)
                for entry in lowered
            )
            and not has_swap
        )
        if has_launch_only:
            return False
        return has_swap

    def _account_keys_from_message(self, message: Any) -> list[str]:
        """Decode message account keys into base58 strings."""
        raw_keys = getattr(message, "account_keys", [])
        keys: list[str] = []
        for key in raw_keys:
            decoded = self._b58(key)
            if decoded:
                keys.append(decoded)
        return keys

    def _full_account_keys_from_message_and_meta(self, message: Any, meta: Any) -> list[str]:
        """Decode static + loaded account keys into full instruction index order."""
        keys = self._account_keys_from_message(message)
        for raw_list_name in ("loaded_writable_addresses", "loaded_readonly_addresses"):
            for key in list(getattr(meta, raw_list_name, []) or []):
                decoded = self._b58(key)
                if decoded:
                    keys.append(decoded)
        return keys

    def _resolve_source(
        self,
        account_keys: Iterable[str],
        update_filters: Iterable[str],
    ) -> tuple[str | None, bool]:
        """Resolve normalized source label from matched filter names or account keys."""
        labels: list[str] = []
        matched_unmapped_program = False
        for filter_name in update_filters:
            program_id = self._filter_name_to_program.get(filter_name)
            if program_id:
                label = PROGRAM_SOURCE_BY_ID.get(program_id)
                if label:
                    labels.append(label)
                else:
                    matched_unmapped_program = True
        if not labels:
            labels = [
                PROGRAM_SOURCE_BY_ID[key] for key in account_keys if key in PROGRAM_SOURCE_BY_ID
            ]
        if not labels and matched_unmapped_program:
            # Program was explicitly subscribed but not mapped in PROGRAM_SOURCE_BY_ID yet.
            return "PROGRAM", True
        if not labels:
            return None, not bool(self._allowed_sources)
        labels = sorted(set(labels))
        if self._allowed_sources:
            allowed = [label for label in labels if label in self._allowed_sources]
            if not allowed:
                return None, False
            return allowed[0], True
        return labels[0], True

    def _token_balances_by_owner(self, rows: Iterable[Any], owner: str) -> dict[str, float]:
        """Aggregate token balances by mint for one owner from protobuf token balance rows."""
        balances: dict[str, float] = {}
        for row in rows:
            row_owner = getattr(row, "owner", "")
            if not isinstance(row_owner, str) or row_owner != owner:
                continue
            mint = getattr(row, "mint", "")
            if not isinstance(mint, str) or not mint:
                continue
            ui = getattr(row, "ui_token_amount", None)
            amount: float | None = None
            if ui is not None:
                ui_amount_string = getattr(ui, "ui_amount_string", None)
                if ui_amount_string:
                    amount = self._to_float(ui_amount_string)
                if amount is None:
                    amount = self._to_float(getattr(ui, "ui_amount", None))
                if amount is None:
                    raw_amount = self._to_float(getattr(ui, "amount", None))
                    decimals = getattr(ui, "decimals", None)
                    try:
                        decimals_int = int(decimals)
                    except (TypeError, ValueError):
                        decimals_int = None
                    if raw_amount is not None and decimals_int is not None and decimals_int >= 0:
                        amount = raw_amount / (10**decimals_int)
            if amount is None:
                continue
            balances[mint] = balances.get(mint, 0.0) + float(amount)
        return balances

    def _candidate_from_update(
        self, update: geyser_pb2.SubscribeUpdate
    ) -> tuple[CandidateEvent | None, str]:
        """Parse one geyser transaction update into CandidateEvent or drop reason."""
        parse_started_at = datetime.now(tz=timezone.utc)
        if not update.HasField("transaction"):
            return None, "not_transaction_update"

        tx_update = update.transaction
        info = tx_update.transaction
        if info is None:
            return None, "missing_transaction_payload"
        if bool(getattr(info, "is_vote", False)):
            return None, "vote_tx"

        signature = self._b58(getattr(info, "signature", b""))
        if not signature:
            return None, "missing_signature"
        with self._sig_lock:
            if signature in self._recent_signature_set:
                return None, "duplicate_signature"

        tx = getattr(info, "transaction", None)
        meta = getattr(info, "meta", None)
        message = getattr(tx, "message", None) if tx is not None else None
        if tx is None or meta is None or message is None:
            return None, "missing_transaction_payload"

        err_obj = getattr(meta, "err", None)
        if err_obj is not None and getattr(err_obj, "err", b""):
            return None, "failed_tx"

        account_keys = self._account_keys_from_message(message)
        if not account_keys:
            return None, "missing_account_keys"

        source_program, source_ok = self._resolve_source(
            account_keys=account_keys,
            update_filters=getattr(update, "filters", []),
        )
        if not source_ok:
            return None, "source_not_allowed"

        logs = list(getattr(meta, "log_messages", []))
        if not self._logs_has_swap_signal(logs, source_program=source_program):
            return None, "not_swap"

        header = getattr(message, "header", None)
        num_required_signatures = (
            int(getattr(header, "num_required_signatures", 0)) if header else 0
        )
        signer_count = max(1, num_required_signatures)
        signer_keys = account_keys[:signer_count]
        triggering_wallet = signer_keys[0] if signer_keys else account_keys[0]
        if not triggering_wallet:
            return None, "missing_wallet_context"

        pre_token = self._token_balances_by_owner(
            getattr(meta, "pre_token_balances", []), triggering_wallet
        )
        post_token = self._token_balances_by_owner(
            getattr(meta, "post_token_balances", []), triggering_wallet
        )

        candidates: list[tuple[str, float]] = []
        for mint in set(pre_token) | set(post_token):
            if not self._is_memecoin_candidate_mint(
                mint, self.config.discovery_require_pump_suffix
            ):
                continue
            delta = float(post_token.get(mint, 0.0) - pre_token.get(mint, 0.0))
            if abs(delta) > 0:
                candidates.append((mint, delta))
        if not candidates:
            return None, "no_token_delta"

        if not self.config.discovery_require_pump_suffix:
            pump_candidates = [item for item in candidates if item[0].lower().endswith("pump")]
            if pump_candidates:
                candidates = pump_candidates

        token_mint, token_delta = max(candidates, key=lambda item: abs(item[1]))
        token_amount = abs(float(token_delta))
        if token_amount <= 0:
            return None, "invalid_amounts"

        try:
            wallet_index = account_keys.index(triggering_wallet)
        except ValueError:
            wallet_index = 0
        pre_balances = list(getattr(meta, "pre_balances", []))
        post_balances = list(getattr(meta, "post_balances", []))
        if wallet_index >= len(pre_balances) or wallet_index >= len(post_balances):
            return None, "missing_native_balances"
        pre_lamports = self._to_float(pre_balances[wallet_index])
        post_lamports = self._to_float(post_balances[wallet_index])
        fee_lamports = self._to_float(getattr(meta, "fee", 0)) or 0.0
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

        tracked_involved = tuple(sorted(set(account_keys) & self._tracked_wallet_set))

        provider_created_at = self._proto_timestamp_to_datetime(getattr(update, "created_at", None))
        block_time = provider_created_at or datetime.now(tz=timezone.utc)
        source_slot_raw = getattr(tx_update, "slot", None)
        try:
            source_slot = int(source_slot_raw) if source_slot_raw is not None else None
        except (TypeError, ValueError):
            source_slot = None
        if source_slot is not None and source_slot <= 0:
            source_slot = None

        # Update local quote engine reserves for constant-product AMM programs.
        # PUMP_AMM and RAYDIUM_LAUNCHLAB both use WSOL vault + token vault (SPL),
        # so post_token_balances contains both sides of the pool.
        reference_price_sol: float | None = None
        if self._local_quote_engine is not None and source_program in {
            "PUMP_AMM",
            "RAYDIUM_LAUNCHLAB",
        }:
            try:
                full_account_keys = self._full_account_keys_from_message_and_meta(message, meta)
                self._local_quote_engine.update_from_swap_meta(
                    token_mint=str(token_mint),
                    post_token_balances=list(getattr(meta, "post_token_balances", [])),
                    triggering_wallet=str(triggering_wallet),
                    source_program=source_program,
                    account_keys=full_account_keys,
                    message_instructions=list(getattr(message, "instructions", []) or []),
                    signature=signature,
                )
                reference_price_sol = self._local_quote_engine.mark_price_sol(str(token_mint))
            except Exception:  # noqa: BLE001
                pass

        self._remember_signature(signature)
        parse_completed_at = datetime.now(tz=timezone.utc)
        return CandidateEvent(
            token_mint=str(token_mint),
            signature=signature,
            block_time=block_time,
            triggering_wallet=str(triggering_wallet),
            side=side,
            sol_amount=float(sol_amount),
            token_amount=float(token_amount),
            reference_price_sol=reference_price_sol,
            source_program=source_program or "UNKNOWN",
            tracked_wallets=tracked_involved,
            discovery_source="pair_first",
            provider_created_at=provider_created_at,
            parse_started_at=parse_started_at,
            parse_completed_at=parse_completed_at,
            source_slot=source_slot,
            event_time_source="yellowstone_created_at"
            if provider_created_at is not None
            else "local_fallback_now",
        ), "accepted"

    async def _connect_and_stream(self) -> AsyncIterator[CandidateEvent]:
        """Open one gRPC stream and yield accepted candidate events."""
        if self.discovery_mode != "pair_first":
            raise WebsocketUnavailableError(
                "Chainstack Yellowstone monitor currently supports DISCOVERY_MODE=pair_first only."
            )
        if not self._subscription_targets:
            raise WebsocketUnavailableError("DISCOVERY_ACCOUNT_INCLUDE is empty.")

        self._channel = self._create_grpc_channel()
        stub = geyser_pb2_grpc.GeyserStub(self._channel)
        request = self._build_subscription_request()

        if self._status_callback is not None:
            self._status_callback(
                websocket_connected=True,
                websocket_failure=None,
                websocket_subscribed_count=0,
                websocket_subscription_total=len(self._subscription_targets),
                websocket_subscribed_wallets=[],
                websocket_subscription_target_type=self._subscription_target_type,
                websocket_subscription_targets=list(self._subscription_targets),
                websocket_subscribed_target_count=0,
            )

        if self._status_callback is not None:
            self._status_callback(
                websocket_ready=True,
                status="running",
                monitoring_mode="chainstack_grpc",
                websocket_connected=True,
                websocket_failure=None,
                websocket_subscribed_count=len(self._subscription_targets),
                websocket_subscription_total=len(self._subscription_targets),
                websocket_latest_wallet=self._subscription_targets[-1],
                websocket_subscription_ids={
                    name: program for name, program in self._filter_name_to_program.items()
                },
                websocket_subscribed_wallets=list(self._subscription_targets),
                websocket_subscription_target_type=self._subscription_target_type,
                websocket_subscription_targets=list(self._subscription_targets),
                websocket_subscribed_target_count=len(self._subscription_targets),
                websocket_parse_request_count=0,
                websocket_parse_batch_count=0,
                websocket_parsed_signature_count=0,
                websocket_avg_batch_size=0.0,
                websocket_parse_estimated_credits=0,
            )

        stream = stub.Subscribe(self._request_iterator(request))
        try:
            async for update in stream:
                stream_received_at = datetime.now(tz=timezone.utc)
                self._notification_count += 1
                if self._status_callback is not None:
                    self._status_callback(
                        monitoring_mode="chainstack_grpc",
                        websocket_notification_count=self._notification_count,
                        websocket_last_notification_at=int(time()),
                        websocket_filter_mode="chainstack_yellowstone_grpc_transactions",
                    )

                # Offload protobuf parsing to a thread so the event loop stays
                # free to process pending entries/exits while parsing runs.
                loop = asyncio.get_running_loop()
                event, reason = await loop.run_in_executor(
                    self._parse_executor, self._candidate_from_update, update
                )
                if event is None:
                    self._record_drop(reason, emit_status=False)
                    self._emit_drop_status()
                    continue
                event.stream_received_at = stream_received_at
                if event.parse_completed_at is None:
                    event.parse_completed_at = datetime.now(tz=timezone.utc)

                if event.side == "BUY" and event.sol_amount < self.config.min_trigger_sol:
                    self._record_drop("below_min_trigger_sol", emit_status=False)
                    self._emit_drop_status()
                    continue

                self._candidate_event_count += 1
                if event.tracked_wallets:
                    self._candidate_tracked_wallet_hit_count += 1
                if self._status_callback is not None:
                    self._status_callback(
                        monitoring_mode="chainstack_grpc",
                        status="running",
                        websocket_candidate_event_count=self._candidate_event_count,
                        websocket_candidate_tracked_wallet_hit_count=self._candidate_tracked_wallet_hit_count,
                        websocket_last_candidate_token=event.token_mint,
                        websocket_last_candidate_wallet=event.triggering_wallet,
                        websocket_last_candidate_signature=event.signature,
                        websocket_last_candidate_tracked_wallet_count=len(event.tracked_wallets),
                    )
                yield event
        finally:
            if self._status_callback is not None:
                self._status_callback(
                    websocket_connected=False,
                    websocket_ready=False,
                )
            if self._channel is not None:
                await self._channel.close()
                self._channel = None

    async def events(self) -> AsyncIterator[CandidateEvent]:
        """Yield events with exponential-retry reconnect logic."""
        retry_count = 0
        while True:
            try:
                async for event in self._connect_and_stream():
                    retry_count = 0
                    yield event
                raise RuntimeError("Chainstack gRPC stream closed.")
            except WebsocketUnavailableError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                retry_count += 1
                if self._max_retries > 0 and retry_count > self._max_retries:
                    raise SystemExit(
                        f"Chainstack Yellowstone stream failed after {self._max_retries} retries: {self._error_label(exc)}"
                    ) from exc
                backoff_sec = min(
                    self._backoff_initial_sec * (2 ** (retry_count - 1)),
                    self._backoff_max_sec,
                )
                jitter_sec = random.uniform(0.0, min(1.0, backoff_sec * 0.25))
                sleep_sec = backoff_sec + jitter_sec
                max_retries_label = "∞" if self._max_retries <= 0 else str(self._max_retries)
                error_label = self._error_label(exc)
                self.logger.warning(
                    "Chainstack Yellowstone stream error (%s/%s): %s. Reconnecting in %ss.",
                    retry_count,
                    max_retries_label,
                    error_label,
                    round(sleep_sec, 2),
                )
                if self._status_callback is not None:
                    self._status_callback(
                        status="running",
                        monitoring_mode="chainstack_grpc_retry_wait",
                        websocket_connected=False,
                        websocket_ready=False,
                        websocket_failure=error_label,
                        websocket_retry_count=retry_count,
                        websocket_retry_next_sec=round(sleep_sec, 2),
                        websocket_retries_max=None if self._max_retries <= 0 else self._max_retries,
                    )
                await asyncio.sleep(sleep_sec)

    def close(self) -> None:
        """Shutdown the parse thread pool (call on bot exit)."""
        self._parse_executor.shutdown(wait=False)
