"""Bot runner supporting paper and live trading modes.

The orchestrator that wires together monitoring, strategy, execution, and
portfolio. Single class (``BotRunner``) with ~3.9k lines because the
runtime control flow is genuinely state-heavy: candidate queueing,
cooldowns, regime tracking, manual session controls, exit scheduling,
and live/paper mode routing all live here. See docs/ARCHITECTURE.md for
the bigger picture.

File table of contents (approximate line ranges):

  ~50- ~235   BotRunner.__init__ and per-instance state setup
  ~235- ~440  Notification + status-write helpers, latency-trace plumbing
  ~440- ~810  Manual-control endpoints (end session, new session, manual close)
  ~810-~1100  Entry/exit notification dispatch, exposure accounting
  ~1100-~1980 Entry decision logic: feature snapshot, rule sizing, ML metadata,
              paper-entry round-trip guard, recovery confirmation, entry-quality
              gates, lane gates
  ~1980-~2570 Ranked-candidate queue + per-strategy cooldown bookkeeping
  ~2570-~2670 Stale-sweep ticker + deferred-candidate handling
  ~2670-~3530 Entry execution paths (sniper, wallet, wallet-copy mirrors)
  ~3530-~3850 Main event loop: _process_events, _process_pending_candidates,
              _exit_and_notify, run_forever
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from src.bot.config import (
    BotConfig,
    effective_max_price_impact_pct,
    effective_min_roundtrip_ratio,
)
from src.bot.models import MatchResult, RuntimeFeatures, RuntimeRule
from src.bot.state import BotStatusWriter
from src.execution.live_reconciler import LiveReconciler
from src.execution.live_sell_cache import LiveSellCache
from src.execution.quote_cache import PositionQuoteCache
from src.strategy.local_quote import PumpAMMQuoteEngine
from src.execution.jupiter_client import LAMPORTS_PER_SOL
from src.execution.trade_executor import PaperTradeEstimate, TradeExecutor
from src.ml.exit_predictor import ExitMLPredictor
from src.ml.live_filter import LiveMLFilter
from src.monitoring.helius_ws import HeliusWebsocketMonitor, WebsocketUnavailableError
from src.monitoring.market_regime import MarketRegimeMonitor
from src.monitoring.token_activity import TokenActivityCache
from src.monitoring.wallet_stream import WalletActivityStream
from src.notifications.telegram import TelegramNotifier
from src.portfolio.position_manager import PositionManager
from src.portfolio.rule_performance import RulePerformanceTracker
from src.storage.bot_db import BotDB
from src.storage.event_log import EventLogger
from src.strategy.entry_engine import EntryEngine
from src.strategy.entry_runtime import determine_entry_lane, score_candidate
from src.strategy.exit_engine import ExitEngine
from src.strategy.feature_runtime import build_runtime_features
from src.strategy.risk_manager import RiskManager
from src.strategy.rule_matcher import closest_rule_misses
from src.strategy.rule_selector import select_rule
from src.strategy.rules_loader import load_runtime_rules
from src.strategy.sniper_engine import SniperEngine
from src.strategy.wallet_engine import WalletEngine


class BotRunner:
    """Main trading loop (paper or live)."""

    # Fields that carry only per-event telemetry — safe to debounce.
    # Any _write_status call whose kwargs are a subset of these is throttled to 1/s.
    _STATUS_HOT_FIELDS = frozenset(
        {
            "last_seen_event_at",
            "last_seen_token",
            "last_seen_wallet",
            "last_seen_side",
            "last_seen_event_time_source",
            "last_seen_provider_created_at",
            "last_seen_stream_received_at",
            "last_seen_parse_completed_at",
            "last_seen_source_slot",
            "pending_candidate_count",
            "queued_candidates",
        }
    )
    _STATUS_DEBOUNCE_SEC = 1.0

    def __init__(
        self,
        config: BotConfig,
        db: BotDB,
        event_log: EventLogger,
        limit_wallets: int | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.event_log = event_log
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ml_filter = LiveMLFilter(config=config, event_log=event_log)
        self.exit_predictor = ExitMLPredictor(
            mode=getattr(config, "ml_exit_mode", "shadow"),
            model_path=getattr(config, "ml_exit_model_path", None),
            samples_path=getattr(config, "ml_exit_samples_path", None),
            sniper_threshold=float(getattr(config, "ml_exit_sniper_threshold", 0.40)),
            main_threshold=float(getattr(config, "ml_exit_main_threshold", 0.45)),
            min_samples=int(getattr(config, "ml_exit_min_samples", 50)),
            retrain_every=int(getattr(config, "ml_exit_retrain_every", 25)),
            sniper_max_hold_sec=float(getattr(config, "sniper_max_hold_sec", 75.0)),
            event_log=event_log,
        )
        self.stream = WalletActivityStream(config, limit_wallets=limit_wallets)
        self.token_cache = TokenActivityCache(
            wallet_scores=self.stream.wallet_scores,
            price_outlier_min_samples=config.price_outlier_min_samples,
            price_outlier_median_window=config.price_outlier_median_window,
            price_outlier_max_multiple=config.price_outlier_max_multiple,
            price_outlier_confirm_signatures=config.price_outlier_confirm_signatures,
            price_outlier_confirm_window_sec=config.price_outlier_confirm_window_sec,
            price_outlier_confirm_tolerance=config.price_outlier_confirm_tolerance,
            db=db,
        )
        self.rules, self.regime_metadata = load_runtime_rules(config)
        # Main strategy must not see rules that belong exclusively to the sniper allowlist.
        # Those rules have wrong exit profiles (sniper 75s hold) when run under main logic.
        _sniper_only: set[str] = set(getattr(config, "sniper_rule_ids", ()) or ())
        _main_allow: set[str] = set(getattr(config, "main_rule_ids", ()) or ())
        # Rules explicitly named in BOTH allowlists are dual-lane by intent;
        # drop them from the sniper-exclusion so they reach the main-lane filter.
        _sniper_only = _sniper_only - _main_allow
        self.main_rules = [r for r in self.rules if r.rule_id not in _sniper_only]
        # When MAIN_RULE_IDS is set, restrict the main lane to that allow-list
        # (mirror of SNIPER_RULE_IDS). Lets the mature-pair pack run in
        # isolation without the 80+ fresh-momentum rules from the primary
        # pack firing on graduated tokens.
        if _main_allow:
            self.main_rules = [r for r in self.main_rules if r.rule_id in _main_allow]
        self.sniper_engine = SniperEngine(config)
        self.wallet_engine = WalletEngine(config)
        self.position_manager = PositionManager(db)
        self.rule_performance = RulePerformanceTracker(db)
        self.risk_manager = RiskManager(config, db)
        self.local_quote_engine = PumpAMMQuoteEngine()

        # --- Execution layer ---
        self.trade_executor = TradeExecutor(config, local_quote_engine=self.local_quote_engine)

        self.entry_engine = EntryEngine(
            db,
            self.position_manager,
            self.rule_performance,
            event_log,
            trade_executor=self.trade_executor,
            risk_manager=self.risk_manager,
        )
        self.quote_cache = PositionQuoteCache(self.trade_executor)
        # Live sell cache: pre-builds Jupiter sell TXs every 2s per open position.
        # Only started in live mode — paper trading never uses it.
        self.live_sell_cache: LiveSellCache | None = None
        self.live_reconciler: LiveReconciler | None = None
        if self.trade_executor.live and self.trade_executor._live_executor is not None:
            self.live_sell_cache = LiveSellCache(self.trade_executor._live_executor)
            if bool(getattr(config, "live_reconciler_enabled", False)):
                try:
                    _wallet = str(self.trade_executor._live_executor.signer.get_public_key() or "")
                except Exception:  # noqa: BLE001
                    _wallet = ""
                if _wallet:
                    self.live_reconciler = LiveReconciler(
                        db=db,
                        broadcaster=self.trade_executor._live_executor.broadcaster,
                        event_log=event_log,
                        wallet_pubkey=_wallet,
                        interval_sec=float(getattr(config, "live_reconciler_interval_sec", 60.0)),
                        drift_threshold_pct=float(
                            getattr(config, "live_reconciler_drift_threshold_pct", 0.10)
                        ),
                        force_close_threshold_pct=float(
                            getattr(
                                config,
                                "live_reconciler_force_close_threshold_pct",
                                0.90,
                            )
                        ),
                    )
        self.exit_engine = ExitEngine(
            db,
            self.position_manager,
            self.rule_performance,
            self.risk_manager,
            event_log,
            trade_executor=self.trade_executor,
            exit_predictor=self.exit_predictor,
            quote_cache=self.quote_cache,
            local_quote_engine=self.local_quote_engine,
            live_sell_cache=self.live_sell_cache,
        )
        if self.live_reconciler is not None:
            self.live_reconciler.set_force_close_callback(
                lambda position, reason, last_error: self.exit_engine.force_close_drifted_position(
                    position, reason=reason, last_error=last_error
                )
            )
        self.notifier = TelegramNotifier(config)
        self.last_seen_signatures: dict[str, str] = {}
        self.ws_monitor = HeliusWebsocketMonitor(
            config,
            list(self.stream.iter_wallets()),
            status_callback=self._write_status,
            local_quote_engine=self.local_quote_engine,
        )
        self.status = BotStatusWriter(config.bot_status_path)
        self._monitoring_mode = "chainstack_grpc_primary"
        self._status_processed_events = 0
        self._candidate_cooldowns: dict[str, datetime] = {}
        self._candidate_cooldown_logged_at: dict[str, datetime] = {}
        self._candidate_defer_logged_at: dict[str, datetime] = {}
        self._sniper_candidate_cooldowns: dict[str, datetime] = {}
        self._wallet_candidate_cooldowns: dict[str, datetime] = {}
        self._pending_candidates: dict[str, dict[str, Any]] = {}
        self._ranked_candidates: dict[str, dict[str, Any]] = {}
        self._last_stale_sweep_at: datetime | None = None
        self._post_close_watch: dict[str, dict[str, Any]] = {}
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._exit_tasks_by_token: dict[str, asyncio.Task[Any]] = {}
        self._pending_exit_inputs: dict[str, tuple[RuntimeFeatures, datetime]] = {}
        self._market_regime_task: asyncio.Task[Any] | None = None
        self._control_lock = threading.Lock()
        self._notify_lock = threading.Lock()
        self._entries_paused = False
        self._market_regime = MarketRegimeMonitor(
            enabled=config.market_regime_enabled,
            win_rate_window=config.market_regime_win_rate_window,
            min_win_rate=config.market_regime_min_win_rate,
            bootstrap_positions=config.market_regime_bootstrap_positions,
            min_candidates_5min=config.market_regime_min_candidates_5min,
            sol_enabled=config.market_regime_sol_enabled,
            sol_drop_threshold=config.market_regime_sol_drop_threshold,
            pause_cooldown_sec=config.market_regime_pause_cooldown_sec,
            event_log=event_log,
        )
        self._session_mode = "active"
        self._end_session_pending = False
        self._end_session_reason: str | None = None
        self._end_session_requested_at: str | None = None
        self._new_session_pending = False
        self._new_session_requested_at: str | None = None
        self._new_session_source: str | None = None
        self._last_status_write_ts: float = 0.0
        self._last_cleanup_ts: float = 0.0

    def _maybe_broadcast_wallet_activity(self, event: Any) -> None:
        """No-op kept for call-site compatibility.

        Originally fanned wallet-activity events out to a streaming broadcaster
        consumed by an external dashboard. The broadcaster path was removed
        from this distribution; the local dashboard reads SQLite + JSONL
        directly instead.
        """
        return

    @property
    def _mode_label(self) -> str:
        return self.trade_executor.mode_label

    def _notify_startup(self) -> None:
        """Send a startup heartbeat."""
        mode = self._mode_label
        ml_status = self.ml_filter.status_fields()
        self.notifier.send(
            "\n".join(
                [
                    "Bot up and running",
                    f"mode={mode.lower()}",
                    f"tracked_wallets={len(self.stream.wallet_df)}",
                    f"active_rules={len(self.rules)}",
                    f"main={'enabled' if self.config.enable_main_strategy else 'disabled'}",
                    f"sniper={'enabled' if self.sniper_engine.enabled else 'disabled'}",
                    f"ml_mode={ml_status.get('ml_mode')}",
                    f"ml_ready={ml_status.get('ml_model_ready')}",
                    f"ml_samples={ml_status.get('ml_training_samples')}",
                    "monitoring=chainstack_yellowstone_grpc_primary",
                ]
            )
        )

    def _handle_loop_exception(
        self, loop: asyncio.AbstractEventLoop, context: dict[str, Any]
    ) -> None:
        """Log loop exceptions without task-stack introspection.

        Python 3.11.1 on macOS has been observed crashing inside Task.get_stack()
        during default asyncio exception formatting. Avoid stringifying Task/Future
        objects here; log only stable fields.
        """
        message = str(context.get("message") or "asyncio loop exception")
        exc = context.get("exception")
        future = context.get("future") or context.get("task")
        future_name: str | None = None
        if future is not None:
            try:
                getter = getattr(future, "get_name", None)
                if callable(getter):
                    future_name = str(getter())
            except Exception:  # noqa: BLE001
                future_name = None
            if not future_name:
                future_name = type(future).__name__
        if isinstance(exc, BaseException):
            self.logger.error(
                "Asyncio loop exception: %s | future=%s | exc=%s",
                message,
                future_name,
                exc,
                exc_info=exc,
            )
        else:
            self.logger.error(
                "Asyncio loop exception: %s | future=%s | context_keys=%s",
                message,
                future_name,
                sorted(context.keys()),
            )

    def _track_background_task(
        self,
        task: asyncio.Task[Any],
        *,
        token_mint: str | None = None,
    ) -> asyncio.Task[Any]:
        """Retain one background task and clean it up safely on completion."""
        self._background_tasks.add(task)

        def _done_callback(done: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(done)
            if token_mint is not None:
                current = self._exit_tasks_by_token.get(token_mint)
                if current is done:
                    self._exit_tasks_by_token.pop(token_mint, None)
            try:
                done.result()
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Background task failed: %s", exc)

        task.add_done_callback(_done_callback)
        return task

    def _schedule_exit_task(self, features: RuntimeFeatures, arrival_time: datetime) -> None:
        """Run at most one exit task per token and coalesce bursty updates."""
        token_mint = str(features.token_mint or "")
        if not token_mint:
            return
        existing = self._exit_tasks_by_token.get(token_mint)
        if existing is not None and not existing.done():
            self._pending_exit_inputs[token_mint] = (features, arrival_time)
            return
        task = asyncio.create_task(
            self._exit_worker(token_mint, features, arrival_time),
            name=f"exit:{token_mint[:8]}",
        )
        self._exit_tasks_by_token[token_mint] = task
        self._track_background_task(task, token_mint=token_mint)

    def _write_status(self, **fields: object) -> None:
        """Persist dashboard-visible bot runtime status.

        Hot-path calls (per-event telemetry only) are throttled to once per second.
        The dashboard refreshes at DASHBOARD_REFRESH_SEC (5s) so sub-second writes
        are wasted I/O and DB queries.
        """
        if "monitoring_mode" in fields and fields["monitoring_mode"]:
            self._monitoring_mode = str(fields["monitoring_mode"])

        # Debounce: skip if this is a hot-path-only call and written recently.
        now_ts = time.monotonic()
        if (
            fields.keys() <= self._STATUS_HOT_FIELDS
            and (now_ts - self._last_status_write_ts) < self._STATUS_DEBOUNCE_SEC
        ):
            return
        self._last_status_write_ts = now_ts

        ml_status = self.ml_filter.status_fields()
        active_session = self.db.active_session()
        # Fetch open positions once and derive all metrics from the snapshot.
        # Previously list_open_positions() was called 4 separate times per write.
        open_positions = self.position_manager.list_open_positions()
        sniper_positions = [p for p in open_positions if self._position_strategy_id(p) == "sniper"]
        wallet_positions = [p for p in open_positions if self._position_strategy_id(p) == "wallet"]
        total_exposure = sum(float(p["size_sol"]) for p in open_positions)
        sniper_exposure = sum(float(p["size_sol"]) for p in sniper_positions)
        wallet_exposure = sum(float(p["size_sol"]) for p in wallet_positions)
        payload = {
            "mode": self._mode_label,
            "monitoring_mode": self._monitoring_mode,
            "tracked_wallets": len(self.stream.wallet_df),
            "tracked_wallet_features_enabled": bool(self.config.tracked_wallet_features_enabled),
            "tracked_wallet_list": list(self.stream.iter_wallets()),
            "discovery_mode": self.config.discovery_mode,
            "discovery_account_include": list(self.config.discovery_account_include),
            "active_rules": len(self.rules),
            "main_enabled": bool(self.config.enable_main_strategy),
            "sniper_enabled": self.sniper_engine.enabled,
            "sniper_open_positions": len(sniper_positions),
            "sniper_exposure_sol": sniper_exposure,
            "wallet_enabled": self.wallet_engine.enabled,
            "wallet_open_positions": len(wallet_positions),
            "wallet_exposure_sol": wallet_exposure,
            "open_positions": len(open_positions),
            "total_exposure_sol": total_exposure,
            "processed_events": self._status_processed_events,
            "queued_candidates": len(self._ranked_candidates),
            "entries_paused": bool(self._entries_paused),
            "session_mode": self._session_mode,
            "end_session_requested_at": self._end_session_requested_at,
            "new_session_requested_at": self._new_session_requested_at,
            "active_session_id": int(active_session["id"]) if active_session is not None else None,
            "active_session_started_at": active_session["started_at"]
            if active_session is not None
            else None,
            "active_session_label": active_session["label"] if active_session is not None else None,
            **ml_status,
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        payload.update(fields)
        self.status.update(**payload)

    @staticmethod
    def _normalize_dt(value: datetime | None) -> datetime | None:
        """Return one UTC-aware datetime or ``None``."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @classmethod
    def _dt_to_iso(cls, value: datetime | None) -> str | None:
        """Return one normalized ISO timestamp or ``None``."""
        normalized = cls._normalize_dt(value)
        return normalized.isoformat() if normalized is not None else None

    @classmethod
    def _ms_between(cls, start: datetime | None, end: datetime | None) -> float | None:
        """Return elapsed milliseconds between two datetimes."""
        start_dt = cls._normalize_dt(start)
        end_dt = cls._normalize_dt(end)
        if start_dt is None or end_dt is None:
            return None
        return max(0.0, (end_dt - start_dt).total_seconds() * 1000.0)

    @staticmethod
    def _copy_trace(trace: dict[str, Any] | None) -> dict[str, Any]:
        """Return one shallow copy of a trace payload."""
        return dict(trace) if isinstance(trace, dict) else {}

    def _new_event_latency_trace(
        self,
        *,
        event: Any,
        arrival_time: datetime,
        source: str,
    ) -> dict[str, Any]:
        """Build the initial per-event latency trace."""
        event_time = self._normalize_dt(getattr(event, "block_time", None))
        provider_created_at = self._normalize_dt(getattr(event, "provider_created_at", None))
        stream_received_at = self._normalize_dt(getattr(event, "stream_received_at", None))
        parse_started_at = self._normalize_dt(getattr(event, "parse_started_at", None))
        parse_completed_at = self._normalize_dt(getattr(event, "parse_completed_at", None))
        trace = {
            "strategy_path": "main_candidate",
            "event_source": str(source),
            "event_signature": str(getattr(event, "signature", "") or ""),
            "token_mint": str(getattr(event, "token_mint", "") or ""),
            "event_time_source": str(getattr(event, "event_time_source", "") or "unknown"),
            "event_effective_time": self._dt_to_iso(event_time),
            "event_block_time": self._dt_to_iso(event_time),
            "event_provider_created_at": self._dt_to_iso(provider_created_at),
            "event_stream_received_at": self._dt_to_iso(stream_received_at),
            "event_parse_started_at": self._dt_to_iso(parse_started_at),
            "event_parse_completed_at": self._dt_to_iso(parse_completed_at),
            "event_runner_arrival_at": self._dt_to_iso(arrival_time),
            "event_arrival_at": self._dt_to_iso(arrival_time),
            "event_source_slot": getattr(event, "source_slot", None),
            "event_effective_to_runner_arrival_ms": self._ms_between(event_time, arrival_time),
            "source_program": getattr(event, "source_program", None),
            "discovery_source": getattr(event, "discovery_source", None),
        }
        if str(getattr(event, "event_time_source", "") or "").lower() == "solana_block_time":
            trace["event_block_to_arrival_ms"] = self._ms_between(event_time, arrival_time)
        else:
            trace["event_block_to_arrival_ms"] = None
        if provider_created_at is not None:
            trace["event_provider_created_to_stream_receive_ms"] = self._ms_between(
                provider_created_at, stream_received_at
            )
            trace["event_provider_created_to_parse_start_ms"] = self._ms_between(
                provider_created_at, parse_started_at
            )
            trace["event_provider_created_to_parse_complete_ms"] = self._ms_between(
                provider_created_at, parse_completed_at
            )
            trace["event_provider_created_to_runner_arrival_ms"] = self._ms_between(
                provider_created_at, arrival_time
            )
        if stream_received_at is not None:
            trace["event_stream_receive_to_parse_start_ms"] = self._ms_between(
                stream_received_at, parse_started_at
            )
            trace["event_stream_receive_to_parse_complete_ms"] = self._ms_between(
                stream_received_at, parse_completed_at
            )
            trace["event_stream_receive_to_runner_arrival_ms"] = self._ms_between(
                stream_received_at, arrival_time
            )
        if parse_started_at is not None:
            trace["event_parse_start_to_complete_ms"] = self._ms_between(
                parse_started_at, parse_completed_at
            )
            trace["event_parse_start_to_runner_arrival_ms"] = self._ms_between(
                parse_started_at, arrival_time
            )
        if parse_completed_at is not None:
            trace["event_parse_complete_to_runner_arrival_ms"] = self._ms_between(
                parse_completed_at, arrival_time
            )
        return trace

    @staticmethod
    def _set_feature_trace(
        features: RuntimeFeatures, key: str, trace: dict[str, Any]
    ) -> dict[str, Any]:
        """Attach one trace payload to runtime features."""
        features.raw[key] = dict(trace)
        return features.raw[key]

    def request_end_session(self, source: str = "dashboard") -> dict[str, object]:
        """Pause new entries and request a graceful close of all open positions."""
        now = datetime.now(tz=timezone.utc)
        with self._control_lock:
            self._entries_paused = True
            self._session_mode = "ending"
            self._end_session_pending = True
            self._end_session_reason = f"session_end:{source}"
            self._end_session_requested_at = now.isoformat()

        if self._event_loop is not None and self._event_loop.is_running():
            self._event_loop.call_soon_threadsafe(self._apply_end_session_if_requested)

        return {
            "ok": True,
            "session_mode": self._session_mode,
            "entries_paused": self._entries_paused,
            "requested_at": self._end_session_requested_at,
            "reason": self._end_session_reason,
        }

    def request_new_session(self, source: str = "dashboard") -> dict[str, object]:
        """Request a new dashboard session boundary without deleting history."""
        now = datetime.now(tz=timezone.utc)
        with self._control_lock:
            self._new_session_pending = True
            self._new_session_requested_at = now.isoformat()
            self._new_session_source = source

        if self._event_loop is not None and self._event_loop.is_running():
            self._event_loop.call_soon_threadsafe(self._apply_new_session_if_requested)

        return {
            "ok": True,
            "requested_at": self._new_session_requested_at,
            "source": source,
        }

    def request_manual_close(self, token_mint: str, source: str = "dashboard") -> dict[str, object]:
        """Force-close every open position for a given mint (dashboard Sell Now)."""
        token_mint = str(token_mint or "").strip()
        if not token_mint:
            return {"ok": False, "error": "token_mint_required"}

        positions = self.position_manager.list_open_positions_for_token(token_mint)
        if not positions:
            return {"ok": False, "error": "no_open_position", "token_mint": token_mint}

        reason = f"manual_close:{source}"
        requested_at = datetime.now(tz=timezone.utc).isoformat()
        self.event_log.log(
            "manual_close_requested",
            {
                "token_mint": token_mint,
                "source": source,
                "position_count": len(positions),
                "requested_at": requested_at,
            },
        )

        def _apply() -> None:
            now = datetime.now(tz=timezone.utc)
            closed = 0
            failed = 0
            for position in positions:
                features = self._build_force_close_features(position, now)
                if features is None:
                    failed += 1
                    continue
                if features.raw is not None:
                    trace = features.raw.get("__exit_latency_trace") or {}
                    trace["strategy_path"] = "manual_close"
                    trace["exit_trigger"] = "manual_close"
                    trace["manual_close_source"] = source
                    features.raw["__exit_latency_trace"] = trace
                ok = self.exit_engine.force_close_position(position, features, reason=reason)
                if ok:
                    closed += 1
                else:
                    failed += 1
            self._notify_exit_events()
            self.event_log.log(
                "manual_close_applied",
                {
                    "token_mint": token_mint,
                    "source": source,
                    "closed_positions": closed,
                    "failed_positions": failed,
                    "applied_at": now.isoformat(),
                },
            )

        if self._event_loop is not None and self._event_loop.is_running():
            self._event_loop.call_soon_threadsafe(_apply)
            dispatched = True
        else:
            # Loop not running — execute synchronously as a best-effort fallback.
            _apply()
            dispatched = False

        return {
            "ok": True,
            "token_mint": token_mint,
            "position_count": len(positions),
            "source": source,
            "requested_at": requested_at,
            "dispatched": dispatched,
        }

    def _build_force_close_features(
        self, position: dict[str, Any], now: datetime
    ) -> RuntimeFeatures | None:
        """Build a minimal feature snapshot for forced session-end exits."""
        features = self._build_stale_sweep_features(position, now)
        if features is not None:
            features.raw["__exit_latency_trace"] = {
                "strategy_path": "force_close",
                "event_source": "session_end",
                "token_mint": str(position.get("token_mint") or ""),
                "exit_trigger": "session_end",
                "force_close_requested_at": now.isoformat(),
            }
            return features
        token_mint = str(position.get("token_mint") or "")
        if not token_mint:
            return None
        try:
            entry_price = float(position.get("entry_price_sol", 0.0) or 0.0)
        except (TypeError, ValueError):
            entry_price = 0.0
        if entry_price <= 0:
            return None
        return RuntimeFeatures(
            token_mint=token_mint,
            entry_time=now,
            entry_price_sol=entry_price,
            token_age_sec=None,
            wallet_cluster_30s=0,
            wallet_cluster_120s=0,
            volume_sol_30s=0.0,
            volume_sol_60s=0.0,
            tx_count_30s=0,
            tx_count_60s=0,
            price_change_30s=0.0,
            price_change_60s=0.0,
            triggering_wallet=str(position.get("triggering_wallet") or ""),
            triggering_wallet_score=float(position.get("triggering_wallet_score", 0.0) or 0.0),
            aggregated_wallet_score=0.0,
            tracked_wallet_present_60s=False,
            tracked_wallet_count_60s=0,
            tracked_wallet_score_sum_60s=0.0,
            raw={
                "__forced_session_end": True,
                "__exit_latency_trace": {
                    "strategy_path": "force_close",
                    "event_source": "session_end",
                    "token_mint": token_mint,
                    "exit_trigger": "session_end",
                    "force_close_requested_at": now.isoformat(),
                },
            },
        )

    def _apply_end_session_if_requested(self) -> None:
        """Close all open positions when an end-session request is pending."""
        now = datetime.now(tz=timezone.utc)
        with self._control_lock:
            if not self._end_session_pending:
                return
            reason = self._end_session_reason or "session_end:manual"
            self._end_session_pending = False
            self._session_mode = "ending"
        self._ranked_candidates.clear()
        self._pending_candidates.clear()

        self.event_log.log(
            "session_end_requested",
            {
                "reason": reason,
                "requested_at": self._end_session_requested_at,
            },
        )

        open_positions = self.position_manager.list_open_positions()
        closed = 0
        failed = 0
        for position in open_positions:
            features = self._build_force_close_features(position, now)
            if features is None:
                failed += 1
                continue
            ok = self.exit_engine.force_close_position(position, features, reason=reason)
            if ok:
                closed += 1
            else:
                failed += 1

        with self._control_lock:
            self._session_mode = "ended"

        self._notify_exit_events()
        self.db.end_active_session(ended_at=now.isoformat())
        self.event_log.log(
            "session_end_applied",
            {
                "reason": reason,
                "open_positions_seen": len(open_positions),
                "closed_positions": closed,
                "failed_positions": failed,
                "applied_at": now.isoformat(),
            },
        )
        self._write_status(
            session_mode=self._session_mode,
            entries_paused=True,
            end_session_applied_at=now.isoformat(),
            end_session_close_attempts=len(open_positions),
            end_session_closed=closed,
            end_session_failed=failed,
        )

    def _apply_new_session_if_requested(self) -> None:
        """Activate a new session boundary for dashboard/session-scoped metrics."""
        now = datetime.now(tz=timezone.utc)
        with self._control_lock:
            if not self._new_session_pending:
                return
            requested_at = self._new_session_requested_at
            source = self._new_session_source or "dashboard"
            self._new_session_pending = False

        open_positions = self.position_manager.list_open_positions()
        if open_positions:
            self.event_log.log(
                "session_new_rejected",
                {
                    "reason": "open_positions_exist",
                    "open_positions": len(open_positions),
                    "requested_at": requested_at,
                    "source": source,
                },
            )
            self._write_status(
                session_new_failed_at=now.isoformat(),
                session_new_failure_reason="open_positions_exist",
                session_new_failure_open_positions=len(open_positions),
            )
            return

        label = f"session_{now.strftime('%Y%m%d_%H%M%S')}"
        session = self.db.start_new_session(label=label)

        with self._control_lock:
            self._entries_paused = False
            self._session_mode = "active"
            self._end_session_pending = False
            self._end_session_reason = None
            self._end_session_requested_at = None

        self._candidate_cooldowns.clear()
        self._candidate_cooldown_logged_at.clear()
        self._candidate_defer_logged_at.clear()
        self._sniper_candidate_cooldowns.clear()
        self._wallet_candidate_cooldowns.clear()
        self._pending_candidates.clear()
        self._ranked_candidates.clear()
        self._status_processed_events = 0

        self.event_log.log(
            "session_new_started",
            {
                "session_id": int(session["id"]),
                "session_started_at": session["started_at"],
                "session_label": session["label"],
                "requested_at": requested_at,
                "source": source,
            },
        )
        self._write_status(
            session_mode=self._session_mode,
            entries_paused=False,
            session_new_applied_at=now.isoformat(),
            active_session_id=int(session["id"]),
            active_session_started_at=session["started_at"],
            active_session_label=session["label"],
            end_session_applied_at=None,
            end_session_close_attempts=0,
            end_session_closed=0,
            end_session_failed=0,
        )

    def _notify_entry(self, position, matched_rule_ids: list[str]) -> None:
        """Send entry notification."""
        mode = self._mode_label
        strategy_id = str(getattr(position, "strategy_id", "main") or "main")
        entry_fee_sol = float(position.metadata.get("paper_entry_fee_sol", 0.0) or 0.0)
        entry_cost_basis_sol = float(
            position.metadata.get("paper_entry_cost_basis_sol", position.size_sol)
            or position.size_sol
        )
        self.notifier.send(
            "\n".join(
                [
                    f"{mode} trade opened",
                    f"strategy={strategy_id}",
                    f"token={position.token_mint}",
                    f"size_sol={position.size_sol:.4f}",
                    f"rule={position.selected_rule_id}",
                    f"regime={position.selected_regime}",
                    f"matched_rules={','.join(matched_rule_ids)}",
                    f"wallet={position.triggering_wallet}",
                    f"net_entry_cost_basis_sol={entry_cost_basis_sol:.6f}",
                    f"entry_fee_sol={entry_fee_sol:.6f}",
                    f"tx_sig={position.metadata.get('tx_signature', 'N/A')}",
                ]
            )
        )

    def _claim_execution_notification(self, execution_id: int, reason: str | None) -> bool:
        """Atomically claim one exit execution row for notification."""
        if reason is None:
            cursor = self.db.execute(
                """
                UPDATE executions
                SET reason = 'notifying:'
                WHERE id = ?
                  AND reason IS NULL
                """,
                (execution_id,),
            )
            return cursor.rowcount == 1
        cursor = self.db.execute(
            """
            UPDATE executions
            SET reason = ?
            WHERE id = ?
              AND reason = ?
              AND COALESCE(reason, '') NOT LIKE 'notified:%'
              AND COALESCE(reason, '') NOT LIKE 'notifying:%'
            """,
            (f"notifying:{reason}", execution_id, reason),
        )
        return cursor.rowcount == 1

    def _restore_execution_notification_reason(self, execution_id: int, reason: str | None) -> None:
        """Restore one exit execution row to its pre-claim reason."""
        if reason is None:
            self.db.execute(
                """
                UPDATE executions
                SET reason = NULL
                WHERE id = ?
                """,
                (execution_id,),
            )
            return
        self.db.execute(
            """
            UPDATE executions
            SET reason = ?
            WHERE id = ?
            """,
            (reason, execution_id),
        )

    def _notify_exit_events(self) -> None:
        """Send notifications for newly recorded exit executions."""
        mode_db = "live" if self.trade_executor.live else "paper"
        mode = self._mode_label
        with self._notify_lock:
            rows = self.db.fetchall(
                """
                SELECT id, token_mint, strategy_id, size_sol, price_sol, status, reason, tx_signature, created_at
                FROM executions
                WHERE action = 'SELL'
                  AND mode = ?
                  AND COALESCE(reason, '') NOT LIKE 'notified:%'
                  AND COALESCE(reason, '') NOT LIKE 'notifying:%'
                ORDER BY id
                """,
                (mode_db,),
            )
            for row in rows:
                row_dict = dict(row)
                original_reason = row_dict.get("reason")
                display_reason = str(original_reason or "")
                if not self._claim_execution_notification(int(row_dict["id"]), original_reason):
                    continue
                try:
                    strategy_id = row_dict.get("strategy_id") or "main"
                    matching_position = self.db.fetchone(
                        """
                        SELECT id, token_mint, strategy_id, selected_rule_id, selected_regime, realized_pnl_sol, metadata_json, status
                        FROM positions
                        WHERE token_mint = ? AND COALESCE(strategy_id, 'main') = ?
                        ORDER BY id DESC
                        LIMIT 1
                        """,
                        (row_dict["token_mint"], strategy_id),
                    )
                    rule_id = (
                        matching_position["selected_rule_id"] if matching_position else "unknown"
                    )
                    regime = (
                        matching_position["selected_regime"] if matching_position else "unknown"
                    )
                    realized_pnl = (
                        float(matching_position["realized_pnl_sol"]) if matching_position else 0.0
                    )
                    metadata: dict[str, Any] = {}
                    if matching_position is not None:
                        try:
                            metadata = json.loads(str(matching_position["metadata_json"] or "{}"))
                        except Exception:  # noqa: BLE001
                            metadata = {}
                    paper_fees_cum_sol = float(
                        metadata.get("paper_cumulative_fees_sol", 0.0) or 0.0
                    )
                    paper_last_exit_fee_sol = float(
                        metadata.get("paper_last_sell_fee_sol", 0.0) or 0.0
                    )

                    if row_dict.get("status") == "CLOSED":
                        closed_position = self.db.fetchone(
                            """
                            SELECT
                                id, token_mint, strategy_id, selected_rule_id, selected_regime,
                                entry_time, entry_price_sol, size_sol, amount_received,
                                triggering_wallet, realized_pnl_sol, metadata_json, status
                            FROM positions
                            WHERE token_mint = ? AND COALESCE(strategy_id, 'main') = ? AND status = 'CLOSED'
                            ORDER BY id DESC
                            LIMIT 1
                            """,
                            (row_dict["token_mint"], strategy_id),
                        )
                        if closed_position is not None:
                            closed_pos_dict = dict(closed_position)
                            self.ml_filter.record_closed_position(closed_pos_dict)

                            # Label exit predictor tick samples and trigger retraining
                            try:
                                cp_meta: dict = {}
                                try:
                                    cp_meta = json.loads(
                                        str(closed_pos_dict.get("metadata_json") or "{}")
                                    )
                                except Exception:
                                    pass
                                self.exit_predictor.record_closed_position(
                                    position_id=int(closed_pos_dict["id"]),
                                    exit_reason=cp_meta.get("last_exit_reason", ""),
                                    realized_pnl_sol=float(
                                        closed_pos_dict.get("realized_pnl_sol") or 0.0
                                    ),
                                    strategy_id=str(closed_pos_dict.get("strategy_id") or "main"),
                                )
                            except Exception as _exc:
                                self.logger.debug(
                                    "exit_predictor.record_closed_position failed: %s",
                                    _exc,
                                )

                            # Feed outcome into market regime monitor (main strategy only)
                            _closed_strategy = str(closed_pos_dict.get("strategy_id") or "main")
                            if "sniper" not in _closed_strategy:
                                self._market_regime.record_position_closed(
                                    pnl_sol=float(closed_pos_dict.get("realized_pnl_sol") or 0.0)
                                )

                            if cp_meta.get("wallet_copy"):
                                self._apply_copy_score_penalty(closed_pos_dict, cp_meta)

                            # Session-scoped burn: any losing close blacklists
                            # the token for the rest of the session. Prevents
                            # re-entering a dying token once its 30-min cooldown
                            # expires.
                            _realized = float(closed_pos_dict.get("realized_pnl_sol") or 0.0)
                            if _realized < 0:
                                _burn_mint = str(closed_pos_dict.get("token_mint") or "")
                                if _burn_mint and not self.risk_manager.is_burned(_burn_mint):
                                    self.risk_manager.mark_burned(
                                        _burn_mint,
                                        reason=str(cp_meta.get("last_exit_reason") or ""),
                                    )
                                    self.event_log.log(
                                        "token_burned_session",
                                        {
                                            "token_mint": _burn_mint,
                                            "position_id": int(closed_pos_dict.get("id") or 0),
                                            "strategy_id": _closed_strategy,
                                            "realized_pnl_sol": round(_realized, 6),
                                            "exit_reason": cp_meta.get("last_exit_reason"),
                                        },
                                    )

                            # Remove from live sell cache on close.
                            if self.live_sell_cache is not None:
                                self.live_sell_cache.unregister(row_dict["token_mint"])

                            # Register for 120s post-close observation
                            _now = datetime.now(tz=timezone.utc)
                            self._post_close_watch[row_dict["token_mint"]] = {
                                "position_id": int(closed_pos_dict["id"]),
                                "strategy_id": strategy_id,
                                "close_time": _now,
                                "exit_pnl_sol": float(
                                    closed_pos_dict.get("realized_pnl_sol") or 0.0
                                ),
                                "exit_reason": cp_meta.get("last_exit_reason", ""),
                                "exit_price_sol": float(row_dict.get("price_sol") or 0.0),
                                "snaps_taken": set(),
                            }

                    self.notifier.send(
                        "\n".join(
                            [
                                f"{mode} trade exit",
                                f"strategy={strategy_id}",
                                f"token={row_dict['token_mint']}",
                                f"rule={rule_id}",
                                f"regime={regime}",
                                f"size_sol={float(row_dict['size_sol']):.4f}",
                                f"reason={display_reason}",
                                f"net_realized_pnl_sol={realized_pnl:.6f}",
                                f"paper_fees_cum_sol={paper_fees_cum_sol:.6f}",
                                f"paper_last_exit_fee_sol={paper_last_exit_fee_sol:.6f}",
                                f"tx_sig={row_dict.get('tx_signature') or 'N/A'}",
                            ]
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(
                        "exit notification failed for execution %s: %s",
                        row_dict["id"],
                        exc,
                    )
                    self._restore_execution_notification_reason(
                        int(row_dict["id"]), original_reason
                    )
                    continue
                self.db.execute(
                    "UPDATE executions SET reason = ? WHERE id = ?",
                    (f"notified:{display_reason}", row_dict["id"]),
                )

    def _position_strategy_id(self, position: dict[str, Any]) -> str:
        """Return strategy id for one persisted position row."""
        strategy_id = position.get("strategy_id")
        if strategy_id:
            return str(strategy_id)
        try:
            metadata = json.loads(position.get("metadata_json") or "{}")
        except Exception:  # noqa: BLE001
            metadata = {}
        return str(metadata.get("strategy_id") or "main")

    def _strategy_open_positions(self, strategy_id: str) -> list[dict[str, Any]]:
        """Return open positions for one strategy id."""
        target = str(strategy_id or "main")
        return [
            row
            for row in self.position_manager.list_open_positions()
            if self._position_strategy_id(row) == target
        ]

    def _strategy_exposure(self, strategy_id: str) -> float:
        """Return open exposure for one strategy id."""
        return sum(
            float(position.get("size_sol", 0.0) or 0.0)
            for position in self._strategy_open_positions(strategy_id)
        )

    def _current_exposure(self) -> float:
        return self.position_manager.total_open_exposure()

    def _apply_copy_score_penalty(
        self,
        closed_position: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Deduct score from copy-source wallets when a copy trade closed at a loss."""
        penalty = float(getattr(self.config, "wallet_copy_score_penalty_on_loss", 0.0))
        if penalty <= 0.0:
            return
        size_sol = float(closed_position.get("size_sol", 0.0) or 0.0)
        if size_sol <= 0.0:
            return
        realized = float(closed_position.get("realized_pnl_sol", 0.0) or 0.0)
        pnl_ratio = realized / size_sol
        loss_threshold = float(getattr(self.config, "wallet_copy_loss_pnl_threshold", -0.02))
        if pnl_ratio > loss_threshold:
            return
        sources = [str(w) for w in (metadata.get("copy_source_wallets") or []) if w]
        if not sources:
            return
        scores = self.stream.wallet_scores
        for wallet in sources:
            prev = float(scores.get(wallet, 0.0) or 0.0)
            scores[wallet] = prev - penalty
        self.event_log.log(
            "wallet_copy_score_penalty",
            {
                "token_mint": closed_position.get("token_mint"),
                "position_id": int(closed_position.get("id") or 0),
                "realized_pnl_sol": round(realized, 6),
                "size_sol": round(size_sol, 6),
                "pnl_ratio": round(pnl_ratio, 6),
                "penalty": round(penalty, 4),
                "penalized_wallets": sources,
            },
        )

    def _annotate_copy_mirror_sell(
        self,
        event,
        features: RuntimeFeatures,
        open_positions: list[dict[str, Any]],
    ) -> None:
        """Flag copy positions whose source wallet just sold so exit_engine can mirror it."""
        seller = str(getattr(event, "triggering_wallet", "") or "")
        if not seller:
            return
        matching_ids: list[int] = []
        for position in open_positions:
            try:
                metadata = json.loads(str(position.get("metadata_json") or "{}"))
            except Exception:
                continue
            if not metadata.get("wallet_copy"):
                continue
            sources = metadata.get("copy_source_wallets") or []
            if seller in sources:
                matching_ids.append(int(position.get("id") or 0))
        if not matching_ids:
            return
        raw = features.raw if isinstance(features.raw, dict) else {}
        raw["copy_mirror_sell_wallet"] = seller
        raw["copy_mirror_sell_position_ids"] = matching_ids
        raw["copy_mirror_sell_signature"] = str(getattr(event, "signature", "") or "")
        features.raw = raw
        self.event_log.log(
            "wallet_copy_mirror_sell_detected",
            {
                "token_mint": event.token_mint,
                "seller": seller,
                "position_ids": matching_ids,
                "signature": raw["copy_mirror_sell_signature"],
            },
        )

    def _feature_snapshot(self, features) -> dict[str, object]:
        """Return a compact runtime feature snapshot for rejection logs."""
        return {
            "token_age_sec": round(float(features.token_age_sec), 3)
            if features.token_age_sec is not None
            else None,
            "wallet_cluster_30s": features.wallet_cluster_30s,
            "wallet_cluster_120s": features.wallet_cluster_120s,
            "volume_sol_30s": round(float(features.volume_sol_30s), 6),
            "volume_sol_60s": round(float(features.volume_sol_60s), 6),
            "tx_count_30s": features.tx_count_30s,
            "tx_count_60s": features.tx_count_60s,
            "buy_volume_sol_30s": round(float(features.buy_volume_sol_30s), 6),
            "buy_volume_sol_60s": round(float(features.buy_volume_sol_60s), 6),
            "sell_volume_sol_30s": round(float(features.sell_volume_sol_30s), 6),
            "sell_volume_sol_60s": round(float(features.sell_volume_sol_60s), 6),
            "buy_sell_ratio_30s": (
                round(float(features.buy_sell_ratio_30s), 6)
                if features.buy_sell_ratio_30s is not None
                else None
            ),
            "buy_sell_ratio_60s": (
                round(float(features.buy_sell_ratio_60s), 6)
                if features.buy_sell_ratio_60s is not None
                else None
            ),
            "net_flow_sol_30s": round(float(features.net_flow_sol_30s), 6),
            "net_flow_sol_60s": round(float(features.net_flow_sol_60s), 6),
            "entry_price_sol": round(float(features.entry_price_sol), 12)
            if features.entry_price_sol is not None
            else None,
            "price_change_30s": round(float(features.price_change_30s), 6)
            if features.price_change_30s is not None
            else None,
            "price_change_60s": round(float(features.price_change_60s), 6)
            if features.price_change_60s is not None
            else None,
            "triggering_wallet": features.triggering_wallet,
            "triggering_wallet_score": round(float(features.triggering_wallet_score), 3),
            "aggregated_wallet_score": round(float(features.aggregated_wallet_score), 3),
            "tracked_wallet_present_60s": bool(features.tracked_wallet_present_60s),
            "tracked_wallet_count_60s": int(features.tracked_wallet_count_60s),
            "tracked_wallet_score_sum_60s": round(float(features.tracked_wallet_score_sum_60s), 3),
            "tracked_wallet_cluster_30s": int(features.raw.get("tracked_wallet_cluster_30s") or 0),
            "tracked_wallet_cluster_120s": int(
                features.raw.get("tracked_wallet_cluster_120s") or 0
            ),
            "tracked_wallet_cluster_300s": int(
                features.raw.get("tracked_wallet_cluster_300s") or 0
            ),
            "buy_streak_count_30s": int(features.raw.get("buy_streak_count_30s") or 0),
            "sell_tx_count_30s": int(getattr(features, "sell_tx_count_30s", 0) or 0),
            "round_trip_wallet_count_30s": int(
                getattr(features, "round_trip_wallet_count_30s", 0) or 0
            ),
            "round_trip_volume_sol_30s": round(
                float(getattr(features, "round_trip_volume_sol_30s", 0.0) or 0.0), 6
            ),
            "swaps_to_1_sol": features.raw.get("swaps_to_1_sol"),
            "swaps_to_5_sol": features.raw.get("swaps_to_5_sol"),
            "swaps_to_10_sol": features.raw.get("swaps_to_10_sol"),
            "swaps_to_30_sol": features.raw.get("swaps_to_30_sol"),
            "launcher_launches": features.raw.get("launcher_launches"),
            "launcher_graduations": features.raw.get("launcher_graduations"),
            "launcher_graduation_ratio": features.raw.get("launcher_graduation_ratio"),
        }

    def _is_relaxed_rule(self, rule: RuntimeRule) -> bool:
        """Return whether a runtime rule is a relaxed variant."""
        source = str(rule.source or "")
        return source.endswith(":relaxed") or rule.rule_id.endswith("_relaxed")

    def _regime_size_multiplier(self, regime: str) -> float:
        """Return configured size multiplier for one regime label."""
        regime_key = str(regime or "unknown")
        if regime_key == "negative_shock_recovery":
            return max(0.0, float(self.config.regime_size_multiplier_negative_shock_recovery))
        if regime_key == "high_cluster_recovery":
            return max(0.0, float(self.config.regime_size_multiplier_high_cluster_recovery))
        if regime_key == "momentum_burst":
            return max(0.0, float(self.config.regime_size_multiplier_momentum_burst))
        return max(0.0, float(self.config.regime_size_multiplier_unknown))

    def _entry_size_for_rule(self, rule: RuntimeRule, features: RuntimeFeatures) -> float:
        """Return proposed position size with rule + regime + wallet sizing."""
        base_size = float(self.config.max_position_sol)
        if self._is_relaxed_rule(rule):
            base_size *= float(self.config.relaxed_rule_size_multiplier)
        base_size *= self._regime_size_multiplier(rule.regime)

        tracked_wallet_boost = min(
            float(self.config.tracked_wallet_size_boost_cap),
            float(self.config.tracked_wallet_size_boost_per_wallet)
            * max(0.0, float(features.tracked_wallet_count_60s)),
        )
        if self.config.tracked_wallet_features_enabled and features.tracked_wallet_present_60s:
            base_size *= 1.0 + tracked_wallet_boost

        return max(0.0, min(base_size, float(self.config.max_position_sol)))

    def _ml_metadata(
        self,
        *,
        probability: float,
        threshold: float,
        mode: str,
        model_ready: bool,
        reason: str,
        feature_map: dict[str, float],
        lane: str | None,
        candidate_score: float | None,
    ) -> dict[str, Any]:
        """Build persisted ML metadata for a taken entry."""
        return {
            "ml_mode": str(mode),
            "ml_model_ready": bool(model_ready),
            "ml_probability": float(probability),
            "ml_threshold": float(threshold),
            "ml_reason": str(reason),
            "ml_feature_map": feature_map,
            "entry_lane": lane,
            "candidate_score": float(candidate_score) if candidate_score is not None else None,
        }

    def _resolve_lp_mint_for_pool(
        self,
        token_mint: str,
        local_quote_engine: Any,
        broadcaster: Any,
        *,
        source_program: str | None = None,
    ) -> str | None:
        """Return the LP mint for ``token_mint`` if we can derive it.

        Dispatches by ``source_program`` — ``PUMP_AMM`` decodes the Pump-AMM
        pool account (bytes [107:139]); ``RAYDIUM`` decodes the Raydium V4
        ``LiquidityStateV4`` account (bytes [464:496]). Pre-migration
        bonding-curve pools and unsupported sources return None, which the
        LP-burn guard treats as fail-closed when ``live_entry_require_lp_burned``
        is enabled.

        The pool pubkey is read from the local-quote-engine pool-state cache
        (no RPC round-trip) when available, with a fallback to the on-chain
        ``getAccountInfo`` path for sources that don't populate the local
        cache yet.
        """
        if broadcaster is None:
            return None
        source = (source_program or "").upper()
        pool_pubkey: str | None = None
        if local_quote_engine is not None:
            getter = getattr(local_quote_engine, "get_native_pool_state", None)
            if getter is not None:
                try:
                    state = getter(token_mint)
                except Exception:  # noqa: BLE001
                    state = None
                if state is not None:
                    pool_candidate = getattr(state, "pool", None)
                    if pool_candidate:
                        pool_pubkey = str(pool_candidate)
        if not pool_pubkey:
            return None
        dispatch_source = source if source in {"PUMP_AMM", "RAYDIUM"} else "PUMP_AMM"
        cached = getattr(broadcaster, "get_pool_lp_mint_cached", None)
        if callable(cached):
            try:
                return cached(pool_pubkey, dispatch_source)
            except Exception:  # noqa: BLE001
                return None
        decoder_name = (
            "get_raydium_v4_lp_mint" if dispatch_source == "RAYDIUM" else "get_pump_amm_lp_mint"
        )
        decoder = getattr(broadcaster, decoder_name, None)
        if decoder is None:
            return None
        try:
            return decoder(pool_pubkey)
        except Exception:  # noqa: BLE001
            return None

    def _paper_entry_guard(
        self,
        *,
        token_mint: str,
        size_sol: float,
        strategy_id: str,
        rejection_event: str,
        rejection_payload: dict[str, Any] | None = None,
        pre_buy_estimate: PaperTradeEstimate | None = None,
        local_quote_engine: Any = None,
        features: RuntimeFeatures | None = None,
        source_program: str | None = None,
    ) -> tuple[bool, PaperTradeEstimate | None, dict[str, Any]]:
        """Run executable-liquidity preflight and return reusable metadata.

        Paper mode runs under ``paper_entry_*`` settings. Live mode runs the
        same round-trip check against the Jupiter/local-quote stack and uses
        ``live_entry_*`` settings; gated off by default until tuned so live
        behavior doesn't change silently.
        """
        live_mode = bool(self.trade_executor.live)
        if live_mode and not bool(
            getattr(self.config, "live_entry_roundtrip_guard_enabled", False)
        ):
            return True, None, {}

        if live_mode:
            pool_age_max = float(getattr(self.config, "live_entry_pool_max_age_sec", 0.0) or 0.0)
            if pool_age_max > 0.0 and local_quote_engine is not None:
                try:
                    pool_state = local_quote_engine.get_native_pool_state(token_mint)
                except Exception:  # noqa: BLE001
                    pool_state = None
                if pool_state is not None:
                    pool_age_sec = max(
                        0.0,
                        time.monotonic() - float(getattr(pool_state, "ts", 0.0) or 0.0),
                    )
                    if pool_age_sec > pool_age_max:
                        payload = {
                            "token_mint": token_mint,
                            "reason": "live_entry_pool_abandoned",
                            "strategy_id": strategy_id,
                            "guard_reason": "pool_state_stale",
                            "size_sol": round(float(size_sol), 6),
                            "pool_age_sec": round(float(pool_age_sec), 3),
                            "pool_age_max_sec": round(float(pool_age_max), 3),
                            "pool_pubkey": str(getattr(pool_state, "pool", "")),
                        }
                        if rejection_payload:
                            payload.update(rejection_payload)
                        self.event_log.log(rejection_event, payload)
                        return False, None, {}

            min_pool_sol = float(
                getattr(self.config, "live_entry_min_pool_sol_reserve", 0.0) or 0.0
            )
            if min_pool_sol > 0.0 and local_quote_engine is not None:
                try:
                    reserves = local_quote_engine.get_reserves(token_mint)
                except Exception:  # noqa: BLE001
                    reserves = None
                if reserves is not None:
                    sol_reserve = float(getattr(reserves, "sol_reserve", 0) or 0) / 1_000_000_000.0
                    if sol_reserve < min_pool_sol:
                        payload = {
                            "token_mint": token_mint,
                            "reason": "live_entry_pool_low_liquidity",
                            "strategy_id": strategy_id,
                            "guard_reason": "pool_sol_reserve_below_floor",
                            "size_sol": round(float(size_sol), 6),
                            "pool_sol_reserve": round(sol_reserve, 6),
                            "min_pool_sol_reserve": round(min_pool_sol, 6),
                        }
                        if rejection_payload:
                            payload.update(rejection_payload)
                        self.event_log.log(rejection_event, payload)
                        return False, None, {}

            if not bool(getattr(self.config, "live_allow_token_2022_buys", False)):
                live_executor = getattr(self.trade_executor, "_live_executor", None)
                resolver = getattr(
                    live_executor, "resolve_mint_program_and_extensions_strict", None
                )
                if live_executor is not None and resolver is not None:
                    try:
                        resolved = resolver(token_mint)
                    except Exception:  # noqa: BLE001
                        resolved = None
                    if resolved is None:
                        payload = {
                            "token_mint": token_mint,
                            "reason": "guard_rpc_unavailable_token_program",
                            "strategy_id": strategy_id,
                            "guard_reason": "token_program_lookup_failed",
                            "size_sol": round(float(size_sol), 6),
                        }
                        if rejection_payload:
                            payload.update(rejection_payload)
                        self.event_log.log(rejection_event, payload)
                        return False, None, {}
                    owning_program, mint_extensions = resolved
                    _SPL_TOKEN = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
                    _TOKEN_2022 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
                    # Extensions that do not affect buy/sell mechanics — safe
                    # to accept. Anything outside this set (transferFeeConfig,
                    # permanentDelegate, transferHook, defaultAccountState,
                    # nonTransferable, confidentialTransferMint, interestBearingConfig, …)
                    # can brick sells or skim value — reject.
                    _SAFE_T22_EXTENSIONS = {
                        "metadataPointer",
                        "tokenMetadata",
                        "metadata",
                        "mintCloseAuthority",
                    }
                    if owning_program == _TOKEN_2022:
                        risky = [ext for ext in mint_extensions if ext not in _SAFE_T22_EXTENSIONS]
                        if risky:
                            payload = {
                                "token_mint": token_mint,
                                "reason": "live_entry_token_2022_risky_extension",
                                "strategy_id": strategy_id,
                                "guard_reason": f"token_2022_extension_{risky[0]}",
                                "size_sol": round(float(size_sol), 6),
                                "mint_token_program": owning_program,
                                "mint_extensions": list(mint_extensions),
                                "risky_extensions": risky,
                            }
                            if rejection_payload:
                                payload.update(rejection_payload)
                            self.event_log.log(rejection_event, payload)
                            return False, None, {}
                    elif owning_program and owning_program != _SPL_TOKEN:
                        payload = {
                            "token_mint": token_mint,
                            "reason": "live_entry_token_program_blocked",
                            "strategy_id": strategy_id,
                            "guard_reason": "mint_not_spl_token",
                            "size_sol": round(float(size_sol), 6),
                            "mint_token_program": str(owning_program),
                        }
                        if rejection_payload:
                            payload.update(rejection_payload)
                        self.event_log.log(rejection_event, payload)
                        return False, None, {}

            # Freeze-authority / mint-authority check. A mint with a live freeze
            # authority can freeze any holder's tokens, blocking sell attempts
            # entirely — the canonical "rug by freeze" pattern. A live mint
            # authority lets the deployer dilute holders without warning. Both
            # can be revoked; honest projects revoke them before launch.
            require_freeze_null = bool(
                getattr(self.config, "live_entry_require_freeze_authority_null", False)
            )
            require_mint_null = bool(
                getattr(self.config, "live_entry_require_mint_authority_null", False)
            )
            if require_freeze_null or require_mint_null:
                live_executor = getattr(self.trade_executor, "_live_executor", None)
                broadcaster = getattr(live_executor, "broadcaster", None) if live_executor else None
                if broadcaster is not None and hasattr(broadcaster, "get_mint_authorities"):
                    try:
                        authorities = broadcaster.get_mint_authorities(token_mint)
                    except Exception:  # noqa: BLE001
                        authorities = None
                    # ``None`` ⇒ unknown. Let the trade through — RPC hiccups
                    # shouldn't block all entries. Treat a parsed result as
                    # authoritative.
                    if authorities is not None:
                        mint_auth, freeze_auth = authorities
                        if require_freeze_null and freeze_auth:
                            payload = {
                                "token_mint": token_mint,
                                "reason": "live_entry_freeze_authority_live",
                                "strategy_id": strategy_id,
                                "guard_reason": "freeze_authority_not_revoked",
                                "size_sol": round(float(size_sol), 6),
                                "freeze_authority": str(freeze_auth),
                            }
                            if rejection_payload:
                                payload.update(rejection_payload)
                            self.event_log.log(rejection_event, payload)
                            return False, None, {}
                        if require_mint_null and mint_auth:
                            payload = {
                                "token_mint": token_mint,
                                "reason": "live_entry_mint_authority_live",
                                "strategy_id": strategy_id,
                                "guard_reason": "mint_authority_not_revoked",
                                "size_sol": round(float(size_sol), 6),
                                "mint_authority": str(mint_auth),
                            }
                            if rejection_payload:
                                payload.update(rejection_payload)
                            self.event_log.log(rejection_event, payload)
                            return False, None, {}

            min_unique_wallets = int(
                getattr(self.config, "live_entry_min_unique_wallets_30s", 0) or 0
            )
            if strategy_id != "wallet" and min_unique_wallets > 0 and features is not None:
                unique_wallets_30s = int(getattr(features, "wallet_cluster_30s", 0) or 0)
                if unique_wallets_30s < min_unique_wallets:
                    payload = {
                        "token_mint": token_mint,
                        "reason": "live_entry_thin_crowd",
                        "strategy_id": strategy_id,
                        "guard_reason": "unique_wallets_30s_below_floor",
                        "size_sol": round(float(size_sol), 6),
                        "unique_wallets_30s": unique_wallets_30s,
                        "min_unique_wallets_30s": min_unique_wallets,
                    }
                    if rejection_payload:
                        payload.update(rejection_payload)
                    self.event_log.log(rejection_event, payload)
                    return False, None, {}

            # Pure-buy-flow filter: on young pools, zero sell volume + nontrivial
            # buy volume is the canonical liquidity-trap signature — the dev/bundle
            # is absorbing every buy and no organic seller has appeared yet.
            # Wallet lane skips: tracked-wallet cluster signal dominates this heuristic.
            if (
                strategy_id != "wallet"
                and bool(getattr(self.config, "entry_pure_buy_filter_enabled", False))
                and features is not None
            ):
                max_age_sec = float(
                    getattr(self.config, "entry_pure_buy_filter_max_age_sec", 0.0) or 0.0
                )
                min_buy_vol = float(
                    getattr(self.config, "entry_pure_buy_filter_min_buy_volume_sol", 0.0) or 0.0
                )
                token_age_sec = features.token_age_sec
                buy_vol_30s = float(getattr(features, "buy_volume_sol_30s", 0.0) or 0.0)
                sell_vol_30s = float(getattr(features, "sell_volume_sol_30s", 0.0) or 0.0)
                if (
                    max_age_sec > 0.0
                    and token_age_sec is not None
                    and float(token_age_sec) < max_age_sec
                    and sell_vol_30s <= 0.0
                    and buy_vol_30s >= min_buy_vol
                ):
                    payload = {
                        "token_mint": token_mint,
                        "reason": "pure_buy_flow_young_token",
                        "strategy_id": strategy_id,
                        "guard_reason": "sell_volume_zero_on_young_pool",
                        "size_sol": round(float(size_sol), 6),
                        "token_age_sec": round(float(token_age_sec), 3),
                        "buy_volume_sol_30s": round(buy_vol_30s, 6),
                        "sell_volume_sol_30s": round(sell_vol_30s, 6),
                        "max_age_sec": round(max_age_sec, 3),
                        "min_buy_volume_sol": round(min_buy_vol, 6),
                    }
                    if rejection_payload:
                        payload.update(rejection_payload)
                    self.event_log.log(rejection_event, payload)
                    return False, None, {}

            # Top-holder concentration check: reject when a non-pool wallet
            # holds a disproportionate share of supply. Proxies for dev/sniper
            # bundling and LP-token concentration. The broadcaster caches the
            # result for 60s per mint so back-to-back evals don't hit Helius.
            max_top_holder_pct = float(
                getattr(self.config, "live_entry_max_top_holder_pct", 0.0) or 0.0
            )
            live_executor = getattr(self.trade_executor, "_live_executor", None)
            broadcaster = getattr(live_executor, "broadcaster", None) if live_executor else None
            if (
                strategy_id != "wallet"
                and max_top_holder_pct > 0.0
                and broadcaster is not None
                and hasattr(broadcaster, "get_top_non_pool_holder_pct")
            ):
                extra_exclude = tuple(
                    getattr(self.config, "live_entry_holder_exclude_pubkeys", ()) or ()
                )
                try:
                    top_holder_pct = broadcaster.get_top_non_pool_holder_pct(
                        token_mint, exclude_owners=extra_exclude
                    )
                except Exception:  # noqa: BLE001
                    top_holder_pct = None
                if top_holder_pct is not None and top_holder_pct > max_top_holder_pct:
                    payload = {
                        "token_mint": token_mint,
                        "reason": "live_entry_holder_concentration",
                        "strategy_id": strategy_id,
                        "guard_reason": "top_non_pool_holder_above_floor",
                        "size_sol": round(float(size_sol), 6),
                        "top_holder_pct": round(float(top_holder_pct), 6),
                        "max_top_holder_pct": round(max_top_holder_pct, 6),
                    }
                    if rejection_payload:
                        payload.update(rejection_payload)
                    self.event_log.log(rejection_event, payload)
                    return False, None, {}

            # Top-5 sum: catches distributed-bundle rugs where a deployer
            # splits allocation across several wallets to slip under the
            # per-wallet top-1 threshold. Uses the same getTokenLargestAccounts
            # RPC as top-1 (already fetched above), so this is effectively free
            # after the first call.
            max_top5_holder_pct = float(
                getattr(self.config, "live_entry_max_top5_holder_pct", 0.0) or 0.0
            )
            if (
                strategy_id != "wallet"
                and max_top5_holder_pct > 0.0
                and broadcaster is not None
                and hasattr(broadcaster, "get_top_n_non_pool_holder_sum_pct")
            ):
                extra_exclude = tuple(
                    getattr(self.config, "live_entry_holder_exclude_pubkeys", ()) or ()
                )
                try:
                    top5_sum_pct = broadcaster.get_top_n_non_pool_holder_sum_pct(
                        token_mint, n=5, exclude_owners=extra_exclude
                    )
                except Exception:  # noqa: BLE001
                    top5_sum_pct = None
                if top5_sum_pct is not None and top5_sum_pct > max_top5_holder_pct:
                    payload = {
                        "token_mint": token_mint,
                        "reason": "live_entry_holder_concentration_top5",
                        "strategy_id": strategy_id,
                        "guard_reason": "top5_non_pool_holders_above_floor",
                        "size_sol": round(float(size_sol), 6),
                        "top5_sum_pct": round(float(top5_sum_pct), 6),
                        "max_top5_holder_pct": round(max_top5_holder_pct, 6),
                    }
                    if rejection_payload:
                        payload.update(rejection_payload)
                    self.event_log.log(rejection_event, payload)
                    return False, None, {}

            # Honeypot simulation: atomic buy→sell bundle simulation via the
            # RPC's simulateBundle method. Opt-in and fail-open — if the RPC
            # doesn't support bundle simulation or the native pool state isn't
            # cached, we allow the trade through. Only rejects when the sell
            # leg concretely reverts in simulation, which is the canonical
            # "buys work, sells don't" honeypot signal.
            if bool(getattr(self.config, "live_entry_honeypot_sim_enabled", False)):
                live_executor = getattr(self.trade_executor, "_live_executor", None)
                if live_executor is not None and hasattr(
                    live_executor, "simulate_honeypot_roundtrip"
                ):
                    size_lamports = int(float(size_sol) * 1_000_000_000)
                    honeypot_trace: dict[str, Any] = {}
                    try:
                        honeypot_err = live_executor.simulate_honeypot_roundtrip(
                            token_mint=token_mint,
                            size_lamports=size_lamports,
                            trace=honeypot_trace,
                        )
                    except Exception:  # noqa: BLE001
                        honeypot_err = None
                    if honeypot_err is not None:
                        is_rpc_unavailable = str(honeypot_err).startswith("rpc_unavailable")
                        payload = {
                            "token_mint": token_mint,
                            "reason": (
                                "guard_rpc_unavailable_honeypot"
                                if is_rpc_unavailable
                                else "live_entry_honeypot_detected"
                            ),
                            "strategy_id": strategy_id,
                            "guard_reason": (
                                "honeypot_sim_rpc_failed"
                                if is_rpc_unavailable
                                else "sell_reverts_in_bundle_sim"
                            ),
                            "size_sol": round(float(size_sol), 6),
                            "honeypot_err": str(honeypot_err),
                            "honeypot_trace": honeypot_trace,
                        }
                        if rejection_payload:
                            payload.update(rejection_payload)
                        self.event_log.log(rejection_event, payload)
                        return False, None, {}

            # LP-burned guard. Fail-closed only for sources where LP-pull is a
            # real rug vector (Raydium V4 by default, where deployers can
            # unlock/unstake LP and drain the pool). Pump-AMM LPs are held by
            # the Pump-AMM program itself — not a pull risk — so skip there.
            # Configurable via LIVE_ENTRY_LP_GUARD_SOURCES. Runs after honeypot
            # sim, before round-trip.
            if bool(getattr(self.config, "live_entry_require_lp_burned", False)):
                source = (source_program or "").upper()
                guarded_sources = {
                    s.upper()
                    for s in (getattr(self.config, "live_entry_lp_guard_sources", ()) or ())
                    if s
                }
                if source in guarded_sources:
                    threshold = float(
                        getattr(self.config, "live_entry_lp_burn_threshold", 0.90) or 0.90
                    )
                    cache_ttl = float(
                        getattr(self.config, "live_entry_lp_burn_cache_ttl_sec", 300.0) or 300.0
                    )
                    lp_mint = self._resolve_lp_mint_for_pool(
                        token_mint,
                        local_quote_engine,
                        broadcaster,
                        source_program=source,
                    )
                    burn_fraction: float | None = None
                    guard_reason: str | None = None
                    if lp_mint is None:
                        guard_reason = "lp_mint_not_found"
                    elif broadcaster is None:
                        guard_reason = "broadcaster_unavailable"
                    else:
                        cached = getattr(broadcaster, "get_lp_burn_fraction_cached", None)
                        try:
                            if callable(cached):
                                burn_fraction = cached(lp_mint, cache_ttl_sec=cache_ttl)
                            elif hasattr(broadcaster, "get_lp_burn_fraction"):
                                burn_fraction = broadcaster.get_lp_burn_fraction(lp_mint)
                            else:
                                guard_reason = "broadcaster_unavailable"
                        except Exception:  # noqa: BLE001
                            burn_fraction = None
                        if guard_reason is None:
                            if burn_fraction is None:
                                guard_reason = "lp_burn_rpc_failed"
                            elif burn_fraction < threshold:
                                guard_reason = "lp_not_burned"
                    if guard_reason is not None:
                        payload = {
                            "token_mint": token_mint,
                            "reason": "live_entry_lp_guard",
                            "strategy_id": strategy_id,
                            "guard_reason": guard_reason,
                            "source_program": source,
                            "size_sol": round(float(size_sol), 6),
                            "lp_mint": lp_mint,
                            "lp_burn_fraction": (
                                round(float(burn_fraction), 6)
                                if burn_fraction is not None
                                else None
                            ),
                            "lp_burn_threshold": round(float(threshold), 6),
                        }
                        if rejection_payload:
                            payload.update(rejection_payload)
                        self.event_log.log(rejection_event, payload)
                        return False, None, {}

            # Dev-wallet serial-creator guard. Pump.fun bonding curve exposes
            # the dev wallet in the BondingCurve PDA. Serial creators spinning
            # up many tokens in a short window are the dominant rug vector on
            # pump.fun pre-graduation. Fail-open on RPC failure / missing
            # creator so Helius hiccups don't block all entries; fail-closed
            # only when we positively identify a creator with too many mints.
            if bool(getattr(self.config, "entry_dev_wallet_check_enabled", False)):
                source = (source_program or "").upper()
                guarded_sources = {
                    s.upper()
                    for s in (getattr(self.config, "entry_dev_wallet_check_sources", ()) or ())
                    if s
                }
                if source in guarded_sources and broadcaster is not None:
                    max_tokens = int(
                        getattr(self.config, "entry_dev_wallet_max_tokens_24h", 3) or 3
                    )
                    creator: str | None = None
                    token_count: int | None = None
                    try:
                        creator = broadcaster.get_pump_fun_creator(token_mint)
                    except Exception:  # noqa: BLE001
                        creator = None
                    if creator is not None:
                        try:
                            # Fetch enough to distinguish "≤threshold" from "over".
                            token_count = broadcaster.get_creator_recent_token_count(
                                creator, limit=max(max_tokens + 1, 5)
                            )
                        except Exception:  # noqa: BLE001
                            token_count = None
                    if creator is not None and token_count is not None and token_count > max_tokens:
                        payload = {
                            "token_mint": token_mint,
                            "reason": "live_entry_dev_serial_creator",
                            "strategy_id": strategy_id,
                            "guard_reason": "creator_too_many_tokens",
                            "source_program": source,
                            "size_sol": round(float(size_sol), 6),
                            "creator_wallet": creator,
                            "creator_token_count": int(token_count),
                            "creator_token_threshold": int(max_tokens),
                        }
                        if rejection_payload:
                            payload.update(rejection_payload)
                        self.event_log.log(rejection_event, payload)
                        return False, None, {}

            min_roundtrip_ratio = effective_min_roundtrip_ratio(
                self.config,
                strategy_id=strategy_id,
                source_program=source_program,
                live_mode=True,
            )
            max_price_impact_pct = effective_max_price_impact_pct(
                self.config,
                strategy_id=strategy_id,
                source_program=source_program,
                live_mode=True,
            )
        else:
            min_roundtrip_ratio = effective_min_roundtrip_ratio(
                self.config,
                strategy_id=strategy_id,
                source_program=source_program,
                live_mode=False,
            )
            max_price_impact_pct = effective_max_price_impact_pct(
                self.config,
                strategy_id=strategy_id,
                source_program=source_program,
                live_mode=False,
            )

        guard = self.trade_executor.evaluate_paper_entry_guard(
            token_mint=token_mint,
            size_sol=size_sol,
            min_roundtrip_ratio=min_roundtrip_ratio,
            max_price_impact_pct=max_price_impact_pct,
            pre_buy_estimate=pre_buy_estimate,
            local_quote_engine=local_quote_engine,
        )
        if not guard.allowed:
            payload = {
                "token_mint": token_mint,
                "reason": "paper_entry_liquidity_guard",
                "strategy_id": strategy_id,
                "guard_reason": guard.reason,
                "size_sol": round(float(size_sol), 6),
                "roundtrip_ratio": round(float(guard.roundtrip_ratio), 6),
                "roundtrip_pnl_sol": round(float(guard.roundtrip_pnl_sol), 6),
                "entry_cost_sol": round(float(guard.entry_cost_sol), 6),
                "immediate_exit_net_sol": round(float(guard.immediate_exit_net_sol), 6),
                "guard_min_roundtrip_ratio": round(float(min_roundtrip_ratio), 6),
                "guard_max_price_impact_pct": round(float(max_price_impact_pct), 6),
            }
            if rejection_payload:
                payload.update(rejection_payload)
            self.event_log.log(rejection_event, payload)
            return False, guard.buy_estimate, {}

        metadata = {
            "paper_entry_guard_passed": True,
            "paper_entry_guard_roundtrip_ratio": float(guard.roundtrip_ratio),
            "paper_entry_guard_roundtrip_pnl_sol": float(guard.roundtrip_pnl_sol),
            "paper_entry_guard_entry_cost_sol": float(guard.entry_cost_sol),
            "paper_entry_guard_immediate_exit_net_sol": float(guard.immediate_exit_net_sol),
            "paper_entry_guard_min_roundtrip_ratio": float(min_roundtrip_ratio),
            "paper_entry_guard_max_price_impact_pct": float(max_price_impact_pct),
        }
        return True, guard.buy_estimate, metadata

    def _recovery_confirmation_failures(self, features, rule: RuntimeRule) -> list[str]:
        """Return recovery confirmation failures for recovery-style entries."""
        if not self.config.enable_recovery_confirmation:
            return []
        # Pair-first mode already applies explicit rule thresholds on live features.
        # Additional recovery delta gating here proved over-restrictive and blocked
        # valid high-cluster/pair-adapted matches.
        if self.config.discovery_mode == "pair_first":
            return []
        if "recovery" not in str(rule.regime):
            return []
        if features.price_change_30s is None or features.price_change_60s is None:
            return ["price_change_confirmation_missing"]
        delta = float(features.price_change_30s) - float(features.price_change_60s)
        if delta <= self.config.recovery_confirmation_min_delta:
            return [
                "recovery_confirmation_failed",
                (
                    f"price_change_delta<={self.config.recovery_confirmation_min_delta:.4f} "
                    f"({delta:.4f})"
                ),
            ]
        return []

    def _entry_quality_failures(self, features) -> list[str]:
        """Return quality-gate failures before rule matching."""
        failures: list[str] = []
        if features.token_age_sec is None:
            failures.append("token_age_sec_missing")
        elif features.token_age_sec < float(self.config.entry_min_token_age_sec):
            failures.append(
                f"token_age_sec<{float(self.config.entry_min_token_age_sec):.0f} ({features.token_age_sec:.1f})"
            )
        if features.wallet_cluster_30s < int(self.config.entry_min_cluster_30s):
            failures.append(
                f"wallet_cluster_30s<{int(self.config.entry_min_cluster_30s)} ({features.wallet_cluster_30s})"
            )
        if features.tx_count_30s < int(self.config.entry_min_tx_count_30s):
            failures.append(
                f"tx_count_30s<{int(self.config.entry_min_tx_count_30s)} ({features.tx_count_30s})"
            )
        if features.volume_sol_30s < float(self.config.entry_min_volume_sol_30s):
            failures.append(
                f"volume_sol_30s<{float(self.config.entry_min_volume_sol_30s):.4f} ({features.volume_sol_30s:.4f})"
            )
        if features.tx_count_30s > 0:
            avg_trade_sol_30s = float(features.volume_sol_30s) / float(features.tx_count_30s)
            min_avg_trade = float(getattr(self.config, "entry_min_avg_trade_sol_30s", 0.0))
            if min_avg_trade > 0 and avg_trade_sol_30s < min_avg_trade:
                failures.append(f"avg_trade_sol_30s<{min_avg_trade:.4f} ({avg_trade_sol_30s:.4f})")
        _pair_first_age_max = float(self.config.pair_first_token_age_max_sec)
        if (
            self.config.discovery_mode == "pair_first"
            and features.token_age_sec is not None
            and _pair_first_age_max > 0.0
            and features.token_age_sec > _pair_first_age_max
        ):
            failures.append(
                f"token_age_sec>{_pair_first_age_max:.0f} ({features.token_age_sec:.1f})"
            )
        return failures

    def _entry_lane_failures(self, features: RuntimeFeatures) -> tuple[str | None, list[str]]:
        """Return lane label and failures for pair-first candidate quality."""
        return determine_entry_lane(features, self.config)

    def _cleanup_ranked_candidates(self, now: datetime) -> None:
        """Drop stale queued candidates to keep ranking queue bounded."""
        max_age_sec = max(30.0, float(self.config.candidate_ranking_window_sec) * 8.0)
        cutoff = now - timedelta(seconds=max_age_sec)
        stale_tokens = [
            token for token, item in self._ranked_candidates.items() if item["queued_at"] < cutoff
        ]
        for token in stale_tokens:
            self._ranked_candidates.pop(token, None)

    def _enqueue_ranked_candidate(
        self,
        event,
        features: RuntimeFeatures,
        match: MatchResult,
        lane: str,
        queued_at: datetime,
        source: str,
        arrival_at: datetime | None = None,
    ) -> bool:
        """Queue one fully validated candidate for short ranking-window selection."""
        if match.selected_rule is None:
            return False
        candidate_score, score_breakdown = score_candidate(
            features=features,
            rule=match.selected_rule,
            detected_regime=match.detected_regime,
            lane=lane,
            config=self.config,
        )
        if candidate_score < float(self.config.candidate_min_score):
            self.event_log.log(
                "entry_rejected",
                {
                    "token_mint": event.token_mint,
                    "reason": "candidate_score_below_threshold",
                    "candidate_score": round(candidate_score, 6),
                    "candidate_min_score": float(self.config.candidate_min_score),
                    "entry_lane": lane,
                    "rule_id": match.selected_rule.rule_id,
                    "tracked_wallet_present_60s": bool(features.tracked_wallet_present_60s),
                    "tracked_wallet_count_60s": int(features.tracked_wallet_count_60s),
                    "tracked_wallet_score_sum_60s": float(features.tracked_wallet_score_sum_60s),
                    "score_breakdown": score_breakdown,
                    "feature_snapshot": self._feature_snapshot(features),
                },
            )
            self._set_candidate_cooldown(event.token_mint, queued_at)
            return False

        item: dict[str, Any] = {
            "event": event,
            "features": features,
            "match": match,
            "lane": lane,
            "score": candidate_score,
            "score_breakdown": score_breakdown,
            "queued_at": queued_at,
            "source": source,
            "arrival_at": self._normalize_dt(arrival_at) or queued_at,
        }
        latency_trace = self._copy_trace(features.raw.get("__latency_trace"))
        latency_trace["candidate_lane"] = lane
        latency_trace["candidate_rank_source"] = str(source)
        latency_trace["candidate_rank_queued_at"] = queued_at.isoformat()
        latency_trace["candidate_score"] = round(candidate_score, 6)
        arrival_ms = self._ms_between(item["arrival_at"], queued_at)
        if arrival_ms is not None:
            latency_trace["arrival_to_rank_queue_ms"] = arrival_ms

        # Pre-fetch the BUY quote in a background thread while the ranking window ticks.
        # By the time _flush_ranked_candidates() fires a short moment later, the quote is
        # usually ready and we avoid one sequential Jupiter call on the critical path.
        if not self.trade_executor.live and self.config.paper_entry_roundtrip_guard_enabled:
            proposed_size = self._entry_size_for_rule(match.selected_rule, features)
            if proposed_size > 0:
                latency_trace["buy_quote_prefetch_submitted_at"] = datetime.now(
                    tz=timezone.utc
                ).isoformat()
                item["buy_quote_future"] = self.trade_executor.submit_buy_quote(
                    token_mint=event.token_mint,
                    size_sol=proposed_size,
                )

        # For live mode: pre-build the Jupiter swap TX during the ranking window.
        # At fire time we only need to sign (~1ms) + broadcast (~50-150ms).
        # Main lane forces Jupiter routing: the native Pump-AMM buy builder's
        # math overflows (preflight Custom 6023) on mature Pump-AMM pools.
        if self.trade_executor.live:
            proposed_size = self._entry_size_for_rule(match.selected_rule, features)
            if proposed_size > 0:
                latency_trace["swap_tx_prefetch_submitted_at"] = datetime.now(
                    tz=timezone.utc
                ).isoformat()
                item["swap_tx_future"] = self.trade_executor.submit_swap_tx_future(
                    token_mint=event.token_mint,
                    size_sol=proposed_size,
                    prefer_jupiter=True,
                    strategy="main",
                    source_program=event.source_program,
                )
        item["latency_trace"] = latency_trace

        existing = self._ranked_candidates.get(event.token_mint)
        if existing is None or candidate_score >= float(existing.get("score", -1.0)):
            self._ranked_candidates[event.token_mint] = item
            self.event_log.log(
                "candidate_ranked",
                {
                    "token_mint": event.token_mint,
                    "entry_lane": lane,
                    "candidate_score": round(candidate_score, 6),
                    "source": source,
                    "detected_regime": match.detected_regime,
                    "rule_id": match.selected_rule.rule_id,
                    "tracked_wallet_present_60s": bool(features.tracked_wallet_present_60s),
                    "tracked_wallet_count_60s": int(features.tracked_wallet_count_60s),
                    "tracked_wallet_score_sum_60s": float(features.tracked_wallet_score_sum_60s),
                    "wallet_bonus_total": round(
                        float(score_breakdown.get("tracked_wallet_bonus_total", 0.0)), 6
                    ),
                    "score_breakdown": score_breakdown,
                    "latency_trace": latency_trace,
                },
            )
        return True

    async def _flush_ranked_candidates(self, now: datetime, force: bool = False) -> int:
        """Select and execute highest-scoring queued candidates."""
        if not self._ranked_candidates:
            return 0
        # Discard queued candidates when market regime is unfavorable.
        # Gate is active when MARKET_REGIME_GATE_ACTIVE=true (independent of ML mode).
        # When inactive, the signal is logged but does not block entries (shadow collection).
        if not self._market_regime.is_favorable():
            if self.config.market_regime_gate_active:
                self._ranked_candidates.clear()
                return 0
            else:
                self.logger.debug(
                    "market_regime: unfavorable (%s) but gate inactive — not blocking entries",
                    self._market_regime.get_state().pause_reason,
                )
        self._cleanup_ranked_candidates(now)
        if not self._ranked_candidates:
            return 0

        if not force:
            oldest = min(item["queued_at"] for item in self._ranked_candidates.values())
            has_waited = (now - oldest).total_seconds() >= float(
                self.config.candidate_ranking_window_sec
            )
            enough_depth = len(self._ranked_candidates) >= int(self.config.candidate_queue_min_size)
            if not (has_waited or enough_depth):
                return 0

        scored_items = sorted(
            self._ranked_candidates.items(),
            key=lambda kv: (
                float(kv[1].get("score", 0.0)),
                kv[1].get("queued_at"),
            ),
            reverse=True,
        )

        # Snapshot positions once for the entire flush loop.
        # Previously has_open_position + _current_exposure + list_open_positions
        # each fired separate DB queries per candidate (3 queries × N candidates).
        _open_positions = self.position_manager.list_open_positions()
        _open_mints = {p["token_mint"] for p in _open_positions}
        _base_exposure = sum(float(p["size_sol"]) for p in _open_positions)
        _base_open_count = len(_open_positions)

        processed = 0
        for token_mint, item in scored_items:
            self._ranked_candidates.pop(token_mint, None)
            event = item["event"]
            features = item["features"]
            match = item["match"]
            lane = str(item.get("lane") or "unknown")
            candidate_score = float(item.get("score", 0.0))
            score_breakdown = dict(item.get("score_breakdown") or {})
            arrival_at = self._normalize_dt(item.get("arrival_at")) or item["queued_at"]
            dispatch_time = datetime.now(tz=timezone.utc)
            latency_trace = self._copy_trace(item.get("latency_trace"))
            latency_trace["candidate_selected_at"] = dispatch_time.isoformat()
            rank_wait_ms = self._ms_between(item["queued_at"], dispatch_time)
            if rank_wait_ms is not None:
                latency_trace["rank_queue_wait_ms"] = rank_wait_ms
            arrival_to_select_ms = self._ms_between(arrival_at, dispatch_time)
            if arrival_to_select_ms is not None:
                latency_trace["arrival_to_candidate_selection_ms"] = arrival_to_select_ms

            if match.selected_rule is None:
                continue
            if token_mint in _open_mints:
                continue

            current_exposure = _base_exposure
            open_count = _base_open_count
            proposed_size = self._entry_size_for_rule(match.selected_rule, features)
            allowed, reason = self.risk_manager.can_open(
                token_mint=token_mint,
                proposed_size_sol=proposed_size,
                open_position_count=open_count,
                total_exposure_sol=current_exposure,
            )
            if not allowed:
                self.event_log.log(
                    "entry_rejected",
                    {
                        "token_mint": token_mint,
                        "reason": reason,
                        "rule_id": match.selected_rule.rule_id,
                        "entry_lane": lane,
                        "candidate_score": round(candidate_score, 6),
                        "wallet_bonus_total": round(
                            float(score_breakdown.get("tracked_wallet_bonus_total", 0.0)),
                            6,
                        ),
                    },
                )
                self._set_candidate_cooldown(token_mint, now)
                continue

            ml_decision = self.ml_filter.evaluate_candidate(
                features=features,
                rule=match.selected_rule,
                detected_regime=match.detected_regime,
                lane=lane,
                candidate_score=candidate_score,
                strategy_id="main",
                rules_pass=True,
                regime_state=self._market_regime.get_state(),
            )
            if not ml_decision.allow_entry:
                self.event_log.log(
                    "entry_rejected",
                    {
                        "token_mint": token_mint,
                        "reason": "ml_gate_rejected",
                        "rule_id": match.selected_rule.rule_id,
                        "entry_lane": lane,
                        "candidate_score": round(candidate_score, 6),
                        "ml_probability": round(float(ml_decision.probability), 6),
                        "ml_threshold": float(ml_decision.threshold),
                        "ml_mode": ml_decision.mode,
                        "ml_model_ready": bool(ml_decision.model_ready),
                    },
                )
                self._set_candidate_cooldown(token_mint, now)
                continue

            extra_metadata = self._ml_metadata(
                probability=ml_decision.probability,
                threshold=ml_decision.threshold,
                mode=ml_decision.mode,
                model_ready=ml_decision.model_ready,
                reason=ml_decision.reason,
                feature_map=ml_decision.feature_map,
                lane=lane,
                candidate_score=candidate_score,
            )
            # Resolve pre-fetched BUY quote (submitted during ranking-window enqueue).
            # If the ranking window was long enough the future is already done — no wait.
            pre_buy: PaperTradeEstimate | None = None
            buy_future = item.get("buy_quote_future")
            if buy_future is not None:
                wait_started = time.monotonic()
                latency_trace["buy_quote_prefetch_ready"] = bool(
                    getattr(buy_future, "done", lambda: False)()
                )
                try:
                    # 0.5s timeout: future has been running since enqueue (~2s ago normally).
                    # Increased from 0.05s — 50ms was too tight for early queue flushes where
                    # the future may have only run for ~100ms (Jupiter takes 200-400ms min).
                    pre_buy = buy_future.result(timeout=0.5)
                except Exception:
                    pre_buy = None
                latency_trace["buy_quote_prefetch_wait_ms"] = (
                    time.monotonic() - wait_started
                ) * 1000.0
                latency_trace["buy_quote_prefetch_hit"] = pre_buy is not None

            # Resolve pre-built swap TX for live mode (sign+broadcast only at fire time).
            prebuilt_tx = None
            swap_tx_future = item.get("swap_tx_future")
            if swap_tx_future is not None:
                wait_started = time.monotonic()
                latency_trace["swap_tx_prefetch_ready"] = bool(
                    getattr(swap_tx_future, "done", lambda: False)()
                )
                try:
                    prebuilt_tx = swap_tx_future.result(timeout=0.5)
                except Exception:
                    prebuilt_tx = None
                latency_trace["swap_tx_prefetch_wait_ms"] = (
                    time.monotonic() - wait_started
                ) * 1000.0
                latency_trace["swap_tx_prefetch_hit"] = prebuilt_tx is not None

            guard_started = time.monotonic()
            _event_source = getattr(item.get("event"), "source_program", None)
            preflight_ok, paper_entry_estimate, guard_metadata = self._paper_entry_guard(
                token_mint=token_mint,
                size_sol=proposed_size,
                strategy_id="main",
                rejection_event="main_entry_rejected",
                rejection_payload={
                    "rule_id": match.selected_rule.rule_id,
                    "entry_lane": lane,
                    "candidate_score": round(candidate_score, 6),
                    "source_program": _event_source,
                },
                pre_buy_estimate=pre_buy,
                local_quote_engine=self.local_quote_engine,
                features=features,
                source_program=_event_source,
            )
            latency_trace["paper_entry_guard_ms"] = (time.monotonic() - guard_started) * 1000.0
            if not preflight_ok:
                self._set_candidate_cooldown(token_mint, now)
                continue
            if guard_metadata:
                extra_metadata = {**extra_metadata, **guard_metadata}

            entry_dispatch_at = datetime.now(tz=timezone.utc)
            latency_trace["entry_dispatch_at"] = entry_dispatch_at.isoformat()
            dispatch_ms = self._ms_between(arrival_at, entry_dispatch_at)
            if dispatch_ms is not None:
                latency_trace["arrival_to_entry_dispatch_ms"] = dispatch_ms
            self._set_feature_trace(features, "__latency_trace", latency_trace)
            extra_metadata["pipeline_latency_trace"] = latency_trace

            position = await self.entry_engine.execute_entry_async(
                features,
                match,
                size_sol=proposed_size,
                current_exposure_sol=current_exposure,
                open_position_count=open_count,
                extra_metadata=extra_metadata,
                paper_entry_estimate=paper_entry_estimate,
                prebuilt_tx=prebuilt_tx,
                source_program=_event_source,
            )
            if position is None:
                self._set_candidate_cooldown(token_mint, now)
                continue

            self.event_log.log(
                "candidate_selected",
                {
                    "token_mint": token_mint,
                    "rule_id": match.selected_rule.rule_id,
                    "entry_lane": lane,
                    "strategy_id": "main",
                    "source_program": _event_source,
                    "candidate_score": round(candidate_score, 6),
                    "wallet_bonus_total": round(
                        float(score_breakdown.get("tracked_wallet_bonus_total", 0.0)), 6
                    ),
                    "score_breakdown": score_breakdown,
                    "latency_trace": latency_trace,
                },
            )
            # Register with quote cache for pre-fetched exit quotes.
            if paper_entry_estimate is not None and paper_entry_estimate.out_amount > 0:
                self.quote_cache.register(token_mint, int(paper_entry_estimate.out_amount))
            # Register with live sell cache: start pre-building sell TX every 2s.
            if self.live_sell_cache is not None:
                import json as _json

                _pos_meta = _json.loads(position.get("metadata_json") or "{}")
                _token_out = int(
                    _pos_meta.get("live_out_amount")
                    or _pos_meta.get("initial_amount_received")
                    or 0
                )
                if _token_out > 0:
                    self.live_sell_cache.register(
                        token_mint,
                        _token_out,
                        strategy="main",
                        source_program=_event_source,
                    )
            self._notify_entry(position, [rule.rule_id for rule in match.matched_rules])
            self._set_candidate_cooldown(token_mint, now)
            processed += 1
            # Update snapshot so subsequent candidates in this flush see the new position.
            _open_mints.add(token_mint)
            _base_exposure += proposed_size
            _base_open_count += 1

        self._write_status(queued_candidates=len(self._ranked_candidates))
        return processed

    def _set_candidate_cooldown(self, token_mint: str, now: datetime) -> None:
        """Mark one token as recently evaluated for short-term duplicate suppression."""
        self._candidate_cooldowns[token_mint] = now

    def _set_sniper_candidate_cooldown(self, token_mint: str, now: datetime) -> None:
        """Mark one token as recently evaluated by sniper strategy."""
        self._sniper_candidate_cooldowns[token_mint] = now

    def _cleanup_candidate_cooldowns(self, now: datetime) -> None:
        """Drop expired token candidate cooldowns."""
        cutoff = now - timedelta(seconds=max(self.config.candidate_cooldown_sec * 3, 60))
        self._candidate_cooldowns = {
            token: seen_at
            for token, seen_at in self._candidate_cooldowns.items()
            if seen_at >= cutoff
        }
        self._candidate_cooldown_logged_at = {
            token: seen_at
            for token, seen_at in self._candidate_cooldown_logged_at.items()
            if seen_at >= cutoff
        }
        self._candidate_defer_logged_at = {
            token: seen_at
            for token, seen_at in self._candidate_defer_logged_at.items()
            if seen_at >= cutoff
        }

    def _cleanup_sniper_candidate_cooldowns(self, now: datetime) -> None:
        """Drop expired sniper token cooldowns."""
        cooldown_sec = max(30, int(self.config.sniper_token_cooldown_sec))
        cutoff = now - timedelta(seconds=max(cooldown_sec * 3, 90))
        self._sniper_candidate_cooldowns = {
            token: seen_at
            for token, seen_at in self._sniper_candidate_cooldowns.items()
            if seen_at >= cutoff
        }

    def _set_wallet_candidate_cooldown(self, token_mint: str, now: datetime) -> None:
        """Mark one token as recently evaluated by wallet strategy."""
        self._wallet_candidate_cooldowns[token_mint] = now

    def _cleanup_wallet_candidate_cooldowns(self, now: datetime) -> None:
        """Drop expired wallet token cooldowns."""
        cooldown_sec = max(30, int(self.config.wallet_token_cooldown_sec))
        cutoff = now - timedelta(seconds=max(cooldown_sec * 3, 90))
        self._wallet_candidate_cooldowns = {
            token: seen_at
            for token, seen_at in self._wallet_candidate_cooldowns.items()
            if seen_at >= cutoff
        }

    def _cleanup_pending_candidates(self, now: datetime) -> None:
        """Drop stale pending candidates."""
        cutoff = now - timedelta(seconds=max(self.config.candidate_maturation_sec * 5, 120))
        self._pending_candidates = {
            token: item
            for token, item in self._pending_candidates.items()
            if item["created_at"] >= cutoff
        }

    def _build_stale_sweep_features(
        self, position: dict[str, Any], now: datetime
    ) -> RuntimeFeatures | None:
        """Construct a synthetic feature snapshot for timeout exits when token updates are quiet."""
        token_mint = str(position.get("token_mint") or "")
        if not token_mint:
            return None
        snapshot = self.token_cache.snapshot(token_mint, now) or {}
        metadata = json.loads(position.get("metadata_json") or "{}")

        entry_time_raw = position.get("entry_time")
        try:
            entry_time = datetime.fromisoformat(str(entry_time_raw).replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            entry_time = now
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        entry_price_sol = snapshot.get("last_price_sol")
        if entry_price_sol is None:
            entry_price_sol = metadata.get("last_price_sol_seen")
        if entry_price_sol is None:
            entry_price_sol = position.get("entry_price_sol")
        try:
            entry_price_sol = float(entry_price_sol or 0.0)
        except (TypeError, ValueError):
            entry_price_sol = 0.0
        if entry_price_sol <= 0:
            return None

        # ── Local AMM price override ──────────────────────────────────────────
        # The snapshot price can be stale (last seen at the entry spike).
        # If the pool has fresh reserves, derive a real executable mark price so
        # stop losses can fire correctly instead of bleeding to sniper_timeout.
        pos_entry_price = float(position.get("entry_price_sol") or 0.0)
        if (
            pos_entry_price > 0
            and self.local_quote_engine is not None
            and self.local_quote_engine.has_reserves(token_mint)
        ):
            remaining_raw = float(metadata.get("paper_remaining_token_raw", 0.0) or 0.0)
            remaining_cost_basis = float(metadata.get("paper_remaining_cost_basis_sol", 0.0) or 0.0)
            token_amount_int = int(round(remaining_raw))
            if token_amount_int > 0 and remaining_cost_basis > 0:
                lamports_out = self.local_quote_engine.quote_sell(token_mint, token_amount_int)
                if lamports_out and lamports_out > 0:
                    sell_ratio = (lamports_out / LAMPORTS_PER_SOL) / remaining_cost_basis
                    amm_mark_price = pos_entry_price * sell_ratio
                    if amm_mark_price > 0:
                        entry_price_sol = amm_mark_price

        token_age_sec = snapshot.get("token_age_sec")
        if token_age_sec is None:
            token_age_sec = max(0.0, (now - entry_time).total_seconds())

        triggering_wallet = str(position.get("triggering_wallet") or "")
        triggering_wallet_score = float(position.get("triggering_wallet_score") or 0.0)

        return RuntimeFeatures(
            token_mint=token_mint,
            entry_time=now,
            entry_price_sol=entry_price_sol,
            token_age_sec=float(token_age_sec),
            wallet_cluster_30s=int(snapshot.get("wallet_cluster_30s", 0)),
            wallet_cluster_120s=int(snapshot.get("wallet_cluster_120s", 0)),
            volume_sol_30s=float(snapshot.get("volume_sol_30s", 0.0)),
            volume_sol_60s=float(snapshot.get("volume_sol_60s", 0.0)),
            tx_count_30s=int(snapshot.get("tx_count_30s", 0)),
            tx_count_60s=int(snapshot.get("tx_count_60s", 0)),
            price_change_30s=snapshot.get("price_change_30s"),
            price_change_60s=snapshot.get("price_change_60s"),
            triggering_wallet=triggering_wallet,
            triggering_wallet_score=triggering_wallet_score,
            aggregated_wallet_score=float(snapshot.get("aggregated_wallet_score", 0.0)),
            tracked_wallet_present_60s=bool(snapshot.get("tracked_wallet_present_60s", False)),
            tracked_wallet_count_60s=int(snapshot.get("tracked_wallet_count_60s", 0)),
            tracked_wallet_score_sum_60s=float(snapshot.get("tracked_wallet_score_sum_60s", 0.0)),
            raw={**snapshot, "__synthetic_sweep": True},
        )

    def _flush_post_close_snapshots(self, now: datetime) -> None:
        """Take feature snapshots at 30/60/90/120s after close and log to events.jsonl."""
        if not self._post_close_watch:
            return
        observe_sec = int(getattr(self.config, "post_close_observe_sec", 120))
        snap_offsets = {30, 60, 90, observe_sec}

        for token_mint in list(self._post_close_watch):
            watch = self._post_close_watch[token_mint]
            elapsed = (now - watch["close_time"]).total_seconds()

            for t in sorted(snap_offsets - watch["snaps_taken"]):
                if elapsed < t:
                    break
                snap = self.token_cache.snapshot(token_mint, now)
                if snap:
                    exit_price = watch["exit_price_sol"]
                    price_now = snap.get("last_price_sol_reliable") or snap.get("entry_price_sol")
                    price_change = None
                    if exit_price and exit_price > 1e-12 and price_now:
                        price_change = round(float(price_now) / float(exit_price) - 1.0, 6)
                    self.event_log.log(
                        "post_close_observation",
                        {
                            "token_mint": token_mint,
                            "position_id": watch["position_id"],
                            "strategy_id": watch["strategy_id"],
                            "seconds_after_close": t,
                            "exit_reason": watch["exit_reason"],
                            "exit_pnl_sol": watch["exit_pnl_sol"],
                            "price_change_from_exit": price_change,
                            "feature_snapshot": snap,
                        },
                    )
                watch["snaps_taken"].add(t)

            if elapsed >= observe_sec:
                self._post_close_watch.pop(token_mint, None)

    async def _stale_sweep_ticker(self) -> None:
        """Wall-clock-driven stale sweep loop.

        Fix B: the in-loop calls from `_process_events` / `run_forever` only
        fire while the ws event stream is delivering events. If the stream
        pauses (or a sniper position's pool goes quiet without anyone else
        triggering `_process_events`), the stale sweep stalls and
        wall-clock exits like `sniper_timeout` never evaluate. This task
        runs independently so the sweep interval is honoured regardless of
        event flow. The interval gate inside `_run_stale_exit_sweep` still
        dedupes against the in-loop calls.
        """
        interval = max(1, int(self.config.stale_sweep_sec))
        tick = max(0.5, float(interval) / 2.0)
        while True:
            try:
                await asyncio.sleep(tick)
                self._run_stale_exit_sweep(datetime.now(tz=timezone.utc))
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("stale sweep ticker iteration failed: %s", exc)

    def _run_stale_exit_sweep(self, now: datetime) -> None:
        """Force timeout exits to be evaluated even if a token stops emitting updates."""
        interval = max(1, int(self.config.stale_sweep_sec))
        if self._last_stale_sweep_at is not None:
            elapsed = (now - self._last_stale_sweep_at).total_seconds()
            if elapsed < interval:
                return
        self._last_stale_sweep_at = now

        open_positions = self.position_manager.list_open_positions()
        token_to_position: dict[str, dict[str, Any]] = {}
        for row in open_positions:
            token = str(row.get("token_mint") or "")
            if token and token not in token_to_position:
                token_to_position[token] = row

        for position in token_to_position.values():
            feature_started = time.monotonic()
            features = self._build_stale_sweep_features(position, now)
            if features is None:
                continue
            exit_trace = {
                "strategy_path": "exit_stale_sweep",
                "event_source": "stale_sweep",
                "token_mint": str(position.get("token_mint") or ""),
                "exit_trigger": "stale_sweep",
                "stale_sweep_started_at": now.isoformat(),
                "stale_sweep_feature_build_ms": (time.monotonic() - feature_started) * 1000.0,
            }
            self._set_feature_trace(features, "__exit_latency_trace", exit_trace)
            self.exit_engine.process(features)
        self._notify_exit_events()
        self._flush_post_close_snapshots(now)

        # Refresh unrealized_pnl_sol for every still-open position so the
        # dashboard shows current mark-to-market values (at most 10s stale).
        # Prefer last_reliable_pnl_multiple (outlier-guard filtered); fall back
        # to last_pnl_multiple only when no reliable tick has landed yet — this
        # avoids displaying thin-pool / synthetic-sweep price spikes the exit
        # engine already chose to ignore.
        for pos in self.position_manager.list_open_positions():
            try:
                meta = json.loads(pos.get("metadata_json") or "{}")
                reliable = meta.get("last_reliable_pnl_multiple")
                if reliable is None:
                    reliable = meta.get("last_pnl_multiple", 0.0)
                last_pnl_mult = float(reliable or 0.0)
                size_sol = float(pos.get("size_sol", 0.0) or 0.0)
                if size_sol > 0:
                    self.position_manager.set_unrealized_pnl(
                        int(pos["id"]), size_sol * last_pnl_mult
                    )
            except Exception:
                pass

        self._write_status(last_stale_sweep_at=now.isoformat())

    def _should_defer_candidate(self, features) -> bool:
        """Return whether a candidate looks too early for immediate entry."""
        return (
            features.wallet_cluster_30s < 2
            or features.tx_count_30s < 2
            or features.volume_sol_30s < 5.0
        )

    def _defer_candidate(self, event, features, arrival_time: datetime) -> None:
        """Soft-reject one early candidate and wait for the next real event."""
        last_logged = self._candidate_defer_logged_at.get(event.token_mint)
        log_interval_sec = max(2.0, float(self.config.candidate_cooldown_sec) / 4.0)
        if (
            last_logged is not None
            and (arrival_time - last_logged).total_seconds() < log_interval_sec
        ):
            return
        latency_trace = self._copy_trace(features.raw.get("__latency_trace"))
        latency_trace["candidate_deferred_at"] = arrival_time.isoformat()
        latency_trace["candidate_reevaluate_mode"] = "next_event"
        latency_trace["candidate_maturation_target_ms"] = 0.0
        self.event_log.log(
            "candidate_deferred",
            {
                "token_mint": event.token_mint,
                "reason": "awaiting_cluster_confirmation",
                "reevaluate_mode": "next_event",
                "triggering_wallet": event.triggering_wallet,
                "feature_snapshot": self._feature_snapshot(features),
                "latency_trace": latency_trace,
            },
        )
        self._candidate_defer_logged_at[event.token_mint] = arrival_time

    async def _maybe_execute_sniper_entry(
        self, event, features: RuntimeFeatures, now: datetime
    ) -> int:
        """Evaluate and execute one sniper entry candidate."""
        latency_trace = self._copy_trace(features.raw.get("__latency_trace"))
        latency_trace["strategy_path"] = "sniper"
        latency_trace["sniper_eval_started_at"] = now.isoformat()
        self._set_feature_trace(features, "__latency_trace", latency_trace)

        if not self.sniper_engine.enabled:
            return 0
        if self._entries_paused:
            return 0
        if self.position_manager.has_open_position(event.token_mint):
            return 0

        self._cleanup_sniper_candidate_cooldowns(now)
        last_seen = self._sniper_candidate_cooldowns.get(event.token_mint)
        cooldown_sec = max(30, int(self.config.sniper_token_cooldown_sec))
        if last_seen is not None and (now - last_seen).total_seconds() < cooldown_sec:
            return 0

        allowed_sources = getattr(self.config, "sniper_allowed_sources", ())
        if allowed_sources and event.source_program not in allowed_sources:
            return 0

        # When filtering by source, use source-specific age so graduated tokens
        # (e.g. PUMP_AMM) aren't rejected because they were seen as PUMP_FUN first.
        if allowed_sources and event.source_program:
            source_age = self.token_cache.source_age_sec(
                event.token_mint, event.source_program, now
            )
            if source_age is not None:
                features.token_age_sec = source_age

        # Anti-wash guard: kill huge buy/sell ratios that don't translate into
        # real price movement. Thresholds widened 2026-04-20 — BSR 5k was too
        # tight (legit low-float tokens routinely hit 10k+ on first sells) and
        # the 15% price floor overlapped the +10% sniper exit target, killing
        # real targets. BSR>30k + price<5% is a much cleaner wash signature.
        ratio_30s = features.buy_sell_ratio_30s
        pchg_30s = features.price_change_30s
        if (
            ratio_30s is not None
            and pchg_30s is not None
            and ratio_30s > 30000.0
            and pchg_30s < 0.05
        ):
            self.event_log.log(
                "sniper_entry_rejected",
                {
                    "token_mint": event.token_mint,
                    "reason": "sniper_antiwash_guard",
                    "buy_sell_ratio_30s": round(float(ratio_30s), 2),
                    "price_change_30s": round(float(pchg_30s), 6),
                    "source_program": event.source_program,
                },
            )
            self._set_sniper_candidate_cooldown(event.token_mint, now)
            return 0

        failures = self.sniper_engine.entry_failures(features)
        if failures:
            return 0

        proposed_size = self.sniper_engine.proposed_size_sol()
        if proposed_size <= 0:
            return 0

        # Pre-warm the swap TX as early as possible so Jupiter runs concurrently
        # while we do exposure/risk/ML checks below (~5-50ms of overlap).
        # The sniper path has no ranking window so this is the only prefetch opportunity.
        # If evaluation later rejects the candidate the future is simply abandoned.
        _sniper_tx_future = self.trade_executor.submit_swap_tx_future(
            event.token_mint,
            proposed_size,
            strategy="sniper",
            source_program=event.source_program,
        )
        latency_trace["swap_tx_prefetch_submitted_at"] = datetime.now(tz=timezone.utc).isoformat()

        sniper_open_positions = self._strategy_open_positions("sniper")
        sniper_open_count = len(sniper_open_positions)
        sniper_exposure = self._strategy_exposure("sniper")
        if sniper_open_count >= int(self.config.sniper_max_open_positions):
            self.event_log.log(
                "sniper_entry_rejected",
                {
                    "token_mint": event.token_mint,
                    "reason": "sniper_max_open_positions",
                    "sniper_open_positions": sniper_open_count,
                    "sniper_max_open_positions": int(self.config.sniper_max_open_positions),
                },
            )
            self._set_sniper_candidate_cooldown(event.token_mint, now)
            return 0
        if sniper_exposure + proposed_size > float(self.config.sniper_max_exposure_sol):
            self.event_log.log(
                "sniper_entry_rejected",
                {
                    "token_mint": event.token_mint,
                    "reason": "sniper_max_exposure",
                    "sniper_exposure_sol": round(float(sniper_exposure), 6),
                    "proposed_size_sol": round(float(proposed_size), 6),
                    "sniper_max_exposure_sol": float(self.config.sniper_max_exposure_sol),
                },
            )
            self._set_sniper_candidate_cooldown(event.token_mint, now)
            return 0

        global_exposure = self._current_exposure()
        global_open_count = self.position_manager.open_position_count()
        allowed, reason = self.risk_manager.can_open(
            token_mint=event.token_mint,
            proposed_size_sol=proposed_size,
            open_position_count=global_open_count,
            total_exposure_sol=global_exposure,
        )
        if not allowed:
            self.event_log.log(
                "sniper_entry_rejected",
                {
                    "token_mint": event.token_mint,
                    "reason": reason,
                    "proposed_size_sol": round(float(proposed_size), 6),
                },
            )
            self._set_sniper_candidate_cooldown(event.token_mint, now)
            return 0

        if self.sniper_engine.use_runtime_rules:
            match = self.sniper_engine.build_runtime_match(features, self.rules)
            if match.selected_rule is None:
                self.event_log.log(
                    "sniper_entry_rejected",
                    {
                        "token_mint": event.token_mint,
                        "reason": match.rejection_reason or "sniper_no_matching_runtime_rule",
                        "detected_regime": match.detected_regime,
                        "feature_snapshot": self._feature_snapshot(features),
                        "closest_rule_misses": closest_rule_misses(
                            features,
                            self.rules,
                            match.detected_regime,
                            limit=3,
                        ),
                    },
                )
                self._set_sniper_candidate_cooldown(event.token_mint, now)
                return 0
        else:
            match = self.sniper_engine.build_match()

        ml_decision = self.ml_filter.evaluate_candidate(
            features=features,
            rule=match.selected_rule,
            detected_regime=match.detected_regime,
            lane="sniper",
            candidate_score=None,
            strategy_id="sniper",
            rules_pass=match.selected_rule is not None,
            regime_state=self._market_regime.get_state(),
        )
        if not ml_decision.allow_entry:
            self.event_log.log(
                "sniper_entry_rejected",
                {
                    "token_mint": event.token_mint,
                    "reason": "ml_gate_rejected",
                    "ml_probability": round(float(ml_decision.probability), 6),
                    "ml_threshold": float(ml_decision.threshold),
                    "ml_mode": ml_decision.mode,
                    "ml_model_ready": bool(ml_decision.model_ready),
                    "feature_snapshot": self._feature_snapshot(features),
                },
            )
            self._set_sniper_candidate_cooldown(event.token_mint, now)
            return 0

        extra_metadata = self._ml_metadata(
            probability=ml_decision.probability,
            threshold=ml_decision.threshold,
            mode=ml_decision.mode,
            model_ready=ml_decision.model_ready,
            reason=ml_decision.reason,
            feature_map=ml_decision.feature_map,
            lane="sniper",
            candidate_score=None,
        )
        preflight_ok, paper_entry_estimate, guard_metadata = self._paper_entry_guard(
            token_mint=event.token_mint,
            size_sol=proposed_size,
            strategy_id="sniper",
            rejection_event="sniper_entry_rejected",
            rejection_payload={
                "rule_id": match.selected_rule.rule_id if match.selected_rule else None,
                "entry_lane": "sniper",
                "source_program": event.source_program,
            },
            local_quote_engine=self.local_quote_engine,
            features=features,
            source_program=event.source_program,
        )
        if not preflight_ok:
            self._set_sniper_candidate_cooldown(event.token_mint, now)
            return 0
        if guard_metadata:
            extra_metadata = {**extra_metadata, **guard_metadata}

        # Resolve the pre-warmed swap TX (started at entry of this function).
        # Give it up to 0.3s; if done the sniper fires as sign+broadcast only.
        # If not yet ready (rare: evaluation faster than Jupiter response) we fall
        # back to execute_buy_async which doesn't block the event loop.
        sniper_prebuilt_tx = None
        if _sniper_tx_future is not None:
            wait_started = time.monotonic()
            latency_trace["swap_tx_prefetch_ready"] = bool(
                getattr(_sniper_tx_future, "done", lambda: False)()
            )
            try:
                sniper_prebuilt_tx = _sniper_tx_future.result(timeout=0.3)
            except Exception:
                sniper_prebuilt_tx = None
            latency_trace["swap_tx_prefetch_wait_ms"] = (time.monotonic() - wait_started) * 1000.0
            latency_trace["swap_tx_prefetch_hit"] = sniper_prebuilt_tx is not None

        dispatch_time = datetime.now(tz=timezone.utc)
        latency_trace["entry_dispatch_at"] = dispatch_time.isoformat()
        arrival_to_dispatch_ms = self._ms_between(now, dispatch_time)
        if arrival_to_dispatch_ms is not None:
            latency_trace["arrival_to_entry_dispatch_ms"] = arrival_to_dispatch_ms
        self._set_feature_trace(features, "__latency_trace", latency_trace)
        extra_metadata["pipeline_latency_trace"] = latency_trace

        position = await self.entry_engine.execute_entry_async(
            features,
            match,
            size_sol=proposed_size,
            current_exposure_sol=global_exposure,
            open_position_count=global_open_count,
            strategy_id="sniper",
            extra_metadata=extra_metadata,
            paper_entry_estimate=paper_entry_estimate,
            prebuilt_tx=sniper_prebuilt_tx,
            source_program=event.source_program,
        )
        if position is None:
            self._set_sniper_candidate_cooldown(event.token_mint, now)
            return 0

        self.event_log.log(
            "sniper_candidate_selected",
            {
                "token_mint": event.token_mint,
                "strategy_id": "sniper",
                "rule_id": match.selected_rule.rule_id
                if match.selected_rule
                else "sniper_scalp_v1",
                "size_sol": round(float(proposed_size), 6),
                "feature_snapshot": self._feature_snapshot(features),
                "latency_trace": latency_trace,
            },
        )
        # Register with quote cache so TP/stop verify uses a pre-fetched quote.
        if paper_entry_estimate is not None and paper_entry_estimate.out_amount > 0:
            self.quote_cache.register(event.token_mint, int(paper_entry_estimate.out_amount))
        # Register with live sell cache.
        if self.live_sell_cache is not None:
            import json as _json

            _pos_meta = _json.loads(position.get("metadata_json") or "{}")
            _token_out = int(
                _pos_meta.get("live_out_amount") or _pos_meta.get("initial_amount_received") or 0
            )
            if _token_out > 0:
                self.live_sell_cache.register(
                    event.token_mint,
                    _token_out,
                    strategy="sniper",
                    source_program=event.source_program,
                )
        self._notify_entry(position, [match.selected_rule.rule_id] if match.selected_rule else [])
        self._set_sniper_candidate_cooldown(event.token_mint, now)
        return 1

    async def _maybe_execute_wallet_entry(
        self, event, features: RuntimeFeatures, now: datetime
    ) -> int:
        """Evaluate and execute one wallet-cluster entry candidate."""
        latency_trace = self._copy_trace(features.raw.get("__latency_trace"))
        latency_trace["strategy_path"] = "wallet"
        latency_trace["wallet_eval_started_at"] = now.isoformat()
        self._set_feature_trace(features, "__latency_trace", latency_trace)

        if not self.wallet_engine.enabled:
            return 0
        if self._entries_paused:
            return 0
        if self.position_manager.has_open_position(event.token_mint):
            return 0

        # Context attached to every wallet rejection payload so the dashboard
        # can surface cluster evidence even when the rejection path strips
        # feature_snapshot (e.g. shared paper-entry guard).
        _matched_wallets = list(getattr(event, "tracked_wallets", ()) or ())
        _raw_snap = features.raw or {}
        _cluster_30s = int(_raw_snap.get("tracked_wallet_cluster_30s") or 0)
        _cluster_120s = int(_raw_snap.get("tracked_wallet_cluster_120s") or 0)
        _cluster_300s = int(
            _raw_snap.get("tracked_wallet_cluster_300s") or features.tracked_wallet_count_60s or 0
        )
        wallet_ctx = {
            "matched_tracked_wallets": _matched_wallets,
            "tracked_wallet_cluster_30s": _cluster_30s,
            "tracked_wallet_cluster_120s": _cluster_120s,
            "tracked_wallet_cluster_300s": _cluster_300s,
            "strategy_id": "wallet",
        }

        self._cleanup_wallet_candidate_cooldowns(now)
        last_seen = self._wallet_candidate_cooldowns.get(event.token_mint)
        cooldown_sec = max(30, int(self.config.wallet_token_cooldown_sec))
        if last_seen is not None and (now - last_seen).total_seconds() < cooldown_sec:
            return 0

        allowed_sources = getattr(self.config, "wallet_allowed_sources", ())
        if allowed_sources and event.source_program not in allowed_sources:
            return 0

        if allowed_sources and event.source_program:
            source_age = self.token_cache.source_age_sec(
                event.token_mint, event.source_program, now
            )
            if source_age is not None:
                features.token_age_sec = source_age

        if self.wallet_engine.copytrading_enabled:
            pool = self.stream.wallet_scores
            trigger_wallet = str(getattr(event, "triggering_wallet", "") or "")
            trigger_in_pool = bool(trigger_wallet) and trigger_wallet in pool
            trigger_score = float(pool.get(trigger_wallet, 0.0)) if trigger_in_pool else 0.0

            # Fallback: some events (esp. pair-first discovery) arrive with a
            # non-tracked triggering_wallet but carry tracked wallets in
            # event.tracked_wallets (cross-referenced by helius_ws). Copy the
            # highest-scoring tracked wallet on the event if the direct
            # trigger isn't a pool member.
            if not trigger_in_pool:
                tracked_on_event = [
                    w for w in (getattr(event, "tracked_wallets", ()) or ()) if w in pool
                ]
                if tracked_on_event:
                    best = max(tracked_on_event, key=lambda w: float(pool.get(w, 0.0)))
                    trigger_wallet = best
                    trigger_score = float(pool.get(best, 0.0))
                    trigger_in_pool = True

            if self.wallet_engine.copy_wallet_qualifies(
                trigger_wallet, trigger_score, in_pool=trigger_in_pool
            ):
                return await self._maybe_execute_wallet_copy_entry(
                    event,
                    features,
                    now,
                    trigger_wallet=trigger_wallet,
                    trigger_score=trigger_score,
                    wallet_ctx=wallet_ctx,
                    latency_trace=latency_trace,
                )
            # Fall through to consensus: the triggering wallet isn't in our
            # pool, but other tracked wallets may have converged on this token.
            # Consensus gates (cluster + score_sum) decide if entry is warranted.

        failures = self.wallet_engine.entry_failures(features)
        if failures:
            # Emit telemetry for near-miss clusters (partial tracked-wallet
            # signal) so the dashboard funnel reflects 5-min-window gate
            # activity. Skip the firehose of zero-signal tokens.
            has_partial_signal = bool(
                _matched_wallets or _cluster_30s or _cluster_120s or _cluster_300s
            )
            if has_partial_signal:
                self.event_log.log(
                    "wallet_entry_rejected",
                    {
                        **wallet_ctx,
                        "token_mint": event.token_mint,
                        "reason": "wallet_engine_gate",
                        "failures": failures,
                    },
                )
            return 0

        proposed_size = self.wallet_engine.proposed_size_sol()
        if proposed_size <= 0:
            return 0

        # Force Jupiter routing: wallet clusters often hit tokens that have
        # already graduated to pump-AMM, where the native builder overflows
        # with preflight Custom 6023. Mirrors the main-lane path.
        _wallet_tx_future = self.trade_executor.submit_swap_tx_future(
            event.token_mint,
            proposed_size,
            prefer_jupiter=True,
            strategy="wallet",
            source_program=event.source_program,
        )
        latency_trace["swap_tx_prefetch_submitted_at"] = datetime.now(tz=timezone.utc).isoformat()

        wallet_open_positions = self._strategy_open_positions("wallet")
        wallet_open_count = len(wallet_open_positions)
        wallet_exposure = self._strategy_exposure("wallet")
        if wallet_open_count >= int(self.config.wallet_max_open_positions):
            self.event_log.log(
                "wallet_entry_rejected",
                {
                    **wallet_ctx,
                    "token_mint": event.token_mint,
                    "reason": "wallet_max_open_positions",
                    "wallet_open_positions": wallet_open_count,
                    "wallet_max_open_positions": int(self.config.wallet_max_open_positions),
                },
            )
            self._set_wallet_candidate_cooldown(event.token_mint, now)
            return 0
        if wallet_exposure + proposed_size > float(self.config.wallet_max_exposure_sol):
            self.event_log.log(
                "wallet_entry_rejected",
                {
                    **wallet_ctx,
                    "token_mint": event.token_mint,
                    "reason": "wallet_max_exposure",
                    "wallet_exposure_sol": round(float(wallet_exposure), 6),
                    "proposed_size_sol": round(float(proposed_size), 6),
                    "wallet_max_exposure_sol": float(self.config.wallet_max_exposure_sol),
                },
            )
            self._set_wallet_candidate_cooldown(event.token_mint, now)
            return 0

        global_exposure = self._current_exposure()
        global_open_count = self.position_manager.open_position_count()
        allowed, reason = self.risk_manager.can_open(
            token_mint=event.token_mint,
            proposed_size_sol=proposed_size,
            open_position_count=global_open_count,
            total_exposure_sol=global_exposure,
        )
        if not allowed:
            self.event_log.log(
                "wallet_entry_rejected",
                {
                    **wallet_ctx,
                    "token_mint": event.token_mint,
                    "reason": reason,
                    "proposed_size_sol": round(float(proposed_size), 6),
                },
            )
            self._set_wallet_candidate_cooldown(event.token_mint, now)
            return 0

        match = self.wallet_engine.build_match()

        ml_decision = self.ml_filter.evaluate_candidate(
            features=features,
            rule=match.selected_rule,
            detected_regime=match.detected_regime,
            lane="wallet",
            candidate_score=None,
            strategy_id="wallet",
            rules_pass=match.selected_rule is not None,
            regime_state=self._market_regime.get_state(),
        )
        if not ml_decision.allow_entry:
            self.event_log.log(
                "wallet_entry_rejected",
                {
                    **wallet_ctx,
                    "token_mint": event.token_mint,
                    "reason": "ml_gate_rejected",
                    "ml_probability": round(float(ml_decision.probability), 6),
                    "ml_threshold": float(ml_decision.threshold),
                    "ml_mode": ml_decision.mode,
                    "ml_model_ready": bool(ml_decision.model_ready),
                    "feature_snapshot": self._feature_snapshot(features),
                },
            )
            self._set_wallet_candidate_cooldown(event.token_mint, now)
            return 0

        extra_metadata = self._ml_metadata(
            probability=ml_decision.probability,
            threshold=ml_decision.threshold,
            mode=ml_decision.mode,
            model_ready=ml_decision.model_ready,
            reason=ml_decision.reason,
            feature_map=ml_decision.feature_map,
            lane="wallet",
            candidate_score=None,
        )

        matched_wallets = _matched_wallets
        if matched_wallets:
            extra_metadata["matched_tracked_wallets"] = matched_wallets

        preflight_ok, paper_entry_estimate, guard_metadata = self._paper_entry_guard(
            token_mint=event.token_mint,
            size_sol=proposed_size,
            strategy_id="wallet",
            rejection_event="wallet_entry_rejected",
            rejection_payload={
                **wallet_ctx,
                "rule_id": match.selected_rule.rule_id if match.selected_rule else None,
                "entry_lane": "wallet",
                "source_program": event.source_program,
            },
            local_quote_engine=self.local_quote_engine,
            features=features,
            source_program=event.source_program,
        )
        if not preflight_ok:
            self._set_wallet_candidate_cooldown(event.token_mint, now)
            return 0
        if guard_metadata:
            extra_metadata = {**extra_metadata, **guard_metadata}

        wallet_prebuilt_tx = None
        if _wallet_tx_future is not None:
            wait_started = time.monotonic()
            latency_trace["swap_tx_prefetch_ready"] = bool(
                getattr(_wallet_tx_future, "done", lambda: False)()
            )
            try:
                # Wallet lane has a ~30s window after cluster trigger, so give
                # Jupiter Metis time to finish building the TX. Prior 300ms was
                # copy-pasted from sniper and caused 29/34 prefetch timeouts in
                # live sessions, forcing fallback to a fresh (and often corrupt)
                # Jupiter quote. 800ms covers typical Metis p99 on fresh pools.
                wallet_prebuilt_tx = _wallet_tx_future.result(timeout=0.8)
            except Exception:
                wallet_prebuilt_tx = None
            latency_trace["swap_tx_prefetch_wait_ms"] = (time.monotonic() - wait_started) * 1000.0
            latency_trace["swap_tx_prefetch_hit"] = wallet_prebuilt_tx is not None

        dispatch_time = datetime.now(tz=timezone.utc)
        latency_trace["entry_dispatch_at"] = dispatch_time.isoformat()
        arrival_to_dispatch_ms = self._ms_between(now, dispatch_time)
        if arrival_to_dispatch_ms is not None:
            latency_trace["arrival_to_entry_dispatch_ms"] = arrival_to_dispatch_ms
        self._set_feature_trace(features, "__latency_trace", latency_trace)
        extra_metadata["pipeline_latency_trace"] = latency_trace

        position = await self.entry_engine.execute_entry_async(
            features,
            match,
            size_sol=proposed_size,
            current_exposure_sol=global_exposure,
            open_position_count=global_open_count,
            strategy_id="wallet",
            extra_metadata=extra_metadata,
            paper_entry_estimate=paper_entry_estimate,
            prebuilt_tx=wallet_prebuilt_tx,
            source_program=event.source_program,
        )
        if position is None:
            self._set_wallet_candidate_cooldown(event.token_mint, now)
            return 0

        self.event_log.log(
            "wallet_candidate_selected",
            {
                "token_mint": event.token_mint,
                "strategy_id": "wallet",
                "rule_id": match.selected_rule.rule_id
                if match.selected_rule
                else "wallet_cluster_v1",
                "size_sol": round(float(proposed_size), 6),
                "matched_tracked_wallets": matched_wallets,
                "feature_snapshot": self._feature_snapshot(features),
                "latency_trace": latency_trace,
            },
        )
        if paper_entry_estimate is not None and paper_entry_estimate.out_amount > 0:
            self.quote_cache.register(event.token_mint, int(paper_entry_estimate.out_amount))
        if self.live_sell_cache is not None:
            import json as _json

            _pos_meta = _json.loads(position.get("metadata_json") or "{}")
            _token_out = int(
                _pos_meta.get("live_out_amount") or _pos_meta.get("initial_amount_received") or 0
            )
            if _token_out > 0:
                self.live_sell_cache.register(
                    event.token_mint,
                    _token_out,
                    strategy="wallet",
                    source_program=event.source_program,
                )
        self._notify_entry(position, [match.selected_rule.rule_id] if match.selected_rule else [])
        self._set_wallet_candidate_cooldown(event.token_mint, now)
        return 1

    async def _maybe_execute_wallet_copy_entry(
        self,
        event,
        features: RuntimeFeatures,
        now: datetime,
        *,
        trigger_wallet: str,
        trigger_score: float,
        wallet_ctx: dict,
        latency_trace: dict,
    ) -> int:
        """Copy-mode wallet entry: follow a qualifying wallet's buy immediately."""
        copy_ctx = {
            **wallet_ctx,
            "copy_source_wallet": trigger_wallet,
            "copy_source_wallet_score": round(float(trigger_score), 4),
            "strategy_id": "wallet",
            "entry_mode": "copytrading",
        }

        failures = self.wallet_engine.copy_entry_failures(
            features,
            triggering_wallet=trigger_wallet,
            triggering_wallet_score=trigger_score,
            event_block_time=getattr(event, "block_time", None),
            now=now,
        )
        if failures:
            self.event_log.log(
                "wallet_copy_entry_rejected",
                {
                    **copy_ctx,
                    "token_mint": event.token_mint,
                    "reason": "wallet_copy_gate",
                    "failures": failures,
                },
            )
            return 0

        proposed_size = self.wallet_engine.copy_proposed_size_sol()
        if proposed_size <= 0:
            return 0

        _copy_tx_future = self.trade_executor.submit_swap_tx_future(
            event.token_mint,
            proposed_size,
            prefer_jupiter=True,
            strategy="wallet",
            source_program=event.source_program,
        )
        latency_trace["swap_tx_prefetch_submitted_at"] = datetime.now(tz=timezone.utc).isoformat()

        wallet_open_positions = self._strategy_open_positions("wallet")
        wallet_open_count = len(wallet_open_positions)
        wallet_exposure = self._strategy_exposure("wallet")
        if wallet_open_count >= int(self.config.wallet_max_open_positions):
            self.event_log.log(
                "wallet_copy_entry_rejected",
                {
                    **copy_ctx,
                    "token_mint": event.token_mint,
                    "reason": "wallet_max_open_positions",
                    "wallet_open_positions": wallet_open_count,
                    "wallet_max_open_positions": int(self.config.wallet_max_open_positions),
                },
            )
            self._set_wallet_candidate_cooldown(event.token_mint, now)
            return 0
        if wallet_exposure + proposed_size > float(self.config.wallet_max_exposure_sol):
            self.event_log.log(
                "wallet_copy_entry_rejected",
                {
                    **copy_ctx,
                    "token_mint": event.token_mint,
                    "reason": "wallet_max_exposure",
                    "wallet_exposure_sol": round(float(wallet_exposure), 6),
                    "proposed_size_sol": round(float(proposed_size), 6),
                    "wallet_max_exposure_sol": float(self.config.wallet_max_exposure_sol),
                },
            )
            self._set_wallet_candidate_cooldown(event.token_mint, now)
            return 0

        global_exposure = self._current_exposure()
        global_open_count = self.position_manager.open_position_count()
        allowed, reason = self.risk_manager.can_open(
            token_mint=event.token_mint,
            proposed_size_sol=proposed_size,
            open_position_count=global_open_count,
            total_exposure_sol=global_exposure,
        )
        if not allowed:
            self.event_log.log(
                "wallet_copy_entry_rejected",
                {
                    **copy_ctx,
                    "token_mint": event.token_mint,
                    "reason": reason,
                    "proposed_size_sol": round(float(proposed_size), 6),
                },
            )
            self._set_wallet_candidate_cooldown(event.token_mint, now)
            return 0

        match = self.wallet_engine.build_match(copy_mode=True)

        ml_bypass = bool(self.config.wallet_copy_ml_bypass)
        if not ml_bypass:
            ml_decision = self.ml_filter.evaluate_candidate(
                features=features,
                rule=match.selected_rule,
                detected_regime=match.detected_regime,
                lane="wallet",
                candidate_score=None,
                strategy_id="wallet",
                rules_pass=match.selected_rule is not None,
                regime_state=self._market_regime.get_state(),
            )
            if not ml_decision.allow_entry:
                self.event_log.log(
                    "wallet_copy_entry_rejected",
                    {
                        **copy_ctx,
                        "token_mint": event.token_mint,
                        "reason": "ml_gate_rejected",
                        "ml_probability": round(float(ml_decision.probability), 6),
                        "ml_threshold": float(ml_decision.threshold),
                        "ml_mode": ml_decision.mode,
                        "ml_model_ready": bool(ml_decision.model_ready),
                        "feature_snapshot": self._feature_snapshot(features),
                    },
                )
                self._set_wallet_candidate_cooldown(event.token_mint, now)
                return 0
            extra_metadata = self._ml_metadata(
                probability=ml_decision.probability,
                threshold=ml_decision.threshold,
                mode=ml_decision.mode,
                model_ready=ml_decision.model_ready,
                reason=ml_decision.reason,
                feature_map=ml_decision.feature_map,
                lane="wallet",
                candidate_score=None,
            )
        else:
            extra_metadata = {}

        copy_source_wallets = list(
            dict.fromkeys(
                [trigger_wallet] + [w for w in getattr(event, "tracked_wallets", ()) or () if w]
            )
        )
        extra_metadata.update(
            {
                "wallet_copy": True,
                "copy_source_wallets": copy_source_wallets,
                "copy_source_wallet_score": round(float(trigger_score), 4),
                "entry_mode": "copytrading",
                "matched_tracked_wallets": copy_source_wallets,
            }
        )

        preflight_ok, paper_entry_estimate, guard_metadata = self._paper_entry_guard(
            token_mint=event.token_mint,
            size_sol=proposed_size,
            strategy_id="wallet",
            rejection_event="wallet_copy_entry_rejected",
            rejection_payload={
                **copy_ctx,
                "rule_id": match.selected_rule.rule_id if match.selected_rule else None,
                "entry_lane": "wallet",
                "source_program": event.source_program,
            },
            local_quote_engine=self.local_quote_engine,
            features=features,
            source_program=event.source_program,
        )
        if not preflight_ok:
            self._set_wallet_candidate_cooldown(event.token_mint, now)
            return 0
        if guard_metadata:
            extra_metadata = {**extra_metadata, **guard_metadata}

        copy_prebuilt_tx = None
        if _copy_tx_future is not None:
            wait_started = time.monotonic()
            latency_trace["swap_tx_prefetch_ready"] = bool(
                getattr(_copy_tx_future, "done", lambda: False)()
            )
            try:
                copy_prebuilt_tx = _copy_tx_future.result(timeout=0.8)
            except Exception:
                copy_prebuilt_tx = None
            latency_trace["swap_tx_prefetch_wait_ms"] = (time.monotonic() - wait_started) * 1000.0
            latency_trace["swap_tx_prefetch_hit"] = copy_prebuilt_tx is not None

        dispatch_time = datetime.now(tz=timezone.utc)
        latency_trace["entry_dispatch_at"] = dispatch_time.isoformat()
        arrival_to_dispatch_ms = self._ms_between(now, dispatch_time)
        if arrival_to_dispatch_ms is not None:
            latency_trace["arrival_to_entry_dispatch_ms"] = arrival_to_dispatch_ms
        self._set_feature_trace(features, "__latency_trace", latency_trace)
        extra_metadata["pipeline_latency_trace"] = latency_trace

        position = await self.entry_engine.execute_entry_async(
            features,
            match,
            size_sol=proposed_size,
            current_exposure_sol=global_exposure,
            open_position_count=global_open_count,
            strategy_id="wallet",
            extra_metadata=extra_metadata,
            paper_entry_estimate=paper_entry_estimate,
            prebuilt_tx=copy_prebuilt_tx,
            source_program=event.source_program,
        )
        if position is None:
            self._set_wallet_candidate_cooldown(event.token_mint, now)
            return 0

        self.event_log.log(
            "wallet_copy_candidate_selected",
            {
                "token_mint": event.token_mint,
                "strategy_id": "wallet",
                "rule_id": match.selected_rule.rule_id if match.selected_rule else "wallet_copy_v1",
                "size_sol": round(float(proposed_size), 6),
                "copy_source_wallets": copy_source_wallets,
                "copy_source_wallet_score": round(float(trigger_score), 4),
                "feature_snapshot": self._feature_snapshot(features),
                "latency_trace": latency_trace,
            },
        )
        if paper_entry_estimate is not None and paper_entry_estimate.out_amount > 0:
            self.quote_cache.register(event.token_mint, int(paper_entry_estimate.out_amount))
        if self.live_sell_cache is not None:
            import json as _json

            _pos_meta = _json.loads(position.get("metadata_json") or "{}")
            _token_out = int(
                _pos_meta.get("live_out_amount") or _pos_meta.get("initial_amount_received") or 0
            )
            if _token_out > 0:
                self.live_sell_cache.register(
                    event.token_mint,
                    _token_out,
                    strategy="wallet",
                    source_program=event.source_program,
                )
        self._notify_entry(position, [match.selected_rule.rule_id] if match.selected_rule else [])
        self._set_wallet_candidate_cooldown(event.token_mint, now)
        return 1

    async def _process_pending_candidates(self, now: datetime) -> int:
        """Timer-based candidate maturation is disabled; re-evaluate on next event."""
        self._pending_candidates.clear()
        return 0

    async def _process_events(self, new_events: list) -> int:
        """Process candidate events."""
        processed = 0
        self._apply_new_session_if_requested()
        self._apply_end_session_if_requested()
        self._run_stale_exit_sweep(datetime.now(tz=timezone.utc))
        for event in new_events:
            arrival_time = datetime.now(tz=timezone.utc)
            event_trace = self._new_event_latency_trace(
                event=event, arrival_time=arrival_time, source="live"
            )
            # Throttle dict cleanup to once every 5s — these scan full in-memory dicts
            # and their cutoffs are in minutes, so per-event execution is pure waste.
            _now_ts = time.monotonic()
            if _now_ts - self._last_cleanup_ts >= 5.0:
                self._cleanup_candidate_cooldowns(arrival_time)
                self._cleanup_sniper_candidate_cooldowns(arrival_time)
                self._cleanup_wallet_candidate_cooldowns(arrival_time)
                self._cleanup_ranked_candidates(arrival_time)
                self._last_cleanup_ts = _now_ts
            self._write_status(
                last_seen_event_at=event.block_time.isoformat(),
                last_seen_token=event.token_mint,
                last_seen_wallet=event.triggering_wallet,
                last_seen_side=event.side,
                last_seen_event_time_source=getattr(event, "event_time_source", None),
                last_seen_provider_created_at=self._dt_to_iso(
                    self._normalize_dt(getattr(event, "provider_created_at", None))
                ),
                last_seen_stream_received_at=self._dt_to_iso(
                    self._normalize_dt(getattr(event, "stream_received_at", None))
                ),
                last_seen_parse_completed_at=self._dt_to_iso(
                    self._normalize_dt(getattr(event, "parse_completed_at", None))
                ),
                last_seen_source_slot=getattr(event, "source_slot", None),
            )
            ingest_started = time.monotonic()
            self.token_cache.ingest(event)
            event_trace["token_cache_ingest_ms"] = (time.monotonic() - ingest_started) * 1000.0
            self._maybe_broadcast_wallet_activity(event)
            feature_started = time.monotonic()
            features = build_runtime_features(event, self.token_cache, self.stream.wallet_scores)
            event_trace["feature_build_ms"] = (time.monotonic() - feature_started) * 1000.0
            if features is None:
                continue
            feature_built_at = datetime.now(tz=timezone.utc)
            event_trace["feature_built_at"] = feature_built_at.isoformat()
            arrival_to_feature_ms = self._ms_between(arrival_time, feature_built_at)
            if arrival_to_feature_ms is not None:
                event_trace["arrival_to_feature_snapshot_ms"] = arrival_to_feature_ms
            self._set_feature_trace(features, "__latency_trace", event_trace)
            exit_trace = self._copy_trace(event_trace)
            exit_trace["exit_signal_received_at"] = arrival_time.isoformat()
            exit_trace["exit_trigger"] = "event"
            # Tag the exit trace with the OWNING position's strategy instead of
            # carrying "main_candidate" (the default assigned at event arrival
            # before we knew whether a position existed). Prior behavior meant
            # every sniper-exit trace reported strategy_path=main_candidate,
            # masking which lane was actually executing the exit. Prefer a
            # sniper position over a main position when both exist for the
            # same token (sniper is the active-exit lane in that rare case).
            open_positions_for_token = self.position_manager.list_open_positions_for_token(
                event.token_mint
            )
            if open_positions_for_token:
                owning_position = next(
                    (
                        p
                        for p in open_positions_for_token
                        if str(p.get("strategy_id") or "") == "sniper"
                    ),
                    open_positions_for_token[0],
                )
                owning_strategy = str(owning_position.get("strategy_id") or "main")
                exit_trace["strategy_path"] = f"{owning_strategy}_exit"
                exit_trace["owning_strategy"] = owning_strategy
                exit_trace["owning_position_id"] = int(owning_position.get("id") or 0)
            self._set_feature_trace(features, "__exit_latency_trace", exit_trace)
            # Track unique token mints for market regime monitor (unique launches/5 min signal)
            self._market_regime.record_candidate_seen(token_mint=event.token_mint, now=arrival_time)
            if (
                event.side == "SELL"
                and bool(getattr(self.config, "wallet_copy_mirror_sell", False))
                and open_positions_for_token
            ):
                self._annotate_copy_mirror_sell(event, features, open_positions_for_token)
            # Fire exit checks as a background task so slow exits (Jupiter quotes,
            # RPC confirmation) don't block processing of the next event.
            self._schedule_exit_task(features, arrival_time)
            processed += await self._flush_ranked_candidates(arrival_time, force=False)
            if event.side != "BUY":
                continue
            if self._entries_paused:
                continue
            if not self._market_regime.is_favorable() and self.config.market_regime_gate_active:
                continue
            wallet_taken = await self._maybe_execute_wallet_entry(event, features, arrival_time)
            processed += wallet_taken
            if wallet_taken > 0 or self.position_manager.has_open_position(event.token_mint):
                continue
            if self.wallet_engine.copytrading_enabled and bool(
                getattr(self.config, "wallet_copy_disable_sniper", True)
            ):
                continue
            processed += await self._maybe_execute_sniper_entry(event, features, arrival_time)
            if not self.config.enable_main_strategy:
                continue
            # Main-lane source + age gate. When ``main_allowed_sources`` is
            # configured, only events from those sources reach the main lane;
            # and the token's source-specific age must fall inside
            # [main_min_token_age_sec, main_max_token_age_sec]. Mirrors the
            # sniper-lane filter pattern above — gates are enforced here
            # (not in rule conditions) because ``rule_matcher`` has no
            # ``token_age_sec_min`` key.
            _main_allowed = getattr(self.config, "main_allowed_sources", ()) or ()
            if _main_allowed and event.source_program not in _main_allowed:
                continue
            _main_min_age = float(getattr(self.config, "main_min_token_age_sec", 0.0) or 0.0)
            _main_max_age = float(getattr(self.config, "main_max_token_age_sec", 0.0) or 0.0)
            if (
                _main_allowed
                and event.source_program
                and (_main_min_age > 0.0 or _main_max_age > 0.0)
            ):
                _source_age = self.token_cache.source_age_sec(
                    event.token_mint, event.source_program, arrival_time
                )
                if _source_age is None:
                    continue
                if _main_min_age > 0.0 and _source_age < _main_min_age:
                    continue
                if _main_max_age > 0.0 and _source_age > _main_max_age:
                    self.event_log.log(
                        "main_entry_rejected",
                        {
                            "token_mint": event.token_mint,
                            "reason": "main_token_age_above_max",
                            "strategy_id": "main",
                            "source_program": event.source_program,
                            "token_age_sec": round(float(_source_age), 3),
                            "main_max_token_age_sec": _main_max_age,
                        },
                    )
                    continue
                features.token_age_sec = _source_age
            if self.position_manager.has_open_position(event.token_mint):
                continue
            last_candidate = self._candidate_cooldowns.get(event.token_mint)
            if last_candidate is not None:
                elapsed_sec = (arrival_time - last_candidate).total_seconds()
                if elapsed_sec < self.config.candidate_cooldown_sec:
                    last_logged = self._candidate_cooldown_logged_at.get(event.token_mint)
                    if last_logged is None or (arrival_time - last_logged).total_seconds() >= max(
                        1.0, self.config.candidate_cooldown_sec / 4
                    ):
                        self.event_log.log(
                            "entry_rejected",
                            {
                                "token_mint": event.token_mint,
                                "reason": "token_candidate_cooldown",
                                "cooldown_sec": self.config.candidate_cooldown_sec,
                                "seconds_since_last_candidate": round(elapsed_sec, 3),
                                "triggering_wallet": event.triggering_wallet,
                            },
                        )
                        self._candidate_cooldown_logged_at[event.token_mint] = arrival_time
                    continue
            if self.position_manager.has_open_position(event.token_mint):
                self.event_log.log(
                    "entry_rejected",
                    {
                        "token_mint": event.token_mint,
                        "reason": "existing_open_position",
                    },
                )
                self._set_candidate_cooldown(event.token_mint, arrival_time)
                continue
            quality_failures = self._entry_quality_failures(features)
            if quality_failures:
                if self._should_defer_candidate(features):
                    self._defer_candidate(event, features, arrival_time)
                    continue
                self.event_log.log(
                    "entry_rejected",
                    {
                        "token_mint": event.token_mint,
                        "reason": "entry_quality_gate",
                        "quality_failures": quality_failures,
                        "triggering_wallet": event.triggering_wallet,
                        "triggering_wallet_score": round(
                            float(features.triggering_wallet_score), 3
                        ),
                        "feature_snapshot": self._feature_snapshot(features),
                    },
                )
                self._set_candidate_cooldown(event.token_mint, arrival_time)
                continue
            lane, lane_failures = self._entry_lane_failures(features)
            if lane is None:
                if self._should_defer_candidate(features):
                    self._defer_candidate(event, features, arrival_time)
                    continue
                self.event_log.log(
                    "entry_rejected",
                    {
                        "token_mint": event.token_mint,
                        "reason": "entry_lane_gate",
                        "lane_failures": lane_failures,
                        "triggering_wallet": event.triggering_wallet,
                        "triggering_wallet_score": round(
                            float(features.triggering_wallet_score), 3
                        ),
                        "feature_snapshot": self._feature_snapshot(features),
                    },
                )
                self._set_candidate_cooldown(event.token_mint, arrival_time)
                continue
            match = select_rule(features, self.main_rules)
            if match.selected_rule is None:
                if self._should_defer_candidate(features):
                    self._defer_candidate(event, features, arrival_time)
                    continue
                rejection_payload = {
                    "token_mint": event.token_mint,
                    "reason": match.rejection_reason,
                    "detected_regime": match.detected_regime,
                    "triggering_wallet": event.triggering_wallet,
                    "triggering_wallet_score": round(float(features.triggering_wallet_score), 3),
                    "feature_snapshot": self._feature_snapshot(features),
                }
                if self.event_log.should_emit(
                    "entry_rejected",
                    {"reason": match.rejection_reason or ""},
                ):
                    rejection_payload["closest_rule_misses"] = closest_rule_misses(
                        features,
                        self.main_rules,
                        match.detected_regime,
                    )
                self.event_log.log(
                    "entry_rejected",
                    rejection_payload,
                )
                self._set_candidate_cooldown(event.token_mint, arrival_time)
                continue
            confirmation_failures = self._recovery_confirmation_failures(
                features, match.selected_rule
            )
            if confirmation_failures:
                self.event_log.log(
                    "entry_rejected",
                    {
                        "token_mint": event.token_mint,
                        "reason": "recovery_confirmation_gate",
                        "confirmation_failures": confirmation_failures,
                        "detected_regime": match.detected_regime,
                        "rule_id": match.selected_rule.rule_id,
                        "feature_snapshot": self._feature_snapshot(features),
                    },
                )
                continue
            queued = self._enqueue_ranked_candidate(
                event=event,
                features=features,
                match=match,
                lane=lane,
                queued_at=arrival_time,
                source="live",
                arrival_at=arrival_time,
            )
        self._pending_candidates.clear()
        processed += await self._flush_ranked_candidates(datetime.now(tz=timezone.utc), force=False)
        self._run_stale_exit_sweep(datetime.now(tz=timezone.utc))
        self._status_processed_events += processed
        self._write_status(pending_candidate_count=len(self._pending_candidates))
        return processed

    async def _exit_and_notify(self, features: RuntimeFeatures, arrival_time: datetime) -> None:
        """Run exit checks and notification as a background task.

        exit_engine.process() is synchronous and may call Jupiter HTTP (300-500ms
        on Layer-3 fallback).  Offloading it to run_in_executor keeps the event
        loop free to continue ingesting gRPC events while exits are processed.
        """
        try:
            exit_trace = self._copy_trace((features.raw or {}).get("__exit_latency_trace"))
            task_started_at = datetime.now(tz=timezone.utc)
            exit_trace["exit_task_started_at"] = task_started_at.isoformat()
            queue_ms = self._ms_between(arrival_time, task_started_at)
            if queue_ms is not None:
                exit_trace["exit_task_queue_ms"] = queue_ms
            self._set_feature_trace(features, "__exit_latency_trace", exit_trace)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.exit_engine.process, features)
            self._notify_exit_events()
            self._flush_post_close_snapshots(arrival_time)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("_exit_and_notify failed: %s", exc)

    async def _exit_worker(
        self, token_mint: str, features: RuntimeFeatures, arrival_time: datetime
    ) -> None:
        """Serialize exit work per token while preserving the latest pending signal."""
        current_features = features
        current_arrival = arrival_time
        while True:
            try:
                await self._exit_and_notify(current_features, current_arrival)
            finally:
                pending = self._pending_exit_inputs.pop(token_mint, None)
            if pending is None:
                break
            current_features, current_arrival = pending

    async def process_once(self) -> int:
        """Fallback polling cycle when websocket is unavailable."""
        return await self._process_events(self.stream.poll(self.last_seen_signatures))

    async def run_forever(self) -> None:
        """Run the bot until interrupted."""
        self._event_loop = asyncio.get_running_loop()
        self._event_loop.set_debug(False)
        self._event_loop.set_exception_handler(self._handle_loop_exception)
        self.quote_cache.start()
        if self.live_sell_cache is not None:
            self.live_sell_cache.start()
        if self.live_reconciler is not None:
            self.live_reconciler.start()
        try:
            if self.config.market_regime_sol_enabled:
                self._market_regime_task = self._track_background_task(
                    asyncio.create_task(
                        self._market_regime.start_sol_price_polling(),
                        name="market-regime-sol-poll",
                    )
                )
            # Fix B: decouple stale-exit sweeps from the event loop.
            self._track_background_task(
                asyncio.create_task(
                    self._stale_sweep_ticker(),
                    name="stale-sweep-ticker",
                )
            )
            self._notify_startup()
            self._write_status(
                status="starting",
                session_mode=self._session_mode,
                entries_paused=self._entries_paused,
            )
            self._apply_new_session_if_requested()
            self._apply_end_session_if_requested()
            mode = self._mode_label
            self.logger.info("Starting %s bot with %s active rules", mode, len(self.rules))
            if self.trade_executor.live:
                self.logger.warning("=" * 60)
                self.logger.warning("⚠️  LIVE TRADING IS ACTIVE – REAL FUNDS AT RISK")
                self.logger.warning("=" * 60)
            while True:
                try:
                    async for event in self.ws_monitor.events():
                        self._monitoring_mode = "chainstack_grpc"
                        try:
                            processed = await self._process_events([event])
                            self._run_stale_exit_sweep(datetime.now(tz=timezone.utc))
                            self._write_status(
                                status="running",
                                last_cycle="chainstack_grpc",
                                last_cycle_processed=processed,
                            )
                            if processed:
                                self.logger.info(
                                    "[%s] Processed %s Chainstack gRPC candidate entries at %s",
                                    mode,
                                    processed,
                                    datetime.now(tz=timezone.utc).isoformat(),
                                )
                        except Exception as exc:  # noqa: BLE001
                            self.logger.warning(
                                "Chainstack Yellowstone event processing failed: %s",
                                exc,
                            )
                            self._write_status(
                                status="running",
                                monitoring_mode="chainstack_grpc",
                                last_cycle="chainstack_grpc_event_error",
                                last_cycle_processed=0,
                                event_processing_failure=str(exc),
                            )
                            continue
                except WebsocketUnavailableError as exc:
                    self.logger.warning("Chainstack Yellowstone stream unavailable: %s", exc)
                    self._write_status(status="running", websocket_unavailable_reason=str(exc))
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("Chainstack Yellowstone monitor failed: %s", exc)
                    self._write_status(status="running", websocket_failure=str(exc))

                if self.config.discovery_mode == "pair_first":
                    self._monitoring_mode = "chainstack_grpc_retry_wait"
                    self._write_status(
                        status="running",
                        monitoring_mode=self._monitoring_mode,
                        last_cycle="chainstack_grpc_retry_wait",
                        last_cycle_processed=0,
                    )
                    self.logger.warning(
                        "[%s] Pair-first mode requires Chainstack Yellowstone stream; polling fallback is disabled. Retrying in %ss.",
                        mode,
                        self.config.poll_interval_sec,
                    )
                    await asyncio.sleep(self.config.poll_interval_sec)
                    continue

                self._monitoring_mode = "polling_fallback"
                self._write_status(status="running", monitoring_mode=self._monitoring_mode)
                while True:
                    processed = await self.process_once()
                    self._run_stale_exit_sweep(datetime.now(tz=timezone.utc))
                    self._write_status(
                        status="running",
                        last_cycle="polling",
                        last_cycle_processed=processed,
                    )
                    self.logger.info(
                        "[%s] Processed %s polling candidate entries at %s",
                        mode,
                        processed,
                        datetime.now(tz=timezone.utc).isoformat(),
                    )
                    await asyncio.sleep(self.config.poll_interval_sec)
        finally:
            for task in list(self._background_tasks):
                task.cancel()
            if self._background_tasks:
                await asyncio.gather(*list(self._background_tasks), return_exceptions=True)
            self.ws_monitor.close()
            self.quote_cache.stop()
            if self.live_sell_cache is not None:
                self.live_sell_cache.stop()
            if self.live_reconciler is not None:
                self.live_reconciler.stop()
            self.trade_executor.close()
