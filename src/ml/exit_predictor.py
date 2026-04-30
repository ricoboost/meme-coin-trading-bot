"""
ML-based exit predictor — decides whether to hold or exit early for open positions.

Two separate models (sniper and main) are trained on tick-level snapshots of open
positions with retrospective labeling:
  - Sniper label=1: position went on to hit sniper_take_profit
  - Sniper label=0: position ended in sniper_timeout or sniper_stop_out
  - Main label=1:   position reached at least TP1 (exit_stage >= 1 or pnl > threshold)
  - Main label=0:   position stopped or timed out before TP1

Feature innovation: uses CURRENT tick features (not entry-time) so the model can
detect mid-hold deterioration — volume collapsing, price reversing, momentum stalling.

Modes:
  shadow  — evaluate and log predictions without affecting exits (data collection)
  gate    — trigger early "ml_exit_early" exit when hold_probability < threshold
  off     — disabled, never fires
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.bot.models import ExitMLDecision, RuntimeFeatures

logger = logging.getLogger(__name__)

# ── Feature names ─────────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    # Position-state features (dynamic — update each tick)
    "hold_time_sec",
    "current_pnl_multiple",
    "max_pnl_multiple_seen",
    "pnl_drawdown_from_peak",
    "time_to_deadline_sec",  # sniper: max_hold-hold_time; main: -1
    # Current market snapshot (live tick, NOT entry-time values)
    "wallet_cluster_30s",
    "volume_sol_30s",
    "buy_volume_sol_30s",
    "tx_count_30s",
    "buy_sell_ratio_30s",
    "net_flow_sol_30s",
    "avg_trade_sol_30s",
    "price_change_30s",
    # Trend comparison: current vs entry (is momentum accelerating or dying?)
    "volume_vs_entry",
    "cluster_vs_entry",
    "buy_ratio_vs_entry",
    "flow_vs_entry",
    # Strategy context
    "strategy_is_sniper",
    "exit_stage",
]

N_FEATURES = len(FEATURE_NAMES)
_FEAT_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# Labels at which early exit wins — different for each strategy
_SNIPER_WIN_REASONS = {"sniper_take_profit"}
_MAIN_WIN_REASONS = {
    "tp1",
    "tp2",
    "tp3",
    "stage0_early_profit",
    "runner_trailing_stop",
    "stage1_timeout_low_positive",
    "pre_tp1_retrace_lock",
}


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
        return default if (f != f or f == float("inf") or f == float("-inf")) else f
    except (TypeError, ValueError):
        return default


def _safe_ratio(num: float, denom: float, default: float = 1.0) -> float:
    if abs(denom) < 1e-9:
        return default
    return num / denom


# ── ExitMLPredictor ────────────────────────────────────────────────────────────


class ExitMLPredictor:
    """
    ML exit predictor with per-tick sample buffering and retrospective labeling.

    Lifecycle:
      1. record_tick_sample() called on each tick for every open position
      2. record_closed_position() called when position closes → labels all buffered ticks
      3. After ml_exit_retrain_every closed positions → _retrain()
      4. evaluate_position() called on each tick to get hold/exit recommendation
    """

    def __init__(
        self,
        mode: str = "shadow",
        model_path: Path | None = None,
        samples_path: Path | None = None,
        sniper_threshold: float = 0.40,
        main_threshold: float = 0.45,
        min_samples: int = 50,
        retrain_every: int = 25,
        sniper_max_hold_sec: float = 75.0,
        event_log: Any = None,
    ) -> None:
        self.mode = mode.lower().strip()
        self.model_path = model_path or Path("models/exit_predictor.joblib")
        self.samples_path = samples_path or Path("data/live/ml_exit_samples.jsonl")
        self.sniper_threshold = sniper_threshold
        self.main_threshold = main_threshold
        self.min_samples = min_samples
        self.retrain_every = retrain_every
        self.sniper_max_hold_sec = sniper_max_hold_sec
        self.event_log = event_log

        # Per-strategy models
        self.models: dict[str, Any] = {"sniper": None, "main": None}
        self._model_versions: dict[str, str] = {"sniper": "", "main": ""}

        # Training buffer: list of (feature_vector, label)
        self._training_rows: dict[str, list[tuple[np.ndarray, int]]] = {
            "sniper": [],
            "main": [],
        }
        # Tick buffer: position_id → deque of tick dicts (capped at 20 per position)
        self._pending_ticks: dict[int, deque[dict]] = defaultdict(lambda: deque(maxlen=20))
        # Track how many positions since last retrain per strategy
        self._closed_since_retrain: dict[str, int] = {"sniper": 0, "main": 0}
        # Dedup guard
        self._recorded_position_ids: set[int] = set()

        self._classifier_cls = None
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier

            self._classifier_cls = HistGradientBoostingClassifier
        except ImportError:
            logger.warning(
                "sklearn not available — ExitMLPredictor will run in shadow/off mode only"
            )

        self.samples_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self._load_model()
        self._load_live_samples()
        self._bootstrap_from_db()

    # ── Feature vector construction ───────────────────────────────────────────

    def _build_vector(
        self,
        features: RuntimeFeatures,
        hold_time_sec: float,
        current_pnl_multiple: float,
        max_pnl_multiple_seen: float,
        exit_stage: int,
        strategy_id: str,
        entry_snapshot: dict[str, float] | None = None,
    ) -> np.ndarray:
        vec = np.zeros(N_FEATURES, dtype=np.float32)

        is_sniper = 1.0 if "sniper" in strategy_id.lower() else 0.0
        deadline = (self.sniper_max_hold_sec - hold_time_sec) if is_sniper else -1.0
        drawdown = max(0.0, max_pnl_multiple_seen - current_pnl_multiple)

        # Position state
        vec[_FEAT_IDX["hold_time_sec"]] = hold_time_sec
        vec[_FEAT_IDX["current_pnl_multiple"]] = current_pnl_multiple
        vec[_FEAT_IDX["max_pnl_multiple_seen"]] = max_pnl_multiple_seen
        vec[_FEAT_IDX["pnl_drawdown_from_peak"]] = drawdown
        vec[_FEAT_IDX["time_to_deadline_sec"]] = deadline

        # Current market features
        vec[_FEAT_IDX["wallet_cluster_30s"]] = _safe_float(features.wallet_cluster_30s)
        vec[_FEAT_IDX["volume_sol_30s"]] = _safe_float(features.volume_sol_30s)
        vec[_FEAT_IDX["buy_volume_sol_30s"]] = _safe_float(features.buy_volume_sol_30s)
        vec[_FEAT_IDX["tx_count_30s"]] = _safe_float(features.tx_count_30s)
        bsr = min(_safe_float(features.buy_sell_ratio_30s), 1000.0)
        vec[_FEAT_IDX["buy_sell_ratio_30s"]] = bsr
        vec[_FEAT_IDX["net_flow_sol_30s"]] = _safe_float(features.net_flow_sol_30s)
        vec[_FEAT_IDX["avg_trade_sol_30s"]] = _safe_float(features.avg_trade_sol_30s)
        vec[_FEAT_IDX["price_change_30s"]] = _safe_float(features.price_change_30s)

        # Trend comparison vs entry snapshot
        if entry_snapshot:
            e_vol = _safe_float(entry_snapshot.get("volume_sol_30s"), 1.0)
            e_cl = _safe_float(entry_snapshot.get("wallet_cluster_30s"), 1.0)
            e_bsr = min(_safe_float(entry_snapshot.get("buy_sell_ratio_30s"), 1.0), 1000.0)
            e_flow = _safe_float(entry_snapshot.get("net_flow_sol_30s"), 1.0)
            vec[_FEAT_IDX["volume_vs_entry"]] = _safe_ratio(
                _safe_float(features.volume_sol_30s), e_vol
            )
            vec[_FEAT_IDX["cluster_vs_entry"]] = _safe_ratio(
                _safe_float(features.wallet_cluster_30s), max(e_cl, 1.0)
            )
            vec[_FEAT_IDX["buy_ratio_vs_entry"]] = _safe_ratio(bsr, max(e_bsr, 0.01))
            vec[_FEAT_IDX["flow_vs_entry"]] = _safe_ratio(
                _safe_float(features.net_flow_sol_30s),
                e_flow if abs(e_flow) > 0.01 else 1.0,
            )
        else:
            vec[_FEAT_IDX["volume_vs_entry"]] = 1.0
            vec[_FEAT_IDX["cluster_vs_entry"]] = 1.0
            vec[_FEAT_IDX["buy_ratio_vs_entry"]] = 1.0
            vec[_FEAT_IDX["flow_vs_entry"]] = 1.0

        vec[_FEAT_IDX["strategy_is_sniper"]] = is_sniper
        vec[_FEAT_IDX["exit_stage"]] = float(exit_stage)

        return vec

    # ── Tick buffering ────────────────────────────────────────────────────────

    def record_tick_sample(
        self,
        position_id: int,
        features: RuntimeFeatures,
        hold_time_sec: float,
        current_pnl_multiple: float,
        max_pnl_multiple_seen: float,
        exit_stage: int,
        strategy_id: str,
        entry_snapshot: dict[str, float] | None,
    ) -> None:
        """Buffer one tick snapshot for a live position. Labeled later when position closes."""
        if self.mode == "off":
            return
        vec = self._build_vector(
            features=features,
            hold_time_sec=hold_time_sec,
            current_pnl_multiple=current_pnl_multiple,
            max_pnl_multiple_seen=max_pnl_multiple_seen,
            exit_stage=exit_stage,
            strategy_id=strategy_id,
            entry_snapshot=entry_snapshot,
        )
        self._pending_ticks[position_id].append(
            {
                "vec": vec,
                "strategy_id": strategy_id,
                "hold_time_sec": hold_time_sec,
                "tick_time": datetime.now(tz=timezone.utc).isoformat(),
            }
        )

    # ── Retrospective labeling on close ──────────────────────────────────────

    def record_closed_position(
        self,
        position_id: int,
        exit_reason: str,
        realized_pnl_sol: float,
        strategy_id: str,
    ) -> int:
        """
        Label and save all buffered tick samples for a closed position.
        Returns number of samples added.
        """
        if position_id in self._recorded_position_ids:
            return 0
        self._recorded_position_ids.add(position_id)

        ticks = list(self._pending_ticks.pop(position_id, []))
        if not ticks:
            return 0

        strategy_key = "sniper" if "sniper" in strategy_id.lower() else "main"

        new_samples = []
        if strategy_key == "sniper":
            # Sniper: position-level label (was it a win or loss overall?)
            label = 1 if exit_reason in _SNIPER_WIN_REASONS else 0
            for tick in ticks:
                self._training_rows[strategy_key].append((tick["vec"], label))
                new_samples.append(
                    {
                        "position_id": position_id,
                        "strategy_id": strategy_key,
                        "exit_reason": exit_reason,
                        "realized_pnl_sol": round(realized_pnl_sol, 6),
                        "label": label,
                        "hold_time_sec": tick["hold_time_sec"],
                        "tick_time": tick["tick_time"],
                        "features": {
                            name: round(float(tick["vec"][feat_i]), 6)
                            for feat_i, name in enumerate(FEATURE_NAMES)
                        },
                    }
                )
        else:
            # Main: tick-level peak proximity labeling.
            # label=1 (hold) when meaningful upside remains; label=0 (exit) when at/past peak.
            # This teaches the model WHEN to exit for maximum profit, not just WHETHER to win.
            tick_pnl_vals = [
                float(tick["vec"][_FEAT_IDX["current_pnl_multiple"]]) for tick in ticks
            ]
            n = len(tick_pnl_vals)
            for tick_i, tick in enumerate(ticks):
                current_pnl = tick_pnl_vals[tick_i]
                future_max = max(tick_pnl_vals[tick_i:]) if tick_i < n else current_pnl
                # Hold if: at least 1.5% more absolute PnL achievable AND future peak is a real win
                has_upside = future_max > current_pnl + 0.015 and future_max > 0.01
                tick_label = 1 if has_upside else 0
                self._training_rows[strategy_key].append((tick["vec"], tick_label))
                new_samples.append(
                    {
                        "position_id": position_id,
                        "strategy_id": strategy_key,
                        "exit_reason": exit_reason,
                        "realized_pnl_sol": round(realized_pnl_sol, 6),
                        "label": tick_label,
                        "hold_time_sec": tick["hold_time_sec"],
                        "tick_time": tick["tick_time"],
                        "features": {
                            name: round(float(tick["vec"][feat_i]), 6)
                            for feat_i, name in enumerate(FEATURE_NAMES)
                        },
                    }
                )

        self._append_jsonl(new_samples)

        self._closed_since_retrain[strategy_key] += 1
        if self._closed_since_retrain[strategy_key] >= self.retrain_every:
            self._retrain(strategy_key)

        logger.debug(
            "exit_predictor: labeled %d tick samples for position %d (strategy=%s exit=%s)",
            len(ticks),
            position_id,
            strategy_key,
            exit_reason,
        )
        return len(ticks)

    # ── Inference ─────────────────────────────────────────────────────────────

    def evaluate_position(
        self,
        position: dict[str, Any],
        features: RuntimeFeatures,
        mark_price_sol: float,
        strategy_id: str,
    ) -> ExitMLDecision:
        """Evaluate whether to exit early or hold. Called on every tick per open position."""
        strategy_key = "sniper" if "sniper" in strategy_id.lower() else "main"
        threshold = self.sniper_threshold if strategy_key == "sniper" else self.main_threshold
        model = self.models.get(strategy_key)
        model_ready = (
            model is not None and len(self._training_rows[strategy_key]) >= self.min_samples
        )

        if self.mode == "off":
            return ExitMLDecision(
                exit_now=False,
                hold_probability=0.5,
                mode="off",
                model_ready=False,
                reason="exit_ml_off",
                strategy_id=strategy_id,
            )

        if not model_ready:
            return ExitMLDecision(
                exit_now=False,
                hold_probability=0.5,
                mode=self.mode,
                model_ready=False,
                reason="exit_ml_not_ready",
                strategy_id=strategy_id,
            )

        # Build feature vector from current tick
        entry_price = float(position.get("entry_price_sol") or 0.0)
        hold_time_sec = 0.0
        try:
            entry_dt_str = position.get("entry_time") or ""
            if entry_dt_str:
                entry_dt = datetime.fromisoformat(str(entry_dt_str))
                hold_time_sec = (datetime.now(tz=timezone.utc) - entry_dt).total_seconds()
        except Exception:
            pass

        current_pnl = (mark_price_sol / entry_price - 1.0) if entry_price > 1e-12 else 0.0
        metadata: dict = {}
        try:
            metadata = (
                json.loads(position.get("metadata_json") or "{}")
                if isinstance(position.get("metadata_json"), str)
                else {}
            )
        except Exception:
            pass
        max_pnl = float(metadata.get("max_pnl_multiple_seen") or max(current_pnl, 0.0))
        exit_stage = int(position.get("exit_stage") or 0)
        entry_snapshot = metadata.get("runtime_features") or {}

        vec = self._build_vector(
            features=features,
            hold_time_sec=hold_time_sec,
            current_pnl_multiple=current_pnl,
            max_pnl_multiple_seen=max_pnl,
            exit_stage=exit_stage,
            strategy_id=strategy_id,
            entry_snapshot=entry_snapshot,
        )

        try:
            prob = float(model.predict_proba(vec.reshape(1, -1))[0][1])
        except Exception as exc:
            logger.debug("exit_predictor: predict failed: %s", exc)
            return ExitMLDecision(
                exit_now=False,
                hold_probability=0.5,
                mode=self.mode,
                model_ready=True,
                reason="exit_ml_predict_error",
                strategy_id=strategy_id,
            )

        if self.mode == "shadow":
            # exit_now reflects what the model *would* do — useful for calibration analysis.
            # It does NOT block exits in shadow mode (caller checks mode before acting).
            would_exit = prob < threshold
            return ExitMLDecision(
                exit_now=would_exit,
                hold_probability=prob,
                mode="shadow",
                model_ready=True,
                reason="exit_ml_shadow",
                strategy_id=strategy_id,
            )

        # gate mode
        if prob < threshold:
            return ExitMLDecision(
                exit_now=True,
                hold_probability=prob,
                mode="gate",
                model_ready=True,
                reason="exit_ml_early_exit",
                strategy_id=strategy_id,
            )
        return ExitMLDecision(
            exit_now=False,
            hold_probability=prob,
            mode="gate",
            model_ready=True,
            reason="exit_ml_hold",
            strategy_id=strategy_id,
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def _retrain(self, strategy_key: str) -> None:
        if self._classifier_cls is None:
            return
        rows = self._training_rows[strategy_key]
        if len(rows) < self.min_samples:
            return

        X = np.vstack([r[0] for r in rows]).astype(np.float32)
        y = np.asarray([r[1] for r in rows], dtype=np.int32)

        if len(np.unique(y)) < 2:
            logger.info("exit_predictor: skipping retrain (%s) — only one class", strategy_key)
            return

        pos = max(1, int((y == 1).sum()))
        neg = max(1, int((y == 0).sum()))
        sample_weight = np.where(y == 1, float(neg / pos), 1.0).astype(np.float32)

        model = self._classifier_cls(
            max_depth=4,
            learning_rate=0.05,
            max_iter=200,
            random_state=42,
            min_samples_leaf=15,
            l2_regularization=0.05,
            verbose=0,
        )
        model.fit(X, y, sample_weight=sample_weight)
        self.models[strategy_key] = model
        self._closed_since_retrain[strategy_key] = 0
        self._model_versions[strategy_key] = (
            f"sklearn_hgb:{datetime.now(tz=timezone.utc).isoformat()}"
        )
        self._save_model()

        win_rate = float(y.mean())
        logger.info(
            "exit_predictor: retrained (%s) samples=%d win_rate=%.1f%%",
            strategy_key,
            len(y),
            win_rate * 100,
        )
        if self.event_log is not None:
            self.event_log.log(
                "exit_predictor_retrained",
                {
                    "strategy_id": strategy_key,
                    "sample_count": len(y),
                    "win_rate": round(win_rate, 4),
                    "model_version": self._model_versions[strategy_key],
                },
            )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_model(self) -> None:
        payload = {
            "models": dict(self.models),
            "feature_names": FEATURE_NAMES,
            "model_versions": dict(self._model_versions),
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump(payload, f)
        except Exception as exc:
            logger.warning("exit_predictor: failed to save model: %s", exc)

    def _load_model(self) -> None:
        if not self.model_path.exists():
            return
        try:
            with open(self.model_path, "rb") as f:
                payload = pickle.load(f)
            if payload.get("feature_names") != FEATURE_NAMES:
                logger.info("exit_predictor: feature mismatch — discarding saved model")
                return
            self.models = payload.get("models", {"sniper": None, "main": None})
            self._model_versions = payload.get("model_versions", {"sniper": "", "main": ""})
            logger.info("exit_predictor: loaded model from %s", self.model_path)
        except Exception as exc:
            logger.warning("exit_predictor: failed to load model: %s", exc)

    def _load_live_samples(self) -> None:
        if not self.samples_path.exists():
            return
        loaded = 0
        try:
            with open(self.samples_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    strategy_key = (
                        "sniper" if "sniper" in str(rec.get("strategy_id", "")) else "main"
                    )
                    feat_dict = rec.get("features") or {}
                    if not feat_dict:
                        continue
                    vec = np.array(
                        [float(feat_dict.get(name, 0.0)) for name in FEATURE_NAMES],
                        dtype=np.float32,
                    )
                    label = int(rec.get("label", 0))
                    self._training_rows[strategy_key].append((vec, label))
                    pos_id = rec.get("position_id")
                    if pos_id:
                        self._recorded_position_ids.add(int(pos_id))
                    loaded += 1
        except Exception as exc:
            logger.warning("exit_predictor: error loading live samples: %s", exc)
            return
        logger.info(
            "exit_predictor: loaded %d samples from %s (sniper=%d main=%d)",
            loaded,
            self.samples_path,
            len(self._training_rows["sniper"]),
            len(self._training_rows["main"]),
        )

    def _bootstrap_from_db(self) -> None:
        """
        Bootstrap from existing closed positions in bot_state.db.
        Uses entry-time features only (no hold-time dynamics) with neutral comparison ratios.
        Only adds positions not already in training buffer.
        """
        db_path = self.samples_path.parent.parent / "live" / "bot_state.db"
        if not db_path.exists():
            return

        import sqlite3

        try:
            conn = sqlite3.connect(db_path)
            rows = conn.execute("""
                SELECT id, strategy_id, realized_pnl_sol, metadata_json
                FROM positions WHERE status='CLOSED'
            """).fetchall()
            conn.close()
        except Exception as exc:
            logger.warning("exit_predictor: bootstrap DB error: %s", exc)
            return

        added = 0
        for pos_id, strat_id, pnl, meta_json in rows:
            if int(pos_id) in self._recorded_position_ids:
                continue
            try:
                meta = json.loads(meta_json) if meta_json else {}
            except Exception:
                continue
            rf = meta.get("runtime_features") or {}
            if not rf:
                continue

            strategy_key = "sniper" if "sniper" in str(strat_id or "") else "main"
            exit_reason = meta.get("last_exit_reason", "")

            if strategy_key == "sniper":
                label = 1 if exit_reason in _SNIPER_WIN_REASONS else 0
            else:
                label = 1 if float(pnl or 0) > 0.005 else 0

            # Build minimal feature vector (hold_time=0, pnl=0, comparison=1)
            vec = np.zeros(N_FEATURES, dtype=np.float32)
            vec[_FEAT_IDX["hold_time_sec"]] = 0.0
            vec[_FEAT_IDX["current_pnl_multiple"]] = 0.0
            vec[_FEAT_IDX["max_pnl_multiple_seen"]] = 0.0
            vec[_FEAT_IDX["pnl_drawdown_from_peak"]] = 0.0
            vec[_FEAT_IDX["time_to_deadline_sec"]] = (
                self.sniper_max_hold_sec if strategy_key == "sniper" else -1.0
            )
            vec[_FEAT_IDX["wallet_cluster_30s"]] = _safe_float(rf.get("wallet_cluster_30s"))
            vec[_FEAT_IDX["volume_sol_30s"]] = _safe_float(rf.get("volume_sol_30s"))
            vec[_FEAT_IDX["buy_volume_sol_30s"]] = _safe_float(rf.get("buy_volume_sol_30s"))
            vec[_FEAT_IDX["tx_count_30s"]] = _safe_float(rf.get("tx_count_30s"))
            vec[_FEAT_IDX["buy_sell_ratio_30s"]] = min(
                _safe_float(rf.get("buy_sell_ratio_30s")), 1000.0
            )
            vec[_FEAT_IDX["net_flow_sol_30s"]] = _safe_float(rf.get("net_flow_sol_30s"))
            vec[_FEAT_IDX["avg_trade_sol_30s"]] = _safe_float(rf.get("avg_trade_sol_30s"))
            vec[_FEAT_IDX["price_change_30s"]] = _safe_float(rf.get("price_change_30s"))
            # Neutral comparison ratios for bootstrap (no history)
            vec[_FEAT_IDX["volume_vs_entry"]] = 1.0
            vec[_FEAT_IDX["cluster_vs_entry"]] = 1.0
            vec[_FEAT_IDX["buy_ratio_vs_entry"]] = 1.0
            vec[_FEAT_IDX["flow_vs_entry"]] = 1.0
            vec[_FEAT_IDX["strategy_is_sniper"]] = 1.0 if strategy_key == "sniper" else 0.0
            vec[_FEAT_IDX["exit_stage"]] = 0.0

            self._training_rows[strategy_key].append((vec, label))
            self._recorded_position_ids.add(int(pos_id))
            added += 1

        if added > 0:
            logger.info("exit_predictor: bootstrapped %d positions from DB", added)
            for strategy_key in ("sniper", "main"):
                if len(self._training_rows[strategy_key]) >= self.min_samples:
                    self._retrain(strategy_key)

    # ── JSONL helpers ─────────────────────────────────────────────────────────

    def _append_jsonl(self, records: list[dict]) -> None:
        if not records:
            return
        try:
            with open(self.samples_path, "a") as fh:
                for rec in records:
                    fh.write(json.dumps(rec) + "\n")
        except Exception as exc:
            logger.warning("exit_predictor: failed to append samples: %s", exc)

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "sniper_model_ready": self.models["sniper"] is not None,
            "main_model_ready": self.models["main"] is not None,
            "sniper_samples": len(self._training_rows["sniper"]),
            "main_samples": len(self._training_rows["main"]),
            "sniper_threshold": self.sniper_threshold,
            "main_threshold": self.main_threshold,
        }
