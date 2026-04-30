"""Runtime ML filter for entry qualification (shadow/gate/off)."""

from __future__ import annotations

import json
import logging
import pickle
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.bot.config import BotConfig
from src.bot.models import RuntimeFeatures, RuntimeRule
from src.monitoring.market_regime import RegimeState
from src.storage.event_log import EventLogger
from src.utils.io import append_jsonl, read_jsonl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MLDecision:
    """One ML scoring decision for a candidate entry."""

    allow_entry: bool
    probability: float
    threshold: float
    mode: str
    model_ready: bool
    reason: str
    feature_map: dict[str, float]


class LiveMLFilter:
    """Lightweight classifier wrapper with bootstrap + live retraining."""

    STRATEGIES = ("main", "sniper")

    FEATURE_NAMES = [
        "token_age_sec",
        "wallet_cluster_30s",
        "wallet_cluster_120s",
        "volume_sol_30s",
        "volume_sol_60s",
        "tx_count_30s",
        "tx_count_60s",
        "buy_volume_sol_30s",
        "buy_volume_sol_60s",
        "sell_volume_sol_30s",
        "sell_volume_sol_60s",
        "buy_sell_ratio_30s",
        "buy_sell_ratio_60s",
        "net_flow_sol_30s",
        "net_flow_sol_60s",
        "avg_trade_sol_30s",
        "avg_trade_sol_60s",
        "entry_price_sol",
        "price_change_30s",
        "price_change_60s",
        "tracked_wallet_present_60s",
        "tracked_wallet_count_60s",
        "tracked_wallet_score_sum_60s",
        "triggering_wallet_score",
        "aggregated_wallet_score",
        "candidate_score",
        "rule_support",
        "rule_hit_2x_rate",
        "rule_hit_5x_rate",
        "rule_rug_rate",
        "strategy_is_sniper",
        "lane_shock",
        "lane_recovery",
        "regime_negative_shock_recovery",
        "regime_high_cluster_recovery",
        "regime_momentum_burst",
        "regime_unknown",
        # Market regime context (added phase 2)
        # Teaches the model to be selective when broader market is cold/dead.
        "market_regime_score",  # 0.0 (dead) to 1.0 (hot), composite
        "recent_win_rate",  # rolling win rate of last 15 main positions (0.5 = neutral/bootstrap)
        "candidate_rate_5min_norm",  # candidates seen in last 5 min ÷ 200 (capped at 1.0)
    ]

    FEATURE_ALIASES = {
        "token_age_sec": ("token_age_sec", "token_age", "age_sec", "age"),
        "wallet_cluster_30s": ("wallet_cluster_30s", "cluster_30s", "wallet_cluster"),
        "wallet_cluster_120s": ("wallet_cluster_120s", "cluster_120s"),
        "volume_sol_30s": (
            "volume_sol_30s",
            "volume_30s",
            "early_buy_volume_sol",
            "buy_volume_sol",
        ),
        "volume_sol_60s": ("volume_sol_60s", "volume_60s"),
        "tx_count_30s": ("tx_count_30s", "tx_30s", "early_tx_count", "tx_count"),
        "tx_count_60s": ("tx_count_60s", "tx_60s"),
        "buy_volume_sol_30s": (
            "buy_volume_sol_30s",
            "buy_volume_30s",
            "buy_volume_sol",
        ),
        "buy_volume_sol_60s": ("buy_volume_sol_60s", "buy_volume_60s"),
        "sell_volume_sol_30s": ("sell_volume_sol_30s", "sell_volume_30s"),
        "sell_volume_sol_60s": ("sell_volume_sol_60s", "sell_volume_60s"),
        "buy_sell_ratio_30s": ("buy_sell_ratio_30s", "buy_sell_ratio"),
        "buy_sell_ratio_60s": ("buy_sell_ratio_60s",),
        "net_flow_sol_30s": ("net_flow_sol_30s", "net_flow_30s"),
        "net_flow_sol_60s": ("net_flow_sol_60s", "net_flow_60s"),
        "avg_trade_sol_30s": ("avg_trade_sol_30s", "avg_trade_30s"),
        "avg_trade_sol_60s": ("avg_trade_sol_60s", "avg_trade_60s"),
        "entry_price_sol": ("entry_price_sol", "price_sol", "launch_price_sol"),
        "price_change_30s": ("price_change_30s", "price_change"),
        "price_change_60s": ("price_change_60s",),
        "tracked_wallet_present_60s": ("tracked_wallet_present_60s", "tracked_present"),
        "tracked_wallet_count_60s": (
            "tracked_wallet_count_60s",
            "tracked_wallet_count",
        ),
        "tracked_wallet_score_sum_60s": (
            "tracked_wallet_score_sum_60s",
            "tracked_wallet_score_sum",
        ),
        "triggering_wallet_score": ("triggering_wallet_score", "wallet_score"),
        "aggregated_wallet_score": ("aggregated_wallet_score", "wallet_score_sum"),
        "candidate_score": ("candidate_score", "score"),
        "rule_support": ("rule_support", "support"),
        "rule_hit_2x_rate": ("rule_hit_2x_rate", "hit_2x_rate"),
        "rule_hit_5x_rate": ("rule_hit_5x_rate", "hit_5x_rate"),
        "rule_rug_rate": ("rule_rug_rate", "rug_rate"),
        "strategy_is_sniper": ("strategy_is_sniper",),
        "lane_shock": ("lane_shock",),
        "lane_recovery": ("lane_recovery",),
        "regime_negative_shock_recovery": ("regime_negative_shock_recovery",),
        "regime_high_cluster_recovery": ("regime_high_cluster_recovery",),
        "regime_momentum_burst": ("regime_momentum_burst",),
        "regime_unknown": ("regime_unknown",),
        "market_regime_score": ("market_regime_score",),
        "recent_win_rate": ("recent_win_rate",),
        "candidate_rate_5min_norm": ("candidate_rate_5min_norm",),
    }

    LABEL_COLUMNS = (
        "label",
        "target",
        "is_win",
        "win",
        "profitable",
        "has_graduated",
        "graduated",
        "y",
    )

    def __init__(self, config: BotConfig, event_log: EventLogger | None = None) -> None:
        self.config = config
        self.event_log = event_log
        self.mode = str(config.ml_mode or "shadow").strip().lower()
        if self.mode not in {"off", "shadow", "gate"}:
            self.mode = "shadow"
        self.backend = str(config.ml_model_backend or "sklearn_hgb").strip().lower()
        self.model_path = config.ml_model_path
        self.samples_path = config.ml_samples_path
        self.bootstrap_path = config.ml_bootstrap_path
        self.bootstrap_glob = str(config.ml_bootstrap_glob or "**/*.csv")
        self.bootstrap_enable = bool(config.ml_bootstrap_enable)
        self.bootstrap_max_rows = max(0, int(config.ml_bootstrap_max_rows))
        self.bootstrap_max_files = max(1, int(config.ml_bootstrap_max_files))
        self.max_training_samples = max(1000, int(config.ml_max_training_samples))
        self.min_samples_activate = max(50, int(config.ml_min_samples_activate))
        self.retrain_every = max(10, int(config.ml_retrain_every))
        self.positive_pnl_threshold_sol = float(config.ml_positive_pnl_threshold_sol)
        self.threshold_main = float(config.ml_threshold_main)
        self.threshold_sniper = float(config.ml_threshold_sniper)

        self.models: dict[str, Any | None] = {strategy: None for strategy in self.STRATEGIES}
        self._classifier_cls: Any | None = None
        self._import_error: str | None = None
        self._training_rows_by_strategy: dict[str, list[tuple[np.ndarray, int]]] = {
            strategy: [] for strategy in self.STRATEGIES
        }
        self._recorded_position_ids: set[int] = set()
        self._recorded_sample_keys: set[str] = set()
        self._trades_since_retrain_by_strategy: dict[str, int] = {
            strategy: 0 for strategy in self.STRATEGIES
        }
        self._last_probability_by_strategy: dict[str, float] = {
            strategy: 0.5 for strategy in self.STRATEGIES
        }
        self._model_version_by_strategy: dict[str, str] = {
            strategy: "untrained" for strategy in self.STRATEGIES
        }

        self._init_backend()
        self._load_model()
        self._load_live_samples()

        if self.bootstrap_enable and self.models["main"] is None:
            bootstrap_rows = self._load_bootstrap_samples()
            if bootstrap_rows > 0:
                logger.info(
                    "ML bootstrap loaded %s samples from %s",
                    bootstrap_rows,
                    self.bootstrap_path,
                )
        for strategy in self.STRATEGIES:
            if (
                self.models[strategy] is None
                and len(self._training_rows_by_strategy[strategy]) >= self.min_samples_activate
            ):
                self._retrain(strategy=strategy, reason="startup_bootstrap")

    @property
    def model_ready(self) -> bool:
        return self._is_strategy_ready("main")

    def status_fields(self) -> dict[str, object]:
        """Expose compact ML status for dashboard/bot status."""
        main_ready = self._is_strategy_ready("main")
        sniper_ready = self._is_strategy_ready("sniper")
        main_samples = len(self._training_rows_by_strategy["main"])
        sniper_samples = len(self._training_rows_by_strategy["sniper"])
        return {
            "ml_mode": self.mode,
            "ml_backend": self.backend,
            # Legacy fields (kept for dashboard compatibility) now reflect `main`.
            "ml_model_ready": main_ready,
            "ml_training_samples": main_samples,
            "ml_min_samples_activate": self.min_samples_activate,
            "ml_retrain_every": self.retrain_every,
            "ml_last_probability": float(self._last_probability_by_strategy["main"]),
            "ml_model_version": self._model_version_by_strategy["main"],
            # Strategy-specific fields.
            "ml_model_ready_main": main_ready,
            "ml_model_ready_sniper": sniper_ready,
            "ml_training_samples_main": main_samples,
            "ml_training_samples_sniper": sniper_samples,
            "ml_last_probability_main": float(self._last_probability_by_strategy["main"]),
            "ml_last_probability_sniper": float(self._last_probability_by_strategy["sniper"]),
            "ml_model_version_main": self._model_version_by_strategy["main"],
            "ml_model_version_sniper": self._model_version_by_strategy["sniper"],
            "ml_import_error": self._import_error,
        }

    def evaluate_candidate(
        self,
        *,
        features: RuntimeFeatures,
        rule: RuntimeRule | None,
        detected_regime: str,
        lane: str | None,
        candidate_score: float | None,
        strategy_id: str,
        rules_pass: bool,
        regime_state: RegimeState | None = None,
    ) -> MLDecision:
        """Score one candidate and return allow/reject decision."""
        strategy = self._strategy_key(strategy_id)
        feature_map = self._build_feature_map(
            features=features,
            rule=rule,
            detected_regime=detected_regime,
            lane=lane,
            candidate_score=candidate_score,
            strategy_id=strategy,
            regime_state=regime_state,
        )
        threshold = self.threshold_sniper if strategy == "sniper" else self.threshold_main
        model_ready = self._is_strategy_ready(strategy)

        if not rules_pass:
            decision = MLDecision(
                allow_entry=False,
                probability=0.0,
                threshold=threshold,
                mode=self.mode,
                model_ready=model_ready,
                reason="rules_failed",
                feature_map=feature_map,
            )
            return decision

        if self.mode == "off":
            decision = MLDecision(
                allow_entry=True,
                probability=0.5,
                threshold=threshold,
                mode=self.mode,
                model_ready=model_ready,
                reason="ml_off",
                feature_map=feature_map,
            )
            return decision

        if not model_ready:
            decision = MLDecision(
                allow_entry=True,
                probability=0.5,
                threshold=threshold,
                mode=self.mode,
                model_ready=False,
                reason="model_not_ready",
                feature_map=feature_map,
            )
            return decision

        vector = self._vector_from_feature_map(feature_map)
        probability = self._predict_probability(vector, strategy=strategy)
        self._last_probability_by_strategy[strategy] = probability

        allow_entry = True
        reason = "ml_shadow_allow"
        if self.mode == "gate":
            if probability >= threshold:
                allow_entry = True
                reason = "ml_gate_allow"
            else:
                allow_entry = False
                reason = "ml_gate_reject"

        if self.event_log is not None:
            self.event_log.log(
                "ml_scored",
                {
                    "strategy_id": strategy,
                    "mode": self.mode,
                    "model_ready": model_ready,
                    "probability": round(float(probability), 6),
                    "threshold": float(threshold),
                    "allow_entry": bool(allow_entry),
                    "reason": reason,
                    "token_mint": features.token_mint,
                    "rule_id": rule.rule_id if rule is not None else None,
                    "detected_regime": detected_regime,
                    "lane": lane,
                },
            )

        return MLDecision(
            allow_entry=allow_entry,
            probability=probability,
            threshold=threshold,
            mode=self.mode,
            model_ready=model_ready,
            reason=reason,
            feature_map=feature_map,
        )

    def record_closed_position(self, position: dict[str, Any]) -> bool:
        """Record one closed position as a training sample and retrain when due."""
        position_id = int(position.get("id") or 0)
        if position_id <= 0:
            return False

        metadata_raw = position.get("metadata_json") or "{}"
        try:
            metadata = (
                json.loads(metadata_raw) if isinstance(metadata_raw, str) else dict(metadata_raw)
            )
        except Exception:  # noqa: BLE001
            metadata = {}

        sample_key = self._position_sample_key(position=position, metadata=metadata)
        if sample_key in self._recorded_sample_keys:
            return False

        runtime_features = metadata.get("runtime_features")
        stored_feature_map = metadata.get("ml_feature_map")
        if isinstance(stored_feature_map, dict):
            feature_map = {
                name: self._f(stored_feature_map.get(name)) for name in self.FEATURE_NAMES
            }
        else:
            if not isinstance(runtime_features, dict):
                return False

            strategy_id = str(position.get("strategy_id") or metadata.get("strategy_id") or "main")
            feature_map = self._build_feature_map_from_snapshot(
                snapshot=runtime_features,
                strategy_id=strategy_id,
                rule_id=str(position.get("selected_rule_id") or ""),
                regime=str(position.get("selected_regime") or ""),
                candidate_score=metadata.get("candidate_score"),
                lane=metadata.get("entry_lane"),
            )
        strategy_id = str(position.get("strategy_id") or metadata.get("strategy_id") or "main")
        strategy = self._strategy_key(strategy_id)
        vector = self._vector_from_feature_map(feature_map)

        pnl_sol = float(position.get("realized_pnl_sol", 0.0) or 0.0)
        label = 1 if pnl_sol > self.positive_pnl_threshold_sol else 0
        self._append_training_row(vector, label, strategy=strategy)
        self._recorded_position_ids.add(position_id)
        self._recorded_sample_keys.add(sample_key)
        self._trades_since_retrain_by_strategy[strategy] += 1

        append_jsonl(
            self.samples_path,
            [
                {
                    "position_id": position_id,
                    "sample_key": sample_key,
                    "recorded_at": datetime.now(tz=timezone.utc).isoformat(),
                    "strategy_id": strategy,
                    "rule_id": str(position.get("selected_rule_id") or ""),
                    "regime": str(position.get("selected_regime") or ""),
                    "token_mint": str(position.get("token_mint") or ""),
                    "label": int(label),
                    "pnl_sol": float(pnl_sol),
                    "feature_map": feature_map,
                }
            ],
        )

        if (
            len(self._training_rows_by_strategy[strategy]) >= self.min_samples_activate
            and self._trades_since_retrain_by_strategy[strategy] >= self.retrain_every
        ):
            self._retrain(strategy=strategy, reason="live_incremental")
        return True

    def _init_backend(self) -> None:
        """Initialize optional ML backend dependencies."""
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._import_error = str(exc)
            self.mode = "off"
            logger.warning("ML disabled: missing backend dependency (%s)", exc)
            return

        self._classifier_cls = HistGradientBoostingClassifier
        self.backend = "sklearn_hgb"

    def _load_model(self) -> None:
        """Load persisted model if available."""
        if not self.model_path.exists():
            return
        try:
            with self.model_path.open("rb") as handle:
                payload = pickle.load(handle)
            feature_names = payload.get("feature_names")
            if list(feature_names or []) != list(self.FEATURE_NAMES):
                logger.warning("Ignoring incompatible ML model at %s", self.model_path)
                return

            # New format: per-strategy model dict.
            models_payload = payload.get("models")
            versions_payload = payload.get("model_versions") or {}
            loaded_any = False
            if isinstance(models_payload, dict):
                for strategy in self.STRATEGIES:
                    model = models_payload.get(strategy)
                    if model is not None:
                        self.models[strategy] = model
                        self._model_version_by_strategy[strategy] = str(
                            versions_payload.get(strategy) or "loaded"
                        )
                        loaded_any = True

            # Backward compatibility: old single-model payload => main model.
            if not loaded_any:
                legacy_model = payload.get("model")
                if legacy_model is not None:
                    self.models["main"] = legacy_model
                    self._model_version_by_strategy["main"] = str(
                        payload.get("model_version") or "loaded"
                    )
                    loaded_any = True

            if loaded_any:
                logger.info("Loaded ML model artifact from %s", self.model_path)
            else:
                logger.warning("No compatible model found in artifact %s", self.model_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load ML model from %s: %s", self.model_path, exc)

    def _save_model(self) -> None:
        """Persist model artifact."""
        if all(self.models[strategy] is None for strategy in self.STRATEGIES):
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        model_map = {
            strategy: model for strategy, model in self.models.items() if model is not None
        }
        payload = {
            "models": model_map,
            "feature_names": list(self.FEATURE_NAMES),
            "trained_at": datetime.now(tz=timezone.utc).isoformat(),
            "sample_count_by_strategy": {
                strategy: len(self._training_rows_by_strategy[strategy])
                for strategy in self.STRATEGIES
            },
            "model_versions": dict(self._model_version_by_strategy),
            "backend": self.backend,
        }
        with self.model_path.open("wb") as handle:
            pickle.dump(payload, handle)

    def _load_live_samples(self) -> int:
        """Load historical live samples from JSONL store."""
        rows = read_jsonl(self.samples_path)
        if not rows:
            return 0
        loaded = 0
        for row in rows:
            feature_map = row.get("feature_map")
            label = row.get("label")
            if not isinstance(feature_map, dict):
                continue
            try:
                label_int = int(label)
            except (TypeError, ValueError):
                continue
            if label_int not in {0, 1}:
                continue
            vector = self._vector_from_feature_map(feature_map)
            strategy = self._strategy_key(row.get("strategy_id"))
            self._append_training_row(vector, label_int, strategy=strategy)
            position_id = row.get("position_id")
            try:
                if position_id is not None:
                    self._recorded_position_ids.add(int(position_id))
            except (TypeError, ValueError):
                pass
            sample_key = row.get("sample_key")
            if isinstance(sample_key, str) and sample_key:
                self._recorded_sample_keys.add(sample_key)
            elif position_id is not None:
                # Backward compatibility for old samples without sample_key.
                self._recorded_sample_keys.add(f"legacy_id:{position_id}")
            loaded += 1
        if loaded:
            logger.info("Loaded %s ML live samples from %s", loaded, self.samples_path)
        return loaded

    def _position_sample_key(self, *, position: dict[str, Any], metadata: dict[str, Any]) -> str:
        """Build a stable fingerprint for one closed position sample.

        This avoids collisions after DB reset where SQLite autoincrement IDs
        start again from 1.
        """
        position_id = int(position.get("id") or 0)
        entry_time = str(position.get("entry_time") or "")
        token_mint = str(position.get("token_mint") or "")
        strategy_id = str(position.get("strategy_id") or metadata.get("strategy_id") or "main")
        rule_id = str(position.get("selected_rule_id") or "")
        regime = str(position.get("selected_regime") or "")
        entry_price_sol = self._f(position.get("entry_price_sol"))
        size_sol = self._f(position.get("size_sol"))
        amount_received = self._f(position.get("amount_received"))
        triggering_wallet = str(
            position.get("triggering_wallet") or metadata.get("triggering_wallet") or ""
        )
        last_exit_at = str(metadata.get("last_exit_at") or "")

        basis = {
            "token_mint": token_mint,
            "strategy_id": strategy_id,
            "rule_id": rule_id,
            "regime": regime,
            "entry_time": entry_time,
            "entry_price_sol": round(entry_price_sol, 12),
            "size_sol": round(size_sol, 12),
            "amount_received": round(amount_received, 6),
            "triggering_wallet": triggering_wallet,
            "last_exit_at": last_exit_at,
        }

        # If entry_time is missing from the fetched row, keep legacy-id behavior.
        if not entry_time:
            return f"legacy_id:{position_id}"

        digest = hashlib.sha1(
            json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"sample:{digest}"

    def _load_bootstrap_samples(self) -> int:
        """Load optional bootstrap samples from CSV files."""
        if self._classifier_cls is None:
            return 0
        if not self.bootstrap_path.exists():
            logger.info("ML bootstrap path not found: %s", self.bootstrap_path)
            return 0
        files = sorted(self.bootstrap_path.glob(self.bootstrap_glob))
        if not files:
            logger.info(
                "ML bootstrap found no files at %s/%s",
                self.bootstrap_path,
                self.bootstrap_glob,
            )
            return 0

        loaded_rows = 0
        for csv_path in files[: self.bootstrap_max_files]:
            if loaded_rows >= self.bootstrap_max_rows:
                break
            if csv_path.suffix.lower() != ".csv":
                continue
            try:
                chunk_iter = pd.read_csv(csv_path, chunksize=25_000, low_memory=False)
            except Exception:  # noqa: BLE001
                continue
            for chunk in chunk_iter:
                if loaded_rows >= self.bootstrap_max_rows:
                    break
                parsed = self._extract_bootstrap_chunk(chunk)
                if parsed is None:
                    continue
                X_chunk, y_chunk, strategy_chunk = parsed
                take = min(len(y_chunk), self.bootstrap_max_rows - loaded_rows)
                if take <= 0:
                    break
                for idx in range(take):
                    strategy = self._strategy_key(strategy_chunk[idx])
                    self._append_training_row(X_chunk[idx], int(y_chunk[idx]), strategy=strategy)
                loaded_rows += take
        return loaded_rows

    def _extract_bootstrap_chunk(
        self, chunk: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Extract aligned feature matrix + labels from one bootstrap chunk."""
        label_col = self._find_first_column(chunk.columns, self.LABEL_COLUMNS)
        if label_col is None:
            return None

        label_series = chunk[label_col].apply(self._normalize_label)
        valid_mask = label_series.notna()
        if not bool(valid_mask.any()):
            return None

        frame = chunk.loc[valid_mask].copy()
        labels = label_series.loc[valid_mask].astype(int).to_numpy(dtype=np.int32)
        if len(labels) == 0:
            return None
        strategy_col = self._find_first_column(frame.columns, ("strategy_id", "strategy"))
        if strategy_col is None:
            strategies = np.asarray(["main"] * len(frame), dtype=object)
        else:
            strategies = frame[strategy_col].astype(str).to_numpy(dtype=object)

        vectors = np.zeros((len(frame), len(self.FEATURE_NAMES)), dtype=np.float32)
        mapped_columns = 0
        for col_idx, feature_name in enumerate(self.FEATURE_NAMES):
            column_name = self._find_first_column(
                frame.columns, self.FEATURE_ALIASES.get(feature_name, (feature_name,))
            )
            if column_name is None:
                continue
            series = pd.to_numeric(frame[column_name], errors="coerce").fillna(0.0).astype(float)
            vectors[:, col_idx] = np.clip(series.to_numpy(dtype=np.float32), -1e9, 1e9)
            mapped_columns += 1

        if mapped_columns < 3:
            return None
        return vectors, labels, strategies

    def _append_training_row(self, vector: np.ndarray, label: int, *, strategy: str) -> None:
        """Append one training row with bounded history."""
        target = self._strategy_key(strategy)
        bucket = self._training_rows_by_strategy[target]
        bucket.append((vector.astype(np.float32, copy=False), int(label)))
        overflow = len(bucket) - self.max_training_samples
        if overflow > 0:
            del bucket[:overflow]

    def _retrain(self, *, strategy: str, reason: str) -> None:
        """Train/retrain the model on current buffered samples."""
        strategy = self._strategy_key(strategy)
        if self._classifier_cls is None:
            return
        training_rows = self._training_rows_by_strategy[strategy]
        if len(training_rows) < self.min_samples_activate:
            return

        X = np.vstack([row[0] for row in training_rows]).astype(np.float32, copy=False)
        y = np.asarray([row[1] for row in training_rows], dtype=np.int32)
        if len(np.unique(y)) < 2:
            logger.info(
                "Skipping ML retrain (%s): only one class present in %s buffer.",
                strategy,
                strategy,
            )
            return

        positive = max(1, int((y == 1).sum()))
        negative = max(1, int((y == 0).sum()))
        positive_weight = float(negative / positive)
        sample_weight = np.where(y == 1, positive_weight, 1.0).astype(np.float32)

        model = self._classifier_cls(
            max_depth=5,
            learning_rate=0.05,
            max_iter=220,
            random_state=42,
            min_samples_leaf=20,
            l2_regularization=0.05,
            verbose=0,
        )
        model.fit(X, y, sample_weight=sample_weight)
        self.models[strategy] = model
        self._trades_since_retrain_by_strategy[strategy] = 0
        self._model_version_by_strategy[strategy] = (
            f"{self.backend}:{datetime.now(tz=timezone.utc).isoformat()}"
        )
        self._save_model()

        win_rate = float(y.mean())
        logger.info(
            "ML retrained (%s:%s) samples=%s win_rate=%.3f backend=%s",
            strategy,
            reason,
            len(y),
            win_rate,
            self.backend,
        )
        if self.event_log is not None:
            self.event_log.log(
                "ml_retrained",
                {
                    "strategy_id": strategy,
                    "reason": reason,
                    "sample_count": int(len(y)),
                    "win_rate": round(win_rate, 6),
                    "backend": self.backend,
                    "model_version": self._model_version_by_strategy[strategy],
                },
            )

    def _predict_probability(self, vector: np.ndarray, *, strategy: str) -> float:
        """Predict win probability for one feature vector."""
        strategy = self._strategy_key(strategy)
        model = self.models.get(strategy)
        if model is None:
            return 0.5
        try:
            proba = float(model.predict_proba(vector.reshape(1, -1))[0][1])
        except Exception:  # noqa: BLE001
            return 0.5
        if not np.isfinite(proba):
            return 0.5
        return float(min(max(proba, 0.0), 1.0))

    def _build_feature_map(
        self,
        *,
        features: RuntimeFeatures,
        rule: RuntimeRule | None,
        detected_regime: str,
        lane: str | None,
        candidate_score: float | None,
        strategy_id: str,
        regime_state: RegimeState | None = None,
    ) -> dict[str, float]:
        # Market regime features — 0.5 (neutral) when state not available
        mkt_score = self._f(regime_state.score if regime_state is not None else 0.5)
        mkt_win_rate = self._f(
            regime_state.win_rate
            if (regime_state is not None and regime_state.win_rate is not None)
            else 0.5
        )
        mkt_cand_norm = self._f(
            min(regime_state.candidate_rate_5min, 200) / 200.0 if regime_state is not None else 0.5
        )
        return {
            "token_age_sec": self._f(features.token_age_sec),
            "wallet_cluster_30s": self._f(features.wallet_cluster_30s),
            "wallet_cluster_120s": self._f(features.wallet_cluster_120s),
            "volume_sol_30s": self._f(features.volume_sol_30s),
            "volume_sol_60s": self._f(features.volume_sol_60s),
            "tx_count_30s": self._f(features.tx_count_30s),
            "tx_count_60s": self._f(features.tx_count_60s),
            "buy_volume_sol_30s": self._f(features.buy_volume_sol_30s),
            "buy_volume_sol_60s": self._f(features.buy_volume_sol_60s),
            "sell_volume_sol_30s": self._f(features.sell_volume_sol_30s),
            "sell_volume_sol_60s": self._f(features.sell_volume_sol_60s),
            "buy_sell_ratio_30s": self._f(self._finite_ratio(features.buy_sell_ratio_30s)),
            "buy_sell_ratio_60s": self._f(self._finite_ratio(features.buy_sell_ratio_60s)),
            "net_flow_sol_30s": self._f(features.net_flow_sol_30s),
            "net_flow_sol_60s": self._f(features.net_flow_sol_60s),
            "avg_trade_sol_30s": self._f(features.avg_trade_sol_30s),
            "avg_trade_sol_60s": self._f(features.avg_trade_sol_60s),
            "entry_price_sol": self._f(features.entry_price_sol),
            "price_change_30s": self._f(features.price_change_30s),
            "price_change_60s": self._f(features.price_change_60s),
            "tracked_wallet_present_60s": 1.0 if features.tracked_wallet_present_60s else 0.0,
            "tracked_wallet_count_60s": self._f(features.tracked_wallet_count_60s),
            "tracked_wallet_score_sum_60s": self._f(features.tracked_wallet_score_sum_60s),
            "triggering_wallet_score": self._f(features.triggering_wallet_score),
            "aggregated_wallet_score": self._f(features.aggregated_wallet_score),
            "candidate_score": self._f(candidate_score),
            "rule_support": self._f(rule.support if rule is not None else 0),
            "rule_hit_2x_rate": self._f(rule.hit_2x_rate if rule is not None else 0.0),
            "rule_hit_5x_rate": self._f(rule.hit_5x_rate if rule is not None else 0.0),
            "rule_rug_rate": self._f(rule.rug_rate if rule is not None else 0.0),
            "strategy_is_sniper": 1.0 if strategy_id == "sniper" else 0.0,
            "lane_shock": 1.0 if lane == "shock" else 0.0,
            "lane_recovery": 1.0 if lane == "recovery" else 0.0,
            "regime_negative_shock_recovery": 1.0
            if detected_regime == "negative_shock_recovery"
            else 0.0,
            "regime_high_cluster_recovery": 1.0
            if detected_regime == "high_cluster_recovery"
            else 0.0,
            "regime_momentum_burst": 1.0 if detected_regime == "momentum_burst" else 0.0,
            "regime_unknown": 1.0 if detected_regime in {"unknown", "pending"} else 0.0,
            "market_regime_score": mkt_score,
            "recent_win_rate": mkt_win_rate,
            "candidate_rate_5min_norm": mkt_cand_norm,
            # Tier A forward-collection features — not in FEATURE_NAMES yet, so
            # the current trained model ignores them. They're persisted into
            # feature_map rows in ml_samples.jsonl for offline ranking once
            # enough post-deploy samples accumulate.
            "trade_size_gini_30s": self._f(features.raw.get("trade_size_gini_30s")),
            "trade_size_gini_60s": self._f(features.raw.get("trade_size_gini_60s")),
            "inter_arrival_cv_30s": self._f(features.raw.get("inter_arrival_cv_30s")),
            "inter_arrival_cv_60s": self._f(features.raw.get("inter_arrival_cv_60s")),
            "max_consecutive_buy_streak_30s": self._f(
                features.raw.get("max_consecutive_buy_streak_30s")
            ),
            "max_consecutive_buy_streak_60s": self._f(
                features.raw.get("max_consecutive_buy_streak_60s")
            ),
            "buy_streak_count_30s": self._f(features.raw.get("buy_streak_count_30s")),
            "buy_streak_count_60s": self._f(features.raw.get("buy_streak_count_60s")),
            "round_trip_wallet_ratio_30s": self._f(features.raw.get("round_trip_wallet_ratio_30s")),
            "real_volume_sol_30s": self._f(features.raw.get("real_volume_sol_30s")),
            "real_buy_volume_sol_30s": self._f(features.raw.get("real_buy_volume_sol_30s")),
            # Sniper-rule features — needed for live v2 rule mining (kaggle_sniper_v1
            # thresholds gate on these). Persisted into ml_samples.jsonl for offline
            # miners; current trained model ignores unknown columns.
            "sell_tx_count_30s": self._f(getattr(features, "sell_tx_count_30s", None)),
            "round_trip_wallet_count_30s": self._f(
                getattr(features, "round_trip_wallet_count_30s", None)
            ),
            "round_trip_volume_sol_30s": self._f(
                getattr(features, "round_trip_volume_sol_30s", None)
            ),
            # Velocity + launcher history (arxiv 2602.14860). Persisted for future
            # mining; None when threshold not reached / launcher unknown.
            "swaps_to_1_sol": self._f(features.raw.get("swaps_to_1_sol")),
            "swaps_to_5_sol": self._f(features.raw.get("swaps_to_5_sol")),
            "swaps_to_10_sol": self._f(features.raw.get("swaps_to_10_sol")),
            "swaps_to_30_sol": self._f(features.raw.get("swaps_to_30_sol")),
            "launcher_launches": self._f(features.raw.get("launcher_launches")),
            "launcher_graduations": self._f(features.raw.get("launcher_graduations")),
            "launcher_graduation_ratio": self._f(features.raw.get("launcher_graduation_ratio")),
        }

    def _build_feature_map_from_snapshot(
        self,
        *,
        snapshot: dict[str, Any],
        strategy_id: str,
        rule_id: str,
        regime: str,
        candidate_score: Any,
        lane: Any,
    ) -> dict[str, float]:
        """Build feature map from persisted entry snapshot."""
        return {
            "token_age_sec": self._f(snapshot.get("token_age_sec")),
            "wallet_cluster_30s": self._f(snapshot.get("wallet_cluster_30s")),
            "wallet_cluster_120s": self._f(snapshot.get("wallet_cluster_120s")),
            "volume_sol_30s": self._f(snapshot.get("volume_sol_30s")),
            "volume_sol_60s": self._f(snapshot.get("volume_sol_60s")),
            "tx_count_30s": self._f(snapshot.get("tx_count_30s")),
            "tx_count_60s": self._f(snapshot.get("tx_count_60s")),
            "buy_volume_sol_30s": self._f(snapshot.get("buy_volume_sol_30s")),
            "buy_volume_sol_60s": self._f(snapshot.get("buy_volume_sol_60s")),
            "sell_volume_sol_30s": self._f(snapshot.get("sell_volume_sol_30s")),
            "sell_volume_sol_60s": self._f(snapshot.get("sell_volume_sol_60s")),
            "buy_sell_ratio_30s": self._f(self._finite_ratio(snapshot.get("buy_sell_ratio_30s"))),
            "buy_sell_ratio_60s": self._f(self._finite_ratio(snapshot.get("buy_sell_ratio_60s"))),
            "net_flow_sol_30s": self._f(snapshot.get("net_flow_sol_30s")),
            "net_flow_sol_60s": self._f(snapshot.get("net_flow_sol_60s")),
            "avg_trade_sol_30s": self._f(snapshot.get("avg_trade_sol_30s")),
            "avg_trade_sol_60s": self._f(snapshot.get("avg_trade_sol_60s")),
            "entry_price_sol": self._f(snapshot.get("entry_price_sol")),
            "price_change_30s": self._f(snapshot.get("price_change_30s")),
            "price_change_60s": self._f(snapshot.get("price_change_60s")),
            "tracked_wallet_present_60s": 1.0
            if bool(snapshot.get("tracked_wallet_present_60s", False))
            else 0.0,
            "tracked_wallet_count_60s": self._f(snapshot.get("tracked_wallet_count_60s")),
            "tracked_wallet_score_sum_60s": self._f(snapshot.get("tracked_wallet_score_sum_60s")),
            "triggering_wallet_score": self._f(snapshot.get("triggering_wallet_score")),
            "aggregated_wallet_score": self._f(snapshot.get("aggregated_wallet_score")),
            "candidate_score": self._f(candidate_score),
            "rule_support": self._f(0.0),
            "rule_hit_2x_rate": self._f(0.0),
            "rule_hit_5x_rate": self._f(0.0),
            "rule_rug_rate": self._f(0.0),
            "strategy_is_sniper": 1.0 if strategy_id == "sniper" else 0.0,
            "lane_shock": 1.0 if str(lane) == "shock" else 0.0,
            "lane_recovery": 1.0 if str(lane) == "recovery" else 0.0,
            "regime_negative_shock_recovery": 1.0 if regime == "negative_shock_recovery" else 0.0,
            "regime_high_cluster_recovery": 1.0 if regime == "high_cluster_recovery" else 0.0,
            "regime_momentum_burst": 1.0 if regime == "momentum_burst" else 0.0,
            "regime_unknown": 1.0 if regime in {"unknown", "pending", ""} else 0.0,
            # Market regime features — neutral for historical samples (not captured at entry time)
            "market_regime_score": 0.5,
            "recent_win_rate": 0.5,
            "candidate_rate_5min_norm": 0.5,
            # Tier A forward-collection features (present on snapshots taken after
            # the token_activity changes land; zero-fill for pre-existing snapshots).
            "trade_size_gini_30s": self._f(snapshot.get("trade_size_gini_30s")),
            "trade_size_gini_60s": self._f(snapshot.get("trade_size_gini_60s")),
            "inter_arrival_cv_30s": self._f(snapshot.get("inter_arrival_cv_30s")),
            "inter_arrival_cv_60s": self._f(snapshot.get("inter_arrival_cv_60s")),
            "max_consecutive_buy_streak_30s": self._f(
                snapshot.get("max_consecutive_buy_streak_30s")
            ),
            "max_consecutive_buy_streak_60s": self._f(
                snapshot.get("max_consecutive_buy_streak_60s")
            ),
            "buy_streak_count_30s": self._f(snapshot.get("buy_streak_count_30s")),
            "buy_streak_count_60s": self._f(snapshot.get("buy_streak_count_60s")),
            "round_trip_wallet_ratio_30s": self._f(snapshot.get("round_trip_wallet_ratio_30s")),
            "real_volume_sol_30s": self._f(snapshot.get("real_volume_sol_30s")),
            "real_buy_volume_sol_30s": self._f(snapshot.get("real_buy_volume_sol_30s")),
            "swaps_to_1_sol": self._f(snapshot.get("swaps_to_1_sol")),
            "swaps_to_5_sol": self._f(snapshot.get("swaps_to_5_sol")),
            "swaps_to_10_sol": self._f(snapshot.get("swaps_to_10_sol")),
            "swaps_to_30_sol": self._f(snapshot.get("swaps_to_30_sol")),
            "launcher_launches": self._f(snapshot.get("launcher_launches")),
            "launcher_graduations": self._f(snapshot.get("launcher_graduations")),
            "launcher_graduation_ratio": self._f(snapshot.get("launcher_graduation_ratio")),
        }

    def _vector_from_feature_map(self, feature_map: dict[str, Any]) -> np.ndarray:
        values = [self._f(feature_map.get(name)) for name in self.FEATURE_NAMES]
        vector = np.asarray(values, dtype=np.float32)
        vector = np.nan_to_num(vector, nan=0.0, posinf=1e6, neginf=-1e6)
        np.clip(vector, -1e9, 1e9, out=vector)
        return vector

    def _strategy_key(self, strategy_id: Any) -> str:
        candidate = str(strategy_id or "main").strip().lower()
        return "sniper" if candidate == "sniper" else "main"

    def _is_strategy_ready(self, strategy: str) -> bool:
        key = self._strategy_key(strategy)
        # Inference readiness only depends on having a trained model artifact.
        # Training-sample thresholds are enforced at retrain time.
        return self.models.get(key) is not None

    @staticmethod
    def _normalize_label(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, np.integer)):
            if int(value) in {0, 1}:
                return float(int(value))
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return None
            if value in {0.0, 1.0}:
                return float(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "win", "positive"}:
            return 1.0
        if text in {"0", "false", "no", "n", "loss", "negative"}:
            return 0.0
        return None

    @staticmethod
    def _find_first_column(columns: Any, candidates: tuple[str, ...]) -> str | None:
        lowered = {str(column).strip().lower(): str(column) for column in columns}
        for candidate in candidates:
            match = lowered.get(candidate.strip().lower())
            if match:
                return match
        return None

    @staticmethod
    def _finite_ratio(value: Any) -> float:
        numeric = LiveMLFilter._f(value)
        if numeric > 1000.0:
            return 1000.0
        if numeric < -1000.0:
            return -1000.0
        return numeric

    @staticmethod
    def _f(value: Any) -> float:
        try:
            numeric = float(pd.to_numeric(value, errors="coerce"))
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(numeric):
            return 0.0
        return float(numeric)
