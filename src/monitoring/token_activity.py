"""In-memory rolling token activity state."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import TYPE_CHECKING, Any

from src.bot.models import CandidateEvent

if TYPE_CHECKING:
    from src.storage.bot_db import BotDB


class TokenActivityCache:
    """Maintain rolling token activity and tracked-wallet clusters."""

    def __init__(
        self,
        wallet_scores: dict[str, float] | None = None,
        price_outlier_min_samples: int = 8,
        price_outlier_median_window: int = 25,
        price_outlier_max_multiple: float = 15.0,
        price_outlier_confirm_signatures: int = 2,
        price_outlier_confirm_window_sec: int = 12,
        price_outlier_confirm_tolerance: float = 0.35,
        db: BotDB | None = None,
    ) -> None:
        self.events_by_token: dict[str, deque[CandidateEvent]] = defaultdict(deque)
        self.accepted_prices_by_token: dict[str, deque[tuple[datetime, float, str]]] = defaultdict(
            deque
        )
        self.pending_outlier_prices_by_token: dict[str, deque[tuple[datetime, float, str]]] = (
            defaultdict(deque)
        )
        self.first_seen: dict[str, datetime] = {}
        self.source_first_seen: dict[str, dict[str, datetime]] = defaultdict(dict)
        self.token_launcher: dict[str, str] = {}
        self._launcher_stats_cache: dict[str, dict[str, int]] = {}
        self.wallet_scores = wallet_scores or {}
        self.tracked_wallet_features_enabled = bool(self.wallet_scores)
        self.price_outlier_filtered_count: dict[str, int] = defaultdict(int)
        self.price_outlier_confirmed_count: dict[str, int] = defaultdict(int)
        self.price_outlier_min_samples = max(1, int(price_outlier_min_samples))
        self.price_outlier_median_window = max(3, int(price_outlier_median_window))
        self.price_outlier_max_multiple = max(1.01, float(price_outlier_max_multiple))
        self.price_outlier_confirm_signatures = max(2, int(price_outlier_confirm_signatures))
        self.price_outlier_confirm_window_sec = max(1, int(price_outlier_confirm_window_sec))
        self.price_outlier_confirm_tolerance = min(
            max(0.01, float(price_outlier_confirm_tolerance)), 1.0
        )
        self.db = db
        # Optional callback invoked once per accepted price tick. Used by the
        # FastAPI streaming service (OHLCV recorder + WS broadcaster). Signature:
        # tick_callback(mint: str, ts: datetime, price_sol: float, volume_sol: float)
        self.tick_callback: Any | None = None

    @staticmethod
    def _parse_time(value: str | None, fallback: datetime) -> datetime:
        if not value:
            return fallback
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return fallback
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    def ingest(self, event: CandidateEvent) -> None:
        """Add one token event into rolling state."""
        queue = self.events_by_token[event.token_mint]
        queue.append(event)
        prior_token_seen = event.token_mint in self.first_seen
        prior_sources = set(self.source_first_seen.get(event.token_mint, {}).keys())
        if self.db is not None and (
            event.token_mint not in self.first_seen
            or (
                event.source_program
                and event.source_program not in self.source_first_seen[event.token_mint]
            )
        ):
            first_seen_raw, source_first_seen_raw = self.db.record_token_observation(
                event.token_mint,
                event.block_time,
                event.source_program,
            )
            self.first_seen[event.token_mint] = self._parse_time(first_seen_raw, event.block_time)
            if event.source_program:
                self.source_first_seen[event.token_mint][event.source_program] = self._parse_time(
                    source_first_seen_raw,
                    event.block_time,
                )
        else:
            if event.token_mint not in self.first_seen:
                self.first_seen[event.token_mint] = event.block_time
            if (
                event.source_program
                and event.source_program not in self.source_first_seen[event.token_mint]
            ):
                self.source_first_seen[event.token_mint][event.source_program] = event.block_time

        # Launcher tracking (arxiv 2602.14860-inspired). "Launcher" = triggering
        # wallet of the first swap we observed for this token — a proxy for the
        # deployer since we don't parse create-tx directly. Idempotent on
        # token_mint. Graduation = first PUMP_AMM event for a token previously
        # seen on PUMP_FUN or RAYDIUM_LAUNCHLAB.
        if (
            not prior_token_seen
            and event.triggering_wallet
            and event.token_mint not in self.token_launcher
        ):
            self.token_launcher[event.token_mint] = event.triggering_wallet
            if self.db is not None:
                self.db.record_token_launcher(
                    event.token_mint,
                    event.triggering_wallet,
                    event.source_program,
                    event.block_time,
                )
                self._launcher_stats_cache.pop(event.triggering_wallet, None)
        if (
            event.source_program == "PUMP_AMM"
            and "PUMP_AMM" not in prior_sources
            and prior_token_seen
            and ("PUMP_FUN" in prior_sources or "RAYDIUM_LAUNCHLAB" in prior_sources)
            and self.db is not None
        ):
            self.db.record_token_graduation(event.token_mint, event.block_time)
            launcher = self.token_launcher.get(event.token_mint)
            if launcher:
                self._launcher_stats_cache.pop(launcher, None)

        price_sol = getattr(event, "reference_price_sol", None)
        if price_sol is None and event.token_amount and event.token_amount > 0:
            price_sol = float(event.sol_amount) / float(event.token_amount)
        if price_sol is not None:
            price_sol = float(price_sol)
            if price_sol > 0:
                accepted_deque = self.accepted_prices_by_token.get(event.token_mint)
                before_len = len(accepted_deque) if accepted_deque is not None else 0
                self._record_price(
                    token_mint=event.token_mint,
                    event_time=event.block_time,
                    signature=event.signature,
                    price_sol=price_sol,
                )
                # If the tick was accepted (or a pending outlier got promoted),
                # invoke the streaming tick callback once with the latest price.
                if self.tick_callback is not None:
                    after_deque = self.accepted_prices_by_token.get(event.token_mint)
                    if after_deque is not None and len(after_deque) > before_len:
                        latest_ts, latest_price, _sig = after_deque[-1]
                        try:
                            self.tick_callback(
                                event.token_mint,
                                latest_ts,
                                float(latest_price),
                                float(event.sol_amount or 0.0),
                            )
                        except Exception:  # noqa: BLE001
                            # Tick callback must never break the ingest path.
                            pass

        self._trim(event.token_mint, event.block_time)

    def source_age_sec(self, token_mint: str, source_program: str, now: datetime) -> float | None:
        """Return seconds since this token was first seen on a specific source program."""
        t = self.source_first_seen.get(token_mint, {}).get(source_program)
        if t is None:
            return None
        return (now - t).total_seconds()

    def launcher_stats_for(self, token_mint: str) -> dict[str, object] | None:
        """Return launcher proxy-wallet + launches/graduations for a token, or None.

        Cached in-memory with invalidation on record_token_launcher /
        record_token_graduation. Returns {launcher_wallet, launches, graduations}
        with zeros if the launcher is known but has no DB row yet.
        """
        launcher = self.token_launcher.get(token_mint)
        if not launcher:
            return None
        cached = self._launcher_stats_cache.get(launcher)
        if cached is not None:
            return {"launcher_wallet": launcher, **cached}
        if self.db is None:
            return {"launcher_wallet": launcher, "launches": 0, "graduations": 0}
        stats = self.db.get_launcher_stats(launcher)
        resolved = {
            "launches": int(stats["launches"]) if stats else 0,
            "graduations": int(stats["graduations"]) if stats else 0,
        }
        self._launcher_stats_cache[launcher] = resolved
        return {"launcher_wallet": launcher, **resolved}

    def _trim(self, token_mint: str, now: datetime) -> None:
        """Trim rolling token state to bounded windows."""
        queue = self.events_by_token[token_mint]
        cutoff = now - timedelta(minutes=10)
        while queue and queue[0].block_time < cutoff:
            queue.popleft()

        accepted_prices = self.accepted_prices_by_token[token_mint]
        while accepted_prices and accepted_prices[0][0] < cutoff:
            accepted_prices.popleft()

        pending_prices = self.pending_outlier_prices_by_token[token_mint]
        while pending_prices and pending_prices[0][0] < cutoff:
            pending_prices.popleft()

    def _record_price(
        self, token_mint: str, event_time: datetime, signature: str, price_sol: float
    ) -> None:
        """Record one implied swap price with median-based outlier confirmation."""
        accepted = self.accepted_prices_by_token[token_mint]
        pending = self.pending_outlier_prices_by_token[token_mint]

        if not accepted:
            accepted.append((event_time, price_sol, signature))
            return

        if len(accepted) < self.price_outlier_min_samples:
            baseline = [point[1] for point in accepted]
        else:
            baseline = [point[1] for point in list(accepted)[-self.price_outlier_median_window :]]
        if not baseline:
            accepted.append((event_time, price_sol, signature))
            return

        baseline_median = float(median(baseline))
        if baseline_median <= 0:
            accepted.append((event_time, price_sol, signature))
            return

        ratio = price_sol / baseline_median
        is_outlier = ratio > self.price_outlier_max_multiple or ratio < (
            1.0 / self.price_outlier_max_multiple
        )
        if not is_outlier:
            accepted.append((event_time, price_sol, signature))
            return

        pending.append((event_time, price_sol, signature))
        confirm_cutoff = event_time - timedelta(seconds=self.price_outlier_confirm_window_sec)
        while pending and pending[0][0] < confirm_cutoff:
            pending.popleft()

        low = price_sol * (1.0 - self.price_outlier_confirm_tolerance)
        high = price_sol * (1.0 + self.price_outlier_confirm_tolerance)
        candidates = [point for point in pending if low <= point[1] <= high]
        candidate_signatures = {point[2] for point in candidates}

        if len(candidate_signatures) < self.price_outlier_confirm_signatures:
            self.price_outlier_filtered_count[token_mint] += 1
            return

        promoted_signatures: set[str] = set()
        promoted_count = 0
        for point in candidates:
            if point[2] in promoted_signatures:
                continue
            accepted.append(point)
            promoted_signatures.add(point[2])
            promoted_count += 1

        if promoted_count > 0:
            self.price_outlier_confirmed_count[token_mint] += promoted_count
            self.pending_outlier_prices_by_token[token_mint] = deque(
                [point for point in pending if point[2] not in promoted_signatures]
            )

    def snapshot(self, token_mint: str, now: datetime) -> dict | None:
        """Return a rolling feature snapshot for one token at a given time."""
        if token_mint not in self.events_by_token:
            return None
        self._trim(token_mint, now)
        queue = list(self.events_by_token[token_mint])
        if not queue:
            return None
        accepted_prices = list(self.accepted_prices_by_token[token_mint])
        last_price_raw = accepted_prices[-1][1] if accepted_prices else None

        recent_prices = [point[1] for point in accepted_prices[-5:]]
        if len(recent_prices) >= 3:
            last_price_reliable = float(median(recent_prices))
        else:
            last_price_reliable = float(last_price_raw) if last_price_raw is not None else None

        def window(seconds: int) -> list[CandidateEvent]:
            cutoff = now - timedelta(seconds=seconds)
            return [event for event in queue if event.block_time >= cutoff]

        def price_window(seconds: int) -> list[tuple[datetime, float, str]]:
            cutoff = now - timedelta(seconds=seconds)
            return [point for point in accepted_prices if point[0] >= cutoff]

        win_30 = window(30)
        win_60 = window(60)
        win_90 = window(90)
        win_120 = window(120)
        win_300 = window(300)
        price_win_30 = price_window(30)
        price_win_60 = price_window(60)
        buy_30 = [event for event in win_30 if event.side == "BUY"]
        buy_60 = [event for event in win_60 if event.side == "BUY"]
        sell_30 = [event for event in win_30 if event.side == "SELL"]
        sell_60 = [event for event in win_60 if event.side == "SELL"]

        def tracked_wallets(events: list[CandidateEvent]) -> set[str]:
            wallets: set[str] = set()
            for event in events:
                for wallet in event.tracked_wallets:
                    if wallet in self.wallet_scores:
                        wallets.add(wallet)
            return wallets

        def price_change(points: list[tuple[datetime, float, str]]) -> float | None:
            if not points:
                return None
            first_price = float(points[0][1])
            last_point_price = float(points[-1][1])
            if first_price <= 0 or last_point_price <= 0:
                return None
            return (last_point_price / first_price) - 1

        tracked_30 = tracked_wallets(win_30)
        tracked_60 = tracked_wallets(win_60)
        tracked_120 = tracked_wallets(win_120)
        tracked_300 = tracked_wallets(win_300)
        tracked_wallet_buys_90s = sum(
            1
            for event in win_90
            if event.side == "BUY" and any(w in self.wallet_scores for w in event.tracked_wallets)
        )
        buy_volume_30 = float(sum(event.sol_amount for event in buy_30))
        buy_volume_60 = float(sum(event.sol_amount for event in buy_60))
        sell_volume_30 = float(sum(event.sol_amount for event in sell_30))
        sell_volume_60 = float(sum(event.sol_amount for event in sell_60))
        tx_count_30 = len(win_30)
        tx_count_60 = len(win_60)

        # Round-trip wallet detection (anti-wash). A wallet that appears on both
        # BUY and SELL sides within the same window is treated as flow-neutral —
        # wash-trade, bundler rotation, or self-hedge. Its SOL is excluded from
        # ``real_volume_sol_*`` so rules can threshold on genuine directional
        # flow. Ratio denominator is the BUY-side wallet set (0.0 = clean flow,
        # 1.0 = every buyer also sold in-window).
        def round_trip_stats(
            events: list[CandidateEvent],
            buy_events: list[CandidateEvent],
            sell_events: list[CandidateEvent],
            total_volume: float,
            buy_volume: float,
        ) -> tuple[int, float, float, float, float, float]:
            buyer_wallets = {e.triggering_wallet for e in buy_events if e.triggering_wallet}
            seller_wallets = {e.triggering_wallet for e in sell_events if e.triggering_wallet}
            rt_wallets = buyer_wallets & seller_wallets
            if not rt_wallets:
                return (0, 0.0, 0.0, 0.0, float(total_volume), float(buy_volume))
            rt_volume = float(
                sum(e.sol_amount for e in events if e.triggering_wallet in rt_wallets)
            )
            rt_buy_volume = float(
                sum(e.sol_amount for e in buy_events if e.triggering_wallet in rt_wallets)
            )
            ratio_denom = max(1, len(buyer_wallets))
            rt_ratio = float(len(rt_wallets)) / float(ratio_denom)
            real_volume = max(0.0, float(total_volume) - rt_volume)
            real_buy_volume = max(0.0, float(buy_volume) - rt_buy_volume)
            return (
                len(rt_wallets),
                rt_volume,
                rt_buy_volume,
                rt_ratio,
                real_volume,
                real_buy_volume,
            )

        total_volume_30 = buy_volume_30 + sell_volume_30
        total_volume_60 = buy_volume_60 + sell_volume_60
        (
            round_trip_count_30,
            round_trip_volume_30,
            round_trip_buy_volume_30,
            round_trip_ratio_30,
            real_volume_30,
            real_buy_volume_30,
        ) = round_trip_stats(win_30, buy_30, sell_30, total_volume_30, buy_volume_30)
        (
            round_trip_count_60,
            round_trip_volume_60,
            round_trip_buy_volume_60,
            round_trip_ratio_60,
            real_volume_60,
            real_buy_volume_60,
        ) = round_trip_stats(win_60, buy_60, sell_60, total_volume_60, buy_volume_60)

        # Tier A forward-collection features: Gini of trade sizes (concentration),
        # inter-arrival coefficient of variation (burstiness), and consecutive-buy
        # streaks (directional persistence). All rolled from the same events we
        # already have in-window, so this is zero extra RPC cost. Values land in
        # the snapshot and flow into ml_samples.jsonl via feature_runtime ->
        # live_filter.build_feature_map for offline ranking.
        def trade_flow_stats(
            events: list[CandidateEvent],
        ) -> tuple[float, float, int, int]:
            n = len(events)
            if n == 0:
                return (0.0, 0.0, 0, 0)
            sizes = [float(e.sol_amount) for e in events if e.sol_amount and e.sol_amount > 0]
            if len(sizes) >= 2:
                sizes.sort()
                total = sum(sizes)
                if total > 0:
                    weighted = sum((i + 1) * s for i, s in enumerate(sizes))
                    gini = (2.0 * weighted) / (len(sizes) * total) - (len(sizes) + 1) / len(sizes)
                    gini = max(0.0, min(1.0, float(gini)))
                else:
                    gini = 0.0
            else:
                gini = 0.0
            if n >= 3:
                gaps = [
                    max(
                        0.0,
                        (events[i].block_time - events[i - 1].block_time).total_seconds(),
                    )
                    for i in range(1, n)
                ]
                mean_gap = sum(gaps) / len(gaps)
                if mean_gap > 0:
                    var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
                    cv = (var**0.5) / mean_gap
                else:
                    cv = 0.0
            else:
                cv = 0.0
            max_streak = 0
            streak_count = 0
            cur = 0
            for e in events:
                if e.side == "BUY":
                    if cur == 0:
                        streak_count += 1
                    cur += 1
                    if cur > max_streak:
                        max_streak = cur
                else:
                    cur = 0
            return (float(gini), float(cv), int(max_streak), int(streak_count))

        (
            trade_size_gini_30,
            inter_arrival_cv_30,
            max_consecutive_buy_streak_30,
            buy_streak_count_30,
        ) = trade_flow_stats(win_30)
        (
            trade_size_gini_60,
            inter_arrival_cv_60,
            max_consecutive_buy_streak_60,
            buy_streak_count_60,
        ) = trade_flow_stats(win_60)
        # Use a large finite sentinel for pure-buy flow instead of float("inf"):
        # the JSON sanitizer in src/utils/io.py collapses inf → None, which made
        # downstream log-based diagnostics mis-read pure-buy windows as "no data".
        # 999.0 clears any realistic BSR threshold we would ever set and survives
        # JSON serialization unchanged.
        _PURE_BUY_RATIO_SENTINEL = 999.0
        buy_sell_ratio_30 = (
            buy_volume_30 / sell_volume_30
            if sell_volume_30 > 0
            else (_PURE_BUY_RATIO_SENTINEL if buy_volume_30 > 0 else None)
        )
        buy_sell_ratio_60 = (
            buy_volume_60 / sell_volume_60
            if sell_volume_60 > 0
            else (_PURE_BUY_RATIO_SENTINEL if buy_volume_60 > 0 else None)
        )
        avg_trade_sol_30 = (
            (buy_volume_30 + sell_volume_30) / tx_count_30 if tx_count_30 > 0 else 0.0
        )
        avg_trade_sol_60 = (
            (buy_volume_60 + sell_volume_60) / tx_count_60 if tx_count_60 > 0 else 0.0
        )

        # Velocity (arxiv 2602.14860): number of swaps to reach a cumulative
        # SOL threshold. Computed across the full 10-min buffer; None if the
        # threshold was never crossed. Lower value = faster accumulation =
        # stronger signal per the paper.
        velocity_thresholds = (1.0, 5.0, 10.0, 30.0)
        swaps_to_threshold: dict[float, int | None] = {t: None for t in velocity_thresholds}
        cumulative_volume = 0.0
        for idx, evt in enumerate(queue, start=1):
            cumulative_volume += float(evt.sol_amount or 0.0)
            for threshold in velocity_thresholds:
                if swaps_to_threshold[threshold] is None and cumulative_volume >= threshold:
                    swaps_to_threshold[threshold] = idx
            if all(v is not None for v in swaps_to_threshold.values()):
                break

        launcher_info = self.launcher_stats_for(token_mint) or {}
        launcher_launches = launcher_info.get("launches")
        launcher_graduations = launcher_info.get("graduations")
        launcher_graduation_ratio: float | None = None
        if isinstance(launcher_launches, int) and launcher_launches > 0:
            launcher_graduation_ratio = float(launcher_graduations or 0) / float(launcher_launches)

        return {
            "token_mint": token_mint,
            "last_price_sol": last_price_reliable,
            "last_price_sol_raw": last_price_raw,
            "last_price_sol_reliable": last_price_reliable,
            "volume_sol_30s": float(sum(event.sol_amount for event in win_30)),
            "volume_sol_60s": float(sum(event.sol_amount for event in win_60)),
            "tx_count_30s": tx_count_30,
            "tx_count_60s": tx_count_60,
            "buy_volume_sol_30s": buy_volume_30,
            "buy_volume_sol_60s": buy_volume_60,
            "sell_volume_sol_30s": sell_volume_30,
            "sell_volume_sol_60s": sell_volume_60,
            "buy_tx_count_30s": len(buy_30),
            "buy_tx_count_60s": len(buy_60),
            "sell_tx_count_30s": len(sell_30),
            "sell_tx_count_60s": len(sell_60),
            "buy_sell_ratio_30s": buy_sell_ratio_30,
            "buy_sell_ratio_60s": buy_sell_ratio_60,
            "net_flow_sol_30s": buy_volume_30 - sell_volume_30,
            "net_flow_sol_60s": buy_volume_60 - sell_volume_60,
            "avg_trade_sol_30s": float(avg_trade_sol_30),
            "avg_trade_sol_60s": float(avg_trade_sol_60),
            "round_trip_wallet_count_30s": int(round_trip_count_30),
            "round_trip_wallet_count_60s": int(round_trip_count_60),
            "round_trip_wallet_ratio_30s": float(round_trip_ratio_30),
            "round_trip_wallet_ratio_60s": float(round_trip_ratio_60),
            "round_trip_volume_sol_30s": float(round_trip_volume_30),
            "round_trip_volume_sol_60s": float(round_trip_volume_60),
            "real_volume_sol_30s": float(real_volume_30),
            "real_volume_sol_60s": float(real_volume_60),
            "real_buy_volume_sol_30s": float(real_buy_volume_30),
            "real_buy_volume_sol_60s": float(real_buy_volume_60),
            "trade_size_gini_30s": float(trade_size_gini_30),
            "trade_size_gini_60s": float(trade_size_gini_60),
            "inter_arrival_cv_30s": float(inter_arrival_cv_30),
            "inter_arrival_cv_60s": float(inter_arrival_cv_60),
            "max_consecutive_buy_streak_30s": int(max_consecutive_buy_streak_30),
            "max_consecutive_buy_streak_60s": int(max_consecutive_buy_streak_60),
            "buy_streak_count_30s": int(buy_streak_count_30),
            "buy_streak_count_60s": int(buy_streak_count_60),
            "wallet_cluster_30s": len({event.triggering_wallet for event in buy_30}),
            "wallet_cluster_120s": len(
                {event.triggering_wallet for event in win_120 if event.side == "BUY"}
            ),
            "tracked_wallet_cluster_30s": len(tracked_30),
            "tracked_wallet_cluster_120s": len(tracked_120),
            "tracked_wallet_cluster_300s": len(tracked_300),
            "tracked_wallet_buys_90s": int(tracked_wallet_buys_90s),
            "tracked_wallet_features_enabled": bool(self.tracked_wallet_features_enabled),
            "tracked_wallet_present_60s": bool(tracked_60),
            "tracked_wallet_count_60s": len(tracked_60),
            "tracked_wallet_score_sum_60s": float(
                sum(self.wallet_scores.get(wallet, 0.0) for wallet in tracked_60)
            ),
            "price_change_30s": price_change(price_win_30),
            "price_change_60s": price_change(price_win_60),
            "token_age_sec": (now - self.first_seen[token_mint]).total_seconds()
            if token_mint in self.first_seen
            else None,
            "aggregated_wallet_score": float(
                sum(self.wallet_scores.get(wallet, 1.0) for wallet in tracked_30)
            ),
            "price_points_30s": len(price_win_30),
            "price_points_60s": len(price_win_60),
            "price_points_total": len(accepted_prices),
            "price_outlier_filtered_10m": int(self.price_outlier_filtered_count[token_mint]),
            "price_outlier_confirmed_10m": int(self.price_outlier_confirmed_count[token_mint]),
            "pending_price_outlier_count": len(self.pending_outlier_prices_by_token[token_mint]),
            "swaps_to_1_sol": swaps_to_threshold[1.0],
            "swaps_to_5_sol": swaps_to_threshold[5.0],
            "swaps_to_10_sol": swaps_to_threshold[10.0],
            "swaps_to_30_sol": swaps_to_threshold[30.0],
            "launcher_wallet": launcher_info.get("launcher_wallet"),
            "launcher_launches": launcher_launches,
            "launcher_graduations": launcher_graduations,
            "launcher_graduation_ratio": launcher_graduation_ratio,
        }
