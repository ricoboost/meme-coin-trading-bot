"""Configuration loading for the Phase 2 paper bot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.utils.io import AppConfig, load_app_config


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _str_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().strip('"').strip("'")


def _csv_env(name: str, default: str = "") -> tuple[str, ...]:
    """Read a comma-separated env var into a tuple of non-empty strings."""
    raw = _str_env(name, default)
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _rule_stop_overrides_env(name: str, default: str = "") -> dict[str, float]:
    """Read rule-specific stop overrides from `rule_id:value` CSV env."""
    raw = _str_env(name, default)
    if not raw:
        return {}
    overrides: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        rule_id, value = item.split(":", 1)
        rule_id = rule_id.strip()
        if not rule_id:
            continue
        try:
            overrides[rule_id] = float(value.strip())
        except ValueError:
            continue
    return overrides


@dataclass(frozen=True)
class BotConfig:
    """Resolved runtime bot configuration."""

    app: AppConfig
    helius_api_key: str
    helius_base_url: str
    helius_rpc_url: str
    chainstack_grpc_endpoint: str
    chainstack_grpc_token: str
    chainstack_reconnect_max_retries: int
    chainstack_reconnect_backoff_initial_sec: int
    chainstack_reconnect_backoff_max_sec: int
    jupiter_base_url: str
    jupiter_api_key: str
    bot_private_key_b58: str
    bot_public_key: str
    solana_cluster: str
    max_position_sol: float
    max_total_exposure_sol: float
    max_daily_loss_sol: float
    max_open_positions: int
    default_slippage_bps: int
    priority_fee_lamports: int
    jito_tip_lamports: int
    jito_tip_accounts: tuple[str, ...]
    live_broadcast_mode: str
    helius_sender_url: str
    helius_bundle_url: str
    live_broadcast_fee_type: str
    live_use_dynamic_priority_fee: bool
    live_use_dynamic_jito_tip: bool
    live_use_jupiter_auto_slippage: bool
    live_buy_slippage_bps: int
    live_sell_slippage_bps: int
    live_buy_slippage_bps_sniper_pump_fun: int
    live_sell_slippage_bps_sniper_pump_fun: int
    live_rebroadcast_interval_ms: int
    live_confirm_poll_interval_ms: int
    live_max_rebroadcast_attempts: int
    live_sender_idle_ping_sec: int
    live_sender_active_warm: bool
    live_sender_warm_interval_sec: float
    live_min_wallet_buffer_lamports: int
    live_buy_ata_rent_buffer_lamports: int
    live_token_account_rent_lamports: int
    live_min_net_exit_lamports: int
    live_close_token_ata_on_full_exit: bool
    live_use_shared_accounts: bool
    live_enable_native_pump_amm: bool
    live_pump_amm_buy_compute_unit_limit: int
    live_pump_amm_sell_compute_unit_limit: int
    live_sell_max_attempts: int
    live_preflight_simulate: bool
    live_priority_fee_lamports_main: int
    live_jito_tip_lamports_main: int
    live_use_dynamic_priority_fee_main: bool
    live_use_dynamic_jito_tip_main: bool
    live_entry_roundtrip_guard_enabled: bool
    live_entry_min_roundtrip_ratio: float
    live_entry_max_price_impact_pct: float
    live_entry_max_price_impact_pct_wallet: float
    live_entry_max_price_impact_pct_wallet_pump_fun: float
    live_entry_max_price_impact_pct_wallet_pump_amm: float
    live_sell_max_price_impact_pct: float
    live_sell_circuit_breaker_threshold: int
    live_sell_circuit_breaker_cooldown_sec: float
    live_sell_slippage_stuck_threshold: int
    live_allow_new_buys: bool
    live_pool_liveness_probe_enabled: bool
    live_pool_liveness_probe_ttl_sec: float
    live_entry_pool_max_age_sec: float
    live_entry_min_pool_sol_reserve: float
    live_entry_min_unique_wallets_30s: int
    live_allow_token_2022_buys: bool
    live_entry_max_top_holder_pct: float
    live_entry_max_top5_holder_pct: float
    live_entry_require_lp_burned: bool
    live_entry_lp_burn_threshold: float
    live_entry_lp_burn_cache_ttl_sec: float
    live_entry_lp_guard_sources: tuple[str, ...]
    live_entry_holder_exclude_pubkeys: tuple[str, ...]
    live_entry_require_freeze_authority_null: bool
    live_entry_require_mint_authority_null: bool
    live_entry_honeypot_sim_enabled: bool
    live_entry_honeypot_sim_fraction_bps: int
    entry_pure_buy_filter_enabled: bool
    entry_pure_buy_filter_max_age_sec: float
    entry_pure_buy_filter_min_buy_volume_sol: float
    entry_dev_wallet_check_enabled: bool
    entry_dev_wallet_max_tokens_24h: int
    entry_dev_wallet_check_sources: tuple[str, ...]
    live_reconciler_enabled: bool
    live_reconciler_interval_sec: float
    live_reconciler_drift_threshold_pct: float
    tracked_wallets_path: Path
    pump_rules_path: Path
    main_rules_path: Path
    sniper_rules_path: Path
    final_summary_path: Path
    top_rules_path: Path
    trusted_rules_path: Path
    regime_comparison_path: Path
    rules_source_mode: str
    allow_legacy_rule_fallback: bool
    enable_auto_trading: bool
    enable_paper_trading: bool
    enable_telegram: bool
    telegram_bot_token: str
    telegram_chat_id: str
    optional_allowed_regimes: tuple[str, ...]
    min_rule_support: int
    max_active_rules: int
    max_strict_rules: int
    disabled_rule_ids: tuple[str, ...]
    enable_runtime_rule_relaxation: bool
    derived_rule_volume_scale: float
    derived_rule_volume_floor: float
    relaxed_rule_size_multiplier: float
    enable_recovery_confirmation: bool
    recovery_confirmation_min_delta: float
    poll_interval_sec: int
    min_trigger_sol: float
    candidate_cooldown_sec: int
    candidate_maturation_sec: int
    discovery_mode: str
    tracked_wallet_features_enabled: bool
    discovery_account_include: tuple[str, ...]
    discovery_allowed_sources: tuple[str, ...]
    discovery_require_pump_suffix: bool
    enable_pair_first_rule_adaptation: bool
    pair_first_price_scale: float
    pair_first_volume_scale: float
    pair_first_cluster_scale: float
    pair_first_token_age_max_sec: float
    entry_min_token_age_sec: float
    entry_min_cluster_30s: int
    entry_min_tx_count_30s: int
    entry_min_volume_sol_30s: float
    entry_min_avg_trade_sol_30s: float
    entry_lane_shock_price_min: float
    entry_lane_shock_price_max: float
    entry_lane_shock_min_cluster: int
    entry_lane_shock_min_tx: int
    entry_lane_shock_min_volume_sol: float
    entry_lane_recovery_price_min: float
    entry_lane_recovery_price_max: float
    entry_lane_recovery_min_cluster: int
    entry_lane_recovery_min_tx: int
    entry_lane_recovery_min_volume_sol: float
    entry_lane_recovery_abs_move_min: float
    entry_lane_recovery_max_cluster: int
    entry_lane_recovery_max_tx: int
    entry_lane_recovery_max_volume_sol: float
    entry_lane_mature_enabled: bool
    entry_lane_mature_price_min: float
    entry_lane_mature_price_max: float
    entry_lane_mature_min_cluster: int
    entry_lane_mature_min_tx: int
    entry_lane_mature_min_volume_sol: float
    entry_overextension_price_max: float
    entry_overextension_fresh_age_sec: float
    entry_overextension_fresh_min_volume_sol: float
    entry_overextension_fresh_min_tx: int
    paper_entry_roundtrip_guard_enabled: bool
    paper_entry_min_roundtrip_ratio: float
    paper_entry_max_price_impact_pct: float
    paper_entry_min_roundtrip_ratio_main: float
    paper_entry_min_roundtrip_ratio_sniper: float
    paper_entry_max_price_impact_pct_main: float
    paper_entry_max_price_impact_pct_sniper: float
    paper_entry_min_roundtrip_ratio_sniper_pump_fun: float
    paper_entry_max_price_impact_pct_sniper_pump_fun: float
    paper_entry_min_roundtrip_ratio_wallet: float
    paper_entry_max_price_impact_pct_wallet: float
    paper_entry_min_roundtrip_ratio_wallet_pump_fun: float
    paper_entry_max_price_impact_pct_wallet_pump_fun: float
    paper_entry_max_price_impact_pct_wallet_pump_amm: float
    candidate_ranking_window_sec: float
    sniper_ranking_window_sec: float
    candidate_queue_min_size: int
    candidate_min_score: float
    ml_mode: str
    ml_model_backend: str
    ml_model_path: Path
    ml_samples_path: Path
    ml_bootstrap_enable: bool
    ml_bootstrap_path: Path
    ml_bootstrap_glob: str
    ml_bootstrap_max_rows: int
    ml_bootstrap_max_files: int
    ml_min_samples_activate: int
    ml_retrain_every: int
    ml_max_training_samples: int
    ml_positive_pnl_threshold_sol: float
    ml_threshold_main: float
    ml_threshold_sniper: float
    # Exit ML predictor
    ml_exit_mode: str
    ml_exit_sniper_threshold: float
    ml_exit_main_threshold: float
    ml_exit_min_samples: int
    ml_exit_retrain_every: int
    ml_exit_model_path: Path
    ml_exit_samples_path: Path
    post_close_observe_sec: int
    ml_exit_peak_lock_enabled: bool
    ml_exit_peak_lock_min_pnl: float
    ml_exit_peak_lock_drawdown: float
    ml_exit_peak_lock_threshold: float
    ml_exit_veto_reasons: tuple[str, ...]
    ml_exit_veto_threshold: float
    ml_exit_min_hold_sec: int
    ml_exit_min_hold_sec_sniper: int
    enable_main_strategy: bool
    enable_sniper_strategy: bool
    sniper_position_sol: float
    sniper_max_open_positions: int
    sniper_max_exposure_sol: float
    sniper_min_token_age_sec: float
    sniper_max_token_age_sec: float
    sniper_min_cluster_30s: int
    sniper_min_tx_count_30s: int
    sniper_min_volume_sol_30s: float
    sniper_min_price_change_30s: float
    sniper_max_price_change_30s: float
    sniper_token_cooldown_sec: int
    sniper_take_profit_pnl: float
    sniper_tp_min_gross_sol_floor: float
    sniper_tp_min_gross_fee_multiplier: float
    sniper_tp_min_gross_size_ratio: float
    sniper_stop_pnl: float
    sniper_max_hold_sec: int
    sniper_tp_confirm_ticks: int
    sniper_stop_confirm_ticks: int
    sniper_stop_min_hold_sec: int
    sniper_min_volume_per_tx_sol_30s: float
    sniper_use_runtime_rules: bool
    sniper_rule_ids: tuple[str, ...]
    sniper_allowed_sources: tuple[str, ...]
    main_rule_ids: tuple[str, ...]
    main_allowed_sources: tuple[str, ...]
    main_min_token_age_sec: float
    main_max_token_age_sec: float
    sniper_tp_jupiter_verify: bool
    sniper_tp_live_bypass_multiplier: float
    enable_wallet_strategy: bool
    wallet_position_sol: float
    wallet_max_open_positions: int
    wallet_max_exposure_sol: float
    wallet_min_cluster_300s: int
    wallet_min_buys_90s: int
    wallet_min_wallet_score_sum: float
    wallet_min_token_age_sec: float
    wallet_max_token_age_sec: float
    wallet_max_price_change_30s: float
    wallet_min_price_change_60s: float
    wallet_min_net_flow_sol_60s: float
    wallet_token_cooldown_sec: int
    wallet_take_profit_pnl: float
    wallet_stop_pnl: float
    wallet_max_hold_sec: int
    wallet_trailing_drawdown: float
    wallet_trailing_arm_confirm_ticks: int
    wallet_trailing_exit_confirm_ticks: int
    wallet_tp1_peak: float
    wallet_tp1_sell_fraction: float
    wallet_tp1_confirm_ticks: int
    wallet_allowed_sources: tuple[str, ...]
    wallet_copytrading_enabled: bool
    wallet_copy_position_sol: float
    wallet_copy_min_wallet_score: float
    wallet_copy_max_token_age_sec: float
    wallet_copy_event_max_age_sec: float
    wallet_copy_ml_bypass: bool
    wallet_copy_disable_sniper: bool
    wallet_copy_trail_arm_pnl: float
    wallet_copy_trail_drawdown: float
    wallet_copy_hard_stop_pnl: float
    wallet_copy_max_hold_sec: int
    wallet_copy_mirror_sell: bool
    wallet_copy_mirror_sell_profit_threshold: float
    wallet_copy_tp1_peak: float
    wallet_copy_tp1_sell_fraction: float
    wallet_copy_tp1_confirm_ticks: int
    wallet_copy_exit_confirm_ticks: int
    wallet_copy_trail_min_floor: float
    wallet_copy_score_penalty_on_loss: float
    wallet_copy_loss_pnl_threshold: float
    entry_score_tracked_wallet_presence_bonus: float
    entry_score_tracked_wallet_count_weight: float
    entry_score_tracked_wallet_count_scale: float
    entry_score_tracked_wallet_score_weight: float
    entry_score_tracked_wallet_score_scale: float
    regime_size_multiplier_negative_shock_recovery: float
    regime_size_multiplier_high_cluster_recovery: float
    regime_size_multiplier_momentum_burst: float
    regime_size_multiplier_unknown: float
    tracked_wallet_size_boost_per_wallet: float
    tracked_wallet_size_boost_cap: float
    exit_tp1_multiple: float
    exit_tp2_multiple: float
    exit_tp3_multiple: float
    exit_tp1_sell_fraction: float
    exit_tp2_sell_fraction: float
    post_tp1_stop_pnl: float
    exit_rule_stop_overrides: dict[str, float]
    post_tp2_trailing_drawdown: float
    post_tp2_timeout_sec: int
    tp1_confirm_ticks: int
    tp2_confirm_ticks: int
    tp1_min_volume_sol_30s: float
    tp2_min_volume_sol_30s: float
    tp3_confirm_ticks: int
    tp2_fast_confirm_ticks: int
    tp2_fast_min_volume_sol_30s: float
    tp3_min_volume_sol_30s: float
    exit_price_max_step_multiple: float
    exit_outlier_max_pnl_jump: float
    exit_outlier_low_volume_sol_30s: float
    exit_max_peak_pnl_multiple: float
    price_outlier_min_samples: int
    price_outlier_median_window: int
    price_outlier_max_multiple: float
    price_outlier_confirm_signatures: int
    price_outlier_confirm_window_sec: int
    price_outlier_confirm_tolerance: float
    stage0_loss_timeout_sec: int
    stage0_loss_timeout_max_pnl: float
    stage0_early_profit_window_sec: int
    stage0_early_profit_min_pnl: float
    stage0_early_profit_max_pnl: float
    stage0_early_profit_confirm_ticks: int
    stage0_early_profit_sell_fraction: float
    stage0_crash_guard_window_sec: int
    stage0_crash_guard_min_pnl: float
    stage0_crash_guard_min_hold_sec: int
    stage0_crash_guard_confirm_ticks: int
    pre_tp1_retrace_lock_min_hold_sec: int
    pre_tp1_retrace_lock_arm_pnl: float
    pre_tp1_retrace_lock_drawdown: float
    pre_tp1_retrace_lock_floor_pnl: float
    pre_tp1_retrace_lock_confirm_ticks: int
    stage0_fast_fail_non_positive_sec: int
    stage0_fast_fail_non_positive_max_pnl: float
    stage0_fast_fail_under_profit_sec: int
    stage0_fast_fail_under_profit_min_pnl: float
    stage0_moderate_positive_timeout_sec: int
    stage0_moderate_positive_min_pnl: float
    stage0_moderate_positive_max_pnl: float
    stage0_moderate_positive_skip_profiles: tuple[str, ...]
    stage1_low_positive_timeout_sec: int
    stage1_low_positive_min_pnl: float
    stage1_low_positive_max_pnl: float
    stage1_sub2x_timeout_sec: int
    stage1_sub2x_min_pnl: float
    stage1_sub2x_max_pnl: float
    absolute_max_hold_sec: int
    stale_sweep_sec: int
    dead_token_hold_sec: int
    dead_token_move_pct: float
    dead_token_confirm_ticks: int
    dead_token_max_tx_count_60s: int
    dead_token_max_volume_sol_60s: float
    bot_db_path: Path
    event_log_path: Path
    event_log_throttle_window_sec: float
    event_log_throttled_event_types: tuple[str, ...]
    bot_status_path: Path
    # Market regime gate
    market_regime_enabled: bool
    market_regime_gate_active: bool
    market_regime_win_rate_window: int
    market_regime_min_win_rate: float
    market_regime_bootstrap_positions: int
    market_regime_min_candidates_5min: int
    market_regime_sol_enabled: bool
    market_regime_sol_drop_threshold: float
    market_regime_pause_cooldown_sec: int


def effective_max_price_impact_pct(
    config: "BotConfig",
    *,
    strategy_id: str | None,
    source_program: str | None,
    live_mode: bool,
) -> float:
    """Pick the price-impact threshold for a given lane / venue / mode.

    Wallet lane on pump.fun needs a much looser cap than the generic 0.25%
    because early-life pump.fun pools routinely quote 5-15% impact on
    0.05 SOL buys; that's venue texture, not a bad trade.
    """
    sid = (strategy_id or "main").lower()
    src = (source_program or "").upper()
    if live_mode:
        if sid == "wallet":
            if src == "PUMP_FUN":
                return float(config.live_entry_max_price_impact_pct_wallet_pump_fun)
            if src == "PUMP_AMM":
                return float(config.live_entry_max_price_impact_pct_wallet_pump_amm)
            return float(config.live_entry_max_price_impact_pct_wallet)
        return float(config.live_entry_max_price_impact_pct)
    if sid == "wallet":
        if src == "PUMP_FUN":
            return float(config.paper_entry_max_price_impact_pct_wallet_pump_fun)
        if src == "PUMP_AMM":
            return float(config.paper_entry_max_price_impact_pct_wallet_pump_amm)
        return float(config.paper_entry_max_price_impact_pct_wallet)
    if sid == "sniper":
        if src == "PUMP_FUN":
            return float(config.paper_entry_max_price_impact_pct_sniper_pump_fun)
        return float(config.paper_entry_max_price_impact_pct_sniper)
    return float(config.paper_entry_max_price_impact_pct_main)


def effective_min_roundtrip_ratio(
    config: "BotConfig",
    *,
    strategy_id: str | None,
    source_program: str | None,
    live_mode: bool,
) -> float:
    sid = (strategy_id or "main").lower()
    src = (source_program or "").upper()
    if live_mode:
        return float(
            getattr(
                config,
                "live_entry_min_roundtrip_ratio",
                config.paper_entry_min_roundtrip_ratio,
            )
        )
    if sid == "wallet":
        if src == "PUMP_FUN":
            return float(config.paper_entry_min_roundtrip_ratio_wallet_pump_fun)
        return float(config.paper_entry_min_roundtrip_ratio_wallet)
    if sid == "sniper":
        if src == "PUMP_FUN":
            return float(config.paper_entry_min_roundtrip_ratio_sniper_pump_fun)
        return float(config.paper_entry_min_roundtrip_ratio_sniper)
    return float(config.paper_entry_min_roundtrip_ratio_main)


def load_bot_config() -> BotConfig:
    """Load bot configuration from env and project config."""
    app = load_app_config()
    root = app.root_dir
    discovery_mode = _str_env("DISCOVERY_MODE", "pair_first").lower()
    if discovery_mode not in {"pair_first", "wallet_first"}:
        discovery_mode = "pair_first"
    tracked_wallet_features_enabled = _bool_env(
        "TRACKED_WALLET_FEATURES_ENABLED",
        discovery_mode != "pair_first",
    )
    rules_source_mode = _str_env("RULES_SOURCE_MODE", "pump").lower()
    if rules_source_mode not in {"auto", "pump", "legacy"}:
        rules_source_mode = "auto"
    live_broadcast_mode = _str_env("LIVE_BROADCAST_MODE", "staked_rpc").lower()
    if live_broadcast_mode not in {
        "staked_rpc",
        "helius_sender",
        "helius_sender_swqos",
        "helius_bundle",
    }:
        live_broadcast_mode = "staked_rpc"
    live_broadcast_fee_type = _str_env("LIVE_BROADCAST_FEE_TYPE", "maxCap")
    if live_broadcast_fee_type not in {"maxCap", "exactFee"}:
        live_broadcast_fee_type = "maxCap"
    default_discovery_accounts = ",".join(
        [
            # Pump.fun + Pump AMM
            "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",
            "term9YPb9mzAsABaqN71A4xdbxHmpBNZavpBiQKZzN3",
            "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA",
            # Raydium LaunchLab + AMM families
            "LanMV9sAd7wArD4vJFi88GyFnpT5z6YGeFCYcRpEPRQ",
            "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
            "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C",
        ]
    )
    default_jito_tip_accounts = ",".join(
        [
            "4ACfpUFoaSD9bfPdeu6DBt89gB6ENTeHBXCAi87NhDEE",
            "D2L6yPZ2FmmmTKPgzaMKdhu6EWZcTpLy1Vhx8uvZe7NZ",
            "9bnz4RShgq1hAnLnZbP8kbgBg1kEmcJBYQq3gQbmnSta",
            "5VY91ws6B2hMmBFRsXkoAAdsPHBJwRfBht4DXox3xkwn",
            "2nyhqdwKcJZR2vcqCyrYsaPVdAnFoJjiksCXJ7hfEYgD",
            "2q5pghRs6arqVjRvT5gfgWfWcHWmw1ZuCzphgd5KfWGJ",
            "wyvPkWjVZz1M8fHQnMMCDTQDbkManefNNhweYk5WkcF",
            "3KCKozbAaF75qEU33jtzozcJ29yJuaLJTy2jFdzUY8bT",
            "4vieeGHPYPG2MmyPRcYjdiDmmhN3ww7hsFNap8pVN3Ey",
            "4TQLFNWK8AovT1gFvda5jfw2oJeRMKEmw7aH6MGBJ3or",
        ]
    )

    return BotConfig(
        app=app,
        helius_api_key=_str_env("HELIUS_API_KEY"),
        helius_base_url=_str_env("HELIUS_BASE_URL", app.env.get("HELIUS_BASE_URL", "")),
        helius_rpc_url=_str_env("HELIUS_RPC_URL"),
        chainstack_grpc_endpoint=_str_env("CHAINSTACK_GRPC_ENDPOINT"),
        chainstack_grpc_token=_str_env("CHAINSTACK_GRPC_TOKEN"),
        chainstack_reconnect_max_retries=int(os.getenv("CHAINSTACK_RECONNECT_MAX_RETRIES", "0")),
        chainstack_reconnect_backoff_initial_sec=int(
            os.getenv("CHAINSTACK_RECONNECT_BACKOFF_INITIAL_SEC", "1")
        ),
        chainstack_reconnect_backoff_max_sec=int(
            os.getenv("CHAINSTACK_RECONNECT_BACKOFF_MAX_SEC", "30")
        ),
        jupiter_base_url=_str_env("JUPITER_BASE_URL", "https://api.jup.ag/swap/v1"),
        jupiter_api_key=_str_env("JUPITER_API_KEY", ""),
        bot_private_key_b58=_str_env("BOT_PRIVATE_KEY_B58"),
        bot_public_key=_str_env("BOT_PUBLIC_KEY"),
        solana_cluster=_str_env("SOLANA_CLUSTER", "mainnet-beta"),
        max_position_sol=float(os.getenv("MAX_POSITION_SOL", "0.05")),
        max_total_exposure_sol=float(os.getenv("MAX_TOTAL_EXPOSURE_SOL", "0.20")),
        max_daily_loss_sol=float(os.getenv("MAX_DAILY_LOSS_SOL", "0.20")),
        max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "5")),
        default_slippage_bps=int(os.getenv("DEFAULT_SLIPPAGE_BPS", "150")),
        priority_fee_lamports=int(os.getenv("PRIORITY_FEE_LAMPORTS", "50000")),
        jito_tip_lamports=int(os.getenv("JITO_TIP_LAMPORTS", "0")),
        jito_tip_accounts=_csv_env("JITO_TIP_ACCOUNTS", default_jito_tip_accounts),
        live_broadcast_mode=live_broadcast_mode,
        helius_sender_url=_str_env("HELIUS_SENDER_URL", "https://sender.helius-rpc.com/fast"),
        helius_bundle_url=_str_env("HELIUS_BUNDLE_URL", ""),
        live_broadcast_fee_type=live_broadcast_fee_type,
        live_use_dynamic_priority_fee=_bool_env("LIVE_USE_DYNAMIC_PRIORITY_FEE", True),
        live_use_dynamic_jito_tip=_bool_env("LIVE_USE_DYNAMIC_JITO_TIP", True),
        live_use_jupiter_auto_slippage=_bool_env("LIVE_USE_JUPITER_AUTO_SLIPPAGE", True),
        live_buy_slippage_bps=int(os.getenv("LIVE_BUY_SLIPPAGE_BPS", "0")),
        live_sell_slippage_bps=int(os.getenv("LIVE_SELL_SLIPPAGE_BPS", "0")),
        live_buy_slippage_bps_sniper_pump_fun=int(
            os.getenv("LIVE_BUY_SLIPPAGE_BPS_SNIPER_PUMP_FUN", "3000")
        ),
        live_sell_slippage_bps_sniper_pump_fun=int(
            os.getenv("LIVE_SELL_SLIPPAGE_BPS_SNIPER_PUMP_FUN", "3000")
        ),
        live_rebroadcast_interval_ms=int(os.getenv("LIVE_REBROADCAST_INTERVAL_MS", "250")),
        live_confirm_poll_interval_ms=int(os.getenv("LIVE_CONFIRM_POLL_INTERVAL_MS", "200")),
        live_max_rebroadcast_attempts=int(os.getenv("LIVE_MAX_REBROADCAST_ATTEMPTS", "8")),
        live_sender_idle_ping_sec=int(os.getenv("LIVE_SENDER_IDLE_PING_SEC", "45")),
        live_sender_active_warm=_bool_env("LIVE_SENDER_ACTIVE_WARM", True),
        live_sender_warm_interval_sec=float(os.getenv("LIVE_SENDER_WARM_INTERVAL_SEC", "1")),
        live_min_wallet_buffer_lamports=int(
            os.getenv("LIVE_MIN_WALLET_BUFFER_LAMPORTS", "1000000")
        ),
        live_buy_ata_rent_buffer_lamports=int(
            os.getenv("LIVE_BUY_ATA_RENT_BUFFER_LAMPORTS", "2200000")
        ),
        live_token_account_rent_lamports=int(
            os.getenv("LIVE_TOKEN_ACCOUNT_RENT_LAMPORTS", "2039280")
        ),
        live_min_net_exit_lamports=int(os.getenv("LIVE_MIN_NET_EXIT_LAMPORTS", "1")),
        live_close_token_ata_on_full_exit=_bool_env("LIVE_CLOSE_TOKEN_ATA_ON_FULL_EXIT", True),
        live_use_shared_accounts=_bool_env("LIVE_USE_SHARED_ACCOUNTS", False),
        live_enable_native_pump_amm=_bool_env("LIVE_ENABLE_NATIVE_PUMP_AMM", True),
        live_pump_amm_buy_compute_unit_limit=int(
            os.getenv("LIVE_PUMP_AMM_BUY_COMPUTE_UNIT_LIMIT", "450000")
        ),
        live_pump_amm_sell_compute_unit_limit=int(
            os.getenv("LIVE_PUMP_AMM_SELL_COMPUTE_UNIT_LIMIT", "350000")
        ),
        live_sell_max_attempts=int(os.getenv("LIVE_SELL_MAX_ATTEMPTS", "3")),
        live_preflight_simulate=_bool_env("LIVE_PREFLIGHT_SIMULATE", True),
        live_priority_fee_lamports_main=int(
            os.getenv(
                "LIVE_PRIORITY_FEE_LAMPORTS_MAIN",
                os.getenv("PRIORITY_FEE_LAMPORTS", "50000"),
            )
        ),
        live_jito_tip_lamports_main=int(
            os.getenv("LIVE_JITO_TIP_LAMPORTS_MAIN", os.getenv("JITO_TIP_LAMPORTS", "0"))
        ),
        live_use_dynamic_priority_fee_main=_bool_env("LIVE_USE_DYNAMIC_PRIORITY_FEE_MAIN", False),
        live_use_dynamic_jito_tip_main=_bool_env("LIVE_USE_DYNAMIC_JITO_TIP_MAIN", False),
        live_entry_roundtrip_guard_enabled=_bool_env("LIVE_ENTRY_ROUNDTRIP_GUARD_ENABLED", True),
        live_entry_min_roundtrip_ratio=float(os.getenv("LIVE_ENTRY_MIN_ROUNDTRIP_RATIO", "0.85")),
        live_entry_max_price_impact_pct=float(os.getenv("LIVE_ENTRY_MAX_PRICE_IMPACT_PCT", "0.25")),
        live_entry_max_price_impact_pct_wallet=float(
            os.getenv(
                "LIVE_ENTRY_MAX_PRICE_IMPACT_PCT_WALLET",
                os.getenv("LIVE_ENTRY_MAX_PRICE_IMPACT_PCT", "0.25"),
            )
        ),
        live_entry_max_price_impact_pct_wallet_pump_fun=float(
            os.getenv("LIVE_ENTRY_MAX_PRICE_IMPACT_PCT_WALLET_PUMP_FUN", "10.0")
        ),
        live_entry_max_price_impact_pct_wallet_pump_amm=float(
            os.getenv("LIVE_ENTRY_MAX_PRICE_IMPACT_PCT_WALLET_PUMP_AMM", "10.0")
        ),
        live_sell_max_price_impact_pct=float(os.getenv("LIVE_SELL_MAX_PRICE_IMPACT_PCT", "0.5")),
        live_sell_circuit_breaker_threshold=int(
            os.getenv("LIVE_SELL_CIRCUIT_BREAKER_THRESHOLD", "3")
        ),
        live_sell_circuit_breaker_cooldown_sec=float(
            os.getenv("LIVE_SELL_CIRCUIT_BREAKER_COOLDOWN_SEC", "300")
        ),
        live_sell_slippage_stuck_threshold=int(
            os.getenv("LIVE_SELL_SLIPPAGE_STUCK_THRESHOLD", "5")
        ),
        live_allow_new_buys=_bool_env("LIVE_ALLOW_NEW_BUYS", True),
        live_pool_liveness_probe_enabled=_bool_env("LIVE_POOL_LIVENESS_PROBE_ENABLED", True),
        live_pool_liveness_probe_ttl_sec=float(os.getenv("LIVE_POOL_LIVENESS_PROBE_TTL_SEC", "30")),
        live_entry_pool_max_age_sec=float(os.getenv("LIVE_ENTRY_POOL_MAX_AGE_SEC", "120")),
        live_entry_min_pool_sol_reserve=float(os.getenv("LIVE_ENTRY_MIN_POOL_SOL_RESERVE", "10")),
        live_entry_min_unique_wallets_30s=int(os.getenv("LIVE_ENTRY_MIN_UNIQUE_WALLETS_30S", "5")),
        live_allow_token_2022_buys=_bool_env("LIVE_ALLOW_TOKEN_2022_BUYS", False),
        live_entry_max_top_holder_pct=float(os.getenv("LIVE_ENTRY_MAX_TOP_HOLDER_PCT", "0.10")),
        live_entry_max_top5_holder_pct=float(os.getenv("LIVE_ENTRY_MAX_TOP5_HOLDER_PCT", "0.25")),
        live_entry_require_lp_burned=_bool_env("LIVE_ENTRY_REQUIRE_LP_BURNED", False),
        live_entry_lp_burn_threshold=float(os.getenv("LIVE_ENTRY_LP_BURN_THRESHOLD", "0.90")),
        live_entry_lp_burn_cache_ttl_sec=float(
            os.getenv("LIVE_ENTRY_LP_BURN_CACHE_TTL_SEC", "300")
        ),
        live_entry_lp_guard_sources=_csv_env("LIVE_ENTRY_LP_GUARD_SOURCES", "RAYDIUM"),
        live_entry_holder_exclude_pubkeys=tuple(
            p.strip()
            for p in os.getenv("LIVE_ENTRY_HOLDER_EXCLUDE_PUBKEYS", "").split(",")
            if p.strip()
        ),
        live_entry_require_freeze_authority_null=_bool_env(
            "LIVE_ENTRY_REQUIRE_FREEZE_AUTHORITY_NULL", True
        ),
        live_entry_require_mint_authority_null=_bool_env(
            "LIVE_ENTRY_REQUIRE_MINT_AUTHORITY_NULL", False
        ),
        live_entry_honeypot_sim_enabled=_bool_env("LIVE_ENTRY_HONEYPOT_SIM_ENABLED", True),
        live_entry_honeypot_sim_fraction_bps=int(
            os.getenv("LIVE_ENTRY_HONEYPOT_SIM_FRACTION_BPS", "500")
        ),
        entry_pure_buy_filter_enabled=_bool_env("ENTRY_PURE_BUY_FILTER_ENABLED", True),
        entry_pure_buy_filter_max_age_sec=float(
            os.getenv("ENTRY_PURE_BUY_FILTER_MAX_AGE_SEC", "180")
        ),
        entry_pure_buy_filter_min_buy_volume_sol=float(
            os.getenv("ENTRY_PURE_BUY_FILTER_MIN_BUY_VOLUME_SOL", "0.05")
        ),
        entry_dev_wallet_check_enabled=_bool_env("ENTRY_DEV_WALLET_CHECK_ENABLED", True),
        entry_dev_wallet_max_tokens_24h=int(os.getenv("ENTRY_DEV_WALLET_MAX_TOKENS_24H", "3")),
        entry_dev_wallet_check_sources=_csv_env("ENTRY_DEV_WALLET_CHECK_SOURCES", "PUMP_FUN"),
        live_reconciler_enabled=_bool_env("LIVE_RECONCILER_ENABLED", True),
        live_reconciler_interval_sec=float(os.getenv("LIVE_RECONCILER_INTERVAL_SEC", "60")),
        live_reconciler_drift_threshold_pct=float(
            os.getenv("LIVE_RECONCILER_DRIFT_THRESHOLD_PCT", "0.10")
        ),
        tracked_wallets_path=root
        / os.getenv("TRACKED_WALLETS_PATH", "data/bronze/wallet_pool.parquet"),
        pump_rules_path=root / os.getenv("PUMP_RULES_PATH", "outputs/rules/pump_rule_packs_v2.csv"),
        main_rules_path=root / os.getenv("MAIN_RULES_PATH", "outputs/rules/mature_pairs_v1.csv"),
        sniper_rules_path=root
        / os.getenv("SNIPER_RULES_PATH", "outputs/rules/kaggle_mined_v1.csv"),
        final_summary_path=root
        / os.getenv("FINAL_SUMMARY_PATH", "outputs/reports/final_research_summary.json"),
        top_rules_path=root
        / os.getenv("TOP_RULES_PATH", "outputs/reports/final_research_summary_top_rules.csv"),
        trusted_rules_path=root
        / os.getenv("TRUSTED_RULES_PATH", "outputs/rules/trusted_rule_shortlist.csv"),
        regime_comparison_path=root
        / os.getenv("REGIME_COMPARISON_PATH", "outputs/reports/regime_comparison.csv"),
        rules_source_mode=rules_source_mode,
        allow_legacy_rule_fallback=_bool_env("ALLOW_LEGACY_RULE_FALLBACK", False),
        enable_auto_trading=_bool_env("ENABLE_AUTO_TRADING", False),
        enable_paper_trading=_bool_env("ENABLE_PAPER_TRADING", True),
        enable_telegram=_bool_env("ENABLE_TELEGRAM", False),
        telegram_bot_token=_str_env("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_str_env("TELEGRAM_CHAT_ID"),
        optional_allowed_regimes=tuple(
            [
                item.strip()
                for item in os.getenv("OPTIONAL_ALLOWED_REGIMES", "").split(",")
                if item.strip()
            ]
        ),
        min_rule_support=int(os.getenv("MIN_RULE_SUPPORT", "50")),
        max_active_rules=int(os.getenv("MAX_ACTIVE_RULES", "10")),
        max_strict_rules=int(os.getenv("MAX_STRICT_RULES", "2")),
        disabled_rule_ids=tuple(
            dict.fromkeys(
                _csv_env("DISABLED_RULE_IDS", "")
                + (
                    "sniper_amm_0001",
                    "exp_vol_02sol",
                    "exp_recovery_-30_+00",
                    "pump_v2_0253",
                    "main_v1_0005",
                )
            )
        ),
        enable_runtime_rule_relaxation=_bool_env("ENABLE_RUNTIME_RULE_RELAXATION", True),
        derived_rule_volume_scale=float(os.getenv("DERIVED_RULE_VOLUME_SCALE", "0.5")),
        derived_rule_volume_floor=float(os.getenv("DERIVED_RULE_VOLUME_FLOOR", "0.5")),
        relaxed_rule_size_multiplier=float(os.getenv("RELAXED_RULE_SIZE_MULTIPLIER", "0.5")),
        enable_recovery_confirmation=_bool_env("ENABLE_RECOVERY_CONFIRMATION", True),
        recovery_confirmation_min_delta=float(os.getenv("RECOVERY_CONFIRMATION_MIN_DELTA", "0.0")),
        poll_interval_sec=int(os.getenv("POLL_INTERVAL_SEC", "15")),
        min_trigger_sol=float(os.getenv("MIN_TRIGGER_SOL", "0.2")),
        candidate_cooldown_sec=int(os.getenv("CANDIDATE_COOLDOWN_SEC", "20")),
        candidate_maturation_sec=int(os.getenv("CANDIDATE_MATURATION_SEC", "0")),
        discovery_mode=discovery_mode,
        tracked_wallet_features_enabled=tracked_wallet_features_enabled,
        discovery_account_include=_csv_env("DISCOVERY_ACCOUNT_INCLUDE", default_discovery_accounts),
        discovery_allowed_sources=_csv_env(
            "DISCOVERY_ALLOWED_SOURCES",
            "PUMP_FUN,PUMP_AMM,RAYDIUM,RAYDIUM_LAUNCHLAB",
        ),
        discovery_require_pump_suffix=_bool_env("DISCOVERY_REQUIRE_PUMP_SUFFIX", False),
        enable_pair_first_rule_adaptation=_bool_env("ENABLE_PAIR_FIRST_RULE_ADAPTATION", True),
        pair_first_price_scale=float(os.getenv("PAIR_FIRST_PRICE_SCALE", "1.0")),
        pair_first_volume_scale=float(os.getenv("PAIR_FIRST_VOLUME_SCALE", "1.0")),
        pair_first_cluster_scale=float(os.getenv("PAIR_FIRST_CLUSTER_SCALE", "1.0")),
        pair_first_token_age_max_sec=float(os.getenv("PAIR_FIRST_TOKEN_AGE_MAX_SEC", "3600")),
        entry_min_token_age_sec=float(os.getenv("ENTRY_MIN_TOKEN_AGE_SEC", "20")),
        entry_min_cluster_30s=int(os.getenv("ENTRY_MIN_CLUSTER_30S", "2")),
        entry_min_tx_count_30s=int(os.getenv("ENTRY_MIN_TX_COUNT_30S", "2")),
        entry_min_volume_sol_30s=float(os.getenv("ENTRY_MIN_VOLUME_SOL_30S", "1.5")),
        entry_min_avg_trade_sol_30s=float(os.getenv("ENTRY_MIN_AVG_TRADE_SOL_30S", "0.5")),
        entry_lane_shock_price_min=float(os.getenv("ENTRY_LANE_SHOCK_PRICE_MIN", "-0.5")),
        entry_lane_shock_price_max=float(os.getenv("ENTRY_LANE_SHOCK_PRICE_MAX", "-0.2")),
        entry_lane_shock_min_cluster=int(os.getenv("ENTRY_LANE_SHOCK_MIN_CLUSTER", "2")),
        entry_lane_shock_min_tx=int(os.getenv("ENTRY_LANE_SHOCK_MIN_TX", "2")),
        entry_lane_shock_min_volume_sol=float(os.getenv("ENTRY_LANE_SHOCK_MIN_VOLUME_SOL", "5.0")),
        entry_lane_recovery_price_min=float(os.getenv("ENTRY_LANE_RECOVERY_PRICE_MIN", "-0.2")),
        entry_lane_recovery_price_max=float(os.getenv("ENTRY_LANE_RECOVERY_PRICE_MAX", "0.35")),
        entry_lane_recovery_min_cluster=int(os.getenv("ENTRY_LANE_RECOVERY_MIN_CLUSTER", "3")),
        entry_lane_recovery_min_tx=int(os.getenv("ENTRY_LANE_RECOVERY_MIN_TX", "3")),
        entry_lane_recovery_min_volume_sol=float(
            os.getenv("ENTRY_LANE_RECOVERY_MIN_VOLUME_SOL", "5.0")
        ),
        entry_lane_recovery_abs_move_min=float(
            os.getenv("ENTRY_LANE_RECOVERY_ABS_MOVE_MIN", "0.05")
        ),
        entry_lane_recovery_max_cluster=int(os.getenv("ENTRY_LANE_RECOVERY_MAX_CLUSTER", "8")),
        entry_lane_recovery_max_tx=int(os.getenv("ENTRY_LANE_RECOVERY_MAX_TX", "8")),
        entry_lane_recovery_max_volume_sol=float(
            os.getenv("ENTRY_LANE_RECOVERY_MAX_VOLUME_SOL", "25.0")
        ),
        entry_lane_mature_enabled=_bool_env("ENTRY_LANE_MATURE_ENABLED", True),
        entry_lane_mature_price_min=float(os.getenv("ENTRY_LANE_MATURE_PRICE_MIN", "-0.10")),
        entry_lane_mature_price_max=float(os.getenv("ENTRY_LANE_MATURE_PRICE_MAX", "0.60")),
        entry_lane_mature_min_cluster=int(os.getenv("ENTRY_LANE_MATURE_MIN_CLUSTER", "3")),
        entry_lane_mature_min_tx=int(os.getenv("ENTRY_LANE_MATURE_MIN_TX", "5")),
        entry_lane_mature_min_volume_sol=float(
            os.getenv("ENTRY_LANE_MATURE_MIN_VOLUME_SOL", "3.0")
        ),
        entry_overextension_price_max=float(os.getenv("ENTRY_OVEREXTENSION_PRICE_MAX", "0.5")),
        entry_overextension_fresh_age_sec=float(
            os.getenv("ENTRY_OVEREXTENSION_FRESH_AGE_SEC", "60")
        ),
        entry_overextension_fresh_min_volume_sol=float(
            os.getenv("ENTRY_OVEREXTENSION_FRESH_MIN_VOLUME_SOL", "5.0")
        ),
        entry_overextension_fresh_min_tx=int(os.getenv("ENTRY_OVEREXTENSION_FRESH_MIN_TX", "2")),
        paper_entry_roundtrip_guard_enabled=_bool_env("PAPER_ENTRY_ROUNDTRIP_GUARD_ENABLED", True),
        paper_entry_min_roundtrip_ratio=float(os.getenv("PAPER_ENTRY_MIN_ROUNDTRIP_RATIO", "0.70")),
        paper_entry_max_price_impact_pct=float(
            os.getenv("PAPER_ENTRY_MAX_PRICE_IMPACT_PCT", "0.35")
        ),
        paper_entry_min_roundtrip_ratio_main=float(
            os.getenv(
                "PAPER_ENTRY_MIN_ROUNDTRIP_RATIO_MAIN",
                os.getenv("PAPER_ENTRY_MIN_ROUNDTRIP_RATIO", "0.70"),
            )
        ),
        paper_entry_min_roundtrip_ratio_sniper=float(
            os.getenv("PAPER_ENTRY_MIN_ROUNDTRIP_RATIO_SNIPER", "0.78")
        ),
        paper_entry_max_price_impact_pct_main=float(
            os.getenv(
                "PAPER_ENTRY_MAX_PRICE_IMPACT_PCT_MAIN",
                os.getenv("PAPER_ENTRY_MAX_PRICE_IMPACT_PCT", "0.35"),
            )
        ),
        paper_entry_max_price_impact_pct_sniper=float(
            os.getenv("PAPER_ENTRY_MAX_PRICE_IMPACT_PCT_SNIPER", "0.25")
        ),
        paper_entry_min_roundtrip_ratio_sniper_pump_fun=float(
            os.getenv(
                "PAPER_ENTRY_MIN_ROUNDTRIP_RATIO_SNIPER_PUMP_FUN",
                os.getenv("PAPER_ENTRY_MIN_ROUNDTRIP_RATIO_SNIPER", "0.60"),
            )
        ),
        paper_entry_max_price_impact_pct_sniper_pump_fun=float(
            os.getenv("PAPER_ENTRY_MAX_PRICE_IMPACT_PCT_SNIPER_PUMP_FUN", "5.0")
        ),
        paper_entry_min_roundtrip_ratio_wallet=float(
            os.getenv("PAPER_ENTRY_MIN_ROUNDTRIP_RATIO_WALLET", "0.85")
        ),
        paper_entry_max_price_impact_pct_wallet=float(
            os.getenv(
                "PAPER_ENTRY_MAX_PRICE_IMPACT_PCT_WALLET",
                os.getenv("PAPER_ENTRY_MAX_PRICE_IMPACT_PCT_MAIN", "0.35"),
            )
        ),
        paper_entry_min_roundtrip_ratio_wallet_pump_fun=float(
            os.getenv("PAPER_ENTRY_MIN_ROUNDTRIP_RATIO_WALLET_PUMP_FUN", "0.85")
        ),
        paper_entry_max_price_impact_pct_wallet_pump_fun=float(
            os.getenv("PAPER_ENTRY_MAX_PRICE_IMPACT_PCT_WALLET_PUMP_FUN", "10.0")
        ),
        paper_entry_max_price_impact_pct_wallet_pump_amm=float(
            os.getenv("PAPER_ENTRY_MAX_PRICE_IMPACT_PCT_WALLET_PUMP_AMM", "10.0")
        ),
        candidate_ranking_window_sec=float(os.getenv("CANDIDATE_RANKING_WINDOW_SEC", "0.5")),
        sniper_ranking_window_sec=float(os.getenv("SNIPER_RANKING_WINDOW_SEC", "0.2")),
        candidate_queue_min_size=int(os.getenv("CANDIDATE_QUEUE_MIN_SIZE", "2")),
        candidate_min_score=float(os.getenv("CANDIDATE_MIN_SCORE", "0.0")),
        ml_mode=_str_env("ML_MODE", "shadow"),
        ml_model_backend=_str_env("ML_MODEL_BACKEND", "auto"),
        ml_model_path=root / os.getenv("ML_MODEL_PATH", "models/entry_filter_model.joblib"),
        ml_samples_path=root / os.getenv("ML_SAMPLES_PATH", "data/live/ml_samples.jsonl"),
        ml_bootstrap_enable=_bool_env("ML_BOOTSTRAP_ENABLE", True),
        ml_bootstrap_path=root / os.getenv("ML_BOOTSTRAP_PATH", "data/external/pump_dataset"),
        ml_bootstrap_glob=_str_env("ML_BOOTSTRAP_GLOB", "ml_bootstrap*.csv"),
        ml_bootstrap_max_rows=int(os.getenv("ML_BOOTSTRAP_MAX_ROWS", "50000")),
        ml_bootstrap_max_files=int(os.getenv("ML_BOOTSTRAP_MAX_FILES", "20")),
        ml_min_samples_activate=int(os.getenv("ML_MIN_SAMPLES_ACTIVATE", "200")),
        ml_retrain_every=int(os.getenv("ML_RETRAIN_EVERY", "50")),
        ml_max_training_samples=int(os.getenv("ML_MAX_TRAINING_SAMPLES", "30000")),
        ml_positive_pnl_threshold_sol=float(os.getenv("ML_POSITIVE_PNL_THRESHOLD_SOL", "0.0")),
        ml_threshold_main=float(os.getenv("ML_THRESHOLD_MAIN", "0.60")),
        ml_threshold_sniper=float(os.getenv("ML_THRESHOLD_SNIPER", "0.62")),
        ml_exit_mode=_str_env("ML_EXIT_MODE", "shadow"),
        ml_exit_sniper_threshold=float(os.getenv("ML_EXIT_SNIPER_THRESHOLD", "0.40")),
        ml_exit_main_threshold=float(os.getenv("ML_EXIT_MAIN_THRESHOLD", "0.45")),
        ml_exit_min_samples=int(os.getenv("ML_EXIT_MIN_SAMPLES", "50")),
        ml_exit_retrain_every=int(os.getenv("ML_EXIT_RETRAIN_EVERY", "25")),
        ml_exit_model_path=root / os.getenv("ML_EXIT_MODEL_PATH", "models/exit_predictor.joblib"),
        ml_exit_samples_path=root
        / os.getenv("ML_EXIT_SAMPLES_PATH", "data/live/ml_exit_samples.jsonl"),
        post_close_observe_sec=int(os.getenv("POST_CLOSE_OBSERVE_SEC", "120")),
        ml_exit_peak_lock_enabled=_bool_env("ML_EXIT_PEAK_LOCK_ENABLED", False),
        ml_exit_peak_lock_min_pnl=float(os.getenv("ML_EXIT_PEAK_LOCK_MIN_PNL", "0.15")),
        ml_exit_peak_lock_drawdown=float(os.getenv("ML_EXIT_PEAK_LOCK_DRAWDOWN", "0.08")),
        ml_exit_peak_lock_threshold=float(os.getenv("ML_EXIT_PEAK_LOCK_THRESHOLD", "0.35")),
        ml_exit_veto_reasons=_csv_env("ML_EXIT_VETO_REASONS", ""),
        ml_exit_veto_threshold=float(os.getenv("ML_EXIT_VETO_THRESHOLD", "0.65")),
        ml_exit_min_hold_sec=int(os.getenv("ML_EXIT_MIN_HOLD_SEC", "20")),
        ml_exit_min_hold_sec_sniper=int(os.getenv("ML_EXIT_MIN_HOLD_SEC_SNIPER", "10")),
        enable_main_strategy=_bool_env("ENABLE_MAIN_STRATEGY", True),
        enable_sniper_strategy=_bool_env("ENABLE_SNIPER_STRATEGY", False),
        sniper_position_sol=float(os.getenv("SNIPER_POSITION_SOL", "0.015")),
        sniper_max_open_positions=int(os.getenv("SNIPER_MAX_OPEN_POSITIONS", "2")),
        sniper_max_exposure_sol=float(os.getenv("SNIPER_MAX_EXPOSURE_SOL", "0.06")),
        sniper_min_token_age_sec=float(os.getenv("SNIPER_MIN_TOKEN_AGE_SEC", "45")),
        sniper_max_token_age_sec=float(os.getenv("SNIPER_MAX_TOKEN_AGE_SEC", "180")),
        sniper_min_cluster_30s=int(os.getenv("SNIPER_MIN_CLUSTER_30S", "4")),
        sniper_min_tx_count_30s=int(os.getenv("SNIPER_MIN_TX_COUNT_30S", "4")),
        sniper_min_volume_sol_30s=float(os.getenv("SNIPER_MIN_VOLUME_SOL_30S", "5.0")),
        sniper_min_price_change_30s=float(os.getenv("SNIPER_MIN_PRICE_CHANGE_30S", "0.05")),
        sniper_max_price_change_30s=float(os.getenv("SNIPER_MAX_PRICE_CHANGE_30S", "0.60")),
        sniper_token_cooldown_sec=int(os.getenv("SNIPER_TOKEN_COOLDOWN_SEC", "120")),
        sniper_take_profit_pnl=float(os.getenv("SNIPER_TAKE_PROFIT_PNL", "0.08")),
        sniper_tp_min_gross_sol_floor=float(os.getenv("SNIPER_TP_MIN_GROSS_SOL_FLOOR", "0.003")),
        sniper_tp_min_gross_fee_multiplier=float(
            os.getenv("SNIPER_TP_MIN_GROSS_FEE_MULTIPLIER", "2.5")
        ),
        sniper_tp_min_gross_size_ratio=float(os.getenv("SNIPER_TP_MIN_GROSS_SIZE_RATIO", "0.015")),
        sniper_stop_pnl=float(os.getenv("SNIPER_STOP_PNL", "-0.10")),
        sniper_max_hold_sec=int(os.getenv("SNIPER_MAX_HOLD_SEC", "75")),
        sniper_tp_confirm_ticks=int(os.getenv("SNIPER_TP_CONFIRM_TICKS", "1")),
        sniper_stop_confirm_ticks=int(os.getenv("SNIPER_STOP_CONFIRM_TICKS", "2")),
        sniper_stop_min_hold_sec=int(os.getenv("SNIPER_STOP_MIN_HOLD_SEC", "3")),
        sniper_min_volume_per_tx_sol_30s=float(
            os.getenv("SNIPER_MIN_VOLUME_PER_TX_SOL_30S", "0.35")
        ),
        sniper_use_runtime_rules=_bool_env("SNIPER_USE_RUNTIME_RULES", False),
        sniper_rule_ids=_csv_env("SNIPER_RULE_IDS", ""),
        sniper_allowed_sources=_csv_env("SNIPER_ALLOWED_SOURCES", ""),
        main_rule_ids=_csv_env("MAIN_RULE_IDS", ""),
        main_allowed_sources=_csv_env("MAIN_ALLOWED_SOURCES", ""),
        main_min_token_age_sec=float(os.getenv("MAIN_MIN_TOKEN_AGE_SEC", "300")),
        main_max_token_age_sec=float(os.getenv("MAIN_MAX_TOKEN_AGE_SEC", "3600")),
        sniper_tp_jupiter_verify=_bool_env("SNIPER_TP_JUPITER_VERIFY", True),
        sniper_tp_live_bypass_multiplier=float(os.getenv("SNIPER_TP_LIVE_BYPASS_MULT", "2.0")),
        enable_wallet_strategy=_bool_env("ENABLE_WALLET_STRATEGY", False),
        wallet_position_sol=float(os.getenv("WALLET_POSITION_SOL", "0.05")),
        wallet_max_open_positions=int(os.getenv("WALLET_MAX_OPEN_POSITIONS", "3")),
        wallet_max_exposure_sol=float(os.getenv("WALLET_MAX_EXPOSURE_SOL", "0.20")),
        wallet_min_cluster_300s=int(os.getenv("WALLET_MIN_CLUSTER_300S", "2")),
        wallet_min_buys_90s=int(os.getenv("WALLET_MIN_BUYS_90S", "3")),
        wallet_min_wallet_score_sum=float(os.getenv("WALLET_MIN_WALLET_SCORE_SUM", "0.0")),
        wallet_min_token_age_sec=float(os.getenv("WALLET_MIN_TOKEN_AGE_SEC", "0")),
        wallet_max_token_age_sec=float(os.getenv("WALLET_MAX_TOKEN_AGE_SEC", "600")),
        wallet_max_price_change_30s=float(os.getenv("WALLET_MAX_PRICE_CHANGE_30S", "1.0")),
        wallet_min_price_change_60s=float(os.getenv("WALLET_MIN_PRICE_CHANGE_60S", "-0.30")),
        wallet_min_net_flow_sol_60s=float(os.getenv("WALLET_MIN_NET_FLOW_SOL_60S", "0.0")),
        wallet_token_cooldown_sec=int(os.getenv("WALLET_TOKEN_COOLDOWN_SEC", "60")),
        wallet_take_profit_pnl=float(os.getenv("WALLET_TAKE_PROFIT_PNL", "0.50")),
        wallet_stop_pnl=float(os.getenv("WALLET_STOP_PNL", "-0.25")),
        wallet_max_hold_sec=int(os.getenv("WALLET_MAX_HOLD_SEC", "900")),
        wallet_trailing_drawdown=float(os.getenv("WALLET_TRAILING_DRAWDOWN", "0.08")),
        wallet_trailing_arm_confirm_ticks=max(
            1, int(os.getenv("WALLET_TRAILING_ARM_CONFIRM_TICKS", "2"))
        ),
        wallet_trailing_exit_confirm_ticks=max(
            1, int(os.getenv("WALLET_TRAILING_EXIT_CONFIRM_TICKS", "2"))
        ),
        wallet_tp1_peak=float(os.getenv("WALLET_TP1_PEAK", "0.30")),
        wallet_tp1_sell_fraction=max(
            0.0, min(float(os.getenv("WALLET_TP1_SELL_FRACTION", "0.5")), 1.0)
        ),
        wallet_tp1_confirm_ticks=max(1, int(os.getenv("WALLET_TP1_CONFIRM_TICKS", "2"))),
        wallet_allowed_sources=_csv_env("WALLET_ALLOWED_SOURCES", ""),
        wallet_copytrading_enabled=_bool_env("WALLET_COPYTRADING_ENABLED", False),
        wallet_copy_position_sol=float(os.getenv("WALLET_COPY_POSITION_SOL", "0.15")),
        wallet_copy_min_wallet_score=float(os.getenv("WALLET_COPY_MIN_WALLET_SCORE", "0.0")),
        wallet_copy_max_token_age_sec=float(os.getenv("WALLET_COPY_MAX_TOKEN_AGE_SEC", "1800")),
        wallet_copy_event_max_age_sec=float(os.getenv("WALLET_COPY_EVENT_MAX_AGE_SEC", "20")),
        wallet_copy_ml_bypass=_bool_env("WALLET_COPY_ML_BYPASS", True),
        wallet_copy_disable_sniper=_bool_env("WALLET_COPY_DISABLE_SNIPER", True),
        wallet_copy_trail_arm_pnl=float(os.getenv("WALLET_COPY_TRAIL_ARM_PNL", "0.05")),
        wallet_copy_trail_drawdown=float(os.getenv("WALLET_COPY_TRAIL_DRAWDOWN", "0.15")),
        wallet_copy_hard_stop_pnl=float(os.getenv("WALLET_COPY_HARD_STOP_PNL", "-0.25")),
        wallet_copy_max_hold_sec=int(os.getenv("WALLET_COPY_MAX_HOLD_SEC", "1800")),
        wallet_copy_mirror_sell=_bool_env("WALLET_COPY_MIRROR_SELL", True),
        wallet_copy_mirror_sell_profit_threshold=float(
            os.getenv("WALLET_COPY_MIRROR_SELL_PROFIT_THRESHOLD", "0.09")
        ),
        wallet_copy_tp1_peak=float(os.getenv("WALLET_COPY_TP1_PEAK", "0.15")),
        wallet_copy_tp1_sell_fraction=max(
            0.0, min(float(os.getenv("WALLET_COPY_TP1_SELL_FRACTION", "0.5")), 1.0)
        ),
        wallet_copy_tp1_confirm_ticks=max(1, int(os.getenv("WALLET_COPY_TP1_CONFIRM_TICKS", "2"))),
        wallet_copy_exit_confirm_ticks=max(
            1, int(os.getenv("WALLET_COPY_EXIT_CONFIRM_TICKS", "2"))
        ),
        wallet_copy_trail_min_floor=float(os.getenv("WALLET_COPY_TRAIL_MIN_FLOOR", "0.02")),
        wallet_copy_score_penalty_on_loss=float(
            os.getenv("WALLET_COPY_SCORE_PENALTY_ON_LOSS", "10.0")
        ),
        wallet_copy_loss_pnl_threshold=float(os.getenv("WALLET_COPY_LOSS_PNL_THRESHOLD", "-0.02")),
        entry_score_tracked_wallet_presence_bonus=float(
            os.getenv("ENTRY_SCORE_TRACKED_WALLET_PRESENCE_BONUS", "0.12")
        ),
        entry_score_tracked_wallet_count_weight=float(
            os.getenv("ENTRY_SCORE_TRACKED_WALLET_COUNT_WEIGHT", "0.08")
        ),
        entry_score_tracked_wallet_count_scale=float(
            os.getenv("ENTRY_SCORE_TRACKED_WALLET_COUNT_SCALE", "3.0")
        ),
        entry_score_tracked_wallet_score_weight=float(
            os.getenv("ENTRY_SCORE_TRACKED_WALLET_SCORE_WEIGHT", "0.06")
        ),
        entry_score_tracked_wallet_score_scale=float(
            os.getenv("ENTRY_SCORE_TRACKED_WALLET_SCORE_SCALE", "400.0")
        ),
        regime_size_multiplier_negative_shock_recovery=float(
            os.getenv("REGIME_SIZE_MULTIPLIER_NEGATIVE_SHOCK_RECOVERY", "1.0")
        ),
        regime_size_multiplier_high_cluster_recovery=float(
            os.getenv("REGIME_SIZE_MULTIPLIER_HIGH_CLUSTER_RECOVERY", "0.9")
        ),
        regime_size_multiplier_momentum_burst=float(
            os.getenv("REGIME_SIZE_MULTIPLIER_MOMENTUM_BURST", "0.65")
        ),
        regime_size_multiplier_unknown=float(os.getenv("REGIME_SIZE_MULTIPLIER_UNKNOWN", "0.6")),
        tracked_wallet_size_boost_per_wallet=float(
            os.getenv("TRACKED_WALLET_SIZE_BOOST_PER_WALLET", "0.03")
        ),
        tracked_wallet_size_boost_cap=float(os.getenv("TRACKED_WALLET_SIZE_BOOST_CAP", "0.15")),
        exit_tp1_multiple=float(os.getenv("EXIT_TP1_MULTIPLE", "1.0")),
        exit_tp2_multiple=float(os.getenv("EXIT_TP2_MULTIPLE", "3.0")),
        exit_tp3_multiple=float(os.getenv("EXIT_TP3_MULTIPLE", "9.0")),
        exit_tp1_sell_fraction=float(os.getenv("EXIT_TP1_SELL_FRACTION", "0.5")),
        exit_tp2_sell_fraction=float(os.getenv("EXIT_TP2_SELL_FRACTION", "0.3")),
        post_tp1_stop_pnl=float(os.getenv("POST_TP1_STOP_PNL", "0.02")),
        exit_rule_stop_overrides=_rule_stop_overrides_env("EXIT_RULE_STOP_OVERRIDES", ""),
        post_tp2_trailing_drawdown=float(os.getenv("POST_TP2_TRAILING_DRAWDOWN", "0.25")),
        post_tp2_timeout_sec=int(os.getenv("POST_TP2_TIMEOUT_SEC", "300")),
        tp1_confirm_ticks=int(os.getenv("TP1_CONFIRM_TICKS", "2")),
        tp2_confirm_ticks=int(os.getenv("TP2_CONFIRM_TICKS", "2")),
        tp1_min_volume_sol_30s=float(os.getenv("TP1_MIN_VOLUME_SOL_30S", "2.0")),
        tp2_min_volume_sol_30s=float(os.getenv("TP2_MIN_VOLUME_SOL_30S", "2.0")),
        tp3_confirm_ticks=int(os.getenv("TP3_CONFIRM_TICKS", "2")),
        tp2_fast_confirm_ticks=int(os.getenv("TP2_FAST_CONFIRM_TICKS", "2")),
        tp2_fast_min_volume_sol_30s=float(os.getenv("TP2_FAST_MIN_VOLUME_SOL_30S", "2.0")),
        tp3_min_volume_sol_30s=float(os.getenv("TP3_MIN_VOLUME_SOL_30S", "2.0")),
        exit_price_max_step_multiple=float(os.getenv("EXIT_PRICE_MAX_STEP_MULTIPLE", "2.5")),
        exit_outlier_max_pnl_jump=float(os.getenv("EXIT_OUTLIER_MAX_PNL_JUMP", "20.0")),
        exit_outlier_low_volume_sol_30s=float(os.getenv("EXIT_OUTLIER_LOW_VOLUME_SOL_30S", "2.0")),
        exit_max_peak_pnl_multiple=float(os.getenv("EXIT_MAX_PEAK_PNL_MULTIPLE", "9.0")),
        price_outlier_min_samples=int(os.getenv("PRICE_OUTLIER_MIN_SAMPLES", "8")),
        price_outlier_median_window=int(os.getenv("PRICE_OUTLIER_MEDIAN_WINDOW", "25")),
        price_outlier_max_multiple=float(os.getenv("PRICE_OUTLIER_MAX_MULTIPLE", "15.0")),
        price_outlier_confirm_signatures=int(os.getenv("PRICE_OUTLIER_CONFIRM_SIGNATURES", "2")),
        price_outlier_confirm_window_sec=int(os.getenv("PRICE_OUTLIER_CONFIRM_WINDOW_SEC", "12")),
        price_outlier_confirm_tolerance=float(os.getenv("PRICE_OUTLIER_CONFIRM_TOLERANCE", "0.35")),
        stage0_loss_timeout_sec=int(os.getenv("STAGE0_LOSS_TIMEOUT_SEC", "900")),
        stage0_loss_timeout_max_pnl=float(os.getenv("STAGE0_LOSS_TIMEOUT_MAX_PNL", "0.0")),
        stage0_early_profit_window_sec=int(os.getenv("STAGE0_EARLY_PROFIT_WINDOW_SEC", "120")),
        stage0_early_profit_min_pnl=float(os.getenv("STAGE0_EARLY_PROFIT_MIN_PNL", "1.0")),
        stage0_early_profit_max_pnl=float(os.getenv("STAGE0_EARLY_PROFIT_MAX_PNL", "1.75")),
        stage0_early_profit_confirm_ticks=int(os.getenv("STAGE0_EARLY_PROFIT_CONFIRM_TICKS", "2")),
        stage0_early_profit_sell_fraction=float(
            os.getenv("STAGE0_EARLY_PROFIT_SELL_FRACTION", "0.30")
        ),
        stage0_crash_guard_window_sec=int(os.getenv("STAGE0_CRASH_GUARD_WINDOW_SEC", "120")),
        stage0_crash_guard_min_pnl=float(os.getenv("STAGE0_CRASH_GUARD_MIN_PNL", "-0.20")),
        stage0_crash_guard_min_hold_sec=int(os.getenv("STAGE0_CRASH_GUARD_MIN_HOLD_SEC", "20")),
        stage0_crash_guard_confirm_ticks=int(os.getenv("STAGE0_CRASH_GUARD_CONFIRM_TICKS", "2")),
        pre_tp1_retrace_lock_min_hold_sec=int(os.getenv("PRE_TP1_RETRACE_LOCK_MIN_HOLD_SEC", "20")),
        pre_tp1_retrace_lock_arm_pnl=float(os.getenv("PRE_TP1_RETRACE_LOCK_ARM_PNL", "0.35")),
        pre_tp1_retrace_lock_drawdown=float(os.getenv("PRE_TP1_RETRACE_LOCK_DRAWDOWN", "0.30")),
        pre_tp1_retrace_lock_floor_pnl=float(os.getenv("PRE_TP1_RETRACE_LOCK_FLOOR_PNL", "0.08")),
        pre_tp1_retrace_lock_confirm_ticks=int(
            os.getenv("PRE_TP1_RETRACE_LOCK_CONFIRM_TICKS", "2")
        ),
        stage0_fast_fail_non_positive_sec=int(os.getenv("STAGE0_FAST_FAIL_NON_POSITIVE_SEC", "0")),
        stage0_fast_fail_non_positive_max_pnl=float(
            os.getenv("STAGE0_FAST_FAIL_NON_POSITIVE_MAX_PNL", "0.0")
        ),
        stage0_fast_fail_under_profit_sec=int(os.getenv("STAGE0_FAST_FAIL_UNDER_PROFIT_SEC", "0")),
        stage0_fast_fail_under_profit_min_pnl=float(
            os.getenv("STAGE0_FAST_FAIL_UNDER_PROFIT_MIN_PNL", "0.10")
        ),
        stage0_moderate_positive_timeout_sec=int(
            os.getenv("STAGE0_MODERATE_POSITIVE_TIMEOUT_SEC", "300")
        ),
        stage0_moderate_positive_min_pnl=float(
            os.getenv("STAGE0_MODERATE_POSITIVE_MIN_PNL", "0.02")
        ),
        stage0_moderate_positive_max_pnl=float(
            os.getenv("STAGE0_MODERATE_POSITIVE_MAX_PNL", "0.99")
        ),
        stage0_moderate_positive_skip_profiles=_csv_env(
            "STAGE0_MODERATE_POSITIVE_SKIP_PROFILES", "mature_pair_v1"
        ),
        stage1_low_positive_timeout_sec=int(os.getenv("STAGE1_LOW_POSITIVE_TIMEOUT_SEC", "900")),
        stage1_low_positive_min_pnl=float(os.getenv("STAGE1_LOW_POSITIVE_MIN_PNL", "1.0")),
        stage1_low_positive_max_pnl=float(os.getenv("STAGE1_LOW_POSITIVE_MAX_PNL", "1.9")),
        stage1_sub2x_timeout_sec=int(os.getenv("STAGE1_SUB2X_TIMEOUT_SEC", "1200")),
        stage1_sub2x_min_pnl=float(os.getenv("STAGE1_SUB2X_MIN_PNL", "0.02")),
        stage1_sub2x_max_pnl=float(os.getenv("STAGE1_SUB2X_MAX_PNL", "0.99")),
        absolute_max_hold_sec=int(os.getenv("ABSOLUTE_MAX_HOLD_SEC", "3600")),
        stale_sweep_sec=int(os.getenv("STALE_SWEEP_SEC", "10")),
        dead_token_hold_sec=int(os.getenv("DEAD_TOKEN_HOLD_SEC", "180")),
        dead_token_move_pct=float(os.getenv("DEAD_TOKEN_MOVE_PCT", "0.02")),
        dead_token_confirm_ticks=int(os.getenv("DEAD_TOKEN_CONFIRM_TICKS", "2")),
        dead_token_max_tx_count_60s=int(os.getenv("DEAD_TOKEN_MAX_TX_COUNT_60S", "3")),
        dead_token_max_volume_sol_60s=float(os.getenv("DEAD_TOKEN_MAX_VOLUME_SOL_60S", "3.0")),
        bot_db_path=root / "data" / "live" / "bot_state.db",
        event_log_path=root / "data" / "live" / "events.jsonl",
        event_log_throttle_window_sec=float(os.getenv("EVENT_LOG_THROTTLE_WINDOW_SEC", "15")),
        event_log_throttled_event_types=_csv_env(
            "EVENT_LOG_THROTTLED_EVENT_TYPES",
            "entry_rejected,sniper_entry_rejected,candidate_deferred",
        ),
        bot_status_path=root / "data" / "live" / "bot_status.json",
        market_regime_enabled=_bool_env("MARKET_REGIME_ENABLED", True),
        market_regime_gate_active=_bool_env("MARKET_REGIME_GATE_ACTIVE", False),
        market_regime_win_rate_window=int(os.getenv("MARKET_REGIME_WIN_RATE_WINDOW", "15")),
        market_regime_min_win_rate=float(os.getenv("MARKET_REGIME_MIN_WIN_RATE", "0.25")),
        market_regime_bootstrap_positions=int(os.getenv("MARKET_REGIME_BOOTSTRAP_POSITIONS", "5")),
        market_regime_min_candidates_5min=int(os.getenv("MARKET_REGIME_MIN_CANDIDATES_5MIN", "0")),
        market_regime_sol_enabled=_bool_env("MARKET_REGIME_SOL_ENABLED", False),
        market_regime_sol_drop_threshold=float(
            os.getenv("MARKET_REGIME_SOL_DROP_THRESHOLD", "-0.05")
        ),
        market_regime_pause_cooldown_sec=int(os.getenv("MARKET_REGIME_PAUSE_COOLDOWN_SEC", "300")),
    )
