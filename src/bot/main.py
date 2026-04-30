"""CLI entrypoint for the Phase 2 trading bot (paper or live)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone

from src.bot.config import BotConfig, load_bot_config
from src.bot.runner import BotRunner
from src.dashboard.main import start_dashboard_thread
from src.storage.bot_db import BotDB
from src.storage.event_log import EventLogger
from src.utils.logging_utils import configure_logging


def validate_startup(config: BotConfig, paper: bool) -> None:
    """Fail fast for obviously invalid startup configuration."""
    if not config.helius_api_key:
        raise SystemExit("HELIUS_API_KEY is required.")
    tracked_wallets_required = config.discovery_mode != "pair_first"
    if tracked_wallets_required and not config.tracked_wallets_path.exists():
        raise SystemExit(f"Tracked wallets file missing: {config.tracked_wallets_path}")
    if config.rules_source_mode == "pump":
        if not config.pump_rules_path.exists():
            raise SystemExit(
                f"RULES_SOURCE_MODE=pump but PUMP_RULES_PATH is missing: {config.pump_rules_path}"
            )
    elif config.rules_source_mode == "legacy":
        if not config.trusted_rules_path.exists() and not config.top_rules_path.exists():
            raise SystemExit("RULES_SOURCE_MODE=legacy but no legacy rule artifact found.")
    else:
        if (
            not config.pump_rules_path.exists()
            and not config.trusted_rules_path.exists()
            and not config.top_rules_path.exists()
        ):
            raise SystemExit(
                "No runtime rule artifact found. Provide PUMP_RULES_PATH or legacy TRUSTED_RULES_PATH/TOP_RULES_PATH."
            )
    if config.discovery_mode == "pair_first" and not config.discovery_account_include:
        raise SystemExit("DISCOVERY_MODE=pair_first requires DISCOVERY_ACCOUNT_INCLUDE to be set.")
    if config.discovery_mode == "pair_first":
        if not config.chainstack_grpc_endpoint:
            raise SystemExit("DISCOVERY_MODE=pair_first requires CHAINSTACK_GRPC_ENDPOINT.")
        if not config.chainstack_grpc_token:
            raise SystemExit("DISCOVERY_MODE=pair_first requires CHAINSTACK_GRPC_TOKEN.")

    # ---- Live-mode pre-flight ----
    if config.enable_auto_trading and not paper:
        missing: list[str] = []
        if not config.bot_private_key_b58:
            missing.append("BOT_PRIVATE_KEY_B58")
        if not config.helius_rpc_url:
            missing.append("HELIUS_RPC_URL")
        if not config.jupiter_base_url:
            missing.append("JUPITER_BASE_URL")
        if missing:
            raise SystemExit(
                f"ENABLE_AUTO_TRADING=true but required env vars are missing: {', '.join(missing)}. "
                "Set them or switch back to paper mode."
            )


def _log_live_readiness(
    logger: logging.Logger, config: BotConfig, *, live_requested: bool, paper: bool
) -> None:
    """Emit a startup readiness report for live-mode troubleshooting."""
    if not live_requested:
        return
    logger.warning(
        "Live request summary: cli_live=%s enable_auto_trading=%s enable_paper_trading=%s effective_mode=%s",
        live_requested,
        config.enable_auto_trading,
        config.enable_paper_trading,
        "PAPER" if paper else "LIVE",
    )
    logger.warning(
        "Live env presence: BOT_PRIVATE_KEY_B58=%s HELIUS_RPC_URL=%s JUPITER_BASE_URL=%s HELIUS_API_KEY=%s",
        bool(config.bot_private_key_b58),
        bool(config.helius_rpc_url),
        bool(config.jupiter_base_url),
        bool(config.helius_api_key),
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 2 Solana memecoin trading bot (paper + live)."
    )
    parser.add_argument("--paper", action="store_true", help="Run in paper trading mode (default).")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live trading mode (requires ENABLE_AUTO_TRADING=true).",
    )
    parser.add_argument("--rules", type=str, default=None, help="Optional rule file override path.")
    parser.add_argument(
        "--allowed-regimes",
        type=str,
        default=None,
        help="Comma-separated regime allowlist.",
    )
    parser.add_argument(
        "--limit-wallets",
        type=int,
        default=None,
        help="Limit number of tracked wallets.",
    )
    parser.add_argument(
        "--with-dashboard",
        action="store_true",
        help="Start the local dashboard (FastAPI + Jinja2) in the same process.",
    )
    parser.add_argument(
        "--dashboard-host", type=str, default=os.getenv("DASHBOARD_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--dashboard-port", type=int, default=int(os.getenv("DASHBOARD_PORT", "8787"))
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    config = load_bot_config()
    if args.allowed_regimes:
        object.__setattr__(
            config,
            "optional_allowed_regimes",
            tuple([item.strip() for item in args.allowed_regimes.split(",") if item.strip()]),
        )
    if args.rules:
        override_path = config.app.root_dir / args.rules
        object.__setattr__(config, "pump_rules_path", override_path)
        object.__setattr__(config, "trusted_rules_path", override_path)

    logger = configure_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        logger_name=__name__,
        force=True,
    )

    # Determine effective paper flag:
    #   --paper is the default; --live overrides only if ENABLE_AUTO_TRADING=true
    paper = True
    if args.live and config.enable_auto_trading:
        paper = False
    elif args.live and not config.enable_auto_trading:
        logger.warning(
            "--live flag ignored because ENABLE_AUTO_TRADING is not true. Running in PAPER mode."
        )

    _log_live_readiness(logger, config, live_requested=args.live, paper=paper)

    validate_startup(config, paper=paper)

    # If paper flag is set, force-disable auto trading in config to guarantee
    # the TradeExecutor stays in paper mode regardless of env.
    if paper:
        object.__setattr__(config, "enable_auto_trading", False)

    db = BotDB(config.bot_db_path)
    event_log = EventLogger(
        config.event_log_path,
        throttle_window_sec=config.event_log_throttle_window_sec,
        throttled_event_types=config.event_log_throttled_event_types,
    )
    runner = BotRunner(config, db, event_log, limit_wallets=args.limit_wallets)

    mode = runner._mode_label
    # Tag every subsequent event with mode + session_id so live audit logs can
    # be filtered cleanly. session_id lets us segregate runs in the same file.
    event_log.set_context(
        mode=mode.lower(),
        session_id=datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    logger.info("Loaded %s bot with %s tracked wallets", mode, len(runner.stream.wallet_df))
    if args.live and mode != "LIVE":
        logger.error("Live mode fallback: bot is running in PAPER mode.")
        reason = runner.trade_executor.fallback_reason or "unknown"
        logger.error("Fallback reason: %s", reason)
        diagnostics = runner.trade_executor.init_diagnostics
        if diagnostics:
            for item in diagnostics:
                logger.error("Live diagnostic: %s", item)
        else:
            logger.error("Live diagnostic: no detailed diagnostics were captured.")
    if mode == "LIVE":
        logger.warning("⚠️  LIVE MODE – real funds will be used for trading!")
    dashboard_server = None
    if args.with_dashboard:
        refresh_sec = int(os.getenv("DASHBOARD_REFRESH_SEC", "5"))
        dashboard_server, _ = start_dashboard_thread(
            host=args.dashboard_host,
            port=args.dashboard_port,
            refresh_sec=refresh_sec,
            logger=logger,
            controller=runner,
        )
    try:
        asyncio.run(runner.run_forever())
    finally:
        event_log.close()
        if dashboard_server is not None:
            dashboard_server.server_close()


if __name__ == "__main__":
    main()
