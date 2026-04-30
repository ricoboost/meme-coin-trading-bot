"""Run the local monitoring dashboard."""

from __future__ import annotations

import argparse
import logging
import os
import threading

from src.bot.config import load_bot_config
from src.dashboard.data import DashboardDataStore, DashboardPaths
from src.dashboard.server import DashboardHttpServer
from src.utils.logging_utils import configure_logging


def build_dashboard_server(
    host: str, port: int, refresh_sec: int, controller: object | None = None
) -> tuple[DashboardHttpServer, str]:
    """Create a configured dashboard server and its display URL."""
    bot_config = load_bot_config()
    store = DashboardDataStore(
        DashboardPaths(
            db_path=bot_config.bot_db_path,
            event_log_path=bot_config.event_log_path,
            status_path=bot_config.bot_status_path,
        )
    )
    server = DashboardHttpServer(
        (host, port), store=store, refresh_sec=refresh_sec, controller=controller
    )
    return server, f"http://{host}:{port}"


def start_dashboard_thread(
    host: str,
    port: int,
    refresh_sec: int,
    logger: logging.Logger,
    controller: object | None = None,
) -> tuple[DashboardHttpServer, threading.Thread]:
    """Start the dashboard server in a daemon thread."""
    server, url = build_dashboard_server(
        host=host, port=port, refresh_sec=refresh_sec, controller=controller
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="dashboard-server")
    thread.start()
    logger.info("Dashboard listening on %s", url)
    return server, thread


def main() -> None:
    """Dashboard CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Local dashboard for the memybot runtime.")
    parser.add_argument("--host", type=str, default=os.getenv("DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("DASHBOARD_PORT", "8787")))
    parser.add_argument(
        "--refresh-sec", type=int, default=int(os.getenv("DASHBOARD_REFRESH_SEC", "5"))
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logger = configure_logging(
        level=logging.DEBUG if args.verbose else logging.INFO, logger_name=__name__
    )
    bot_config = load_bot_config()
    server, url = build_dashboard_server(
        host=args.host, port=args.port, refresh_sec=args.refresh_sec, controller=None
    )
    logger.info("Dashboard listening on %s", url)
    logger.info("Using db=%s", bot_config.bot_db_path)
    logger.info("Using events=%s", bot_config.event_log_path)
    logger.info("Using status=%s", bot_config.bot_status_path)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Dashboard stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
