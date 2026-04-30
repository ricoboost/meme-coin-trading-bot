"""[TEMPLATE — won't run as-is. See docs/COLLECTORS.md.

This file shows the wallet-pool input schema and a sample fetch shape.
The actual scrape/API call has been replaced with placeholders so the
data flow is intact but no specific vendor is named. Plug in your own
data source by rewriting the fetch function — the rest of the pipeline
(refresh_wallet_pool, score_wallets_helius) consumes the parquet output
unchanged.]

Scrape Provider A leaderboard wallets and build the wallet pool.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from playwright.sync_api import Locator, Page, sync_playwright

from src.utils.io import dataset_path, load_app_config, write_parquet
from src.utils.logging_utils import configure_logging
from src.utils.time_utils import utcnow


BASE58_RE = re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b")
TAB_CONTAINER_SELECTOR = "div[class*='leaderboard_timeFilterContainer']"
SELECTED_TAB_SELECTOR = "p[class*='leaderboard_selected']"
ACCOUNT_LINK_SELECTOR = "a[href*='/account/']"


def wallet_from_account_href(href: str | None) -> str | None:
    """Extract a wallet address from a Provider A account link."""
    if not href:
        return None
    path = urlparse(href).path.strip("/")
    if not path.startswith("account/"):
        return None
    wallet = path.split("/", 1)[1]
    if BASE58_RE.fullmatch(wallet):
        return wallet
    return None


def extract_wallet_rows(
    page: Page, timeframe: str, captured_at: datetime, top_n: int
) -> list[dict[str, Any]]:
    """Extract ranked wallet rows from account links on the active leaderboard tab."""
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    hrefs = page.locator(ACCOUNT_LINK_SELECTOR).evaluate_all(
        "(links) => links.map((link) => link.href)"
    )
    for href in hrefs:
        wallet = wallet_from_account_href(href)
        if not wallet:
            continue
        if wallet in seen:
            continue
        seen.add(wallet)
        rows.append(
            {
                "wallet": wallet,
                "rank": len(rows) + 1,
                "timeframe": timeframe,
                "pnl_text": None,
                "collected_at": captured_at,
            }
        )
        if len(rows) >= top_n:
            break
    return rows


def tab_locator(page: Page, timeframe: str) -> Locator:
    """Return the locator for a leaderboard timeframe tab."""
    label = timeframe.capitalize()
    container = page.locator(TAB_CONTAINER_SELECTOR).first
    return container.get_by_text(label, exact=True)


def wait_for_tab_state(page: Page, timeframe: str, previous_text: str | None = None) -> None:
    """Wait until the requested tab is selected and the content has had a chance to refresh."""
    label = timeframe.capitalize()
    page.locator(SELECTED_TAB_SELECTOR).filter(has_text=label).first.wait_for(timeout=15_000)
    if previous_text:
        try:
            page.wait_for_function(
                """([selector, text]) => {
                    const body = document.querySelector(selector);
                    return !!body && body.innerText !== text;
                }""",
                arg=["body", previous_text],
                timeout=5_000,
            )
        except Exception:
            # The selected class check is the main gate. Body text does not always change immediately.
            pass
    page.wait_for_timeout(2_500)


def scrape_all_timeframes(page: Page, top_n: int, raw_dir: Path, logger) -> list[dict[str, Any]]:
    """Scrape all leaderboard timeframes from a single interactive Provider A session."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    previous_text: str | None = None

    for timeframe in ["daily", "weekly", "monthly"]:
        captured_at = utcnow()
        tab = tab_locator(page, timeframe)
        tab.wait_for(timeout=20_000)
        tab.click()
        wait_for_tab_state(page, timeframe, previous_text)
        content = page.content()
        text = page.locator("body").inner_text()
        previous_text = text

        rows = extract_wallet_rows(page, timeframe, captured_at, top_n)
        (raw_dir / f"{timeframe}.html").write_text(content, encoding="utf-8")
        (raw_dir / f"{timeframe}.json").write_text(
            json.dumps(rows, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Scraped %s wallets for %s", len(rows), timeframe)
        all_rows.extend(rows)

    return all_rows


def build_wallet_pool(
    provider_a_df: pd.DataFrame, wallets_config: dict[str, Any], top_n: int
) -> pd.DataFrame:
    """Build wallet pool scores from timeframe rankings and optional manual tags."""
    if provider_a_df.empty:
        return pd.DataFrame(
            columns=[
                "wallet",
                "score",
                "appears_daily",
                "appears_weekly",
                "appears_monthly",
                "best_rank",
                "tags",
            ]
        )

    weights = {"daily": 3, "weekly": 2, "monthly": 1}
    work = provider_a_df.copy()
    work["weight"] = work["timeframe"].map(weights).fillna(0)
    work["score_component"] = (top_n + 1 - work["rank"]) * work["weight"]

    grouped = (
        work.groupby("wallet", as_index=False)
        .agg(
            score=("score_component", "sum"),
            appears_daily=("timeframe", lambda s: int((s == "daily").any())),
            appears_weekly=("timeframe", lambda s: int((s == "weekly").any())),
            appears_monthly=("timeframe", lambda s: int((s == "monthly").any())),
            best_rank=("rank", "min"),
        )
        .sort_values(["score", "best_rank"], ascending=[False, True])
    )

    manual_include = set(wallets_config.get("manual_include", []))
    manual_exclude = set(wallets_config.get("manual_exclude", []))
    tag_map = wallets_config.get("wallet_tags", {})

    if manual_include:
        existing = set(grouped["wallet"].tolist())
        include_rows = [
            {
                "wallet": wallet,
                "score": 0,
                "appears_daily": 0,
                "appears_weekly": 0,
                "appears_monthly": 0,
                "best_rank": None,
            }
            for wallet in manual_include
            if wallet not in existing
        ]
        if include_rows:
            grouped = pd.concat([grouped, pd.DataFrame(include_rows)], ignore_index=True)

    grouped = grouped[~grouped["wallet"].isin(manual_exclude)].copy()
    grouped["tags"] = grouped["wallet"].map(lambda wallet: tag_map.get(wallet, []))
    return grouped.reset_index(drop=True)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Collect top Provider A wallets.")
    parser.add_argument("--top-n", type=int, default=None, help="Number of wallets per timeframe.")
    parser.add_argument(
        "--headed", action="store_true", help="Run Playwright with a visible browser."
    )
    parser.add_argument(
        "--slow-mo-ms",
        type=int,
        default=0,
        help="Optional Playwright slow motion in milliseconds.",
    )
    args = parser.parse_args()

    config = load_app_config()
    logger = configure_logging(logger_name=__name__)
    top_n = args.top_n or config.env["PROVIDER_A_TOP_N"]
    raw_dir = config.paths["raw_provider_a_dir"]
    start_url = next(iter(config.settings["provider_a"]["base_urls"].values()))

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed, slow_mo=args.slow_mo_ms)
        page = browser.new_page()
        page.goto(start_url, wait_until="domcontentloaded", timeout=60_000)
        page.locator(TAB_CONTAINER_SELECTOR).first.wait_for(timeout=20_000)
        page.wait_for_timeout(3_000)
        all_rows = scrape_all_timeframes(page, top_n, raw_dir, logger)
        browser.close()

    provider_a_df = pd.DataFrame(all_rows)
    write_parquet(provider_a_df, dataset_path(config, "bronze", "provider_a_wallets.parquet"))
    logger.info("Saved Provider A raw dataset")
    # wallet_pool.parquet is now owned by src.collectors.refresh_wallet_pool,
    # which unions provider_a + helius_token_buyers + manual imports, applies a
    # Helius-backed PnL filter, and writes the final pool. Run:
    #   .venv/bin/python3 -m src.collectors.refresh_wallet_pool


if __name__ == "__main__":
    main()
