"""End-to-end wallet pool refresh.

1. Union all bronze wallet-source parquets (provider_a, helius token buyers,
   manual imports) into a single candidate list tagged with sources.
2. Merge with silver/wallet_scores.parquet (produced by score_wallets_helius).
3. Apply filters: 30d PnL floor, 30d win-rate floor, 30d trade-count floor,
   7d activity requirement.
4. Rank by composite score and write the top N to
   data/bronze/wallet_pool.parquet (consumed by wallet_stream.py).

Preserves the legacy schema (wallet, score, appears_daily, appears_weekly,
appears_monthly, best_rank, tags) so all existing loaders keep working.
Appends new columns (sources, realized_pnl_30d, win_rate_30d, n_trades_30d,
last_active_ts) for post-hoc inspection.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import dataset_path, load_app_config, read_parquet, write_parquet
from src.utils.logging_utils import configure_logging
from src.utils.time_utils import utcnow


SOURCE_PARQUETS = [
    ("provider_a_wallets.parquet", "provider_a"),
    ("helius_token_buyers.parquet", "helius_token_buyer"),
    ("manual_wallets.parquet", "manual"),
    ("provider_b_wallets.parquet", "provider_b"),
    ("provider_b_signal_wallets.parquet", "provider_b_signal"),
    ("provider_c_wallets.parquet", "provider_c"),
    ("provider_d_wallets.parquet", "provider_d"),
]


def load_candidates(root: Path, logger) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name, fallback_source in SOURCE_PARQUETS:
        path = root / "data" / "bronze" / name
        df = read_parquet(path)
        if df.empty or "wallet" not in df.columns:
            logger.info("  source %s: empty/missing — skipping", name)
            continue
        src_col = df.get("source")
        if src_col is None:
            df = df.assign(source=fallback_source)
        logger.info("  source %s: %d rows", name, len(df))
        frames.append(df[["wallet", "source"]].copy())
    if not frames:
        return pd.DataFrame(columns=["wallet", "sources"])
    merged = pd.concat(frames, ignore_index=True).dropna(subset=["wallet"])
    merged = merged.drop_duplicates(["wallet", "source"])
    grouped = (
        merged.groupby("wallet", as_index=False)["source"]
        .agg(lambda s: sorted(set(s)))
        .rename(columns={"source": "sources"})
    )
    return grouped


def load_provider_a_compat_cols(root: Path) -> pd.DataFrame:
    """Pull appears_daily/weekly/monthly & best_rank from the provider_a raw table."""
    path = root / "data" / "bronze" / "provider_a_wallets.parquet"
    raw = read_parquet(path)
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "wallet",
                "appears_daily",
                "appears_weekly",
                "appears_monthly",
                "best_rank",
            ]
        )
    agg = raw.groupby("wallet", as_index=False).agg(
        appears_daily=("timeframe", lambda s: int((s == "daily").any())),
        appears_weekly=("timeframe", lambda s: int((s == "weekly").any())),
        appears_monthly=("timeframe", lambda s: int((s == "monthly").any())),
        best_rank=("rank", "min"),
    )
    return agg


def _is_proven_winner(sources) -> bool:
    if sources is None:
        return False
    try:
        return "proven_winner" in sources
    except TypeError:
        return False


def apply_filters(
    df: pd.DataFrame,
    *,
    min_trades: int,
    min_realized_pnl: float,
    min_win_rate: float,
    max_last_active_age_days: int,
    logger,
) -> pd.DataFrame:
    if df.empty:
        return df
    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    active_cutoff = now_ts - max_last_active_age_days * 86400
    before = len(df)

    # proven_winner wallets come from live closed trades that we verified
    # fired a profitable entry — bypass score-floor filters so they stay
    # in the pool even if Helius hasn't scored them yet.
    is_proven = df["sources"].map(_is_proven_winner)
    proven_count = int(is_proven.sum())
    logger.info("  proven_winner wallets (filter bypass): %d", proven_count)

    # Wallets never seen by Helius (no score row) get dropped — except proven_winners.
    df = df[is_proven | df["n_trades_30d"].notna()].copy()
    logger.info("  after has-score filter: %d (was %d)", len(df), before)

    is_proven = df["sources"].map(_is_proven_winner)
    df = df[is_proven | (df["n_trades_30d"].fillna(0) >= min_trades)]
    logger.info("  after n_trades_30d>=%d: %d", min_trades, len(df))

    is_proven = df["sources"].map(_is_proven_winner)
    df = df[is_proven | (df["realized_pnl_30d"].fillna(-1e9) > min_realized_pnl)]
    logger.info("  after realized_pnl_30d>%s: %d", min_realized_pnl, len(df))

    is_proven = df["sources"].map(_is_proven_winner)
    df = df[is_proven | (df["win_rate_30d"].fillna(-1.0) > min_win_rate)]
    logger.info("  after win_rate_30d>%s: %d", min_win_rate, len(df))

    is_proven = df["sources"].map(_is_proven_winner)
    df = df[is_proven | (df["last_active_ts"].fillna(0) >= active_cutoff)]
    logger.info("  after last_active within %dd: %d", max_last_active_age_days, len(df))

    return df


def build_scores(df: pd.DataFrame) -> pd.Series:
    """Composite = realized_pnl × log(n_trades) × (1 + 0.3·sources_count).

    Heavily weights absolute PnL but boosts activity and multi-source
    corroboration so a single-source noise wallet doesn't dominate.
    proven_winner wallets (captured from our own profitable closes) get a
    2× multiplier so they outrank scraped sources.
    """
    pnl = df["realized_pnl_30d"].fillna(0.0).clip(lower=0.01)
    trade_term = df["n_trades_30d"].fillna(1).clip(lower=1).map(math.log)
    src_count = df["sources"].map(len).clip(lower=1)
    base = pnl * trade_term * (1.0 + 0.3 * src_count)
    proven_bonus = df["sources"].map(lambda s: 2.0 if _is_proven_winner(s) else 1.0)
    return (base * proven_bonus).astype(float)


def run_pipeline(
    root: Path,
    *,
    skip_collect: bool,
    skip_score: bool,
    min_trades: int,
    min_realized_pnl: float,
    min_win_rate: float,
    max_last_active_age_days: int,
    top_n: int,
    logger,
) -> pd.DataFrame:
    if not skip_collect:
        logger.info("[step 1/4] collectors")
        for mod in (
            "src.collectors.collect_provider_a_wallets",
            "src.collectors.refresh_seed_tokens",
            "src.collectors.collect_helius_token_buyers",
            "src.collectors.collect_manual_wallets",
            "src.collectors.collect_provider_b_wallets",
            "src.collectors.collect_provider_b_signal_wallets",
            "src.collectors.collect_provider_c_wallets",
            "src.collectors.collect_provider_d_wallets",
        ):
            logger.info("  → %s", mod)
            try:
                subprocess.run(
                    [sys.executable, "-m", mod],
                    cwd=root,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                logger.warning("  %s failed (%s) — continuing with what we have", mod, exc)

    if not skip_score:
        logger.info("[step 2/4] Helius PnL scoring")
        subprocess.run(
            [sys.executable, "-m", "src.collectors.score_wallets_helius"],
            cwd=root,
            check=True,
        )

    logger.info("[step 3/4] union + filter + rank")
    candidates = load_candidates(root, logger)
    scores = read_parquet(root / "data" / "silver" / "wallet_scores.parquet")
    provider_a_compat = load_provider_a_compat_cols(root)

    if candidates.empty:
        logger.error("No candidates found — aborting")
        return pd.DataFrame()

    merged = candidates.merge(scores, on="wallet", how="left")
    merged = merged.merge(provider_a_compat, on="wallet", how="left")
    for col, default in [
        ("appears_daily", 0),
        ("appears_weekly", 0),
        ("appears_monthly", 0),
        ("best_rank", np.nan),
    ]:
        merged[col] = merged[col].fillna(default)

    logger.info("  merged: %d candidates", len(merged))
    filtered = apply_filters(
        merged,
        min_trades=min_trades,
        min_realized_pnl=min_realized_pnl,
        min_win_rate=min_win_rate,
        max_last_active_age_days=max_last_active_age_days,
        logger=logger,
    )

    if filtered.empty:
        logger.error("All candidates filtered out — pool would be empty. Relax thresholds.")
        return pd.DataFrame()

    filtered = filtered.copy()
    filtered["score"] = build_scores(filtered)
    filtered = filtered.sort_values("score", ascending=False)
    if top_n > 0:
        filtered = filtered.head(top_n)
    filtered["tags"] = [[] for _ in range(len(filtered))]
    filtered["refreshed_at"] = utcnow()

    # Preserve legacy schema as the leading columns.
    ordered_cols = [
        "wallet",
        "score",
        "appears_daily",
        "appears_weekly",
        "appears_monthly",
        "best_rank",
        "tags",
        "sources",
        "realized_pnl_30d",
        "total_invested_30d",
        "total_realized_30d",
        "n_trades_30d",
        "n_closed_30d",
        "win_rate_30d",
        "active_trades_7d",
        "last_active_ts",
        "refreshed_at",
    ]
    for col in ordered_cols:
        if col not in filtered.columns:
            filtered[col] = None
    return filtered[ordered_cols].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the wallet pool end-to-end.")
    parser.add_argument("--skip-collect", action="store_true", help="Skip running collectors")
    parser.add_argument("--skip-score", action="store_true", help="Skip Helius scoring")
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--min-pnl", type=float, default=5.0)
    parser.add_argument("--min-win-rate", type=float, default=0.35)
    parser.add_argument("--max-last-active-days", type=int, default=7)
    parser.add_argument("--top-n", type=int, default=150)
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't overwrite wallet_pool.parquet"
    )
    args = parser.parse_args()

    config = load_app_config()
    logger = configure_logging(logger_name=__name__)

    pool = run_pipeline(
        config.root_dir,
        skip_collect=args.skip_collect,
        skip_score=args.skip_score,
        min_trades=args.min_trades,
        min_realized_pnl=args.min_pnl,
        min_win_rate=args.min_win_rate,
        max_last_active_age_days=args.max_last_active_days,
        top_n=args.top_n,
        logger=logger,
    )

    if pool.empty:
        logger.error("Pipeline produced empty pool — NOT overwriting existing wallet_pool.parquet")
        return

    logger.info("[step 4/4] ranked pool preview (top 10)")
    for _, row in pool.head(10).iterrows():
        logger.info(
            "  %s  score=%7.2f  pnl=%+7.2f  trades=%4d  wr=%.2f  sources=%s",
            row["wallet"],
            float(row["score"]),
            float(row["realized_pnl_30d"]),
            int(row["n_trades_30d"]),
            float(row["win_rate_30d"]),
            row["sources"],
        )

    out_path = dataset_path(config, "bronze", "wallet_pool.parquet")
    if args.dry_run:
        logger.info("--dry-run: would have written %d rows to %s", len(pool), out_path)
    else:
        write_parquet(pool, out_path)
        logger.info("Wrote %s (%d rows)", out_path, len(pool))


if __name__ == "__main__":
    main()
