"""48-hour rule performance report.

Queries data/live/bot_state.db and prints a ranked winner/loser breakdown
by rule × regime × strategy for the last 48 hours of closed positions.

Usage:
    python src/analysis/rule_performance_48h.py
    python src/analysis/rule_performance_48h.py --hours 72
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "live" / "bot_state.db"

ROW_W = 110


def bar(value: float, max_abs: float, width: int = 20) -> str:
    if max_abs == 0:
        return " " * width
    ratio = min(abs(value) / max_abs, 1.0)
    filled = int(ratio * width)
    sym = "█" if value >= 0 else "▓"
    return (sym * filled).ljust(width)


def fmt_pnl(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.4f}"


def fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def section(title: str) -> None:
    print()
    print("─" * ROW_W)
    print(f"  {title}")
    print("─" * ROW_W)


def run(hours: int = 48) -> None:
    if not DB_PATH.exists():
        print(f"ERROR: database not found at {DB_PATH}")
        return

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    # ── 1. Timespan check ────────────────────────────────────────────────────
    cutoff_sql = f"datetime('now', '-{hours} hours')"
    span = con.execute(
        f"""
        SELECT
            MIN(entry_time)                                           AS oldest,
            MAX(entry_time)                                           AS newest,
            COUNT(*)                                                  AS total,
            SUM(CASE WHEN status != 'OPEN' THEN 1 ELSE 0 END)        AS closed,
            SUM(CASE WHEN status  = 'OPEN' THEN 1 ELSE 0 END)        AS open_pos,
            SUM(CASE WHEN status  = 'CLOSED' THEN realized_pnl_sol ELSE 0 END) AS total_pnl
        FROM positions
        WHERE entry_time >= {cutoff_sql}
        """
    ).fetchone()

    print()
    print("=" * ROW_W)
    print(f"  RULE PERFORMANCE REPORT — last {hours}h")
    print("=" * ROW_W)
    print(f"  Window : {span['oldest']}  →  {span['newest']}")
    print(
        f"  Total  : {span['total']} entries  |  {span['closed']} closed  |  {span['open_pos']} open"
    )
    pnl_c = "+" if span["total_pnl"] >= 0 else ""
    print(f"  Net PnL: {pnl_c}{span['total_pnl']:.4f} SOL  (closed only)")

    # ── 2. Per rule-regime-strategy breakdown ────────────────────────────────
    rows = con.execute(
        f"""
        SELECT
            selected_rule_id                                                          AS rule_id,
            COALESCE(NULLIF(selected_regime,''), 'unknown')                           AS regime,
            strategy_id,
            COUNT(*)                                                                  AS entries,
            SUM(CASE WHEN status != 'OPEN'    THEN 1   ELSE 0   END)                 AS closed,
            SUM(CASE WHEN status  = 'OPEN'    THEN 1   ELSE 0   END)                 AS open_pos,
            SUM(CASE WHEN status  = 'CLOSED' AND realized_pnl_sol  > 0 THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN status  = 'CLOSED' AND realized_pnl_sol <= 0 THEN 1 ELSE 0 END) AS losses,
            SUM(CASE WHEN status  = 'CLOSED' THEN realized_pnl_sol ELSE 0 END)       AS total_pnl,
            AVG(CASE WHEN status  = 'CLOSED' THEN realized_pnl_sol ELSE NULL END)    AS avg_pnl,
            SUM(CASE WHEN exit_stage = 99  THEN 1 ELSE 0 END)                        AS stop_outs,
            SUM(CASE WHEN exit_stage = 201 THEN 1 ELSE 0 END)                        AS sniper_tp,
            SUM(CASE WHEN exit_stage = 298 THEN 1 ELSE 0 END)                        AS sniper_timeout,
            SUM(CASE WHEN exit_stage = 299 THEN 1 ELSE 0 END)                        AS sniper_stop
        FROM positions
        WHERE entry_time >= {cutoff_sql}
          AND selected_rule_id IS NOT NULL
          AND selected_rule_id != ''
        GROUP BY selected_rule_id, selected_regime, strategy_id
        ORDER BY total_pnl DESC
        """
    ).fetchall()

    if not rows:
        print("\n  No positions found in this window.")
        con.close()
        return

    winners = [r for r in rows if r["total_pnl"] > 0]
    losers = [r for r in rows if r["total_pnl"] <= 0]
    max_abs = max(abs(r["total_pnl"]) for r in rows) or 1.0

    hdr = f"  {'Rule':<40} {'Regime':<28} {'Strat':<7} {'N':>4} {'Cl':>4} {'WR':>6} {'TotPnL':>10} {'AvgPnL':>10} {'Stops':>5}  Chart"
    row_sep = "  " + "-" * (ROW_W - 2)

    # ── WINNERS ──────────────────────────────────────────────────────────────
    section(f"WINNERS  ({len(winners)} rules / rule-regime pairs with net positive PnL)")
    print(hdr)
    print(row_sep)
    for r in winners:
        wr = r["wins"] / r["closed"] if r["closed"] else 0.0
        sniper_info = ""
        if r["strategy_id"] == "sniper":
            sniper_info = f" [tp={r['sniper_tp']} to={r['sniper_timeout']} st={r['sniper_stop']}]"
        print(
            f"  {str(r['rule_id']):<40} {str(r['regime']):<28} {str(r['strategy_id']):<7}"
            f" {r['entries']:>4} {r['closed']:>4} {fmt_pct(wr):>6}"
            f" {fmt_pnl(r['total_pnl']):>10} {fmt_pnl(r['avg_pnl'] or 0):>10}"
            f" {r['stop_outs']:>5}  {bar(r['total_pnl'], max_abs)}{sniper_info}"
        )

    # ── LOSERS ───────────────────────────────────────────────────────────────
    section(f"LOSERS   ({len(losers)} rules / rule-regime pairs with net negative or zero PnL)")
    print(hdr)
    print(row_sep)
    for r in sorted(losers, key=lambda x: x["total_pnl"]):
        wr = r["wins"] / r["closed"] if r["closed"] else 0.0
        sniper_info = ""
        if r["strategy_id"] == "sniper":
            sniper_info = f" [tp={r['sniper_tp']} to={r['sniper_timeout']} st={r['sniper_stop']}]"
        print(
            f"  {str(r['rule_id']):<40} {str(r['regime']):<28} {str(r['strategy_id']):<7}"
            f" {r['entries']:>4} {r['closed']:>4} {fmt_pct(wr):>6}"
            f" {fmt_pnl(r['total_pnl']):>10} {fmt_pnl(r['avg_pnl'] or 0):>10}"
            f" {r['stop_outs']:>5}  {bar(r['total_pnl'], max_abs)}{sniper_info}"
        )

    # ── STRATEGY SUMMARY ─────────────────────────────────────────────────────
    strats = con.execute(
        f"""
        SELECT
            strategy_id,
            COUNT(*)                                                              AS entries,
            SUM(CASE WHEN status  = 'CLOSED' AND realized_pnl_sol  > 0 THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN status  = 'CLOSED' AND realized_pnl_sol <= 0 THEN 1 ELSE 0 END) AS losses,
            SUM(CASE WHEN status  = 'CLOSED' THEN realized_pnl_sol ELSE 0 END)   AS total_pnl,
            AVG(CASE WHEN status  = 'CLOSED' THEN realized_pnl_sol ELSE NULL END) AS avg_pnl,
            SUM(CASE WHEN status != 'OPEN'   THEN 1 ELSE 0 END)                  AS closed
        FROM positions
        WHERE entry_time >= {cutoff_sql}
        GROUP BY strategy_id
        ORDER BY total_pnl DESC
        """
    ).fetchall()

    section("BY STRATEGY")
    print(
        f"  {'Strategy':<12} {'Entries':>8} {'Closed':>8} {'Wins':>6} {'Losses':>8} {'WinRate':>8} {'TotPnL':>11} {'AvgPnL':>10}"
    )
    print(row_sep)
    for s in strats:
        wr = s["wins"] / s["closed"] if s["closed"] else 0.0
        print(
            f"  {str(s['strategy_id']):<12} {s['entries']:>8} {s['closed']:>8}"
            f" {s['wins']:>6} {s['losses']:>8} {fmt_pct(wr):>8}"
            f" {fmt_pnl(s['total_pnl']):>11} {fmt_pnl(s['avg_pnl'] or 0):>10}"
        )

    # ── REGIME SUMMARY ───────────────────────────────────────────────────────
    regimes = con.execute(
        f"""
        SELECT
            COALESCE(NULLIF(selected_regime,''), 'unknown')                       AS regime,
            strategy_id,
            COUNT(*)                                                              AS entries,
            SUM(CASE WHEN status  = 'CLOSED' AND realized_pnl_sol  > 0 THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN status  = 'CLOSED' AND realized_pnl_sol <= 0 THEN 1 ELSE 0 END) AS losses,
            SUM(CASE WHEN status  = 'CLOSED' THEN realized_pnl_sol ELSE 0 END)   AS total_pnl,
            AVG(CASE WHEN status  = 'CLOSED' THEN realized_pnl_sol ELSE NULL END) AS avg_pnl,
            SUM(CASE WHEN status != 'OPEN'   THEN 1 ELSE 0 END)                  AS closed
        FROM positions
        WHERE entry_time >= {cutoff_sql}
        GROUP BY selected_regime, strategy_id
        ORDER BY total_pnl DESC
        """
    ).fetchall()

    section("BY REGIME × STRATEGY")
    print(
        f"  {'Regime':<35} {'Strat':<8} {'Entries':>8} {'Closed':>8} {'WinRate':>8} {'TotPnL':>11} {'AvgPnL':>10}"
    )
    print(row_sep)
    for r in regimes:
        wr = r["wins"] / r["closed"] if r["closed"] else 0.0
        pnl_flag = (
            " ◀ LOSS" if r["total_pnl"] < -0.05 else (" ◀ WIN" if r["total_pnl"] > 0.05 else "")
        )
        print(
            f"  {str(r['regime']):<35} {str(r['strategy_id']):<8} {r['entries']:>8} {r['closed']:>8}"
            f" {fmt_pct(wr):>8} {fmt_pnl(r['total_pnl']):>11} {fmt_pnl(r['avg_pnl'] or 0):>10}{pnl_flag}"
        )

    # ── TOP EXIT REASONS ─────────────────────────────────────────────────────
    exits = con.execute(
        f"""
        SELECT
            exit_stage,
            strategy_id,
            COUNT(*)                                   AS n,
            SUM(realized_pnl_sol)                      AS total_pnl,
            AVG(realized_pnl_sol)                      AS avg_pnl
        FROM positions
        WHERE entry_time >= {cutoff_sql}
          AND status = 'CLOSED'
        GROUP BY exit_stage, strategy_id
        ORDER BY n DESC
        """
    ).fetchall()

    STAGE_LABELS = {
        0: "open/unknown",
        87: "stage0_crash_guard",
        89: "stage0_fast_fail",
        90: "stage0_timeout",
        92: "absolute_timeout",
        93: "stage0_loss_timeout",
        94: "stage0_moderate_pos_timeout",
        98: "pre_tp1_retrace_lock",
        99: "stop_out",
        201: "sniper_take_profit",
        298: "sniper_timeout",
        299: "sniper_stop_out",
    }

    section("EXIT STAGE BREAKDOWN  (closed positions)")
    print(f"  {'Stage':>6}  {'Label':<35} {'Strat':<8} {'N':>5} {'TotPnL':>11} {'AvgPnL':>10}")
    print(row_sep)
    for e in exits:
        label = STAGE_LABELS.get(e["exit_stage"], f"stage_{e['exit_stage']}")
        print(
            f"  {e['exit_stage']:>6}  {label:<35} {str(e['strategy_id']):<8}"
            f" {e['n']:>5} {fmt_pnl(e['total_pnl']):>11} {fmt_pnl(e['avg_pnl']):>10}"
        )

    print()
    print("=" * ROW_W)
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=48)
    args = parser.parse_args()
    run(hours=args.hours)
