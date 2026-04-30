"""ML performance report — entry ML, exit ML, and market regime signal.

Reads position metadata_json to extract ML probabilities and compares them
against actual realized PnL outcomes. Runs against the last N hours (default 48h)
or all data with --all flag.

Usage:
    python src/analysis/ml_performance.py
    python src/analysis/ml_performance.py --hours 72
    python src/analysis/ml_performance.py --all
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "live" / "bot_state.db"
ROW_W = 112


# ── helpers ──────────────────────────────────────────────────────────────────


def section(title: str) -> None:
    print()
    print("─" * ROW_W)
    print(f"  {title}")
    print("─" * ROW_W)


def pnl_bar(v: float, max_abs: float, width: int = 18) -> str:
    if max_abs == 0:
        return " " * width
    ratio = min(abs(v) / max_abs, 1.0)
    filled = int(ratio * width)
    sym = "█" if v >= 0 else "▓"
    return (sym * filled).ljust(width)


def fmt(v: float | None, d: int = 4, sign: bool = True) -> str:
    if v is None:
        return "-"
    prefix = ("+" if v >= 0 else "") if sign else ""
    return f"{prefix}{v:.{d}f}"


def fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def gate_sim_line(threshold: float, positions: list[dict]) -> None:
    """Print one row of the gate simulation table."""
    passed = [p for p in positions if p["prob"] >= threshold]
    blocked = [p for p in positions if p["prob"] < threshold]
    p_wins = sum(1 for p in passed if p["pnl"] > 0)
    p_pnl = sum(p["pnl"] for p in passed)
    b_pnl = sum(p["pnl"] for p in blocked)
    p_wr = p_wins / len(passed) if passed else 0.0
    saved = -b_pnl  # negative blocked PnL = saved from losses
    print(
        f"  ≥{threshold:.2f}  │"
        f"  {len(passed):>5} passed ({fmt_pct(len(passed) / len(positions) if positions else 0):>6})  │"
        f"  {len(blocked):>5} blocked  │"
        f"  WR {fmt_pct(p_wr):>6}  │"
        f"  PnL {fmt(p_pnl):>10} SOL  │"
        f"  saved {fmt(saved, sign=False):>9} SOL"
    )


# ── main ─────────────────────────────────────────────────────────────────────


def run(hours: int | None = 48) -> None:
    if not DB_PATH.exists():
        print(f"ERROR: database not found at {DB_PATH}")
        return

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    # Build time filter
    if hours is not None:
        time_filter = f"AND entry_time >= datetime('now', '-{hours} hours')"
        window_label = f"last {hours}h"
    else:
        time_filter = ""
        window_label = "all time"

    # Load all closed positions with metadata
    raw = con.execute(
        f"""
        SELECT id, strategy_id, status, realized_pnl_sol, entry_time, exit_stage,
               metadata_json
        FROM positions
        WHERE status = 'CLOSED'
          AND metadata_json IS NOT NULL
          AND metadata_json != '{{}}'
          {time_filter}
        ORDER BY entry_time ASC
        """
    ).fetchall()

    # Parse metadata
    positions: list[dict] = []
    for r in raw:
        try:
            meta = json.loads(r["metadata_json"])
        except Exception:
            continue
        prob = meta.get("ml_probability")
        if prob is None:
            continue
        positions.append(
            {
                "id": r["id"],
                "strategy": r["strategy_id"],
                "pnl": float(r["realized_pnl_sol"]),
                "exit_stage": r["exit_stage"],
                "prob": float(prob),
                "threshold": float(meta.get("ml_threshold", 0.5)),
                "ml_mode": meta.get("ml_mode", "unknown"),
                "ml_reason": meta.get("ml_reason", ""),
                "model_ready": bool(meta.get("ml_model_ready", False)),
                "exit_hold": meta.get("exit_ml_hold_probability"),
                "regime_score": float(
                    meta.get("ml_feature_map", {}).get("market_regime_score", -1) or -1
                ),
            }
        )

    total_raw = con.execute(
        f"SELECT COUNT(*) FROM positions WHERE status='CLOSED' {time_filter}"
    ).fetchone()[0]

    print()
    print("=" * ROW_W)
    print(f"  ML PERFORMANCE REPORT — {window_label}")
    print("=" * ROW_W)
    print(f"  Closed positions in window : {total_raw}")
    print(f"  Positions with ML score    : {len(positions)}")
    if not positions:
        print("\n  No ML-scored positions found.")
        con.close()
        return
    covered = len(positions) / total_raw if total_raw else 0
    ml_ready_count = sum(1 for p in positions if p["model_ready"])
    print(f"  Coverage                   : {fmt_pct(covered)}")
    print(f"  Model-ready ticks          : {ml_ready_count} / {len(positions)}")
    ml_modes = {}
    for p in positions:
        ml_modes[p["ml_mode"]] = ml_modes.get(p["ml_mode"], 0) + 1
    print(f"  ML modes seen              : {dict(ml_modes)}")

    main_pos = [p for p in positions if p["strategy"] == "main"]
    sniper_pos = [p for p in positions if p["strategy"] == "sniper"]

    # ── ENTRY ML — probability buckets ───────────────────────────────────────
    BUCKETS = [
        ("<0.30", None, 0.30),
        ("0.30–0.40", 0.30, 0.40),
        ("0.40–0.50", 0.40, 0.50),
        ("0.50–0.60", 0.50, 0.60),
        ("0.60–0.70", 0.60, 0.70),
        ("0.70–0.80", 0.70, 0.80),
        ("≥0.80", 0.80, None),
    ]

    def bucket_stats(pool: list[dict]) -> None:
        max_abs = max((abs(p["pnl"]) for p in pool), default=1.0)
        hdr = f"  {'Bucket':<12} {'N':>5} {'Wins':>6} {'WinRate':>8} {'TotPnL':>11} {'AvgPnL':>10}  Chart"
        print(hdr)
        print("  " + "-" * (ROW_W - 2))
        for label, lo, hi in BUCKETS:
            rows = [
                p
                for p in pool
                if (lo is None or p["prob"] >= lo) and (hi is None or p["prob"] < hi)
            ]
            if not rows:
                continue
            wins = sum(1 for p in rows if p["pnl"] > 0)
            tot_pnl = sum(p["pnl"] for p in rows)
            avg_pnl = tot_pnl / len(rows)
            wr = wins / len(rows)
            flag = " ◀ BAD" if wr < 0.32 else (" ◀ GOOD" if wr > 0.50 else "")
            print(
                f"  {label:<12} {len(rows):>5} {wins:>6} {fmt_pct(wr):>8}"
                f" {fmt(tot_pnl):>11} {fmt(avg_pnl):>10}  {pnl_bar(tot_pnl, max_abs)}{flag}"
            )

    section("ENTRY ML — ALL STRATEGIES  (probability vs outcome)")
    bucket_stats(positions)

    section("ENTRY ML — MAIN STRATEGY")
    if main_pos:
        bucket_stats(main_pos)
    else:
        print("  No main-strategy positions with ML score.")

    section("ENTRY ML — SNIPER STRATEGY")
    if sniper_pos:
        bucket_stats(sniper_pos)
    else:
        print("  No sniper-strategy positions with ML score.")

    # ── GATE SIMULATION ───────────────────────────────────────────────────────
    section("GATE SIMULATION — what if ML had been in gate mode?")
    print(
        f"  Baseline (no gate): {len(positions)} trades  |  PnL {fmt(sum(p['pnl'] for p in positions))} SOL  |  WR {fmt_pct(sum(1 for p in positions if p['pnl'] > 0) / len(positions))}"
    )
    print()
    print(
        f"  {'Thresh':<7} │ {'Passed':^26} │ {'Blocked':^15} │ {'WinRate':^12} │ {'Passed PnL':^18} │ {'Saved from blocked'}"
    )
    print("  " + "─" * (ROW_W - 2))
    for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        gate_sim_line(t, positions)

    print()
    print("  ── MAIN strategy only ──")
    if main_pos:
        for t in [0.35, 0.40, 0.45, 0.50]:
            gate_sim_line(t, main_pos)

    print()
    print("  ── SNIPER strategy only ──")
    if sniper_pos:
        for t in [0.35, 0.40, 0.45, 0.50]:
            gate_sim_line(t, sniper_pos)

    # ── EXIT ML ───────────────────────────────────────────────────────────────
    exit_pos = [p for p in positions if p["exit_hold"] is not None]
    section(
        f"EXIT ML — hold_probability vs outcome  ({len(exit_pos)} positions with exit ML score)"
    )

    if exit_pos:
        EXIT_BUCKETS = [
            ("<0.10", None, 0.10),
            ("0.10–0.20", 0.10, 0.20),
            ("0.20–0.30", 0.20, 0.30),
            ("0.30–0.40", 0.30, 0.40),
            ("0.40–0.50", 0.40, 0.50),
            ("≥0.50", 0.50, None),
        ]
        max_abs = max((abs(p["pnl"]) for p in exit_pos), default=1.0)
        print(
            f"  {'Bucket':<12} {'N':>5} {'Wins':>6} {'WinRate':>8} {'TotPnL':>11} {'AvgPnL':>10}  Note"
        )
        print("  " + "-" * (ROW_W - 2))
        for label, lo, hi in EXIT_BUCKETS:
            rows = [
                p
                for p in exit_pos
                if (lo is None or p["exit_hold"] >= lo) and (hi is None or p["exit_hold"] < hi)
            ]
            if not rows:
                continue
            wins = sum(1 for p in rows if p["pnl"] > 0)
            tot_pnl = sum(p["pnl"] for p in rows)
            avg_pnl = tot_pnl / len(rows)
            wr = wins / len(rows)
            # Low hold_prob = model wanted to exit = check if that was right
            note = (
                " [model said exit — correct if pnl<0]"
                if hi is not None and hi <= 0.20
                else " [model said hold — correct if pnl>0]"
                if lo is not None and lo >= 0.40
                else ""
            )
            print(
                f"  {label:<12} {len(rows):>5} {wins:>6} {fmt_pct(wr):>8}"
                f" {fmt(tot_pnl):>11} {fmt(avg_pnl):>10}  {note}"
            )
    else:
        print(
            "  No exit ML scores found (field added recently — future positions will populate this)."
        )

    # ── MARKET REGIME SCORE ───────────────────────────────────────────────────
    regime_pos = [p for p in positions if p["regime_score"] >= 0]
    section(f"MARKET REGIME SCORE — composite signal vs outcome  ({len(regime_pos)} positions)")

    if regime_pos:
        REG_BUCKETS = [
            ("<0.30", None, 0.30),
            ("0.30–0.45", 0.30, 0.45),
            ("0.45–0.55", 0.45, 0.55),
            ("0.55–0.65", 0.55, 0.65),
            ("≥0.65", 0.65, None),
        ]
        max_abs = max((abs(p["pnl"]) for p in regime_pos), default=1.0)
        print(
            f"  {'Score range':<14} {'N':>5} {'Wins':>6} {'WinRate':>8} {'TotPnL':>11} {'AvgPnL':>10}  Chart"
        )
        print("  " + "-" * (ROW_W - 2))
        for label, lo, hi in REG_BUCKETS:
            rows = [
                p
                for p in regime_pos
                if (lo is None or p["regime_score"] >= lo)
                and (hi is None or p["regime_score"] < hi)
            ]
            if not rows:
                continue
            wins = sum(1 for p in rows if p["pnl"] > 0)
            tot_pnl = sum(p["pnl"] for p in rows)
            avg_pnl = tot_pnl / len(rows)
            wr = wins / len(rows)
            print(
                f"  {label:<14} {len(rows):>5} {wins:>6} {fmt_pct(wr):>8}"
                f" {fmt(tot_pnl):>11} {fmt(avg_pnl):>10}  {pnl_bar(tot_pnl, max_abs)}"
            )

    # ── PROBABILITY TREND (last 50 positions) ────────────────────────────────
    section("ENTRY ML PROBABILITY — recent trend (last 50 scored positions)")
    recent = positions[-50:]
    if recent:
        avg_prob = sum(p["prob"] for p in recent) / len(recent)
        avg_pnl_r = sum(p["pnl"] for p in recent) / len(recent)
        wins_r = sum(1 for p in recent if p["pnl"] > 0)
        above_thr = sum(1 for p in recent if p["prob"] >= p["threshold"])
        print(f"  Avg probability (last 50): {avg_prob:.3f}")
        print(f"  Avg PnL        (last 50): {fmt(avg_pnl_r)} SOL")
        print(f"  Win rate       (last 50): {fmt_pct(wins_r / len(recent))}")
        print(
            f"  Above threshold          : {above_thr} / {len(recent)}  ({fmt_pct(above_thr / len(recent))})"
        )

        # Mini sparkline of prob trend
        print()
        print("  Probability sparkline (each char = 1 position, scale 0.0→1.0):")
        spark_chars = " ▁▂▃▄▅▆▇█"
        line = "  "
        for p in recent:
            idx = min(int(p["prob"] * 8), 8)
            line += spark_chars[idx]
        print(line)
        print("  " + "0" + " " * 24 + "0.5" + " " * 20 + "1.0")

    print()
    print("=" * ROW_W)
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--hours", type=int, default=48, help="Hours to look back (default 48)")
    group.add_argument("--all", action="store_true", help="Analyze all data")
    args = parser.parse_args()
    run(hours=None if args.all else args.hours)
