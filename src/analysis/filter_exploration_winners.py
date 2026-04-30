"""Filter exploration_v1 rules after 12h live run.

Reads rule_performance from bot_state.db and ranks exploration rules by:
  1. Positive avg_pnl  (must be > 0)
  2. Win rate ≥ 50%
  3. ≥ MIN_ENTRIES to be statistically meaningful

Outputs two files:
  outputs/rules/winners_v1.csv        — rules that passed all filters
  outputs/rules/winners_v1_sniper.csv — subset suitable for sniper lane

Usage:
    python -m src.analysis.filter_exploration_winners [--min-entries N] [--min-winrate 0.50]
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

DB_PATH = Path("data/live/bot_state.db")
EXPLORATION_CSV = Path("outputs/rules/exploration_v1.csv")
WINNERS_OUT = Path("outputs/rules/winners_v1.csv")
SNIPER_OUT = Path("outputs/rules/winners_v1_sniper.csv")

# Sniper-appropriate families: fast, fresh-token focused
SNIPER_FAMILIES = {"fresh_momentum", "clean_sweep", "combo_signal", "buy_dominance"}

# Conditions that suggest a rule fires early enough for sniper (≤90s age or ratio-based)
SNIPER_AGE_MAX = 90.0

COLUMNS = [
    "rule_id",
    "family",
    "support",
    "support_valid",
    "precision",
    "precision_valid",
    "recall",
    "f1",
    "lift",
    "score",
    "score_valid",
    "pack_name",
    "pack_rank",
    "conditions_obj",
    "conditions_json",
    "conditions",
    "notes",
    # appended live stats
    "live_entries",
    "live_wins",
    "live_losses",
    "live_stop_outs",
    "live_win_rate",
    "live_avg_pnl",
    "live_total_pnl",
]


def _load_exploration_rules() -> dict[str, dict]:
    """Return exploration rule rows keyed by rule_id."""
    if not EXPLORATION_CSV.exists():
        raise FileNotFoundError(f"Rule file not found: {EXPLORATION_CSV}")
    with EXPLORATION_CSV.open() as f:
        reader = csv.DictReader(f)
        return {row["rule_id"]: row for row in reader}


def _load_rule_performance(min_entries: int) -> dict[str, dict]:
    """Return rule_performance rows for rules with ≥ min_entries."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT rule_id, entries, wins, losses, stop_outs, average_pnl, realized_pnl "
        "FROM rule_performance WHERE entries >= ?",
        (min_entries,),
    ).fetchall()
    conn.close()
    return {r["rule_id"]: dict(r) for r in rows}


def _is_sniper_suitable(rule_row: dict) -> bool:
    """Return True if this rule is a good candidate for the sniper lane."""
    family = rule_row.get("family", "")
    if family not in SNIPER_FAMILIES:
        return False
    try:
        cond = json.loads(rule_row.get("conditions_obj", "{}"))
    except Exception:
        return False
    age_max = cond.get("token_age_sec_max")
    if age_max is not None and float(age_max) <= SNIPER_AGE_MAX:
        return True
    # Also include rules that are ratio/dominance-based even without age cap
    if "buy_sell_ratio_30s_min" in cond and float(cond["buy_sell_ratio_30s_min"]) >= 20.0:
        return True
    return False


def _print_summary(winners: list[dict], label: str) -> None:
    print(f"\n{'=' * 70}")
    print(f" {label} — {len(winners)} rules")
    print(f"{'=' * 70}")
    if not winners:
        print("  (none)")
        return
    header = f"  {'rule_id':<38} {'entries':>7} {'win%':>6} {'avg_pnl':>9} {'total_pnl':>10}"
    print(header)
    print("  " + "-" * 74)
    for r in winners:
        print(
            f"  {r['rule_id']:<38} {r['live_entries']:>7} "
            f"{r['live_win_rate'] * 100:>5.1f}% {r['live_avg_pnl']:>9.4f} "
            f"{r['live_total_pnl']:>10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-entries",
        type=int,
        default=5,
        help="Minimum live entries for a rule to be considered (default: 5)",
    )
    parser.add_argument(
        "--min-winrate",
        type=float,
        default=0.50,
        help="Minimum win rate (default: 0.50)",
    )
    parser.add_argument(
        "--min-avgpnl",
        type=float,
        default=0.0,
        help="Minimum average PnL in SOL (default: 0.0)",
    )
    args = parser.parse_args()

    exploration = _load_exploration_rules()
    perf = _load_rule_performance(min_entries=args.min_entries)

    print(f"\nExploration rules loaded: {len(exploration)}")
    print(f"Rules with ≥{args.min_entries} live entries: {len(perf)}")

    # Print all exploration rule performance (inc. those below threshold)
    print(f"\n{'All exploration_v1 rule performance':^70}")
    print(
        f"  {'rule_id':<38} {'entries':>7} {'wins':>5} {'win%':>6} {'avg_pnl':>9} {'total_pnl':>10}"
    )
    print("  " + "-" * 78)
    exp_perf = {k: v for k, v in perf.items() if k.startswith("exp_")}
    for rid, p in sorted(exp_perf.items(), key=lambda x: -x[1]["average_pnl"]):
        wp = p["wins"] / p["entries"] * 100 if p["entries"] > 0 else 0
        print(
            f"  {rid:<38} {p['entries']:>7} {p['wins']:>5} {wp:>5.1f}% "
            f"{p['average_pnl']:>9.4f} {p['realized_pnl']:>10.4f}"
        )

    # Apply filters
    winners = []
    for rule_id, rule_row in exploration.items():
        p = perf.get(rule_id)
        if p is None:
            continue  # not enough entries yet
        entries = p["entries"]
        wins = p["wins"]
        win_rate = wins / entries if entries > 0 else 0.0
        avg_pnl = p["average_pnl"]
        if win_rate < args.min_winrate:
            continue
        if avg_pnl < args.min_avgpnl:
            continue
        merged = dict(rule_row)
        merged["live_entries"] = entries
        merged["live_wins"] = wins
        merged["live_losses"] = p["losses"]
        merged["live_stop_outs"] = p["stop_outs"]
        merged["live_win_rate"] = round(win_rate, 4)
        merged["live_avg_pnl"] = round(avg_pnl, 6)
        merged["live_total_pnl"] = round(p["realized_pnl"], 6)
        # Update precision/score fields with live stats
        merged["precision"] = round(win_rate, 4)
        merged["precision_valid"] = round(win_rate, 4)
        merged["score"] = round(max(avg_pnl * 10, 0.0), 6)
        merged["score_valid"] = merged["score"]
        merged["support"] = entries
        merged["support_valid"] = entries
        merged["notes"] = (
            f"{rule_row.get('notes', '')} | Live: {entries} trades, "
            f"{win_rate * 100:.1f}% win, {avg_pnl:+.4f} avg SOL"
        )
        winners.append(merged)

    winners.sort(key=lambda r: (-r["live_avg_pnl"], -r["live_win_rate"]))
    sniper_winners = [r for r in winners if _is_sniper_suitable(r)]

    _print_summary(winners, "WINNERS (all lanes)")
    _print_summary(sniper_winners, "WINNERS (sniper-suitable)")

    # Write outputs
    WINNERS_OUT.parent.mkdir(parents=True, exist_ok=True)

    with WINNERS_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(winners)

    with SNIPER_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(sniper_winners)

    print("\nWritten to:")
    print(f"  {WINNERS_OUT}  ({len(winners)} rules)")
    print(f"  {SNIPER_OUT}  ({len(sniper_winners)} sniper rules)")

    if winners:
        all_ids = ",".join(r["rule_id"] for r in winners)
        sniper_ids = ",".join(r["rule_id"] for r in sniper_winners)
        print("\nTo deploy winners, set in .env:")
        print("  PUMP_RULES_PATH=outputs/rules/winners_v1.csv")
        print(f"  MAX_ACTIVE_RULES={len(winners)}")
        if sniper_ids:
            print(f"  SNIPER_RULE_IDS={sniper_ids}")
    else:
        print("\nNo winners yet — run longer or lower --min-entries / --min-winrate thresholds.")


if __name__ == "__main__":
    main()
