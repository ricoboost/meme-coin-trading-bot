"""Generate a diverse exploration rule pack (~100 rules) for 12h live testing.

Each rule covers a different slice of the feature space. After 12h, filter by
win% and avg_pnl from rule_performance to find what actually works.

Usage:
    python -m src.analysis.generate_exploration_rules
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

OUTFILE = Path("outputs/rules/exploration_v1.csv")

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
]

rules: list[dict] = []
_rank = 0


def add(rule_id: str, family: str, conditions: dict, notes: str = "") -> None:
    global _rank
    _rank += 1
    cj = json.dumps(conditions)
    rules.append(
        {
            "rule_id": rule_id,
            "family": family,
            "support": 10,
            "support_valid": 10,
            "precision": 0.5,
            "precision_valid": 0.5,
            "recall": 0.0,
            "f1": 0.0,
            "lift": 1.0,
            "score": 0.05,
            "score_valid": 0.05,
            "pack_name": "exploration_v1",
            "pack_rank": _rank,
            "conditions_obj": cj,
            "conditions_json": cj,
            "conditions": str(conditions),
            "notes": notes or f"Exploration rule {rule_id}",
        }
    )


# ── Group A: Fresh Token + Positive Momentum (14 rules) ──────────────────────
# Core hypothesis: fresh token + upward price pressure = early pump entry
for age in [30, 45, 60, 90, 120]:
    for price_min in [0.05, 0.10, 0.20]:
        if age == 30 and price_min == 0.20:
            continue  # 30s age + 20% move is near-impossible to catch
        add(
            f"exp_fresh_mom_{age}s_p{int(price_min * 100):02d}",
            "fresh_momentum",
            {
                "token_age_sec_max": age,
                "price_change_30s_min": price_min,
                "tx_count_30s_min": 3,
                "volume_sol_30s_min": 1.5,
            },
            f"Fresh token (≤{age}s) ≥{int(price_min * 100)}% 30s move, ≥3 tx, ≥1.5 SOL",
        )

add(
    "exp_ultra_fresh_any",
    "fresh_momentum",
    {"token_age_sec_max": 30, "tx_count_30s_min": 3, "volume_sol_30s_min": 1.0},
    "Ultra-fresh (≤30s) any activity — cast wide net",
)

add(
    "exp_ultra_fresh_buyvol",
    "fresh_momentum",
    {"token_age_sec_max": 45, "buy_volume_sol_30s_min": 3.0, "tx_count_30s_min": 4},
    "Ultra-fresh ≤45s, 3 SOL buy volume",
)

# ── Group B: Volume Surge (11 rules) ─────────────────────────────────────────
# Raw volume signal — total and buy-only
for vol in [2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
    add(
        f"exp_vol_{int(vol):02d}sol",
        "volume_surge",
        {"volume_sol_30s_min": vol, "tx_count_30s_min": 3},
        f"≥{vol} SOL 30s volume, ≥3 tx",
    )

for buy_vol in [2.0, 5.0, 10.0, 20.0, 40.0]:
    add(
        f"exp_buyvol_{int(buy_vol):02d}sol",
        "volume_surge",
        {"buy_volume_sol_30s_min": buy_vol, "tx_count_30s_min": 3},
        f"≥{buy_vol} SOL 30s BUY volume, ≥3 tx",
    )

# ── Group C: Buy/Sell Ratio Dominance (8 rules) ───────────────────────────────
for ratio in [2.0, 5.0, 10.0, 20.0, 50.0, 80.0]:
    add(
        f"exp_bsr_{int(ratio):03d}",
        "buy_dominance",
        {
            "buy_sell_ratio_30s_min": ratio,
            "volume_sol_30s_min": 1.5,
            "tx_count_30s_min": 3,
        },
        f"Buy/sell ratio ≥{ratio}x with volume",
    )

add(
    "exp_bsr_fresh_020",
    "buy_dominance",
    {"buy_sell_ratio_30s_min": 20.0, "token_age_sec_max": 90, "tx_count_30s_min": 3},
    "Fresh token + extreme buy dominance ≥20x",
)

add(
    "exp_bsr_fresh_050",
    "buy_dominance",
    {"buy_sell_ratio_30s_min": 50.0, "token_age_sec_max": 120, "tx_count_30s_min": 4},
    "Fresh token + extreme buy dominance ≥50x",
)

# ── Group D: Transaction Frequency (8 rules) ──────────────────────────────────
for tx in [8, 12, 20, 30, 50, 80]:
    add(
        f"exp_tx_{tx:03d}",
        "high_frequency",
        {"tx_count_30s_min": tx, "volume_sol_30s_min": 1.0},
        f"≥{tx} tx in 30s, ≥1 SOL volume",
    )

add(
    "exp_tx_020_pos",
    "high_frequency",
    {"tx_count_30s_min": 20, "price_change_30s_min": 0.05, "volume_sol_30s_min": 2.0},
    "High frequency trading with positive price move",
)

add(
    "exp_tx_050_vol",
    "high_frequency",
    {"tx_count_30s_min": 50, "volume_sol_30s_min": 5.0},
    "Very high frequency — strong interest signal",
)

# ── Group E: Average Trade Size / Whale Detection (8 rules) ──────────────────
for avg_sol in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
    add(
        f"exp_avgtr_{int(avg_sol * 10):02d}",
        "whale_signal",
        {
            "avg_trade_sol_30s_min": avg_sol,
            "tx_count_30s_min": 3,
            "volume_sol_30s_min": 1.0,
        },
        f"≥{avg_sol} SOL avg trade — large buyers present",
    )

add(
    "exp_whale_fresh",
    "whale_signal",
    {"avg_trade_sol_30s_min": 1.0, "token_age_sec_max": 90, "tx_count_30s_min": 3},
    "Large avg trade on fresh token — early whale entry",
)

# ── Group F: Net Flow Surge (7 rules) ────────────────────────────────────────
# virtual_sol_growth_60s_min uses net_flow_sol_60s as proxy at runtime
for nf in [2.0, 5.0, 10.0, 20.0, 40.0]:
    add(
        f"exp_netflow_{int(nf):02d}",
        "net_flow",
        {"virtual_sol_growth_60s_min": nf, "tx_count_30s_min": 3},
        f"≥{nf} SOL net buy flow in 60s",
    )

add(
    "exp_netflow_mom",
    "net_flow",
    {
        "virtual_sol_growth_60s_min": 5.0,
        "price_change_30s_min": 0.05,
        "tx_count_30s_min": 4,
    },
    "Net flow surge with positive price direction",
)

add(
    "exp_netflow_fresh",
    "net_flow",
    {
        "virtual_sol_growth_60s_min": 10.0,
        "token_age_sec_max": 90,
        "tx_count_30s_min": 3,
    },
    "Strong net flow on fresh token",
)

# ── Group G: Low Sell Pressure / Clean Sweep (8 rules) ───────────────────────
# Sellers absent — token in pure accumulation phase
for max_sell in [0, 1, 2, 3]:
    add(
        f"exp_nosell_{max_sell:02d}tx",
        "clean_sweep",
        {
            "sell_tx_count_30s_max": max_sell,
            "tx_count_30s_min": 4,
            "volume_sol_30s_min": 1.5,
        },
        f"≤{max_sell} sell txs in 30s with activity",
    )

add(
    "exp_nosell_fresh",
    "clean_sweep",
    {
        "sell_tx_count_30s_max": 1,
        "token_age_sec_max": 60,
        "tx_count_30s_min": 4,
        "volume_sol_30s_min": 2.0,
    },
    "No sellers on fresh token — pure accumulation",
)

add(
    "exp_nosell_whale",
    "clean_sweep",
    {"sell_tx_count_30s_max": 2, "avg_trade_sol_30s_min": 0.5, "tx_count_30s_min": 4},
    "No sellers + whale-sized buys",
)

add(
    "exp_nosell_dominant",
    "clean_sweep",
    {"sell_tx_count_30s_max": 0, "buy_volume_sol_30s_min": 3.0},
    "Zero sellers + solid buy volume — maximum clean sweep",
)

add(
    "exp_nosell_bsr",
    "clean_sweep",
    {
        "sell_tx_count_30s_max": 1,
        "buy_sell_ratio_30s_min": 20.0,
        "volume_sol_30s_min": 2.0,
    },
    "Almost no sellers + extreme buy ratio",
)

# ── Group H: Recovery Plays (7 rules) ────────────────────────────────────────
# Token dipped and bouncing — new buyers confirmed by cluster + volume
for pmin, pmax in [
    (-0.40, -0.10),
    (-0.30, 0.0),
    (-0.20, 0.10),
    (-0.15, 0.15),
    (-0.10, 0.20),
]:
    add(
        f"exp_recovery_{int(pmin * 100):+03d}_{int(pmax * 100):+03d}",
        "recovery",
        {
            "price_change_30s_min": pmin,
            "price_change_30s_max": pmax,
            "wallet_cluster_30s_min": 3,
            "volume_sol_30s_min": 3.0,
            "tx_count_30s_min": 4,
        },
        f"Price {pmin * 100:.0f}% to {pmax * 100:.0f}% with cluster confirmation",
    )

add(
    "exp_recovery_clean",
    "recovery",
    {
        "price_change_30s_min": -0.25,
        "price_change_30s_max": 0.0,
        "sell_tx_count_30s_max": 3,
        "tx_count_30s_min": 5,
        "volume_sol_30s_min": 4.0,
    },
    "Recovery + low sell pressure — capitulation complete",
)

add(
    "exp_recovery_whale",
    "recovery",
    {
        "price_change_30s_min": -0.30,
        "price_change_30s_max": 0.05,
        "avg_trade_sol_30s_min": 0.8,
        "tx_count_30s_min": 4,
        "volume_sol_30s_min": 3.0,
    },
    "Recovery + whale-sized buys entering the dip",
)

# ── Group I: Cluster + Volume Combos (9 rules) ───────────────────────────────
# Many unique buyers with significant volume — broad market participation
for clust, vol in [
    (3, 2.0),
    (4, 3.0),
    (5, 5.0),
    (6, 8.0),
    (8, 10.0),
    (10, 15.0),
    (12, 20.0),
    (15, 30.0),
]:
    add(
        f"exp_cluster_{clust:02d}_vol_{int(vol):02d}",
        "broad_participation",
        {
            "wallet_cluster_30s_min": clust,
            "volume_sol_30s_min": vol,
            "tx_count_30s_min": clust,
        },
        f"≥{clust} unique buyers, ≥{vol} SOL volume",
    )

add(
    "exp_cluster_fresh",
    "broad_participation",
    {"wallet_cluster_30s_min": 5, "token_age_sec_max": 90, "volume_sol_30s_min": 3.0},
    "Strong buyer cluster on fresh token",
)

# ── Group J: Combined High-Signal Rules (9 rules) ────────────────────────────
# Multiple strong signals together — lower fire rate, higher conviction
add(
    "exp_combo_strong_momentum",
    "combo_signal",
    {
        "price_change_30s_min": 0.15,
        "buy_sell_ratio_30s_min": 10.0,
        "volume_sol_30s_min": 5.0,
        "tx_count_30s_min": 8,
    },
    "Strong price move + buy dominance + volume",
)

add(
    "exp_combo_whale_fresh",
    "combo_signal",
    {
        "token_age_sec_max": 60,
        "avg_trade_sol_30s_min": 1.0,
        "buy_volume_sol_30s_min": 5.0,
    },
    "Fresh token + whales buying hard (≥1 SOL avg)",
)

add(
    "exp_combo_clean_momentum",
    "combo_signal",
    {
        "price_change_30s_min": 0.10,
        "sell_tx_count_30s_max": 2,
        "volume_sol_30s_min": 5.0,
        "tx_count_30s_min": 6,
    },
    "Rising price with almost no sellers",
)

add(
    "exp_combo_all_green",
    "combo_signal",
    {
        "price_change_30s_min": 0.05,
        "buy_sell_ratio_30s_min": 5.0,
        "wallet_cluster_30s_min": 4,
        "volume_sol_30s_min": 3.0,
        "tx_count_30s_min": 5,
    },
    "All signals green: price+ratio+cluster+volume",
)

add(
    "exp_combo_ultra_buy",
    "combo_signal",
    {
        "buy_sell_ratio_30s_min": 80.0,
        "buy_volume_sol_30s_min": 5.0,
        "token_age_sec_max": 90,
    },
    "Extreme buy ratio on fresh token — maximum buy pressure",
)

add(
    "exp_combo_flow_cluster",
    "combo_signal",
    {
        "virtual_sol_growth_60s_min": 8.0,
        "wallet_cluster_30s_min": 5,
        "tx_count_30s_min": 8,
    },
    "Strong net flow + many buyers + high frequency",
)

add(
    "exp_combo_fresh_explosion",
    "combo_signal",
    {
        "token_age_sec_max": 45,
        "tx_count_30s_min": 10,
        "volume_sol_30s_min": 5.0,
        "price_change_30s_min": 0.05,
    },
    "Ultra-fresh token exploding with activity",
)

add(
    "exp_combo_whale_dominant",
    "combo_signal",
    {
        "avg_trade_sol_30s_min": 2.0,
        "buy_sell_ratio_30s_min": 10.0,
        "volume_sol_30s_min": 8.0,
    },
    "Large whale buys overwhelming sellers",
)

add(
    "exp_combo_netflow_fresh_burst",
    "combo_signal",
    {
        "virtual_sol_growth_60s_min": 15.0,
        "token_age_sec_max": 60,
        "price_change_30s_min": 0.05,
        "tx_count_30s_min": 5,
    },
    "Massive net flow on fresh token with upward price",
)


def main() -> None:
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTFILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rules)

    from collections import Counter

    fams = Counter(r["family"] for r in rules)
    print(f"Written {len(rules)} rules to {OUTFILE}")
    print("Families:")
    for fam, cnt in sorted(fams.items()):
        print(f"  {fam:<25} {cnt:>3}")


if __name__ == "__main__":
    main()
