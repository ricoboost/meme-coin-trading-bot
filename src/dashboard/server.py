"""Live dashboard HTTP server.

Single-file Python server that serves a Tailwind/shadcn-style HTML
shell and streams live bot state via Server-Sent Events. The frontend
is vanilla JS + Chart.js — no React, no build step.

Endpoints
---------
GET /                       — dashboard HTML
GET /api/stream             — SSE tick feed (compact live snapshot)
GET /api/live               — one-shot `live_tick()` JSON (used on initial render)
GET /api/metrics            — hot-path latency metrics
GET /api/summary            — legacy summary (kept for detail views)
GET /api/open_positions     — legacy
GET /api/recent_positions   — legacy
GET /api/executions         — legacy
GET /api/rule_performance   — legacy
GET /api/events             — legacy
GET /api/rejections         — legacy
GET /api/rejection_summary  — legacy
GET /api/scoreboard         — session win/loss scoreboard + funnel + per-strategy
GET /api/pnl_series         — legacy
GET /api/rule_pnl_series    — legacy
GET /api/activity_series    — legacy
GET /api/token_detail       — legacy
GET /api/rule_detail        — legacy
GET /api/subscribed_wallets — legacy
GET /api/wallet_panel       — wallet-lane pool health + cluster activity + lane funnel
GET /api/health             — legacy
POST /api/session/new       — session control
POST /api/session/end       — session control
POST /api/positions/close   — manual force-close a single position (Sell Now)
"""

from __future__ import annotations

import errno
import json
import logging
import math
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from src.dashboard.data import DashboardDataStore

logger = logging.getLogger(__name__)


# Dashboard HTML — Tailwind CDN + Chart.js + vanilla JS client.
# All state updates flow through /api/stream; the JSON filter endpoints
# stay available for the detail panels.
_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en" class="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Memybot · Live</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          colors: {
            bg: "#09090b",
            panel: "#0f0f12",
            panel2: "#151519",
            border: "#27272a",
            borderhi: "#3f3f46",
            mute: "#71717a",
            mute2: "#a1a1aa",
            fg: "#e4e4e7",
            accent: "#22c55e",
            danger: "#ef4444",
            warn: "#f59e0b",
            blue: "#3b82f6",
            purple: "#a855f7",
          },
          fontFamily: {
            sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
            mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
          },
        },
      },
    };
  </script>
  <style>
    html, body { background: #09090b; color: #e4e4e7; }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #27272a; border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: #3f3f46; }
    .card { background: linear-gradient(180deg, #0f0f12 0%, #0c0c0f 100%); }
    .pulse-dot::after {
      content: "";
      position: absolute; inset: 0; border-radius: 9999px;
      box-shadow: 0 0 0 0 currentColor;
      animation: pulse-ring 1.8s cubic-bezier(0.24, 0, 0.38, 1) infinite;
    }
    @keyframes pulse-ring {
      0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.45); }
      70% { box-shadow: 0 0 0 8px rgba(34, 197, 94, 0); }
      100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }
    .num { font-variant-numeric: tabular-nums; }
    .sparkline { height: 28px; width: 100%; }
  </style>
</head>
<body class="font-sans antialiased">
  <!-- Top bar ------------------------------------------------------------ -->
  <header class="sticky top-0 z-30 border-b border-border bg-bg/90 backdrop-blur">
    <div class="mx-auto flex h-14 max-w-[1600px] items-center gap-4 px-6">
      <div class="flex items-center gap-2">
        <div class="h-7 w-7 rounded-md bg-gradient-to-br from-accent to-blue flex items-center justify-center">
          <svg viewBox="0 0 24 24" class="h-4 w-4 text-bg" fill="none" stroke="currentColor" stroke-width="2.5">
            <path d="M3 17l6-6 4 4 8-8" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </div>
        <span class="font-semibold tracking-tight">memybot</span>
        <span class="text-xs text-mute">live</span>
      </div>

      <nav class="ml-6 flex items-center gap-1 text-sm text-mute2">
        <a href="#scoreboard" class="rounded-md px-2 py-1 hover:bg-panel2 hover:text-fg">Scoreboard</a>
        <a href="#pnl" class="rounded-md px-2 py-1 hover:bg-panel2 hover:text-fg">PnL</a>
        <a href="#funnel" class="rounded-md px-2 py-1 hover:bg-panel2 hover:text-fg">Funnel</a>
        <a href="#positions" class="rounded-md px-2 py-1 hover:bg-panel2 hover:text-fg">Positions</a>
        <a href="#metrics" class="rounded-md px-2 py-1 hover:bg-panel2 hover:text-fg">Metrics</a>
        <a href="#feed" class="rounded-md px-2 py-1 hover:bg-panel2 hover:text-fg">Feed</a>
        <a href="#events" class="rounded-md px-2 py-1 hover:bg-panel2 hover:text-fg">Events</a>
      </nav>

      <div class="ml-auto flex items-center gap-3">
        <div id="stream-status" class="flex items-center gap-2 rounded-md border border-border bg-panel px-2.5 py-1 text-xs">
          <span class="relative h-2 w-2 rounded-full bg-mute pulse-dot" id="stream-dot"></span>
          <span id="stream-label" class="text-mute2">connecting…</span>
        </div>
        <div id="session-info" class="text-xs text-mute"></div>
        <button id="btn-session-new" class="rounded-md border border-border bg-panel px-3 py-1.5 text-xs hover:bg-panel2">New session</button>
        <button id="btn-session-end" class="rounded-md border border-border bg-panel px-3 py-1.5 text-xs hover:bg-panel2">End</button>
      </div>
    </div>
  </header>

  <main class="mx-auto max-w-[1600px] space-y-6 px-6 py-6">
    <!-- KPI row -->
    <section class="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
      <div class="card rounded-xl border border-border p-4">
        <div class="flex items-center justify-between">
          <div class="text-[11px] uppercase tracking-wider text-mute">Unrealized PnL</div>
          <div id="kpi-ts" class="text-[10px] text-mute"></div>
        </div>
        <div id="kpi-unrealized" class="mt-2 text-3xl font-semibold num">—</div>
        <div id="kpi-exposure" class="mt-1 text-xs text-mute">exposure —</div>
      </div>
      <div class="card rounded-xl border border-border p-4">
        <div class="text-[11px] uppercase tracking-wider text-mute">Realized PnL</div>
        <div id="kpi-realized" class="mt-2 text-3xl font-semibold num">—</div>
        <div id="kpi-closed" class="mt-1 text-xs text-mute">closed —</div>
      </div>
      <div class="card rounded-xl border border-border p-4">
        <div class="text-[11px] uppercase tracking-wider text-mute">Open positions</div>
        <div id="kpi-open" class="mt-2 text-3xl font-semibold num">—</div>
        <div id="kpi-openhint" class="mt-1 text-xs text-mute">streaming</div>
      </div>
      <div class="card rounded-xl border border-border p-4">
        <div class="text-[11px] uppercase tracking-wider text-mute">Executions</div>
        <div class="mt-2 flex items-baseline gap-3">
          <div><span id="kpi-buys" class="text-3xl font-semibold num">—</span><span class="ml-1 text-xs text-mute">buys</span></div>
          <div><span id="kpi-sells" class="text-3xl font-semibold num">—</span><span class="ml-1 text-xs text-mute">sells</span></div>
        </div>
        <div class="mt-1 flex items-baseline gap-3 text-xs text-mute">
          <span>avoided entry <span id="kpi-avoided" class="text-fg num">—</span></span>
          <span>sell fail <span id="kpi-sellfail" class="text-danger num">—</span></span>
        </div>
      </div>
    </section>

    <!-- Scoreboard (wins / losses / rate / avg / best / worst) -->
    <section id="scoreboard" class="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
      <div class="card rounded-xl border border-border p-4">
        <div class="text-[11px] uppercase tracking-wider text-mute">Win rate</div>
        <div id="sb-winrate" class="mt-2 text-3xl font-semibold num">—</div>
        <div id="sb-wlsplit" class="mt-1 text-xs text-mute">— W / — L</div>
      </div>
      <div class="card rounded-xl border border-border p-4">
        <div class="text-[11px] uppercase tracking-wider text-mute">Closed</div>
        <div id="sb-closed" class="mt-2 text-3xl font-semibold num">—</div>
        <div id="sb-breakeven" class="mt-1 text-xs text-mute">be —</div>
      </div>
      <div class="card rounded-xl border border-border p-4">
        <div class="text-[11px] uppercase tracking-wider text-mute">Avg PnL</div>
        <div id="sb-avg" class="mt-2 text-3xl font-semibold num">—</div>
        <div id="sb-realized" class="mt-1 text-xs text-mute">realized —</div>
      </div>
      <div class="card rounded-xl border border-border p-4">
        <div class="text-[11px] uppercase tracking-wider text-mute">Best</div>
        <div id="sb-best" class="mt-2 text-xl font-semibold num text-accent">—</div>
        <div id="sb-best-mint" class="mt-1 font-mono text-[11px] text-mute">—</div>
      </div>
      <div class="card rounded-xl border border-border p-4">
        <div class="text-[11px] uppercase tracking-wider text-mute">Worst</div>
        <div id="sb-worst" class="mt-2 text-xl font-semibold num text-danger">—</div>
        <div id="sb-worst-mint" class="mt-1 font-mono text-[11px] text-mute">—</div>
      </div>
      <div class="card rounded-xl border border-border p-4">
        <div class="text-[11px] uppercase tracking-wider text-mute">Hit 2× / 5×</div>
        <div id="sb-hits" class="mt-2 text-3xl font-semibold num">—</div>
        <div id="sb-stops" class="mt-1 text-xs text-mute">stops —</div>
      </div>
    </section>

    <!-- Per-strategy breakdown (sniper vs main) -->
    <section id="strategy-breakdown" class="card rounded-xl border border-border p-5">
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-sm font-semibold tracking-tight">Per-strategy</h2>
          <p class="mt-0.5 text-xs text-mute">closed-trade split by lane</p>
        </div>
      </div>
      <div id="strategy-body" class="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3"></div>
    </section>

    <!-- Wallet lane monitor -->
    <section id="wallet-panel" class="card rounded-xl border border-border p-5">
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-sm font-semibold tracking-tight">Wallet lane</h2>
          <p class="mt-0.5 text-xs text-mute">provider_a cluster monitor · last 15 min</p>
        </div>
        <div id="wallet-panel-status" class="text-xs text-mute">—</div>
      </div>
      <div class="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-3">
        <div class="rounded-lg border border-border bg-panel p-4">
          <div class="text-[11px] uppercase tracking-wider text-mute">Pool health</div>
          <dl id="wallet-pool-grid" class="mt-3 grid grid-cols-2 gap-2 text-xs"></dl>
        </div>
        <div class="rounded-lg border border-border bg-panel p-4">
          <div class="text-[11px] uppercase tracking-wider text-mute">Live cluster activity</div>
          <dl id="wallet-activity-grid" class="mt-3 grid grid-cols-2 gap-2 text-xs"></dl>
          <div class="mt-3 text-[11px] uppercase tracking-wider text-mute">Top wallets</div>
          <ul id="wallet-top-list" class="mt-2 space-y-1 text-xs text-mute2"></ul>
        </div>
        <div class="rounded-lg border border-border bg-panel p-4">
          <div class="text-[11px] uppercase tracking-wider text-mute">Lane funnel</div>
          <dl id="wallet-funnel-grid" class="mt-3 grid grid-cols-2 gap-2 text-xs"></dl>
          <div class="mt-3 text-[11px] uppercase tracking-wider text-mute">Top rejections</div>
          <ul id="wallet-rejections-list" class="mt-2 space-y-1 text-xs text-mute2"></ul>
        </div>
      </div>
    </section>

    <!-- PnL chart -->
    <section id="pnl" class="card rounded-xl border border-border p-5">
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-sm font-semibold tracking-tight">Session PnL</h2>
          <p class="mt-0.5 text-xs text-mute">cumulative realized · current open unrealized overlay</p>
        </div>
        <div class="flex items-center gap-3 text-xs">
          <span class="flex items-center gap-1.5"><span class="h-2 w-2 rounded-full bg-accent"></span>realized</span>
          <span class="flex items-center gap-1.5"><span class="h-2 w-2 rounded-full bg-blue"></span>combined</span>
        </div>
      </div>
      <div class="mt-4 h-64"><canvas id="chart-pnl"></canvas></div>
    </section>

    <!-- Candidate funnel -->
    <section id="funnel" class="card rounded-xl border border-border p-5">
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-sm font-semibold tracking-tight">Candidate funnel</h2>
          <p class="mt-0.5 text-xs text-mute">events → candidates → entries → wins</p>
        </div>
      </div>
      <div id="funnel-stages" class="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6"></div>
      <div class="mt-4 grid grid-cols-1 gap-3 xl:grid-cols-2">
        <div class="rounded-lg border border-border bg-panel p-3">
          <div class="text-[11px] uppercase tracking-wider text-mute">Top rejection reasons</div>
          <ul id="funnel-reasons" class="mt-2 space-y-1 text-xs text-mute2"></ul>
        </div>
        <div class="rounded-lg border border-border bg-panel p-3">
          <div class="text-[11px] uppercase tracking-wider text-mute">Drop breakdown</div>
          <div id="funnel-drops" class="mt-2 grid grid-cols-2 gap-2 text-xs text-mute2"></div>
        </div>
      </div>
    </section>

    <!-- Hot-path metrics -->
    <section id="metrics" class="card rounded-xl border border-border p-5">
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-sm font-semibold tracking-tight">Hot-path latency</h2>
          <p class="mt-0.5 text-xs text-mute">p50 / p95 / last over the last 5 minutes</p>
        </div>
        <div id="metrics-window" class="text-xs text-mute">—</div>
      </div>
      <div id="metrics-grid" class="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"></div>
    </section>

    <!-- Execution graph + Feed health -->
    <section class="grid grid-cols-1 gap-4 xl:grid-cols-3">
      <div class="card col-span-1 rounded-xl border border-border p-5 xl:col-span-2">
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-sm font-semibold tracking-tight">Execution timeline</h2>
            <p class="mt-0.5 text-xs text-mute">buy_total_ms vs sell_total_ms · last 60 samples</p>
          </div>
          <div class="flex items-center gap-3 text-xs">
            <span class="flex items-center gap-1.5"><span class="h-2 w-2 rounded-full bg-accent"></span>buy</span>
            <span class="flex items-center gap-1.5"><span class="h-2 w-2 rounded-full bg-danger"></span>sell</span>
          </div>
        </div>
        <div class="mt-4 h-64"><canvas id="chart-exec"></canvas></div>
      </div>

      <div id="feed" class="card rounded-xl border border-border p-5">
        <h2 class="text-sm font-semibold tracking-tight">Feed health</h2>
        <p class="mt-0.5 text-xs text-mute">yellowstone + parser queue</p>
        <dl id="feed-grid" class="mt-4 grid grid-cols-2 gap-3 text-xs"></dl>
      </div>
    </section>

    <!-- Open positions -->
    <section id="positions" class="card rounded-xl border border-border">
      <div class="flex items-center justify-between border-b border-border px-5 py-4">
        <div>
          <h2 class="text-sm font-semibold tracking-tight">Open positions</h2>
          <p class="mt-0.5 text-xs text-mute">live marks from exit engine</p>
        </div>
        <div id="positions-count" class="text-xs text-mute">—</div>
      </div>
      <div class="overflow-x-auto">
        <table class="min-w-full text-sm">
          <thead class="text-left text-[11px] uppercase tracking-wider text-mute">
            <tr class="border-b border-border">
              <th class="px-5 py-3 font-medium">Mint</th>
              <th class="px-5 py-3 font-medium">Strategy</th>
              <th class="px-5 py-3 font-medium">Rule</th>
              <th class="px-5 py-3 font-medium text-right">Size</th>
              <th class="px-5 py-3 font-medium text-right">Entry</th>
              <th class="px-5 py-3 font-medium text-right">Unreal. PnL</th>
              <th class="px-5 py-3 font-medium text-right">Age</th>
              <th class="px-5 py-3 font-medium">Stage</th>
              <th class="px-5 py-3 font-medium text-right">Action</th>
            </tr>
          </thead>
          <tbody id="positions-body" class="divide-y divide-border text-mute2"></tbody>
        </table>
      </div>
    </section>

    <!-- Closed positions -->
    <section id="closed-positions" class="card rounded-xl border border-border">
      <div class="flex items-center justify-between border-b border-border px-5 py-4">
        <div>
          <h2 class="text-sm font-semibold tracking-tight">Closed positions</h2>
          <p class="mt-0.5 text-xs text-mute">this session · latest 30</p>
        </div>
        <div id="closed-count" class="text-xs text-mute">—</div>
      </div>
      <div class="overflow-x-auto">
        <table class="min-w-full text-sm">
          <thead class="text-left text-[11px] uppercase tracking-wider text-mute">
            <tr class="border-b border-border">
              <th class="px-5 py-3 font-medium">Mint</th>
              <th class="px-5 py-3 font-medium">Strategy</th>
              <th class="px-5 py-3 font-medium">Rule</th>
              <th class="px-5 py-3 font-medium text-right">Size</th>
              <th class="px-5 py-3 font-medium text-right">PnL</th>
              <th class="px-5 py-3 font-medium text-right">PnL %</th>
              <th class="px-5 py-3 font-medium text-right">Hold</th>
              <th class="px-5 py-3 font-medium">Closed</th>
            </tr>
          </thead>
          <tbody id="closed-body" class="divide-y divide-border text-mute2"></tbody>
        </table>
      </div>
    </section>

    <!-- Live activity feed -->
    <section id="events" class="card rounded-xl border border-border p-5">
      <div class="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 class="text-sm font-semibold tracking-tight">Live activity</h2>
          <p class="mt-0.5 text-xs text-mute">structured event tail · click chips to filter</p>
        </div>
        <div id="events-chips" class="flex flex-wrap items-center gap-1.5 text-xs"></div>
      </div>
      <div id="events-body" class="mt-4 space-y-2 font-mono text-xs text-mute2"></div>
    </section>
  </main>

  <script>
    // ---------- helpers ----------
    const fmtSol = (v, digits = 4) => {
      if (v === null || v === undefined || Number.isNaN(+v)) return "—";
      const n = Number(v);
      const sign = n > 0 ? "+" : "";
      return sign + n.toFixed(digits) + " SOL";
    };
    const fmtMs = (v) => {
      if (v === null || v === undefined || Number.isNaN(+v)) return "—";
      const n = Number(v);
      if (n < 1) return n.toFixed(2) + " ms";
      if (n < 100) return n.toFixed(1) + " ms";
      if (n < 1000) return n.toFixed(0) + " ms";
      return (n / 1000).toFixed(2) + " s";
    };
    const fmtNum = (v) => (v === null || v === undefined || Number.isNaN(+v)) ? "—" : Number(v).toLocaleString();
    const shortMint = (m) => (m || "").length > 12 ? m.slice(0, 6) + "…" + m.slice(-4) : (m || "—");
    function mintCell(mint) {
      const m = String(mint || "");
      if (!m) return `<td class="px-5 py-3 font-mono text-xs text-fg">—</td>`;
      const ds = `https://dexscreener.com/solana/${encodeURIComponent(m)}`;
      const ax = `https://axiom.trade/meme/${encodeURIComponent(m)}`;
      const stop = `event.stopPropagation()`;
      return `<td class="px-5 py-3 font-mono text-xs text-fg">
        <div class="flex items-center gap-1.5">
          <a href="${ds}" target="_blank" rel="noopener noreferrer" onclick="${stop}"
             title="Open on DexScreener — ${m}"
             class="text-accent hover:underline">${shortMint(m)}</a>
          <a href="${ax}" target="_blank" rel="noopener noreferrer" onclick="${stop}"
             title="Open on Axiom"
             class="rounded bg-panel2 px-1 text-[9px] uppercase tracking-wider text-mute2 hover:text-fg">Ax</a>
          <button type="button" onclick="${stop};navigator.clipboard.writeText('${m}')"
             title="Copy mint to clipboard"
             class="rounded bg-panel2 px-1 text-[9px] uppercase tracking-wider text-mute2 hover:text-fg">Copy</button>
        </div>
      </td>`;
    }
    const relTime = (iso) => {
      if (!iso) return "—";
      const t = new Date(iso).getTime();
      if (!t) return "—";
      const s = Math.max(0, (Date.now() - t) / 1000);
      if (s < 60) return Math.round(s) + "s";
      if (s < 3600) return Math.round(s / 60) + "m";
      if (s < 86400) return Math.round(s / 3600) + "h";
      return Math.round(s / 86400) + "d";
    };
    const pnlClass = (v) => {
      if (v === null || v === undefined || Number.isNaN(+v)) return "text-mute";
      const n = Number(v);
      if (n > 0.0001) return "text-accent";
      if (n < -0.0001) return "text-danger";
      return "text-mute2";
    };

    // ---------- stream status indicator ----------
    const streamDot = document.getElementById("stream-dot");
    const streamLabel = document.getElementById("stream-label");
    function setStreamState(state) {
      streamDot.classList.remove("bg-mute", "bg-accent", "bg-danger", "bg-warn", "pulse-dot");
      if (state === "live") { streamDot.classList.add("bg-accent", "pulse-dot"); streamLabel.textContent = "live"; streamLabel.className = "text-accent"; }
      else if (state === "reconnecting") { streamDot.classList.add("bg-warn"); streamLabel.textContent = "reconnecting"; streamLabel.className = "text-warn"; }
      else if (state === "error") { streamDot.classList.add("bg-danger"); streamLabel.textContent = "offline"; streamLabel.className = "text-danger"; }
      else { streamDot.classList.add("bg-mute"); streamLabel.textContent = state; streamLabel.className = "text-mute2"; }
    }

    // ---------- KPI row ----------
    function renderKPIs(summary) {
      const s = summary || {};
      const unrealized = Number(s.unrealized_pnl_sol ?? 0);
      const realized = Number(s.realized_pnl_sol ?? 0);
      const elU = document.getElementById("kpi-unrealized");
      const elR = document.getElementById("kpi-realized");
      elU.textContent = fmtSol(unrealized);
      elU.className = "mt-2 text-3xl font-semibold num " + pnlClass(unrealized);
      elR.textContent = fmtSol(realized);
      elR.className = "mt-2 text-3xl font-semibold num " + pnlClass(realized);
      document.getElementById("kpi-exposure").textContent = "exposure " + fmtSol(s.open_exposure_sol || 0, 3);
      document.getElementById("kpi-closed").textContent = "closed " + fmtNum(s.closed_positions || 0);
      document.getElementById("kpi-open").textContent = fmtNum(s.open_positions || 0);
      document.getElementById("kpi-buys").textContent = fmtNum(s.buy_count || 0);
      document.getElementById("kpi-sells").textContent = fmtNum(s.sell_count || 0);
      document.getElementById("kpi-avoided").textContent = fmtNum(s.avoided_entry_count || 0);
      document.getElementById("kpi-sellfail").textContent = fmtNum(s.sell_failed_count || 0);
      document.getElementById("kpi-ts").textContent = s.last_execution_at ? "last exec " + relTime(s.last_execution_at) : "";

      const sessionEl = document.getElementById("session-info");
      if (s.active_session_id) {
        const label = s.active_session_label ? " · " + s.active_session_label : "";
        sessionEl.textContent = "session #" + s.active_session_id + label;
      } else {
        sessionEl.textContent = "no session";
      }
    }

    // ---------- hot-path metrics ----------
    const STAGE_LABELS = {
      event_to_runner_ms: "Event → runner",
      feature_build_ms: "Feature build",
      arrival_to_dispatch_ms: "Arrival → dispatch",
      buy_total_ms: "Buy total",
      sell_total_ms: "Sell total",
      broadcast_send_ms: "Broadcast send",
      broadcast_confirm_ms: "Broadcast confirm",
      jupiter_order_ms: "Jupiter order",
      preflight_ms: "Preflight sim",
      reconcile_ms: "Reconcile",
    };
    const STAGE_ORDER = [
      "event_to_runner_ms",
      "arrival_to_dispatch_ms",
      "buy_total_ms",
      "sell_total_ms",
      "broadcast_send_ms",
      "broadcast_confirm_ms",
      "jupiter_order_ms",
      "preflight_ms",
      "reconcile_ms",
      "feature_build_ms",
    ];
    const stageSparklines = new Map();
    function renderMetrics(metrics) {
      if (!metrics) return;
      document.getElementById("metrics-window").textContent = "window " + Math.round(metrics.window_sec || 0) + "s";
      const grid = document.getElementById("metrics-grid");
      const stages = metrics.stages || {};
      // Build/update each card
      const cards = STAGE_ORDER.filter((k) => stages[k]);
      // Ensure cards exist
      const existing = new Set(Array.from(grid.children).map((el) => el.dataset.stage));
      cards.forEach((key) => {
        if (!existing.has(key)) {
          const card = document.createElement("div");
          card.dataset.stage = key;
          card.className = "rounded-lg border border-border bg-panel p-3";
          card.innerHTML = `
            <div class="flex items-center justify-between">
              <div class="text-[11px] uppercase tracking-wider text-mute">${STAGE_LABELS[key] || key}</div>
              <div class="text-[10px] text-mute" data-role="count">—</div>
            </div>
            <div class="mt-1 flex items-baseline gap-2">
              <div class="text-xl font-semibold num" data-role="p50">—</div>
              <div class="text-[11px] text-mute">p50</div>
              <div class="ml-auto text-xs text-mute2 num" data-role="p95">—</div>
              <div class="text-[10px] text-mute">p95</div>
            </div>
            <canvas class="sparkline mt-2" data-role="spark"></canvas>
            <div class="mt-1 flex justify-between text-[10px] text-mute"><span data-role="last">last —</span><span data-role="p99">p99 —</span></div>
          `;
          grid.appendChild(card);
        }
      });
      // Remove stale cards
      Array.from(grid.children).forEach((el) => {
        if (!cards.includes(el.dataset.stage)) el.remove();
      });

      // Update content
      cards.forEach((key) => {
        const s = stages[key];
        const card = grid.querySelector(`[data-stage="${key}"]`);
        if (!card || !s) return;
        card.querySelector('[data-role="p50"]').textContent = fmtMs(s.p50);
        card.querySelector('[data-role="p95"]').textContent = fmtMs(s.p95);
        card.querySelector('[data-role="p99"]').textContent = "p99 " + fmtMs(s.p99);
        card.querySelector('[data-role="last"]').textContent = "last " + fmtMs(s.last);
        card.querySelector('[data-role="count"]').textContent = "n " + s.count;
        const canvas = card.querySelector('[data-role="spark"]');
        drawSpark(canvas, (s.samples || []).map((p) => p[1]));
      });
    }

    function drawSpark(canvas, values) {
      const ctx = canvas.getContext("2d");
      const w = canvas.width = canvas.clientWidth * devicePixelRatio;
      const h = canvas.height = canvas.clientHeight * devicePixelRatio;
      ctx.clearRect(0, 0, w, h);
      if (!values.length) return;
      const min = Math.min(...values);
      const max = Math.max(...values);
      const range = (max - min) || 1;
      ctx.strokeStyle = "#22c55e";
      ctx.lineWidth = 1.5 * devicePixelRatio;
      ctx.beginPath();
      values.forEach((v, i) => {
        const x = (i / Math.max(1, values.length - 1)) * w;
        const y = h - ((v - min) / range) * h;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      // Fill under the line
      ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
      ctx.fillStyle = "rgba(34, 197, 94, 0.08)";
      ctx.fill();
    }

    // ---------- execution timeline chart ----------
    let execChart = null;
    function renderExecChart(metrics) {
      const stages = metrics?.stages || {};
      const buy = (stages.buy_total_ms?.samples || []).map((p) => ({ x: p[0] * 1000, y: p[1] }));
      const sell = (stages.sell_total_ms?.samples || []).map((p) => ({ x: p[0] * 1000, y: p[1] }));
      if (!execChart) {
        const ctx = document.getElementById("chart-exec").getContext("2d");
        execChart = new Chart(ctx, {
          type: "line",
          data: {
            datasets: [
              { label: "buy", data: buy, borderColor: "#22c55e", backgroundColor: "rgba(34,197,94,0.08)", tension: 0.35, pointRadius: 0, borderWidth: 1.5, fill: true },
              { label: "sell", data: sell, borderColor: "#ef4444", backgroundColor: "rgba(239,68,68,0.08)", tension: 0.35, pointRadius: 0, borderWidth: 1.5, fill: true },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "nearest", intersect: false },
            plugins: { legend: { display: false }, tooltip: { backgroundColor: "#0f0f12", borderColor: "#27272a", borderWidth: 1, titleColor: "#e4e4e7", bodyColor: "#a1a1aa" } },
            scales: {
              x: { type: "time", time: { unit: "minute" }, ticks: { color: "#71717a", maxTicksLimit: 6 }, grid: { color: "#1a1a1e" } },
              y: { ticks: { color: "#71717a", callback: (v) => v >= 1000 ? (v / 1000).toFixed(1) + "s" : v + "ms" }, grid: { color: "#1a1a1e" } },
            },
          },
        });
      } else {
        execChart.data.datasets[0].data = buy;
        execChart.data.datasets[1].data = sell;
        execChart.update("none");
      }
    }

    // ---------- positions table ----------
    function positionDetailHtml(p) {
      const meta = (p && p.metadata_json) || {};
      const matched = Array.isArray(p.matched_rule_ids) ? p.matched_rule_ids : [];
      const guardKeys = [
        "paper_entry_guard_passed",
        "paper_entry_guard_roundtrip_ratio",
        "paper_entry_guard_roundtrip_pnl_sol",
        "paper_entry_guard_entry_cost_sol",
        "paper_entry_guard_immediate_exit_net_sol",
        "paper_entry_guard_min_roundtrip_ratio",
        "paper_entry_guard_max_price_impact_pct",
      ];
      const guardRows = guardKeys
        .filter((k) => meta[k] !== undefined && meta[k] !== null)
        .map((k) => `<div><dt class="text-[10px] uppercase tracking-wider text-mute">${k.replace("paper_entry_guard_", "")}</dt><dd class="mt-0.5 text-fg num">${typeof meta[k] === "number" ? Number(meta[k]).toFixed(6) : String(meta[k])}</dd></div>`)
        .join("");
      const interestingMeta = Object.entries(meta).filter(([k]) =>
        !guardKeys.includes(k) && !k.startsWith("paper_entry_guard_")
      ).slice(0, 12);
      const metaRows = interestingMeta
        .map(([k, v]) => `<div><dt class="text-[10px] uppercase tracking-wider text-mute">${k}</dt><dd class="mt-0.5 truncate text-fg num" title="${String(v).replace(/"/g, "&quot;")}">${typeof v === "object" ? JSON.stringify(v).slice(0, 80) : String(v)}</dd></div>`)
        .join("");
      const wallet = p.triggering_wallet
        ? `<div><dt class="text-[10px] uppercase tracking-wider text-mute">triggering wallet</dt><dd class="mt-0.5 truncate font-mono text-[11px] text-fg" title="${p.triggering_wallet}">${p.triggering_wallet}</dd></div><div><dt class="text-[10px] uppercase tracking-wider text-mute">wallet score</dt><dd class="mt-0.5 text-fg num">${Number(p.triggering_wallet_score || 0).toFixed(4)}</dd></div>`
        : "";
      const regime = p.selected_regime ? `<div><dt class="text-[10px] uppercase tracking-wider text-mute">regime</dt><dd class="mt-0.5 text-fg">${p.selected_regime}</dd></div>` : "";
      const matchedBlock = matched.length
        ? `<div class="col-span-full"><dt class="text-[10px] uppercase tracking-wider text-mute">matched rules (${matched.length})</dt><dd class="mt-1 flex flex-wrap gap-1">${matched.slice(0, 20).map((r) => `<span class="rounded bg-panel2 px-1.5 py-0.5 font-mono text-[10px] text-mute2">${r}</span>`).join("")}</dd></div>`
        : "";
      const guardSection = guardRows ? `<div class="col-span-full mt-2"><div class="mb-1 text-[10px] uppercase tracking-wider text-accent">entry guard</div><div class="grid grid-cols-2 gap-x-3 gap-y-1 sm:grid-cols-3 md:grid-cols-4">${guardRows}</div></div>` : "";
      const metaSection = metaRows ? `<div class="col-span-full mt-2"><div class="mb-1 text-[10px] uppercase tracking-wider text-mute">metadata</div><div class="grid grid-cols-2 gap-x-3 gap-y-1 sm:grid-cols-3 md:grid-cols-4">${metaRows}</div></div>` : "";
      return `<div class="grid grid-cols-2 gap-x-3 gap-y-1 sm:grid-cols-3 md:grid-cols-4">${wallet}${regime}${matchedBlock}${guardSection}${metaSection}</div>`;
    }

    // Persist expanded-row state across table re-renders so opening a detail
    // panel doesn't auto-collapse on the next refresh tick.
    const expandedKeys = { pos: new Set(), closed: new Set(), ev: new Set() };
    function attachRowExpand(tbody, group) {
      tbody.querySelectorAll("tr[data-row]").forEach((row) => {
        const key = row.dataset.row;
        const detail = tbody.querySelector(`tr[data-detail="${key}"]`);
        if (!detail) return;
        if (group && expandedKeys[group] && expandedKeys[group].has(key)) {
          detail.classList.remove("hidden");
        }
        row.addEventListener("click", () => {
          detail.classList.toggle("hidden");
          if (!group || !expandedKeys[group]) return;
          if (detail.classList.contains("hidden")) expandedKeys[group].delete(key);
          else expandedKeys[group].add(key);
        });
      });
    }
    function safeKey(s) {
      return String(s || "").replace(/[^A-Za-z0-9_-]/g, "_");
    }

    function renderPositions(positions) {
      const body = document.getElementById("positions-body");
      const countEl = document.getElementById("positions-count");
      countEl.textContent = positions.length + " open";
      if (!positions.length) {
        body.innerHTML = `<tr><td colspan="9" class="px-5 py-8 text-center text-mute">no open positions</td></tr>`;
        return;
      }
      body.innerHTML = positions.map((p, i) => {
        // Prefer the exit-engine-sourced multiple (always in correct units,
        // outlier-guarded). Only fall back to DB unrealized_pnl_sol when the
        // metadata has not produced a tick yet (brand-new positions).
        const hasLive = p.live_pnl_sol != null;
        const pnl = hasLive ? Number(p.live_pnl_sol) : Number(p.unrealized_pnl_sol ?? 0);
        const pnlPct = p.size_sol > 0 ? (pnl / p.size_sol) * 100 : null;
        const cls = pnlClass(pnl);
        const key = "op-" + safeKey(p.token_mint || i);
        const mint = String(p.token_mint || "");
        const entryPx = Number(p.entry_price_sol);
        const markPx = p.mark_price_sol != null ? Number(p.mark_price_sol) : null;
        const markPct = (markPx != null && entryPx > 0) ? ((markPx - entryPx) / entryPx) * 100 : null;
        const priceCell = markPx != null
          ? `<div class="text-right text-mute2">${entryPx.toExponential(2)}</div>
             <div class="text-right text-[10px] ${markPct != null && markPct >= 0 ? "text-ok" : "text-danger"}">
               ⚡ ${markPx.toExponential(2)}${markPct != null ? ` (${markPct >= 0 ? "+" : ""}${markPct.toFixed(1)}%)` : ""}
             </div>`
          : `<div class="text-right text-mute2">${entryPx.toExponential(2)}</div>
             <div class="text-right text-[10px] text-mute">—</div>`;
        const srcTip = p.live_source === "amm_quote"
          ? "From pool reserves (matches Jupiter)"
          : p.live_source === "metadata"
          ? "From last-trade price (exit-engine tick; may lag Jupiter after a whale trade)"
          : "No tick yet";
        const liveBadge = !hasLive
          ? `<span class="ml-1 rounded bg-panel2 px-1 text-[9px] uppercase tracking-wider text-mute" title="${srcTip}">pending</span>`
          : (p.live_price_stale
            ? `<span class="ml-1 rounded bg-warn/10 px-1 text-[9px] uppercase tracking-wider text-warn" title="${srcTip} — last tick >20s ago">stale</span>`
            : (p.live_source === "amm_quote"
              ? `<span class="ml-1 rounded bg-ok/10 px-1 text-[9px] uppercase tracking-wider text-ok" title="${srcTip}">live</span>`
              : `<span class="ml-1 rounded bg-accent/10 px-1 text-[9px] uppercase tracking-wider text-accent" title="${srcTip}">tick</span>`));
        return `
          <tr data-row="${key}" class="cursor-pointer hover:bg-panel2">
            ${mintCell(p.token_mint)}
            <td class="px-5 py-3"><span class="rounded-md bg-panel2 px-1.5 py-0.5 text-[11px]">${p.strategy_id || "main"}</span></td>
            <td class="px-5 py-3 font-mono text-xs">${p.selected_rule_id || "—"}</td>
            <td class="px-5 py-3 text-right num">${Number(p.size_sol).toFixed(3)}</td>
            <td class="px-5 py-3 num">${priceCell}</td>
            <td class="px-5 py-3 text-right num ${cls}">
              ${fmtSol(pnl)}${pnlPct !== null ? ` <span class="text-[10px] text-mute">(${pnlPct >= 0 ? "+" : ""}${pnlPct.toFixed(1)}%)</span>` : ""}${liveBadge}
            </td>
            <td class="px-5 py-3 text-right text-mute">${relTime(p.entry_time)}</td>
            <td class="px-5 py-3 text-[11px] text-mute2">stage ${p.exit_stage ?? 0}</td>
            <td class="px-5 py-3 text-right">
              <button type="button" data-sell-now="${mint}"
                class="rounded-md border border-danger/40 bg-danger/10 px-2 py-1 text-[11px] font-medium text-danger hover:bg-danger/20 disabled:opacity-50 disabled:cursor-not-allowed">
                Sell now
              </button>
            </td>
          </tr>
          <tr data-detail="${key}" class="hidden bg-panel/50">
            <td colspan="9" class="px-5 py-3 text-[11px]">${positionDetailHtml(p)}</td>
          </tr>
        `;
      }).join("");
      attachRowExpand(body, "pos");
      attachSellNowHandlers(body);
    }

    async function sellNowRequest(tokenMint, btn) {
      if (!tokenMint) return;
      const label = btn.textContent;
      btn.disabled = true;
      btn.textContent = "Selling…";
      try {
        const res = await fetch("/api/positions/close", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token_mint: tokenMint }),
        });
        const body = await res.json().catch(() => ({}));
        if (!res.ok || !body.ok) {
          const msg = body.error || ("HTTP " + res.status);
          btn.textContent = "Failed";
          btn.title = msg;
          setTimeout(() => { btn.textContent = label; btn.disabled = false; btn.title = ""; }, 2500);
          return;
        }
        btn.textContent = "Queued";
        btn.classList.remove("border-danger/40", "bg-danger/10", "text-danger");
        btn.classList.add("border-accent/40", "bg-accent/10", "text-accent");
        // Row will disappear on next SSE tick once the close settles.
      } catch (err) {
        btn.textContent = "Failed";
        btn.title = String(err || "");
        setTimeout(() => { btn.textContent = label; btn.disabled = false; btn.title = ""; }, 2500);
      }
    }

    function attachSellNowHandlers(body) {
      body.querySelectorAll("button[data-sell-now]").forEach((btn) => {
        btn.addEventListener("click", (ev) => {
          ev.stopPropagation();  // don't trigger row expand
          const mint = btn.getAttribute("data-sell-now");
          if (!window.confirm(`Sell now: ${mint}?\n\nThis force-closes the position immediately.`)) return;
          sellNowRequest(mint, btn);
        });
      });
    }

    // ---------- feed health ----------
    function renderFeed(health) {
      const h = health || {};
      const entries = [
        ["Bot", h.bot_status || "—"],
        ["Mode", h.monitoring_mode || "—"],
        ["Pending cand.", fmtNum(h.pending_candidate_count)],
        ["Entries", h.entries_paused ? "paused" : "active"],
        ["WS ready", h.websocket_ready === true ? "yes" : (h.websocket_ready === false ? "no" : "—")],
        ["WS subs", `${h.websocket_subscribed_count ?? "—"} / ${h.websocket_subscription_total ?? "—"}`],
        ["WS notifs", fmtNum(h.websocket_notification_count)],
        ["WS dropped", fmtNum(h.websocket_dropped_notification_count)],
        ["Parse queue", fmtNum(h.websocket_pending_parse_queue)],
        ["Last cycle", fmtNum(h.last_cycle_processed)],
      ];
      document.getElementById("feed-grid").innerHTML = entries.map(([k, v]) =>
        `<div><dt class="text-[10px] uppercase tracking-wider text-mute">${k}</dt><dd class="mt-0.5 text-fg num">${v}</dd></div>`
      ).join("");
    }

    // ---------- scoreboard ----------
    function renderScoreboard(sb) {
      if (!sb) return;
      const winRate = Number(sb.win_rate || 0) * 100;
      const elWR = document.getElementById("sb-winrate");
      elWR.textContent = (Number(sb.decisive || 0) > 0) ? winRate.toFixed(1) + "%" : "—";
      elWR.className = "mt-2 text-3xl font-semibold num " + (winRate >= 50 ? "text-accent" : winRate > 0 ? "text-warn" : "text-mute");
      document.getElementById("sb-wlsplit").textContent = `${fmtNum(sb.wins)} W / ${fmtNum(sb.losses)} L`;
      document.getElementById("sb-closed").textContent = fmtNum(sb.closed_positions);
      document.getElementById("sb-breakeven").textContent = "be " + fmtNum(sb.breakevens);
      const avg = Number(sb.avg_pnl_sol || 0);
      const elAvg = document.getElementById("sb-avg");
      elAvg.textContent = fmtSol(avg);
      elAvg.className = "mt-2 text-3xl font-semibold num " + pnlClass(avg);
      document.getElementById("sb-realized").textContent = "realized " + fmtSol(sb.realized_pnl_sol || 0, 3);
      const best = sb.best_trade, worst = sb.worst_trade;
      document.getElementById("sb-best").textContent = best ? fmtSol(best.pnl_sol) : "—";
      document.getElementById("sb-best-mint").textContent = best ? shortMint(best.token_mint) : "—";
      document.getElementById("sb-worst").textContent = worst ? fmtSol(worst.pnl_sol) : "—";
      document.getElementById("sb-worst-mint").textContent = worst ? shortMint(worst.token_mint) : "—";
      document.getElementById("sb-hits").textContent = `${fmtNum(sb.hit_2x)} / ${fmtNum(sb.hit_5x)}`;
      document.getElementById("sb-stops").textContent = "stops " + fmtNum(sb.stop_outs);
    }

    // ---------- per-strategy breakdown ----------
    function renderStrategies(sb) {
      const body = document.getElementById("strategy-body");
      const rows = Array.isArray(sb?.per_strategy) ? sb.per_strategy : [];
      if (!rows.length) {
        body.innerHTML = `<div class="col-span-full rounded-lg border border-border/60 bg-panel/40 p-4 text-center text-xs text-mute">no closed trades yet this session</div>`;
        return;
      }
      body.innerHTML = rows.map((r) => {
        const pnl = Number(r.realized_pnl_sol || 0);
        const avg = Number(r.avg_pnl_sol || 0);
        const wr = Number(r.win_rate || 0) * 100;
        const wl = Number(r.wins) + Number(r.losses);
        return `
          <div class="rounded-lg border border-border bg-panel p-4">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-2">
                <span class="rounded-md bg-panel2 px-2 py-0.5 text-xs font-semibold">${r.strategy_id}</span>
                <span class="text-xs text-mute">${fmtNum(r.trades)} trades</span>
              </div>
              <span class="text-[11px] ${wr >= 50 ? "text-accent" : "text-mute2"}">${wl ? wr.toFixed(1) + "%" : "—"}</span>
            </div>
            <div class="mt-3 flex items-baseline gap-3">
              <div class="text-xl font-semibold num ${pnlClass(pnl)}">${fmtSol(pnl)}</div>
              <div class="text-[11px] text-mute">avg ${fmtSol(avg)}</div>
            </div>
            <div class="mt-2 flex items-center gap-2 text-[11px] text-mute2">
              <span class="text-accent">${fmtNum(r.wins)} W</span>
              <span class="text-danger">${fmtNum(r.losses)} L</span>
              <span class="text-mute">${fmtNum(r.breakevens)} BE</span>
            </div>
          </div>
        `;
      }).join("");
    }

    // ---------- PnL chart ----------
    let pnlChart = null;
    async function refreshPnlChart(summary) {
      try {
        const series = await fetch("/api/pnl_series?limit=500").then((r) => r.json());
        if (!Array.isArray(series)) return;
        const realized = series.map((p) => ({
          x: new Date(p.time).getTime(),
          y: Number(p.cumulative_realized_pnl_sol ?? 0),
        }));
        const lastRealized = realized.length ? realized[realized.length - 1].y : 0;
        const unreal = Number(summary?.unrealized_pnl_sol || 0);
        const combined = realized.length ? realized.map((p) => ({ x: p.x, y: p.y })) : [];
        if (combined.length) {
          combined[combined.length - 1] = { x: combined[combined.length - 1].x, y: lastRealized + unreal };
        } else if (unreal) {
          combined.push({ x: Date.now(), y: unreal });
        }
        if (!pnlChart) {
          const ctx = document.getElementById("chart-pnl").getContext("2d");
          pnlChart = new Chart(ctx, {
            type: "line",
            data: {
              datasets: [
                { label: "realized", data: realized, borderColor: "#22c55e", backgroundColor: "rgba(34,197,94,0.10)", tension: 0.3, pointRadius: 0, borderWidth: 1.8, fill: true, stepped: "before" },
                { label: "combined", data: combined, borderColor: "#3b82f6", backgroundColor: "rgba(59,130,246,0.06)", tension: 0.3, pointRadius: 0, borderWidth: 1.3, borderDash: [4, 3], fill: false },
              ],
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              interaction: { mode: "nearest", intersect: false },
              plugins: { legend: { display: false }, tooltip: { backgroundColor: "#0f0f12", borderColor: "#27272a", borderWidth: 1, titleColor: "#e4e4e7", bodyColor: "#a1a1aa", callbacks: { label: (ctx) => ctx.dataset.label + ": " + ctx.parsed.y.toFixed(4) + " SOL" } } },
              scales: {
                x: { type: "time", time: { unit: "minute" }, ticks: { color: "#71717a", maxTicksLimit: 6 }, grid: { color: "#1a1a1e" } },
                y: { ticks: { color: "#71717a", callback: (v) => Number(v).toFixed(3) }, grid: { color: "#1a1a1e" } },
              },
            },
          });
        } else {
          pnlChart.data.datasets[0].data = realized;
          pnlChart.data.datasets[1].data = combined;
          pnlChart.update("none");
        }
      } catch (e) { /* ignore */ }
    }

    // ---------- candidate funnel ----------
    function renderFunnel(sb) {
      const f = sb?.funnel || {};
      const stages = [
        { key: "notifications", label: "Notifications", color: "text-mute2", bg: "bg-panel2" },
        { key: "candidates", label: "Candidates", color: "text-blue", bg: "bg-blue/10" },
        { key: "entries", label: "Entries", color: "text-purple", bg: "bg-purple/10" },
        { key: "open", label: "Open", color: "text-warn", bg: "bg-warn/10" },
        { key: "closed", label: "Closed", color: "text-mute2", bg: "bg-panel2" },
        { key: "wins", label: "Wins", color: "text-accent", bg: "bg-accent/10" },
      ];
      const max = Math.max(1, ...stages.map((s) => Number(f[s.key] || 0)));
      document.getElementById("funnel-stages").innerHTML = stages.map((s) => {
        const v = Number(f[s.key] || 0);
        const pct = Math.round((v / max) * 100);
        return `
          <div class="rounded-lg border border-border ${s.bg} p-3">
            <div class="text-[10px] uppercase tracking-wider text-mute">${s.label}</div>
            <div class="mt-1 flex items-baseline gap-2">
              <div class="text-2xl font-semibold num ${s.color}">${fmtNum(v)}</div>
            </div>
            <div class="mt-2 h-1 w-full overflow-hidden rounded-full bg-border/40">
              <div class="h-1 rounded-full bg-accent/70" style="width: ${pct}%"></div>
            </div>
          </div>
        `;
      }).join("");
      const reasons = Array.isArray(f.top_reject_reasons) ? f.top_reject_reasons : [];
      const reasonTotal = reasons.reduce((acc, r) => acc + Number(r.count || 0), 0) || 1;
      document.getElementById("funnel-reasons").innerHTML = reasons.length
        ? reasons.map((r) => {
            const pct = Math.round((Number(r.count || 0) / reasonTotal) * 100);
            return `
              <li class="flex items-center gap-2">
                <span class="w-16 shrink-0 text-right text-mute">${fmtNum(r.count)}</span>
                <span class="relative h-1.5 flex-1 overflow-hidden rounded-full bg-border/40">
                  <span class="absolute inset-y-0 left-0 bg-danger/70" style="width: ${pct}%"></span>
                </span>
                <span class="w-[55%] truncate text-mute2">${r.reason}</span>
              </li>
            `;
          }).join("")
        : `<li class="text-mute">no rejections</li>`;
      const drops = [
        ["Notif dropped", f.dropped],
        ["Rejected", f.rejected],
        ["Losses", f.losses],
      ];
      document.getElementById("funnel-drops").innerHTML = drops.map(([k, v]) =>
        `<div><dt class="text-[10px] uppercase tracking-wider text-mute">${k}</dt><dd class="mt-0.5 text-fg num">${fmtNum(v)}</dd></div>`
      ).join("");
    }

    // ---------- closed positions table ----------
    function renderClosedPositions(rows) {
      const body = document.getElementById("closed-body");
      const countEl = document.getElementById("closed-count");
      const list = Array.isArray(rows) ? rows.filter((r) => String(r.status || "").toUpperCase() !== "OPEN").slice(0, 30) : [];
      countEl.textContent = list.length + " closed";
      if (!list.length) {
        body.innerHTML = `<tr><td colspan="8" class="px-5 py-8 text-center text-mute">no closed positions yet</td></tr>`;
        return;
      }
      body.innerHTML = list.map((p, i) => {
        const pnl = Number(p.realized_pnl_sol ?? 0);
        const meta = p.metadata_json || {};
        // size_sol is the REMAINING size and resets to 0 on full close, so fall
        // back to the initial cost basis captured in metadata for closed rows.
        const initialSize = Number(meta.initial_size_sol ?? 0);
        const currentSize = Number(p.size_sol ?? 0);
        const size = initialSize > 0 ? initialSize : currentSize;
        const pnlPct = size > 0 ? (pnl / size) * 100 : null;
        const exitAt = meta.last_exit_at || p.exit_time || null;
        const entryTs = p.entry_time ? new Date(p.entry_time).getTime() : null;
        const exitTs = exitAt ? new Date(exitAt).getTime() : null;
        const holdSec = (entryTs && exitTs && exitTs > entryTs) ? (exitTs - entryTs) / 1000 : null;
        const holdLabel = holdSec === null ? "—" : (holdSec < 60 ? Math.round(holdSec) + "s" : holdSec < 3600 ? Math.round(holdSec / 60) + "m" : (holdSec / 3600).toFixed(1) + "h");
        const cls = pnlClass(pnl);
        const key = "cp-" + safeKey((p.token_mint || "") + "_" + (p.entry_time || i));
        return `
          <tr data-row="${key}" class="cursor-pointer hover:bg-panel2">
            ${mintCell(p.token_mint)}
            <td class="px-5 py-3"><span class="rounded-md bg-panel2 px-1.5 py-0.5 text-[11px]">${p.strategy_id || "main"}</span></td>
            <td class="px-5 py-3 font-mono text-xs">${p.selected_rule_id || "—"}</td>
            <td class="px-5 py-3 text-right num">${size.toFixed(3)}</td>
            <td class="px-5 py-3 text-right num ${cls}">${fmtSol(pnl)}</td>
            <td class="px-5 py-3 text-right num ${cls}">${pnlPct === null ? "—" : (pnlPct >= 0 ? "+" : "") + pnlPct.toFixed(1) + "%"}</td>
            <td class="px-5 py-3 text-right text-mute">${holdLabel}</td>
            <td class="px-5 py-3 text-[11px] text-mute2">${relTime(exitAt)}</td>
          </tr>
          <tr data-detail="${key}" class="hidden bg-panel/50">
            <td colspan="8" class="px-5 py-3 text-[11px]">${positionDetailHtml(p)}</td>
          </tr>
        `;
      }).join("");
      attachRowExpand(body, "closed");
    }

    // ---------- activity feed with chips ----------
    const EVENT_COLOR = {
      live_entry: "text-accent",
      live_exit: "text-blue",
      live_entry_failed: "text-danger",
      live_exit_failed: "text-danger",
      live_entry_failed_fee_burn: "text-danger",
      live_exit_failed_fee_burn: "text-danger",
      sniper_entry_rejected: "text-warn",
      entry_rejected: "text-warn",
      sniper_candidate_selected: "text-purple",
      candidate_selected: "text-purple",
      candidate_dropped: "text-mute2",
    };
    const ACTIVITY_CHIPS = [
      { id: "all", label: "All", events: null },
      { id: "entries", label: "Entries", events: ["live_entry", "paper_entry", "sniper_entry"] },
      { id: "exits", label: "Exits", events: ["live_exit", "paper_exit", "sniper_exit"] },
      { id: "rejections", label: "Rejections", events: ["entry_rejected", "sniper_entry_rejected"] },
      { id: "errors", label: "Errors", events: ["live_entry_failed", "live_exit_failed", "live_entry_failed_fee_burn", "live_exit_failed_fee_burn"] },
      { id: "candidates", label: "Candidates", events: ["candidate_selected", "sniper_candidate_selected", "candidate_dropped"] },
    ];
    let activeChipId = "all";
    function renderActivityChips() {
      const host = document.getElementById("events-chips");
      host.innerHTML = ACTIVITY_CHIPS.map((chip) => {
        const active = chip.id === activeChipId;
        return `<button data-chip="${chip.id}" class="rounded-full border px-2.5 py-1 ${active ? "border-accent bg-accent/10 text-accent" : "border-border bg-panel text-mute2 hover:bg-panel2"}">${chip.label}</button>`;
      }).join("");
      host.querySelectorAll("button[data-chip]").forEach((btn) => {
        btn.addEventListener("click", () => {
          activeChipId = btn.dataset.chip;
          renderActivityChips();
          refreshActivity();
        });
      });
    }
    async function refreshActivity() {
      const chip = ACTIVITY_CHIPS.find((c) => c.id === activeChipId) || ACTIVITY_CHIPS[0];
      try {
        const qs = new URLSearchParams({ limit: "60" });
        const rows = await fetch("/api/events?" + qs.toString()).then((r) => r.json());
        const body = document.getElementById("events-body");
        if (!Array.isArray(rows) || !rows.length) {
          body.innerHTML = `<div class="text-mute">no events</div>`;
          return;
        }
        const filtered = chip.events
          ? rows.filter((r) => chip.events.some((pref) => String(r.event_type || "").startsWith(pref) || String(r.event_type || "") === pref))
          : rows;
        const slice = filtered.slice(0, 30);
        if (!slice.length) {
          body.innerHTML = `<div class="text-mute">no events match ${chip.label.toLowerCase()}</div>`;
          return;
        }
        body.innerHTML = slice.map((r, i) => {
          const et = r.event_type || "event";
          const color = EVENT_COLOR[et] || "text-mute2";
          // /api/events returns raw JSONL rows with fields at the TOP level
          // (reason, guard_reason, rule_id, …). Older code paths wrap payload
          // under `r.payload` — support both shapes for robustness.
          const payload = (r.payload && typeof r.payload === "object") ? r.payload : r;
          const META_KEYS = new Set(["event_type", "logged_at", "timestamp", "created_at", "ts", "mode", "session_id", "payload"]);
          const mint = payload.token_mint || r.token_mint || "";
          const reason = payload.reason || r.reason || "";
          const guardReason = payload.guard_reason || r.guard_reason || "";
          const ruleId = payload.rule_id || r.rule_id || "";
          const strategyId = payload.strategy_id || r.strategy_id || "";
          const errorStr = payload.error || r.error || "";
          const detailBadges = [
            reason ? `<span class="rounded bg-panel2 px-1.5 py-0.5 text-[10px] text-warn">${reason}</span>` : "",
            guardReason ? `<span class="rounded bg-panel2 px-1.5 py-0.5 text-[10px] text-mute2">${guardReason}</span>` : "",
            ruleId ? `<span class="rounded bg-panel2 px-1.5 py-0.5 font-mono text-[10px] text-blue">${ruleId}</span>` : "",
            strategyId ? `<span class="rounded bg-panel2 px-1.5 py-0.5 text-[10px]">${strategyId}</span>` : "",
            errorStr ? `<span class="rounded bg-panel2 px-1.5 py-0.5 text-[10px] text-danger">err</span>` : "",
          ].filter(Boolean).join(" ");
          const ts = r.logged_at || r.timestamp || r.created_at || r.ts;
          const key = "ev-" + safeKey((ts || "") + "_" + et + "_" + (mint || i));
          const payloadEntries = Object.entries(payload).filter(([k]) => k !== "token_mint" && !META_KEYS.has(k)).slice(0, 40);
          const payloadRows = payloadEntries
            .map(([k, v]) => {
              const valStr = typeof v === "object" ? JSON.stringify(v) : String(v);
              const clipped = valStr.length > 160 ? valStr.slice(0, 160) + "…" : valStr;
              return `<div><dt class="text-[10px] uppercase tracking-wider text-mute">${k}</dt><dd class="mt-0.5 break-all text-mute2" title="${valStr.replace(/"/g, "&quot;")}">${clipped}</dd></div>`;
            })
            .join("");
          const detailPanel = payloadRows
            ? `<div data-detail="${key}" class="hidden mt-2 rounded-md border border-border/40 bg-panel2/40 p-3"><div class="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">${payloadRows}</div></div>`
            : "";
          return `
            <div data-row="${key}" class="cursor-pointer rounded-md border border-border/60 bg-panel/40 px-3 py-2 hover:bg-panel2">
              <div class="flex flex-wrap items-start gap-2">
                <span class="shrink-0 text-[10px] text-mute" title="${ts || ""}">${relTime(ts)}</span>
                <span class="${color} shrink-0 font-medium">${et}</span>
                ${mint ? `<span class="shrink-0 font-mono text-[11px] text-mute2">${shortMint(mint)}</span>` : ""}
                <span class="flex flex-wrap gap-1 text-[11px]">${detailBadges}</span>
              </div>
              ${detailPanel}
            </div>
          `;
        }).join("");
        body.querySelectorAll("div[data-row]").forEach((row) => {
          const rowKey = row.dataset.row;
          const detail = body.querySelector(`div[data-detail="${rowKey}"]`);
          if (!detail) return;
          if (expandedKeys.ev.has(rowKey)) detail.classList.remove("hidden");
          row.addEventListener("click", () => {
            detail.classList.toggle("hidden");
            if (detail.classList.contains("hidden")) expandedKeys.ev.delete(rowKey);
            else expandedKeys.ev.add(rowKey);
          });
        });
      } catch (e) { /* ignore */ }
    }
    async function refreshScoreboard() {
      try {
        const sb = await fetch("/api/scoreboard").then((r) => r.json());
        renderScoreboard(sb);
        renderStrategies(sb);
        renderFunnel(sb);
      } catch (e) { /* ignore */ }
    }
    async function refreshClosedPositions() {
      try {
        const rows = await fetch("/api/recent_positions?limit=60").then((r) => r.json());
        renderClosedPositions(rows);
      } catch (e) { /* ignore */ }
    }

    // ---------- wallet lane panel ----------
    function fmtAge(sec) {
      if (sec == null) return "—";
      const s = Math.max(0, Number(sec));
      if (s < 60) return s.toFixed(0) + "s";
      if (s < 3600) return (s / 60).toFixed(1) + "m";
      if (s < 86400) return (s / 3600).toFixed(1) + "h";
      return (s / 86400).toFixed(1) + "d";
    }
    function ageFromIso(iso) {
      if (!iso) return null;
      const t = Date.parse(iso);
      if (Number.isNaN(t)) return null;
      return Math.max(0, (Date.now() - t) / 1000);
    }
    function renderWalletPanel(w) {
      if (!w) return;
      const head = document.getElementById("wallet-panel-status");
      const enabled = !!(w.pool && w.pool.wallet_enabled);
      head.textContent = enabled ? "ACTIVE" : "disabled";
      head.className = "text-xs " + (enabled ? "text-accent" : "text-mute");

      const pool = w.pool || {};
      const pg = document.getElementById("wallet-pool-grid");
      const lastBuyAge = w.activity ? ageFromIso((w.activity.last_tracked_buy || {}).ts) : null;
      const lastSeenAge = ageFromIso(pool.last_seen_event_at);
      pg.innerHTML = `
        <dt class="text-mute">Tracked wallets</dt><dd class="text-right font-semibold num">${fmtNum(pool.tracked_wallets || 0)}</dd>
        <dt class="text-mute">Features on</dt><dd class="text-right">${pool.features_enabled ? "yes" : "no"}</dd>
        <dt class="text-mute">Open / Exposure</dt><dd class="text-right num">${fmtNum(pool.wallet_open_positions || 0)} · ${fmtSol(pool.wallet_exposure_sol || 0)}</dd>
        <dt class="text-mute">Pool refreshed</dt><dd class="text-right">${fmtAge(pool.pool_refresh_age_sec)} ago</dd>
        <dt class="text-mute">Last event seen</dt><dd class="text-right">${fmtAge(lastSeenAge)} ago</dd>
        <dt class="text-mute">Last tracked buy</dt><dd class="text-right">${lastBuyAge == null ? "—" : fmtAge(lastBuyAge) + " ago"}</dd>
      `;

      const act = w.activity || {};
      const ag = document.getElementById("wallet-activity-grid");
      const share = (act.tracked_wallet_share || 0) * 100;
      ag.innerHTML = `
        <dt class="text-mute">Events (15m)</dt><dd class="text-right num">${fmtNum(act.events_seen || 0)}</dd>
        <dt class="text-mute">With tracked wallet</dt><dd class="text-right num">${fmtNum(act.events_with_tracked_wallet || 0)} <span class="text-mute2">(${share.toFixed(1)}%)</span></dd>
        <dt class="text-mute">Biggest cluster/5m</dt><dd class="text-right num ${act.biggest_cluster >= 2 ? "text-accent" : ""}">${fmtNum(act.biggest_cluster || 0)}</dd>
        <dt class="text-mute">On mint</dt><dd class="text-right font-mono text-[11px]">${act.biggest_cluster_token ? shortMint(act.biggest_cluster_token) : "—"}</dd>
      `;
      const topList = document.getElementById("wallet-top-list");
      const top = Array.isArray(act.top_wallets) ? act.top_wallets : [];
      topList.innerHTML = top.length
        ? top.map((r) => `<li class="flex items-center justify-between"><span class="font-mono text-[11px]">${shortMint(r.wallet)}</span><span class="num text-mute">${fmtNum(r.appearances)}</span></li>`).join("")
        : `<li class="text-mute">no tracked wallet activity in window</li>`;

      const f = w.funnel || {};
      const fg = document.getElementById("wallet-funnel-grid");
      const wr = f.win_rate == null ? "—" : (Number(f.win_rate) * 100).toFixed(1) + "%";
      const pnl = Number(f.realized_pnl_sol || 0);
      fg.innerHTML = `
        <dt class="text-mute">Candidates</dt><dd class="text-right num">${fmtNum(f.candidates_selected || 0)}</dd>
        <dt class="text-mute">Rejections</dt><dd class="text-right num">${fmtNum(f.rejections_total || 0)}</dd>
        <dt class="text-mute">Entries fired</dt><dd class="text-right num">${fmtNum(f.entries_executed || 0)}</dd>
        <dt class="text-mute">Exec failures</dt><dd class="text-right num ${f.failures ? "text-danger" : ""}">${fmtNum(f.failures || 0)}</dd>
        <dt class="text-mute">Closed · W/L</dt><dd class="text-right num">${fmtNum(f.closed_trades || 0)} · ${fmtNum(f.wins || 0)}/${fmtNum(f.losses || 0)}</dd>
        <dt class="text-mute">Win rate · PnL</dt><dd class="text-right num">${wr} · <span class="${pnlClass(pnl)}">${fmtSol(pnl)}</span></dd>
      `;
      const rejList = document.getElementById("wallet-rejections-list");
      const rejs = Array.isArray(f.top_rejections) ? f.top_rejections : [];
      rejList.innerHTML = rejs.length
        ? rejs.map((r) => `<li class="flex items-center justify-between"><span class="truncate pr-2">${r.reason}</span><span class="num text-mute">${fmtNum(r.count)}</span></li>`).join("")
        : `<li class="text-mute">no rejections in window</li>`;
    }
    async function refreshWalletPanel() {
      try {
        const w = await fetch("/api/wallet_panel").then((r) => r.json());
        renderWalletPanel(w);
      } catch (e) { /* ignore */ }
    }

    // ---------- SSE client ----------
    function connectStream() {
      setStreamState("reconnecting");
      const es = new EventSource("/api/stream");
      es.onopen = () => setStreamState("live");
      es.onerror = () => { setStreamState("error"); es.close(); setTimeout(connectStream, 2000); };
      es.onmessage = (evt) => {
        if (!evt.data) return;
        try {
          const tick = JSON.parse(evt.data);
          applyTick(tick);
        } catch (e) { /* ignore */ }
      };
    }
    let lastTickSummary = null;
    function applyTick(tick) {
      lastTickSummary = tick.summary || null;
      renderKPIs(tick.summary);
      renderPositions(tick.positions || []);
      renderMetrics(tick.metrics);
      renderExecChart(tick.metrics);
      renderFeed(tick.health);
    }

    // ---------- session control buttons ----------
    document.getElementById("btn-session-new").addEventListener("click", async () => {
      try {
        const r = await fetch("/api/session/new", { method: "POST" });
        const payload = await r.json();
        if (!payload.ok && payload.error) alert(payload.error);
      } catch (e) { alert(String(e)); }
    });
    document.getElementById("btn-session-end").addEventListener("click", async () => {
      if (!confirm("End current session?")) return;
      try {
        const r = await fetch("/api/session/end", { method: "POST" });
        const payload = await r.json();
        if (!payload.ok && payload.error) alert(payload.error);
      } catch (e) { alert(String(e)); }
    });
    // ---------- boot ----------
    (async () => {
      try {
        const initial = await fetch("/api/live").then((r) => r.json());
        applyTick(initial);
      } catch (e) { /* ignore */ }
      renderActivityChips();
      refreshActivity();
      refreshScoreboard();
      refreshClosedPositions();
      refreshWalletPanel();
      refreshPnlChart(lastTickSummary);
      setInterval(refreshActivity, 4000);
      setInterval(refreshScoreboard, 6000);
      setInterval(refreshClosedPositions, 8000);
      setInterval(refreshWalletPanel, 5000);
      setInterval(() => refreshPnlChart(lastTickSummary), 6000);
      connectStream();
    })();
  </script>
</body>
</html>
"""


def render_dashboard_html(refresh_sec: int) -> str:
    """Return the dashboard HTML. ``refresh_sec`` kept for API compatibility."""
    _ = refresh_sec  # unused — stream drives updates
    return _DASHBOARD_HTML


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve dashboard HTML, SSE stream, and JSON APIs."""

    server: "DashboardHttpServer"

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            self.close_connection = True

    def handle(self) -> None:
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        limit = self._parse_limit(query.get("limit", ["50"])[0])

        if parsed.path == "/":
            self._send_html(render_dashboard_html(self.server.refresh_sec))
            return
        if parsed.path == "/api/stream":
            self._stream_events()
            return
        if parsed.path == "/api/live":
            tick = self.server.store.live_tick()
            tick["positions"] = self._enrich_positions_live(tick.get("positions") or [])
            self._send_json(tick)
            return
        if parsed.path == "/api/metrics":
            window = 300.0
            try:
                window = float(query.get("window", ["300"])[0])
            except (TypeError, ValueError):
                window = 300.0
            self._send_json(self.server.store.hot_path_metrics(window_sec=window))
            return

        routes = {
            "/api/summary": lambda: self.server.store.summary(),
            "/api/open_positions": lambda: self.server.store.open_positions(
                token=self._first(query, "token"),
                rule_id=self._first(query, "rule_id"),
                regime=self._first(query, "regime"),
            ),
            "/api/recent_positions": lambda: self.server.store.recent_positions(
                limit=limit,
                token=self._first(query, "token"),
                rule_id=self._first(query, "rule_id"),
                regime=self._first(query, "regime"),
                status=self._first(query, "status"),
            ),
            "/api/executions": lambda: self.server.store.recent_executions(
                limit=limit,
                token=self._first(query, "token"),
                action=self._first(query, "action"),
                mode=self._first(query, "mode"),
                status=self._first(query, "status"),
            ),
            "/api/rule_performance": lambda: self.server.store.rule_performance(
                limit=limit,
                rule_id=self._first(query, "rule_id"),
                regime=self._first(query, "regime"),
            ),
            "/api/events": lambda: self.server.store.recent_events(
                limit=limit,
                token=self._first(query, "token"),
                rule_id=self._first(query, "rule_id"),
                regime=self._first(query, "regime"),
                event_type=self._first(query, "event_type"),
            ),
            "/api/rejections": lambda: self.server.store.rejected_trades(
                limit=limit,
                token=self._first(query, "token"),
                rule_id=self._first(query, "rule_id"),
                regime=self._first(query, "regime"),
            ),
            "/api/rejection_summary": lambda: self.server.store.rejection_summary(limit=limit),
            "/api/scoreboard": lambda: self.server.store.session_scoreboard(),
            "/api/pnl_series": lambda: self.server.store.pnl_series(
                limit=limit,
                rule_id=self._first(query, "rule_id"),
                regime=self._first(query, "regime"),
            ),
            "/api/rule_pnl_series": lambda: self.server.store.rule_pnl_series(limit=limit),
            "/api/activity_series": lambda: self.server.store.activity_series(limit=limit),
            "/api/token_detail": lambda: self.server.store.token_detail(
                self._first(query, "token") or ""
            ),
            "/api/rule_detail": lambda: self.server.store.rule_detail(
                self._first(query, "rule_id") or ""
            ),
            "/api/subscribed_wallets": lambda: self.server.store.subscribed_wallets(),
            "/api/wallet_panel": lambda: self.server.store.wallet_panel(),
            "/api/health": lambda: self.server.store.health(),
        }
        if parsed.path in routes:
            try:
                self._send_json(routes[parsed.path]())
            except Exception as exc:  # noqa: BLE001
                logger.exception("dashboard_http | route error on %s", parsed.path)
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/session/new":
            controller = getattr(self.server, "controller", None)
            if controller is None or not hasattr(controller, "request_new_session"):
                self._send_json(
                    {"ok": False, "error": "dashboard control unavailable"},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            try:
                payload = controller.request_new_session(source="dashboard")
                self._send_json(payload, status=HTTPStatus.ACCEPTED)
            except Exception as exc:  # noqa: BLE001
                logger.exception("dashboard_http | session new request failed")
                self._send_json(
                    {"ok": False, "error": str(exc)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return
        if parsed.path == "/api/session/end":
            controller = getattr(self.server, "controller", None)
            if controller is None or not hasattr(controller, "request_end_session"):
                self._send_json(
                    {"ok": False, "error": "dashboard control unavailable"},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            try:
                payload = controller.request_end_session(source="dashboard")
                self._send_json(payload, status=HTTPStatus.ACCEPTED)
            except Exception as exc:  # noqa: BLE001
                logger.exception("dashboard_http | session end request failed")
                self._send_json(
                    {"ok": False, "error": str(exc)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return
        if parsed.path == "/api/positions/close":
            controller = getattr(self.server, "controller", None)
            if controller is None or not hasattr(controller, "request_manual_close"):
                self._send_json(
                    {"ok": False, "error": "dashboard control unavailable"},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            token_mint = self._read_post_field("token_mint")
            if not token_mint:
                self._send_json(
                    {"ok": False, "error": "token_mint_required"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            try:
                payload = controller.request_manual_close(token_mint=token_mint, source="dashboard")
                status = HTTPStatus.ACCEPTED if payload.get("ok") else HTTPStatus.BAD_REQUEST
                self._send_json(payload, status=status)
            except Exception as exc:  # noqa: BLE001
                logger.exception("dashboard_http | manual close request failed")
                self._send_json(
                    {"ok": False, "error": str(exc)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    # ---------------------------------------------------------------- SSE
    def _stream_events(self) -> None:
        """Push `live_tick()` as a Server-Sent Event stream."""
        interval_sec = max(0.25, float(getattr(self.server, "stream_interval_sec", 0.5)))
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
            TimeoutError,
        ):
            return
        except OSError as exc:
            if exc.errno in {errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED}:
                return
            raise

        last_heartbeat = 0.0
        while True:
            try:
                tick = self.server.store.live_tick()
                tick["positions"] = self._enrich_positions_live(tick.get("positions") or [])
                payload = json.dumps(
                    self._sanitize_json_payload(tick),
                    default=str,
                    allow_nan=False,
                )
                self.wfile.write(b"data: " + payload.encode("utf-8") + b"\n\n")
                now = time.time()
                if now - last_heartbeat > 15:
                    self.wfile.write(b": ping\n\n")
                    last_heartbeat = now
                self.wfile.flush()
            except (
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
                TimeoutError,
            ):
                return
            except OSError as exc:
                if exc.errno in {errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED}:
                    return
                logger.exception("dashboard_http | stream OS error")
                return
            except Exception:  # noqa: BLE001
                logger.exception("dashboard_http | stream tick failed")
                try:
                    self.wfile.write(b"event: error\ndata: {}\n\n")
                    self.wfile.flush()
                except Exception:  # noqa: BLE001
                    return
            time.sleep(interval_sec)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        logger.info("dashboard_http | " + format, *args)

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        clean_payload = self._sanitize_json_payload(payload)
        body = json.dumps(clean_payload, default=str, allow_nan=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status=status)

    def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self._send_bytes(body, "text/html; charset=utf-8", status=status)

    def _send_bytes(
        self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        """Write a response while tolerating client disconnects."""
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
            TimeoutError,
        ):
            logger.debug("dashboard_http | client disconnected before response write")
        except OSError as exc:
            if exc.errno in {errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED}:
                logger.debug(
                    "dashboard_http | client disconnected before response write (%s)",
                    exc.errno,
                )
                return
            raise

    def _enrich_positions_live(self, positions: list[dict]) -> list[dict]:
        """Overlay mark-to-market fields on open positions.

        Primary source: ``local_quote_engine.quote_sell(mint, amount_raw)`` — this is
        the same constant-product quote Jupiter's aggregator produces for the pool,
        so the reported PnL matches what you'd actually realize by selling now.

        Fallback: the exit engine's ``last_reliable_pnl_multiple`` (tick-synced but
        derived from last-trade prices, which can be skewed by whale dumps / copycat
        panic sells). Used only if no fresh reserves are cached for the mint.
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        STALE_SEC = 20.0
        controller = getattr(self.server, "controller", None)
        engine = getattr(controller, "local_quote_engine", None) if controller else None
        has_engine = engine is not None and hasattr(engine, "quote_sell")

        for p in positions:
            mint = str(p.get("token_mint") or "")
            size_sol = float(p.get("size_sol") or 0.0)
            amount_raw = int(float(p.get("amount_received") or 0.0))

            # ---- Primary: live AMM quote (Jupiter-equivalent) -----------------
            quoted_ok = False
            if has_engine and mint and size_sol > 0 and amount_raw > 0:
                try:
                    lamports_out = engine.quote_sell(mint, amount_raw)
                except Exception:  # noqa: BLE001
                    lamports_out = None
                if lamports_out is not None and lamports_out > 0:
                    live_value_sol = float(lamports_out) / 1_000_000_000.0
                    live_pnl = live_value_sol - size_sol
                    p["live_pnl_sol"] = live_pnl
                    p["live_pnl_multiple"] = live_pnl / size_sol
                    p["live_price_stale"] = False
                    p["live_source"] = "amm_quote"
                    # Also surface spot mark for the price cell.
                    try:
                        spot = engine.mark_price_sol(mint)
                        if spot is not None and float(spot) > 0:
                            p["mark_price_sol"] = float(spot)
                    except Exception:  # noqa: BLE001
                        pass
                    quoted_ok = True

            if quoted_ok:
                continue

            # ---- Fallback: exit-engine metadata -----------------------------
            meta = p.get("metadata_json")
            if not isinstance(meta, dict):
                p["live_price_stale"] = True
                continue
            mult = meta.get("last_reliable_pnl_multiple")
            if mult is None:
                mult = meta.get("last_pnl_multiple")
            if mult is None:
                p["live_price_stale"] = True
                continue
            try:
                mult_f = float(mult)
            except (TypeError, ValueError):
                p["live_price_stale"] = True
                continue
            stale = True
            ts_iso = meta.get("last_token_update_at")
            if ts_iso:
                try:
                    ts = datetime.fromisoformat(str(ts_iso).replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    stale = (now - ts).total_seconds() > STALE_SEC
                except (TypeError, ValueError):
                    stale = True
            mark = meta.get("last_price_sol_reliable_seen") or meta.get("last_price_sol_seen")
            if mark is not None:
                try:
                    p["mark_price_sol"] = float(mark)
                except (TypeError, ValueError):
                    pass
            p["live_pnl_multiple"] = mult_f
            p["live_pnl_sol"] = size_sol * mult_f
            p["live_price_stale"] = stale
            p["live_source"] = "metadata"
        return positions

    @staticmethod
    def _parse_limit(raw: str) -> int:
        try:
            return max(1, min(int(raw), 500))
        except ValueError:
            return 50

    @staticmethod
    def _first(query: dict[str, list[str]], key: str) -> str | None:
        values = query.get(key) or []
        if not values:
            return None
        value = values[0].strip()
        return value or None

    def _read_post_field(self, key: str) -> str | None:
        """Read a single field from a POST body (JSON or x-www-form-urlencoded)."""
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except (TypeError, ValueError):
            length = 0
        if length <= 0 or length > 65536:
            return None
        raw = self.rfile.read(length)
        content_type = (self.headers.get("Content-Type") or "").lower()
        if "application/json" in content_type:
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except (ValueError, UnicodeDecodeError):
                return None
            if isinstance(data, dict):
                value = data.get(key)
                if isinstance(value, str):
                    return value.strip() or None
            return None
        try:
            parsed = parse_qs(raw.decode("utf-8"))
        except UnicodeDecodeError:
            return None
        values = parsed.get(key) or []
        if not values:
            return None
        value = values[0].strip()
        return value or None

    @classmethod
    def _sanitize_json_payload(cls, payload: object) -> object:
        """Replace NaN/Infinity values so API responses stay browser-parseable."""
        if isinstance(payload, float):
            return payload if math.isfinite(payload) else None
        if isinstance(payload, dict):
            return {key: cls._sanitize_json_payload(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [cls._sanitize_json_payload(item) for item in payload]
        if isinstance(payload, tuple):
            return [cls._sanitize_json_payload(item) for item in payload]
        return payload


class DashboardHttpServer(ThreadingHTTPServer):
    """HTTP server with attached dashboard store."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        store: DashboardDataStore,
        refresh_sec: int,
        controller: object | None = None,
        stream_interval_sec: float = 0.5,
    ) -> None:
        super().__init__(server_address, DashboardHandler)
        self.store = store
        self.refresh_sec = refresh_sec
        self.controller = controller
        self.stream_interval_sec = stream_interval_sec
