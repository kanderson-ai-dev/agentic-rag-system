// Dashboard (Phase 6): renders real usage metadata into (a) metric cards,
// (b) a dependency-free SVG chart of cost/latency over time, and (c) a
// sortable, paginated recent-requests table. All server-derived values are
// written with `textContent` (never interpolated into innerHTML), and empty +
// loading (skeleton) states are explicit.

import { $, $$, debounce } from "/js/util.js";
import { api } from "/js/api.js";
import { showToast } from "./toast.js";

const DASHBOARD_PAGE_SIZE = 8;

// Dashboard state is session-scoped and re-fetched wholesale on each refresh;
// sorting is done client-side over that (small) result set.
const dashboard = {
  recent: [],
  sortKey: "timestamp",
  sortDir: -1, // -1 = descending, +1 = ascending
  page: 0,
};

function formatUsd(value) {
  return `$${Number(value).toFixed(4)}`;
}

function formatLatency(ms) {
  return `${Math.round(Number(ms))}ms`;
}

function formatTokens(n) {
  return Number(n).toLocaleString();
}

function statusOf(row) {
  if (row.blocked) return "blocked";
  if (row.escalated) return "escalated";
  return "ok";
}

function statusLabel(status) {
  return { ok: "ok", blocked: "blocked", escalated: "reviewed" }[status] || status;
}

// --- Loading skeleton ------------------------------------------------------
function renderDashboardSkeleton() {
  const summary = $("#summary");
  summary.replaceChildren();
  for (let i = 0; i < 4; i += 1) {
    summary.appendChild(
      Object.assign(document.createElement("div"), {
        className: "skeleton stat",
      })
    );
  }

  $("#chart-figure").hidden = true;
  $("#chart-empty").hidden = true;
  $("#chart").replaceChildren(
    Object.assign(document.createElement("div"), {
      className: "skeleton chart",
    })
  );

  const tbody = $("#recent-table tbody");
  tbody.replaceChildren();
  for (let i = 0; i < 5; i += 1) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.className = "skeleton row";
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
  $("#recent-empty").hidden = true;
  $("#pagination").replaceChildren();
}

// --- Metric cards ----------------------------------------------------------
function renderSummary(summary) {
  const cards = [
    { value: formatUsd(summary.total_cost_usd), label: "Total cost" },
    { value: formatLatency(summary.avg_latency_ms), label: "Avg latency" },
    { value: String(summary.blocked_requests ?? 0), label: "Blocked" },
    { value: String(summary.escalated_requests ?? 0), label: "Escalated to human" },
  ];

  const el = $("#summary");
  el.replaceChildren();
  for (const card of cards) {
    const stat = document.createElement("div");
    stat.className = "stat";
    const value = document.createElement("div");
    value.className = "value";
    value.textContent = card.value;
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = card.label;
    stat.append(value, label);
    el.appendChild(stat);
  }
}

// --- Quality (EDD) ---------------------------------------------------------
function renderQuality(quality) {
  const el = $("#quality");
  el.replaceChildren();

  if (!quality.available) {
    const p = document.createElement("p");
    p.className = "muted small";
    p.textContent = "No evaluation scorecard yet — run ";
    const code = document.createElement("code");
    code.textContent = "python evaluation/run_ragas.py";
    p.appendChild(code);
    p.appendChild(document.createTextNode("."));
    el.appendChild(p);
    return;
  }

  for (const [name, metric] of Object.entries(quality.metrics || {})) {
    const stat = document.createElement("div");
    stat.className = `stat quality-stat ${metric.passing ? "pass" : "fail"}`;
    const value = document.createElement("div");
    value.className = "value";
    value.textContent = Number(metric.score ?? 0).toFixed(2);
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = name.replaceAll("_", " ");
    const threshold = document.createElement("span");
    threshold.className = "threshold";
    threshold.textContent = ` (≥ ${metric.threshold})`;
    label.appendChild(threshold);
    stat.append(value, label);
    el.appendChild(stat);
  }
}

// --- Performance (Phase 8) -------------------------------------------------
// Green/amber/red vs. the project targets, always paired with a text label —
// colour is never the only signal.
function perfBand(value, target, slack) {
  if (value <= target) return { cls: "pass", label: "on target" };
  if (value <= target * slack) return { cls: "warn", label: "above target" };
  return { cls: "fail", label: "over target" };
}

function perfCard(value, label, band) {
  const stat = document.createElement("div");
  stat.className = `stat perf-stat ${band.cls}`;
  const v = document.createElement("div");
  v.className = "value";
  v.textContent = value;
  const l = document.createElement("div");
  l.className = "label";
  l.textContent = label;
  const tag = document.createElement("span");
  tag.className = "threshold";
  tag.textContent = ` ${band.label}`;
  l.appendChild(tag);
  stat.append(v, l);
  return stat;
}

function renderPerformance(perf) {
  const cards = $("#perf-cards");
  const nodes = $("#perf-nodes");
  const hva = $("#human-vs-agent");
  const versionsBody = $("#perf-versions tbody");
  cards.replaceChildren();
  nodes.replaceChildren();
  hva.replaceChildren();
  versionsBody.replaceChildren();

  const empty = !perf || !perf.requests;
  $("#perf-empty").hidden = !empty;
  $("#perf-nodes-empty").hidden = !empty;

  const lat = perf?.latency ?? {};
  const cost = perf?.cost ?? {};
  const tokens = perf?.tokens ?? {};

  cards.append(
    perfCard(
      formatLatency(lat.p50_ms ?? 0),
      "p50 latency (≤ 3s)",
      empty ? { cls: "", label: "no data" } : perfBand(lat.p50_ms, lat.target_p50_ms, 1.5)
    ),
    perfCard(
      formatLatency(lat.p95_ms ?? 0),
      "p95 latency (≤ 6s)",
      empty ? { cls: "", label: "no data" } : perfBand(lat.p95_ms, lat.target_p95_ms, 1.5)
    ),
    perfCard(
      formatUsd(cost.avg_per_query_usd ?? 0),
      "Avg cost / query (≤ $0.01)",
      empty ? { cls: "", label: "no data" } : perfBand(cost.avg_per_query_usd, cost.target_per_query_usd, 2)
    ),
    perfCard(formatUsd(cost.total_usd ?? 0), "Session cost", { cls: "", label: `${perf?.requests ?? 0} requests` }),
    perfCard(
      formatTokens(Math.round(tokens.avg_per_query ?? 0)),
      "Avg tokens / query",
      { cls: "", label: "" }
    )
  );

  if (!empty) {
    // Per-node p95 bars, normalized to the slowest node.
    const perNode = perf.per_node || [];
    const maxP95 = Math.max(...perNode.map((n) => n.p95_ms), 1);
    for (const node of perNode) {
      const row = document.createElement("div");
      row.className = "perf-bar-row";
      row.setAttribute("role", "listitem");
      const name = document.createElement("span");
      name.className = "perf-bar-name";
      name.textContent = node.node;
      const track = document.createElement("div");
      track.className = "perf-bar-track";
      const bar = document.createElement("div");
      bar.className = "perf-bar";
      bar.style.width = `${Math.max(2, (node.p95_ms / maxP95) * 100)}%`;
      track.appendChild(bar);
      const value = document.createElement("span");
      value.className = "perf-bar-value";
      value.textContent = `p50 ${formatLatency(node.p50_ms)} · p95 ${formatLatency(node.p95_ms)} · n=${node.samples}`;
      row.append(name, track, value);
      nodes.appendChild(row);
    }

    // Human vs. Agent — the human side is always labelled an estimate.
    const h = perf.human_vs_agent || {};
    const pq = h.per_query || {};
    const tt = h.totals || {};
    const cols = [
      {
        title: "Human (estimate)",
        lines: [
          `${formatUsd(pq.human_cost_usd ?? 0)} / query`,
          `${Math.round((pq.human_seconds ?? 0) / 60)} min / query`,
          `${formatUsd(tt.human_cost_usd ?? 0)} for ${tt.requests ?? 0} queries`,
        ],
      },
      {
        title: "Agent (measured)",
        lines: [
          `${formatUsd(pq.agent_cost_usd ?? 0)} / query`,
          `${(pq.agent_seconds ?? 0).toFixed(1)}s / query`,
          `${formatUsd(tt.agent_cost_usd ?? 0)} for ${tt.requests ?? 0} queries`,
        ],
      },
    ];
    for (const col of cols) {
      const box = document.createElement("div");
      box.className = "perf-col";
      const title = document.createElement("strong");
      title.textContent = col.title;
      box.appendChild(title);
      for (const line of col.lines) {
        const p = document.createElement("p");
        p.className = "muted small";
        p.textContent = line;
        box.appendChild(p);
      }
      hva.appendChild(box);
    }
    const savings = document.createElement("div");
    savings.className = "perf-col perf-savings";
    const big = document.createElement("strong");
    big.textContent = `${(h.cost_savings_pct ?? 0).toFixed(1)}%`;
    const sub = document.createElement("p");
    sub.className = "muted small";
    sub.textContent = "cheaper than the human baseline";
    savings.append(big, sub);
    hva.appendChild(savings);

    const note = $("#perf-hva-note");
    const a = h.assumptions || {};
    note.textContent =
      `Assumes a human reviewer spends ~${a.human_minutes_per_query ?? "—"} min ` +
      `per query at $${a.human_hourly_usd ?? "—"}/h. Human side is an estimate; ` +
      "agent side is measured telemetry for this session.";
  } else {
    $("#perf-hva-note").textContent = "";
  }

  const versions = perf?.per_prompt_version || [];
  $("#perf-versions-empty").hidden = versions.length > 0;
  for (const v of versions) {
    const tr = document.createElement("tr");
    for (const text of [
      `v${v.generate_answer_version}`,
      String(v.requests),
      formatUsd(v.avg_cost_usd),
      formatLatency(v.avg_latency_ms),
    ]) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    }
    versionsBody.appendChild(tr);
  }
}

// --- SVG chart (no libraries) ---------------------------------------------
// A single small viewBox with two normalized polylines — latency (accent) and
// cost (success) — plotted against request order (oldest → newest). Values are
// normalized to their own maximum so the two different units can share an axis;
// the legend reports the actual latest value + unit for each series.
function renderChart(recent) {
  const chart = $("#chart");
  const caption = $("#chart-caption");
  chart.replaceChildren();
  caption.replaceChildren();

  const ordered = [...recent].reverse(); // oldest → newest for a time axis

  if (ordered.length < 2) {
    $("#chart-figure").hidden = true;
    $("#chart-empty").hidden = !ordered.length;
    return;
  }

  $("#chart-figure").hidden = false;
  $("#chart-empty").hidden = true;

  const W = 100;
  const H = 40;
  const pad = 6;
  const n = ordered.length;

  const latencies = ordered.map((r) => r.latency_ms);
  const costs = ordered.map((r) => r.cost_usd);
  const maxLatency = Math.max(...latencies, 0);
  const maxCost = Math.max(...costs, 0);

  const x = (i) => pad + (i / (n - 1)) * (W - pad * 2);
  const y = (v, max) => H - pad - (max ? (v / max) * (H - pad * 2) : 0);

  const rawLatency = latencies
    .map((v, i) => `${x(i).toFixed(2)},${y(v, maxLatency).toFixed(2)}`)
    .join(" ");
  const rawCost = costs
    .map((v, i) => `${x(i).toFixed(2)},${y(v, maxCost).toFixed(2)}`)
    .join(" ");

  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");

  // Baseline + optional midway grid line.
  const baseline = document.createElementNS(NS, "line");
  baseline.setAttribute("x1", String(pad));
  baseline.setAttribute("x2", String(W - pad));
  baseline.setAttribute("y1", String(H - pad));
  baseline.setAttribute("y2", String(H - pad));
  baseline.setAttribute("class", "grid");
  svg.appendChild(baseline);

  const series = (cls) => {
    const line = document.createElementNS(NS, "polyline");
    line.setAttribute("class", `series ${cls}`);
    return line;
  };
  const latencyLine = series("series-latency");
  latencyLine.setAttribute("points", rawLatency);
  const costLine = series("series-cost");
  costLine.setAttribute("points", rawCost);
  svg.append(latencyLine, costLine);

  // Dots on each series so individual points remain visible at small counts.
  const points = (values, max, cls) => {
    for (let i = 0; i < values.length; i += 1) {
      const dot = document.createElementNS(NS, "circle");
      dot.setAttribute("class", `point ${cls}`);
      dot.setAttribute("cx", x(i).toFixed(2));
      dot.setAttribute("cy", y(values[i], max).toFixed(2));
      dot.setAttribute("r", "1.3");
      svg.appendChild(dot);
    }
  };
  points(latencies, maxLatency, "point-latency");
  points(costs, maxCost, "point-cost");

  chart.appendChild(svg);

  // Time range + legend with the latest actual value per series. These are
  // HTML in the figcaption, not SVG <text>: the chart uses
  // preserveAspectRatio="none", which would non-uniformly stretch any text.
  const firstLabel = new Date(ordered[0].timestamp).toLocaleTimeString();
  const lastLabel = new Date(ordered[n - 1].timestamp).toLocaleTimeString();
  const range = document.createElement("span");
  range.className = "chart-range";
  range.textContent = `${firstLabel} – ${lastLabel}`;
  caption.appendChild(range);

  const legend = document.createElement("div");
  legend.className = "chart-legend";
  const legendItems = [
    { cls: "latency", label: `Latency · ${formatLatency(maxLatency)} peak` },
    { cls: "cost", label: `Cost · ${formatUsd(maxCost)} peak` },
  ];
  for (const item of legendItems) {
    const wrap = document.createElement("span");
    wrap.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = `swatch ${item.cls}`;
    wrap.append(swatch, document.createTextNode(item.label));
    legend.appendChild(wrap);
  }
  caption.appendChild(legend);
}

// --- Sort + paginate -------------------------------------------------------
function sortDashboardRows() {
  const dir = dashboard.sortDir;
  dashboard.recent.sort((a, b) => {
    let av;
    let bv;
    switch (dashboard.sortKey) {
      case "timestamp":
        av = a.timestamp;
        bv = b.timestamp;
        break;
      case "tokens":
        av = a.prompt_tokens + a.completion_tokens;
        bv = b.prompt_tokens + b.completion_tokens;
        break;
      case "cost":
        av = a.cost_usd;
        bv = b.cost_usd;
        break;
      case "latency":
        av = a.latency_ms;
        bv = b.latency_ms;
        break;
      case "status":
        av = statusOf(a);
        bv = statusOf(b);
        break;
      default:
        av = 0;
        bv = 0;
    }
    if (av < bv) return -1 * dir;
    if (av > bv) return 1 * dir;
    return 0;
  });
}

function pageCount() {
  return Math.max(1, Math.ceil(dashboard.recent.length / DASHBOARD_PAGE_SIZE));
}

function pageRows() {
  const start = dashboard.page * DASHBOARD_PAGE_SIZE;
  return dashboard.recent.slice(start, start + DASHBOARD_PAGE_SIZE);
}

function renderTableHeadSortState() {
  $$("#recent-table th").forEach((th) => {
    const btn = th.querySelector(".th-sort");
    if (!btn) return;
    if (btn.dataset.sort === dashboard.sortKey) {
      th.setAttribute("aria-sort", dashboard.sortDir === 1 ? "ascending" : "descending");
    } else {
      th.removeAttribute("aria-sort");
    }
  });
}

function renderRecentTable() {
  const tbody = $("#recent-table tbody");
  const empty = $("#recent-empty");
  const wrap = $("#table-wrap");

  if (!dashboard.recent.length) {
    wrap.hidden = true;
    empty.hidden = false;
    $("#pagination").replaceChildren();
    return;
  }

  wrap.hidden = false;
  empty.hidden = true;

  sortDashboardRows();
  renderTableHeadSortState();
  tbody.replaceChildren();

  for (const row of pageRows()) {
    const status = statusOf(row);
    const tr = document.createElement("tr");
    const cells = [
      new Date(row.timestamp).toLocaleTimeString(),
      formatTokens(row.prompt_tokens + row.completion_tokens),
      formatUsd(row.cost_usd),
      formatLatency(row.latency_ms),
    ];
    for (const text of cells) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    }
    const tdStatus = document.createElement("td");
    const pill = document.createElement("span");
    pill.className = `status-pill ${status}`;
    pill.textContent = statusLabel(status);
    tdStatus.appendChild(pill);
    tr.appendChild(tdStatus);
    tbody.appendChild(tr);
  }

  renderPagination();
}

function renderPagination() {
  const nav = $("#pagination");
  nav.replaceChildren();
  const pages = pageCount();
  if (pages <= 1) return;

  const addButton = (label, page, opts = {}) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "page-btn";
    btn.textContent = label;
    btn.disabled = !!opts.disabled;
    if (opts.current) btn.setAttribute("aria-current", "page");
    btn.addEventListener("click", () => {
      dashboard.page = page;
      renderRecentTable();
    });
    return btn;
  };

  nav.appendChild(
    addButton("Previous", dashboard.page - 1, {
      disabled: dashboard.page === 0,
    })
  );
  for (let p = 0; p < pages; p += 1) {
    nav.appendChild(addButton(String(p + 1), p, { current: p === dashboard.page }));
  }
  nav.appendChild(
    addButton("Next", dashboard.page + 1, {
      disabled: dashboard.page === pages - 1,
    })
  );

  const info = document.createElement("span");
  info.className = "page-info";
  info.textContent = `Page ${dashboard.page + 1} of ${pages}`;
  nav.appendChild(info);
}

// --- Wire sortable headers ------------------------------------------------
function setupDashboardSort() {
  $$("#recent-table .th-sort").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.sort;
      if (dashboard.sortKey === key) {
        dashboard.sortDir *= -1;
      } else {
        dashboard.sortKey = key;
        dashboard.sortDir = key === "timestamp" ? -1 : 1;
      }
      dashboard.page = 0;
      renderRecentTable();
    });
  });
}

// --- Refresh ---------------------------------------------------------------
async function refreshDashboard() {
  renderDashboardSkeleton();
  try {
    const [summary, recent, quality, perf] = await Promise.all([
      api("/api/v1/dashboard/summary"),
      api("/api/v1/dashboard/recent?limit=200"),
      api("/api/v1/dashboard/quality"),
      api("/api/v1/dashboard/performance"),
    ]);

    dashboard.recent = Array.isArray(recent) ? recent : [];
    dashboard.page = 0;

    renderSummary(summary);
    renderQuality(quality);
    renderPerformance(perf);
    renderChart(dashboard.recent);
    renderRecentTable();
  } catch {
    // Dashboard is best-effort; stay in the empty state if the endpoint is
    // unavailable.
    $("#summary").replaceChildren();
    $("#perf-cards").replaceChildren();
    $("#perf-nodes").replaceChildren();
    $("#perf-empty").hidden = false;
    $("#perf-nodes-empty").hidden = false;
    $("#human-vs-agent").replaceChildren();
    $("#perf-versions tbody").replaceChildren();
    $("#chart").replaceChildren();
    $("#chart-figure").hidden = true;
    $("#chart-empty").hidden = true;
    $("#recent-table tbody").replaceChildren();
    $("#recent-empty").hidden = false;
    $("#pagination").replaceChildren();
    showToast("Could not load the dashboard.", "error");
  }
}

function initDashboard() {
  setupDashboardSort();

  // The chart is a responsive SVG; re-render it on window resize but debounced
  // so rapid resize events don't thrash the layout (the SVG is cheap to
  // redraw, but the debounce avoids jank during an interactive drag-resize).
  window.addEventListener(
    "resize",
    debounce(() => {
      if (dashboard.recent.length) renderChart(dashboard.recent);
    }, 150)
  );
}

export { refreshDashboard, initDashboard };
