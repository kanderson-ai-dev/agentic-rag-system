// Agentic RAG frontend — no build step, no framework, no external deps.

let accessToken = null; // kept in memory only (not localStorage)
let currentThreadId = null;

// A random id per browser session (sessionStorage: cleared when the tab
// closes, unique per new tab/incognito window). Sent as `X-Session-Id` so the
// dashboard can scope its aggregates to "this visit" instead of the
// service's lifetime totals — a first-time viewer always sees zeros.
function getSessionId() {
  let sessionId = sessionStorage.getItem("session_id");
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    sessionStorage.setItem("session_id", sessionId);
  }
  return sessionId;
}

const $ = (sel) => document.querySelector(sel);

// --- Theme toggle ---
// The initial theme is applied inline in <head> (before first paint). This
// handler only flips between themes and persists the choice to localStorage,
// so the preference survives reloads and new tabs within the same browser.
function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem("theme", theme);
  } catch {
    // Storage may be unavailable (private mode, blocked cookies); the theme
    // still applies for this page view, it just won't persist.
  }
  const label = theme === "light" ? "Switch to dark theme" : "Switch to light theme";
  $("#theme-toggle").setAttribute("aria-label", label);
  $("#theme-toggle").setAttribute("title", label);
}

$("#theme-toggle").addEventListener("click", () => {
  setTheme(currentTheme() === "light" ? "dark" : "light");
});

// Keep the toggle's accessible label in sync with the theme applied at load.
setTheme(currentTheme());

// --- Chat experience helpers (Phase 4) ------------------------------------

// XSS-safe Markdown renderer. Builds DOM nodes with `textContent` / safe tag
// construction only — it never parses untrusted strings through innerHTML, so
// LLM output cannot inject <script>/event handlers. Raw HTML in the answer is
// always treated as literal text, never as markup.
//
// Supported subset: fenced ```code```, inline `code`, headings (#..######),
// unordered (-/*) and ordered (1.) lists, blockquotes (>), paragraphs, and
// inline **bold** / *italic*.
function renderMarkdown(source) {
  const root = document.createDocumentFragment();
  const lines = String(source ?? "").split(/\r?\n/);
  let i = 0;

  const inline = (text) => {
    const frag = document.createDocumentFragment();
    // Bold before italic so `**a *b* c**` doesn't mis-nest. Regex alternates
    // **bold**, *italic*, and inline `code`.
    const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
    let last = 0;
    let match;
    while ((match = re.exec(text)) !== null) {
      if (match.index > last) {
        frag.appendChild(document.createTextNode(text.slice(last, match.index)));
      }
      const token = match[0];
      if (token.startsWith("**")) {
        const el = document.createElement("strong");
        el.textContent = token.slice(2, -2);
        frag.appendChild(el);
      } else if (token.startsWith("`")) {
        const el = document.createElement("code");
        el.textContent = token.slice(1, -1);
        frag.appendChild(el);
      } else {
        const el = document.createElement("em");
        el.textContent = token.slice(1, -1);
        frag.appendChild(el);
      }
      last = match.index + token.length;
    }
    if (last < text.length) {
      frag.appendChild(document.createTextNode(text.slice(last)));
    }
    return frag;
  };

  const newEl = (tag) => document.createElement(tag);

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block: ``` (optionally with a language).
    if (/^\s*```/.test(line)) {
      const codeLines = [];
      i += 1;
      while (i < lines.length && !/^\s*```/.test(lines[i])) {
        codeLines.push(lines[i]);
        i += 1;
      }
      i += 1; // skip the closing fence
      const pre = newEl("pre");
      const code = newEl("code");
      code.textContent = codeLines.join("\n");
      pre.appendChild(code);
      root.appendChild(pre);
      continue;
    }

    // Blank line separates blocks.
    if (/^\s*$/.test(line)) {
      i += 1;
      continue;
    }

    // Headings.
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      const el = newEl(`h${heading[1].length}`);
      el.appendChild(inline(heading[2]));
      root.appendChild(el);
      i += 1;
      continue;
    }

    // Unordered list.
    if (/^\s*[-*]\s+/.test(line)) {
      const ul = newEl("ul");
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        const li = newEl("li");
        li.appendChild(inline(lines[i].replace(/^\s*[-*]\s+/, "")));
        ul.appendChild(li);
        i += 1;
      }
      root.appendChild(ul);
      continue;
    }

    // Ordered list.
    if (/^\s*\d+\.\s+/.test(line)) {
      const ol = newEl("ol");
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        const li = newEl("li");
        li.appendChild(inline(lines[i].replace(/^\s*\d+\.\s+/, "")));
        ol.appendChild(li);
        i += 1;
      }
      root.appendChild(ol);
      continue;
    }

    // Blockquote.
    if (/^\s*>/.test(line)) {
      const quote = newEl("blockquote");
      while (i < lines.length && /^\s*>/.test(lines[i])) {
        const p = newEl("p");
        p.appendChild(inline(lines[i].replace(/^\s*>\s?/, "")));
        quote.appendChild(p);
        i += 1;
      }
      root.appendChild(quote);
      continue;
    }

    // Paragraph (possibly spanning contiguous text lines).
    const paraLines = [line];
    i += 1;
    while (
      i < lines.length &&
      !/^\s*$/.test(lines[i]) &&
      !/^\s*(```|#{1,6}\s|[-*]\s|\d+\.\s|>)/.test(lines[i])
    ) {
      paraLines.push(lines[i]);
      i += 1;
    }
    const p = newEl("p");
    p.appendChild(inline(paraLines.join("\n")));
    root.appendChild(p);
  }

  return root;
}

// Build (or locate) the assistant message row and render `markdown` into the
// `.message-body` slot using the XSS-safe renderer, then attach source chips
// and a copy button.
function addAssistantMessage(markdown, sources = []) {
  const message = createMessage("assistant");

  const meta = document.createElement("div");
  meta.className = "message-meta";

  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "copy-btn";
  copyBtn.textContent = "Copy";
  copyBtn.addEventListener("click", () => copyMessageText(copyBtn, markdown));
  meta.appendChild(copyBtn);

  const body = document.createElement("div");
  body.className = "message-body";
  body.appendChild(renderMarkdown(markdown));

  const text = document.createElement("div");
  text.className = "message-text";
  text.appendChild(body);

  message.appendChild(meta);
  message.appendChild(text);

  if (sources && sources.length) {
    const sourceRow = document.createElement("div");
    sourceRow.className = "sources";
    (sources.slice(0, 4)).forEach((source) => {
      const chip = document.createElement("span");
      chip.className = "source-chip";
      chip.textContent = source;
      sourceRow.appendChild(chip);
    });
    message.appendChild(sourceRow);
  }

  $("#messages").appendChild(message);
  smartScroll();
  return message;
}

// Copy the raw answer text and give transient feedback on the button.
async function copyMessageText(button, text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // Clipboard API may be unavailable (insecure context); fell back to a
    // hidden textarea selection so the copy button still works.
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  const original = button.textContent;
  button.textContent = "Copied!";
  button.classList.add("copied");
  setTimeout(() => {
    button.textContent = original;
    button.classList.remove("copied");
  }, 1500);
}

// Create a message row container (user / assistant / system / error / typing).
function createMessage(kind) {
  // Dismiss the welcome state once real content arrives.
  const welcome = $("#chat-welcome");
  if (welcome) welcome.remove();

  const el = document.createElement("div");
  el.className = `message ${kind}`;
  el.setAttribute("role", "listitem");
  return el;
}

function addMessage(text, kind = "assistant") {
  const el = createMessage(kind);
  el.textContent = text;
  $("#messages").appendChild(el);
  smartScroll();
  return el;
}

// --- Typing indicator -----------------------------------------------------
// A transient assistant bubble with animated dots, shown while a query is in
// flight and removed/resolved when the answer (or error) arrives.
function showTyping() {
  hideTyping();
  const el = document.createElement("div");
  el.className = "message assistant typing-message";
  el.id = "typing-indicator";
  const dots = document.createElement("span");
  dots.className = "typing";
  dots.setAttribute("aria-label", "Assistant is typing");
  for (let n = 0; n < 3; n += 1) {
    const dot = document.createElement("span");
    dot.className = "dot";
    dots.appendChild(dot);
  }
  el.appendChild(dots);
  $("#messages").appendChild(el);
  smartScroll();
}

function hideTyping() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

// --- Smart auto-scroll ---------------------------------------------------
// Keep the newest message visible, but only auto-scroll when the user is
// already near the bottom — otherwise let them read earlier messages without
// being yanked down.
function smartScroll() {
  const container = $("#messages");
  const threshold = 48; // px from bottom considered "at the bottom"
  const atBottom =
    container.scrollHeight - container.scrollTop - container.clientHeight <=
    threshold;
  if (atBottom) {
    container.scrollTop = container.scrollHeight;
  }
}

// --- Composer: adaptive height + send/error/loading state -----------------
(function setupComposer() {
  const textarea = $("#question");
  const sendBtn = $("#chat-send");

  // Grow the textarea up to a max height as content wraps.
  textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
  });

  function setBusy(busy) {
    sendBtn.disabled = busy;
    sendBtn.textContent = busy ? "Sending…" : "Send";
    textarea.disabled = busy;
  }

  window.chatBusy = setBusy;
})();

async function api(path, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    "X-Session-Id": getSessionId(),
    ...(options.headers || {}),
  };
  if (accessToken) headers.Authorization = `Bearer ${accessToken}`;
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

// --- Auth gating ---
// Ask the backend whether login is required before deciding what to show.
// Without this check, a deployment that leaves JWT_SECRET_KEY/AUTH_USERNAME/
// AUTH_PASSWORD_HASH unset (open quickstart mode) would show a sign-in form
// that can never succeed, hiding the chat/dashboard behind it.

// The header badge has three mutually-exclusive states, rendered distinctly:
//   * "signed out"   (default) — auth is enabled and no valid token is held.
//   * "signed in"    — auth is enabled and a token was issued this session.
//   * "auth disabled" — auth is not configured (open quickstart mode).
function setAuthStatus(text, state) {
  const el = $("#auth-status");
  el.textContent = text;
  el.classList.remove("signed-in", "auth-disabled");
  if (state) el.classList.add(state);
}

async function initAuthGate() {
  try {
    const { auth_required: authRequired } = await api("/api/v1/auth/status");
    if (!authRequired) {
      setAuthStatus("auth disabled", "auth-disabled");
      $("#login-panel").hidden = true;
      $("#chat-panel").hidden = false;
      $("#dashboard-panel").hidden = false;
      refreshDashboard();
    }
  } catch {
    // If the check itself fails, fall back to the login panel (safe default).
  }
}
initAuthGate();

// --- Login ---
$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("#login-error").hidden = true;
  try {
    const result = await api("/api/v1/auth/login", {
      method: "POST",
      body: JSON.stringify({
        username: $("#username").value,
        password: $("#password").value,
      }),
    });
    accessToken = result.access_token;
    setAuthStatus("signed in", "signed-in");
    $("#login-panel").hidden = true;
    $("#chat-panel").hidden = false;
    $("#dashboard-panel").hidden = false;
    refreshDashboard();
  } catch (error) {
    $("#login-error").textContent = error.message;
    $("#login-error").hidden = false;
  }
});

// --- Chat ---
// The question field is a <textarea> (so Shift+Enter can add a newline),
// but browsers only auto-submit <input>s on Enter. Submit explicitly on
// plain Enter, keeping Shift+Enter for multi-line questions.
$("#question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#chat-form").requestSubmit();
  }
});

$("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("#question").value.trim();
  if (!question) return;
  addMessage(question, "user");
  $("#question").value = "";
  resetComposerHeight();
  await sendQuestion(question);
});

// Send a question to the backend, driving the typing indicator and handling
// the completed / interrupted / error outcome. Exposes a retry affordance for
// transient failures.
async function sendQuestion(question) {
  if (window.chatBusy) window.chatBusy(true);
  showTyping();
  try {
    const result = await api("/api/v1/rag/query", {
      method: "POST",
      body: JSON.stringify({ question }),
    });
    hideTyping();
    currentThreadId = result.thread_id;

    if (result.status === "interrupted") {
      openReviewModal(result.interrupt);
    } else {
      addAssistantMessage(result.answer || "(no answer)", result.sources || []);
      refreshDashboard();
    }
  } catch (error) {
    hideTyping();
    addErrorWithRetry(`Something went wrong: ${error.message}`, () =>
      sendQuestion(question)
    );
  } finally {
    if (window.chatBusy) window.chatBusy(false);
  }
}

// Render an error bubble with a "Retry" button wired to re-run the action.
function addErrorWithRetry(text, retry) {
  const el = createMessage("system");
  el.classList.add("error");
  el.textContent = text;
  const retryBtn = document.createElement("button");
  retryBtn.type = "button";
  retryBtn.className = "btn secondary retry-btn";
  retryBtn.textContent = "Retry";
  retryBtn.addEventListener("click", () => {
    el.remove();
    retry();
  });
  el.appendChild(retryBtn);
  $("#messages").appendChild(el);
  smartScroll();
}

function resetComposerHeight() {
  const textarea = $("#question");
  textarea.style.height = "auto";
}

// --- Human review modal (Phase 5) -----------------------------------------
// The review flow is fully keyboard-operable and accessible:
//   * A native `<dialog>` traps focus and closes on Esc automatically.
//   * The three decisions (approve / retry / override) are radio inputs with
//     a clear explanation each, selectable via arrow keys / click.
//   * Retry and override reveal a labeled input that is validated before
//     submission — empty input blocks submission with a visible error.
//   * Submitting shows a busy state on the confirm button and disables the
//     controls; failures surface inside the modal (with retry) instead of
//     dumping an error into the chat.

const REVIEW_INPUT_LABELS = {
  retry: "Revised question",
  override: "Manual answer",
};
const REVIEW_INPUT_PLACEHOLDERS = {
  retry: "Enter a clearer question to re-run retrieval…",
  override: "Write the correct answer…",
};

let reviewMode = null;
let reviewRetryPayload = null; // remembered decision shape for error retry

function selectedReviewDecision() {
  const checked = $('input[name="review-decision"]:checked') || null;
  return checked ? checked.value : null;
}

function setReviewMode(mode) {
  reviewMode = mode;
  const inputRow = $("#review-input-row");
  const input = $("#review-input");
  const label = $("#review-input-label");
  if (mode === "retry" || mode === "override") {
    label.textContent = REVIEW_INPUT_LABELS[mode];
    input.placeholder = REVIEW_INPUT_PLACEHOLDERS[mode];
    input.value = "";
    inputRow.hidden = false;
    input.focus();
  } else {
    input.value = "";
    inputRow.hidden = true;
  }
  clearReviewErrors();
}

function clearReviewErrors() {
  $("#review-input-error").hidden = true;
  $("#review-input-error").textContent = "";
  $("#review-submit-error").hidden = true;
  $("#review-submit-error").textContent = "";
}

function setReviewBusy(busy) {
  const submit = $("#review-submit");
  const cancel = $("#review-cancel");
  submit.disabled = busy;
  cancel.disabled = busy;
  submit.textContent = busy ? "Submitting…" : "Confirm";
  document
    .querySelectorAll('input[name="review-decision"]')
    .forEach((radio) => {
      radio.disabled = busy;
    });
  $("#review-input").disabled = busy;
}

function openReviewModal(interrupt) {
  // Reset radio selection and mode on each open.
  document
    .querySelectorAll('input[name="review-decision"]')
    .forEach((radio) => {
      radio.checked = false;
    });
  setReviewMode(null);
  reviewRetryPayload = null;

  $("#review-question").textContent = interrupt.question || "(no question)";
  const docs = (interrupt.best_documents || []).slice(0, 2).join("\n\n");
  $("#review-docs").textContent =
    docs || "(no retrieved documents available)";

  $("#review-modal").showModal();
  // Move focus to the first decision for immediate keyboard use.
  $("#review-approve").focus();
}

// Persist the chosen mode when the user picks a decision.
document
  .querySelectorAll('input[name="review-decision"]')
  .forEach((radio) => {
    radio.addEventListener("change", () => setReviewMode(radio.value));
  });

// Clear the input error as soon as the user starts typing again.
$("#review-input").addEventListener("input", () => {
  $("#review-input-error").hidden = true;
});

$("#review-cancel").addEventListener("click", () => {
  $("#review-modal").close();
});

$("#review-submit").addEventListener("click", () => {
  submitReview(selectedReviewDecision() || reviewMode);
});

async function submitReview(decision) {
  if (!decision) {
    showReviewError("Please choose one of the three options before confirming.");
    return;
  }

  const input = $("#review-input").value.trim();
  if ((decision === "retry" || decision === "override") && !input) {
    const inputError = $("#review-input-error");
    inputError.textContent =
      `Please enter a ${decision === "retry" ? "revised question" : "manual answer"}.`;
    inputError.hidden = false;
    $("#review-input").focus();
    return;
  }

  const payload = { decision };
  if (decision === "retry") payload.revised_question = input;
  if (decision === "override") payload.override_answer = input;
  reviewRetryPayload = payload;

  setReviewBusy(true);
  clearReviewErrors();
  try {
    const result = await api(`/api/v1/rag/query/${currentThreadId}/review`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    $("#review-modal").close();
    if (result.status === "interrupted") {
      openReviewModal(result.interrupt);
    } else {
      addAssistantMessage(result.answer || "(no answer)", result.sources || []);
      refreshDashboard();
    }
  } catch (error) {
    // Keep the modal open so the user can correct and retry in place.
    showReviewError(`Something went wrong: ${error.message}`);
  } finally {
    setReviewBusy(false);
  }
}

function showReviewError(message) {
  const submitError = $("#review-submit-error");
  submitError.textContent = message;
  submitError.hidden = false;
}

// --- Dashboard (Phase 6) ---------------------------------------------------
// Renders real usage metadata into (a) metric cards, (b) a dependency-free SVG
// chart of cost/latency over time, and (c) a sortable, paginated recent-requests
// table. All server-derived values are written with `textContent` (never
// interpolated into innerHTML), and empty + loading (skeleton) states are
// explicit.

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
    summary.appendChild(Object.assign(document.createElement("div"), {
      className: "skeleton stat",
    }));
  }

  $("#chart-figure").hidden = true;
  $("#chart-empty").hidden = true;
  $("#chart").replaceChildren(Object.assign(document.createElement("div"), {
    className: "skeleton chart",
  }));

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

  const rawLatency = latencies.map((v, i) => `${x(i).toFixed(2)},${y(v, maxLatency).toFixed(2)}`).join(" ");
  const rawCost = costs.map((v, i) => `${x(i).toFixed(2)},${y(v, maxCost).toFixed(2)}`).join(" ");

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

  // Axis labels: first/last timestamps and the peak value of each series.
  const axisLabel = (text, cx, cy) => {
    const t = document.createElementNS(NS, "text");
    t.setAttribute("class", "axis-label");
    t.setAttribute("x", cx.toFixed(2));
    t.setAttribute("y", cy.toFixed(2));
    t.textContent = text;
    svg.appendChild(t);
  };
  const firstLabel = new Date(ordered[0].timestamp).toLocaleTimeString();
  const lastLabel = new Date(ordered[n - 1].timestamp).toLocaleTimeString();
  axisLabel(firstLabel, pad, H - 1);
  axisLabel(lastLabel, W - pad, H - 1);

  chart.appendChild(svg);

  // Legend with the latest actual value per series.
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
  document.querySelectorAll("#recent-table th").forEach((th) => {
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

  nav.appendChild(addButton("Previous", dashboard.page - 1, {
    disabled: dashboard.page === 0,
  }));
  for (let p = 0; p < pages; p += 1) {
    nav.appendChild(addButton(String(p + 1), p, { current: p === dashboard.page }));
  }
  nav.appendChild(addButton("Next", dashboard.page + 1, {
    disabled: dashboard.page === pages - 1,
  }));

  const info = document.createElement("span");
  info.className = "page-info";
  info.textContent = `Page ${dashboard.page + 1} of ${pages}`;
  nav.appendChild(info);
}

// --- Wire sortable headers ------------------------------------------------
function setupDashboardSort() {
  document.querySelectorAll("#recent-table .th-sort").forEach((btn) => {
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
    const [summary, recent, quality] = await Promise.all([
      api("/api/v1/dashboard/summary"),
      api("/api/v1/dashboard/recent?limit=200"),
      api("/api/v1/dashboard/quality"),
    ]);

    dashboard.recent = Array.isArray(recent) ? recent : [];
    dashboard.page = 0;

    renderSummary(summary);
    renderQuality(quality);
    renderChart(dashboard.recent);
    renderRecentTable();
  } catch {
    // Dashboard is best-effort; stay in the empty state on failure while
    // signed out or if the endpoint is unavailable.
    $("#summary").replaceChildren();
    $("#chart").replaceChildren();
    $("#chart-figure").hidden = true;
    $("#chart-empty").hidden = true;
    $("#recent-table tbody").replaceChildren();
    $("#recent-empty").hidden = false;
    $("#pagination").replaceChildren();
  }
}

setupDashboardSort();
