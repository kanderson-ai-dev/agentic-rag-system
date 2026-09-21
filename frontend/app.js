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

// --- Human review modal ---
let reviewMode = null;

function openReviewModal(interrupt) {
  reviewMode = null;
  $("#review-question").textContent = interrupt.question;
  $("#review-docs").textContent =
    interrupt.best_documents.slice(0, 2).join("\n\n") || "(no documents)";
  $("#review-input").hidden = true;
  $("#review-confirm-row").hidden = true;
  $("#review-modal").showModal();
}

$("#review-approve").addEventListener("click", () => submitReview("approve"));
$("#review-retry").addEventListener("click", () => {
  reviewMode = "retry";
  $("#review-input").hidden = false;
  $("#review-input").placeholder = "Revised question";
  $("#review-input").value = "";
  $("#review-confirm-row").hidden = false;
});
$("#review-override").addEventListener("click", () => {
  reviewMode = "override";
  $("#review-input").hidden = false;
  $("#review-input").placeholder = "Manual answer";
  $("#review-input").value = "";
  $("#review-confirm-row").hidden = false;
});
$("#review-confirm").addEventListener("click", () => {
  submitReview(reviewMode);
});
$("#review-cancel").addEventListener("click", () => {
  $("#review-modal").close();
});

async function submitReview(decision) {
  const payload = { decision };
  if (decision === "retry") payload.revised_question = $("#review-input").value;
  if (decision === "override") payload.override_answer = $("#review-input").value;

  $("#review-modal").close();
  if (window.chatBusy) window.chatBusy(true);
  showTyping();
  try {
    const result = await api(`/api/v1/rag/query/${currentThreadId}/review`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    hideTyping();
    if (result.status === "interrupted") {
      openReviewModal(result.interrupt);
    } else {
      addAssistantMessage(result.answer || "(no answer)", result.sources || []);
      refreshDashboard();
    }
  } catch (error) {
    hideTyping();
    addErrorWithRetry(`Something went wrong: ${error.message}`, () => {
      submitReview(decision);
    });
  } finally {
    if (window.chatBusy) window.chatBusy(false);
  }
}

// --- Dashboard ---
function renderQuality(quality) {
  const el = $("#quality");
  if (!quality.available) {
    el.innerHTML = `<p class="muted small">No evaluation scorecard yet — run <code>python evaluation/run_ragas.py</code>.</p>`;
    return;
  }
  const rows = Object.entries(quality.metrics)
    .map(([name, m]) => {
      const cls = m.passing ? "pass" : "fail";
      const label = name.replaceAll("_", " ");
      return `
        <div class="stat quality-stat ${cls}">
          <div class="value">${m.score.toFixed(2)}</div>
          <div class="label">${label} <span class="threshold">(≥ ${m.threshold})</span></div>
        </div>
      `;
    })
    .join("");
  el.innerHTML = rows;
}

async function refreshDashboard() {
  try {
    const summary = await api("/api/v1/dashboard/summary");
    const recent = await api("/api/v1/dashboard/recent?limit=10");
    const quality = await api("/api/v1/dashboard/quality");

    $("#summary").innerHTML = `
      <div class="stat"><div class="value">$${summary.total_cost_usd.toFixed(4)}</div><div class="label">Total cost</div></div>
      <div class="stat"><div class="value">${summary.avg_latency_ms.toFixed(0)}ms</div><div class="label">Avg latency</div></div>
      <div class="stat"><div class="value">${summary.blocked_requests}</div><div class="label">Blocked</div></div>
      <div class="stat"><div class="value">${summary.escalated_requests}</div><div class="label">Escalated to human</div></div>
    `;
    renderQuality(quality);

    const tbody = $("#recent-table tbody");
    tbody.innerHTML = "";
    for (const row of recent) {
      const status = row.blocked ? "blocked" : row.escalated ? "escalated" : "ok";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${new Date(row.timestamp).toLocaleTimeString()}</td>
        <td>${row.prompt_tokens + row.completion_tokens}</td>
        <td>$${row.cost_usd.toFixed(5)}</td>
        <td>${row.latency_ms}ms</td>
        <td>${status}</td>
      `;
      tbody.appendChild(tr);
    }
  } catch {
    // Dashboard is best-effort; ignore failures while signed out.
  }
}
