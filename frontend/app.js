// Agentic RAG frontend — no build step, no framework, no external deps.

let accessToken = null; // kept in memory only (not localStorage)
let currentThreadId = null;

const $ = (sel) => document.querySelector(sel);

function addMessage(text, kind = "assistant") {
  const el = document.createElement("div");
  el.className = `message ${kind}`;
  el.textContent = text;
  $("#messages").appendChild(el);
  el.scrollIntoView({ behavior: "smooth" });
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (accessToken) headers.Authorization = `Bearer ${accessToken}`;
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

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
    $("#auth-status").textContent = "signed in";
    $("#auth-status").classList.add("signed-in");
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
$("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("#question").value.trim();
  if (!question) return;
  addMessage(question, "user");
  $("#question").value = "";

  try {
    const result = await api("/api/v1/rag/query", {
      method: "POST",
      body: JSON.stringify({ question }),
    });
    currentThreadId = result.thread_id;

    if (result.status === "interrupted") {
      openReviewModal(result.interrupt);
    } else {
      addMessage(result.answer || "(no answer)");
      refreshDashboard();
    }
  } catch (error) {
    addMessage(`Error: ${error.message}`, "system");
  }
});

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
  try {
    const result = await api(`/api/v1/rag/query/${currentThreadId}/review`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    if (result.status === "interrupted") {
      openReviewModal(result.interrupt);
    } else {
      addMessage(result.answer || "(no answer)");
      refreshDashboard();
    }
  } catch (error) {
    addMessage(`Error: ${error.message}`, "system");
  }
}

// --- Dashboard ---
async function refreshDashboard() {
  try {
    const summary = await api("/api/v1/dashboard/summary");
    const recent = await api("/api/v1/dashboard/recent?limit=10");

    $("#summary").innerHTML = `
      <div class="stat"><div class="value">$${summary.total_cost_usd.toFixed(4)}</div><div class="label">Total cost</div></div>
      <div class="stat"><div class="value">${summary.avg_latency_ms.toFixed(0)}ms</div><div class="label">Avg latency</div></div>
      <div class="stat"><div class="value">${summary.blocked_requests}</div><div class="label">Blocked</div></div>
      <div class="stat"><div class="value">${summary.escalated_requests}</div><div class="label">Escalated to human</div></div>
    `;

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
