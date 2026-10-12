// Prompt Registry console section: version timeline, structured diffs,
// candidate editor, async A/B runs, and gated promote/rollback.
//
// Every identifier goes through `label(kind, key)` — the UI never prints a
// raw `snake_case` value. Mutations reuse the native-<dialog> pattern from
// the review modal and are only attempted when the deployment allows writes
// (`writes_enabled` from the list endpoint); when JWT auth is configured the
// operator sign-in appears on demand, only for mutating actions.

import { $, el } from "/js/util.js";
import { api } from "/js/api.js";
import { label } from "/js/labels.js";
import { announce, showToast } from "./toast.js";

// Scorecard metrics worth surfacing in timelines and the A/B comparison.
const SCORECARD_METRICS = [
  "faithfulness",
  "answer_relevancy",
  "context_precision",
  "context_recall",
  "llm_as_judge",
  "cost_per_query_usd",
  "latency_p50_ms",
  "latency_p95_ms",
];

const POLL_MS = 2000;

// Module-scoped state.
let authToken = null;
let authRequired = false;
let writesEnabled = false;
let selected = null; // { name, promoted_version, versions }
let abJobs = new Map(); // `${name}:v${version}` -> job payload
let pollTimer = null;
let confirmAction = null;

function jobKey(name, version) {
  return `${name}:v${version}`;
}

function canWrite() {
  return writesEnabled && (!authRequired || !!authToken);
}

function writeHeaders() {
  return authToken ? { Authorization: `Bearer ${authToken}` } : {};
}

function showError(message) {
  const box = $("#prompts-error");
  box.textContent = message;
  box.hidden = false;
}

function clearError() {
  const box = $("#prompts-error");
  box.textContent = "";
  box.hidden = true;
}

// Gate every mutating action: disabled deployments explain themselves via
// `aria-describedby`; JWT-protected ones reveal the operator sign-in only
// when a write is attempted.
function guarded(fn) {
  return (...args) => {
    if (!writesEnabled) return;
    if (authRequired && !authToken) {
      const panel = $("#prompts-auth");
      panel.hidden = false;
      $("#prompts-auth-note").textContent =
        "Sign in to create candidates, run A/B tests, or promote versions.";
      $("#prompts-username").focus();
      return;
    }
    fn(...args);
  };
}

// --- List ---------------------------------------------------------------

function renderList(data) {
  const list = $("#prompt-list");
  list.replaceChildren();
  const prompts = data.prompts || [];
  $("#prompts-empty").hidden = prompts.length > 0;

  const chain = $("#prompts-chain");
  chain.textContent = data.audit_chain_verified
    ? "Audit chain ✓ verified"
    : "Audit chain ✗ broken";
  chain.className = `prompt-chain ${data.audit_chain_verified ? "ok" : "bad"}`;

  for (const p of prompts) {
    const row = el("button", {
      type: "button",
      className: "prompt-row",
    });
    row.appendChild(
      el("span", { className: "prompt-row-name", textContent: label("prompt", p.name) })
    );
    row.appendChild(
      el("span", {
        className: "prompt-row-meta muted small",
        textContent: `Active v${p.active_version} · ${p.version_count} version${
          p.version_count === 1 ? "" : "s"
        }`,
      })
    );
    row.addEventListener("click", () => selectPrompt(p.name));
    list.appendChild(row);
  }
}

async function refreshPrompts(keepSelection = true) {
  try {
    const data = await api("/api/v1/prompts/");
    writesEnabled = !!data.writes_enabled;
    applyWritesState();
    renderList(data);
    if (keepSelection && selected) await selectPrompt(selected.name);
  } catch (error) {
    showError(`Could not load the prompt registry: ${error.message}`);
  }
}

// Disable every mutation control when the deployment rejects writes; the
// shared note explains why (aria-describedby target).
function applyWritesState() {
  const note = $("#prompts-writes-note");
  if (writesEnabled) {
    note.textContent = authRequired
      ? "Writes need an operator sign-in — attempted actions will ask for it."
      : "";
  } else {
    note.textContent = "Prompt writes are disabled in this deployment.";
  }
  note.hidden = !note.textContent;
  document.querySelectorAll("[data-write]").forEach((btn) => {
    btn.disabled = !writesEnabled;
  });
}

// --- Detail: timeline + diff + actions -----------------------------------

function scorecardSummary(evaluation) {
  const sc = evaluation?.scorecard;
  if (!sc) return null;
  const parts = SCORECARD_METRICS.filter((m) => sc[m] !== undefined)
    .slice(0, 3)
    .map((m) => `${label("metric", m)} ${formatMetric(m, sc[m])}`);
  return parts.length ? `A/B · ${parts.join(" · ")}` : null;
}

function formatMetric(metric, value) {
  if (metric.startsWith("cost_")) return `$${Number(value).toFixed(4)}`;
  if (metric.startsWith("latency_")) return `${Math.round(value)}ms`;
  if (metric === "llm_as_judge") return Number(value).toFixed(2);
  return Number(value).toFixed(2);
}

function statusPill(status) {
  return el("span", {
    className: `status-pill prompt-status-${status}`,
    textContent: label("status", status),
  });
}

function versionCard(v) {
  const card = el("article", { className: "prompt-version" });
  const head = el("div", { className: "prompt-version-head" });
  head.appendChild(el("strong", { textContent: `v${v.version}` }));
  head.appendChild(statusPill(v.status));
  if (v.version === selected.promoted_version) {
    head.appendChild(el("span", { className: "pill pill-accent", textContent: "live" }));
  }
  card.appendChild(head);
  card.appendChild(
    el("p", {
      className: "muted small",
      textContent: `${v.author} · ${new Date(v.created_at).toLocaleString()}`,
    })
  );
  card.appendChild(el("p", { className: "small", textContent: v.changelog }));
  const summary = scorecardSummary(v.evaluation);
  if (summary) {
    card.appendChild(el("p", { className: "muted small", textContent: summary }));
  }

  const actions = el("div", { className: "prompt-version-actions" });
  if (v.status === "candidate" || v.status === "draft") {
    const job = abJobs.get(jobKey(selected.name, v.version));
    const runBtn = el("button", {
      type: "button",
      className: "btn secondary small",
      textContent: "Run A/B test",
      "data-write": "",
    });
    runBtn.addEventListener("click", guarded(() => startAb(v.version)));
    actions.appendChild(runBtn);

    const promoteBtn = el("button", {
      type: "button",
      className: "btn small",
      textContent: "Promote",
      "data-write": "",
      "aria-describedby": "hint-promote",
    });
    const hasEvidence = !!v.evaluation?.ab_test_id || job?.status === "succeeded";
    promoteBtn.disabled = !hasEvidence;
    promoteBtn.addEventListener("click", guarded(() => askPromote(v.version)));
    actions.appendChild(promoteBtn);
  }
  if (actions.childElementCount) card.appendChild(actions);
  return card;
}

async function selectPrompt(name) {
  clearError();
  try {
    selected = await api(`/api/v1/prompts/${name}`);
  } catch (error) {
    showError(`Could not load ${label("prompt", name)}: ${error.message}`);
    return;
  }
  const detail = $("#prompt-detail");
  detail.hidden = false;
  $("#prompt-detail-title").textContent = label("prompt", name);

  const timeline = $("#prompt-versions");
  timeline.replaceChildren();
  [...selected.versions]
    .sort((a, b) => b.version - a.version)
    .forEach((v) => timeline.appendChild(versionCard(v)));

  renderDiffControls();
  renderActions();
  renderAbPanel();
  applyWritesState();
}

function renderDiffControls() {
  const controls = $("#prompt-diff-controls");
  controls.replaceChildren();
  if (selected.versions.length < 2) {
    $("#prompt-diff").replaceChildren();
    return;
  }
  const mk = (id) => {
    const sel = el("select", { id, className: "diff-select" });
    selected.versions.forEach((v) => {
      sel.appendChild(el("option", { value: v.version, textContent: `v${v.version}` }));
    });
    return sel;
  };
  const selA = mk("diff-a");
  const selB = mk("diff-b");
  selA.value = selected.promoted_version;
  const other = selected.versions.find((v) => v.version !== selA.value);
  if (other) selB.value = other.version;
  const btn = el("button", {
    type: "button",
    className: "btn secondary small",
    textContent: "Compare",
  });
  btn.addEventListener("click", () => showDiff(Number(selA.value), Number(selB.value)));
  controls.appendChild(selA);
  controls.appendChild(selB);
  controls.appendChild(btn);
  showDiff(Number(selA.value), Number(selB.value));
}

async function showDiff(a, b) {
  const host = $("#prompt-diff");
  try {
    const data = await api(`/api/v1/prompts/${selected.name}/diff?a=${a}&b=${b}`);
    host.replaceChildren();
    const table = el("table", { className: "diff-table" });
    const tbody = el("tbody");
    for (const h of data.hunks) {
      const tr = el("tr", { className: `diff-${h.op}` });
      // Meaningful markers — +/- text glyphs, never colour alone.
      const marker = h.op === "add" ? "+" : h.op === "del" ? "−" : " ";
      tr.appendChild(el("td", { className: "diff-marker", textContent: marker }));
      tr.appendChild(
        el("td", {
          className: "diff-lines muted",
          textContent: `${h.old_line ?? ""}→${h.new_line ?? ""}`,
        })
      );
      tr.appendChild(el("td", { className: "diff-text", textContent: h.text }));
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    host.appendChild(table);
  } catch (error) {
    showError(`Could not diff versions: ${error.message}`);
  }
}

// --- Write actions: candidate / A-B / promote / rollback -----------------

function renderActions() {
  const host = $("#prompt-actions");
  host.replaceChildren();
  const candBtn = el("button", {
    type: "button",
    className: "btn secondary small",
    textContent: "New candidate",
    "data-write": "",
  });
  candBtn.addEventListener("click", guarded(openCandidateModal));
  host.appendChild(candBtn);

  const hasRollback = selected.versions.some(
    (v) => v.status === "superseded" || v.status === "rolled_back"
  );
  const rollbackBtn = el("button", {
    type: "button",
    className: "btn ghost small",
    textContent: "Rollback",
    "data-write": "",
    "aria-describedby": "hint-rollback",
  });
  rollbackBtn.disabled = !hasRollback;
  rollbackBtn.addEventListener("click", guarded(askRollback));
  host.appendChild(rollbackBtn);
}

async function openCandidateModal() {
  clearError();
  try {
    const promoted = await api(
      `/api/v1/prompts/${selected.name}/${selected.promoted_version}`
    );
    $("#cand-system").value = promoted.system || "";
    $("#cand-human").value = promoted.human || "";
    $("#cand-vars").textContent = `Required variables: ${promoted.variables
      .map((v) => `{${v}}`)
      .join(", ")}`;
  } catch (error) {
    showError(`Could not preload the live prompt: ${error.message}`);
    return;
  }
  $("#cand-author").value = "";
  $("#cand-description").value = "";
  $("#cand-changelog").value = "";
  $("#cand-error").hidden = true;
  $("#cand-title").textContent = `New candidate — ${label("prompt", selected.name)}`;
  $("#candidate-modal").showModal();
  $("#cand-system").focus();
}

async function submitCandidate() {
  const errBox = $("#cand-error");
  errBox.hidden = true;
  const payload = {
    system: $("#cand-system").value.trim(),
    human: $("#cand-human").value.trim() || null,
    author: $("#cand-author").value.trim(),
    description: $("#cand-description").value.trim(),
    changelog: $("#cand-changelog").value.trim(),
  };
  try {
    await api(`/api/v1/prompts/${selected.name}/candidates`, {
      method: "POST",
      headers: writeHeaders(),
      body: JSON.stringify(payload),
    });
    $("#candidate-modal").close();
    showToast("Candidate created.", "success");
    await refreshPrompts();
  } catch (error) {
    // Validation failures arrive as plain-English `detail` strings — show
    // them verbatim inside the modal so the author can fix and resubmit.
    errBox.textContent = `Could not create the candidate: ${error.message}`;
    errBox.hidden = false;
  }
}

async function startAb(version) {
  try {
    const job = await api(`/api/v1/prompts/${selected.name}/ab-test`, {
      method: "POST",
      headers: writeHeaders(),
      body: JSON.stringify({ candidate_version: version }),
    });
    abJobs.set(jobKey(selected.name, version), {
      ...job,
      candidate_version: version,
    });
    announce("A/B test queued.");
    pollAb(version);
  } catch (error) {
    showError(`Could not start the A/B test: ${error.message}`);
  }
}

async function pollAb(version) {
  if (pollTimer) window.clearTimeout(pollTimer);
  const key = jobKey(selected.name, version);
  const job = abJobs.get(key);
  if (!job) return;
  try {
    const fresh = await api(
      `/api/v1/prompts/${selected.name}/ab-test/${job.job_id}`
    );
    abJobs.set(key, { ...fresh, candidate_version: version });
  } catch (error) {
    showError(`Could not poll the A/B job: ${error.message}`);
    return;
  }
  renderAbPanel();
  const status = abJobs.get(key).status;
  if (status === "queued" || status === "running") {
    pollTimer = window.setTimeout(() => pollAb(version), POLL_MS);
  } else {
    selectPrompt(selected.name); // statuses/evidence may have changed
  }
}

function renderAbPanel() {
  const host = $("#prompt-ab");
  host.replaceChildren();
  const entry = [...abJobs.entries()]
    .filter(([k]) => k.startsWith(`${selected.name}:`))
    .pop();
  if (!entry) return;
  const [, job] = entry;

  const statusText = label("job_status", job.status);
  if (job.status === "queued" || job.status === "running") {
    const note = el("p", {
      className: "muted small",
      textContent: `A/B run ${statusText.toLowerCase()} — evaluating v${job.candidate_version} against the live version…`,
    });
    note.setAttribute("aria-live", "polite");
    host.appendChild(note);
    return;
  }
  if (job.status === "failed") {
    host.appendChild(
      el("p", { className: "error", textContent: `A/B run failed: ${job.error || "unknown error"}` })
    );
    return;
  }

  const result = job.result || {};
  const baseline = result.baseline_scorecard || {};
  const candidate = result.candidate_scorecard || {};
  const deltas = result.deltas || {};

  const table = el("table", { className: "ab-table" });
  table.appendChild(
    el("caption", {
      className: "sr-only",
      textContent: `Baseline v${result.baseline_version} versus candidate v${result.candidate_version}`,
    })
  );
  const thead = el("thead");
  const htr = el("tr");
  for (const h of ["Metric", `v${result.baseline_version} (live)`, `v${result.candidate_version}`, "Delta"]) {
    htr.appendChild(el("th", { scope: "col", textContent: h }));
  }
  thead.appendChild(htr);
  table.appendChild(thead);

  const tbody = el("tbody");
  for (const m of SCORECARD_METRICS) {
    if (baseline[m] === undefined && candidate[m] === undefined) continue;
    const tr = el("tr");
    tr.appendChild(el("td", { textContent: label("metric", m) }));
    tr.appendChild(el("td", { textContent: formatMetric(m, baseline[m] ?? 0) }));
    tr.appendChild(el("td", { textContent: formatMetric(m, candidate[m] ?? 0) }));
    const d = deltas[m];
    tr.appendChild(
      el("td", {
        className: "ab-delta",
        textContent: d === undefined ? "—" : `${d > 0 ? "+" : ""}${formatMetric(m, d)}`,
      })
    );
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  host.appendChild(table);

  const gate = result.gate || {};
  const verdict = el("p", { className: `ab-verdict ${gate.allowed ? "ok" : "bad"}` });
  verdict.textContent = gate.allowed
    ? "Gate verdict: passed — safe to promote."
    : "Gate verdict: blocked.";
  host.appendChild(verdict);
  (gate.failures || []).forEach((f) =>
    host.appendChild(el("p", { className: "muted small", textContent: `• ${f}` }))
  );
}

// --- Confirmation modal (promote / rollback) ------------------------------

function askConfirm(title, text, onConfirm) {
  $("#confirm-title").textContent = title;
  $("#confirm-text").textContent = text;
  confirmAction = onConfirm;
  $("#confirm-modal").showModal();
  $("#confirm-ok").focus();
}

function askPromote(version) {
  const job = abJobs.get(jobKey(selected.name, version));
  const v = selected.versions.find((x) => x.version === version);
  const abId = v?.evaluation?.ab_test_id || job?.job_id;
  askConfirm(
    "Promote candidate?",
    `${label("prompt", selected.name)} v${version} becomes the live prompt for new queries. The current live version moves to superseded.`,
    async () => {
      try {
        await api(`/api/v1/prompts/${selected.name}/promote`, {
          method: "POST",
          headers: writeHeaders(),
          body: JSON.stringify({
            version,
            evaluation: { ab_test_id: abId },
          }),
        });
        showToast(`v${version} promoted.`, "success");
        await refreshPrompts();
      } catch (error) {
        const failures = error.message?.failures;
        showError(
          Array.isArray(failures) && failures.length
            ? `Promotion blocked: ${failures.join(" · ")}`
            : `Promotion failed: ${typeof error.message === "string" ? error.message : "gate rejected the evidence"}`
        );
      }
    }
  );
}

function askRollback() {
  askConfirm(
    "Rollback prompt?",
    `Re-promotes the most recent superseded version of ${label("prompt", selected.name)}.`,
    async () => {
      try {
        await api(`/api/v1/prompts/${selected.name}/rollback`, {
          method: "POST",
          headers: writeHeaders(),
          body: "{}",
        });
        showToast("Rollback applied.", "success");
        await refreshPrompts();
      } catch (error) {
        showError(`Rollback failed: ${error.message}`);
      }
    }
  );
}

// --- Operator sign-in (mutations only) ------------------------------------

async function submitAuth() {
  const errBox = $("#prompts-auth-error");
  errBox.hidden = true;
  try {
    const data = await api("/api/v1/auth/login", {
      method: "POST",
      body: JSON.stringify({
        username: $("#prompts-username").value.trim(),
        password: $("#prompts-password").value,
      }),
    });
    authToken = data.access_token;
    $("#prompts-auth").hidden = true;
    showToast("Signed in — writes enabled.", "success");
    applyWritesState();
  } catch {
    errBox.textContent = "Sign-in failed — check the credentials and try again.";
    errBox.hidden = false;
  }
}

// --- Init ------------------------------------------------------------------

async function initPrompts() {
  $("#prompts-auth").addEventListener("submit", (e) => {
    e.preventDefault();
    submitAuth();
  });
  $("#cand-submit").addEventListener("click", submitCandidate);
  $("#cand-cancel").addEventListener("click", () => $("#candidate-modal").close());
  $("#confirm-cancel").addEventListener("click", () => $("#confirm-modal").close());
  $("#confirm-ok").addEventListener("click", async () => {
    $("#confirm-modal").close();
    if (confirmAction) await confirmAction();
  });

  try {
    const status = await api("/api/v1/auth/status");
    authRequired = !!status.auth_required;
  } catch {
    authRequired = false; // status endpoint absent — assume open deployment
  }
  await refreshPrompts(false);
}

export { initPrompts };
