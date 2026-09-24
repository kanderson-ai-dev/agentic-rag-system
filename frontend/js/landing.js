// Public landing page: one input, one button, one grounded answer.
//
// Minimalist surface — no dashboard, no review modal. It reuses the shared
// modules (`api`, `markdown`, `util`) and mirrors the console's composer
// pattern (Enter to send, Shift+Enter for a newline, adaptive height). When a
// query escalates to human review, the landing points to `/console` instead of
// exposing the operator-only review flow here.

import { $ } from "./util.js";
import { api } from "./api.js";
import { renderMarkdown } from "./markdown.js";

const MAX_TEXTAREA_HEIGHT = 160; // px — same cap as the console composer

// Busy state is module-scoped; the submit handler reads it so a double-submit
// can never fire two in-flight queries.
let busy = false;

function setBusy(next) {
  busy = next;
  const button = $("#ask-button");
  const textarea = $("#question");
  button.disabled = next;
  button.textContent = next ? "Asking…" : "Ask";
  textarea.disabled = next;
}

// --- Answer region ---------------------------------------------------------
// `#answer` holds the rendered content; `#answer-meta` holds secondary info
// (the echoed question, source chips, escalation link). Both are cleared
// together on every state transition.

function clearAnswer() {
  $("#answer").replaceChildren();
  $("#answer-meta").replaceChildren();
}

// Lightweight skeleton shown while the query is in flight. The surrounding
// <section> has a fixed min-height, so this never shifts the layout.
function showSkeleton() {
  clearAnswer();
  const skeleton = document.createElement("div");
  skeleton.className = "space-y-3";
  skeleton.setAttribute("role", "status");
  skeleton.setAttribute("aria-label", "Generating an answer");
  for (const width of ["w-full", "w-11/12", "w-2/3"]) {
    const line = document.createElement("div");
    line.className = `h-3 ${width} animate-pulse rounded-full bg-neutral-800`;
    skeleton.appendChild(line);
  }
  $("#answer").appendChild(skeleton);
}

function echoQuestion(question) {
  const el = document.createElement("p");
  el.className = "text-xs font-medium uppercase tracking-widest text-neutral-500";
  el.textContent = question;
  $("#answer-meta").appendChild(el);
}

function renderSources(sources) {
  if (!sources || !sources.length) return;
  const row = document.createElement("div");
  row.className = "mt-3 flex flex-wrap gap-2";
  sources.slice(0, 4).forEach((source) => {
    const chip = document.createElement("span");
    chip.className =
      "rounded-full border border-neutral-800 bg-neutral-900 px-2.5 py-1 font-mono text-xs text-neutral-400";
    chip.textContent = source;
    row.appendChild(chip);
  });
  $("#answer-meta").appendChild(row);
}

// The correction loop escalated to a human reviewer. The full approve / retry /
// override flow is an operator feature, so the landing links to the console
// instead of failing silently.
function renderEscalation() {
  const box = document.createElement("div");
  box.className =
    "rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-sm leading-relaxed text-amber-200/90";
  box.textContent =
    "This question needs a second look — the automated checks could not resolve it confidently.";

  const link = document.createElement("a");
  link.href = "/console";
  link.className =
    "mt-2 inline-block font-medium text-amber-300 underline decoration-amber-500/50 underline-offset-4 transition-colors hover:text-amber-200";
  link.textContent = "Open the operator console for the full review flow →";
  box.appendChild(link);

  $("#answer").appendChild(box);
}

function renderError(message, retry) {
  const box = document.createElement("div");
  box.className =
    "rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-200/90";
  box.setAttribute("role", "alert");
  box.textContent = message;

  const retryBtn = document.createElement("button");
  retryBtn.type = "button";
  retryBtn.className =
    "ml-3 inline-block rounded-lg border border-red-500/40 px-3 py-1 text-xs font-medium text-red-200 transition-colors hover:bg-red-500/20";
  retryBtn.textContent = "Retry";
  retryBtn.addEventListener("click", () => {
    clearAnswer();
    retry();
  });
  box.appendChild(retryBtn);

  $("#answer").appendChild(box);
}

async function ask(question) {
  if (busy) return;
  setBusy(true);
  showSkeleton();
  try {
    const result = await api("/api/v1/rag/query", {
      method: "POST",
      body: JSON.stringify({ question }),
    });
    clearAnswer();
    echoQuestion(question);

    if (result.status === "interrupted") {
      renderEscalation();
    } else {
      $("#answer").appendChild(renderMarkdown(result.answer || "(no answer)"));
      renderSources(result.sources);
    }
  } catch (error) {
    clearAnswer();
    renderError(`Something went wrong: ${error.message}`, () => ask(question));
  } finally {
    setBusy(false);
  }
}

function init() {
  const textarea = $("#question");
  const form = $("#ask-form");

  // Grow the textarea up to a max height as content wraps.
  textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, MAX_TEXTAREA_HEIGHT)}px`;
  });

  // The question field is a <textarea> (so Shift+Enter can add a newline),
  // but browsers only auto-submit <input>s on Enter. Submit explicitly on
  // plain Enter — the same listener drives both the Ask button and the key.
  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = textarea.value.trim();
    if (!question) return;
    textarea.value = "";
    textarea.style.height = "auto";
    ask(question);
  });
}

init();
