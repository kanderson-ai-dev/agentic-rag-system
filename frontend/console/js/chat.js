// Chat experience: messages, Markdown answers, copy, typing indicator,
// adaptive composer, and query submission. State is module-scoped and shared
// through explicit getters/setters rather than globals.

import { $ } from "/js/util.js";
import { api } from "/js/api.js";
import { renderMarkdown } from "/js/markdown.js";
import { announce } from "./toast.js";

// The current thread id, set after a query; needed by the review flow to
// resume the right checkpoint. Module-scoped, exposed via accessors so it is
// shared without leaking onto `window`.
let currentThreadId = null;

export function getThreadId() {
  return currentThreadId;
}

// Injected side-effects (review modal open + dashboard refresh) avoid a
// circular import between chat, review, and dashboard. Set during init.
let onInterrupted = null;
let onDashboardRefresh = null;

export function wireChat({ openReview, refreshDashboard }) {
  onInterrupted = openReview;
  onDashboardRefresh = refreshDashboard;
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
    sources.slice(0, 4).forEach((source) => {
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
    // Clipboard API may be unavailable (insecure context); fall back to a
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
  announce("Answer copied to clipboard.");
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
function resetComposerHeight() {
  const textarea = $("#question");
  textarea.style.height = "auto";
}

// Busy state is kept module-internal; the composer toggles it and the submit
// handler reads it, so no `window` global is needed.
let composerBusy = false;

function setBusy(busy) {
  composerBusy = busy;
  const sendBtn = $("#chat-send");
  const textarea = $("#question");
  sendBtn.disabled = busy;
  sendBtn.textContent = busy ? "Sending…" : "Send";
  textarea.disabled = busy;
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

// Send a question to the backend, driving the typing indicator and handling
// the completed / interrupted / error outcome.
async function sendQuestion(question) {
  if (composerBusy) return;
  setBusy(true);
  showTyping();
  try {
    const result = await api("/api/v1/rag/query", {
      method: "POST",
      body: JSON.stringify({ question }),
    });
    hideTyping();
    currentThreadId = result.thread_id;

    if (result.status === "interrupted") {
      announce("A human review is required to continue.");
      onInterrupted?.(result.interrupt);
    } else if (result.blocked) {
      announce("Request blocked by the input guardrail.");
      addMessage(
        result.answer || "This request was blocked by the safety guardrail.",
        "system"
      );
      onDashboardRefresh?.();
    } else {
      addAssistantMessage(result.answer || "(no answer)", result.sources || []);
      announce("Answer received.");
      onDashboardRefresh?.();
    }
  } catch (error) {
    hideTyping();
    addErrorWithRetry(`Something went wrong: ${error.message}`, () =>
      sendQuestion(question)
    );
  } finally {
    setBusy(false);
  }
}

// Wire the composer: adaptive height, Enter-to-send, and form submission.
function initChat() {
  const textarea = $("#question");
  const chatForm = $("#chat-form");

  // Grow the textarea up to a max height as content wraps.
  textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
  });

  // The question field is a <textarea> (so Shift+Enter can add a newline),
  // but browsers only auto-submit <input>s on Enter. Submit explicitly on
  // plain Enter, keeping Shift+Enter for multi-line questions.
  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      chatForm.requestSubmit();
    }
  });

  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = textarea.value.trim();
    if (!question) return;
    addMessage(question, "user");
    textarea.value = "";
    resetComposerHeight();
    await sendQuestion(question);
  });
}

export { addAssistantMessage, initChat, sendQuestion };
