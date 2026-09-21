// Transient notifications (toasts) + live announcements + global error
// handling. Both toasts and the announcer are populated with plain text only
// (never HTML), so nothing dynamic can inject markup.

import { $ } from "./util.js";

/**
 * Announce a message to screen readers via a visually-hidden polite live
 * region, without moving visual focus.
 */
function announce(message) {
  const region = $("#chat-announcer");
  if (!region) return;
  // Clear then set so repeated identical announcements are still spoken.
  region.textContent = "";
  region.textContent = message;
}

function removeToast(toast) {
  toast.classList.add("leaving");
  window.setTimeout(() => toast.remove(), 200);
}

function showToast(message, type = "info", duration = 4000) {
  const container = $("#toasts");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.setAttribute("role", "status");

  const text = document.createElement("span");
  text.className = "toast-text";
  text.textContent = message;

  const dismiss = document.createElement("button");
  dismiss.type = "button";
  dismiss.className = "toast-dismiss";
  dismiss.setAttribute("aria-label", "Dismiss notification");
  dismiss.textContent = "×";
  dismiss.addEventListener("click", () => removeToast(toast));

  toast.append(text, dismiss);
  container.appendChild(toast);

  window.setTimeout(() => removeToast(toast), duration);
}

/**
 * Route unexpected runtime errors and unhandled promise rejections to a
 * visible toast instead of a silent failure or a bare console stack.
 */
function initGlobalErrorHandling() {
  window.addEventListener("error", (event) => {
    showToast(`Unexpected error: ${event.message}`, "error");
  });

  window.addEventListener("unhandledrejection", (event) => {
    const reason =
      event.reason instanceof Error ? event.reason.message : String(event.reason);
    showToast(`Request failed: ${reason}`, "error");
  });
}

export { announce, showToast, removeToast, initGlobalErrorHandling };
