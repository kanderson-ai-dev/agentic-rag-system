// Shared, dependency-free helpers used across the frontend modules.
// Everything here is pure and side-effect free so it can be imported by any
// module without creating implicit global state or unexpected coupling.

/** Query a single element by CSS selector (throws if absent). */
export const $ = (sel) => document.querySelector(sel);

/** Query all matching elements. */
export const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/**
 * Create an element, optionally with its attributes set. A tiny helper that
 * keeps the DOM-construction code readable without pulling in a framework.
 */
export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "className") {
      node.className = value;
    } else if (key === "textContent") {
      node.textContent = value;
    } else {
      node.setAttribute(key, value);
    }
  }
  for (const child of children) {
    node.appendChild(child);
  }
  return node;
}

/**
 * A random id per browser tab (sessionStorage: cleared when the tab closes,
 * unique per new tab/incognito window). Sent as `X-Session-Id` so the
 * dashboard can scope its aggregates to "this visit" instead of the service's
 * lifetime totals — a first-time viewer always sees zeros.
 */
export function getSessionId() {
  let sessionId = null;
  try {
    sessionId = sessionStorage.getItem("session_id");
  } catch {
    // Storage may be unavailable; fall back to an ephemeral per-page id.
  }
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    try {
      sessionStorage.setItem("session_id", sessionId);
    } catch {
      // Non-fatal: the id just won't persist across reloads for this tab.
    }
  }
  return sessionId;
}

/**
 * Debounce a function so it only runs after `waitMs` of inactivity.
 * Used for high-frequency inputs (resize, scroll, resize-driven chart
 * redraws) to avoid thrashing the layout on every event.
 */
export function debounce(fn, waitMs = 150) {
  let timer = null;
  return (...args) => {
    if (timer !== null) window.clearTimeout(timer);
    timer = window.setTimeout(() => {
      timer = null;
      fn(...args);
    }, waitMs);
  };
}
