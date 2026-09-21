// Thin fetch wrapper for the backend API.
//
// The app has no login flow — it is open by design — so requests never carry
// an Authorization header. Every request is tagged with the per-session id
// so the dashboard can scope aggregates to "this visit".

import { getSessionId } from "./util.js";

export async function api(path, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    "X-Session-Id": getSessionId(),
    ...(options.headers || {}),
  };
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json();
}
