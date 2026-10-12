// Thin fetch wrapper for the backend API.
//
// The app has no login flow — it is open by design — so requests never carry
// an Authorization header. Every request is tagged with the per-session id
// so the dashboard can scope aggregates to "this visit".

import { getSessionId } from "./util.js";

// Error carrying the HTTP status and the Retry-After header (when present).
// Callers use `status === 429` to render friendly rate-limit/budget copy and
// `retryAfter` to decide whether a Retry button is meaningful at all — a
// budget 429 resets at UTC midnight, so retrying now is pointless.
export class ApiError extends Error {
  constructor(message, status, retryAfter) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

export async function api(path, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    "X-Session-Id": getSessionId(),
    ...(options.headers || {}),
  };
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(
      body.detail || `HTTP ${response.status}`,
      response.status,
      response.headers.get("Retry-After")
    );
  }
  return response.json();
}
