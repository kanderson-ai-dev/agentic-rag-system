// Thin fetch wrapper for the backend API.
//
// The app has no login flow — it is open by design — so requests never carry
// an Authorization header. Every request is tagged with the per-session id
// so the dashboard can scope aggregates to "this visit".

import { getSessionId } from "./util.js";

// Error carrying the HTTP status, the Retry-After header (when present) and
// the raw `detail` body — callers use `status === 429` to render friendly
// rate-limit/budget copy, `retryAfter` to decide whether a Retry button is
// meaningful at all (a budget 429 resets at UTC midnight), and `detail` for
// structured payloads like the promotion gate's `failures` list.
export class ApiError extends Error {
  constructor(message, status, retryAfter, detail) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.retryAfter = retryAfter;
    this.detail = detail;
  }
}

// FastAPI `detail` has three shapes: a plain string, an object (e.g.
// `{message, failures}` on a gated promotion), or a 422 validation array of
// `{loc, msg, type}` items. Reduce each to a human string — passing the raw
// value to `Error` would render "[object Object]".
function detailMessage(detail, status) {
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => item?.msg).filter(Boolean);
    if (messages.length) return messages.join("; ");
  }
  if (detail && typeof detail === "object" && typeof detail.message === "string") {
    return detail.message;
  }
  return `HTTP ${status}`;
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
    const detail = body.detail;
    throw new ApiError(
      detailMessage(detail, response.status),
      response.status,
      response.headers.get("Retry-After"),
      detail
    );
  }
  return response.json();
}
