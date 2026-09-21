// Thin authenticated fetch wrapper for the backend API.
//
// The JWT lives in module scope (in memory only, never localStorage) to reduce
// the surface for token theft via XSS. It is exposed through explicit
// getter/setter functions rather than a mutable module-level variable that any
// importer could overwrite.

import { getSessionId } from "./util.js";

let accessToken = null;

export function setAccessToken(token) {
  accessToken = token;
}

export function getAccessToken() {
  return accessToken;
}

export async function api(path, options = {}) {
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
