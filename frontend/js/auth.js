// Authentication gating and login.
//
// Asks the backend whether login is required before deciding what to show.
// Without this check, a deployment that leaves JWT_SECRET_KEY/AUTH_USERNAME/
// AUTH_PASSWORD_HASH unset (open quickstart mode) would show a sign-in form
// that can never succeed, hiding the chat/dashboard behind it.

import { $ } from "./util.js";
import { api, setAccessToken } from "./api.js";
import { showToast } from "./toast.js";

// The header badge has three mutually-exclusive states, rendered distinctly:
//   * "signed out"   (default) — auth is enabled and no valid token is held.
//   * "signed in"    — auth is enabled and a token was issued this session.
//   * "auth disabled" — auth is not configured (open quickstart mode).
function setAuthStatus(text, state) {
  const el = $("#auth-status");
  el.textContent = text;
  el.classList.remove("signed-in", "auth-disabled");
  if (state) el.classList.add(state);
}

// Injected side-effect: reveal the app panels + refresh the dashboard once
// access is established, whether via login or open quickstart mode.
let onAuthenticated = null;

export function wireAuth({ onReady }) {
  onAuthenticated = onReady;
}

async function initAuthGate() {
  try {
    const { auth_required: authRequired } = await api("/api/v1/auth/status");
    if (!authRequired) {
      setAuthStatus("auth disabled", "auth-disabled");
      $("#login-panel").hidden = true;
      $("#chat-panel").hidden = false;
      $("#dashboard-panel").hidden = false;
      onAuthenticated?.();
    }
  } catch {
    // If the check itself fails, fall back to the login panel (safe default).
  }
}

function initLogin() {
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
      setAccessToken(result.access_token);
      setAuthStatus("signed in", "signed-in");
      showToast("Signed in successfully.", "success");
      $("#login-panel").hidden = true;
      $("#chat-panel").hidden = false;
      $("#dashboard-panel").hidden = false;
      onAuthenticated?.();
    } catch (error) {
      $("#login-error").textContent = error.message;
      $("#login-error").hidden = false;
    }
  });
}

export { setAuthStatus, initAuthGate, initLogin };
