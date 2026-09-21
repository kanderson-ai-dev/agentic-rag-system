// Frontend application entrypoint. Orchestrates the ES modules, wires their
// cross-cutting dependencies (no circular imports, no globals), and kicks off
// the auth gate. This is the only module with side effects at import time —
// every other module exposes explicit `init*`/`wire*` functions.

import { initTheme } from "./theme.js";
import { initGlobalErrorHandling } from "./toast.js";
import { initChat, wireChat } from "./chat.js";
import { initReview, wireReview, openReviewModal } from "./review.js";
import { initLogin, initAuthGate, wireAuth } from "./auth.js";
import { initDashboard, refreshDashboard } from "./dashboard.js";

function init() {
  initGlobalErrorHandling();
  initTheme();

  // Wire the cross-module callbacks before anything can trigger them.
  wireChat({ openReview: openReviewModal, refreshDashboard });
  wireReview({ refreshDashboard });
  wireAuth({ onReady: refreshDashboard });

  initChat();
  initReview();
  initLogin();
  initDashboard();

  initAuthGate();
}

init();
