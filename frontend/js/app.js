// Frontend application entrypoint. Orchestrates the ES modules and wires
// their cross-cutting dependencies (no circular imports, no globals). This
// is the only module with side effects at import time — every other module
// exposes explicit `init*`/`wire*` functions. There is no login gate: the
// app is open by design, so chat and dashboard are available immediately.

import { initTheme } from "./theme.js";
import { initGlobalErrorHandling } from "./toast.js";
import { initChat, wireChat } from "./chat.js";
import { initReview, wireReview, openReviewModal } from "./review.js";
import { initDashboard, refreshDashboard } from "./dashboard.js";

function init() {
  initGlobalErrorHandling();
  initTheme();

  // Wire the cross-module callbacks before anything can trigger them.
  wireChat({ openReview: openReviewModal, refreshDashboard });
  wireReview({ refreshDashboard });

  initChat();
  initReview();
  initDashboard();

  refreshDashboard();
}

init();
