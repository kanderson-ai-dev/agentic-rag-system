// Human-in-the-loop review modal (Phase 5). Fully keyboard-operable and
// accessible:
//   * A native `<dialog>` traps focus and closes on Esc automatically.
//   * The three decisions (approve / retry / override) are radio inputs with a
//     clear explanation each, selectable via arrow keys / click.
//   * Retry and override reveal a labeled input that is validated before
//     submission — empty input blocks submission with a visible error.
//   * Submitting shows a busy state on the confirm button and disables the
//     controls; failures surface inside the modal (with retry).

import { $ } from "./util.js";
import { api } from "./api.js";
import { addAssistantMessage, getThreadId } from "./chat.js";
import { announce, showToast } from "./toast.js";

const REVIEW_INPUT_LABELS = {
  retry: "Revised question",
  override: "Manual answer",
};
const REVIEW_INPUT_PLACEHOLDERS = {
  retry: "Enter a clearer question to re-run retrieval…",
  override: "Write the correct answer…",
};

// Module-scoped state (no globals).
let reviewMode = null;
let lastFocusedElement = null; // element focused before the modal opened

// Injected so review can refresh the dashboard after a resolved decision
// without importing the dashboard module directly (avoids a cycle).
let onDashboardRefresh = null;

export function wireReview({ refreshDashboard }) {
  onDashboardRefresh = refreshDashboard;
}

function selectedReviewDecision() {
  const checked = $('input[name="review-decision"]:checked') || null;
  return checked ? checked.value : null;
}

function setReviewMode(mode) {
  reviewMode = mode;
  const inputRow = $("#review-input-row");
  const input = $("#review-input");
  const label = $("#review-input-label");
  if (mode === "retry" || mode === "override") {
    label.textContent = REVIEW_INPUT_LABELS[mode];
    input.placeholder = REVIEW_INPUT_PLACEHOLDERS[mode];
    input.value = "";
    inputRow.hidden = false;
    input.focus();
  } else {
    input.value = "";
    inputRow.hidden = true;
  }
  clearReviewErrors();
}

function clearReviewErrors() {
  $("#review-input-error").hidden = true;
  $("#review-input-error").textContent = "";
  $("#review-submit-error").hidden = true;
  $("#review-submit-error").textContent = "";
}

function setReviewBusy(busy) {
  const submit = $("#review-submit");
  const cancel = $("#review-cancel");
  submit.disabled = busy;
  cancel.disabled = busy;
  submit.textContent = busy ? "Submitting…" : "Confirm";
  document
    .querySelectorAll('input[name="review-decision"]')
    .forEach((radio) => {
      radio.disabled = busy;
    });
  $("#review-input").disabled = busy;
}

function showReviewError(message) {
  const submitError = $("#review-submit-error");
  submitError.textContent = message;
  submitError.hidden = false;
}

function openReviewModal(interrupt) {
  // Reset radio selection and mode on each open.
  document
    .querySelectorAll('input[name="review-decision"]')
    .forEach((radio) => {
      radio.checked = false;
    });
  setReviewMode(null);

  // Remember what had focus so we can return to it when the dialog closes.
  lastFocusedElement = document.activeElement;

  $("#review-question").textContent = interrupt.question || "(no question)";
  const docs = (interrupt.best_documents || []).slice(0, 2).join("\n\n");
  $("#review-docs").textContent =
    docs || "(no retrieved documents available)";

  const modal = $("#review-modal");
  modal.showModal();
  // Move focus to the first decision for immediate keyboard use.
  $("#review-approve").focus();

  // Restore focus to the triggering control when the dialog closes (e.g. Esc
  // or Cancel); native <dialog> returns focus to the invoker, but we opened
  // programmatically so we restore explicitly for a predictable keyboard flow.
  modal.addEventListener(
    "close",
    () => {
      if (lastFocusedElement && lastFocusedElement.isConnected) {
        lastFocusedElement.focus();
      }
    },
    { once: true }
  );
}

async function submitReview(decision) {
  if (!decision) {
    showReviewError("Please choose one of the three options before confirming.");
    return;
  }

  const inputValue = $("#review-input").value.trim();
  if ((decision === "retry" || decision === "override") && !inputValue) {
    const inputError = $("#review-input-error");
    inputError.textContent = `Please enter a ${
      decision === "retry" ? "revised question" : "manual answer"
    }.`;
    inputError.hidden = false;
    $("#review-input").focus();
    return;
  }

  const payload = { decision };
  if (decision === "retry") payload.revised_question = inputValue;
  if (decision === "override") payload.override_answer = inputValue;

  setReviewBusy(true);
  clearReviewErrors();
  try {
    const result = await api(`/api/v1/rag/query/${getThreadId()}/review`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    $("#review-modal").close();
    showToast("Review submitted.", "success");
    if (result.status === "interrupted") {
      openReviewModal(result.interrupt);
    } else {
      addAssistantMessage(result.answer || "(no answer)", result.sources || []);
      announce("Answer received.");
      onDashboardRefresh?.();
    }
  } catch (error) {
    // Keep the modal open so the user can correct and retry in place.
    showReviewError(`Something went wrong: ${error.message}`);
  } finally {
    setReviewBusy(false);
  }
}

function initReview() {
  // Persist the chosen mode when the user picks a decision.
  document
    .querySelectorAll('input[name="review-decision"]')
    .forEach((radio) => {
      radio.addEventListener("change", () => setReviewMode(radio.value));
    });

  // Clear the input error as soon as the user starts typing again.
  $("#review-input").addEventListener("input", () => {
    $("#review-input-error").hidden = true;
  });

  $("#review-cancel").addEventListener("click", () => {
    $("#review-modal").close();
  });

  $("#review-submit").addEventListener("click", () => {
    submitReview(selectedReviewDecision() || reviewMode);
  });
}

export { initReview, openReviewModal, submitReview };
