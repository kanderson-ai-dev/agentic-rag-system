// Theme management: applies the persisted/OS preference before first paint
// (see the inline bootstrap in index.html) and toggles between light/dark.

import { $ } from "/js/util.js";

const STORAGE_KEY = "theme";

export function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "light"
    ? "light"
    : "dark";
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem("theme", theme);
  } catch {
    // Storage may be unavailable (private mode, blocked cookies); the theme
    // still applies for this page view, it just won't persist.
  }
  const label =
    theme === "light" ? "Switch to dark theme" : "Switch to light theme";
  $("#theme-toggle").setAttribute("aria-label", label);
  $("#theme-toggle").setAttribute("title", label);
}

/**
 * Wire the toggle button and sync its accessible label with the theme that was
 * already applied by the inline bootstrap in <head>.
 */
export function initTheme() {
  $("#theme-toggle").addEventListener("click", () => {
    setTheme(currentTheme() === "light" ? "dark" : "light");
  });
  setTheme(currentTheme());
}
