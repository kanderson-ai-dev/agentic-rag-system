// Apply the persisted theme before first paint to avoid a flash of the
// wrong color scheme (FOUC). Loaded as a classic (blocking) external script
// from <head> so it runs before the stylesheet paints; intentionally not an
// ES module — module scripts are deferred, which would defeat the purpose.
(function () {
  var stored = null;
  try {
    stored = localStorage.getItem("theme");
  } catch (_) {
    stored = null;
  }
  var prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  var theme = stored === "light" || stored === "dark"
    ? stored
    : prefersDark
      ? "dark"
      : "light";
  document.documentElement.setAttribute("data-theme", theme);
})();
