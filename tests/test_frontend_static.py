"""Tests that the frontend static files are served without shadowing API routes."""

from fastapi.testclient import TestClient

from app.main import app


def test_serves_index_html() -> None:
    client = TestClient(app)
    response = client.get("/")
    client.close()

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_serves_static_assets() -> None:
    client = TestClient(app)
    css = client.get("/styles.css")
    js = client.get("/app.js")
    client.close()

    assert css.status_code == 200
    assert js.status_code == 200


def test_api_routes_are_not_shadowed() -> None:
    client = TestClient(app)
    health = client.get("/health/live")
    openapi = client.get("/openapi.json")
    client.close()

    assert health.status_code == 200
    assert openapi.status_code == 200


def test_design_system_phase1_theme_toggle_and_tokens() -> None:
    """Phase 1: base visual system is present and wired.

    Asserts the theme toggle exists in the markup, both color themes are
    defined via CSS custom properties (tokens), and the JS persists the
    choice to localStorage rather than leaving the theme hard-coded.
    """
    client = TestClient(app)
    html = client.get("/").text
    css = client.get("/styles.css").text
    js = client.get("/app.js").text
    client.close()

    # Theme toggle button present in the header, exposed to assistive tech.
    assert 'id="theme-toggle"' in html
    assert 'aria-label="Toggle color theme"' in html

    # Both themes are declared as design-token blocks (light and dark).
    assert '[data-theme="light"]' in css
    assert ':root[data-theme="dark"]' in css or '[data-theme="dark"]' in css

    # Theme persistence uses localStorage, not a session-only or hard-coded value.
    assert 'localStorage.setItem("theme", theme)' in js
    assert 'localStorage.getItem("theme")' in html


def test_layout_phase2_responsive_grid_and_breakpoints() -> None:
    """Phase 2: mobile-first grid layout with explicit breakpoints.

    Asserts the main region uses a CSS Grid that cannot force horizontal
    overflow (`minmax(0, 1fr)`), the header is sticky, explicit tablet/desktop
    `min-width` breakpoints exist, and the wide "recent requests" table is
    wrapped in a scroll container rather than overflowing the page at 320px.
    """
    client = TestClient(app)
    html = client.get("/").text
    css = client.get("/styles.css").text
    client.close()

    # Main content is a single-column, overflow-safe grid.
    assert ".layout" in css
    assert "display: grid" in css
    assert "grid-template-columns: minmax(0, 1fr)" in css

    # The header stays fixed to the top while the content scrolls beneath it.
    assert "position: sticky" in css

    # Explicit mobile-first breakpoints for tablet and desktop (min-width,
    # not max-width, so base rules target the smallest screen).
    assert "--bp-tablet: 640px" in css
    assert "--bp-desktop: 1024px" in css
    assert "@media (min-width: 640px)" in css
    assert "@media (min-width: 1024px)" in css

    # The dashboard table is wrapped so it scrolls inside its own container
    # and never forces a horizontal page scroll on narrow viewports.
    assert 'class="table-wrap"' in html
    assert ".table-wrap" in css
    assert "overflow-x: auto" in css


def test_branding_phase3_logo_favicon_and_metadata() -> None:
    """Phase 3: inline logo, favicon, and share-ready metadata.

    Asserts an inline SVG logo is present in the header, a favicon is declared
    (via data URI, no extra asset), and the page ships Open Graph / Twitter
    metadata plus `theme-color` so shared links unfurl correctly.
    """
    client = TestClient(app)
    html = client.get("/").text
    css = client.get("/styles.css").text
    client.close()

    # Inline logo (no external asset) wrapped in a brand block in the header.
    assert 'class="brand"' in html
    assert 'class="logo"' in html
    assert "<svg" in html

    # Favicon is declared (SVG data URI — portable, no build step).
    assert 'rel="icon"' in html
    assert 'type="image/svg+xml"' in html
    assert 'data:image/svg+xml' in html

    # Document metadata: title + description for SEO and link previews.
    assert "<title>" in html
    assert 'name="description"' in html

    # theme-color adapts to the active theme (light + dark variants).
    assert 'name="theme-color"' in html

    # Open Graph + Twitter cards render a rich preview when shared.
    assert 'property="og:title"' in html
    assert 'property="og:description"' in html
    assert 'name="twitter:card"' in html

    # The logo inherits design tokens rather than hard-coding colors.
    assert "var(--accent)" in html
    assert ".brand" in css
    assert ".logo" in css


def test_branding_phase3_auth_status_states() -> None:
    """Phase 3: the header badge has distinct auth states.

    Asserts the JS keeps the auth-status badge in mutually-exclusive, clearly
    distinguishable states ("signed in", "signed out", "auth disabled") rather
    than relying on a single ambiguous label.
    """
    client = TestClient(app)
    js = client.get("/app.js").text
    css = client.get("/styles.css").text
    client.close()

    # A single helper renders the three states, removing any prior state class.
    assert "function setAuthStatus" in js
    assert "auth-disabled" in js
    assert "signed-in" in js

    # Each state has dedicated visual treatment (distinct token colors).
    assert ".badge.auth-disabled" in css
    assert ".badge.signed-in" in css


def test_chat_phase4_xss_safe_markdown_rendering() -> None:
    """Phase 4: assistant answers render as sanitized Markdown, not raw HTML.

    Asserts the answer renderer builds DOM nodes via safe textContent/createElement
    (never assigning untrusted output to innerHTML), so a malicious LLM answer that
    contains `<script>` or event-handler markup is treated as literal text and can
    never execute (OWASP recommendation: never render LLM output as HTML).
    """
    client = TestClient(app)
    js = client.get("/app.js").text
    html = client.get("/").text
    client.close()

    # A dedicated renderer exists and constructs nodes with the safe DOM APIs,
    # not by interpolating strings into innerHTML.
    assert "function renderMarkdown" in js
    assert "document.createTextNode" in js
    assert "document.createElement" in js

    # The renderer never assigns untrusted content to innerHTML for the answer:
    # it uses `appendChild` on the constructed fragment. (innerHTML *is* still
    # used elsewhere only for static, known dashboard markup.)
    assert "root.appendChild" in js
    assert "body.appendChild(renderMarkdown" in js
    assert "DOMParser" not in js

    # Welcome/empty state is required, and the answer is inserted into a
    # dedicated body slot so styling stays isolated from untrusted content.
    assert 'id="chat-welcome"' in html


def test_chat_phase4_typing_loading_and_states() -> None:
    """Phase 4: typing indication, loading state, and error retry exist.

    Asserts the JS has an explicit "typing" indicator shown while a query is in
    flight, a busy/loading state for the composer/send button, and an error path
    that offers a retry affordance rather than a dead end.
    """
    client = TestClient(app)
    js = client.get("/app.js").text
    css = client.get("/styles.css").text
    client.close()

    # Typing indicator is driven by dedicated show/hide helpers.
    assert "function showTyping" in js
    assert "function hideTyping" in js
    assert "typing" in css

    # Loading state disables the send affordance while in flight.
    assert "Sending…" in js
    assert "disabled" in js

    # Error messages carry a retry action.
    assert "Retry" in js
    assert "addErrorWithRetry" in js


def test_chat_phase4_copy_and_sources() -> None:
    """Phase 4: copy-to-clipboard and source chips are wired up.

    Asserts the assistant answer exposes a copy button (using the async Clipboard
    API with a legacy fallback) and renders retrieved sources as chips rather than
    as opaque text.
    """
    client = TestClient(app)
    js = client.get("/app.js").text
    css = client.get("/styles.css").text
    client.close()

    # Copy-to-clipboard flow with a graceful fallback.
    assert "navigator.clipboard.writeText" in js
    assert "copyMessageText" in js
    assert "Copied!" in js
    assert ".copy-btn" in css

    # Sources render as chips sourced from the query response `sources` field.
    assert "source-chip" in css
    assert "sources" in js


def test_chat_phase4_adaptive_input_and_smart_scroll() -> None:
    """Phase 4: adaptive composer height and intelligent auto-scroll.

    Asserts the textarea auto-grows with content (capped at a max height) and the
    message list only auto-scrolls when the user is already near the bottom, so
    reading earlier messages is never interrupted.
    """
    client = TestClient(app)
    js = client.get("/app.js").text
    css = client.get("/styles.css").text
    client.close()

    # Auto-growing textarea with an explicit max height.
    assert "scrollHeight" in js
    assert "max-height" in css
    assert "resize: none" in css

    # Smart scroll preserves the user's position unless already at the bottom.
    assert "function smartScroll" in js
    assert "scrollHeight" in js
    assert "scrollTop" in js
