# Frontend

Plain HTML/CSS/JS — no framework, no build step, no external runtime
dependencies beyond the Tailwind Play CDN on the landing. Served by FastAPI via
`StaticFiles` mounts in `app/main.py`.

## Two surfaces

| Route | Surface | Directory |
|---|---|---|
| `/` | **Public landing** — one input, one button, one grounded answer. Fixed dark theme, Tailwind via CDN. | `frontend/landing/` |
| `/console` | **Operator console** — full chat, metrics dashboard, and the HITL review modal. Token-based `styles.css`, light/dark themes. | `frontend/console/` |
| `/js/*` | **Shared ES modules** imported by both surfaces (`util`, `api`, `markdown`) plus the landing entrypoint (`landing.js`). | `frontend/js/` |

The split is a product decision: `/` is the 10-second first impression (ask a
question, get a grounded answer — no login, no dashboard), while `/console`
keeps the complete operator tooling for anyone who wants to dig deeper. When a
landing query escalates to human review, the page shows an honest notice with a
link to `/console` instead of exposing the operator-only review flow.

Console modules import the shared modules by absolute path (`/js/util.js`), so
each surface's directory stays self-contained under its mount.

## Structure

| File | Responsibility |
|---|---|
| `landing/index.html` | Minimalist landing markup + metadata (SEO/OG/Twitter). Tailwind Play CDN (`https://cdn.tailwindcss.com`) — no `tailwind.config.js`, no build. |
| `js/landing.js` | Landing logic: adaptive textarea, Enter-to-send, busy/skeleton/answer/error states, HITL escalation link to `/console`. |
| `js/util.js` | Pure helpers: `$`, `$$`, `el`, `getSessionId`, `debounce`. |
| `js/api.js` | `fetch` wrapper + `X-Session-Id`. No login flow: the UI never sends credentials. |
| `js/markdown.js` | **Sanitized** Markdown renderer (no XSS), builds DOM with `textContent`/`createElement`. |
| `console/index.html` | Console markup + metadata, inline SVG logo/favicon, anti-FOUC theme bootstrap script. |
| `console/styles.css` | Console design system: tokens, reset, components, light/dark themes, responsive layout. |
| `console/js/app.js` | Console **entrypoint** (ES module). Orchestrates modules and wires cross-dependencies. Only module with import-time side effects. No auth gate: chat and dashboard are available immediately. |
| `console/js/theme.js` | Light/dark theme toggle persisted to `localStorage`. |
| `console/js/toast.js` | Transient notifications (`showToast`), `aria-live` announcements (`announce`), global error handling. |
| `console/js/chat.js` | Chat experience: messages, typing, copy, adaptive composer, query submission. |
| `console/js/review.js` | Human-review modal (HITL). |
| `console/js/dashboard.js` | Dashboard: metric cards, sortable/paginated table, SVG chart, Quality (EDD). |

## ES modules (no bundler)

Both surfaces use **native ES modules** (`<script type="module">`) — the browser
loads them with `import`/`export`, no build step:

- **No unnecessary global state**: `thread_id`, composer state, and modal state
  live in module scope (not `window`), shared via explicit accessors
  (`getThreadId`, …) or injected callbacks (`wireChat`, `wireReview`) — no
  circular imports.
- **Unidirectional dependencies**: `app.js` is the only module that knows the
  console's init graph; each module exposes an idempotent `init*`.
- **`debounce` on high-frequency inputs**: the `resize` handler that redraws the
  SVG chart is decoupled via `debounce` from `util.js`.
- **No layout shift**: vertical space is reserved (`min-height`) in metric
  cards, the chart region, the recent-requests table, and the landing answer
  area, so skeleton → data → empty transitions never move content.

## Branding and metadata

- **Logo and favicon**: inline SVG (no external binary assets, no build step).
  The favicon is served as a data URI (portable, no extra request); the console
  header logo uses `currentColor`/design-system tokens to adapt to the theme.
- **Metadata**: descriptive `<title>`, `<meta name="description">`,
  `theme-color` (light/dark `media` variants on the console), and Open Graph +
  Twitter Card so shared links unfurl richly.

## Design system (console)

Every visual value derives from **custom properties** (CSS variables); there are
no hard-coded "magic" values in components.

- **Palette** (`--bg`, `--surface*`, `--border*`, `--text*`, `--accent*`,
  `--success`, `--danger`, `--warning`).
- **Typography** (`--font-sans`, `--font-mono`, scale `--text-xs`…`--text-3xl`,
  `--leading-*`, `--weight-*`).
- **Spacing** base 4px (`--space-1`…`--space-7`).
- **Radii** (`--radius-sm/md/lg/full`), **shadows** (`--shadow-sm/md/lg`) and
  **transitions** (`--transition-fast`, `--transition`).

### Themes

Two themes defined as token blocks: `[data-theme="dark"]` (default) and
`[data-theme="light"]`. The header toggle:

1. Applies the theme saved in `localStorage` (`key = "theme"`) before first
   paint via an inline script in `<head>` (the only inline script — allowed by
   CSP through its SHA-256 hash).
2. Falls back to `prefers-color-scheme` when no preference is stored.
3. On click, flips `<html>`'s `data-theme` and persists to `localStorage`.

The landing does not theme-switch: it is a single fixed dark theme, styled with
Tailwind utilities.

### Base reusable components

- **Buttons**: `.btn` (primary) and `.secondary`, `.danger`, `.ghost` variants,
  with `:hover`, `:active`, `:disabled`, and visible focus states.
- **Inputs / textarea**: `:focus` accent ring.
- **Cards**: `.card`.
- **Badges**: `.badge`.
- **Tables**: `table.recent` (wrapped in `.table-wrap` for its own horizontal
  scroll on narrow screens, without breaking page scroll).
- **Stats**: `.stat` (+ `.quality-stat.pass/.fail`).

## Chat experience (console)

- **Sanitized Markdown (no XSS)**: LLM answers render through a custom renderer
  (`renderMarkdown` in `js/markdown.js`, shared with the landing) that builds
  DOM nodes only with `textContent`/`createElement` — untrusted output is
  **never** assigned to `innerHTML`. Raw HTML in an answer is always literal
  text (XSS neutralized by design). Supported subset: fenced and inline code,
  headings, ordered/unordered lists, blockquotes, paragraphs, and
  `**bold**` / `*italic*`.
- **"Typing…" indicator and loading states**: animated-dots bubble
  (`showTyping`/`hideTyping`) while a query is in flight; the *Send* button
  switches to "*Sending…*" and stays disabled until resolved.
- **Copy answer and sources**: each answer carries a *Copy* button (Clipboard
  API with `execCommand("copy")` fallback) and retrieved sources render as
  chips (`sources` field of `QueryResponse`).
- **States**: welcome (empty), loading (typing), and error with retry
  (`addErrorWithRetry`).
- **Smart auto-scroll** (only when already near the bottom) and an **adaptive
  textarea** that grows with content up to a max height — the same composer
  pattern the landing reuses.

## Human review flow — HITL (console)

When the Self-RAG correction loop exhausts its retries, the graph pauses
(`interrupt()`) and the console opens an **accessible modal** (`<dialog>`) so a
human can decide how to continue. The whole flow is keyboard-operable.

- **Accessible modal**: native `<dialog>` with `aria-labelledby` and
  `aria-describedby`, **trapped focus** and **Esc to close** (native element
  behavior). On open, focus moves to the first decision control.
- **Three explained options**: each decision is a radio input under a
  `<fieldset>` (arrow-key navigable) with a visible explanation:
  - **Approve** — accept the best available answer as-is.
  - **Retry** — re-run retrieval with a revised question you write.
  - **Override** — write the correct answer manually.
- **Input validation**: `retry`/`override` reveal a labelled field; submitting
  empty blocks the action and shows a visible error with focus on the field
  (cleared on typing). Choosing nothing asks for a choice.
- **Loading/error states**: on confirm, the button switches to "*Submitting…*"
  and controls disable; on failure the error shows **inside** the modal
  (`role="alert"`) and the modal stays open to correct and resubmit — it never
  closes nor dumps the error into the chat.

## Dashboard (console)

The `#dashboard-panel` shows real backend data (`/api/v1/dashboard/summary`,
`/recent`, `/quality`), all written with `textContent` (never `innerHTML` with
server data):

- **Metric cards**: total cost (`$`), average latency (`ms`), requests blocked
  by the guardrail and escalated to human review (`renderSummary`).
- **Quality (EDD)**: pass/fail indicators per RAGAS scorecard threshold
  (`renderQuality`), with an implicit "no scorecard yet" state.
- **Cost/latency chart**: inline SVG built with `createElementNS` (no libraries,
  no `<canvas>`) with two normalized series — *latency* (accent) and *cost*
  (success) — plus a legend with each series' real peak (`renderChart`).
- **Recent requests table**: keyboard-sortable columns (`button.th-sort` with
  `aria-sort` on the `<th>`), client-side pagination (`renderPagination`), and
  `$`/`ms`/thousands formats (`formatUsd`, `formatLatency`, `formatTokens`).
- **Empty and loading states**: shimmer skeleton while data is in flight
  (`renderDashboardSkeleton`) and explicit empty messages when there's no
  activity (`#recent-empty`, `#chart-empty`).

## Responsive layout (console, mobile-first)

The layout is **mobile-first**: base styles target the smallest screen (320px
mobile) and improve progressively with `min-width` media queries. No horizontal
scroll at any size.

| Breakpoint | CSS token | What changes |
|---|---|---|
| Mobile | (base) | `.layout` single column, compact gutters, sticky header with truncated (ellipsis) title. |
| Tablet | `--bp-tablet` (≥640px) | Wider gutters and card padding; header padding. |
| Desktop | `--bp-desktop` (≥1024px) | Stats in more columns. |

Hierarchy and spacing conventions:

- All content lives in `.layout` (CSS Grid, `minmax(0, 1fr)` to prevent
  overflow), aligned and centered with `max-width: var(--max-width)`.
- The header is `position: sticky` with `z-index` above content.
- Tables use `.table-wrap` (`overflow-x: auto`) instead of forcing page scroll
  on mobile.

## Content Security Policy

`SecurityHeadersMiddleware` (`app/api/middleware.py`) sets a strict CSP on every
response:

- `default-src 'self'` — everything is same-origin by default.
- `script-src 'self' 'sha256-…' https://cdn.tailwindcss.com` — the SHA-256 hash
  covers the console's inline anti-FOUC theme bootstrap (the only inline
  script); the Tailwind origin is allowed for the landing's no-build utility
  CSS. Arbitrary inline/eval'd scripts remain forbidden.
- `style-src 'self' 'unsafe-inline'` — required because the Tailwind CDN
  injects its generated stylesheet as a `<style>` element at runtime.
- `img-src 'self' data:` — for the inline SVG favicon.

## Accessibility and security

- **Live announcements (`aria-live`)**: the console conversation is a polite
  `role="log"`, and a visually-hidden announcer (`#chat-announcer`,
  `aria-live="polite"`) speaks key events — "*Answer received*", "*A human
  review is required*", "*Request blocked by the input guardrail*" — without
  moving focus. The landing answer region is itself a polite live region.
- **Full keyboard navigation**: skip link ("Skip to main content") jumping to
  `<main id="main">` (`tabindex="-1"`), and explicit focus management in the
  modal: on open, focus moves to the first decision radio and, on close,
  returns to the previously focused element (`lastFocusedElement`).
- **Contrast ≥ AA and `prefers-reduced-motion`**: palette verified against the
  active theme; the `@media (prefers-reduced-motion: reduce)` block collapses
  all animations/transitions (WCAG 2.3.3).
- **Total sanitization**: the ES modules **never** assign to `innerHTML` — all
  dynamic content (LLM answer, dashboard, errors) uses
  `textContent`/`createElement`. There is no code path that turns an XSS
  payload into executable markup.
- **Toasts and global error handling**: a `#toasts` container (`role="status"`,
  `aria-live="polite"`) shows dismissible transient notifications
  (`showToast`), and the global `window.addEventListener("error")` /
  `("unhandledrejection")` listeners turn unexpected failures into a visible
  notification instead of dying silently in the console.
