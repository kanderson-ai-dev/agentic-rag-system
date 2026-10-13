# Visual UI QA — checklist & evidence

Self-assessment pass over the rendered UI. Captures are generated deterministically
by `scripts/ui_qa.py` (stub graph, same technique as the functional suite) into
`docs/ui-qa/`; a machine-readable index lives in `docs/ui-qa/manifest.json`.

Automated rules are enforced by `tests/e2e/test_ui_content.py` (run with
`pytest -m e2e`, deterministic — no keys). Manual inspection was performed on the
committed PNGs.

## Visual & layout checks

| # | Check | Evidence | Status |
|---|-------|----------|--------|
| 1 | Landing empty state: headline, placeholder, char counter `0/1000`, send hint | `landing-empty-1280-dark.png` | ✅ |
| 2 | Landing at 768px: no overflow, composer stays usable | `landing-empty-768-dark.png` | ✅ |
| 3 | Landing at 375px: single column, counter visible, no clipping | `landing-empty-375-dark.png` | ✅ |
| 4 | Landing answer state: typography, question echoed, source chip rendered | `landing-answer.png` | ✅ |
| 5 | Landing escalated state (F6): friendly copy + console link, no fabricated sources | `landing-escalated.png` | ✅ |
| 6 | Landing guardrail-blocked state: calm safe-refusal, not an error banner | `landing-blocked.png` | ✅ |
| 7 | Console chat: user bubble, assistant answer, prompt-version footer, copy control | `console-chat.png` | ✅ |
| 8 | HITL modal: question, humanised reason, three actions with hints, focus ring on Approve | `console-hitl-modal.png` | ✅ |
| 9 | Dashboard cards + recent table: session scoping note, Prompt column (`v2`), status pills | `console-dashboard.png` | ✅ |
| 10 | Performance panel: p50/p95 vs targets with text labels, node latency bars (human names), daily-budget bar, Human-vs-Agent table with stated assumptions | `console-performance.png` | ✅ |
| 11 | Prompts list: human labels, active-version badges, version counts, "Audit chain ✓ verified" | `console-prompts-list.png` | ✅ |
| 12 | Prompt diff: explicit `−`/`+` markers, line numbers, red/green rows | `console-prompts-diff.png` | ✅ |
| 13 | A/B panel: baseline vs candidate vs delta, gate verdict explained in words | `console-prompts-ab.png` | ✅ |
| 14 | Candidate editor: preloaded promoted text, required variables listed, modal layout | `console-prompts-editor.png` | ✅ |
| 15 | Console at 375px: all sections stack, cards readable, no horizontal scroll | `console-mobile-375.png` | ✅ |
| 16 | Console at 768px: dashboard cards wrap cleanly, tables scroll within cards | `console-tablet-768.png` | ✅ |
| 17 | Light theme (console owns the theme tokens): full palette swap, no dark leftovers | `console-dashboard-1280-light.png` | ✅ |

The public landing is intentionally dark-only — its stylesheet declares no
`prefers-color-scheme` variants, so the light/dark pair is exercised on the
console, where `[data-theme="light"]` tokens exist.

## Content rules (automated)

`tests/e2e/test_ui_content.py` sweeps rendered `innerText` on every screen
(landing ×4, console dashboard, HITL modal, prompts surfaces, light theme):

| Rule | How enforced | Status |
|------|--------------|--------|
| No `snake_case` identifiers | `\b[a-z]+(?:_[a-z0-9]+)+\b` sweep + named-token list (`thread_id`, `cost_usd`, `latency_ms`, `session_id`, `max_retries_exhausted`, …) → zero hits | ✅ |
| No broken templating | `undefined`, `null`, `NaN`, `[object Object]`, `Invalid Date` → zero hits | ✅ |
| Consistent formats | measured money 4-decimal (`$0.0003`), targets may use 2 (`$0.01`), bare integers only as rates (`$30/h`); latency always with `ms`/`s`; percentages explicit | ✅ |
| `<img>` alt | every image carries an `alt` attribute | ✅ |
| Accessible names | every visible button/link has text or `aria-label`; every field labelled (label/aria/placeholder) | ✅ |
| Focus ring | Tab-driven focus paints outline or box-shadow (`:focus-visible`) | ✅ |
| Contrast ≥ 4.5:1 | computed sRGB luminance on representative text, dark **and** light themes | ✅ |
| Empty states | fresh session shows `$0.0000` zeros + "No requests recorded this session yet" + audit-chain status — never blank areas | ✅ |

CSP: `Content-Security-Policy` (`default-src 'self'`, `script-src 'self'` +
Tailwind CDN only) is set by `SecurityHeadersMiddleware`; the captures were taken
through the real middleware with zero console CSP violations.

## Issues found during this pass — and fixed

| # | Found | Fix |
|---|-------|-----|
| 1 | "Latency by graph node" rendered raw ids (`budget_gate`, `grade_documents`, …) | `dashboard.js` now renders `label("node", …)` — visible in `console-performance.png` |
| 2 | Landing Ask button white-on-`indigo-500` measured 4.47:1 (below AA) | `bg-indigo-600` + hover `bg-indigo-700` → ~5.9:1 |
| 3 | `generate_answer` v2 changelog leaked shorthand (`selfrag_dependencies crec=0.5`) into the diff view | changelog rewritten in plain prose |
| 4 | Sticky `.topbar` painted over scrolled element captures | capture script hides it with `visibility` during element shots (layout preserved) |
| 5 | A/B capture timed out waiting for `.ab-verdict` | `ui_qa.py` built a fresh `AbTestJobManager` per request — the poll 404'd the job. Single shared instance now |
| 6 | "Light" landing capture was a no-op (dark-only page) | replaced with console light-theme capture, which owns the real tokens |

Re-captured all PNGs after the fixes — statuses above reflect the final set.
