# Langfuse evidence — Phase LF

Why Langfuse: LangSmith is the default provider, but its monthly unique-trace
quota is exhausted (HTTP 429 `tenant exceeded usage limits` on ingestion —
auth and reads still work). Per the fallback decision, Langfuse carries the
tracing + experiment evidence until the quota recovers.

Switch: `TRACING_PROVIDER=none|langsmith|langfuse` in `.env` (default `none`).

## Smoke evidence (real keys, 2026-10-12)

Three `POST /api/v1/rag/query` calls — one trace each, 13 typed observations
(CHAIN / GENERATION / SPAN tree: guardrail → retrieve → grade_documents →
generate → output_guardrail), `X-Trace-Id` response header set:

- <https://us.cloud.langfuse.com/project/cmv2ur3v108f6ad0jyrwm1hw2/traces/6aeaca7f06464ec35f4bcf6ff7b1bec8>
- <https://us.cloud.langfuse.com/project/cmv2ur3v108f6ad0jyrwm1hw2/traces/6b932cf9522d0952a290cd330e849827>
- <https://us.cloud.langfuse.com/project/cmv2ur3v108f6ad0jyrwm1hw2/traces/ada395671cd2474a3b8c47262e7c24d4>

One evaluation run — dataset `agentic-rag-system-smoke` (2 items), experiment
`lf-smoke-run-1`, evaluator score `expected_behavior = 1.0` on both items
(injection case blocked via `error_output`, zero LLM spend):

- Dataset: <https://us.cloud.langfuse.com/project/cmv2ur3v108f6ad0jyrwm1hw2/datasets>
  → `agentic-rag-system-smoke` → experiment `lf-smoke-run-1`

Note: this Langfuse organization is on the v4 API — "dataset runs" appear in
the UI under **Experiments**, and public read APIs are
`/api/public/v2/observations` + `/api/public/experiments` (legacy
`/api/public/datasets/.../runs/...` returns HTTP 410).

## Screenshots to capture for the README (Phase 8+)

- Trace tree: open any trace URL above → the span waterfall.
- Experiment table: Datasets → `agentic-rag-system-smoke` → `lf-smoke-run-1`
  (per-item scores column).
- For the full evidence run (Phase 6/7): dataset `agentic-rag-system-eval-v<hash>`
  and experiment `agentic-rag-system-<prompt-version-tag>` via
  `evaluation/run_evaluation.py` with `TRACING_PROVIDER=langfuse`.

## Phase 7 evidence — generate_answer v2 promotion (2026-10-12)

Full-dataset experiment (18 items) on the promoted v2 prompts:

- Dataset `agentic-rag-system-eval-v6ec2c719f9`, experiment
  **`ga2-gd1-lj1-tq1-f964ae3-2026-10-12`** (prompt tag encodes
  generate_answer v2, grade_documents v1, llm_judge v1, transform_query v1).
- UI path: Datasets → `agentic-rag-system-eval-v6ec2c719f9` → experiment
  `ga2-gd1-lj1-tq1-f964ae3-2026-10-12` → per-item evaluator scores.

Measured A/B backing the promotion (committed):
`evaluation/results/ab/0b2856ed2f27.json` — llm_as_judge 4.76 → 4.86,
p95 latency 6758 ms → 4083 ms, all other metrics equal or better; the
promotion gate initially blocked on a sign bug (lower-is-better metrics
flagged as regressions) — fixed and re-evaluated to ALLOWED.
