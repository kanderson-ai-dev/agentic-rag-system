# Iteration 1 — generate_answer v1 → v2

Date: 2026-10-12 · Dataset: `agentic-rag-system-eval-v6ec2c719f9` (18 cases)
· Baseline: `v1_baseline.json` (hybrid mode, promoted v1 prompts, real
Pinecone + Neo4j backends, `gpt-4o-mini`).

## Weakest real signal

`selfrag_dependencies` (multi-hop): `context_recall = 0.5`. All other
answer-cases scored 1.0 on every RAGAS metric; five cases sat at
`llm_as_judge = 4.33`.

## Root cause (evidence, not guess)

Retrieval did its job — `ctx1` of the case contains the missing fact
verbatim ("When retries are exhausted, the system escalates to
human-in-the-loop review rather than hallucinating from weak context").
The generated answer covers critique + rewrite + LangGraph but omits the
HITL escalation leg entirely. The v1 prompt imposes a hard cap
("max three sentences") which compresses multi-part answers and drops
trailing concepts.

## Hypothesis

`generate_answer` v2: replace the hard sentence cap with a completeness
instruction ("cover every distinct concept or relationship the context
supports; completeness over brevity"). Expected: `selfrag_dependencies`
context_recall 0.5 → 1.0 and judge 4.33 → ~5, without regressing
faithfulness/answer_relevancy elsewhere. Risk: longer answers cost
slightly more tokens — bounded by the ≤ $0.01/query gate.

## Method

A/B via `evaluation/ab_test.py` (shared dataset, identical graph, only
the pinned prompt version differs) → promotion gate (thresholds +
regression vs. v1 + behavior invariants). A/B result: `../ab/`.

## Outcome — v2 promoted

A/B `0b2856ed2f27` (real keys, pinecone + neo4j, `gpt-4o-mini`):

| metric | v1 | v2 | delta |
|---|---|---|---|
| llm_as_judge | 4.762 | 4.857 | +0.095 |
| latency_p50_ms | 3165 | 2789 | -376 |
| latency_p95_ms | 6758 | 4083 | -2675 |
| faithfulness / relevancy / precision / recall | unchanged | unchanged | 0 |
| cost_per_query_usd | 0.0004 | 0.0008 | +0.0004 |

Target case `selfrag_dependencies`: judge 4.33 → 5.0 and the HITL
escalation leg is back in the answer. `context_recall` stayed 0.5 —
that metric measures the retrieved context vs. ground truth, which a
generation prompt cannot move (retrieval-side iteration is a future
candidate).

Gate note: the first verdict was BLOCKED — the generic regression check
treated lower-is-better metrics (cost, latency) as regressing when they
improved. Fixed in `promotion_gate.py` (lower-is-better set excluded
from the generic check; p50/p95 get a dedicated ≤ baseline × 1.20 rule);
re-evaluated on the same saved scorecards → ALLOWED.

Langfuse evidence: experiment `ga2-gd1-lj1-tq1-f964ae3-2026-10-12` on
dataset `agentic-rag-system-eval-v6ec2c719f9` (see
`docs/eval/langfuse-evidence.md`).
