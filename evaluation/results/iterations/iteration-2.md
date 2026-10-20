# Iteration 2 — generate_answer v2 → v3 (live A/B evidence)

Date: 2026-10-19 · Dataset: `agentic-rag-system-eval-v6ec2c719f9` (18 cases)
· Baseline: promoted `generate_answer v2` · Backends as configured:
**Pinecone + NetworkX** (`gpt-4o-mini`; the Neo4j Aura instance was
unreachable at run time, so the graph arm honestly ran the local NetworkX
fallback — `meta.graph_backend` records what actually served).

## Weakest real signal

Iteration 1's per-case judge scores: `interrupt_fn` and
`checkpointer_role` sit at `llm_as_judge = 4.33` — the lowest in the set.
Both answers *paraphrase* the mechanism ("the graph pauses and can be
resumed") where the retrieved context names it concretely
("a checkpointer", `Command(resume=...)`, `thread_id`).

## Hypothesis

`generate_answer` v3: add a name-the-mechanism instruction — when the
context provides a concrete component (`checkpointer`, `thread_id`,
`Command(resume=...)`, …) the answer must name it instead of
paraphrasing. Expected: judge 4.33 → ~5 on the two target cases, no
faithfulness/relevancy regression. This is a real semantic edit, not a
no-op candidate.

## Method

- Console/API runs — through the **product path**: create candidate →
  `POST …/ab-test` → polled job → `evaluation/ab_test.py` engine, via
  `scripts/ui_qa.py --live` and `scripts/record_demo.py --live`; hard
  caps `AB_TEST_MAX_COST_USD=0.50` / `AB_TEST_TIMEOUT_SECONDS=600`.
- Engine run — `python -m evaluation.ab_test` directly, `PROMPTS_DIR`
  pointed at a throwaway registry copy, `--max-cost-usd 0.50`.

Registry writes for every run lived in a temp copy of `prompts/` — the
tracked tree was never mutated (v3 remains a candidate, not promoted).

## Outcome — five live runs, verdict split

Same candidate, same engine, five real runs — the spread between them is
judge/latency variance, not a code change:

| run | path | judge v2 → v3 | p50 | p95 | spend (both arms) | gate |
|---|---|---|---|---|---|---|
| `f0918333d964` | console job `776daf547254` | 4.952 → 4.905 | 4 168 → 2 718 | 10 321 → 29 761 | $0.0077 + $0.0162 | **BLOCKED** (judge −0.047, p95 +188%) |
| `f0ef98a18f98` | console job `547e02986317` | 4.952 → 4.905 | 3 510 → 2 640 | 19 624 → 11 594 | $0.0076 + $0.0161 | **BLOCKED** (judge −0.047) |
| `b704310f28cb` | console job `1fef299d6e13` | 5.000 → 4.976 | 5 633 → 2 818 | 19 350 → 4 516 | $0.0076 + $0.0160 | **BLOCKED** (judge −0.024) |
| `1ef09bac29aa` | console job `6eaebd37a1be` | 4.952 → 4.976 | 3 446 → 2 864 | 7 878 → 5 527 | $0.0076 + $0.0161 | **ALLOWED** (+0.024) |
| `737bddf9fa85` | engine CLI | 4.905 → 4.976 | 3 558 → 2 882 | 6 919 → 4 358 | $0.0076 + $0.0161 | **ALLOWED** (+0.071) |

All results carry `meta.live: true`, `meta.model: gpt-4o-mini`,
`meta.git_sha: 7475264`, 18 cases per arm, nonzero `cost_total_usd` on
both arms; the later runs also stamp `meta.tokens` (~76k prompt +
~7.7k completion ≈ $0.024 per run). Every file — blocked or allowed —
is committed under `evaluation/results/ab/`; a blocked gate is valid
evidence, published rather than re-tuned. The `1ef09bac29aa` run is the
one shown live in `docs/screenshots/rag-demo-live.gif`.

Reading: 2 of 5 runs allowed. The judge deltas (−0.047 … +0.071) sit
inside typical 1–5 judge noise and the p95 swings trace to single
outlier cases — at n=18 the evidence is inconclusive, so v2 stays
promoted and v3 stays a candidate. The honest next step is a larger
dataset or a repeated-run promotion policy, not a candidate tweak
after a blocked verdict.
