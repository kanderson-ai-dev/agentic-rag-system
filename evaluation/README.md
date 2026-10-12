# Evaluation

One dataset, one harness, two runners.

## Dataset (`dataset.py`)

`build_dataset()` is the single source of evaluation cases — `run_ragas.py`,
`run_evaluation.py` and the A/B engine all consume it. Each case:

| field | meaning |
|---|---|
| `id` | stable slug (never reused for a different case) |
| `category` | `single_hop`, `multi_hop`, `out_of_domain`, `injection`, `other_language`, `vague` |
| `question` | the input sent to the graph |
| `ground_truth` | reference answer (or, for behavior cases, the expected outcome) |
| `expected_behavior` | `answer`, `refuse_out_of_domain`, or `block_injection` |

`dataset_version()` is a short SHA-256 over the canonical case list — it feeds
the dataset name (`agentic-rag-system-eval-v<hash>`, identical in LangSmith
and Langfuse) and is stamped on every scorecard, so a scorecard can never be
silently compared against a different question set.

## Tracing provider (`TRACING_PROVIDER`)

`run_evaluation.py` dispatches on `TRACING_PROVIDER`:

| value | behavior |
|---|---|
| `none` (default) | local harness scorecard only — daily development, CI, offline work |
| `langsmith` | real LangSmith Experiment via `evaluate()` (dataset + per-run evaluators) |
| `langfuse` | real Langfuse experiment via `dataset.run_experiment()` (dataset run + one trace per item + evaluator scores) |

Both provider paths fall back to the local scorecard when their backend is
missing keys or unreachable — **the local scorecard is always the
promotion-gate source of truth**; provider dashboards are evidence, not the
gate. The `run_ragas.py` harness never emits traces on purpose: it produces
the scorecard numbers, while `run_evaluation.py` produces the per-item
experiment traces.

Langfuse is the conditional fallback for when LangSmith is unusable (401/403
key problems or HTTP 429 monthly-trace-quota exhaustion — the latter hit in
practice, which is why this integration exists). With the provider selected,
every API request is traced too — `X-Trace-Id` response header included —
so flip it on only for deliberate evidence runs, then set it back to `none`.
For self-hosted Langfuse, only `LANGFUSE_BASE_URL` changes (e.g.
`http://localhost:3001`); the same code path produces local evidence with
unlimited trace budget.

## How non-answer cases are scored

Quality metrics (faithfulness, answer_relevancy, context_precision,
context_recall, llm_as_judge) are only meaningful when the system actually
answers — they are averaged **over `expected_behavior: answer` cases only**.

Behavior cases are scored as pass/fail checks instead:

| expected_behavior | passes when | scorecard field |
|---|---|---|
| `block_injection` | the input guardrail sets `blocked` before retrieval — zero LLM tokens spent | `injection_block_rate` |
| `refuse_out_of_domain` | the answer contains the refusal text or the run escalates to human review | `ood_refusal_rate` |

`behavior_accuracy` is the mean pass rate across all behavior cases. A system
that answers out-of-domain questions confidently scores *worse* here — that is
the point: the scorecard rewards not hallucinating.

## Flags (both runners)

```
python evaluation/run_ragas.py \
    --prompt-version generate_answer=2   # repeatable; pins a version without promoting it
    --retrieval-mode hybrid              # hybrid | vector_only | graph_only | no_retrieval
    --max-cost-usd 0.50                  # abort between cases past this spend
    --output results/my_scorecard.json
```

Prompt pins resolve through the registry (candidate/draft versions are fine)
and are recorded in the scorecard's `meta.prompt_versions` — pins never touch
registry state or write audit entries.

## Scorecard provenance

Every scorecard stamps `meta`: `prompt_versions`, `retrieval_mode`,
`dataset_version` (content hash), `git_sha`, `model`, `vector_backend`
(`pinecone|chroma`), `graph_backend` (`neo4j|networkx`), `generated_at`.
Per-case results include end-to-end `latency_ms` (wall clock around the full
graph call) aggregated into `latency_p50_ms` / `latency_p95_ms`, plus a
`per_category` breakdown — the `single_hop` vs `multi_hop` split is what the
retrieval-mode comparison reads.

## Files

- `run_ragas.py` — local scorecard + `results/history/` + `TREND.md`
- `run_evaluation.py` — provider experiment: LangSmith `evaluate()` or Langfuse `run_experiment()` per `TRACING_PROVIDER` (local scorecard fallback)
- `harness.py` — shared case runner / scorer / provenance
- `evaluators.py` — the LLM-judged metric functions
- `results/` — scorecards, history, A/B results (`ab/`)
