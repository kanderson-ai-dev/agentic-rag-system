// Human-readable label dictionary — the single map for every internal
// identifier the UI can render (prompt names, lifecycle statuses, graph
// nodes, interrupt reasons, backends). `label(kind, key)` falls back to a
// humanizer and warns on a missing entry, so a new identifier can never
// leak raw snake_case into the interface — a Python test also sweeps this
// file against the server-side enums to catch drift.
//
// Keep entries one-per-line (`"key": "Label"`) — the drift test parses them.

const LABELS = {
  prompt: {
    "generate_answer": "Answer generation",
    "grade_documents": "Document grading",
    "transform_query": "Query rewriting",
    "llm_judge": "LLM judge",
  },
  status: {
    "draft": "Draft",
    "candidate": "Candidate",
    "promoted": "Promoted",
    "superseded": "Superseded",
    "rolled_back": "Rolled back",
  },
  node: {
    "guardrail": "Input guardrail",
    "budget_gate": "Budget gate",
    "error_output": "Safe refusal",
    "retrieve": "Hybrid retrieval",
    "grade_documents": "Document grading",
    "generate": "Answer generation",
    "transform_query": "Query rewriting",
    "human_review": "Human review",
    "output_guardrail": "Output guardrail",
  },
  interrupt_reason: {
    "max_retries_exhausted": "Retries exhausted",
  },
  backend: {
    "pinecone": "Pinecone",
    "chroma": "Chroma",
    "neo4j": "Neo4j",
    "networkx": "NetworkX",
    "vector": "Vector search",
    "graph": "Knowledge graph",
  },
  metric: {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "llm_as_judge": "LLM judge",
    "injection_block_rate": "Injection block rate",
    "ood_refusal_rate": "OOD refusal rate",
    "behavior_accuracy": "Behavior accuracy",
    "cost_per_query_usd": "Cost per query",
    "cost_total_usd": "Total cost",
    "latency_p50_ms": "p50 latency",
    "latency_p95_ms": "p95 latency",
    "cases_count": "Cases",
    "answered_cases": "Answered cases",
  },
  job_status: {
    "queued": "Queued",
    "running": "Running",
    "succeeded": "Succeeded",
    "failed": "Failed",
  },
};

// Last-resort humanizer for an identifier missing from the dictionary:
// "some_raw_name" -> "Some raw name". Always warns so the gap gets fixed.
function humanize(key) {
  const words = String(key).replace(/[_-]+/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export function label(kind, key) {
  if (key === null || key === undefined || key === "") return "";
  const table = LABELS[kind] || {};
  if (Object.prototype.hasOwnProperty.call(table, key)) return table[key];
  console.warn(`labels: missing entry for ${kind}:${key}`);
  return humanize(key);
}
