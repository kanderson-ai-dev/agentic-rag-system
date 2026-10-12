"""Shared evaluation harness: run cases, score behavior, stamp evidence.

Both entrypoints (`run_ragas.py`, `run_evaluation.py`) and the A/B engine run
the *same* dataset through the *same* code path so numbers stay comparable:

- ``run_cases`` invokes the compiled graph once per dataset case, measuring
  end-to-end wall-clock latency around the full graph call.
- ``score_results`` turns raw results into a flat scorecard: RAGAS-style
  metrics over ``answer`` cases, behavior rates for injection/out-of-domain
  cases, cost and p50/p95 latency.
- ``run_metadata`` stamps provenance (prompt versions, dataset hash, git sha,
  model, backends, retrieval mode) on every scorecard — README claims may only
  cite backends these fields actually name.
"""

import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.core.config import Settings
from app.core.cost_tracking import TokenUsageCallbackHandler, calculate_cost_usd
from app.graph.graph import initial_state
from app.graph.prompts import REFUSAL_TEXT, active_versions
from app.services.prompt_registry import PromptName
from evaluation.dataset import dataset_version

MetricScorer = Callable[..., float]
"""(context=..., answer=..., question=...) -> 0.0-1.0, or judge 1.0-5.0."""

QUALITY_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "llm_as_judge",
)

BEHAVIOR_METRICS = ("injection_block_rate", "ood_refusal_rate", "behavior_accuracy")


class CostCapExceeded(RuntimeError):
    """The run spent more than ``--max-cost-usd`` — abort before more spend."""

    def __init__(self, spent_usd: float, cap_usd: float) -> None:
        super().__init__(
            f"cost cap exceeded: spent ${spent_usd:.4f} > cap ${cap_usd:.4f}"
        )
        self.spent_usd = spent_usd
        self.cap_usd = cap_usd


@dataclass
class CaseResult:
    """One dataset case after a graph invocation (+ scores once judged)."""

    id: str
    category: str
    expected_behavior: str
    question: str
    ground_truth: str
    answer: str
    contexts: list[str]
    sources: list[str]
    blocked: bool
    escalated: bool
    latency_ms: int
    behavior_ok: bool
    scores: dict[str, float] = field(default_factory=dict)


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile over a sorted-or-unsorted list (p in 0..100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, round(len(ordered) * p / 100 + 0.499999))
    return ordered[min(rank - 1, len(ordered) - 1)]


def _behavior_ok(expected: str, *, blocked: bool, escalated: bool, answer: str) -> bool:
    """Whether the observed behavior matches the case's expected_behavior."""
    if expected == "block_injection":
        return blocked
    if expected == "refuse_out_of_domain":
        return blocked or escalated or REFUSAL_TEXT in answer
    # "answer": any non-blocked path counts — the *quality* of the answer is
    # what the LLM metrics score, not the behavior check.
    return not blocked


def run_cases(
    graph: Any,
    cases: list[dict[str, str]],
    *,
    thread_prefix: str = "eval",
    token_handler: TokenUsageCallbackHandler | None = None,
    max_cost_usd: float | None = None,
    settings: Settings | None = None,
    deadline_seconds: float | None = None,
) -> list[CaseResult]:
    """Invoke the graph once per case with per-case latency and behavior checks.

    ``max_cost_usd`` aborts the run between cases (never mid-LLM-call) when the
    accumulated spend in ``token_handler`` exceeds it — measured with the
    settings' per-token prices. ``deadline_seconds`` stops the loop similarly
    and raises ``CostCapExceeded``-style timeout via ``TimeoutError``.
    """
    results: list[CaseResult] = []
    started = time.monotonic()
    for case in cases:
        _check_budget(token_handler, max_cost_usd, settings)
        if deadline_seconds is not None and time.monotonic() - started > deadline_seconds:
            raise TimeoutError(
                f"evaluation exceeded deadline of {deadline_seconds}s "
                f"after {len(results)} cases"
            )

        start = time.perf_counter()
        result = graph.invoke(
            initial_state(case["question"]),
            config={
                "configurable": {"thread_id": f"{thread_prefix}-{case['id']}"}
            },
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        documents = result.get("documents", [])
        answer = result.get("generation", "")
        blocked = bool(result.get("blocked", False))
        escalated = bool(result.get("__interrupt__"))
        results.append(
            CaseResult(
                id=case["id"],
                category=case["category"],
                expected_behavior=case["expected_behavior"],
                question=case["question"],
                ground_truth=case["ground_truth"],
                answer=answer,
                contexts=[doc.page_content for doc in documents],
                sources=sorted(
                    {str(doc.metadata.get("source", "")) for doc in documents}
                    - {""}
                ),
                blocked=blocked,
                escalated=escalated,
                latency_ms=latency_ms,
                behavior_ok=_behavior_ok(
                    case["expected_behavior"],
                    blocked=blocked,
                    escalated=escalated,
                    answer=answer,
                ),
            )
        )
    _check_budget(token_handler, max_cost_usd, settings)
    return results


def score_results(
    results: list[CaseResult],
    scorers: Mapping[str, MetricScorer],
    *,
    token_handler: TokenUsageCallbackHandler | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Score results into a flat scorecard.

    ``scorers`` maps metric name -> function taking (question=, answer=,
    context=) kwargs; ``llm_as_judge`` returns a 1-5 mean, the rest 0-1.
    Scoring itself calls the LLM — its tokens land in ``token_handler`` so the
    reported cost covers judging too.
    """
    answer_cases = [r for r in results if r.expected_behavior == "answer"]
    for case in answer_cases:
        context = "\n\n".join(case.contexts)
        for name, scorer in scorers.items():
            if name not in QUALITY_METRICS:
                continue
            case.scores[name] = scorer(
                question=case.question, answer=case.answer, context=context
            )

    sums: dict[str, float] = {name: 0.0 for name in QUALITY_METRICS}
    for case in answer_cases:
        for name in QUALITY_METRICS:
            sums[name] += case.scores.get(name, 0.0)
    n = len(answer_cases) or 1

    injection = [r for r in results if r.expected_behavior == "block_injection"]
    ood = [r for r in results if r.expected_behavior == "refuse_out_of_domain"]
    behavior_cases = injection + ood
    latencies = [float(r.latency_ms) for r in results]
    total_cost = _cost_so_far(token_handler, settings)

    return {
        "faithfulness": round(sums["faithfulness"] / n, 4),
        "answer_relevancy": round(sums["answer_relevancy"] / n, 4),
        "context_precision": round(sums["context_precision"] / n, 4),
        "context_recall": round(sums["context_recall"] / n, 4),
        "llm_as_judge": round(sums["llm_as_judge"] / n, 4),
        "injection_block_rate": _rate(injection),
        "ood_refusal_rate": _rate(ood),
        "behavior_accuracy": _rate(behavior_cases),
        "cost_total_usd": round(total_cost, 6),
        "cost_per_query_usd": round(total_cost / (len(results) or 1), 6),
        "latency_p50_ms": int(percentile(latencies, 50)),
        "latency_p95_ms": int(percentile(latencies, 95)),
        "cases_count": len(results),
        "answered_cases": len(answer_cases),
        "per_category": _per_category(results),
        "cases": [asdict(r) for r in results],
    }


def _rate(cases: list[CaseResult]) -> float | None:
    if not cases:
        return None
    return round(sum(1 for c in cases if c.behavior_ok) / len(cases), 4)


def _per_category(results: list[CaseResult]) -> dict[str, Any]:
    buckets: dict[str, list[CaseResult]] = {}
    for r in results:
        buckets.setdefault(r.category, []).append(r)
    per_cat: dict[str, Any] = {}
    for category, cases in sorted(buckets.items()):
        answered = [c for c in cases if c.expected_behavior == "answer"]
        entry: dict[str, Any] = {"cases": len(cases)}
        if answered:
            for name in QUALITY_METRICS:
                entry[name] = round(
                    sum(c.scores.get(name, 0.0) for c in answered) / len(answered),
                    4,
                )
        entry["behavior_rate"] = _rate(cases)
        entry["latency_p50_ms"] = int(
            percentile([float(c.latency_ms) for c in cases], 50)
        )
        per_cat[category] = entry
    return per_cat


def _cost_so_far(
    token_handler: TokenUsageCallbackHandler | None,
    settings: Settings | None,
) -> float:
    if token_handler is None or settings is None:
        return 0.0
    return calculate_cost_usd(
        token_handler.prompt_tokens,
        token_handler.completion_tokens,
        settings.cost_input_price_per_1m,
        settings.cost_output_price_per_1m,
    )


def _check_budget(
    token_handler: TokenUsageCallbackHandler | None,
    max_cost_usd: float | None,
    settings: Settings | None,
) -> None:
    if max_cost_usd is None:
        return
    spent = _cost_so_far(token_handler, settings)
    if spent > max_cost_usd:
        raise CostCapExceeded(spent, max_cost_usd)


def git_sha() -> str:
    """Short sha of HEAD, or 'unknown' outside a git checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return "unknown"


def run_metadata(
    settings: Settings,
    *,
    retrieval_mode: str = "hybrid",
) -> dict[str, Any]:
    """Provenance stamped on every scorecard.

    ``graph_backend``/``vector_backend`` mirror the backend-selection logic in
    the service factories — a scorecard may only name Neo4j/Pinecone when the
    corresponding credentials drove this run.
    """
    return {
        "prompt_versions": active_versions(),
        "retrieval_mode": retrieval_mode,
        "dataset_version": dataset_version(),
        "git_sha": git_sha(),
        "model": settings.chat_model_name,
        "vector_backend": (
            "pinecone" if settings.pinecone_api_key else "chroma"
        ),
        "graph_backend": (
            "neo4j"
            if settings.neo4j_uri and settings.neo4j_password
            else "networkx"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
    }


def parse_prompt_version_overrides(values: list[str] | None) -> dict[str, int]:
    """Parse repeatable ``--prompt-version name=N`` flags into {name: int}."""
    pins: dict[str, int] = {}
    for raw in values or []:
        name, sep, version = raw.partition("=")
        if not sep or not name.strip() or not version.strip().isdigit():
            raise ValueError(
                f"invalid --prompt-version {raw!r}; expected name=N"
            )
        PromptName(name.strip())  # raises ValueError on unknown prompt names
        pins[name.strip()] = int(version.strip())
    return pins


def prompt_version_tag(versions: dict[str, int]) -> str:
    """Compact ``ga1-gd1-tq1-lj1`` tag for experiment names."""
    abbrev = {
        "generate_answer": "ga",
        "grade_documents": "gd",
        "transform_query": "tq",
        "llm_judge": "lj",
    }
    return "-".join(
        f"{abbrev.get(name, name)}{version}"
        for name, version in sorted(versions.items())
    )
