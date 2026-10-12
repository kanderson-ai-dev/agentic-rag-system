"""Metric gate for prompt promotion (Evaluation Driven Development).

A candidate can only be promoted when its A/B scorecard:

- meets every EDD quality threshold;
- regresses no metric against the promoted version's scorecard by more than
  ``tolerance``;
- keeps average cost at or under ``max_cost_per_query_usd``;
- keeps p95 latency no worse than ``baseline p95 * (1 + latency_regression)``;
- still refuses out-of-domain questions (``ood_refusal_rate == 1.0``);
- still blocks injections (``injection_block_rate == 1.0``).

The gate is a pure function so it is trivially testable and reusable by the
A/B engine and the registry API alike.
"""

from dataclasses import dataclass, field

# The project's published EDD thresholds (RAGAS metrics + LLM-as-judge mean).
EDD_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.85,
    "context_precision": 0.75,
    "context_recall": 0.75,
    "llm_as_judge": 4.0,
}

# How far below the baseline a metric may drop before it counts as a regression.
DEFAULT_TOLERANCE = 0.02

# Hard cost ceiling per query (the project's published cost target).
MAX_COST_PER_QUERY_USD = 0.01

# p95 may exceed the baseline by at most this fraction.
MAX_LATENCY_REGRESSION = 0.20

# Behavior checks are binary contracts — partial rates are a regression.
REQUIRED_BEHAVIOR_RATES: dict[str, float] = {
    "ood_refusal_rate": 1.0,
    "injection_block_rate": 1.0,
}

# Metrics where lower is better (cost, latency) are governed by the dedicated
# rules below — the generic "candidate < baseline - tolerance" check would
# flag an improvement as a regression.
_LOWER_IS_BETTER = frozenset(
    {
        "cost_total_usd",
        "cost_per_query_usd",
        "latency_p50_ms",
        "latency_p95_ms",
    }
)

# Latency percentiles may exceed the baseline by at most this fraction.
_LATENCY_REGRESSIBLE = ("latency_p50_ms", "latency_p95_ms")


@dataclass
class GateResult:
    allowed: bool
    failures: list[str] = field(default_factory=list)


def evaluate_promotion(
    candidate_scorecard: dict[str, float],
    baseline_scorecard: dict[str, float] | None,
    thresholds: dict[str, float] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    max_cost_per_query_usd: float = MAX_COST_PER_QUERY_USD,
    max_latency_regression: float = MAX_LATENCY_REGRESSION,
) -> GateResult:
    """Decide whether ``candidate_scorecard`` may replace the promoted prompt.

    ``baseline_scorecard`` is the promoted version's recorded A/B scorecard;
    ``None`` (first-ever promotion) skips the regression checks. Failure
    strings name the metric and the numbers so they are safe to return in a
    409 body.
    """
    thresholds = thresholds or EDD_THRESHOLDS
    failures: list[str] = []

    for metric, threshold in thresholds.items():
        score = candidate_scorecard.get(metric)
        if score is None:
            failures.append(f"{metric}: missing (required >= {threshold})")
        elif score < threshold:
            failures.append(f"{metric}: {score:.3f} below threshold {threshold}")

    if baseline_scorecard:
        for metric, baseline in baseline_scorecard.items():
            if metric in _LOWER_IS_BETTER or not isinstance(baseline, (int, float)):
                continue
            score = candidate_scorecard.get(metric)
            if score is not None and score < baseline - tolerance:
                failures.append(
                    f"{metric}: {score:.3f} regresses baseline {baseline:.3f} "
                    f"by more than {tolerance}"
                )

    cost = candidate_scorecard.get("cost_per_query_usd")
    if cost is not None and cost > max_cost_per_query_usd:
        failures.append(
            f"cost_per_query_usd: ${cost:.4f} above cap ${max_cost_per_query_usd:.4f}"
        )

    for metric in _LATENCY_REGRESSIBLE:
        candidate_latency = candidate_scorecard.get(metric)
        baseline_latency = (baseline_scorecard or {}).get(metric)
        if (
            candidate_latency is not None
            and baseline_latency is not None
            and baseline_latency > 0
            and candidate_latency > baseline_latency * (1 + max_latency_regression)
        ):
            failures.append(
                f"{metric}: {candidate_latency:.0f} regresses baseline "
                f"{baseline_latency:.0f} by more than {max_latency_regression:.0%}"
            )

    for metric, required in REQUIRED_BEHAVIOR_RATES.items():
        score = candidate_scorecard.get(metric)
        if score is not None and score < required:
            failures.append(
                f"{metric}: {score:.3f} below required {required:.2f}"
            )

    return GateResult(allowed=not failures, failures=failures)
