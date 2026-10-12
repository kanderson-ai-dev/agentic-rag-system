"""Metric gate for prompt promotion (Evaluation Driven Development).

A candidate can only be promoted when its A/B scorecard (a) meets every EDD
threshold and (b) does not regress any metric against the currently promoted
version's scorecard by more than a small tolerance. The gate is a pure
function so it is trivially testable and reusable by the A/B engine and the
registry API alike.
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


@dataclass
class GateResult:
    allowed: bool
    failures: list[str] = field(default_factory=list)


def evaluate_promotion(
    candidate_scorecard: dict[str, float],
    baseline_scorecard: dict[str, float] | None,
    thresholds: dict[str, float] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> GateResult:
    """Decide whether ``candidate_scorecard`` may replace the promoted prompt.

    ``baseline_scorecard`` is the promoted version's recorded A/B scorecard;
    ``None`` (first-ever promotion) skips the regression check. Failure strings
    name the metric and the numbers so they are safe to return in a 409 body.
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
            score = candidate_scorecard.get(metric)
            if score is not None and score < baseline - tolerance:
                failures.append(
                    f"{metric}: {score:.3f} regresses baseline {baseline:.3f} "
                    f"by more than {tolerance}"
                )

    return GateResult(allowed=not failures, failures=failures)
