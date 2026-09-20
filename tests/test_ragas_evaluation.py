"""Tests for the evaluation dataset and RAGAS-style scorecard."""

from evaluation.dataset import SAMPLE_CASES, build_dataset
from tests.conftest import requires_openai_key


def test_build_dataset_returns_cases() -> None:
    cases = build_dataset()
    assert len(cases) == len(SAMPLE_CASES)
    for case in cases:
        assert case["question"]
        assert case["ground_truth"]


def test_dataset_contains_out_of_domain_case() -> None:
    questions = [case["question"] for case in build_dataset()]
    assert any("capital of France" in q for q in questions)


@requires_openai_key
def test_compute_scorecard_returns_metrics() -> None:
    from evaluation.run_ragas import compute_scorecard

    scorecard = compute_scorecard()

    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        assert metric in scorecard
        assert 0.0 <= scorecard[metric] <= 1.0


def test_metric_prompts_are_defined() -> None:
    # Ensure the evaluators module imports and exposes the four metric functions.
    from evaluation.evaluators import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    assert callable(faithfulness)
    assert callable(answer_relevancy)
    assert callable(context_precision)
    assert callable(context_recall)
