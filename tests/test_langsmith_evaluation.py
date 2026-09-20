"""Tests for the LangSmith evaluation wiring (skipped without credentials)."""

from tests.conftest import requires_langsmith_key, requires_openai_key


@requires_openai_key
@requires_langsmith_key
def test_evaluators_are_wired() -> None:
    from evaluation.evaluators import (
        groundedness_evaluator,
        llm_as_judge_evaluator,
        relevance_evaluator,
    )

    assert callable(llm_as_judge_evaluator)
    assert groundedness_evaluator is not None
    assert relevance_evaluator is not None
