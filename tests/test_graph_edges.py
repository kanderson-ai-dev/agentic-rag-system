"""Unit tests for the Self-RAG conditional edges.

These exercise pure functions of `GraphState`, so no LLM, retriever or
graph execution is involved.
"""

from app.graph.edges import (
    make_decide_to_generate,
    route_after_guardrail,
    route_after_human_review,
)


def _state(**overrides):
    base = {
        "question": "what is self-rag?",
        "generation": "",
        "documents": [],
        "web_search_needed": False,
        "retry_count": 0,
        "human_decision": None,
        "blocked": False,
        "output_flagged": False,
    }
    base.update(overrides)
    return base


class TestRouteAfterGuardrail:
    def test_routes_to_retrieve_when_not_blocked(self) -> None:
        assert route_after_guardrail(_state(blocked=False)) == "retrieve"

    def test_routes_to_error_output_when_blocked(self) -> None:
        assert route_after_guardrail(_state(blocked=True)) == "error_output"


class TestDecideToGenerate:
    def test_generates_when_documents_are_sufficient(self) -> None:
        decide = make_decide_to_generate(max_retries=2)
        assert decide(_state(web_search_needed=False)) == "generate"

    def test_retries_when_documents_insufficient_and_budget_remains(self) -> None:
        decide = make_decide_to_generate(max_retries=2)
        assert decide(_state(web_search_needed=True, retry_count=1)) == "transform_query"

    def test_escalates_to_human_when_retries_exhausted(self) -> None:
        decide = make_decide_to_generate(max_retries=2)
        assert decide(_state(web_search_needed=True, retry_count=2)) == "human_review"

    def test_never_loops_forever_regardless_of_retry_count(self) -> None:
        """Even a runaway retry_count must terminate the loop deterministically."""
        decide = make_decide_to_generate(max_retries=2)
        assert decide(_state(web_search_needed=True, retry_count=999)) == "human_review"


class TestRouteAfterHumanReview:
    def test_override_routes_to_output_guardrail(self) -> None:
        assert route_after_human_review(_state(human_decision="override")) == "output_guardrail"

    def test_retry_goes_back_to_retrieve(self) -> None:
        assert route_after_human_review(_state(human_decision="retry")) == "retrieve"

    def test_approve_proceeds_to_generate(self) -> None:
        assert route_after_human_review(_state(human_decision="approve")) == "generate"

    def test_missing_decision_defaults_to_generate(self) -> None:
        assert route_after_human_review(_state(human_decision=None)) == "generate"
