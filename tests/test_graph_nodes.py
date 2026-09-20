"""Unit tests for individual Self-RAG node factories, using test doubles."""

import app.graph.nodes as nodes_module
from app.graph.nodes import (
    make_generate_node,
    make_grade_documents_node,
    make_human_review_node,
    make_retrieve_node,
    make_transform_query_node,
)
from tests.conftest import (
    IRRELEVANT_DOC,
    RELEVANT_DOC,
    FakeGenerationChain,
    FakeGrader,
    FakeRetriever,
    FakeRewriterChain,
)


def _state(**overrides):
    base = {
        "question": "what is langgraph?",
        "generation": "",
        "documents": [],
        "web_search_needed": False,
        "retry_count": 0,
        "human_decision": None,
    }
    base.update(overrides)
    return base


def test_retrieve_node_returns_retriever_documents() -> None:
    retriever = FakeRetriever([RELEVANT_DOC])
    node = make_retrieve_node(retriever)

    result = node(_state())

    assert result["documents"] == [RELEVANT_DOC]
    assert retriever.calls == ["what is langgraph?"]


def test_grade_documents_node_filters_out_irrelevant_documents() -> None:
    node = make_grade_documents_node(FakeGrader())

    result = node(_state(documents=[RELEVANT_DOC, IRRELEVANT_DOC]))

    assert result["documents"] == [RELEVANT_DOC]
    assert result["web_search_needed"] is False


def test_grade_documents_node_flags_web_search_needed_when_nothing_relevant() -> None:
    node = make_grade_documents_node(FakeGrader())

    result = node(_state(documents=[IRRELEVANT_DOC]))

    assert result["documents"] == []
    assert result["web_search_needed"] is True


def test_generate_node_uses_document_context() -> None:
    node = make_generate_node(FakeGenerationChain())

    result = node(_state(documents=[RELEVANT_DOC]))

    assert result["generation"].startswith("Answer based on:")


def test_transform_query_node_rewrites_question_and_increments_retry_count() -> None:
    node = make_transform_query_node(FakeRewriterChain())

    result = node(_state(question="original question", retry_count=1))

    assert result["question"] == "original question (rewritten)"
    assert result["retry_count"] == 2


class TestHumanReviewNode:
    """`interrupt()` only works inside a running, checkpointed graph (see
    `test_graph_integration.py` for the end-to-end pause/resume behavior).
    Here we monkeypatch it to unit test the surfaced payload and the
    state updates applied for each resume decision, in isolation.
    """

    def test_surfaces_expected_payload_to_interrupt(self, monkeypatch) -> None:
        captured: dict = {}

        def fake_interrupt(value):
            captured.update(value)
            return {"decision": "approve"}

        monkeypatch.setattr(nodes_module, "interrupt", fake_interrupt)
        node = make_human_review_node()

        node(_state(question="hard question", retry_count=2, documents=[RELEVANT_DOC]))

        assert captured == {
            "reason": "max_retries_exhausted",
            "question": "hard question",
            "retry_count": 2,
            "best_documents": [RELEVANT_DOC.page_content],
        }

    def test_approve_decision_keeps_documents_for_generation(self, monkeypatch) -> None:
        monkeypatch.setattr(nodes_module, "interrupt", lambda value: {"decision": "approve"})
        node = make_human_review_node()

        result = node(_state())

        assert result == {"human_decision": "approve"}

    def test_override_decision_sets_generation_directly(self, monkeypatch) -> None:
        monkeypatch.setattr(
            nodes_module,
            "interrupt",
            lambda value: {"decision": "override", "override_answer": "human-provided answer"},
        )
        node = make_human_review_node()

        result = node(_state())

        assert result == {"human_decision": "override", "generation": "human-provided answer"}

    def test_retry_decision_resets_question_and_retry_count(self, monkeypatch) -> None:
        monkeypatch.setattr(
            nodes_module,
            "interrupt",
            lambda value: {"decision": "retry", "revised_question": "a clearer question"},
        )
        node = make_human_review_node()

        result = node(_state(question="original question", retry_count=2))

        assert result == {
            "human_decision": "retry",
            "question": "a clearer question",
            "retry_count": 0,
            "web_search_needed": False,
        }
