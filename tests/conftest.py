"""Shared pytest fixtures.

All fixtures here are deterministic stand-ins for LLM chains and retrievers:
none of them perform network calls, so the suite never requires
`OPENAI_API_KEY` or `LANGSMITH_API_KEY` to pass.
"""

from typing import Any

import pytest
from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.graph import build_graph

RELEVANT_DOC = Document(
    page_content="LangGraph orchestrates stateful multi-actor LLM applications.",
    metadata={"source": "langgraph-overview"},
)
IRRELEVANT_DOC = Document(
    page_content="Bananas are a good source of potassium.",
    metadata={"source": "nutrition-facts"},
)


class FakeRetriever:
    """Returns a fixed set of documents regardless of the query."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.calls: list[str] = []

    def invoke(self, question: str) -> list[Document]:
        self.calls.append(question)
        return self.documents


class FakeSequentialRetriever:
    """Returns a different fixed batch of documents on each successive call.

    Useful to simulate retrieval quality improving after a query rewrite.
    """

    def __init__(self, batches: list[list[Document]]) -> None:
        self.batches = batches
        self.calls: list[str] = []

    def invoke(self, question: str) -> list[Document]:
        index = min(len(self.calls), len(self.batches) - 1)
        self.calls.append(question)
        return self.batches[index]


class FakeGrader:
    """Grades documents based on a keyword contained in their content."""

    def __init__(self, relevant_keyword: str = "LangGraph") -> None:
        self.relevant_keyword = relevant_keyword

    def invoke(self, inputs: dict[str, Any]) -> dict[str, str]:
        score = "yes" if self.relevant_keyword in inputs["document"] else "no"
        return {"binary_score": score}


class FakeGenerationChain:
    """Returns a deterministic answer derived from the provided context."""

    def invoke(self, inputs: dict[str, Any]) -> str:
        return f"Answer based on: {inputs['context'][:40]}"


class FakeRewriterChain:
    """Appends a marker to the question to simulate a rewrite."""

    def invoke(self, inputs: dict[str, Any]) -> str:
        return f"{inputs['question']} (rewritten)"


@pytest.fixture
def relevant_doc() -> Document:
    return RELEVANT_DOC


@pytest.fixture
def irrelevant_doc() -> Document:
    return IRRELEVANT_DOC


@pytest.fixture
def fake_generation_chain() -> FakeGenerationChain:
    return FakeGenerationChain()


@pytest.fixture
def fake_rewriter_chain() -> FakeRewriterChain:
    return FakeRewriterChain()


@pytest.fixture
def fake_grader() -> FakeGrader:
    return FakeGrader()


def build_test_graph(
    documents: list[Document] | None = None,
    *,
    retriever: Any | None = None,
    max_retries: int = 2,
    relevant_keyword: str = "LangGraph",
):
    """Build a fully-stubbed, deterministic Self-RAG graph for tests."""
    return build_graph(
        retriever=retriever if retriever is not None else FakeRetriever(documents or []),
        grader_chain=FakeGrader(relevant_keyword),
        generation_chain=FakeGenerationChain(),
        rewriter_chain=FakeRewriterChain(),
        checkpointer=InMemorySaver(),
        max_retries=max_retries,
    )
