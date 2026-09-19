"""Seed corpus for the demo knowledge base.

In a production deployment this would be replaced by an ingestion pipeline
(document loaders, chunking, embeddings) feeding a persistent vector store.
For this demo, a small curated corpus about agentic AI concepts is enough
to exercise the full Self-RAG correction loop end to end.
"""

from langchain_core.documents import Document

SAMPLE_DOCUMENTS: list[Document] = [
    Document(
        page_content=(
            "LangGraph is a low-level orchestration framework for building "
            "stateful, multi-actor applications with large language models. "
            "It models workflows as graphs of nodes and edges, where each "
            "node can read and write a shared state object."
        ),
        metadata={"source": "langgraph-overview", "topic": "langgraph"},
    ),
    Document(
        page_content=(
            "Self-RAG is a retrieval-augmented generation pattern where the "
            "system critiques its own retrieved context before generating an "
            "answer. If the retrieved documents are graded as irrelevant or "
            "insufficient, the query is rewritten and retrieval is retried "
            "instead of generating an answer from weak context."
        ),
        metadata={"source": "self-rag-pattern", "topic": "self-rag"},
    ),
    Document(
        page_content=(
            "Human-in-the-loop (HITL) patterns in LangGraph rely on the "
            "interrupt() function together with a checkpointer. Calling "
            "interrupt() pauses graph execution and persists the state; the "
            "graph resumes later when the caller invokes it again with a "
            "Command(resume=...) carrying the human's decision."
        ),
        metadata={"source": "langgraph-hitl", "topic": "human-in-the-loop"},
    ),
    Document(
        page_content=(
            "Evaluation-driven development treats evaluation datasets and "
            "metrics as first-class artifacts of the software lifecycle. "
            "LangSmith supports this by letting teams version datasets, "
            "define code or LLM-as-judge evaluators, and run evaluate() to "
            "produce comparable experiments over time."
        ),
        metadata={"source": "langsmith-evaluation", "topic": "evaluation"},
    ),
    Document(
        page_content=(
            "FastAPI is a modern Python web framework built on top of "
            "Starlette and Pydantic. It provides automatic request "
            "validation, dependency injection, and OpenAPI documentation "
            "generation out of the box."
        ),
        metadata={"source": "fastapi-overview", "topic": "fastapi"},
    ),
    Document(
        page_content=(
            "A checkpointer in LangGraph persists the graph state between "
            "invocations, keyed by a thread_id. SqliteSaver stores this "
            "state in a local SQLite file, which is durable across process "
            "restarts, unlike the in-memory InMemorySaver used only for "
            "tests and short-lived scripts."
        ),
        metadata={"source": "langgraph-checkpointing", "topic": "checkpointing"},
    ),
]
