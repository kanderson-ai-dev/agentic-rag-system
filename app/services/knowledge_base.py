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
            "instead of generating an answer from weak context. When retries "
            "are exhausted, the system escalates to human-in-the-loop review "
            "rather than hallucinating from weak context."
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
            "tests and short-lived scripts. LangGraph can work without a "
            "checkpointer for single-execution workflows, but checkpointer "
            "is required for human-in-the-loop patterns and state persistence."
        ),
        metadata={"source": "langgraph-checkpointing", "topic": "checkpointing"},
    ),
    Document(
        page_content=(
            "LangGraph and Self-RAG are related in that LangGraph provides the "
            "orchestration framework to implement the Self-RAG pattern. Self-RAG "
            "is a specific retrieval-augmented generation approach that can be "
            "built using LangGraph's graph-based workflow system, nodes, edges, "
            "and state management capabilities."
        ),
        metadata={"source": "langgraph-selfrag-relationship", "topic": "langgraph"},
    ),
    Document(
        page_content=(
            "When Self-RAG exhausts its retry attempts without finding sufficient "
            "context, it escalates to human-in-the-loop review instead of generating "
            "an answer from weak context. This prevents hallucinations by refusing "
            "to answer when the retrieved documents are inadequate, allowing a human "
            "reviewer to provide guidance or override the system's decision."
        ),
        metadata={"source": "self-rag-escalation", "topic": "self-rag"},
    ),
]
