"""Compiled Self-RAG LangGraph workflow.

The graph shape is:

    START -> guardrail --(blocked)--> error_output -> END
                      '--(safe)-----> retrieve -> grade_documents
                                           |--(enough docs)------> generate
                                           |--(retry available)---> transform_query -> retrieve
                                           '--(retries exhausted)--> human_review
                                                    |--(approve)--> generate
                                                    |--(retry)----> retrieve
                                                    '--(override)--> output_guardrail
    generate -> output_guardrail -> END

The `retrieve` node performs hybrid retrieval: vector search (Pinecone/Chroma)
combined with graph search (Neo4j/NetworkX).
"""

import contextlib
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.config import Settings
from app.core.cost_tracking import TokenUsageCallbackHandler
from app.graph.edges import (
    make_decide_to_generate,
    route_after_guardrail,
    route_after_human_review,
)
from app.graph.nodes import (
    Invokable,
    make_error_output_node,
    make_generate_node,
    make_grade_documents_node,
    make_guardrail_node,
    make_human_review_node,
    make_hybrid_retrieve_node,
    make_output_guardrail_node,
    make_transform_query_node,
)
from app.graph.state import GraphState


def build_graph(
    retriever: Invokable,
    grader_chain: Invokable,
    generation_chain: Invokable,
    rewriter_chain: Invokable,
    graph_search_fn: Callable[[str], list[Document]],
    checkpointer: BaseCheckpointSaver[Any],
    max_retries: int = 2,
) -> CompiledStateGraph[Any, None, Any, Any]:
    """Wire the Self-RAG nodes and edges into a compiled, runnable graph."""
    workflow = StateGraph(GraphState)

    workflow.add_node("guardrail", make_guardrail_node())
    workflow.add_node("error_output", make_error_output_node())
    workflow.add_node("retrieve", make_hybrid_retrieve_node(retriever, graph_search_fn))
    workflow.add_node(
        "grade_documents", make_grade_documents_node(grader_chain, generation_chain)
    )
    workflow.add_node("generate", make_generate_node(generation_chain))
    workflow.add_node("transform_query", make_transform_query_node(rewriter_chain))
    workflow.add_node("human_review", make_human_review_node())
    workflow.add_node("output_guardrail", make_output_guardrail_node())

    workflow.add_edge(START, "guardrail")
    workflow.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"error_output": "error_output", "retrieve": "retrieve"},
    )
    workflow.add_edge("error_output", END)
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        make_decide_to_generate(max_retries),
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "human_review": "human_review",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "generate": "generate",
            "retrieve": "retrieve",
            "output_guardrail": "output_guardrail",
        },
    )
    workflow.add_edge("generate", "output_guardrail")
    workflow.add_edge("output_guardrail", END)

    return workflow.compile(checkpointer=checkpointer)


def initial_state(question: str) -> dict[str, Any]:
    """Build the initial state dict for a fresh graph invocation."""
    return {
        "question": question,
        "generation": "",
        "documents": [],
        "web_search_needed": False,
        "retry_count": 0,
        "human_decision": None,
        "blocked": False,
        "output_flagged": False,
        "speculative_generation": "",
        "speculative_context": "",
        "speculative_question": "",
    }


def warmup_backends(
    retriever: Invokable,
    graph_search_fn: Callable[[str], list[Document]],
    llm: BaseChatModel,
) -> None:
    """Pre-open backend connections in a daemon thread at startup.

    The first real query otherwise pays the full connection-setup cost: a
    sleeping Neo4j Aura free-tier instance takes seconds to wake, and the
    first OpenAI/Pinecone calls pay TLS handshake and pool setup. Firing
    cheap calls in the background moves that latency off the request path.
    Best-effort: failures are ignored since the request path retries anyway.
    """

    def _run() -> None:
        calls = (
            lambda: retriever.invoke("warmup"),
            lambda: graph_search_fn("warmup"),
            lambda: llm.invoke(
                "ping",
                config={"tags": ["warmup"], "metadata": {"purpose": "connection-warmup"}},
            ),
        )
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            futures = [pool.submit(call) for call in calls]
            for future in futures:
                with contextlib.suppress(Exception):  # warmup is best-effort
                    future.result()

    threading.Thread(target=_run, daemon=True, name="backend-warmup").start()


def build_default_graph(
    settings: Settings,
    checkpointer: BaseCheckpointSaver[Any],
    token_handler: TokenUsageCallbackHandler | None = None,
) -> CompiledStateGraph[Any, None, Any, Any]:
    """Build the production graph, wired with real LLM and hybrid retrieval."""
    from app.services.graph_store import build_graph_store_service
    from app.services.llm import (
        build_chat_model,
        build_generation_chain,
        build_grader_chain,
        build_rewriter_chain,
    )
    from app.services.vector_store import build_vector_store_service

    llm = build_chat_model(settings, token_handler)
    vector_store_service = build_vector_store_service(settings)
    graph_store_service = build_graph_store_service(settings)

    retriever = vector_store_service.as_retriever(top_k=settings.retriever_top_k)
    graph_search_fn = graph_store_service.graph_search
    warmup_backends(retriever, graph_search_fn, llm)

    return build_graph(
        retriever=retriever,
        grader_chain=build_grader_chain(llm),
        generation_chain=build_generation_chain(llm),
        rewriter_chain=build_rewriter_chain(llm),
        graph_search_fn=graph_search_fn,
        checkpointer=checkpointer,
        max_retries=settings.max_retries,
    )
