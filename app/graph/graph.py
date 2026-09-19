"""Compiled Self-RAG LangGraph workflow.

The graph shape is:

    START -> retrieve -> grade_documents --(enough docs)--> generate -> END
                              |--(retry available)--> transform_query -> retrieve
                              '--(retries exhausted)--> human_review
                                       |--(approve)--> generate -> END
                                       |--(retry)-----> retrieve
                                       '--(override)--> END
"""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.config import Settings
from app.graph.edges import make_decide_to_generate, route_after_human_review
from app.graph.nodes import (
    Invokable,
    make_generate_node,
    make_grade_documents_node,
    make_human_review_node,
    make_retrieve_node,
    make_transform_query_node,
)
from app.graph.state import GraphState


def build_graph(
    retriever: Invokable,
    grader_chain: Invokable,
    generation_chain: Invokable,
    rewriter_chain: Invokable,
    checkpointer: BaseCheckpointSaver,
    max_retries: int = 2,
) -> CompiledStateGraph:
    """Wire the Self-RAG nodes and edges into a compiled, runnable graph."""
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", make_retrieve_node(retriever))
    workflow.add_node("grade_documents", make_grade_documents_node(grader_chain))
    workflow.add_node("generate", make_generate_node(generation_chain))
    workflow.add_node("transform_query", make_transform_query_node(rewriter_chain))
    workflow.add_node("human_review", make_human_review_node())

    workflow.add_edge(START, "retrieve")
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
        {"generate": "generate", "retrieve": "retrieve", END: END},
    )
    workflow.add_edge("generate", END)

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
    }


def build_default_graph(
    settings: Settings, checkpointer: BaseCheckpointSaver
) -> CompiledStateGraph:
    """Build the production graph, wired with real LLM and vector store."""
    from app.services.llm import (
        build_chat_model,
        build_generation_chain,
        build_grader_chain,
        build_rewriter_chain,
    )
    from app.services.vector_store import build_vector_store_service

    llm = build_chat_model(settings)
    vector_store_service = build_vector_store_service(settings)

    return build_graph(
        retriever=vector_store_service.as_retriever(top_k=settings.retriever_top_k),
        grader_chain=build_grader_chain(llm),
        generation_chain=build_generation_chain(llm),
        rewriter_chain=build_rewriter_chain(llm),
        checkpointer=checkpointer,
        max_retries=settings.max_retries,
    )
