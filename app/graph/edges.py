"""Conditional edges for the Self-RAG graph.

Both routing functions are pure functions of `GraphState`, which makes the
correction-loop and human-in-the-loop logic trivial to unit test without any
LLM calls or graph execution.
"""

from langgraph.graph import END

from app.graph.state import GraphState


def make_decide_to_generate(max_retries: int):
    """Route after grading: generate, retry, or escalate to a human.

    - Enough relevant documents -> generate the answer.
    - Not enough documents, but retries remain -> rewrite the query and
      retrieve again.
    - Not enough documents and retries exhausted -> escalate to a human
      reviewer instead of generating from weak context or looping forever.
    """

    def decide_to_generate(state: GraphState) -> str:
        if not state["web_search_needed"]:
            return "generate"
        if state["retry_count"] < max_retries:
            return "transform_query"
        return "human_review"

    return decide_to_generate


def route_after_human_review(state: GraphState) -> str:
    """Route after a human resolves an escalation.

    - "override": the human supplied the final answer directly -> end.
    - "retry": the human supplied a revised question -> retrieve again.
    - "approve" (default): generate with the documents currently held.
    """
    decision = state.get("human_decision")
    if decision == "override":
        return END
    if decision == "retry":
        return "retrieve"
    return "generate"
