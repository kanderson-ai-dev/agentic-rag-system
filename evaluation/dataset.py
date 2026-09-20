"""Evaluation dataset: curated questions with ground-truth answers.

Each case targets a concept in the demo corpus (LangGraph, Self-RAG, HITL,
FastAPI, checkpointing) so the evaluation metrics are meaningful rather than
artificially easy.
"""

SAMPLE_CASES: list[dict[str, str]] = [
    {
        "question": "What is LangGraph?",
        "ground_truth": (
            "LangGraph is a low-level orchestration framework for building stateful, "
            "multi-actor applications with large language models, modeling workflows "
            "as graphs of nodes and edges."
        ),
    },
    {
        "question": "How does Self-RAG decide whether to generate an answer?",
        "ground_truth": (
            "Self-RAG critiques its retrieved context before generating. If documents "
            "are graded as irrelevant or insufficient, it rewrites the query and retries "
            "retrieval instead of answering from weak context."
        ),
    },
    {
        "question": "What does the interrupt() function do in LangGraph?",
        "ground_truth": (
            "interrupt() pauses graph execution and persists state via a checkpointer; "
            "the graph resumes later when invoked again with a Command carrying a resume value."
        ),
    },
    {
        "question": "What is the role of a checkpointer?",
        "ground_truth": (
            "A checkpointer persists graph state between invocations keyed by a thread_id, "
            "making execution durable across process restarts."
        ),
    },
    {
        "question": "How does LangSmith support evaluation-driven development?",
        "ground_truth": (
            "LangSmith lets teams version datasets, define code or LLM-as-judge evaluators, "
            "and run evaluate() to produce comparable experiments over time."
        ),
    },
    {
        "question": "What does FastAPI provide out of the box?",
        "ground_truth": (
            "FastAPI provides automatic request validation, dependency injection, and "
            "OpenAPI documentation generation on top of Starlette and Pydantic."
        ),
    },
    {
        "question": "What is the capital of France?",  # out-of-domain on purpose
        "ground_truth": (
            "The retrieved context does not contain the answer, so the system should say "
            "it does not know rather than hallucinate."
        ),
    },
]


def build_dataset() -> list[dict[str, str]]:
    """Return a copy of the curated dataset."""
    return [dict(case) for case in SAMPLE_CASES]
