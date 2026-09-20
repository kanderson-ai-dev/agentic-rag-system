"""Run a LangSmith experiment against the compiled graph (LLM-as-judge)."""

from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver
from langsmith import Client, evaluate

from app.core.config import get_settings
from app.graph.graph import build_default_graph, initial_state
from evaluation.dataset import build_dataset
from evaluation.evaluators import (
    groundedness_evaluator,
    llm_as_judge_evaluator,
    relevance_evaluator,
)


@lru_cache
def _graph():
    settings = get_settings()
    settings.configure_langsmith_env()
    return build_default_graph(settings, InMemorySaver())


def _target(inputs: dict) -> dict:
    result = _graph().invoke(
        initial_state(inputs["question"]),
        config={"configurable": {"thread_id": "eval"}},
    )
    return {
        "answer": result.get("generation", ""),
        "contexts": [doc.page_content for doc in result.get("documents", [])],
    }


def main() -> None:
    client = Client()
    dataset = client.create_dataset(
        dataset_name="agentic-rag-evaluation",
        description="Curated questions for the Agentic RAG system.",
    )
    for case in build_dataset():
        client.create_example(
            inputs={"question": case["question"]},
            outputs={"answer": case["ground_truth"]},
            dataset_id=dataset.id,
        )

    evaluate(
        _target,
        data=dataset.name,
        evaluators=[llm_as_judge_evaluator, groundedness_evaluator, relevance_evaluator],
        experiment_prefix="agentic-rag-system",
    )


if __name__ == "__main__":
    main()
