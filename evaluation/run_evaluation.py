"""Run a LangSmith experiment against the compiled graph (LLM-as-judge).

This script uses `langsmith.evaluate()` to produce a real, versioned Experiment
in the LangSmith dashboard (dataset + evaluator runs), falling back to a local
evaluation only if the LangSmith API key is absent or unauthorized.
"""

import json
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import get_settings
from app.graph.graph import build_default_graph, initial_state
from evaluation.dataset import build_dataset
from evaluation.evaluators import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    llm_as_judge_evaluator,
)

DATASET_NAME = "agentic-rag-system-eval"


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


def run_local_evaluation() -> dict:
    """Run evaluation locally without LangSmith dataset creation."""
    cases = build_dataset()
    results = []

    for case in cases:
        result = _target({"question": case["question"]})
        answer = result["answer"]
        context = "\n\n".join(result["contexts"])

        # Compute metrics
        faithfulness_score = faithfulness(context, answer)
        relevancy_score = answer_relevancy(case["question"], answer)
        precision_score = context_precision(case["question"], context)
        recall_score = context_recall(case["question"], context)

        results.append({
            "question": case["question"],
            "ground_truth": case["ground_truth"],
            "answer": answer,
            "faithfulness": faithfulness_score,
            "answer_relevancy": relevancy_score,
            "context_precision": precision_score,
            "context_recall": recall_score,
        })

    # Calculate averages
    if results:
        avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
        avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)
        avg_precision = sum(r["context_precision"] for r in results) / len(results)
        avg_recall = sum(r["context_recall"] for r in results) / len(results)
    else:
        avg_faithfulness = avg_relevancy = avg_precision = avg_recall = 0.0

    return {
        "averages": {
            "faithfulness": round(avg_faithfulness, 4),
            "answer_relevancy": round(avg_relevancy, 4),
            "context_precision": round(avg_precision, 4),
            "context_recall": round(avg_recall, 4),
        },
        "individual_results": results,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _get_or_create_dataset(client, name: str, description: str):
    """Return the dataset, creating it (and its examples) only the first time."""
    from langsmith.utils import LangSmithConflictError, LangSmithNotFoundError

    try:
        return client.read_dataset(dataset_name=name), False
    except LangSmithNotFoundError:
        pass

    try:
        dataset = client.create_dataset(dataset_name=name, description=description)
        return dataset, True
    except LangSmithConflictError:
        return client.read_dataset(dataset_name=name), False


def _rag_evaluators() -> list:
    """Evaluators registered with `langsmith.evaluate()` for this experiment."""

    def _faithfulness(run, example) -> dict:
        return {
            "key": "faithfulness",
            "score": faithfulness(
                "\n\n".join(run.outputs.get("contexts", [])), run.outputs.get("answer", "")
            ),
        }

    def _relevancy(run, example) -> dict:
        return {
            "key": "answer_relevancy",
            "score": answer_relevancy(
                run.inputs.get("question", ""), run.outputs.get("answer", "")
            ),
        }

    def _precision(run, example) -> dict:
        return {
            "key": "context_precision",
            "score": context_precision(
                run.inputs.get("question", ""), "\n\n".join(run.outputs.get("contexts", []))
            ),
        }

    def _recall(run, example) -> dict:
        return {
            "key": "context_recall",
            "score": context_recall(
                run.inputs.get("question", ""), "\n\n".join(run.outputs.get("contexts", []))
            ),
        }

    def _judge(run, example) -> dict:
        return llm_as_judge_evaluator(run, example)

    return [_faithfulness, _relevancy, _precision, _recall, _judge]


def try_langsmith_evaluation() -> dict | None:
    """Run a real LangSmith Experiment (dataset + evaluate()), falling back to a
    local evaluation only if LangSmith is unavailable or unauthorized.
    """
    try:
        from langsmith import Client, evaluate
    except ImportError:
        print("LangSmith not installed. Running local evaluation.")
        return run_local_evaluation()

    settings = get_settings()
    settings.configure_langsmith_env()

    if not settings.langsmith_api_key:
        print("LANGCHAIN_API_KEY not configured. Running local evaluation instead.")
        return run_local_evaluation()

    client = Client()

    try:
        list(client.list_datasets(limit=1))
    except Exception as e:
        print(f"LangSmith API key lacks permissions ({e}). Running local evaluation instead.")
        return run_local_evaluation()

    try:
        dataset, created = _get_or_create_dataset(
            client,
            DATASET_NAME,
            description="Curated questions for the Agentic RAG system (LangGraph, Self-RAG, "
            "HITL, FastAPI, checkpointing), including an out-of-domain case.",
        )
        if created:
            for case in build_dataset():
                client.create_example(
                    inputs={"question": case["question"]},
                    outputs={"answer": case["ground_truth"]},
                    dataset_id=dataset.id,
                )

        results = evaluate(
            _target,
            data=dataset.name,
            evaluators=_rag_evaluators(),
            experiment_prefix="agentic-rag-system",
            description="Self-RAG hybrid retrieval (Pinecone/Chroma + Neo4j/NetworkX) "
            "evaluated with RAGAS-style LLM-as-judge metrics.",
            max_concurrency=2,
        )
        print(f"LangSmith experiment created: {results.experiment_name}")
        print(f"View it at: {results.url}")
        return None
    except Exception as e:
        print(f"LangSmith evaluate() failed: {e}. Running local evaluation instead.")
        return run_local_evaluation()


def main() -> None:
    print("Running LLM-as-judge evaluation...")
    result = try_langsmith_evaluation()

    if result:
        # Save local results
        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
        output_file = results_dir / f"llm_judge_evaluation_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"\nLocal evaluation results saved to: {output_file}")
        print("\nAverage Scores:")
        for metric, score in result["averages"].items():
            print(f"  {metric}: {score}")

        print("\nIndividual Results:")
        for i, r in enumerate(result["individual_results"], 1):
            print(f"\n{i}. Question: {r['question']}")
            print(f"   Answer: {r['answer'][:100]}...")
            print(f"   Faithfulness: {r['faithfulness']}, Relevancy: {r['answer_relevancy']}")


if __name__ == "__main__":
    main()
