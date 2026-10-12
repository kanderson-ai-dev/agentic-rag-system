"""Run a provider-backed experiment against the compiled graph (LLM-as-judge).

The provider is selected by ``TRACING_PROVIDER``:

- ``langsmith`` — `langsmith.evaluate()` produces a versioned Experiment in
  the LangSmith dashboard (dataset
  ``agentic-rag-system-eval-v<dataset_version>``, experiment prefix carrying
  the prompt versions that ran).
- ``langfuse`` — the same versioned dataset and one trace per item land in
  Langfuse via ``dataset.run_experiment()`` with evaluator scores attached.
- ``none`` — local harness scorecard only (default for daily development).

Every provider path falls back to the local harness scorecard when its
backend is unavailable — the local scorecard stays the promotion-gate
source of truth either way.

CLI flags mirror `run_ragas.py`: ``--prompt-version NAME=N`` (repeatable),
``--retrieval-mode``, ``--max-cost-usd``, ``--output``.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from app.core import tracing
from app.core.config import get_settings
from app.core.cost_tracking import TokenUsageCallbackHandler, calculate_cost_usd
from app.graph.graph import build_default_graph, initial_state
from app.graph.prompts import REFUSAL_TEXT, active_versions, pin_versions
from evaluation import evaluators
from evaluation.dataset import build_dataset, dataset_version
from evaluation.evaluators import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    llm_as_judge_evaluator,
)
from evaluation.harness import (
    CostCapExceeded,
    git_sha,
    parse_prompt_version_overrides,
    prompt_version_tag,
    run_cases,
    run_metadata,
    score_results,
)


def dataset_name() -> str:
    """LangSmith dataset name, versioned by dataset content hash."""
    return f"agentic-rag-system-eval-v{dataset_version()}"


@lru_cache
def _graph(token_handler: TokenUsageCallbackHandler | None = None):
    settings = get_settings()
    settings.configure_langsmith_env()
    return build_default_graph(settings, InMemorySaver(), token_handler)


def _target(inputs: dict) -> dict:
    result = _graph().invoke(
        initial_state(inputs["question"]),
        config={"configurable": {"thread_id": "eval"}},
    )
    return {
        "answer": result.get("generation", ""),
        "contexts": [doc.page_content for doc in result.get("documents", [])],
        "blocked": bool(result.get("blocked", False)),
        "escalated": bool(result.get("__interrupt__")),
    }


def run_local_evaluation(
    *,
    prompt_versions: dict[str, int] | None = None,
    max_cost_usd: float | None = None,
) -> dict:
    """Run evaluation locally without LangSmith dataset creation."""
    settings = get_settings()
    settings.configure_langsmith_env()

    token_handler = TokenUsageCallbackHandler()
    evaluators.set_token_handler(token_handler)
    pin_versions(prompt_versions)
    try:
        cases = build_dataset()
        results = run_cases(
            _graph(token_handler),
            cases,
            token_handler=token_handler,
            max_cost_usd=max_cost_usd,
            settings=settings,
        )
        scorecard = score_results(
            results,
            evaluators.default_scorers(),
            token_handler=token_handler,
            settings=settings,
        )
    finally:
        pin_versions(None)
        evaluators.set_token_handler(None)

    scorecard["meta"] = run_metadata(settings)
    return scorecard


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

    def _behavior(run, example) -> dict:
        """1 when the observed behavior matches the expected_behavior label."""
        expected = (example.metadata or {}).get("expected_behavior", "answer")
        outputs = run.outputs or {}
        if expected == "block_injection":
            ok = bool(outputs.get("blocked"))
        elif expected == "refuse_out_of_domain":
            ok = bool(outputs.get("blocked") or outputs.get("escalated"))
        else:
            ok = not outputs.get("blocked", False)
        return {"key": "expected_behavior", "score": 1.0 if ok else 0.0}

    return [_faithfulness, _relevancy, _precision, _recall, _judge, _behavior]


def try_langsmith_evaluation(
    *,
    prompt_versions: dict[str, int] | None = None,
    max_cost_usd: float | None = None,
) -> dict | None:
    """Run a real LangSmith Experiment (dataset + evaluate()), falling back to a
    local evaluation only if LangSmith is unavailable or unauthorized.
    """
    try:
        from langsmith import Client, evaluate
    except ImportError:
        print("LangSmith not installed. Running local evaluation.")
        return run_local_evaluation(
            prompt_versions=prompt_versions, max_cost_usd=max_cost_usd
        )

    settings = get_settings()
    settings.configure_langsmith_env()

    if not settings.langsmith_api_key:
        print("LANGCHAIN_API_KEY not configured. Running local evaluation instead.")
        return run_local_evaluation(
            prompt_versions=prompt_versions, max_cost_usd=max_cost_usd
        )

    client = Client()

    try:
        list(client.list_datasets(limit=1))
    except Exception as e:
        print(f"LangSmith API key lacks permissions ({e}). Running local evaluation instead.")
        return run_local_evaluation(
            prompt_versions=prompt_versions, max_cost_usd=max_cost_usd
        )

    pin_versions(prompt_versions)
    token_handler = TokenUsageCallbackHandler()
    evaluators.set_token_handler(token_handler)
    try:
        name = dataset_name()
        dataset, created = _get_or_create_dataset(
            client,
            name,
            description="Unified Agentic RAG eval set (single-hop, multi-hop, "
            "out-of-domain, injection, other-language, vague).",
        )
        if created:
            for case in build_dataset():
                client.create_example(
                    inputs={"question": case["question"]},
                    outputs={"answer": case["ground_truth"]},
                    metadata={
                        "id": case["id"],
                        "category": case["category"],
                        "expected_behavior": case["expected_behavior"],
                    },
                    dataset_id=dataset.id,
                )

        tag = prompt_version_tag(active_versions())

        # Build the target over the handler-wired graph so spend is measured;
        # the cap is checked after the run (evaluate() can't abort mid-flight).
        graph = _graph(token_handler)

        def target(inputs: dict) -> dict:
            result = graph.invoke(
                initial_state(inputs["question"]),
                config={"configurable": {"thread_id": "eval"}},
            )
            return {
                "answer": result.get("generation", ""),
                "contexts": [
                    doc.page_content for doc in result.get("documents", [])
                ],
                "blocked": bool(result.get("blocked", False)),
                "escalated": bool(result.get("__interrupt__")),
            }

        results = evaluate(
            target,
            data=name,
            evaluators=_rag_evaluators(),
            experiment_prefix=f"agentic-rag-system-{tag}",
            description=f"prompt_versions={tag}; dataset_version={dataset_version()}",
            max_concurrency=2,
        )
        spent = calculate_cost_usd(
            token_handler.prompt_tokens,
            token_handler.completion_tokens,
            settings.cost_input_price_per_1m,
            settings.cost_output_price_per_1m,
        )
        if max_cost_usd is not None and spent > max_cost_usd:
            print(f"WARNING: run spent ${spent:.4f} > cap ${max_cost_usd:.4f}")
        print(f"LangSmith experiment created: {results.experiment_name}")
        print(f"View it at: {results.url}")
        return None
    except Exception as e:
        print(f"LangSmith evaluate() failed: {e}. Running local evaluation instead.")
        return run_local_evaluation(
            prompt_versions=prompt_versions, max_cost_usd=max_cost_usd
        )
    finally:
        pin_versions(None)
        evaluators.set_token_handler(None)


def _get_or_create_langfuse_dataset(client: object, name: str) -> object:
    """Fetch the Langfuse dataset, creating it (with items) on first use."""
    try:
        return client.get_dataset(name)
    except Exception:  # noqa: BLE001 - any lookup miss falls through to create
        pass

    client.create_dataset(
        name=name,
        description="Unified Agentic RAG eval set (single-hop, multi-hop, "
        "out-of-domain, injection, other-language, vague).",
    )
    for case in build_dataset():
        client.create_dataset_item(
            dataset_name=name,
            input={"question": case["question"]},
            expected_output=case["ground_truth"],
            metadata={
                "id": case["id"],
                "category": case["category"],
                "expected_behavior": case["expected_behavior"],
            },
        )
    return client.get_dataset(name)


def _langfuse_evaluations(
    input: dict,  # noqa: A002 - Langfuse evaluator contract names it `input`
    output: dict,
    expected_output: object,
    metadata: dict | None,
) -> list[dict]:
    """Item-level evaluator for ``dataset.run_experiment()``.

    Mirrors the LangSmith evaluator set: quality metrics on ``answer`` cases
    plus the behavior check on every case, returned as Langfuse scores.
    """
    expected = (metadata or {}).get("expected_behavior", "answer")
    question = (input or {}).get("question", "")
    output = output or {}
    answer = output.get("answer", "")
    context = "\n\n".join(output.get("contexts", []))
    blocked = bool(output.get("blocked"))
    escalated = bool(output.get("escalated"))

    evaluations: list[dict] = []
    if expected == "answer":
        for name, scorer in evaluators.default_scorers().items():
            evaluations.append(
                {
                    "name": name,
                    "value": scorer(question=question, answer=answer, context=context),
                }
            )

    if expected == "block_injection":
        ok = blocked
    elif expected == "refuse_out_of_domain":
        ok = blocked or escalated or REFUSAL_TEXT in answer
    else:
        ok = not blocked
    evaluations.append({"name": "expected_behavior", "value": 1.0 if ok else 0.0})
    return evaluations


def try_langfuse_evaluation(
    *,
    prompt_versions: dict[str, int] | None = None,
    max_cost_usd: float | None = None,
) -> dict | None:
    """Run a real Langfuse experiment, falling back to local when unavailable.

    ``dataset.run_experiment`` creates the dataset run, one trace per item
    and evaluator scores natively; the local scorecard remains the
    promotion-gate source of truth and is produced by the fallback path or
    by `run_ragas.py`.
    """
    settings = get_settings()
    if tracing.init_tracing(settings) != "langfuse":
        print("Langfuse unavailable or misconfigured. Running local evaluation.")
        return run_local_evaluation(
            prompt_versions=prompt_versions, max_cost_usd=max_cost_usd
        )

    try:
        import langfuse

        client = langfuse.get_client()
    except Exception as e:
        print(f"Langfuse client unavailable ({e}). Running local evaluation.")
        return run_local_evaluation(
            prompt_versions=prompt_versions, max_cost_usd=max_cost_usd
        )

    pin_versions(prompt_versions)
    token_handler = TokenUsageCallbackHandler()
    evaluators.set_token_handler(token_handler)
    try:
        dataset = _get_or_create_langfuse_dataset(client, dataset_name())
        graph = _graph(token_handler)
        tag = prompt_version_tag(active_versions())

        def task(*, item: object, **kwargs: object) -> dict:
            result = graph.invoke(
                initial_state(item.input["question"]),
                config={
                    "configurable": {"thread_id": f"eval-{item.id}"},
                    "callbacks": tracing.build_callbacks(settings),
                },
            )
            return {
                "answer": result.get("generation", ""),
                "contexts": [
                    doc.page_content for doc in result.get("documents", [])
                ],
                "blocked": bool(result.get("blocked", False)),
                "escalated": bool(result.get("__interrupt__")),
            }

        def evaluator(
            *, input: dict, output: dict, expected_output: object, metadata: dict, **kwargs: object
        ) -> list[dict]:
            return _langfuse_evaluations(input, output, expected_output, metadata)

        result = dataset.run_experiment(
            # `<prompt_versions>-<git sha>-<date>` so A/B runs compare side by side
            name=f"{tag}-{git_sha()}-{datetime.now(UTC):%Y-%m-%d}",
            task=task,
            evaluators=[evaluator],
            max_concurrency=2,
            metadata={
                "prompt_versions": tag,
                "dataset_version": dataset_version(),
            },
        )
        spent = calculate_cost_usd(
            token_handler.prompt_tokens,
            token_handler.completion_tokens,
            settings.cost_input_price_per_1m,
            settings.cost_output_price_per_1m,
        )
        if max_cost_usd is not None and spent > max_cost_usd:
            print(f"WARNING: run spent ${spent:.4f} > cap ${max_cost_usd:.4f}")
        tracing.flush()
        print(f"Langfuse experiment complete: {result.name}")
        return None
    except Exception as e:
        print(f"Langfuse experiment failed: {e}. Running local evaluation instead.")
        return run_local_evaluation(
            prompt_versions=prompt_versions, max_cost_usd=max_cost_usd
        )
    finally:
        pin_versions(None)
        evaluators.set_token_handler(None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--prompt-version",
        action="append",
        metavar="NAME=N",
        dest="prompt_versions",
        help="pin a prompt version for the run (repeatable)",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("hybrid", "vector_only", "graph_only", "no_retrieval"),
        default="hybrid",
        help="retrieval backend combination (non-hybrid lands in a later phase)",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="abort the run when accumulated spend exceeds this cap",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="local-fallback scorecard path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        pins = parse_prompt_version_overrides(args.prompt_versions)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.retrieval_mode != "hybrid":
        print(
            f"error: retrieval_mode={args.retrieval_mode!r} lands in a later phase",
            file=sys.stderr,
        )
        return 2

    provider = get_settings().tracing_provider
    print(f"Running LLM-as-judge evaluation (provider={provider})...")
    try:
        if provider == "langfuse":
            result = try_langfuse_evaluation(
                prompt_versions=pins or None, max_cost_usd=args.max_cost_usd
            )
        elif provider == "langsmith":
            result = try_langsmith_evaluation(
                prompt_versions=pins or None, max_cost_usd=args.max_cost_usd
            )
        else:
            result = run_local_evaluation(
                prompt_versions=pins or None, max_cost_usd=args.max_cost_usd
            )
    except CostCapExceeded as exc:
        print(f"aborted: {exc}", file=sys.stderr)
        return 3

    if result:
        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        output = args.output or (
            results_dir / "llm_judge_evaluation.json"
        )
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nLocal evaluation results saved to: {output}")
        print("\nAverage Scores:")
        for metric in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "llm_as_judge",
        ):
            print(f"  {metric}: {result.get(metric)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
