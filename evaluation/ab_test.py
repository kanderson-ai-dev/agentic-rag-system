"""A/B evaluation engine: score a candidate prompt against the promoted one.

Runs the shared dataset twice through an identically-configured graph, the
only difference being the pinned version of one prompt — baseline (promoted,
or an explicit ``--baseline``) vs candidate. Produces both scorecards,
per-metric deltas, cost/latency deltas, a promotion-gate preview, and persists
the evidence to ``evaluation/results/ab/<id>.json``.

CLI::

    python -m evaluation.ab_test --name generate_answer --candidate 2 \
        [--baseline promoted|N] [--max-cost-usd 1.0] [--timeout-seconds 600]
"""

import argparse
import json
import sys
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import Settings, get_settings
from app.core.cost_tracking import TokenUsageCallbackHandler
from app.graph.graph import build_default_graph
from app.graph.prompts import get_registry, pin_versions
from app.services.promotion_gate import evaluate_promotion
from app.services.prompt_registry import PromptName, PromptRegistry
from evaluation import evaluators
from evaluation.dataset import build_dataset
from evaluation.harness import (
    MetricScorer,
    run_cases,
    run_metadata,
    score_results,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "ab"


class AbTestError(RuntimeError):
    """Invalid A/B request (unknown version, comparing a version to itself)."""


def _arm_scorecard(
    graph: Any,
    cases: list[dict[str, str]],
    *,
    prompt_name: str,
    version: int,
    scorers: Mapping[str, MetricScorer],
    token_handler: TokenUsageCallbackHandler,
    settings: Settings,
    max_cost_usd: float | None,
    deadline_seconds: float | None,
    thread_prefix: str,
) -> dict[str, Any]:
    pin_versions({prompt_name: version})
    try:
        results = run_cases(
            graph,
            cases,
            thread_prefix=thread_prefix,
            token_handler=token_handler,
            max_cost_usd=max_cost_usd,
            settings=settings,
            deadline_seconds=deadline_seconds,
        )
        return score_results(
            results, scorers, token_handler=token_handler, settings=settings
        )
    finally:
        pin_versions(None)


def run_ab_test(
    prompt_name: str,
    candidate_version: int,
    *,
    baseline_version: int | None = None,
    settings: Settings | None = None,
    registry: PromptRegistry | None = None,
    max_cost_usd: float | None = None,
    deadline_seconds: float | None = None,
    graph: Any | None = None,
    scorers: Mapping[str, MetricScorer] | None = None,
    results_dir: str | Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Run the A/B comparison and persist the evidence.

    One token handler is shared across both arms so ``max_cost_usd`` and the
    deadline bound the *whole* comparison (pipeline + judging calls).
    """
    settings = settings or get_settings()
    settings.configure_langsmith_env()
    registry = registry or get_registry()
    name = PromptName(prompt_name)

    baseline_version = (
        baseline_version
        if baseline_version is not None
        else registry.promoted(name).version
    )
    if candidate_version == baseline_version:
        raise AbTestError(
            f"{name} v{candidate_version} is already the baseline — "
            "an A/B run needs a different version on each side"
        )
    candidate = registry.get(name, candidate_version)  # 404s on unknown version
    baseline = registry.get(name, baseline_version)

    if scorers is None:
        scorers = evaluators.default_scorers()
    token_handler = TokenUsageCallbackHandler()
    evaluators.set_token_handler(token_handler)
    # `live` marks evidence produced by real model calls: true only when the
    # engine built the production graph itself with a configured key. An
    # injected graph (tests, fixtures) means the calls never happened here.
    live = graph is None and settings.openai_api_key is not None
    if graph is None:
        graph = build_default_graph(settings, InMemorySaver(), token_handler)

    cases = build_dataset()
    try:
        baseline_card = _arm_scorecard(
            graph,
            cases,
            prompt_name=name.value,
            version=baseline_version,
            scorers=scorers,
            token_handler=token_handler,
            settings=settings,
            max_cost_usd=max_cost_usd,
            deadline_seconds=deadline_seconds,
            thread_prefix="ab-baseline",
        )
        candidate_card = _arm_scorecard(
            graph,
            cases,
            prompt_name=name.value,
            version=candidate.version,
            scorers=scorers,
            token_handler=token_handler,
            settings=settings,
            max_cost_usd=max_cost_usd,
            deadline_seconds=deadline_seconds,
            thread_prefix="ab-candidate",
        )
    finally:
        evaluators.set_token_handler(None)
        pin_versions(None)

    flat_base = _flat(baseline_card)
    flat_cand = _flat(candidate_card)
    deltas = {
        metric: round(flat_cand.get(metric, 0.0) - flat_base.get(metric, 0.0), 4)
        for metric in sorted(set(flat_base) | set(flat_cand))
        if isinstance(flat_cand.get(metric), (int, float))
        and isinstance(flat_base.get(metric), (int, float))
    }
    gate = evaluate_promotion(flat_cand, flat_base)

    ab_test_id = uuid.uuid4().hex[:12]
    result: dict[str, Any] = {
        "ab_test_id": ab_test_id,
        "prompt_name": name.value,
        "baseline_version": baseline_version,
        "candidate_version": candidate.version,
        "baseline_scorecard": flat_base,
        "candidate_scorecard": flat_cand,
        "baseline_detail": baseline_card,
        "candidate_detail": candidate_card,
        "deltas": deltas,
        "gate": {"allowed": gate.allowed, "failures": gate.failures},
        "meta": {
            **run_metadata(settings, retrieval_mode="hybrid"),
            "live": live,
            "tokens": {
                "prompt_tokens": token_handler.prompt_tokens,
                "completion_tokens": token_handler.completion_tokens,
            },
        },
        "candidate_status": candidate.status.value,
        "baseline_status": baseline.status.value,
        "generated_at": datetime.now(UTC).isoformat(),
    }

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ab_test_id}.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["result_path"] = str(path)
    return result


def _flat(scorecard: dict[str, Any]) -> dict[str, Any]:
    """Flat metric view of a harness scorecard (drops cases/meta for the gate)."""
    return {
        key: value
        for key, value in scorecard.items()
        if key not in {"cases", "per_category", "meta"}
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="prompt name")
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument(
        "--baseline",
        default="promoted",
        help="baseline version number or 'promoted' (default)",
    )
    parser.add_argument("--max-cost-usd", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    baseline = None if args.baseline == "promoted" else int(args.baseline)
    try:
        result = run_ab_test(
            args.name,
            args.candidate,
            baseline_version=baseline,
            max_cost_usd=args.max_cost_usd,
            deadline_seconds=args.timeout_seconds,
            results_dir=args.output or RESULTS_DIR,
        )
    except (AbTestError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    print(
        f"\nGate verdict: {'ALLOWED' if result['gate']['allowed'] else 'BLOCKED'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
