"""Measured retrieval-mode comparison over the shared eval dataset (F4).

Runs the same dataset through the graph once per ``retrieval_mode`` —
``hybrid``, ``vector_only``, ``graph_only`` and a bare-LLM ``no_retrieval``
arm — and writes ``evaluation/results/baseline_comparison.json`` plus a
Markdown table. Metrics are reported per category so the single-hop vs.
multi-hop split is explicit: the README's hybrid claim stands or falls on
the *multi-hop* numbers, which is where a graph backend should earn its
cost.

Usage::

    python evaluation/run_baseline.py --max-cost-usd 1.0
    python evaluation/run_baseline.py --modes hybrid,vector_only

One token handler spans every arm so ``--max-cost-usd`` bounds the whole
run; per-mode cost is measured as a delta on the handler counters.
"""

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import Settings, get_settings
from app.core.cost_tracking import TokenUsageCallbackHandler, calculate_cost_usd
from app.graph.graph import build_default_graph
from app.graph.prompts import pin_versions
from evaluation import evaluators
from evaluation.dataset import build_dataset
from evaluation.harness import (
    CostCapExceeded,
    MetricScorer,
    parse_prompt_version_overrides,
    run_cases,
    run_metadata,
    score_results,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
COMPARISON_PATH = RESULTS_DIR / "baseline_comparison.json"

MODES = ("hybrid", "vector_only", "graph_only", "no_retrieval")


class NoRetrievalGraphAdapter:
    """Bare-LLM baseline: answers with the promoted generation prompt and
    no retrieval context, exposing the ``invoke`` shape `run_cases` expects.

    The generation prompt (and its versioned registry resolution) is kept so
    the arm isolates *retrieval itself* — prompt, model and scoring are
    identical to the retrieval modes.
    """

    def __init__(
        self, settings: Settings, token_handler: TokenUsageCallbackHandler
    ) -> None:
        from app.services.llm import build_chat_model, build_generation_chain

        self._chain = build_generation_chain(build_chat_model(settings, token_handler))

    def invoke(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        answer = self._chain.invoke(
            {
                "context": "",
                "question": state["question"],
                "prompt_version": None,
            },
            config=config,
        )
        return {
            "generation": answer,
            "documents": [],
            "blocked": False,
            "prompt_versions": {},
        }


def _build_mode_graph(
    mode: str,
    mode_settings: Settings,
    token_handler: TokenUsageCallbackHandler,
) -> Any:
    if mode == "no_retrieval":
        return NoRetrievalGraphAdapter(mode_settings, token_handler)
    return build_default_graph(mode_settings, InMemorySaver(), token_handler)


def _mode_cost_usd(
    tokens_before: tuple[int, int],
    token_handler: TokenUsageCallbackHandler,
    settings: Settings,
) -> float:
    """Spend accumulated since the snapshot — covers pipeline + judge calls."""
    prompt_delta = token_handler.prompt_tokens - tokens_before[0]
    completion_delta = token_handler.completion_tokens - tokens_before[1]
    return calculate_cost_usd(
        prompt_delta,
        completion_delta,
        settings.cost_input_price_per_1m,
        settings.cost_output_price_per_1m,
    )


def run_baseline(
    *,
    settings: Settings | None = None,
    prompt_versions: dict[str, int] | None = None,
    modes: tuple[str, ...] = MODES,
    max_cost_usd: float | None = None,
    scorers: Mapping[str, MetricScorer] | None = None,
    graph: Any | None = None,
) -> dict[str, Any]:
    """Run every mode over the shared dataset and build the comparison.

    ``graph`` may be injected by tests (a single stub serves all modes —
    mode isolation itself is unit-tested at the retrieve-node level).
    """
    settings = settings or get_settings()
    settings.configure_langsmith_env()
    if scorers is None:
        scorers = evaluators.default_scorers()

    token_handler = TokenUsageCallbackHandler()
    evaluators.set_token_handler(token_handler)
    pin_versions(prompt_versions)
    try:
        cases = build_dataset()
        comparison: dict[str, Any] = {"modes": {}}
        for mode in modes:
            if mode not in MODES:
                raise ValueError(f"unknown retrieval_mode {mode!r}")
            mode_settings = settings.model_copy(update={"retrieval_mode": mode})
            mode_graph = graph or _build_mode_graph(
                mode, mode_settings, token_handler
            )
            tokens_before = (
                token_handler.prompt_tokens,
                token_handler.completion_tokens,
            )
            results = run_cases(
                mode_graph,
                cases,
                thread_prefix=f"baseline-{mode}",
                token_handler=token_handler,
                max_cost_usd=max_cost_usd,
                settings=mode_settings,
            )
            scorecard = score_results(
                results, scorers, token_handler=token_handler, settings=mode_settings
            )
            spent = _mode_cost_usd(tokens_before, token_handler, settings)
            # score_results reports cumulative handler cost; replace with the
            # per-mode delta so each arm's cost stands alone.
            scorecard["cost_total_usd"] = round(spent, 6)
            scorecard["cost_per_query_usd"] = round(
                spent / (len(results) or 1), 6
            )
            scorecard["meta"] = run_metadata(mode_settings)
            comparison["modes"][mode] = scorecard
    finally:
        pin_versions(None)
        evaluators.set_token_handler(None)

    meta = run_metadata(settings)
    meta.pop("retrieval_mode", None)
    meta["retrieval_modes"] = list(modes)
    comparison["meta"] = meta
    return comparison


def comparison_markdown(comparison: dict[str, Any]) -> str:
    """Render the comparison as a Markdown table — headline metrics per mode
    plus the multi-hop rows the hybrid claim is decided on."""
    modes = comparison["modes"]
    header = (
        "| metric | " + " | ".join(modes) + " |\n"
        "|---|" + "---|" * len(modes)
    )
    rows: list[str] = []

    def _row(label: str, key: str, source: str = "top") -> None:
        cells = []
        for card in modes.values():
            value = (
                card.get(key)
                if source == "top"
                else (card.get("per_category", {}).get(source, {}) or {}).get(key)
            )
            cells.append("—" if value is None else str(value))
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    for key in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "llm_as_judge",
    ):
        _row(key, key)
    for key in ("injection_block_rate", "ood_refusal_rate", "behavior_accuracy"):
        _row(key, key)
    for key in ("cost_per_query_usd", "latency_p50_ms", "latency_p95_ms"):
        _row(key, key)
    _row("multi_hop faithfulness", "faithfulness", "multi_hop")
    _row("multi_hop context_recall", "context_recall", "multi_hop")
    _row("multi_hop llm_as_judge", "llm_as_judge", "multi_hop")
    _row("single_hop faithfulness", "faithfulness", "single_hop")
    return header + "\n" + "\n".join(rows) + "\n"


def hybrid_verdict(comparison: dict[str, Any]) -> dict[str, Any]:
    """Whether the multi-hop numbers justify the hybrid claim — the
    honesty-rule check from the plan, kept explicit so the README wording
    follows the measurement, not the other way round.

    "Wins" means: never loses on any multi-hop quality metric *and* wins on
    at least one — a ceiling tie (both at 1.0) is not a loss.
    """
    modes = comparison["modes"]
    hybrid = modes.get("hybrid", {}).get("per_category", {}).get("multi_hop", {})
    vector = (
        modes.get("vector_only", {}).get("per_category", {}).get("multi_hop", {})
    )
    deltas = {
        metric: round((hybrid.get(metric) or 0.0) - (vector.get(metric) or 0.0), 4)
        for metric in ("faithfulness", "context_recall", "llm_as_judge")
        if hybrid.get(metric) is not None and vector.get(metric) is not None
    }
    wins = (
        bool(deltas)
        and all(delta >= 0 for delta in deltas.values())
        and any(delta > 0 for delta in deltas.values())
    )
    return {
        "multi_hop_deltas_vs_vector_only": deltas,
        "hybrid_wins": wins,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--modes",
        default=",".join(MODES),
        help="comma-separated subset of modes (default: all four)",
    )
    parser.add_argument(
        "--prompt-version",
        action="append",
        metavar="NAME=N",
        dest="prompt_versions",
        help="pin a prompt version for the run (repeatable)",
    )
    parser.add_argument("--max-cost-usd", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        pins = parse_prompt_version_overrides(args.prompt_versions)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    invalid = [m for m in modes if m not in MODES]
    if invalid:
        print(f"error: unknown modes {invalid}; valid: {list(MODES)}", file=sys.stderr)
        return 2

    try:
        comparison = run_baseline(
            prompt_versions=pins or None,
            modes=modes,
            max_cost_usd=args.max_cost_usd,
        )
    except CostCapExceeded as exc:
        print(f"aborted: {exc}", file=sys.stderr)
        return 3

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = args.output or COMPARISON_PATH
    output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nBaseline comparison saved to: {output}")
    print("\n" + comparison_markdown(comparison))
    verdict = hybrid_verdict(comparison)
    print(f"hybrid vs vector_only (multi-hop): {verdict['multi_hop_deltas_vs_vector_only']}")
    if not verdict["hybrid_wins"]:
        print("NOTE: hybrid did not beat vector_only on all multi-hop metrics —")
        print("      per the honesty rule, README claims must follow the numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
