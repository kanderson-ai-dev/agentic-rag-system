"""Compute the RAGAS-style quality scorecard and serialize it to JSON.

Usage:
    python evaluation/run_ragas.py                 # scorecard + history entry
    python evaluation/run_ragas.py --report        # regenerate TREND.md
    python evaluation/run_ragas.py --prompt-version generate_answer=2 \
        --retrieval-mode hybrid --max-cost-usd 0.50 --output out.json

The scorecard is stamped with provenance — prompt versions, dataset hash,
git sha, model, retrieval mode and the real backends — so a reported number
can never be read against a different configuration than the one that ran.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import Settings, get_settings
from app.core.cost_tracking import TokenUsageCallbackHandler
from app.graph.graph import build_default_graph
from app.graph.prompts import pin_versions
from evaluation import evaluators
from evaluation.dataset import build_dataset
from evaluation.harness import (
    CostCapExceeded,
    parse_prompt_version_overrides,
    run_cases,
    run_metadata,
    score_results,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
HISTORY_DIR = RESULTS_DIR / "history"

RETRIEVAL_MODES = ("hybrid", "vector_only", "graph_only", "no_retrieval")


def compute_scorecard(
    *,
    settings: Settings | None = None,
    prompt_versions: dict[str, int] | None = None,
    retrieval_mode: str = "hybrid",
    max_cost_usd: float | None = None,
    graph: object | None = None,
) -> dict:
    """Run the shared dataset and compute metrics, behavior and latency.

    ``prompt_versions`` pins specific prompt versions for the run (the A/B
    engine uses this to score candidates without mutating the registry).
    """
    settings = settings or get_settings()
    settings.configure_langsmith_env()
    if retrieval_mode != "hybrid":
        raise NotImplementedError(
            f"retrieval_mode={retrieval_mode!r} lands in a later phase"
        )

    token_handler = TokenUsageCallbackHandler()
    evaluators.set_token_handler(token_handler)
    pin_versions(prompt_versions)
    try:
        if graph is None:
            graph = build_default_graph(
                settings, InMemorySaver(), token_handler
            )
        cases = build_dataset()
        results = run_cases(
            graph,
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

    scorecard["meta"] = run_metadata(settings, retrieval_mode=retrieval_mode)
    return scorecard


def _write_trend() -> None:
    """Regenerate TREND.md from history entries (with provenance columns)."""
    rows: list[tuple[str, dict]] = []
    if HISTORY_DIR.exists():
        for path in sorted(HISTORY_DIR.glob("*.json")):
            rows.append((path.stem, json.loads(path.read_text(encoding="utf-8"))))
    header = (
        "| run | faithfulness | answer_relevancy | context_precision | "
        "context_recall | prompt_versions | retrieval_mode |"
    )
    lines = [header, "|---|---|---|---|---|---|---|"]
    for name, data in rows:
        meta = data.get("meta", {})
        versions = meta.get("prompt_versions") or {}
        pv = ",".join(f"{k}:v{v}" for k, v in sorted(versions.items())) or "?"
        mode = meta.get("retrieval_mode", "hybrid")
        lines.append(
            f"| {name} | {data.get('faithfulness')} | "
            f"{data.get('answer_relevancy')} | {data.get('context_precision')} "
            f"| {data.get('context_recall')} | {pv} | {mode} |"
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "TREND.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report", action="store_true", help="generate TREND.md from history"
    )
    parser.add_argument(
        "--prompt-version",
        action="append",
        metavar="NAME=N",
        dest="prompt_versions",
        help="pin a prompt version for the run (repeatable)",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=RETRIEVAL_MODES,
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
        help="scorecard output path (default: results/ragas_scorecard.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.report:
        _write_trend()
        return 0

    try:
        pins = parse_prompt_version_overrides(args.prompt_versions)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        scorecard = compute_scorecard(
            prompt_versions=pins or None,
            retrieval_mode=args.retrieval_mode,
            max_cost_usd=args.max_cost_usd,
        )
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except CostCapExceeded as exc:
        print(f"aborted: {exc}", file=sys.stderr)
        return 3

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")

    output = args.output or (RESULTS_DIR / "ragas_scorecard.json")
    output.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    (HISTORY_DIR / f"{timestamp}.json").write_text(
        json.dumps(scorecard, indent=2), encoding="utf-8"
    )
    print(json.dumps(scorecard, indent=2))
    print(f"\nScorecard written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
