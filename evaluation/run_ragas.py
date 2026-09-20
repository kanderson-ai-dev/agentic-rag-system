"""Compute the RAGAS-style quality scorecard and serialize it to JSON.

Usage:
    python evaluation/run_ragas.py            # write scorecard + history entry
    python evaluation/run_ragas.py --report   # regenerate TREND.md from history
"""

import argparse
import json
from datetime import UTC, datetime
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
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
HISTORY_DIR = RESULTS_DIR / "history"

METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}


def _metric_inputs(question: str, answer: str, context: str) -> dict[str, dict[str, str]]:
    return {
        "faithfulness": {"context": context, "answer": answer},
        "answer_relevancy": {"question": question, "answer": answer},
        "context_precision": {"question": question, "context": context},
        "context_recall": {"question": question, "context": context},
    }


def compute_scorecard() -> dict[str, float]:
    """Run the graph over the dataset and compute the four metrics."""
    graph = build_default_graph(get_settings(), InMemorySaver())
    sums = {name: 0.0 for name in METRICS}
    cases = build_dataset()
    for case in cases:
        result = graph.invoke(
            initial_state(case["question"]),
            config={"configurable": {"thread_id": "eval"}},
        )
        answer = result.get("generation", "")
        context = "\n\n".join(doc.page_content for doc in result.get("documents", []))
        inputs = _metric_inputs(case["question"], answer, context)
        for name, fn in METRICS.items():
            sums[name] += fn(**inputs[name])
    total = len(cases) or 1
    return {name: round(sums[name] / total, 4) for name in METRICS}


def _write_trend() -> None:
    rows: list[tuple[str, dict]] = []
    if HISTORY_DIR.exists():
        for path in sorted(HISTORY_DIR.glob("*.json")):
            rows.append((path.stem, json.loads(path.read_text(encoding="utf-8"))))
    header = "| run | faithfulness | answer_relevancy | context_precision | context_recall |"
    lines = [header, "|---|---|---|---|---|"]
    for name, data in rows:
        lines.append(
            f"| {name} | {data.get('faithfulness')} | {data.get('answer_relevancy')} "
            f"| {data.get('context_precision')} | {data.get('context_recall')} |"
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "TREND.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="generate TREND.md from history")
    args = parser.parse_args()

    if args.report:
        _write_trend()
        return

    scorecard = compute_scorecard()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")

    payload = {**scorecard, "generated_at": timestamp}
    (RESULTS_DIR / "ragas_scorecard.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (HISTORY_DIR / f"{timestamp}.json").write_text(
        json.dumps(scorecard, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
