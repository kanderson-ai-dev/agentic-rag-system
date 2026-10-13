"""Gate a scorecard on the EDD thresholds and check its provenance is current.

Two independent checks:

- **Thresholds** — the scorecard must meet every published EDD threshold
  (delegates to ``evaluate_promotion`` with no baseline, so promotion and CI
  enforce the same numbers).
- **Freshness** — ``meta.dataset_version`` and ``meta.prompt_versions`` must
  match the current dataset hash and promoted registry versions; a scorecard
  produced by a different configuration is an unverified claim.

Usage:
    python evaluation/check_scorecard.py [path/to/scorecard.json]

Exit code 0 = pass, 1 = at least one check failed.
"""

import json
import sys
from pathlib import Path
from typing import Any

from app.services.promotion_gate import evaluate_promotion
from evaluation.dataset import dataset_version

DEFAULT_PATH = Path(__file__).resolve().parent / "results" / "ragas_scorecard.json"


def check_thresholds(scorecard: dict[str, Any]) -> list[str]:
    """EDD threshold failures — empty list when the scorecard passes."""
    return evaluate_promotion(scorecard, baseline_scorecard=None).failures


def check_freshness(scorecard: dict[str, Any]) -> list[str]:
    """Drift between the scorecard's provenance and the current config."""
    from app.graph.prompts import active_versions

    meta = scorecard.get("meta") or {}
    failures: list[str] = []

    recorded = meta.get("dataset_version")
    current = dataset_version()
    if recorded != current:
        failures.append(
            f"dataset_version: scorecard ran {recorded!r}, current is {current!r}"
        )

    recorded_versions = meta.get("prompt_versions") or {}
    current_versions = active_versions()
    if recorded_versions != current_versions:
        failures.append(
            "prompt_versions: scorecard ran "
            f"{recorded_versions!r}, current promoted set is {current_versions!r}"
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    path = Path(argv[0]) if argv else DEFAULT_PATH
    if not path.exists():
        print(f"error: scorecard not found: {path}", file=sys.stderr)
        return 1

    scorecard = json.loads(path.read_text(encoding="utf-8"))
    failures = check_thresholds(scorecard) + check_freshness(scorecard)
    if failures:
        print(f"scorecard gate FAILED ({path}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"scorecard gate passed ({path})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
