"""Fixture data must never masquerade as measured evidence (Phase 8 guard).

``scripts/ui_qa.py`` ships a canned A/B payload (``CANNED_AB``) used purely
to exercise the console UI offline. Its markers — the ``ui-qa-stub`` dataset
version and the fixture author names — must never appear in the README or
in committed evidence under ``evaluation/results/`` presented as real runs.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_FIXTURE_MARKERS = ("ui-qa-stub", "ui-qa candidate", "demo candidate")


def test_fixture_markers_not_in_readme() -> None:
    content = (ROOT / "README.md").read_text(encoding="utf-8")
    for marker in _FIXTURE_MARKERS:
        assert marker not in content, f"fixture marker {marker!r} in README.md"


def test_committed_ab_results_are_not_fixture_runs() -> None:
    ab_dir = ROOT / "evaluation" / "results" / "ab"
    results = list(ab_dir.glob("*.json"))
    assert results, "no committed A/B evidence found"
    for path in results:
        content = path.read_text(encoding="utf-8")
        for marker in _FIXTURE_MARKERS:
            assert marker not in content, f"fixture marker {marker!r} in {path.name}"
        result = json.loads(content)
        meta = result.get("meta") or {}
        assert meta.get("dataset_version") != "ui-qa-stub", path.name
