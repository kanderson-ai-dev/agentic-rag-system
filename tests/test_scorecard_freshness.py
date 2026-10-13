"""Committed scorecard must match the current dataset and promoted prompts.

A scorecard produced by a different dataset hash or prompt-version set is an
unverified claim — CI fails until a fresh run regenerates it. Thresholds are
checked by the same gate the promotion path uses.
"""

import json
from pathlib import Path

import pytest

from evaluation.check_scorecard import check_freshness, check_thresholds
from evaluation.dataset import dataset_version

SCORECARD = (
    Path(__file__).resolve().parent.parent
    / "evaluation"
    / "results"
    / "ragas_scorecard.json"
)


def _committed() -> dict:
    assert SCORECARD.exists(), "ragas_scorecard.json is missing"
    return json.loads(SCORECARD.read_text(encoding="utf-8"))


def test_committed_scorecard_meets_edd_thresholds() -> None:
    assert check_thresholds(_committed()) == []


def test_committed_scorecard_is_fresh() -> None:
    """Meta provenance must equal the current dataset hash + promoted set."""
    assert check_freshness(_committed()) == []


def test_freshness_catches_stale_dataset() -> None:
    scorecard = _committed()
    scorecard.setdefault("meta", {})["dataset_version"] = "not-current"
    failures = check_freshness(scorecard)
    assert any("dataset_version" in f for f in failures)


def test_freshness_catches_stale_prompt_versions() -> None:
    scorecard = _committed()
    scorecard.setdefault("meta", {})["prompt_versions"] = {"generate_answer": 999}
    failures = check_freshness(scorecard)
    assert any("prompt_versions" in f for f in failures)


def test_thresholds_catch_quality_regression() -> None:
    assert check_thresholds({"faithfulness": 0.5}) != []


@pytest.mark.skipif(not SCORECARD.exists(), reason="no scorecard committed")
def test_scorecard_meta_declares_dataset_version() -> None:
    meta = _committed().get("meta") or {}
    assert meta.get("dataset_version") == dataset_version()
