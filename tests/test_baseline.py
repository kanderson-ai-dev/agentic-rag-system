"""Tests for `evaluation/run_baseline.py` — mode runner, table, verdict.

Fully offline: a stub graph serves every mode (mode isolation itself is
unit-tested at the retrieve-node level in `test_hybrid_retrieval.py`).
"""

from typing import Any

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.cost_tracking import TokenUsageCallbackHandler, calculate_cost_usd
from evaluation.run_baseline import (
    NoRetrievalGraphAdapter,
    comparison_markdown,
    hybrid_verdict,
    run_baseline,
)


class _StubGraph:
    """Deterministic stand-in for the compiled graph."""

    def __init__(self, answer: str = "stub answer") -> None:
        self.answer = answer
        self.invoke_count = 0

    def invoke(self, state: dict, config: dict | None = None) -> dict:
        self.invoke_count += 1
        return {
            "generation": self.answer,
            "documents": [],
            "blocked": False,
            "prompt_versions": {"generate_answer": 1},
        }


def _fake_scorers() -> dict:
    return {
        name: (lambda **kw: 0.9)
        for name in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "llm_as_judge",
        )
    }


def test_settings_accepts_valid_modes() -> None:
    for mode in ("hybrid", "vector_only", "graph_only", "no_retrieval"):
        assert Settings(retrieval_mode=mode).retrieval_mode == mode


def test_settings_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        Settings(retrieval_mode="everything")


def test_settings_retrieval_mode_defaults_to_hybrid() -> None:
    assert Settings.model_fields["retrieval_mode"].default == "hybrid"


def test_run_baseline_produces_scorecard_per_mode() -> None:
    stub = _StubGraph()
    comparison = run_baseline(
        settings=Settings(),
        graph=stub,
        scorers=_fake_scorers(),
        modes=("hybrid", "vector_only"),
    )

    assert list(comparison["modes"]) == ["hybrid", "vector_only"]
    for card in comparison["modes"].values():
        assert card["cases_count"] > 0
        assert card["faithfulness"] == 0.9
        assert card["meta"]["retrieval_mode"] in ("hybrid", "vector_only")
    assert comparison["meta"]["retrieval_modes"] == ["hybrid", "vector_only"]
    assert "git_sha" in comparison["meta"]


def test_mode_cost_usd_measures_delta_since_snapshot() -> None:
    """Each arm's cost is a delta on the shared handler, not cumulative spend."""
    from evaluation.run_baseline import _mode_cost_usd

    settings = Settings()
    handler = TokenUsageCallbackHandler()
    handler.prompt_tokens = 1_000
    handler.completion_tokens = 500
    before = (handler.prompt_tokens, handler.completion_tokens)

    handler.prompt_tokens = 2_000
    handler.completion_tokens = 1_500

    expected = calculate_cost_usd(
        1_000,
        1_000,
        settings.cost_input_price_per_1m,
        settings.cost_output_price_per_1m,
    )
    assert _mode_cost_usd(before, handler, settings) == pytest.approx(expected)


def test_comparison_markdown_renders_table() -> None:
    comparison = {
        "modes": {
            "hybrid": {
                "faithfulness": 0.9,
                "answer_relevancy": 0.88,
                "context_precision": 0.8,
                "context_recall": 0.75,
                "llm_as_judge": 4.2,
                "injection_block_rate": 1.0,
                "ood_refusal_rate": 1.0,
                "behavior_accuracy": 1.0,
                "cost_per_query_usd": 0.001,
                "latency_p50_ms": 3000,
                "latency_p95_ms": 6000,
                "per_category": {
                    "multi_hop": {
                        "faithfulness": 0.92,
                        "context_recall": 0.8,
                        "llm_as_judge": 4.3,
                    },
                    "single_hop": {"faithfulness": 0.89},
                },
            },
            "vector_only": {
                "faithfulness": 0.85,
                "answer_relevancy": 0.86,
                "context_precision": 0.78,
                "context_recall": 0.7,
                "llm_as_judge": 4.0,
                "injection_block_rate": 1.0,
                "ood_refusal_rate": 1.0,
                "behavior_accuracy": 1.0,
                "cost_per_query_usd": 0.0009,
                "latency_p50_ms": 2500,
                "latency_p95_ms": 5000,
                "per_category": {
                    "multi_hop": {
                        "faithfulness": 0.84,
                        "context_recall": 0.7,
                        "llm_as_judge": 4.0,
                    },
                    "single_hop": {"faithfulness": 0.86},
                },
            },
        }
    }
    table = comparison_markdown(comparison)
    assert "| metric | hybrid | vector_only |" in table
    assert "multi_hop faithfulness" in table
    assert "0.92" in table and "0.84" in table


def test_hybrid_verdict_detects_win_and_loss() -> None:
    win = {
        "modes": {
            "hybrid": {
                "per_category": {
                    "multi_hop": {
                        "faithfulness": 0.92,
                        "context_recall": 0.8,
                        "llm_as_judge": 4.3,
                    }
                }
            },
            "vector_only": {
                "per_category": {
                    "multi_hop": {
                        "faithfulness": 0.84,
                        "context_recall": 0.7,
                        "llm_as_judge": 4.0,
                    }
                }
            },
        }
    }
    verdict = hybrid_verdict(win)
    assert verdict["hybrid_wins"] is True
    assert verdict["multi_hop_deltas_vs_vector_only"]["faithfulness"] == pytest.approx(
        0.08, abs=1e-3
    )

    # A ceiling tie (1.0 vs 1.0 on faithfulness) is not a loss — hybrid still
    # wins when it leads on another metric.
    tie = {
        "modes": {
            "hybrid": {
                "per_category": {
                    "multi_hop": {
                        "faithfulness": 1.0,
                        "context_recall": 0.9167,
                        "llm_as_judge": 4.7778,
                    }
                }
            },
            "vector_only": {
                "per_category": {
                    "multi_hop": {
                        "faithfulness": 1.0,
                        "context_recall": 0.8333,
                        "llm_as_judge": 4.7222,
                    }
                }
            },
        }
    }
    assert hybrid_verdict(tie)["hybrid_wins"] is True

    loss = {
        "modes": {
            "hybrid": {
                "per_category": {
                    "multi_hop": {
                        "faithfulness": 0.80,
                        "context_recall": 0.7,
                        "llm_as_judge": 4.0,
                    }
                }
            },
            "vector_only": {
                "per_category": {
                    "multi_hop": {
                        "faithfulness": 0.84,
                        "context_recall": 0.75,
                        "llm_as_judge": 4.1,
                    }
                }
            },
        }
    }
    assert hybrid_verdict(loss)["hybrid_wins"] is False


def test_no_retrieval_adapter_returns_empty_documents() -> None:
    """The bare-LLM arm never retrieves — documents are always empty."""

    class _Chain:
        def invoke(self, inputs: dict, config: Any = None) -> str:
            assert inputs["context"] == ""
            return "no-context answer"

    adapter = NoRetrievalGraphAdapter.__new__(NoRetrievalGraphAdapter)
    adapter._chain = _Chain()
    result = adapter.invoke({"question": "q"})

    assert result["documents"] == []
    assert result["generation"] == "no-context answer"
    assert result["blocked"] is False


def test_dashboard_quality_includes_baseline_when_file_exists(
    monkeypatch, tmp_path
) -> None:
    import json

    from fastapi.testclient import TestClient

    import app.api.v1.routes.dashboard as dashboard
    from app.main import app

    scorecard = tmp_path / "scorecard.json"
    scorecard.write_text(
        json.dumps({"faithfulness": 0.9, "generated_at": "2026-01-01"}),
        encoding="utf-8",
    )
    comparison = tmp_path / "baseline.json"
    comparison.write_text(
        json.dumps(
            {
                "meta": {"generated_at": "2026-01-01"},
                "modes": {
                    "hybrid": {
                        "faithfulness": 0.9,
                        "per_category": {"multi_hop": {"faithfulness": 0.92}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "_SCORECARD_PATH", scorecard)
    monkeypatch.setattr(dashboard, "_BASELINE_PATH", comparison)

    with TestClient(app) as client:
        body = client.get("/api/v1/dashboard/quality").json()

    assert body["available"] is True
    assert body["baseline_comparison"]["modes"]["hybrid"]["faithfulness"] == 0.9
    assert body["baseline_comparison"]["multi_hop"]["hybrid"]["faithfulness"] == 0.92


def test_dashboard_quality_omits_baseline_when_missing(monkeypatch, tmp_path) -> None:
    import json

    from fastapi.testclient import TestClient

    import app.api.v1.routes.dashboard as dashboard
    from app.main import app

    scorecard = tmp_path / "scorecard.json"
    scorecard.write_text(json.dumps({"faithfulness": 0.9}), encoding="utf-8")
    monkeypatch.setattr(dashboard, "_SCORECARD_PATH", scorecard)
    monkeypatch.setattr(dashboard, "_BASELINE_PATH", tmp_path / "nope.json")

    with TestClient(app) as client:
        body = client.get("/api/v1/dashboard/quality").json()

    assert body["available"] is True
    assert "baseline_comparison" not in body
