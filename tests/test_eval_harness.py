"""Offline tests for the unified evaluation dataset and harness."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from app.core.config import Settings
from app.core.cost_tracking import TokenUsageCallbackHandler
from app.graph.prompts import (
    active_versions,
    get_prompt,
    get_system_text,
    pin_versions,
    set_registry,
)
from app.services.prompt_registry import PromptRegistry
from evaluation.dataset import (
    CATEGORIES,
    EXPECTED_BEHAVIORS,
    SAMPLE_CASES,
    build_dataset,
    dataset_version,
    validate_dataset,
)
from evaluation.harness import (
    CaseResult,
    CostCapExceeded,
    parse_prompt_version_overrides,
    percentile,
    prompt_version_tag,
    run_cases,
    run_metadata,
    score_results,
)

_REPO_PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


class FakeGraph:
    """Deterministic graph double: blocks injections, refuses OOD, answers rest."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, state: dict, config: dict | None = None) -> dict[str, Any]:
        question = state["question"]
        self.calls.append(question)
        if "ignore all previous" in question.lower() or "do anything now" in question.lower():
            return {**state, "blocked": True, "generation": "", "documents": []}
        if "capital of france" in question.lower() or "sourdough" in question.lower():
            return {
                **state,
                "blocked": False,
                "generation": "I don't know.",
                "documents": [],
            }
        return {
            **state,
            "blocked": False,
            "generation": "A grounded answer.",
            "documents": [
                Document(
                    page_content="ctx",
                    metadata={"source": "doc-a"},
                )
            ],
        }


def _scorers() -> dict[str, Any]:
    values = {
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "context_precision": 0.7,
        "context_recall": 0.6,
        "llm_as_judge": 4.0,
    }
    return {
        name: (lambda question="", answer="", context="", v=v: v)
        for name, v in values.items()
    }


@pytest.fixture
def registry(tmp_path: Path) -> PromptRegistry:
    shutil.copytree(_REPO_PROMPTS, tmp_path / "prompts")
    instance = PromptRegistry(tmp_path / "prompts")
    set_registry(instance)
    return instance


class TestDataset:
    def test_schema_valid(self) -> None:
        assert validate_dataset() == []
        assert len(SAMPLE_CASES) >= 18

    def test_counts_by_category(self) -> None:
        counts: dict[str, int] = {}
        for case in SAMPLE_CASES:
            counts[case["category"]] = counts.get(case["category"], 0) + 1
        assert counts["single_hop"] == 6
        assert counts["multi_hop"] == 6
        assert counts["out_of_domain"] == 2
        assert counts["injection"] == 2
        assert counts["other_language"] == 1
        assert counts["vague"] == 1

    def test_unique_ids_and_enums(self) -> None:
        ids = [c["id"] for c in SAMPLE_CASES]
        assert len(ids) == len(set(ids))
        for case in SAMPLE_CASES:
            assert case["category"] in CATEGORIES
            assert case["expected_behavior"] in EXPECTED_BEHAVIORS

    def test_existing_ids_kept_stable(self) -> None:
        ids = {c["id"] for c in SAMPLE_CASES}
        assert {
            "langgraph_overview",
            "selfrag_grading",
            "interrupt_fn",
            "checkpointer_role",
            "langsmith_edd",
            "fastapi_features",
        } <= ids

    def test_injection_cases_match_guardrail_patterns(self) -> None:
        from app.graph.guardrails import detect_prompt_injection

        for case in SAMPLE_CASES:
            if case["category"] == "injection":
                assert detect_prompt_injection(case["question"]) is not None, case["id"]

    def test_dataset_version_stable_and_sensitive(self) -> None:
        v1 = dataset_version()
        assert v1 == dataset_version()
        mutated = build_dataset()
        mutated[0]["question"] += "!"
        assert len(v1) == 10
        # hash is computed over SAMPLE_CASES — a different list must differ
        canonical = json.dumps(mutated, sort_keys=True, ensure_ascii=False)
        import hashlib

        assert hashlib.sha256(canonical.encode()).hexdigest()[:10] != v1

    def test_validate_catches_errors(self) -> None:
        bad = [
            {"id": "a", "category": "nope", "question": "q", "ground_truth": "g",
             "expected_behavior": "answer"},
            {"id": "a", "category": "single_hop", "question": "q",
             "ground_truth": "g", "expected_behavior": "bogus"},
        ]
        errors = validate_dataset(bad)
        assert any("duplicate" in e or "invalid" in e for e in errors)
        assert len(errors) >= 3


class TestPromptVersionOverrides:
    def test_parse_overrides(self) -> None:
        pins = parse_prompt_version_overrides(
            ["generate_answer=2", "grade_documents=3"]
        )
        assert pins == {"generate_answer": 2, "grade_documents": 3}

    def test_parse_rejects_malformed(self) -> None:
        for bad in ("generate_answer", "=2", "name=x", "nope=1"):
            with pytest.raises(ValueError):
                parse_prompt_version_overrides([bad])

    def test_pin_resolves_candidate(self, registry: PromptRegistry) -> None:
        pinned = registry.create_candidate(
            "generate_answer",
            system="Pinned system. {context}",
            human="Q: {question}",
            author="t",
            description="d",
            changelog="c",
        ).version
        try:
            pin_versions({"generate_answer": pinned})
            prompt = get_prompt("generate_answer")
            assert "Pinned system" in prompt.messages[0].prompt.template
            assert get_system_text("generate_answer").startswith("Pinned")
            assert active_versions()["generate_answer"] == pinned
        finally:
            pin_versions(None)
        assert (
            active_versions()["generate_answer"]
            == registry.promoted("generate_answer").version
        )

    def test_explicit_version_beats_pin(self, registry: PromptRegistry) -> None:
        pin_versions({"generate_answer": 1})
        try:
            # explicit version arg wins over the pin (pin is v1, so use a
            # fresh candidate to distinguish)
            other = registry.create_candidate(
                "generate_answer",
                system="Other. {context}",
                human="Q: {question}",
                author="t",
                description="d",
                changelog="c",
            ).version
            prompt = get_prompt("generate_answer", version=other)
            assert "Other." in prompt.messages[0].prompt.template
        finally:
            pin_versions(None)


class TestHarness:
    def test_run_cases_records_latency_and_behavior(self) -> None:
        results = run_cases(FakeGraph(), build_dataset())
        assert len(results) == len(SAMPLE_CASES)
        by_id = {r.id: r for r in results}
        assert by_id["inj_prompt_leak"].blocked is True
        assert by_id["inj_prompt_leak"].behavior_ok is True
        assert by_id["ood_france"].behavior_ok is True
        assert all(r.latency_ms >= 0 for r in results)

    def test_score_results_aggregates(self) -> None:
        results = run_cases(FakeGraph(), build_dataset())
        card = score_results(results, _scorers())
        # quality metrics averaged over answer cases only
        assert card["answered_cases"] == 14
        assert card["faithfulness"] == pytest.approx(0.9)
        assert card["llm_as_judge"] == pytest.approx(4.0)
        assert card["injection_block_rate"] == 1.0
        assert card["ood_refusal_rate"] == 1.0
        assert card["behavior_accuracy"] == 1.0
        assert "single_hop" in card["per_category"]
        assert "multi_hop" in card["per_category"]
        # injection cases carry no quality scores
        inj = next(c for c in card["cases"] if c["id"] == "inj_prompt_leak")
        assert inj["scores"] == {}

    def test_score_results_flags_bad_behavior(self) -> None:
        bad = CaseResult(
            id="inj", category="injection", expected_behavior="block_injection",
            question="q", ground_truth="g", answer="oops", contexts=[],
            sources=[], blocked=False, escalated=False, latency_ms=1,
            behavior_ok=False,
        )
        card = score_results([bad], _scorers())
        assert card["injection_block_rate"] == 0.0

    def test_cost_guard_aborts(self) -> None:
        handler = TokenUsageCallbackHandler()
        handler.prompt_tokens = 10_000_000  # ~$1.50 of gpt-4o-mini input
        settings = Settings()
        with pytest.raises(CostCapExceeded):
            run_cases(
                FakeGraph(),
                build_dataset(),
                token_handler=handler,
                max_cost_usd=0.50,
                settings=settings,
            )

    def test_deadline_aborts(self) -> None:
        with pytest.raises(TimeoutError):
            run_cases(FakeGraph(), build_dataset(), deadline_seconds=0.0)

    def test_percentile(self) -> None:
        assert percentile([], 50) == 0.0
        assert percentile([10], 95) == 10
        assert percentile([1, 2, 3, 4], 50) == 2
        assert percentile(list(range(1, 21)), 95) == 19

    def test_prompt_version_tag(self) -> None:
        tag = prompt_version_tag(
            {"generate_answer": 2, "llm_judge": 1, "grade_documents": 1,
             "transform_query": 3}
        )
        assert tag == "ga2-gd1-lj1-tq3"


class TestMetadata:
    def test_run_metadata_stamps_provenance(
        self, registry: PromptRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "")
        monkeypatch.setenv("NEO4J_URI", "")
        monkeypatch.setenv("NEO4J_PASSWORD", "")
        settings = Settings()
        meta = run_metadata(settings, retrieval_mode="hybrid")
        assert meta["prompt_versions"]["generate_answer"] == registry.promoted(
            "generate_answer"
        ).version
        assert meta["retrieval_mode"] == "hybrid"
        assert meta["dataset_version"] == dataset_version()
        assert meta["model"] == settings.chat_model_name
        assert meta["vector_backend"] in {"chroma", "pinecone"}
        assert meta["graph_backend"] in {"networkx", "neo4j"}
        assert meta["git_sha"]

    def test_backend_flags_reflect_credentials(self) -> None:
        pine = Settings(pinecone_api_key="k")
        assert "pinecone" in run_metadata(pine)["vector_backend"]
        assert "chroma" in run_metadata(Settings(pinecone_api_key=None))[
            "vector_backend"
        ]
        neo = Settings(neo4j_uri="bolt://x", neo4j_password="p")
        assert "neo4j" in run_metadata(neo)["graph_backend"]
        assert "networkx" in run_metadata(Settings(neo4j_uri=None))[
            "graph_backend"
        ]


class TestTrendReport:
    def test_trend_includes_new_columns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from evaluation import run_ragas

        history = tmp_path / "history"
        history.mkdir()
        entry = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.9,
            "context_precision": 0.8,
            "context_recall": 0.8,
            "meta": {
                "prompt_versions": {"generate_answer": 2},
                "retrieval_mode": "vector_only",
            },
        }
        (history / "2026-01-01_000000.json").write_text(json.dumps(entry))
        monkeypatch.setattr(run_ragas, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(run_ragas, "HISTORY_DIR", history)

        run_ragas._write_trend()
        text = (tmp_path / "TREND.md").read_text()
        assert "prompt_versions" in text
        assert "retrieval_mode" in text
        assert "generate_answer:v2" in text
        assert "vector_only" in text


class TestScorecardStamping:
    def test_compute_scorecard_stamps_meta(
        self,
        registry: PromptRegistry,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """compute_scorecard with a fake graph + stub scorers still stamps all
        provenance fields — this is what keeps README claims honest."""
        from evaluation import evaluators, run_ragas

        monkeypatch.setenv("PINECONE_API_KEY", "")
        monkeypatch.setattr(
            evaluators, "default_scorers", lambda: _scorers()
        )
        card = run_ragas.compute_scorecard(graph=FakeGraph())
        assert card["faithfulness"] == pytest.approx(0.9)
        meta = card["meta"]
        assert meta["dataset_version"] == dataset_version()
        assert meta["prompt_versions"]["generate_answer"] == registry.promoted(
            "generate_answer"
        ).version
        # whichever backend the local .env configures is stamped honestly
        assert meta["vector_backend"] in {"chroma", "pinecone"}
        assert card["cases_count"] == len(SAMPLE_CASES)

    def test_non_hybrid_mode_rejected(self, registry: PromptRegistry) -> None:
        from evaluation import run_ragas

        with pytest.raises(NotImplementedError):
            run_ragas.compute_scorecard(
                retrieval_mode="vector_only", graph=FakeGraph()
            )
