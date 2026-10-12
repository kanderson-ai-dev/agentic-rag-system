"""Unit tests for the provider-agnostic tracing seam (`app.core.tracing`).

Langfuse is never really imported or contacted — a fake module is injected
into ``sys.modules`` so every test runs fully offline.
"""

import sys
import threading
import types
from contextlib import contextmanager
from typing import Any

import pytest
from pydantic import SecretStr

from app.core import tracing
from app.core.config import Settings


class _FakeSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.metadata: dict[str, Any] = {}
        self.ended = False

    def update(self, **kwargs: Any) -> None:
        self.metadata.update(kwargs.get("metadata") or {})

    def end(self) -> None:
        self.ended = True


class _FakeObservationCtx:
    def __init__(self, client: "_FakeLangfuseClient", name: str, as_type: str) -> None:
        self._client = client
        self._name = name
        self._as_type = as_type
        self.span: _FakeSpan | None = None

    def __enter__(self) -> _FakeSpan:
        self._client.observations.append((self._name, self._as_type))
        self.span = _FakeSpan(self._name)
        self._client._depth += 1
        return self.span

    def __exit__(self, *exc: Any) -> bool:
        self._client._depth -= 1
        if self._client._depth == 0:
            self._client.active_trace_id = None
        return False


class _FakeAttrsCtx:
    def __init__(self, client: "_FakeLangfuseClient", attrs: dict[str, Any]) -> None:
        self._client = client
        client.attribute_calls.append(attrs)

    def __enter__(self) -> "_FakeAttrsCtx":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class _FakeLangfuseClient:
    """Minimal stand-in for `langfuse.Langfuse` — records what it was asked."""

    def __init__(self, *, trace_id: str = "trace-1") -> None:
        self.auth_ok = True
        self.observations: list[tuple[str, str]] = []
        self.attribute_calls: list[dict[str, Any]] = []
        self.active_trace_id: str | None = None
        self._depth = 0
        self._trace_id = trace_id
        self.flushed = False

    def auth_check(self) -> bool:
        return self.auth_ok

    def start_as_current_observation(
        self, *, name: str, as_type: str, **kwargs: Any
    ) -> _FakeObservationCtx:
        if self._depth == 0:
            self.active_trace_id = self._trace_id
        return _FakeObservationCtx(self, name, as_type)

    def get_current_trace_id(self) -> str | None:
        return self.active_trace_id

    def flush(self) -> None:
        self.flushed = True

    def shutdown(self) -> None:
        self.flushed = True


def _fake_langfuse_module(client: _FakeLangfuseClient) -> types.ModuleType:
    module = types.ModuleType("langfuse")
    module.Langfuse = lambda **kwargs: client  # type: ignore[attr-defined]
    module.get_client = lambda **kwargs: client  # type: ignore[attr-defined]

    @contextmanager
    def _propagate_attributes(**attrs: Any):
        with _FakeAttrsCtx(client, attrs):
            yield

    module.propagate_attributes = _propagate_attributes  # type: ignore[attr-defined]
    return module


def _langchain_module(handler_cls: type | None = None) -> types.ModuleType:
    from langchain_core.callbacks.base import BaseCallbackHandler

    module = types.ModuleType("langfuse.langchain")
    module.CallbackHandler = handler_cls or type(  # type: ignore[attr-defined]
        "CallbackHandler", (BaseCallbackHandler,), {}
    )
    return module


@pytest.fixture(autouse=True)
def _reset_tracing_state():
    tracing._reset_state()
    yield
    tracing._reset_state()


@pytest.fixture
def langfuse_settings() -> Settings:
    return Settings(
        tracing_provider="langfuse",
        langfuse_public_key=SecretStr("pk-test"),
        langfuse_secret_key=SecretStr("sk-test"),
        langfuse_base_url="https://cloud.langfuse.com",
    )


def test_provider_defaults_to_none() -> None:
    assert Settings.model_fields["tracing_provider"].default == "none"


def test_langfuse_configured_requires_both_keys(langfuse_settings: Settings) -> None:
    assert tracing.langfuse_configured(langfuse_settings)
    assert not tracing.langfuse_configured(
        Settings(
            tracing_provider="langfuse",
            langfuse_public_key=SecretStr("pk"),
            langfuse_secret_key=None,
        )
    )
    assert not tracing.langfuse_configured(
        Settings(tracing_provider="none", langfuse_public_key=SecretStr("pk"),
                 langfuse_secret_key=SecretStr("sk"))
    )


def test_init_tracing_none_is_free() -> None:
    assert tracing.init_tracing(Settings(tracing_provider="none")) == "none"
    assert tracing.active_provider() == "none"


def test_init_tracing_langfuse_missing_keys_degrades(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "langfuse", _fake_langfuse_module(_FakeLangfuseClient()))
    with caplog.at_level("WARNING"):
        result = tracing.init_tracing(
            Settings(
                tracing_provider="langfuse",
                langfuse_public_key=None,
                langfuse_secret_key=None,
            )
        )
    assert result == "none"
    assert tracing.active_provider() == "none"


def test_init_tracing_langfuse_auth_failure_degrades(
    monkeypatch: pytest.MonkeyPatch, langfuse_settings: Settings
) -> None:
    client = _FakeLangfuseClient()
    client.auth_ok = False
    monkeypatch.setitem(sys.modules, "langfuse", _fake_langfuse_module(client))
    assert tracing.init_tracing(langfuse_settings) == "none"


def test_init_tracing_langfuse_init_exception_degrades(
    monkeypatch: pytest.MonkeyPatch, langfuse_settings: Settings
) -> None:
    module = types.ModuleType("langfuse")
    module.Langfuse = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[attr-defined]
    module.get_client = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", module)
    assert tracing.init_tracing(langfuse_settings) == "none"


def test_init_tracing_langfuse_success(
    monkeypatch: pytest.MonkeyPatch, langfuse_settings: Settings
) -> None:
    client = _FakeLangfuseClient()
    monkeypatch.setitem(sys.modules, "langfuse", _fake_langfuse_module(client))
    assert tracing.init_tracing(langfuse_settings) == "langfuse"
    assert tracing.active_provider() == "langfuse"


def test_init_tracing_disables_langsmith_env_for_other_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale LANGCHAIN_TRACING_V2=true can never double-trace."""
    settings = Settings(
        tracing_provider="langfuse",
        langchain_tracing_v2=True,
        langfuse_public_key=None,  # unkeyed → no client built, no network
        langfuse_secret_key=None,
    )
    tracing.init_tracing(settings)
    import os

    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_init_tracing_langsmith_flag_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        tracing_provider="langsmith",
        langchain_tracing_v2=True,
        langsmith_api_key=SecretStr("ls-key"),
    )
    tracing.init_tracing(settings)
    import os

    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert tracing.active_provider() == "langsmith"


def test_build_callbacks_returns_empty_when_disabled() -> None:
    assert tracing.build_callbacks() == []


def test_build_callbacks_returns_fresh_handler_per_call(
    monkeypatch: pytest.MonkeyPatch, langfuse_settings: Settings
) -> None:
    client = _FakeLangfuseClient()
    monkeypatch.setitem(sys.modules, "langfuse", _fake_langfuse_module(client))
    monkeypatch.setitem(sys.modules, "langfuse.langchain", _langchain_module())
    tracing._set_active("langfuse")

    first, second = tracing.build_callbacks(), tracing.build_callbacks()
    assert len(first) == 1 and len(second) == 1
    assert first[0] is not second[0]


def test_build_callbacks_handler_failure_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> None:
        raise RuntimeError("no handler")

    module = types.ModuleType("langfuse.langchain")
    module.CallbackHandler = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse.langchain", module)
    tracing._set_active("langfuse")
    assert tracing.build_callbacks() == []


def test_trace_run_yields_none_when_disabled() -> None:
    with tracing.trace_run(Settings(), name="t") as trace_id:
        assert trace_id is None


def test_trace_run_yields_trace_id_and_attributes(
    monkeypatch: pytest.MonkeyPatch, langfuse_settings: Settings
) -> None:
    client = _FakeLangfuseClient(trace_id="trace-xyz")
    monkeypatch.setitem(sys.modules, "langfuse", _fake_langfuse_module(client))
    tracing._set_active("langfuse")

    with tracing.trace_run(
        langfuse_settings,
        name="rag.query",
        session_id="sess-1",
        metadata={"thread_id": "t-1"},
        tags=["api"],
    ) as trace_id:
        assert trace_id == "trace-xyz"

    assert ("rag.query", "span") in client.observations
    assert client.attribute_calls[0]["session_id"] == "sess-1"
    assert client.attribute_calls[0]["metadata"] == {"thread_id": "t-1"}
    assert client.attribute_calls[0]["trace_name"] == "rag.query"


def test_trace_run_body_exceptions_propagate(
    monkeypatch: pytest.MonkeyPatch, langfuse_settings: Settings
) -> None:
    client = _FakeLangfuseClient()
    monkeypatch.setitem(sys.modules, "langfuse", _fake_langfuse_module(client))
    tracing._set_active("langfuse")

    with (
        pytest.raises(ValueError, match="body failed"),
        tracing.trace_run(langfuse_settings, name="t"),
    ):
        raise ValueError("body failed")


def test_trace_run_setup_failure_yields_none(
    monkeypatch: pytest.MonkeyPatch, langfuse_settings: Settings
) -> None:
    client = _FakeLangfuseClient()
    client.start_as_current_observation = lambda **kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("span start failed")
    )
    monkeypatch.setitem(sys.modules, "langfuse", _fake_langfuse_module(client))
    tracing._set_active("langfuse")

    with tracing.trace_run(langfuse_settings, name="t") as trace_id:
        assert trace_id is None


def test_concurrent_trace_runs_do_not_share_handlers(
    monkeypatch: pytest.MonkeyPatch, langfuse_settings: Settings
) -> None:
    """Each thread gets its own handler + trace id — no cross-contamination."""
    barrier = threading.Barrier(2)
    seen: dict[str, Any] = {}

    class _PerThreadClient(_FakeLangfuseClient):
        def get_current_trace_id(self) -> str | None:
            return f"trace-{threading.current_thread().name}"

    client = _PerThreadClient()
    handler_instances: list[Any] = []
    handler_cls = type("CallbackHandler", (), {})
    monkeypatch.setitem(sys.modules, "langfuse", _fake_langfuse_module(client))
    monkeypatch.setitem(sys.modules, "langfuse.langchain", _langchain_module(handler_cls))
    tracing._set_active("langfuse")

    def worker(name: str) -> None:
        callbacks = tracing.build_callbacks()
        handler_instances.extend(callbacks)
        with tracing.trace_run(langfuse_settings, name="req") as trace_id:
            barrier.wait()
            seen[name] = trace_id

    threads = [threading.Thread(target=worker, args=(n,), name=n) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen == {"a": "trace-a", "b": "trace-b"}
    assert handler_instances[0] is not handler_instances[1]


def test_mask_trace_payload_redacts_and_truncates() -> None:
    payload = {
        "api_key": "sk-live-secret",
        "nested": {"authorization": "Bearer xyz", "note": "ok"},
        "long": "x" * 900,
        "items": ["y" * 600, "short"],
        "count": 3,
    }
    masked = tracing.mask_trace_payload(payload)
    assert masked["api_key"] == "[redacted]"
    assert masked["nested"]["authorization"] == "[redacted]"
    assert masked["nested"]["note"] == "ok"
    assert masked["long"].endswith("[truncated]")
    assert len(masked["long"]) < 600
    assert masked["items"][0].endswith("[truncated]")
    assert masked["items"][1] == "short"
    assert masked["count"] == 3


def test_flush_is_noop_when_disabled() -> None:
    tracing.flush()  # must not raise


def test_flush_calls_client(
    monkeypatch: pytest.MonkeyPatch, langfuse_settings: Settings
) -> None:
    client = _FakeLangfuseClient()
    monkeypatch.setitem(sys.modules, "langfuse", _fake_langfuse_module(client))
    tracing._set_active("langfuse")
    tracing.flush()
    assert client.flushed


# --- API wiring: X-Trace-Id header ---


def test_query_response_carries_trace_id_header(
    monkeypatch: pytest.MonkeyPatch, langfuse_settings: Settings
) -> None:
    """An enabled provider puts the trace id on the wire; none omits it."""
    from fastapi.testclient import TestClient

    import app.main as main_module
    from app.core.dependencies import get_graph
    from app.main import app
    from tests.conftest import RELEVANT_DOC, build_test_graph

    client = _FakeLangfuseClient(trace_id="api-trace-7")
    monkeypatch.setitem(sys.modules, "langfuse", _fake_langfuse_module(client))
    monkeypatch.setitem(sys.modules, "langfuse.langchain", _langchain_module())
    # The lifespan runs init_tracing() — it must resolve langfuse itself.
    monkeypatch.setattr(main_module, "get_settings", lambda: langfuse_settings)

    graph = build_test_graph([RELEVANT_DOC])
    app.dependency_overrides[get_graph] = lambda: graph

    with TestClient(app, raise_server_exceptions=True) as http:
        traced = http.post("/api/v1/rag/query", json={"question": "what is langgraph?"})

    assert traced.status_code == 200
    assert traced.headers.get("X-Trace-Id") == "api-trace-7"

    monkeypatch.setattr(
        main_module, "get_settings", lambda: Settings(tracing_provider="none")
    )
    with TestClient(app, raise_server_exceptions=True) as http:
        plain = http.post("/api/v1/rag/query", json={"question": "what is langgraph?"})
    app.dependency_overrides.clear()

    assert plain.status_code == 200
    assert "x-trace-id" not in plain.headers


def test_query_succeeds_when_trace_setup_fails(
    monkeypatch: pytest.MonkeyPatch, langfuse_settings: Settings
) -> None:
    """A broken provider must never change the HTTP response."""
    from fastapi.testclient import TestClient

    import app.main as main_module
    from app.core.dependencies import get_graph
    from app.main import app
    from tests.conftest import RELEVANT_DOC, build_test_graph

    # Startup succeeds, but the per-request client lookup raises — the
    # request must still complete untraced.
    client = _FakeLangfuseClient()
    module = _fake_langfuse_module(client)
    module.get_client = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("down"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", module)
    monkeypatch.setattr(main_module, "get_settings", lambda: langfuse_settings)

    graph = build_test_graph([RELEVANT_DOC])
    app.dependency_overrides[get_graph] = lambda: graph
    with TestClient(app, raise_server_exceptions=True) as http:
        response = http.post("/api/v1/rag/query", json={"question": "what is langgraph?"})
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "x-trace-id" not in response.headers


# --- Offline Langfuse evaluation path ---


class _FakeDatasetItem:
    def __init__(self, case: dict[str, str]) -> None:
        self.id = case["id"]
        self.input = {"question": case["question"]}
        self.expected_output = case["ground_truth"]
        self.metadata = {
            "id": case["id"],
            "category": case["category"],
            "expected_behavior": case["expected_behavior"],
        }


class _FakeDataset:
    """`dataset.run_experiment` contract: task + evaluators per item."""

    def __init__(self, items: list[_FakeDatasetItem]) -> None:
        self.items = items
        self.experiment_calls: list[dict[str, Any]] = []

    def run_experiment(self, *, name, task, evaluators, max_concurrency, metadata):
        self.experiment_calls.append({"name": name, "metadata": metadata})
        results = []
        for item in self.items:
            output = task(item=item)
            evaluations = [
                ev(
                    input=item.input,
                    output=output,
                    expected_output=item.expected_output,
                    metadata=item.metadata,
                )
                for ev in evaluators
            ]
            results.append({"output": output, "evaluations": evaluations})
        return types.SimpleNamespace(name=name, item_results=results)


class _FakeLangfuseEvalClient(_FakeLangfuseClient):
    """`get_dataset`/`create_dataset*` on top of the trace fakes."""

    def __init__(self, dataset: _FakeDataset) -> None:
        super().__init__()
        self.dataset = dataset
        self._lookups = 0
        self.created_items = 0

    def get_dataset(self, name: str, **kwargs: Any) -> _FakeDataset:
        self._lookups += 1
        if self._lookups == 1:
            raise LookupError("dataset not found")  # first call → create path
        return self.dataset

    def create_dataset(self, *, name: str, description: str | None = None, **kw: Any) -> None:
        pass

    def create_dataset_item(self, *, dataset_name: str, **kwargs: Any) -> None:
        self.created_items += 1


def test_langfuse_evaluation_runs_experiment_offline(
    monkeypatch: pytest.MonkeyPatch, langfuse_settings: Settings
) -> None:
    """`try_langfuse_evaluation` drives task+evaluators over every dataset
    item with the graph stubbed and scorers faked — fully offline."""
    import evaluation.run_evaluation as runner
    from evaluation import evaluators
    from evaluation.dataset import build_dataset

    cases = build_dataset()
    dataset = _FakeDataset([_FakeDatasetItem(c) for c in cases])
    client = _FakeLangfuseEvalClient(dataset)
    monkeypatch.setitem(sys.modules, "langfuse", _fake_langfuse_module(client))
    monkeypatch.setitem(sys.modules, "langfuse.langchain", _langchain_module())
    monkeypatch.setattr(runner, "get_settings", lambda: langfuse_settings)

    class _StubGraph:
        def invoke(self, state: dict, config: dict | None = None) -> dict:
            return {
                "generation": "stub answer",
                "documents": [],
                "blocked": False,
                "prompt_versions": {"generate_answer": 1},
            }

    monkeypatch.setattr(runner, "_graph", lambda token_handler=None: _StubGraph())
    monkeypatch.setattr(
        evaluators,
        "default_scorers",
        lambda: {name: (lambda **kw: 0.9) for name in
                 ("faithfulness", "answer_relevancy", "context_precision",
                  "context_recall", "llm_as_judge")},
    )

    result = runner.try_langfuse_evaluation()

    assert result is None  # provider path returns None (evidence lives in Langfuse)
    assert dataset.experiment_calls, "run_experiment was never invoked"
    # `<prompt_version_tag>-<git sha>-<date>` naming convention (A/B comparable)
    import re

    assert re.fullmatch(r"[a-z0-9-]+-[a-z0-9]+-\d{4}-\d{2}-\d{2}",
                        dataset.experiment_calls[0]["name"])
    assert client.created_items == len(cases)
    assert client.flushed  # buffered spans pushed at end of run


def test_langfuse_evaluation_falls_back_when_misconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Langfuse keys → local scorecard, never a crash."""
    import evaluation.run_evaluation as runner

    monkeypatch.setattr(
        runner,
        "get_settings",
        lambda: Settings(
            tracing_provider="langfuse",
            langfuse_public_key=None,
            langfuse_secret_key=None,
        ),
    )

    class _StubGraph:
        def invoke(self, state: dict, config: dict | None = None) -> dict:
            return {"generation": "x", "documents": [], "prompt_versions": {}}

    from evaluation import evaluators

    monkeypatch.setattr(runner, "_graph", lambda token_handler=None: _StubGraph())
    monkeypatch.setattr(evaluators, "default_scorers", lambda: {})

    result = runner.try_langfuse_evaluation()
    assert isinstance(result, dict)
    assert "cases_count" in result  # local scorecard shape
