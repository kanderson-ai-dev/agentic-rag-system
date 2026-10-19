"""Concurrency safety for token accounting and the usage store (I-10, I-11)."""

import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from app.core.cost_tracking import TokenUsageCallbackHandler
from app.core.dependencies import get_graph
from app.main import app
from app.services.usage_store import UsageStore
from tests.conftest import RELEVANT_DOC, build_test_graph


class _BarrierChain:
    """Reports per-question token counts through the shared callback.

    A real LangChain call ends in ``on_llm_end`` on the registered handler —
    the app-wide ``token_handler``. The barrier forces both requests to be
    inside the graph at the same time, which is when a shared accumulator
    would mix their counts.
    """

    def __init__(self, barrier: threading.Barrier) -> None:
        self._barrier = barrier

    def invoke(self, inputs: dict, config: dict | None = None) -> str:
        self._barrier.wait(timeout=15)

        class _Response:
            llm_output = {
                "token_usage": {
                    "prompt_tokens": len(inputs["question"]),
                    "completion_tokens": 0,
                }
            }

        app.state.token_handler.on_llm_end(_Response())
        return "answer"


def test_concurrent_requests_record_their_own_token_counts(tmp_path) -> None:
    barrier = threading.Barrier(2)
    graph = build_test_graph(
        [RELEVANT_DOC], generation_chain=_BarrierChain(barrier)
    )
    store = UsageStore(str(tmp_path / "usage.sqlite"))
    app.dependency_overrides[get_graph] = lambda: graph
    client = TestClient(app)
    app.state.usage_store = store
    app.state.token_handler = TokenUsageCallbackHandler()
    try:
        results: dict[str, object] = {}

        def ask(thread_id: str, question: str) -> None:
            results[thread_id] = client.post(
                "/api/v1/rag/query",
                json={"question": question, "thread_id": thread_id},
            )

        question_a, question_b = "aaa", "bbbbbbb"
        threads = [
            threading.Thread(target=ask, args=("thread-a", question_a)),
            threading.Thread(target=ask, args=("thread-b", question_b)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
    finally:
        app.dependency_overrides.clear()
        del app.state.usage_store
        del app.state.token_handler
        client.close()

    assert results["thread-a"].status_code == 200
    assert results["thread-b"].status_code == 200
    by_thread = {row["thread_id"]: row for row in store.recent(limit=10)}
    assert by_thread["thread-a"]["prompt_tokens"] == len(question_a)
    assert by_thread["thread-b"]["prompt_tokens"] == len(question_b)


def test_usage_store_records_25_concurrent_rows(tmp_path) -> None:
    store = UsageStore(str(tmp_path / "usage.sqlite"))

    def record(index: int) -> None:
        store.record(
            thread_id=f"t{index}",
            prompt_tokens=index,
            completion_tokens=0,
            cost_usd=0.0,
            latency_ms=1,
            blocked=False,
            escalated=False,
            retry_count=0,
        )

    with ThreadPoolExecutor(max_workers=25) as pool:
        list(pool.map(record, range(25)))

    assert store.summary()["total_requests"] == 25
