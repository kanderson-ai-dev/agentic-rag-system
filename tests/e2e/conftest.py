"""End-to-end fixtures: a real uvicorn server plus a Playwright browser.

Two modes:

- **Deterministic** (default, ``-m e2e``): the app's ``get_graph`` dependency is
  overridden with ``build_test_graph`` — real guardrail/edges/HITL interrupt,
  fake chains and retrievers. No keys, no network beyond localhost. Skipped
  entirely when Playwright or its chromium build is missing.
- **Live** (``E2E_LIVE=1``, ``tests/e2e/test_live.py``): no overrides — the
  lifespan builds the real graph from ``.env`` keys; observed numbers feed
  ``docs/functional-qa/CHECKLIST.md``.

The suite's default addopts exclude the ``e2e`` marker, so ``pytest`` stays
offline and fast; these only run on explicit opt-in.
"""

import os
import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.documents import Document

pw = pytest.importorskip(
    "playwright.sync_api",
    reason="Playwright not installed — maintainer-only e2e tooling",
)
sync_playwright = pw.sync_playwright

from app.core.config import get_settings  # noqa: E402
from app.core.dependencies import get_graph  # noqa: E402
from app.main import app  # noqa: E402
from tests.conftest import (  # noqa: E402
    IRRELEVANT_DOC,
    RELEVANT_DOC,
    build_test_graph,
)

LIVE = os.getenv("E2E_LIVE") == "1"

# Env keys the deterministic server must not see — with them absent the
# lifespan falls back to local stores and the stub graph decides everything.
_KEY_VARS = [
    "OPENAI_API_KEY",
    "PINECONE_API_KEY",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_TRACING_V2",
]


class TopicRetriever:
    """Routes by keyword so one graph can serve several scenarios:

    - in-domain questions (``langgraph``, ``self-rag``, ``hitl``) get a
      relevant document and answer in one hop;
    - anything else gets an irrelevant document, fails grading, and exhausts
      the rewrite retries — driving a real human-review interrupt.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, question: str) -> list[Document]:
        self.calls.append(question)
        q = question.lower()
        if any(k in q for k in ("langgraph", "self-rag", "hitl", "human review")):
            return [RELEVANT_DOC]
        return [IRRELEVANT_DOC]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Server:
    """uvicorn wrapped so tests can start/stop it like a fixture."""

    def __init__(self, port: int) -> None:
        import uvicorn

        config = uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self.base_url = f"http://127.0.0.1:{port}"

    def __enter__(self) -> "_Server":
        self._thread.start()
        import urllib.request

        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"{self.base_url}/health/live", timeout=1)
                return self
            except OSError:
                time.sleep(0.2)
        raise RuntimeError("e2e server did not start within 30s")

    def __exit__(self, *exc: Any) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


@pytest.fixture(scope="session")
def stub_server() -> Iterator[str]:
    """Deterministic app: real graph edges, fake chains — no external keys."""
    saved = {k: os.environ.pop(k, None) for k in _KEY_VARS}
    os.environ["TRACING_PROVIDER"] = "none"
    get_settings.cache_clear()
    retriever = TopicRetriever()
    # The budget gate consults this callable per request — the dedicated test
    # flips E2E_BUDGET_EXHAUSTED mid-session to exercise the 429 path.
    graph = build_test_graph(
        retriever=retriever,
        max_retries=2,
        budget_check=lambda: os.environ.get("E2E_BUDGET_EXHAUSTED") == "1",
    )
    app.dependency_overrides[get_graph] = lambda: graph
    try:
        with _Server(_free_port()) as server:
            yield server.base_url
    finally:
        app.dependency_overrides.pop(get_graph, None)
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value
        get_settings.cache_clear()


@pytest.fixture(scope="session")
def live_server() -> Iterator[str]:
    """Real app with real keys — used only by the opt-in live suite."""
    if not LIVE:
        pytest.skip("set E2E_LIVE=1 to run the live functional QA suite")
    get_settings.cache_clear()
    with _Server(_free_port()) as server:
        yield server.base_url


@pytest.fixture(scope="session")
def browser() -> Iterator[Any]:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            yield browser
            browser.close()
    except Exception as exc:  # browser binaries missing
        pytest.skip(f"chromium unavailable: {exc}")


@pytest.fixture()
def page(browser: Any) -> Iterator[Any]:
    context = browser.new_context(
        viewport={"width": 1280, "height": 900}, color_scheme="dark"
    )
    page = context.new_page()
    yield page
    context.close()


def wait_for_send(page: Any, button: str, timeout_ms: int = 60_000) -> None:
    """One ask cycle: busy → idle.

    Guardrail-blocked responses land in <1s — the disabled state can flicker
    past between polls, so a missed busy-signal is tolerated as "already done";
    callers still wait for rendered content before asserting.
    """
    import contextlib

    with contextlib.suppress(Exception):
        # response may be faster than the busy-state poll
        page.wait_for_selector(f"{button}[disabled]", timeout=2_000)
    page.wait_for_selector(f"{button}:not([disabled])", timeout=timeout_ms)


pytestmark = pytest.mark.e2e
