"""Lifespan resource lifecycle — backend handles must not leak (I-22)."""

from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app


class _StubGraphStore:
    backend = "stub"

    def __init__(self, closed: list[str]) -> None:
        self._closed = closed

    def graph_search(self, term: str) -> list:
        return []

    def close(self) -> None:
        self._closed.append("closed")


def test_lifespan_closes_graph_store_on_shutdown(monkeypatch) -> None:
    """The Neo4j driver holds a connection pool — closing it is the point;
    the local backends close() is a no-op so the call is always safe."""
    closed: list[str] = []
    monkeypatch.setattr(
        main_module,
        "build_graph_store_service",
        lambda settings: _StubGraphStore(closed),
    )

    with TestClient(app):
        pass  # entering/exiting runs the full lifespan

    assert closed == ["closed"]
    assert not hasattr(app.state, "graph_store")
