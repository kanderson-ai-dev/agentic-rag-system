"""Security tests for the graph store (anti Cypher-injection)."""

import pytest

from app.core.cache import TTLCache
from app.services.graph_store import (
    Neo4jGraphStoreService,
    NetworkXGraphStoreService,
    sanitize_graph_term,
)


class TestSanitizeGraphTerm:
    def test_allows_benign_term(self) -> None:
        assert sanitize_graph_term("langgraph orchestration") == "langgraph orchestration"

    def test_strips_dangerous_characters(self) -> None:
        # Quotes, semicolons and backticks are stripped by the allow-list.
        assert sanitize_graph_term('langgraph"`; x') == "langgraph x"

    def test_caps_length(self) -> None:
        assert len(sanitize_graph_term("a" * 1000)) == 200

    @pytest.mark.parametrize("keyword", ["MATCH", "delete", "MERGE", "Create"])
    def test_rejects_cypher_keywords(self, keyword: str) -> None:
        with pytest.raises(ValueError):
            sanitize_graph_term(f"hello {keyword} world")


class _FakeSession:
    def __init__(self) -> None:
        self.query = None
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False

    def run(self, query, **params):
        self.query = query
        self.params = params
        return []


class _FakeDriver:
    def __init__(self) -> None:
        self.session_obj = _FakeSession()

    def session(self):
        return self.session_obj

    def close(self) -> None:
        pass


def _neo4j_service() -> Neo4jGraphStoreService:
    # Bypass __init__ to avoid importing/connecting a real driver.
    service = Neo4jGraphStoreService.__new__(Neo4jGraphStoreService)
    service._driver = _FakeDriver()
    service._cache = TTLCache()
    return service


class TestNeo4jInjectionDefense:
    def test_uses_parameterized_query(self) -> None:
        service = _neo4j_service()
        service.graph_search("langgraph")
        assert service._driver.session_obj.query is not None
        assert "$term" in service._driver.session_obj.query
        assert service._driver.session_obj.params == {"term": "langgraph"}

    def test_rejects_cypher_injection_without_running_query(self) -> None:
        service = _neo4j_service()
        result = service.graph_search("MATCH (n) RETURN n")
        assert result == []
        assert service._driver.session_obj.query is None


def test_networkx_rejects_cypher_injection_without_error(tmp_path) -> None:
    service = NetworkXGraphStoreService(str(tmp_path / "missing.json"))
    assert service.graph_search("MATCH (n) DELETE n") == []
