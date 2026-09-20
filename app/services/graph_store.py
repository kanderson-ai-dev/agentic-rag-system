"""Graph store service: Neo4j (cloud) with a local NetworkX fallback.

This is the second retrieval source in the hybrid pipeline. When Neo4j is
configured (`NEO4J_URI` + password), queries run against it using parameterized
Cypher (anti Cypher-injection). Otherwise a local `networkx` graph loaded from a
JSON file is used, so graph search always produces real results without any
external account.
"""

import json
import re
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document

from app.core.config import Settings

# Cypher keywords rejected by `sanitize_graph_term` as defense-in-depth on top
# of parameterization.
_CYPHER_KEYWORDS = {
    "MATCH",
    "MERGE",
    "CREATE",
    "DELETE",
    "DETACH",
    "REMOVE",
    "SET",
    "DROP",
    "RETURN",
    "WITH",
    "WHERE",
    "UNWIND",
    "CALL",
    "YIELD",
    "LOAD",
    "CSV",
    "FOREACH",
    "UNION",
    "LIMIT",
    "SKIP",
    "OPTIONAL",
}

_MAX_TERM_LENGTH = 200


def sanitize_graph_term(term: str) -> str:
    """Sanitize a graph-search term via an allow-list and keyword rejection.

    Strips characters outside a conservative allow-list, caps length, and raises
    `ValueError` if the term contains a Cypher keyword. This is defense-in-depth
    on top of always using parameterized queries.
    """
    cleaned = "".join(char for char in term if char.isalnum() or char in " -_.")
    cleaned = " ".join(cleaned.split())[:_MAX_TERM_LENGTH]
    for keyword in _CYPHER_KEYWORDS:
        if re.search(rf"\b{keyword}\b", cleaned, re.IGNORECASE):
            raise ValueError(f"forbidden graph-search term: {keyword}")
    return cleaned


class GraphStoreService(Protocol):
    """Interface exposed by every graph store backend."""

    backend: str

    def graph_search(self, term: str) -> list[Document]: ...


class NetworkXGraphStoreService:
    """Local in-memory graph store (default backend, no external account)."""

    backend = "networkx"

    def __init__(self, graph_path: str) -> None:
        import networkx as nx

        self._graph = self._load(nx, Path(graph_path))

    def _load(self, nx, path: Path):
        if path.exists():
            try:
                return nx.node_link_graph(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # fall through to the default graph on a corrupt file
        return self._default_graph(nx)

    def _default_graph(self, nx):
        """Build a small graph from the demo corpus so the fallback works
        out of the box without running the ingest script."""
        from app.services.knowledge_base import SAMPLE_DOCUMENTS

        graph = nx.Graph()
        for doc in SAMPLE_DOCUMENTS:
            source = doc.metadata.get("source", "unknown")
            graph.add_node(
                source,
                label=doc.metadata.get("topic", source),
                description=doc.page_content,
            )
        edges = [
            ("langgraph-overview", "self-rag-pattern"),
            ("langgraph-overview", "langgraph-hitl"),
            ("langgraph-overview", "langgraph-checkpointing"),
            ("self-rag-pattern", "langsmith-evaluation"),
        ]
        for left, right in edges:
            if graph.has_node(left) and graph.has_node(right):
                graph.add_edge(left, right, relation="RELATED_TO")
        return graph

    def graph_search(self, term: str) -> list[Document]:
        try:
            safe = sanitize_graph_term(term)
        except ValueError:
            return []
        if not safe:
            return []
        needle = safe.lower()
        results: list[Document] = []
        seen: set[str] = set()
        for node, data in self._graph.nodes(data=True):
            label = data.get("label", node)
            description = data.get("description", "")
            haystack = f"{node} {label} {description}".lower()
            if needle in haystack:
                content = description or str(node)
                if content in seen:
                    continue
                seen.add(content)
                results.append(
                    Document(
                        page_content=content,
                        metadata={"source": "graph", "backend": self.backend, "node": node},
                    )
                )
        return results


class Neo4jGraphStoreService:
    """Neo4j-backed graph store (cloud, used when configured)."""

    backend = "neo4j"

    def __init__(self, uri: str, username: str, password: str) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        self._driver.close()

    def graph_search(self, term: str) -> list[Document]:
        try:
            safe = sanitize_graph_term(term)
        except ValueError:
            return []
        if not safe:
            return []
        # Parameterized Cypher: user input is bound via `$term`, never
        # interpolated into the query string.
        query = (
            "MATCH (n) "
            "WHERE toLower(coalesce(n.label, '')) CONTAINS toLower($term) "
            "OR toLower(coalesce(n.description, '')) CONTAINS toLower($term) "
            "RETURN n.label AS label, n.description AS description LIMIT 5"
        )
        with self._driver.session() as session:
            records = list(session.run(query, term=safe))
        return [
            Document(
                page_content=record.get("description") or record.get("label") or "",
                metadata={
                    "source": "graph",
                    "backend": self.backend,
                    "node": record.get("label"),
                },
            )
            for record in records
            if record.get("description") or record.get("label")
        ]


def build_graph_store_service(settings: Settings) -> GraphStoreService:
    """Create the graph store service, choosing backend by configuration.

    Uses Neo4j when `NEO4J_URI` is configured, otherwise falls back to a local
    NetworkX graph so graph search always works out of the box.
    """
    if settings.neo4j_uri and settings.neo4j_password:
        return Neo4jGraphStoreService(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password_value() or "",
        )
    return NetworkXGraphStoreService(settings.knowledge_graph_path)
