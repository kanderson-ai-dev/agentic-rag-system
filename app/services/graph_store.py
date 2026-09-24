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
from typing import Any, Protocol

from langchain_core.documents import Document

from app.core.cache import TTLCache
from app.core.config import Settings

# Cloud graph queries (Neo4j Aura) are the single largest contributor to
# retrieval latency; results are cached briefly since the demo knowledge
# graph changes only via the ingest script, not per-request.
_GRAPH_SEARCH_CACHE_TTL_SECONDS = 300.0

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

# English stopwords skipped when extracting graph-search terms from a natural
# language question — they would match nearly every node's text.
_GRAPH_SEARCH_STOPWORDS = {
    "what", "how", "does", "the", "and", "for", "are", "with", "that", "this",
    "from", "you", "your", "can", "could", "would", "should", "about", "tell",
    "explain", "give", "show", "which", "who", "when", "where", "why", "there",
    "here", "not", "yes", "any", "all", "some", "into", "out", "over", "under",
    "again", "once", "just", "only", "very", "too", "also", "was", "were", "been",
    "is", "a", "an", "of", "in", "to", "or", "on", "it", "its", "be", "by", "at",
    "as", "do", "if", "then", "than", "so", "such", "i", "we", "they", "me", "my",
    "their", "between", "work", "works", "working", "use", "used", "using",
}

# Cap on extracted terms, so a long question can't fan out into a huge query.
_MAX_GRAPH_TERMS = 6

# Cap on graph-search results, matching the Cypher `LIMIT 5`.
_MAX_GRAPH_RESULTS = 5


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


def extract_graph_terms(text: str) -> list[str]:
    """Reduce a natural-language question to content terms for graph lookup.

    Splits the input into words, drops stopwords and very short tokens, and
    sanitizes each surviving term independently — a single term that trips
    the Cypher-keyword check is skipped rather than voiding the whole search.
    """
    terms: list[str] = []
    for word in text.split():
        if word.lower() in _GRAPH_SEARCH_STOPWORDS:
            continue
        try:
            safe = sanitize_graph_term(word).lower()
        except ValueError:
            continue
        # Length is checked after sanitizing: "(n)" collapses to "n", and a
        # one-character term would substring-match nearly every node.
        if len(safe) >= 3 and safe not in terms:
            terms.append(safe)
    return terms[:_MAX_GRAPH_TERMS]


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

    def _load(self, nx: Any, path: Path) -> Any:
        if path.exists():
            try:
                return nx.node_link_graph(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # fall through to the default graph on a corrupt file
        return self._default_graph(nx)

    def _default_graph(self, nx: Any) -> Any:
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
        terms = extract_graph_terms(term)
        if not terms:
            return []
        results: list[Document] = []
        seen: set[str] = set()
        for node, data in self._graph.nodes(data=True):
            label = data.get("label", node)
            description = data.get("description", "")
            haystack = f"{node} {label} {description}".lower()
            if not any(t in haystack for t in terms):
                continue
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
            if len(results) >= _MAX_GRAPH_RESULTS:
                break
        return results


class Neo4jGraphStoreService:
    """Neo4j-backed graph store (cloud, used when configured)."""

    backend = "neo4j"

    def __init__(self, uri: str, username: str, password: str) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._cache: TTLCache[list[Document]] = TTLCache(
            ttl_seconds=_GRAPH_SEARCH_CACHE_TTL_SECONDS
        )

    def close(self) -> None:
        self._driver.close()

    def graph_search(self, term: str) -> list[Document]:
        terms = extract_graph_terms(term)
        if not terms:
            return []
        cache_key = ",".join(sorted(terms))
        return self._cache.get_or_compute(cache_key, lambda: self._query(terms))

    def _query(self, terms: list[str]) -> list[Document]:
        # Parameterized Cypher: extracted terms are bound via `$terms`, never
        # interpolated into the query string.
        query = (
            "MATCH (n) "
            "WHERE any(term IN $terms WHERE "
            "toLower(coalesce(n.label, '')) CONTAINS term "
            "OR toLower(coalesce(n.description, '')) CONTAINS term) "
            f"RETURN n.label AS label, n.description AS description LIMIT {_MAX_GRAPH_RESULTS}"
        )
        with self._driver.session() as session:
            records = list(session.run(query, terms=terms))
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
