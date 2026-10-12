"""Tests for hybrid retrieval: backend selection, fallback and combined node."""

from langchain_core.documents import Document

from app.core.config import Settings
from app.graph.nodes import make_hybrid_retrieve_node
from app.services.graph_store import (
    NetworkXGraphStoreService,
    build_graph_store_service,
)
from tests.conftest import RELEVANT_DOC, FakeRetriever


def _state(**overrides):
    base = {
        "question": "what is langgraph?",
        "generation": "",
        "documents": [],
        "web_search_needed": False,
        "retry_count": 0,
        "human_decision": None,
        "blocked": False,
        "output_flagged": False,
    }
    base.update(overrides)
    return base


class _FakeService:
    def __init__(self, backend: str) -> None:
        self.backend = backend


class TestVectorStoreSelection:
    def test_defaults_to_chroma_without_pinecone_key(self, monkeypatch) -> None:
        import app.services.vector_store as vs

        monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", lambda **kw: object())
        monkeypatch.setattr(vs, "ChromaVectorStoreService", lambda **kw: _FakeService("chroma"))
        monkeypatch.setattr(vs, "PineconeVectorStoreService", lambda **kw: _FakeService("pinecone"))

        service = vs.build_vector_store_service(Settings(pinecone_api_key=None))
        assert service.backend == "chroma"

    def test_uses_pinecone_when_key_present(self, monkeypatch) -> None:
        import app.services.vector_store as vs

        monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", lambda **kw: object())
        monkeypatch.setattr(vs, "ChromaVectorStoreService", lambda **kw: _FakeService("chroma"))
        monkeypatch.setattr(vs, "PineconeVectorStoreService", lambda **kw: _FakeService("pinecone"))

        service = vs.build_vector_store_service(Settings(pinecone_api_key="key"))
        assert service.backend == "pinecone"


class TestGraphStoreSelection:
    def test_defaults_to_networkx_without_neo4j(self) -> None:
        service = build_graph_store_service(Settings(neo4j_uri=None, neo4j_password=None))
        assert service.backend == "networkx"

    def test_uses_neo4j_when_configured(self) -> None:
        service = build_graph_store_service(
            Settings(neo4j_uri="bolt://localhost:7687", neo4j_password="secret")
        )
        assert service.backend == "neo4j"


class TestNetworkXFallback:
    def test_graph_search_returns_matching_documents(self, tmp_path) -> None:
        service = NetworkXGraphStoreService(str(tmp_path / "missing.json"))
        results = service.graph_search("langgraph")
        assert results
        assert all(doc.metadata["source"] == "graph" for doc in results)
        assert all(doc.metadata["backend"] == "networkx" for doc in results)


class TestHybridRetrieveNode:
    def test_combines_vector_and_graph_results(self) -> None:
        graph_doc = Document(
            page_content="graph result about langgraph",
            metadata={"source": "graph", "backend": "networkx"},
        )
        node = make_hybrid_retrieve_node(
            FakeRetriever([RELEVANT_DOC]),
            graph_search_fn=lambda _q: [graph_doc],
        )

        result = node(_state())

        assert len(result["documents"]) == 2
        assert result["documents"][0] == RELEVANT_DOC
        assert result["documents"][1].metadata["source"] == "graph"


class TestRetrievalModeIsolation:
    """Each mode must touch only its own backend — the baseline numbers are
    meaningless if a "vector_only" arm secretly queries the graph store."""

    def _recording_backends(self):
        calls = {"vector": 0, "graph": 0}

        class _RecordingRetriever:
            def invoke(self, _q: str) -> list[Document]:
                calls["vector"] += 1
                return [RELEVANT_DOC]

        def _graph_search(_q: str) -> list[Document]:
            calls["graph"] += 1
            return [Document(page_content="graph doc", metadata={"source": "g"})]

        return calls, _RecordingRetriever(), _graph_search

    def test_hybrid_calls_both_backends(self) -> None:
        calls, retriever, graph_search = self._recording_backends()
        node = make_hybrid_retrieve_node(
            retriever, graph_search, retrieval_mode="hybrid"
        )
        result = node(_state())
        assert calls == {"vector": 1, "graph": 1}
        assert len(result["documents"]) == 2

    def test_vector_only_never_touches_graph(self) -> None:
        calls, retriever, graph_search = self._recording_backends()
        node = make_hybrid_retrieve_node(
            retriever, graph_search, retrieval_mode="vector_only"
        )
        result = node(_state())
        assert calls == {"vector": 1, "graph": 0}
        assert result["documents"] == [RELEVANT_DOC]

    def test_graph_only_never_touches_vector(self) -> None:
        calls, retriever, graph_search = self._recording_backends()
        node = make_hybrid_retrieve_node(
            retriever, graph_search, retrieval_mode="graph_only"
        )
        result = node(_state())
        assert calls == {"vector": 0, "graph": 1}
        assert len(result["documents"]) == 1

    def test_no_retrieval_calls_neither_backend(self) -> None:
        calls, retriever, graph_search = self._recording_backends()
        node = make_hybrid_retrieve_node(
            retriever, graph_search, retrieval_mode="no_retrieval"
        )
        assert node(_state())["documents"] == []
        assert calls == {"vector": 0, "graph": 0}
