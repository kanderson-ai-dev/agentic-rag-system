"""Manual ingestion script for the hybrid retrieval backends.

Seeds the vector store (Chroma by default, Pinecone if configured) and the graph
store (a JSON file for the local NetworkX fallback, Neo4j if configured) from the
demo corpus. Running this once leaves the system fully functional with or without
external accounts:

    python scripts/ingest_knowledge_base.py
"""

import json
from pathlib import Path

from app.core.config import get_settings
from app.services.knowledge_base import SAMPLE_DOCUMENTS

_RELATED_EDGES = [
    ("langgraph-overview", "self-rag-pattern"),
    ("langgraph-overview", "langgraph-hitl"),
    ("langgraph-overview", "langgraph-checkpointing"),
    ("self-rag-pattern", "langsmith-evaluation"),
]


def _build_knowledge_graph() -> dict:
    """Extract a simple Concept -[:RELATED_TO]-> Concept structure from the corpus."""
    nodes = [
        {
            "id": doc.metadata.get("source", f"doc-{index}"),
            "label": doc.metadata.get("topic", doc.metadata.get("source", "")),
            "description": doc.page_content,
        }
        for index, doc in enumerate(SAMPLE_DOCUMENTS)
    ]
    return {
        "directed": False,
        "multigraph": False,
        "nodes": nodes,
        "links": [
            {"source": left, "target": right, "relation": "RELATED_TO"}
            for left, right in _RELATED_EDGES
        ],
    }


def _write_local_graph(settings) -> Path:
    graph = _build_knowledge_graph()
    path = Path(settings.knowledge_graph_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return path


def _seed_pinecone(settings) -> None:
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model_name,
        api_key=settings.openai_api_key_value(),
    )
    pc = Pinecone(api_key=settings.pinecone_api_key_value())
    index = pc.Index(settings.pinecone_index_name)
    PineconeVectorStore(index=index, embedding=embeddings).add_documents(SAMPLE_DOCUMENTS)


def _seed_neo4j(settings, graph: dict) -> None:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password_value() or ""),
    )
    with driver.session() as session:
        for node in graph["nodes"]:
            session.run(
                "MERGE (n:Concept {id: $id}) "
                "SET n.label = $label, n.description = $description",
                id=node["id"],
                label=node["label"],
                description=node["description"],
            )
        for link in graph["links"]:
            session.run(
                "MATCH (a:Concept {id: $source}), (b:Concept {id: $target}) "
                "MERGE (a)-[:RELATED_TO]->(b)",
                source=link["source"],
                target=link["target"],
            )
    driver.close()


def main() -> None:
    settings = get_settings()

    if settings.pinecone_api_key:
        _seed_pinecone(settings)
        print("Vector store seeded: pinecone")
    else:
        from app.services.vector_store import build_vector_store_service

        build_vector_store_service(settings)
        print("Vector store seeded: chroma")

    graph_path = _write_local_graph(settings)
    print(f"Knowledge graph written: {graph_path}")

    if settings.neo4j_uri and settings.neo4j_password:
        _seed_neo4j(settings, _build_knowledge_graph())
        print("Graph store seeded: neo4j")

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
