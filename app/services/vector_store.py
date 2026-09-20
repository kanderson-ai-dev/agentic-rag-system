"""Vector store service.

Wraps the concrete vector store implementation behind a small factory so the
rest of the application only depends on the `VectorStoreRetriever` interface
from `langchain_core`. Swapping `InMemoryVectorStore` for a persistent store
(Chroma, pgvector, Pinecone, ...) only requires changing this module.
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever

from app.core.config import Settings
from app.services.knowledge_base import SAMPLE_DOCUMENTS


class VectorStoreService:
    """Builds and holds the retriever used by the Self-RAG graph."""

    def __init__(self, embeddings: Embeddings, documents: list[Document] | None = None) -> None:
        self._store = InMemoryVectorStore(embeddings)
        self._store.add_documents(documents if documents is not None else SAMPLE_DOCUMENTS)

    def as_retriever(self, top_k: int = 4) -> VectorStoreRetriever:
        return self._store.as_retriever(search_kwargs={"k": top_k})


def build_vector_store_service(settings: Settings) -> VectorStoreService:
    """Create a `VectorStoreService` seeded with the demo knowledge base."""
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model_name,
        api_key=settings.openai_api_key_value(),
    )
    return VectorStoreService(embeddings=embeddings)
