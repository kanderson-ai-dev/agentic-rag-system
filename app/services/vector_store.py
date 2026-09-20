"""Vector store service.

Wraps the concrete vector store backend behind a small factory so the rest of
the application only depends on the `.as_retriever()` interface. The backend is
selected automatically: Pinecone when configured, otherwise a local persistent
Chroma store (the default, so the demo works without any external account).
"""

from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.config import Settings
from app.services.knowledge_base import SAMPLE_DOCUMENTS


class ChromaVectorStoreService:
    """Local persistent vector store (default backend, no external account)."""

    backend = "chroma"

    def __init__(self, embeddings: Embeddings, persist_dir: str) -> None:
        from langchain_chroma import Chroma

        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        is_first_run = not any(persist_path.iterdir())
        self._store = Chroma(
            embedding_function=embeddings,
            persist_directory=str(persist_path),
        )
        if is_first_run:
            self._store.add_documents(SAMPLE_DOCUMENTS)

    def as_retriever(self, top_k: int = 4) -> VectorStoreRetriever:
        return self._store.as_retriever(search_kwargs={"k": top_k})


class PineconeVectorStoreService:
    """Pinecone-backed vector store (cloud, used when an API key is configured)."""

    backend = "pinecone"

    def __init__(
        self,
        embeddings: Embeddings,
        api_key: str,
        index_name: str,
    ) -> None:
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone

        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        self._store = PineconeVectorStore(index=index, embedding=embeddings)

    def as_retriever(self, top_k: int = 4) -> VectorStoreRetriever:
        return self._store.as_retriever(search_kwargs={"k": top_k})


def build_vector_store_service(settings: Settings):
    """Create the vector store service, choosing the backend by configuration.

    Uses Pinecone when `PINECONE_API_KEY` is configured, otherwise falls back to
    a local persistent Chroma store seeded with the demo corpus.
    """
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model_name,
        api_key=settings.openai_api_key_value(),
    )

    if settings.pinecone_api_key:
        return PineconeVectorStoreService(
            embeddings=embeddings,
            api_key=settings.pinecone_api_key_value() or "",
            index_name=settings.pinecone_index_name,
        )
    return ChromaVectorStoreService(
        embeddings=embeddings,
        persist_dir=settings.chroma_persist_dir,
    )
