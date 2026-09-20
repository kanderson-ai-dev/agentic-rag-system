"""Application configuration.

Settings are loaded from environment variables (and an optional `.env` file)
using pydantic-settings. This is the single source of truth for runtime
configuration across the API layer, the LangGraph workflow and the
evaluation scripts.
"""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Agentic RAG service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "Agentic RAG & Knowledge Systems"
    environment: str = "development"
    log_level: str = "INFO"

    # --- OpenAI / LLM ---
    openai_api_key: SecretStr | None = None
    chat_model_name: str = "gpt-4o-mini"
    chat_model_temperature: float = 0.0
    embedding_model_name: str = "text-embedding-3-small"

    # --- Hybrid retrieval: vector store (Pinecone cloud / Chroma local) ---
    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str = "agentic-rag-system"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    chroma_persist_dir: str = "data/chroma"

    # --- Hybrid retrieval: graph store (Neo4j cloud / NetworkX local) ---
    neo4j_uri: str | None = None
    neo4j_username: str = "neo4j"
    neo4j_password: SecretStr | None = None
    knowledge_graph_path: str = "data/knowledge_graph.json"

    # --- Authentication (single-user JWT, optional) ---
    jwt_secret_key: SecretStr | None = None
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    auth_username: str | None = None
    auth_password_hash: SecretStr | None = None

    # --- Self-RAG graph ---
    max_retries: int = 2
    retriever_top_k: int = 4

    # --- Checkpointing (required for human-in-the-loop interrupts) ---
    checkpoint_db_path: str = "data/checkpoints.sqlite"

    # --- LangSmith observability / evaluation-driven development ---
    langchain_tracing_v2: bool = False
    langchain_project: str = "agentic-rag-system"
    langchain_endpoint: str = "https://api.smith.langchain.com"
    # `LANGCHAIN_API_KEY` is the standard LangChain/LangSmith env var name.
    langsmith_api_key: SecretStr | None = Field(
        default=None, validation_alias="LANGCHAIN_API_KEY"
    )

    @field_validator(
        "openai_api_key",
        "langsmith_api_key",
        "pinecone_api_key",
        "neo4j_password",
        "jwt_secret_key",
        "auth_password_hash",
        mode="before",
    )
    @classmethod
    def _empty_str_to_none(cls, value: object) -> object:
        """Treat empty strings as `None` so an unconfigured CI secret (which
        arrives as an empty string) never reads as a real key.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def openai_api_key_value(self) -> str | None:
        """Return the plaintext OpenAI API key, or `None` if not configured."""
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None

    def pinecone_api_key_value(self) -> str | None:
        """Return the plaintext Pinecone API key, or `None` if not configured."""
        return self.pinecone_api_key.get_secret_value() if self.pinecone_api_key else None

    def neo4j_password_value(self) -> str | None:
        """Return the plaintext Neo4j password, or `None` if not configured."""
        return self.neo4j_password.get_secret_value() if self.neo4j_password else None

    def jwt_secret_key_value(self) -> str | None:
        """Return the plaintext JWT signing secret, or `None` if not configured."""
        return self.jwt_secret_key.get_secret_value() if self.jwt_secret_key else None

    def auth_password_hash_value(self) -> str | None:
        """Return the bcrypt password hash, or `None` if not configured."""
        return self.auth_password_hash.get_secret_value() if self.auth_password_hash else None

    def configure_langsmith_env(self) -> None:
        """Propagate LangSmith settings to the environment variables that
        LangChain/LangGraph read implicitly for tracing.
        """
        import os

        if self.langchain_tracing_v2:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
            os.environ["LANGCHAIN_ENDPOINT"] = self.langchain_endpoint
            if self.langsmith_api_key:
                os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
