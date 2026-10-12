"""Application configuration.

Settings are loaded from environment variables (and an optional `.env` file)
using pydantic-settings. This is the single source of truth for runtime
configuration across the API layer, the LangGraph workflow and the
evaluation scripts.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default cap on accepted question length (OWASP LLM04). Configurable via the
# `MAX_QUESTION_CHARS` environment variable; enforced both at the API boundary
# (422) and inside the guardrail node so evaluation scripts see identical
# behaviour.
DEFAULT_MAX_QUESTION_CHARS = 1000


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

    # --- Cost tracking ---
    cost_input_price_per_1m: float = 0.15  # gpt-4o-mini input price (USD per 1M tokens)
    cost_output_price_per_1m: float = 0.60  # gpt-4o-mini output price (USD per 1M tokens)
    usage_db_path: str = "data/usage.sqlite"
    # Human-vs-Agent comparison on the performance dashboard. These are
    # stated assumptions for the human side of the comparison — an estimate,
    # never a measured value.
    human_minutes_per_query: float = 6.0
    human_hourly_usd: float = 30.0
    # Hard cap on LLM spend per UTC day, for the public demo. Enforced
    # in-graph after the (free) input guardrail and before any LLM call —
    # blocked input never consumes budget. `GET /health` and the console
    # performance panel surface the live status.
    daily_budget_usd: float = 1.00

    # --- Abuse protection ---
    # Per-IP rate limit on the request-path endpoints (`POST /rag/query` and
    # its human-review resume). `/auth/login` keeps its own stricter limit.
    rate_limit_per_minute: int = 20

    # --- Self-RAG graph ---
    max_retries: int = 2
    retriever_top_k: int = 4
    max_question_chars: int = DEFAULT_MAX_QUESTION_CHARS
    # Retrieval backend combination baked into the graph at build time.
    # Config/evaluation-only — deliberately not a request parameter: callers
    # cannot weaken retrieval per query. `no_retrieval` never reaches the
    # graph (the eval harness drives a bare-LLM chain for that baseline).
    retrieval_mode: Literal["hybrid", "vector_only", "graph_only", "no_retrieval"] = (
        "hybrid"
    )

    # --- Prompt registry (versioned prompt artifacts) ---
    prompts_dir: str = "prompts"
    # Escape hatch for open (unauthenticated) deployments: when no JWT secret
    # is configured, registry mutations are refused unless this is true.
    prompt_registry_write_enabled: bool = False
    # A/B API runs spend real tokens on the live eval dataset — hard caps.
    ab_test_max_cost_usd: float = 1.0
    ab_test_timeout_seconds: float = 600.0

    # --- Checkpointing (required for human-in-the-loop interrupts) ---
    checkpoint_db_path: str = "data/checkpoints.sqlite"

    # --- Tracing provider (single switch for observability backends) ---
    # `none` for daily development and offline tests, `langsmith` for the
    # default EDD provider, `langfuse` for the conditional fallback when
    # LangSmith is unusable (quota, missing/invalid key).
    tracing_provider: Literal["none", "langsmith", "langfuse"] = "none"

    # --- LangSmith observability / evaluation-driven development ---
    langchain_tracing_v2: bool = False
    langchain_project: str = "agentic-rag-system"
    langchain_endpoint: str = "https://api.smith.langchain.com"
    # `LANGCHAIN_API_KEY` is the standard LangChain/LangSmith env var name.
    langsmith_api_key: SecretStr | None = Field(
        default=None, validation_alias="LANGCHAIN_API_KEY"
    )

    # --- Langfuse observability (fallback provider) ---
    # Cloud base URL by default; a self-hosted deployment only changes
    # `LANGFUSE_BASE_URL` (e.g. http://localhost:3001).
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"

    @field_validator(
        "openai_api_key",
        "langsmith_api_key",
        "pinecone_api_key",
        "neo4j_password",
        "jwt_secret_key",
        "auth_password_hash",
        "langfuse_public_key",
        "langfuse_secret_key",
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

    def langfuse_public_key_value(self) -> str | None:
        """Return the plaintext Langfuse public key, or `None` if not configured."""
        return self.langfuse_public_key.get_secret_value() if self.langfuse_public_key else None

    def langfuse_secret_key_value(self) -> str | None:
        """Return the plaintext Langfuse secret key, or `None` if not configured."""
        return self.langfuse_secret_key.get_secret_value() if self.langfuse_secret_key else None

    def configure_langsmith_env(self) -> None:
        """Propagate the implicit LangChain tracing env vars for the provider.

        LangSmith env tracing is only exported when ``tracing_provider`` is
        ``"langsmith"`` *and* ``langchain_tracing_v2`` is true. For ``none``
        and ``langfuse`` the flags are forced off so a stale
        ``LANGCHAIN_TRACING_V2=true`` can never double-trace a request or
        silently burn the (quota-limited) LangSmith budget alongside another
        provider.
        """
        import os

        enabled = self.tracing_provider == "langsmith" and self.langchain_tracing_v2
        flag = "true" if enabled else "false"
        os.environ["LANGCHAIN_TRACING_V2"] = flag
        os.environ["LANGSMITH_TRACING"] = flag
        if enabled:
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
            os.environ["LANGCHAIN_ENDPOINT"] = self.langchain_endpoint
            if self.langsmith_api_key:
                os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
