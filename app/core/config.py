"""Application configuration.

Settings are loaded from environment variables (and an optional `.env` file)
using pydantic-settings. This is the single source of truth for runtime
configuration across the API layer, the LangGraph workflow and the
evaluation scripts.
"""

from functools import lru_cache

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
    openai_api_key: str | None = None
    chat_model_name: str = "gpt-4o-mini"
    chat_model_temperature: float = 0.0
    embedding_model_name: str = "text-embedding-3-small"

    # --- Self-RAG graph ---
    max_retries: int = 2
    retriever_top_k: int = 4

    # --- Checkpointing (required for human-in-the-loop interrupts) ---
    checkpoint_db_path: str = "data/checkpoints.sqlite"

    # --- LangSmith observability / evaluation-driven development ---
    langchain_tracing_v2: bool = False
    langchain_project: str = "agentic-rag-system"
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str | None = None

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
                os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
