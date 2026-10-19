"""LLM client and chain factories.

Every chain returned here only needs to expose an `.invoke()` method, which
keeps the graph nodes trivially testable with hand-written stub objects
instead of real language models.
"""

from typing import Any, Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.graph.prompts import RegistryPrompt
from app.services.prompt_registry import PromptName

# Hard cap on generated answer length. The prompt asks for concise but
# complete answers; the bound only protects tail latency.
GENERATION_MAX_TOKENS = 512


class GradeDocuments(BaseModel):
    """Structured output for the batched document relevance grader."""

    scores: list[Literal["yes", "no"]] = Field(
        description=(
            "One 'yes'/'no' verdict per retrieved document, in the same order "
            "the documents were provided."
        )
    )


def build_chat_model(
    settings: Settings, callback_handler: BaseCallbackHandler | None = None
) -> BaseChatModel:
    """Build the ChatOpenAI model used across all chains."""
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": settings.chat_model_name,
        "temperature": settings.chat_model_temperature,
        "api_key": settings.openai_api_key_value(),
    }
    if callback_handler is not None:
        kwargs["callbacks"] = [callback_handler]
    return ChatOpenAI(**kwargs)


def build_grader_chain(llm: BaseChatModel) -> Runnable[Any, Any]:
    """Chain that grades all retrieved documents in a single LLM call.

    The prompt resolves lazily from the registry on every invoke, so a
    promotion takes effect without rebuilding the chain or the graph.
    """
    structured_llm = llm.with_structured_output(GradeDocuments)
    return RegistryPrompt(PromptName.GRADE_DOCUMENTS) | structured_llm


def build_generation_chain(llm: BaseChatModel) -> Runnable[Any, Any]:
    """Chain that generates the final answer from relevant context."""
    bounded_llm = llm.bind(max_tokens=GENERATION_MAX_TOKENS)
    return RegistryPrompt(PromptName.GENERATE_ANSWER) | bounded_llm | StrOutputParser()


def build_rewriter_chain(llm: BaseChatModel) -> Runnable[Any, Any]:
    """Chain that rewrites the question to improve retrieval."""
    return RegistryPrompt(PromptName.TRANSFORM_QUERY) | llm | StrOutputParser()
