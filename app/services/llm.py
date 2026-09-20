"""LLM client and chain factories.

Every chain returned here only needs to expose an `.invoke()` method, which
keeps the graph nodes trivially testable with hand-written stub objects
instead of real language models.
"""

from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.graph.prompts import (
    GENERATE_ANSWER_PROMPT,
    GRADE_DOCUMENTS_PROMPT,
    TRANSFORM_QUERY_PROMPT,
)


class GradeDocuments(BaseModel):
    """Structured output for the document relevance grader."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the question, otherwise 'no'."
    )


def build_chat_model(settings: Settings, callback_handler=None):
    """Build the ChatOpenAI model used across all chains."""
    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": settings.chat_model_name,
        "temperature": settings.chat_model_temperature,
        "api_key": settings.openai_api_key_value(),
    }
    if callback_handler is not None:
        kwargs["callbacks"] = [callback_handler]
    return ChatOpenAI(**kwargs)


def build_grader_chain(llm) -> Runnable:
    """Chain that grades a single document's relevance to a question."""
    structured_llm = llm.with_structured_output(GradeDocuments)
    return GRADE_DOCUMENTS_PROMPT | structured_llm


def build_generation_chain(llm) -> Runnable:
    """Chain that generates the final answer from relevant context."""
    return GENERATE_ANSWER_PROMPT | llm | StrOutputParser()


def build_rewriter_chain(llm) -> Runnable:
    """Chain that rewrites the question to improve retrieval."""
    return TRANSFORM_QUERY_PROMPT | llm | StrOutputParser()
