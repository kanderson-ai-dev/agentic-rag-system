"""Pydantic schemas for the v1 API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for `POST /api/v1/rag/query`."""

    question: str = Field(..., min_length=1, description="The user's natural language question.")
    thread_id: str | None = Field(
        default=None,
        description="Existing conversation/thread id. A new one is generated if omitted.",
    )


class InterruptPayload(BaseModel):
    """Payload surfaced to the caller when the graph pauses for human review."""

    reason: str
    question: str
    retry_count: int
    best_documents: list[str]


class QueryResponse(BaseModel):
    """Response body shared by the query and review endpoints."""

    status: Literal["completed", "interrupted"]
    thread_id: str
    answer: str | None = None
    sources: list[str] = Field(default_factory=list)
    retries: int = 0
    interrupt: InterruptPayload | None = None

    @classmethod
    def from_graph_result(cls, result: dict[str, Any], thread_id: str) -> "QueryResponse":
        """Build a response from a raw LangGraph invocation result."""
        interrupts = result.get("__interrupt__")
        if interrupts:
            payload = interrupts[0].value
            return cls(
                status="interrupted",
                thread_id=thread_id,
                retries=payload.get("retry_count", 0),
                interrupt=InterruptPayload(**payload),
            )

        documents = result.get("documents", [])
        return cls(
            status="completed",
            thread_id=thread_id,
            answer=result.get("generation", ""),
            sources=[doc.metadata.get("source", "unknown") for doc in documents],
            retries=result.get("retry_count", 0),
        )


class HumanReviewRequest(BaseModel):
    """Request body for `POST /api/v1/rag/query/{thread_id}/review`."""

    decision: Literal["approve", "retry", "override"]
    revised_question: str | None = Field(
        default=None, description="Required when decision is 'retry'."
    )
    override_answer: str | None = Field(
        default=None, description="Required when decision is 'override'."
    )
