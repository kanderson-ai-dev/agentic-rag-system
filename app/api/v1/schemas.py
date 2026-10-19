"""Pydantic schemas for the v1 API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import get_settings
from app.graph.prompts import REFUSAL_TEXT

# The closed set of prompt names — `Literal` on path params makes FastAPI
# answer 422 (not a file lookup) for anything else, including `../` names.
PromptNameParam = Literal[
    "grade_documents", "generate_answer", "transform_query", "llm_judge"
]


class QueryRequest(BaseModel):
    """Request body for `POST /api/v1/rag/query`."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=get_settings().max_question_chars,
        description="The user's natural language question.",
    )
    thread_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:-]+$",
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
    blocked: bool = False
    sources: list[str] = Field(default_factory=list)
    retries: int = 0
    interrupt: InterruptPayload | None = None
    # Prompt versions pinned for this thread — the chat footer attributes the
    # answer ("Answered with Answer generation v2") from this map.
    prompt_versions: dict[str, int] = Field(default_factory=dict)

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

        answer = result.get("generation", "")
        # Honest sources: an "I don't know" refusal used none of the retrieved
        # documents, an output-flagged answer was replaced by the safe refusal
        # (grounded in nothing), and the same source can appear twice (vector +
        # graph hit the same document), so report each source once, in order.
        sources: list[str] = []
        if answer.strip() != REFUSAL_TEXT and not result.get("output_flagged"):
            seen: set[str] = set()
            for doc in result.get("documents", []):
                source = doc.metadata.get("source", "unknown")
                if source not in seen:
                    seen.add(source)
                    sources.append(source)
        return cls(
            status="completed",
            thread_id=thread_id,
            answer=answer,
            blocked=bool(result.get("blocked", False)),
            sources=sources,
            retries=result.get("retry_count", 0),
            prompt_versions=result.get("prompt_versions") or {},
        )


class HumanReviewRequest(BaseModel):
    """Request body for `POST /api/v1/rag/query/{thread_id}/review`."""

    decision: Literal["approve", "retry", "override"]
    revised_question: str | None = Field(
        default=None,
        max_length=get_settings().max_question_chars,
        description="Required when decision is 'retry'.",
    )
    override_answer: str | None = Field(
        default=None,
        max_length=4000,
        description="Required when decision is 'override'.",
    )

    @model_validator(mode="after")
    def _require_payload_for_decision(self) -> "HumanReviewRequest":
        if self.decision == "retry" and not (self.revised_question or "").strip():
            raise ValueError("revised_question is required for 'retry'")
        if self.decision == "override" and not (self.override_answer or "").strip():
            raise ValueError("override_answer is required for 'override'")
        return self


class LoginRequest(BaseModel):
    """Request body for `POST /api/v1/auth/login`."""

    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    """Response body for a successful login."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


class AuthStatusResponse(BaseModel):
    """Response body for `GET /api/v1/auth/status`.

    Lets clients (e.g. the frontend) know upfront whether the service requires
    a login, instead of guessing or showing a sign-in form that can never
    succeed when `JWT_SECRET_KEY`/`AUTH_USERNAME`/`AUTH_PASSWORD_HASH` are unset.
    """

    auth_required: bool


class CandidateCreateRequest(BaseModel):
    """Request body for `POST /api/v1/prompts/{name}/candidates`."""

    model_config = ConfigDict(extra="forbid")

    system: str = Field(min_length=1, max_length=32_000)
    human: str | None = Field(default=None, max_length=32_000)
    author: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=500)
    changelog: str = Field(min_length=1, max_length=500)
    target_model: str | None = Field(default=None, max_length=100)


class AbTestCreateRequest(BaseModel):
    """Request body for `POST /api/v1/prompts/{name}/ab-test`."""

    model_config = ConfigDict(extra="forbid")

    candidate_version: int = Field(ge=1)


class PromptEvaluationBody(BaseModel):
    """A/B evidence attached to a promotion decision."""

    model_config = ConfigDict(extra="forbid")

    dataset_version: str | None = None
    scorecard: dict[str, float] = Field(default_factory=dict)
    ab_test_id: str | None = None


class PromoteRequest(BaseModel):
    """Request body for `POST /api/v1/prompts/{name}/promote`."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    evaluation: PromptEvaluationBody | None = None
