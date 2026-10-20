"""Prompt registry API — read-only views plus protected lifecycle mutations.

Reads (list, versions, single version, diff, A/B job status) are open so the
operator console can render prompt provenance without a token. Mutations
(create candidate, A/B run, promote, rollback) require `require_prompt_write`
(JWT when auth is configured, or the `PROMPT_REGISTRY_WRITE_ENABLED` escape
hatch otherwise) and are rate-limited — registry writes are a prompt-tampering
surface, so every mutation is also appended to the registry's hash-chained
audit log with the authenticated subject as actor.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.v1.schemas import (
    AbTestCreateRequest,
    CandidateCreateRequest,
    PromoteRequest,
    PromptNameParam,
)
from app.core.config import Settings, get_settings
from app.core.dependencies import (
    get_ab_test_manager,
    get_ab_test_runner,
    get_prompt_registry,
    require_prompt_write,
)
from app.core.rate_limit import limiter
from app.services.ab_test_jobs import (
    AbTestJobManager,
    AbTestRunner,
    JobAlreadyRunningError,
)
from app.services.promotion_gate import evaluate_promotion
from app.services.prompt_registry import (
    LifecycleError,
    PromptEvaluation,
    PromptNotFoundError,
    PromptRegistry,
    PromptStatus,
    PromptValidationError,
)

router = APIRouter(prefix="/api/v1/prompts", tags=["prompts"])


@contextmanager
def _registry_errors() -> Iterator[None]:
    """Translate registry exceptions into generic HTTP errors.

    No stack traces, file paths, or prompt internals are echoed back —
    validation failures return a plain reason; lifecycle/not-found errors
    carry only the prompt name and status involved.
    """
    try:
        yield
    except PromptNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Prompt version not found") from exc
    except LifecycleError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PromptValidationError as exc:
        raise HTTPException(
            status_code=422, detail="Prompt content failed validation"
        ) from exc


def _version_public(version: dict[str, Any]) -> dict[str, Any]:
    """Strip prompt bodies from list views — content lives on the detail route."""
    return {k: v for k, v in version.items() if k not in {"system", "human"}}


# ---------------------------------------------------------------------------
# Open read endpoints
# ---------------------------------------------------------------------------


@router.get("/")
def list_prompts(
    registry: PromptRegistry = Depends(get_prompt_registry),
) -> dict[str, Any]:
    """All prompts: active version, version count, audit-chain integrity.

    ``writes_enabled`` tells the console upfront whether mutations are even
    possible in this deployment (JWT configured, or the open-deployment write
    flag on) so edit/promote controls can render disabled with a reason
    instead of failing on first submit.
    """
    settings = get_settings()
    return {
        "prompts": registry.list_prompts(),
        "audit_chain_verified": registry.verify_chain(),
        "writes_enabled": bool(
            settings.jwt_secret_key or settings.prompt_registry_write_enabled
        ),
    }


@router.get("/{name}")
def list_versions(
    name: PromptNameParam,
    registry: PromptRegistry = Depends(get_prompt_registry),
) -> dict[str, Any]:
    """All versions of one prompt: status, author, changelog, evaluation."""
    with _registry_errors():
        versions = registry.versions(name)
        promoted = registry.promoted(name).version
    return {
        "name": name,
        "promoted_version": promoted,
        "versions": [_version_public(v.model_dump(mode="json")) for v in versions],
    }


@router.get("/{name}/diff")
def diff_versions(
    name: PromptNameParam,
    a: int = Query(ge=1),
    b: int = Query(ge=1),
    registry: PromptRegistry = Depends(get_prompt_registry),
) -> dict[str, Any]:
    """Structured line-level diff between two versions of the same prompt."""
    with _registry_errors():
        hunks = registry.diff(name, a, b)
    return {"name": name, "a": a, "b": b, "hunks": hunks}


@router.get("/{name}/{version}")
def get_version(
    name: PromptNameParam,
    version: int,
    registry: PromptRegistry = Depends(get_prompt_registry),
) -> dict[str, Any]:
    """Full content of one prompt version."""
    with _registry_errors():
        prompt = registry.get(name, version)
    return prompt.model_dump(mode="json")


@router.get("/{name}/ab-test/{job_id}")
def ab_test_status(
    name: PromptNameParam,
    job_id: str,
    manager: AbTestJobManager = Depends(get_ab_test_manager),
) -> dict[str, Any]:
    """Poll the status of a queued/running/finished A/B run."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="A/B job not found")
    return job


# ---------------------------------------------------------------------------
# Protected mutations
# ---------------------------------------------------------------------------


@router.post("/{name}/candidates", status_code=201)
@limiter.limit("10/minute")
def create_candidate(
    request: Request,
    name: PromptNameParam,
    payload: CandidateCreateRequest,
    actor: str = Depends(require_prompt_write),
    registry: PromptRegistry = Depends(get_prompt_registry),
) -> dict[str, Any]:
    """Create a draft→candidate version. Content is validated on write
    (required variables, length cap, secret/injection pattern rejection)."""
    with _registry_errors():
        candidate = registry.create_candidate(
            name,
            system=payload.system,
            human=payload.human,
            author=payload.author,
            description=payload.description,
            changelog=payload.changelog,
            target_model=payload.target_model,
            actor=actor,
        )
    return candidate.model_dump(mode="json")


@router.post("/{name}/ab-test", status_code=202)
@limiter.limit("10/minute")
def start_ab_test(
    request: Request,
    name: PromptNameParam,
    payload: AbTestCreateRequest,
    actor: str = Depends(require_prompt_write),
    registry: PromptRegistry = Depends(get_prompt_registry),
    manager: AbTestJobManager = Depends(get_ab_test_manager),
    runner: AbTestRunner = Depends(get_ab_test_runner),
) -> dict[str, Any]:
    """Queue an A/B run of a candidate against the promoted version.

    Returns 202 with a job id to poll. Single-flight: a second request while
    a run is in progress is rejected with 409.
    """
    del actor  # the A/B job is not a registry mutation — no audit entry
    with _registry_errors():
        candidate = registry.get(name, payload.candidate_version)
    if candidate.status not in {PromptStatus.CANDIDATE, PromptStatus.DRAFT}:
        raise HTTPException(
            status_code=409,
            detail=f"{name} v{payload.candidate_version} is not a candidate",
        )
    try:
        job_id = manager.start(name, payload.candidate_version, runner)
    except JobAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"job_id": job_id, "status": "queued"}


@router.post("/{name}/promote")
@limiter.limit("10/minute")
def promote(
    request: Request,
    name: PromptNameParam,
    payload: PromoteRequest,
    actor: str = Depends(require_prompt_write),
    registry: PromptRegistry = Depends(get_prompt_registry),
    manager: AbTestJobManager = Depends(get_ab_test_manager),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Promote a candidate — gated on EDD thresholds and non-regression.

    Evidence comes from the request body's `evaluation`, from the candidate's
    stored A/B result, or resolved from an `ab_test_id` job. Failures return
    409 naming the metrics.
    """
    with _registry_errors():
        candidate = registry.get(name, payload.version)
        if candidate.status is not PromptStatus.CANDIDATE:
            raise LifecycleError(
                f"{name} v{payload.version} is {candidate.status}, not candidate"
            )
        promoted = registry.promoted(name)

    evidence = payload.evaluation or candidate.evaluation
    # An `ab_test_id` resolves against the job manager — the typical API
    # flow: run A/B, poll, promote by id. `meta.live` on the persisted
    # result is what proves the scorecard came from real model calls.
    live_evidence = False
    ab_test_id = getattr(evidence, "ab_test_id", None) if evidence else None
    if ab_test_id:
        job = manager.get(str(ab_test_id))
        if job is not None and job.get("status") == "succeeded":
            result = job.get("result") or {}
            live_evidence = bool((result.get("meta") or {}).get("live"))
            if not getattr(evidence, "scorecard", None):
                evidence = PromptEvaluation(
                    dataset_version=(result.get("meta") or {}).get("dataset_version"),
                    scorecard=result.get("candidate_scorecard") or {},
                    ab_test_id=str(ab_test_id),
                )
    # In a deployment with an OpenAI key configured, a fixture/stub run can
    # never promote a prompt — only evidence from a real-model A/B counts.
    if settings.openai_api_key and not live_evidence:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Promotion requires live A/B evidence",
                "failures": ["evaluation: not a live run"],
            },
        )
    if evidence is None or not getattr(evidence, "scorecard", None):
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Promotion requires A/B evaluation evidence",
                "failures": ["evaluation: missing"],
            },
        )
    baseline_card = promoted.evaluation.scorecard if promoted.evaluation else None
    gate = evaluate_promotion(evidence.scorecard, baseline_card)
    if not gate.allowed:
        raise HTTPException(
            status_code=409,
            detail={"message": "Promotion gate failed", "failures": gate.failures},
        )

    with _registry_errors():
        result = registry.promote(
            name,
            payload.version,
            actor=actor,
            evaluation=evidence
            if isinstance(evidence, PromptEvaluation)
            else PromptEvaluation(
                dataset_version=evidence.dataset_version,
                scorecard=evidence.scorecard,
                ab_test_id=evidence.ab_test_id,
            ),
        )
    return result.model_dump(mode="json")


@router.post("/{name}/rollback")
@limiter.limit("10/minute")
def rollback(
    request: Request,
    name: PromptNameParam,
    actor: str = Depends(require_prompt_write),
    registry: PromptRegistry = Depends(get_prompt_registry),
) -> dict[str, Any]:
    """Re-promote the most recent superseded/rolled-back version."""
    with _registry_errors():
        promoted = registry.rollback(name, actor=actor)
    return promoted.model_dump(mode="json")
