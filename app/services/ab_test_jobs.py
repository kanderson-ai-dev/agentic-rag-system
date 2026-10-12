"""Single-flight job manager for prompt A/B evaluation runs.

The API's `POST /{name}/ab-test` returns 202 immediately; the run executes in a
daemon thread and its status is polled via `GET /{name}/ab-test/{job_id}`. Only
one job may be in flight at a time (evaluation runs spend real tokens — a pile
of concurrent A/B runs would multiply cost). Finished jobs persist their result
JSON under `evaluation/results/ab/` so the evidence survives restarts.

The actual experiment engine is injected as ``runner`` — the API default
delegates to the evaluation engine; tests inject a synchronous stub.
"""

import json
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

AbTestRunner = Callable[[str, int], dict[str, Any]]
"""(prompt_name, candidate_version) -> result dict."""


class JobAlreadyRunningError(Exception):
    """Raised when a second A/B run is requested while one is in flight."""


class AbTestJobManager:
    """Tracks at most one running A/B job; persists results to disk."""

    def __init__(self, results_dir: str | Path = "evaluation/results/ab") -> None:
        self._results_dir = Path(results_dir)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def start(
        self,
        prompt_name: str,
        candidate_version: int,
        runner: AbTestRunner,
    ) -> str:
        """Queue a run; returns the job id. Raises when a job is in flight."""
        with self._lock:
            if any(
                job["status"] in {"queued", "running"} for job in self._jobs.values()
            ):
                raise JobAlreadyRunningError("an A/B run is already in progress")
            job_id = uuid.uuid4().hex[:12]
            self._jobs[job_id] = {
                "job_id": job_id,
                "prompt_name": prompt_name,
                "candidate_version": candidate_version,
                "status": "queued",
                "queued_at": datetime.now(UTC).isoformat(),
                "result": None,
                "error": None,
            }
        thread = threading.Thread(
            target=self._run,
            args=(job_id, prompt_name, candidate_version, runner),
            daemon=True,
            name=f"ab-test-{job_id}",
        )
        thread.start()
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def _run(
        self,
        job_id: str,
        prompt_name: str,
        candidate_version: int,
        runner: AbTestRunner,
    ) -> None:
        with self._lock:
            self._jobs[job_id]["status"] = "running"
            self._jobs[job_id]["started_at"] = datetime.now(UTC).isoformat()
        try:
            result = runner(prompt_name, candidate_version)
        except Exception as exc:  # noqa: BLE001 - job failure is data, not a crash
            with self._lock:
                self._jobs[job_id]["status"] = "failed"
                # Only the exception class is stored — an error body could
                # carry internals or credentials into the result file.
                self._jobs[job_id]["error"] = type(exc).__name__
            return
        result_path = self._persist(job_id, result)
        with self._lock:
            self._jobs[job_id]["status"] = "succeeded"
            self._jobs[job_id]["result"] = result
            self._jobs[job_id]["result_path"] = str(result_path)
            self._jobs[job_id]["finished_at"] = datetime.now(UTC).isoformat()

    def _persist(self, job_id: str, result: dict[str, Any]) -> Path:
        self._results_dir.mkdir(parents=True, exist_ok=True)
        path = self._results_dir / f"{job_id}.json"
        path.write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        return path


def default_ab_test_runner(prompt_name: str, candidate_version: int) -> dict[str, Any]:
    """Run the evaluation A/B engine with the configured hard cost/time caps."""
    from app.core.config import get_settings
    from evaluation.ab_test import run_ab_test

    settings = get_settings()
    result: dict[str, Any] = run_ab_test(
        prompt_name,
        candidate_version,
        max_cost_usd=settings.ab_test_max_cost_usd,
        deadline_seconds=settings.ab_test_timeout_seconds,
    )
    return result
