"""File-based prompt registry: versioned prompts with an audited lifecycle.

Prompts are git-tracked YAML artifacts under ``prompts/<name>/vN.yaml`` — the
filesystem (and git history) is the source of truth, no database involved.
Each prompt name follows the lifecycle

    draft -> candidate -> promoted -> superseded | rolled_back

with exactly one ``promoted`` version per name at any time. Promotion is
metric-gated by the caller (the API/A/B engine supplies the evidence); the
registry enforces the lifecycle and records every mutation in an append-only
hash-chained audit log (``prompts/AUDIT.jsonl``) so tampering is detectable.

Validation is fail-closed: names are a fixed enum (paths are built only from
enum + int, so traversal is impossible by construction), template variables
must match the per-name contract, oversized prompts, embedded secrets and
prompt-injection payloads are rejected.
"""

import contextlib
import difflib
import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.graph.guardrails import detect_prompt_injection

_AUDIT_FILE = "AUDIT.jsonl"
_GENESIS_HASH = "0" * 64
_VERSION_FILE = re.compile(r"^v(\d+)\.yaml$")

# Hard caps on prompt content — a prompt is a few paragraphs, not a document.
_MAX_TEXT_CHARS = 32_000
_MAX_DESCRIPTION_CHARS = 500

# Defense against secrets being committed into prompt artifacts (the mirror
# image of the safe-logging redaction list, applied to content).
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{16,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*['\"]?[^\s'\"]{8,}"
    ),
]

_TEMPLATE_VARIABLE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class PromptName(StrEnum):
    """The closed set of prompts the system knows — never a user-supplied path."""

    GRADE_DOCUMENTS = "grade_documents"
    GENERATE_ANSWER = "generate_answer"
    TRANSFORM_QUERY = "transform_query"
    LLM_JUDGE = "llm_judge"


class PromptStatus(StrEnum):
    DRAFT = "draft"
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    SUPERSEDED = "superseded"
    ROLLED_BACK = "rolled_back"


# Deterministic lifecycle transition table; anything not listed is rejected.
_ALLOWED_TRANSITIONS: dict[PromptStatus, set[PromptStatus]] = {
    PromptStatus.DRAFT: {PromptStatus.CANDIDATE},
    PromptStatus.CANDIDATE: {PromptStatus.PROMOTED},
    PromptStatus.PROMOTED: {PromptStatus.SUPERSEDED, PromptStatus.ROLLED_BACK},
    PromptStatus.SUPERSEDED: {PromptStatus.PROMOTED},
    PromptStatus.ROLLED_BACK: set(),
}

# The template-variable contract each prompt must honour exactly: required
# variables must be present and unknown ones are rejected.
_REQUIRED_VARIABLES: dict[PromptName, frozenset[str]] = {
    PromptName.GRADE_DOCUMENTS: frozenset({"documents", "question"}),
    PromptName.GENERATE_ANSWER: frozenset({"context", "question"}),
    PromptName.TRANSFORM_QUERY: frozenset({"question"}),
    PromptName.LLM_JUDGE: frozenset(),
}


class PromptRegistryError(Exception):
    """Base error for registry failures."""


class PromptNotFoundError(PromptRegistryError, KeyError):
    """Unknown prompt name or version."""


class PromptValidationError(PromptRegistryError, ValueError):
    """Prompt content failed validation."""


class LifecycleError(PromptRegistryError, ValueError):
    """Illegal lifecycle transition or broken invariant."""


class PromptEvaluation(BaseModel):
    """Evidence attached to a version when it is promoted after an A/B run."""

    model_config = ConfigDict(extra="forbid")

    dataset_version: str | None = None
    scorecard: dict[str, float] = Field(default_factory=dict)
    ab_test_id: str | None = None


class PromptVersion(BaseModel):
    """One versioned prompt artifact (mirrors the on-disk YAML schema)."""

    model_config = ConfigDict(extra="forbid")

    name: PromptName
    version: int = Field(ge=1)
    status: PromptStatus
    author: str = Field(min_length=1, max_length=100)
    created_at: datetime
    target_model: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=_MAX_DESCRIPTION_CHARS)
    changelog: str = Field(min_length=1, max_length=_MAX_DESCRIPTION_CHARS)
    variables: list[str]
    system: str = Field(min_length=1, max_length=_MAX_TEXT_CHARS)
    human: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    evaluation: PromptEvaluation | None = None

    @field_validator("system", "human", mode="before")
    @classmethod
    def _strip_surrounding_whitespace(cls, value: object) -> object:
        """Normalise YAML block-scalar trailing newlines out of the text."""
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def _check_template_contract(self) -> Self:
        found = set(_TEMPLATE_VARIABLE.findall(self.system))
        if self.human:
            found |= set(_TEMPLATE_VARIABLE.findall(self.human))
        declared = set(self.variables)
        if declared != found:
            raise PromptValidationError(
                f"{self.name} v{self.version}: declared variables {sorted(declared)} "
                f"do not match template variables {sorted(found)}"
            )
        required = _REQUIRED_VARIABLES[self.name]
        if declared != required:
            raise PromptValidationError(
                f"{self.name} v{self.version}: variables {sorted(declared)} "
                f"do not satisfy the required contract {sorted(required)}"
            )
        return self


def _entry_hash(entry: dict[str, Any]) -> str:
    """sha256 over ``prev_hash + canonical JSON`` of the entry minus ``hash``."""
    payload = {key: value for key, value in entry.items() if key != "hash"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((str(payload["prev_hash"]) + canonical).encode()).hexdigest()


class _LiteralDumper(yaml.SafeDumper):  # type: ignore
    """YAML dumper that renders long text fields as literal blocks."""


def _literal_str(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


_LiteralDumper.add_representer(str, _literal_str)


def _validate_content(name: PromptName, system: str, human: str | None) -> None:
    """Reject prompt text carrying secrets or injection payloads."""
    for label, text in (("system", system), ("human", human)):
        if text is None:
            continue
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                raise PromptValidationError(
                    f"{name}: {label} text matches a secret pattern"
                )
        category = detect_prompt_injection(text)
        if category is not None:
            raise PromptValidationError(
                f"{name}: {label} text matches injection pattern '{category}'"
            )


class PromptRegistry:
    """Read/mutate the versioned prompt store rooted at ``root``.

    All versions of every known prompt are loaded and validated once at
    construction; mutations write YAML atomically (temp file + ``os.replace``)
    under a lock and append to the audit chain.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._lock = threading.Lock()
        self._versions: dict[PromptName, dict[int, PromptVersion]] = self._load_all()

    @property
    def root(self) -> Path:
        return self._root

    # ------------------------------------------------------------------ load
    def _load_all(self) -> dict[PromptName, dict[int, PromptVersion]]:
        versions: dict[PromptName, dict[int, PromptVersion]] = {}
        for name in PromptName:
            directory = self._root / name.value
            if not directory.is_dir():
                raise PromptRegistryError(
                    f"missing prompt directory: {directory}"
                )
            prompt_versions: dict[int, PromptVersion] = {}
            for path in sorted(directory.iterdir()):
                match = _VERSION_FILE.match(path.name)
                if not match:
                    continue
                file_version = int(match.group(1))
                prompt_versions[file_version] = self._load_version(
                    name, file_version, path
                )
            if not prompt_versions:
                raise PromptRegistryError(f"{name}: no versions found in {directory}")
            promoted = [
                v.version
                for v in prompt_versions.values()
                if v.status == PromptStatus.PROMOTED
            ]
            if len(promoted) != 1:
                raise LifecycleError(
                    f"{name}: expected exactly one promoted version, found {promoted}"
                )
            versions[name] = prompt_versions
        return versions

    def _load_version(self, name: PromptName, file_version: int, path: Path) -> PromptVersion:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise PromptValidationError(f"{path}: invalid YAML") from exc
        if not isinstance(data, dict):
            raise PromptValidationError(f"{path}: expected a YAML mapping")
        try:
            prompt = PromptVersion.model_validate(data)
        except (PromptValidationError, ValueError) as exc:
            raise PromptValidationError(f"{path}: {exc}") from exc
        if prompt.name != name:
            raise PromptValidationError(
                f"{path}: name '{prompt.name}' does not match directory '{name}'"
            )
        if prompt.version != file_version:
            raise PromptValidationError(
                f"{path}: version {prompt.version} does not match filename v{file_version}"
            )
        _validate_content(name, prompt.system, prompt.human)
        return prompt

    # ------------------------------------------------------------------ reads
    def _coerce_name(self, name: PromptName | str) -> PromptName:
        try:
            return name if isinstance(name, PromptName) else PromptName(str(name))
        except ValueError as exc:
            raise PromptNotFoundError(f"unknown prompt name: {name!r}") from exc

    def list_prompts(self) -> list[dict[str, Any]]:
        """Summary per prompt: active version, version count, statuses."""
        return [
            {
                "name": name.value,
                "active_version": self.promoted(name).version,
                "version_count": len(self._versions[name]),
                "versions": [
                    {"version": v.version, "status": v.status.value}
                    for v in sorted(
                        self._versions[name].values(), key=lambda p: p.version
                    )
                ],
            }
            for name in PromptName
        ]

    def versions(self, name: PromptName | str) -> list[PromptVersion]:
        """All versions of a prompt, ascending."""
        prompt_name = self._coerce_name(name)
        return sorted(self._versions[prompt_name].values(), key=lambda p: p.version)

    def get(self, name: PromptName | str, version: int | None = None) -> PromptVersion:
        """Return one version (the promoted one when ``version`` is ``None``)."""
        prompt_name = self._coerce_name(name)
        if version is None:
            return self.promoted(prompt_name)
        try:
            return self._versions[prompt_name][version]
        except KeyError as exc:
            raise PromptNotFoundError(f"{prompt_name}: no version v{version}") from exc

    def promoted(self, name: PromptName | str) -> PromptVersion:
        prompt_name = self._coerce_name(name)
        for prompt in self._versions[prompt_name].values():
            if prompt.status == PromptStatus.PROMOTED:
                return prompt
        raise LifecycleError(f"{prompt_name}: no promoted version")

    @property
    def version_token(self) -> str:
        """Stable token for the current promoted set (changes on promotion)."""
        return "|".join(
            f"{name.value}@{self.promoted(name).version}" for name in PromptName
        )

    def diff(
        self, name: PromptName | str, a: int, b: int
    ) -> list[dict[str, Any]]:
        """Structured diff between two versions' rendered text.

        Returns hunks ``{op: add|del|same, old_line, new_line, text}`` computed
        over the ``system`` + ``human`` text.
        """
        old, new = self.get(name, a), self.get(name, b)
        old_lines = _render_text(old).splitlines()
        new_lines = _render_text(new).splitlines()
        hunks: list[dict[str, Any]] = []
        old_no = new_no = 0
        for line in difflib.ndiff(old_lines, new_lines):
            marker, text = line[:2], line[2:]
            if marker == "  ":
                old_no += 1
                new_no += 1
                hunks.append(
                    {"op": "same", "old_line": old_no, "new_line": new_no, "text": text}
                )
            elif marker == "- ":
                old_no += 1
                hunks.append(
                    {"op": "del", "old_line": old_no, "new_line": None, "text": text}
                )
            elif marker == "+ ":
                new_no += 1
                hunks.append(
                    {"op": "add", "old_line": None, "new_line": new_no, "text": text}
                )
        return hunks

    # -------------------------------------------------------------- mutations
    def create_candidate(
        self,
        name: PromptName | str,
        *,
        system: str,
        human: str | None = None,
        author: str,
        description: str,
        changelog: str,
        target_model: str | None = None,
        actor: str = "local",
    ) -> PromptVersion:
        """Create the next ``vN`` for ``name`` with status ``candidate``."""
        with self._lock:
            prompt_name = self._coerce_name(name)
            base = self.promoted(prompt_name)
            try:
                candidate = PromptVersion(
                    name=prompt_name,
                    version=max(self._versions[prompt_name]) + 1,
                    status=PromptStatus.CANDIDATE,
                    author=author,
                    created_at=datetime.now(UTC),
                    target_model=target_model or base.target_model,
                    description=description,
                    changelog=changelog,
                    variables=sorted(_REQUIRED_VARIABLES[prompt_name]),
                    system=system,
                    human=human,
                )
            except ValueError as exc:
                raise PromptValidationError(str(exc)) from exc
            _validate_content(prompt_name, candidate.system, candidate.human)
            self._write_version(candidate)
            self._versions[prompt_name][candidate.version] = candidate
            self._append_audit(actor, "create_candidate", prompt_name, candidate.version)
            return candidate

    def promote(
        self,
        name: PromptName | str,
        version: int,
        *,
        actor: str,
        evaluation: PromptEvaluation | dict[str, Any] | None = None,
    ) -> PromptVersion:
        """Promote a candidate; the previously promoted version is superseded.

        Metric gating is the caller's job (the A/B engine must have already
        verified the candidate); the registry enforces the lifecycle and
        stores the evidence on the promoted artifact.
        """
        with self._lock:
            prompt_name = self._coerce_name(name)
            candidate = self.get(prompt_name, version)
            if candidate.status != PromptStatus.CANDIDATE:
                raise LifecycleError(
                    f"{prompt_name} v{version}: only a candidate can be promoted "
                    f"(status is {candidate.status})"
                )
            current = self.promoted(prompt_name)
            self._transition(current, PromptStatus.SUPERSEDED)
            if evaluation is not None:
                candidate = candidate.model_copy(
                    update={
                        "evaluation": (
                            evaluation
                            if isinstance(evaluation, PromptEvaluation)
                            else PromptEvaluation.model_validate(evaluation)
                        )
                    }
                )
            promoted = self._transition(candidate, PromptStatus.PROMOTED)
            self._append_audit(actor, "promote", prompt_name, promoted.version)
            return promoted

    def rollback(self, name: PromptName | str, *, actor: str) -> PromptVersion:
        """Re-promote the most recently superseded version.

        The currently promoted version is marked ``rolled_back`` — the trail
        of what ran when stays in the artifacts and the audit chain.
        """
        with self._lock:
            prompt_name = self._coerce_name(name)
            superseded = [
                v
                for v in self._versions[prompt_name].values()
                if v.status == PromptStatus.SUPERSEDED
            ]
            if not superseded:
                raise LifecycleError(f"{prompt_name}: nothing to roll back to")
            current = self.promoted(prompt_name)
            target = max(superseded, key=lambda v: v.version)
            self._transition(current, PromptStatus.ROLLED_BACK)
            promoted = self._transition(target, PromptStatus.PROMOTED)
            self._append_audit(actor, "rollback", prompt_name, promoted.version)
            return promoted

    def _transition(self, prompt: PromptVersion, target: PromptStatus) -> PromptVersion:
        """Apply a lifecycle transition, persist it and return the updated version."""
        allowed = _ALLOWED_TRANSITIONS.get(prompt.status, set())
        if target not in allowed:
            raise LifecycleError(
                f"{prompt.name} v{prompt.version}: forbidden transition "
                f"{prompt.status} -> {target}"
            )
        updated = prompt.model_copy(update={"status": target})
        self._write_version(updated)
        self._versions[prompt.name][prompt.version] = updated
        return updated

    # ------------------------------------------------------------------ audit
    def _audit_path(self) -> Path:
        return self._root / _AUDIT_FILE

    def _append_audit(
        self, actor: str, action: str, name: PromptName, version: int
    ) -> None:
        entries = self._read_audit()
        prev_hash = entries[-1]["hash"] if entries else _GENESIS_HASH
        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "actor": actor,
            "action": action,
            "name": name.value,
            "version": version,
            "prev_hash": prev_hash,
        }
        entry["hash"] = _entry_hash(entry)
        with self._audit_path().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, separators=(",", ":")) + "\n")

    def _read_audit(self) -> list[dict[str, Any]]:
        path = self._audit_path()
        if not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
        return entries

    def audit_entries(self) -> list[dict[str, Any]]:
        """Return the raw audit chain (oldest first)."""
        return self._read_audit()

    def verify_chain(self) -> bool:
        """Recompute every hash link; ``False`` on any tampering/corruption."""
        try:
            prev_hash = _GENESIS_HASH
            for entry in self._read_audit():
                if entry.get("prev_hash") != prev_hash:
                    return False
                if entry.get("hash") != _entry_hash(entry):
                    return False
                prev_hash = entry["hash"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return False
        return True

    # ------------------------------------------------------------------ write
    def _write_version(self, prompt: PromptVersion) -> None:
        """Persist a version atomically: temp file + ``os.replace``."""
        directory = self._root / prompt.name.value
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"v{prompt.version}.yaml"
        payload = _version_to_yaml_dict(prompt)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(directory), prefix=".v", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                yaml.dump(
                    payload,
                    handle,
                    Dumper=_LiteralDumper,
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                )
            os.replace(tmp_name, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise


def _render_text(prompt: PromptVersion) -> str:
    text = f"system:\n{prompt.system}"
    if prompt.human is not None:
        text += f"\nhuman:\n{prompt.human}"
    return text


def _version_to_yaml_dict(prompt: PromptVersion) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": prompt.name.value,
        "version": prompt.version,
        "status": prompt.status.value,
        "author": prompt.author,
        "created_at": prompt.created_at.isoformat().replace("+00:00", "Z"),
        "target_model": prompt.target_model,
        "description": prompt.description,
        "changelog": prompt.changelog,
        "variables": list(prompt.variables),
        "system": prompt.system,
    }
    if prompt.human is not None:
        data["human"] = prompt.human
    data["evaluation"] = (
        prompt.evaluation.model_dump(mode="json") if prompt.evaluation else None
    )
    return data
