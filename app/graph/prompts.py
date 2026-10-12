"""Prompt template access via the file-based Prompt Registry.

This module is a thin facade over ``app.services.prompt_registry``: the graph
no longer binds prompt *constants* — it resolves the currently ``promoted``
version of each prompt from the versioned YAML artifacts under ``prompts/``.

- ``get_prompt(name, version=None)`` builds a ``ChatPromptTemplate`` from a
  stored version (the promoted one when ``version`` is ``None``).
- ``RegistryPrompt`` is a ``Runnable`` that resolves its prompt **lazily on
  every invoke** — a promotion takes effect on the next LLM call with no
  graph rebuild or restart. Callers pin an explicit version by passing
  ``prompt_version`` in the input dict (in-flight HITL threads keep the
  versions they started with, recorded in ``GraphState["prompt_versions"]``).
- ``active_versions()`` returns ``{prompt_name: promoted_version}`` and refreshes
  the ``agent_prompt_active_version`` gauge.
"""

import threading
from pathlib import Path
from typing import Any, cast

from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

from app.core.config import get_settings
from app.core.metrics import agent_prompt_active_version
from app.services.prompt_registry import PromptName, PromptRegistry

# Exact answer the generation prompt instructs the model to give when the
# retrieved context cannot answer the question. Shared with the API layer so a
# refusal can be detected in code (and reported with honest `sources: []`)
# without an extra LLM call. Lives outside the registry because the detection
# contract must survive prompt edits.
REFUSAL_TEXT = "I don't know."

_registry: PromptRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> PromptRegistry:
    """Return the process-wide registry, built lazily from ``prompts_dir``."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = PromptRegistry(Path(get_settings().prompts_dir))
    return _registry


def set_registry(registry: PromptRegistry | None) -> None:
    """Swap the registry instance (API wiring, tests); ``None`` resets to lazy."""
    global _registry
    with _registry_lock:
        _registry = registry


def get_prompt(
    name: PromptName | str, version: int | None = None
) -> ChatPromptTemplate:
    """Build a ``ChatPromptTemplate`` from a stored prompt version.

    ``version=None`` resolves the currently promoted version, so prompts built
    through this facade always reflect the registry at call time.
    """
    prompt = get_registry().get(name, version)
    messages: list[tuple[str, str]] = [("system", prompt.system)]
    if prompt.human is not None:
        messages.append(("human", prompt.human))
    return ChatPromptTemplate.from_messages(messages)


def get_system_text(name: PromptName | str, version: int | None = None) -> str:
    """Return just the system text of a stored prompt version."""
    return get_registry().get(name, version).system


def active_versions() -> dict[str, int]:
    """Map ``{prompt_name: promoted_version}`` and refresh the gauge."""
    registry = get_registry()
    versions = {
        name.value: registry.promoted(name).version for name in PromptName
    }
    for prompt_name, version in versions.items():
        agent_prompt_active_version.labels(prompt=prompt_name).set(version)
    return versions


class RegistryPrompt(Runnable[dict[str, Any], ChatPromptValue]):
    """Runnable that formats a registry prompt at invoke time.

    Reads ``prompt_version`` out of the input dict when present (the rest of
    the keys are the prompt's template variables); a missing or ``None``
    version resolves the currently promoted one. This keeps promotions
    restart-free while letting a HITL thread pin the versions it began with.
    """

    def __init__(self, name: PromptName | str) -> None:
        super().__init__()
        self._name = name

    def invoke(
        self,
        input: dict[str, Any],  # noqa: A002 - Runnable API names it `input`
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ChatPromptValue:
        variables = dict(input)
        version = variables.pop("prompt_version", None)
        return cast(ChatPromptValue, get_prompt(self._name, version).invoke(variables))
