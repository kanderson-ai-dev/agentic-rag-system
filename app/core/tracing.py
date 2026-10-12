"""Provider-agnostic tracing seam: ``none`` | ``langsmith`` | ``langfuse``.

Single integration point for observability backends. The rest of the app
talks only to this module — provider quirks (LangChain env tracing for
LangSmith, an OTel span context + fresh ``CallbackHandler`` for Langfuse)
stay behind four functions:

- ``init_tracing(settings)`` — startup hook: configure env vars, build the
  provider client once, run a single auth check. Stores the *effective*
  provider so a failed init cleanly disables tracing instead of retrying
  every request.
- ``build_callbacks(settings)`` — LangChain callbacks for one request. A
  new handler per call: handlers carry per-trace state and must never be
  shared between concurrent requests.
- ``trace_run(...)`` — context manager wrapping one logical unit of work in
  a root observation; yields the trace id (or ``None`` when disabled).
- ``flush()`` — force-export buffered spans at shutdown / end of eval runs.

Every failure mode degrades to "no tracing": a down or misconfigured
provider never changes an HTTP response or eats a request's latency budget.
"""

import re
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Effective provider, written once by ``init_tracing``. Anything that never
# saw an init (unit tests, ad-hoc scripts) reads "none" and stays untraced.
_active_provider = "none"
_active_lock = threading.Lock()

# Warn-once events: a broken provider logs once per process, not per request.
_warned: set[str] = set()
_warned_lock = threading.Lock()

# Keys whose values are never sent to a tracing backend (OWASP LLM06 / data
# minimization — mirrors the logging redaction policy).
_SECRET_KEY_RE = re.compile(r"(?i)(api_?key|secret|token|password|authorization)")

# Span payload bound: traces carry enough context to debug a pipeline (short
# strings, scores, metadata) but never multi-KB documents or raw user input.
_MAX_TRACE_STRING_CHARS = 500


def _warn_once(event: str, **fields: Any) -> None:
    """Log ``event`` once per process so a broken provider can't spam logs."""
    with _warned_lock:
        if event in _warned:
            return
        _warned.add(event)
    logger.warning(event, **fields)


def _reset_state() -> None:
    """Reset module state — for tests only."""
    global _active_provider
    with _active_lock:
        _active_provider = "none"
    with _warned_lock:
        _warned.clear()


def active_provider() -> str:
    """Effective provider resolved at init: ``none``, ``langsmith`` or ``langfuse``."""
    with _active_lock:
        return _active_provider


def _set_active(provider: str) -> None:
    global _active_provider
    with _active_lock:
        _active_provider = provider


def langfuse_configured(settings: Settings) -> bool:
    """True when the langfuse provider is selected and fully keyed."""
    return (
        settings.tracing_provider == "langfuse"
        and bool(settings.langfuse_public_key)
        and bool(settings.langfuse_secret_key)
    )


def mask_trace_payload(data: Any, *, max_chars: int = _MAX_TRACE_STRING_CHARS) -> Any:
    """Bound what reaches the tracing backend.

    - System-role message content is replaced outright — prompt *templates*
      are registry IP; the trace keeps ``prompt_version``/``prompt_hash``
      metadata instead (versions/hashes are the auditable reference, the
      template text stays in the repo).
    - Values under secret-looking keys are replaced outright.
    - Long strings are truncated (question text, document bodies).

    Traces therefore carry enough context to evaluate and debug the pipeline
    without exporting raw PII, full documents, prompt templates, or
    credentials. Note that user input and model output still leave the
    process for the Langfuse region in ``LANGFUSE_BASE_URL`` (bounded).
    """
    if isinstance(data, str):
        return data if len(data) <= max_chars else data[:max_chars] + "...[truncated]"
    if isinstance(data, dict):
        if data.get("role") == "system":
            return {**data, "content": "[system prompt redacted]"}
        return {
            key: ("[redacted]" if _SECRET_KEY_RE.search(str(key)) else mask_trace_payload(v))
            for key, v in data.items()
        }
    if isinstance(data, (list, tuple)):
        return [mask_trace_payload(v) for v in data]
    return data


def _new_langfuse_client(settings: Settings) -> Any:
    """Build the masked Langfuse client (singleton per public key).

    The SDK registers the instance under its public key, so the per-request
    ``CallbackHandler()`` and ``get_client()`` calls reuse this client and
    its mask — one construction point, consistent privacy behavior.
    """
    import langfuse

    return langfuse.Langfuse(
        public_key=settings.langfuse_public_key_value(),
        secret_key=settings.langfuse_secret_key_value(),
        host=settings.langfuse_base_url,
        mask=mask_trace_payload,
    )


def init_tracing(settings: Settings) -> str:
    """Resolve and initialize the tracing provider; returns the active one.

    LangSmith is env-driven (``configure_langsmith_env``). Langfuse gets a
    real client plus a startup auth check: on missing keys, init failure or
    a failed auth check the provider resolves to ``none`` — the request path
    then takes the zero-cost no-op branch.
    """
    settings.configure_langsmith_env()
    provider = settings.tracing_provider

    if provider == "langfuse":
        if not langfuse_configured(settings):
            _warn_once(
                "langfuse_provider_misconfigured",
                detail="TRACING_PROVIDER=langfuse requires LANGFUSE_PUBLIC_KEY "
                "and LANGFUSE_SECRET_KEY",
            )
            _set_active("none")
            return "none"
        try:
            client = _new_langfuse_client(settings)
            ok = bool(client.auth_check())
        except Exception as exc:  # noqa: BLE001 - init failure must not break startup
            _warn_once("langfuse_init_failed", exception=type(exc).__name__)
            _set_active("none")
            return "none"
        if not ok:
            _warn_once("langfuse_auth_failed")
            _set_active("none")
            return "none"
        logger.info("tracing_enabled", provider="langfuse")
        _set_active("langfuse")
        return "langfuse"

    if provider == "langsmith":
        logger.info(
            "tracing_enabled",
            provider="langsmith",
            tracing_v2=settings.langchain_tracing_v2,
        )
    _set_active(provider)
    return provider


def build_callbacks(settings: Settings | None = None) -> list[Any]:
    """Fresh LangChain callbacks for one request / one eval item.

    Returns ``[]`` whenever tracing is off or handler construction fails —
    the invoke then runs untraced at zero cost. A new handler is built per
    call because the handler holds per-trace state; sharing one between
    concurrent requests would cross-contaminate traces.
    """
    if active_provider() != "langfuse":
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception as exc:  # noqa: BLE001 - broken provider degrades, never 500s
        _warn_once("langfuse_handler_init_failed", exception=type(exc).__name__)
        return []


@contextmanager
def trace_run(
    settings: Settings,
    *,
    name: str,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[str | None]:
    """Root observation for one unit of work; yields the trace id or ``None``.

    Setup failures yield ``None`` so callers treat tracing as optional. Body
    exceptions propagate normally — a traced failed request still fails, it
    is simply recorded on the span first.
    """
    if active_provider() != "langfuse":
        yield None
        return
    try:
        import langfuse

        client = langfuse.get_client()
    except Exception as exc:  # noqa: BLE001
        _warn_once("langfuse_client_unavailable", exception=type(exc).__name__)
        yield None
        return

    stack = ExitStack()
    try:
        stack.enter_context(client.start_as_current_observation(name=name, as_type="span"))
        stack.enter_context(
            langfuse.propagate_attributes(
                session_id=session_id,
                user_id=user_id,
                metadata=dict(metadata or {}),
                tags=list(tags or []),
                trace_name=name,
            )
        )
    except Exception as exc:  # noqa: BLE001
        stack.close()
        _warn_once("langfuse_trace_start_failed", exception=type(exc).__name__)
        yield None
        return

    with stack:
        yield client.get_current_trace_id()


@contextmanager
def observe_llm(
    name: str, *, metadata: dict[str, Any] | None = None
) -> Iterator[Any | None]:
    """Manual generation-level observation for calls LangChain doesn't trace.

    Only meaningful inside ``trace_run`` — otherwise there is no ambient
    trace context and this is a no-op.
    """
    if active_provider() != "langfuse":
        yield None
        return
    try:
        import langfuse

        client = langfuse.get_client()
    except Exception as exc:  # noqa: BLE001
        _warn_once("langfuse_client_unavailable", exception=type(exc).__name__)
        yield None
        return

    stack = ExitStack()
    try:
        span = stack.enter_context(
            client.start_as_current_observation(name=name, as_type="generation")
        )
        if metadata:
            span.update(metadata=dict(metadata))
    except Exception as exc:  # noqa: BLE001
        stack.close()
        _warn_once("langfuse_observe_failed", exception=type(exc).__name__)
        yield None
        return

    with stack:
        yield span


def flush() -> None:
    """Force-export buffered spans (shutdown, end of CLI eval runs)."""
    if active_provider() != "langfuse":
        return
    try:
        import langfuse

        langfuse.get_client().flush()
    except Exception as exc:  # noqa: BLE001
        _warn_once("langfuse_flush_failed", exception=type(exc).__name__)


def shutdown() -> None:
    """Flush and release the provider client; a failure cannot block shutdown."""
    if active_provider() != "langfuse":
        return
    try:
        import langfuse

        client = langfuse.get_client()
        client.flush()
        client.shutdown()
    except Exception as exc:  # noqa: BLE001 - teardown must never hang shutdown
        _warn_once("langfuse_shutdown_failed", exception=type(exc).__name__)
