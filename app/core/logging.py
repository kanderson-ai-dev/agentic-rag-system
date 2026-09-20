"""Structured logging configuration using structlog.

Emits JSON in production (for log aggregation) and a human-readable console
format in development. A secret-redaction processor masks any field whose name
looks like a credential before the record is written.
"""

import sys
from typing import Any

import structlog

_SENSITIVE_SUFFIXES = ("_key", "_token", "_secret")
_SENSITIVE_SUBSTRINGS = ("password", "authorization")


def _redact_secrets(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Replace values of credential-like fields with ``***``."""
    redacted: dict[str, Any] = {}
    for key, value in event_dict.items():
        lower = key.lower()
        if lower.endswith(_SENSITIVE_SUFFIXES) or any(
            token in lower for token in _SENSITIVE_SUBSTRINGS
        ):
            redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted


def configure_logging(level: str = "INFO", environment: str = "development") -> None:
    """Configure structlog with the renderer appropriate for the environment."""
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _redact_secrets,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer = (
        structlog.processors.JSONRenderer()
        if environment == "production"
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Return a module-level structlog logger."""
    return structlog.get_logger(name)
