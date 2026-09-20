"""Input and output guardrails for the Self-RAG graph (OWASP LLM01/LLM02).

Implements the guardrail-first architecture from PLAN.md Fase 1: malicious or
malformed input is detected and blocked *before* it reaches the retriever or
the LLM, and the final generated output is screened for system-prompt leakage
or reflected injected instructions before being returned.
"""

import re
import unicodedata
from collections.abc import Iterable

# Cap on the sanitized input length, as a cheap defense against model DoS
# (OWASP LLM04) and to keep retrieval/LLM costs bounded.
MAX_INPUT_LENGTH = 2000

# Minimum number of shared 5-grams between the output and a system prompt that
# counts as a likely system-prompt leak.
LEAK_NGRAM_THRESHOLD = 3

# (category, raw regex) pairs. `detect_prompt_injection` returns the category of
# the first match, which doubles as a stable, testable reason code.
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (
        "ignore_previous_instructions",
        r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+(instructions|prompts?)",
    ),
    (
        "disregard_instructions",
        r"disregard\s+(all\s+)?(previous\s+)?(instructions|rules)",
    ),
    (
        "reveal_system_prompt",
        r"reveal\s+(your|the)\s+(system\s+)?prompt",
    ),
    (
        "fake_role_header",
        r"(^|\n)\s*(system|assistant|developer)\s*:\s*(ignore|you\s+are|your\s+role)",
    ),
    (
        "fake_special_token",
        r"<\|[^|>]{1,64}\|>",
    ),
    (
        "jailbreak_dan",
        r"\b(do\s+anything\s+now|jailbreak)\b",
    ),
    (
        "bypass_safety_filters",
        r"bypass\s+(the\s+)?(safety|content|security|moderation)\s+filters?",
    ),
]

INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (category, re.compile(pattern, re.IGNORECASE))
    for category, pattern in _INJECTION_PATTERNS
]

# Unicode ranges treated as "invisible"/control characters and stripped out.
_ZERO_WIDTH_RANGES = (
    (0x200B, 0x200F),  # zero-width space, non-joiner, joiner, LRM/RLM
    (0x202A, 0x202E),  # bidi embedding/override control
    (0x2060, 0x206F),  # word joiner, invisible operators
    (0xFEFF, 0xFEFF),  # byte order mark / zero-width no-break space
)


def _is_allowed_char(char: str) -> bool:
    """Return True for ordinary printable text characters."""
    code = ord(char)
    if code < 32 and char not in "\t\n\r":
        return False
    if code == 0x7F or 0x80 <= code <= 0x9F:
        return False
    return all(not start <= code <= end for start, end in _ZERO_WIDTH_RANGES)


def sanitize_input(raw_text: str) -> str:
    """Normalize Unicode (NFKC), strip control/invisible characters, cap length."""
    normalized = unicodedata.normalize("NFKC", raw_text)
    cleaned = "".join(char for char in normalized if _is_allowed_char(char))
    collapsed = " ".join(cleaned.split())
    return collapsed[:MAX_INPUT_LENGTH]


def detect_prompt_injection(text: str) -> str | None:
    """Return the category of the first injection pattern matched, else `None`."""
    for category, pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return category
    return None


def _shared_ngrams(a: str, b: str, n: int = 5) -> int:
    """Count distinct word n-grams present in both strings (case-insensitive)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    tokens_a = a.lower().split()
    tokens_b = b.lower().split()
    if len(tokens_a) < n or len(tokens_b) < n:
        return 0
    ngrams_a = {" ".join(tokens_a[i : i + n]) for i in range(len(tokens_a) - n + 1)}
    ngrams_b = {" ".join(tokens_b[i : i + n]) for i in range(len(tokens_b) - n + 1)}
    return len(ngrams_a & ngrams_b)


def screen_output(text: str, protected_texts: Iterable[str] | None = None) -> str | None:
    """Detect reflected injected instructions or system-prompt leakage.

    Returns a reason string if the output is flagged, else `None`. Callers can
    inject their own ``protected_texts`` for unit testing; by default the graph's
    system prompts are used.
    """
    reflected = detect_prompt_injection(text)
    if reflected:
        return f"reflected_injection:{reflected}"

    protected = protected_texts if protected_texts is not None else SYSTEM_PROMPT_TEXTS
    for prompt_text in protected:
        if _shared_ngrams(text, prompt_text) >= LEAK_NGRAM_THRESHOLD:
            return "system_prompt_leak"
    return None


def _system_prompt_texts() -> list[str]:
    """Extract the system-message text from the graph's prompt templates."""
    from app.graph.prompts import (
        GENERATE_ANSWER_PROMPT,
        GRADE_DOCUMENTS_PROMPT,
        TRANSFORM_QUERY_PROMPT,
    )

    texts: list[str] = []
    for prompt in (GRADE_DOCUMENTS_PROMPT, GENERATE_ANSWER_PROMPT, TRANSFORM_QUERY_PROMPT):
        for message in prompt.messages:
            template = getattr(getattr(message, "prompt", None), "template", None)
            if template:
                texts.append(template)
    return texts


SYSTEM_PROMPT_TEXTS = _system_prompt_texts()
