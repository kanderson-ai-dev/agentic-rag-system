"""Phase 10 — Prompts console tab and the human-readable label dictionary.

Covers the static mount of the new modules, the ``label(kind, key)`` contract
(no raw identifiers in the UI), and a drift sweep: every identifier the
backend can emit (prompt names, lifecycle statuses, graph node names,
interrupt reasons) must have a dictionary entry, so a new enum member fails
tests instead of leaking ``snake_case`` into the console.
"""

import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services.prompt_registry import PromptName, PromptStatus

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_JS = REPO_ROOT / "frontend" / "js" / "labels.js"
GRAPH_PY = REPO_ROOT / "app" / "graph" / "graph.py"


def _label_entries(kind: str) -> set[str]:
    """Parse one ``kind: { ... }`` block out of labels.js."""
    source = LABELS_JS.read_text(encoding="utf-8")
    block = re.search(rf"\b{kind}:\s*{{(.*?)}}", source, re.DOTALL)
    assert block, f"labels.js has no `{kind}` block"
    return set(re.findall(r'"([a-z0-9_]+)":', block.group(1)))


def test_labels_module_served_and_exposes_label_helper() -> None:
    client = TestClient(app)
    response = client.get("/js/labels.js")
    client.close()

    assert response.status_code == 200
    assert "export function label" in response.text
    # Missing entries must warn (drift signal) rather than render raw ids.
    assert "console.warn" in response.text


def test_label_dictionary_covers_prompt_names() -> None:
    entries = _label_entries("prompt")
    assert {p.value for p in PromptName} <= entries


def test_label_dictionary_covers_lifecycle_statuses() -> None:
    entries = _label_entries("status")
    assert {s.value for s in PromptStatus} <= entries


def test_label_dictionary_covers_graph_nodes() -> None:
    node_names = set(re.findall(r'add_node\("(\w+)"', GRAPH_PY.read_text(encoding="utf-8")))
    assert node_names, "no graph nodes found — graph.py layout changed?"
    entries = _label_entries("node") | _label_entries("prompt")
    assert node_names <= entries


def test_label_dictionary_covers_interrupt_reasons() -> None:
    # The only reason emitted by the retry loop today; the sweep keeps the
    # dictionary honest if edges.py gains new interrupt reasons.
    assert "max_retries_exhausted" in _label_entries("interrupt_reason")


def test_prompts_module_served_and_wired() -> None:
    client = TestClient(app)
    module = client.get("/console/js/prompts.js")
    app_js = client.get("/console/js/app.js").text
    html = client.get("/console/").text
    client.close()

    assert module.status_code == 200
    assert "import" in module.text and "export" in module.text
    assert "initPrompts" in app_js

    # Section skeleton + accessible dialogs live in the markup (CSP-safe:
    # nothing rendered inline by JS string templates).
    for marker in (
        'id="prompts-panel"',
        'id="prompt-list"',
        'id="prompt-detail"',
        'id="candidate-modal"',
        'id="confirm-modal"',
        'id="prompts-auth"',
        'id="prompts-writes-note"',
    ):
        assert marker in html


def test_prompts_ui_humanizes_every_identifier() -> None:
    """prompts.js must route identifiers through `label()` — no raw prints."""
    client = TestClient(app)
    source = client.get("/console/js/prompts.js").text
    client.close()

    assert 'label("prompt"' in source
    assert 'label("status"' in source
    assert 'label("metric"' in source
    assert 'label("job_status"' in source
    # Diff markers are text glyphs (+/−), not colour-only signalling.
    assert '"+"' in source and '"−"' in source


def test_prompts_ui_explains_disabled_writes() -> None:
    """Mutating controls degrade visibly when writes/auth are unavailable."""
    client = TestClient(app)
    source = client.get("/console/js/prompts.js").text
    client.close()

    assert "writes_enabled" in source
    assert "aria-describedby" in source
    assert "auth_required" in source or "/api/v1/auth/status" in source


def test_chat_footer_attributes_prompt_version() -> None:
    client = TestClient(app)
    source = client.get("/console/js/chat.js").text
    client.close()

    assert "prompt_versions" in source
    assert "Answered with" in source
    assert 'label("prompt"' in source


def test_dashboard_recent_table_has_prompt_column() -> None:
    client = TestClient(app)
    html = client.get("/console/").text
    source = client.get("/console/js/dashboard.js").text
    client.close()

    assert "<th scope=\"col\">Prompt</th>" in html
    assert "prompt_versions" in source


def test_query_response_carries_prompt_versions() -> None:
    """The chat footer needs the pinned versions on the query response."""
    from app.api.v1.schemas import QueryResponse

    resp = QueryResponse.from_graph_result(
        {
            "generation": "An answer.",
            "documents": [],
            "prompt_versions": {"generate_answer": 2, "grade_documents": 1},
        },
        "t-1",
    )
    assert resp.prompt_versions["generate_answer"] == 2


def test_landing_counter_and_escalation_copy() -> None:
    client = TestClient(app)
    html = client.get("/").text
    js = client.get("/js/landing.js").text
    client.close()

    assert 'id="char-count"' in html
    assert "MAX_QUESTION_CHARS" in js
    assert "updateCounter" in js
    assert "I couldn't find a reliable answer" in js
