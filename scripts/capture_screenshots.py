"""Capture real UI screenshots for ``docs/screenshots/``.

Requires the app running locally and Playwright installed in the venv
(intentionally not a project dependency — this is a maintainer-only tool):

    uv run uvicorn app.main:app --port 8000
    uv pip install playwright && playwright install chromium
    python scripts/capture_screenshots.py

Captures, all against the real running service:

- ``frontend.png`` — the public landing (``/``) answering a real question.
- ``dashboard-cost.png`` — the operator console's metric cards.
- ``dashboard-latency.png`` — the console's latency chart + recent-requests
  table, populated by real queries sent through the UI.
- ``ragas-scorecard.png`` — the versioned ``evaluation/results/
  ragas_scorecard.json`` rendered as a terminal-style capture.
- ``demo.webm`` (and ``demo.gif`` when ffmpeg is available via
  ``imageio-ffmpeg``) — a short recording of the landing answering a question.

The LangSmith experiment screenshot (``langsmith-experiment.png``) is not
automatable — it requires an interactive LangSmith session; see
``docs/screenshots/README.md``.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8000"
ROOT = Path(__file__).resolve().parent.parent
SHOTS_DIR = ROOT / "docs" / "screenshots"
SCORECARD_JSON = ROOT / "evaluation" / "results" / "ragas_scorecard.json"

LANDING_QUESTION = "What is LangGraph and how does it enable stateful agents?"
CONSOLE_QUESTIONS = [
    "What is LangGraph?",
    "How does the Self-RAG correction loop work?",
]
QUERY_TIMEOUT_MS = 120_000


def wait_for_ask_cycle(page, button_sel: str) -> None:
    """Wait for one full ask cycle: busy → idle (covers answer, error, or HITL).

    The Ask/Send button disables while the query is in flight and re-enables
    in the submit handler's `finally` for every outcome — the observable
    completion signal that needs no JS evaluation (CSP forbids eval).
    """
    page.wait_for_selector(f"{button_sel}[disabled]", timeout=10_000)
    page.wait_for_selector(f"{button_sel}:not([disabled])", timeout=QUERY_TIMEOUT_MS)


def save_screenshot(locator_or_page, path: Path, **kwargs) -> None:
    """Write a screenshot via a temp file + atomic replace, retrying briefly.

    The target may be momentarily locked by an editor/viewer previewing the
    previous capture — a short retry avoids flaky failures on Windows.
    """
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    last_error = None
    for _ in range(10):
        try:
            locator_or_page.screenshot(path=str(tmp), **kwargs)
            os.replace(tmp, path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.5)
    raise last_error


def capture_landing(browser) -> None:
    """Screenshot the landing answering a real question; record a demo clip."""
    with tempfile.TemporaryDirectory() as video_dir:
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            device_scale_factor=2,
            record_video_dir=video_dir,
            record_video_size={"width": 1280, "height": 900},
        )
        page = context.new_page()
        try:
            page.goto(BASE_URL + "/", wait_until="networkidle")
            page.fill("#question", LANDING_QUESTION)
            page.press("#question", "Enter")
            wait_for_ask_cycle(page, "#ask-button")
            # A transient backend error renders a Retry button — use it once.
            retry = page.locator("#answer [role='alert'] button")
            if retry.count():
                retry.first.click()
                wait_for_ask_cycle(page, "#ask-button")
            page.wait_for_timeout(400)  # let source chips / fonts settle
            save_screenshot(page.locator("main"), SHOTS_DIR / "frontend.png")
        finally:
            context.close()
        video_path = page.video.path() if page.video else None
        if video_path and Path(video_path).exists():
            shutil.copy(video_path, SHOTS_DIR / "demo.webm")


def ask_console(page, question: str) -> None:
    """Send a question through the console UI and wait for the outcome."""
    page.fill("#question", question)
    page.press("#question", "Enter")
    # The composer disables while the query is in flight and re-enables in the
    # submit handler's `finally` for every outcome (answer, blocked, error,
    # escalation).
    wait_for_ask_cycle(page, "#chat-send")
    if page.locator("#review-modal[open]").count():
        page.click("#review-cancel")
        page.wait_for_selector("#review-modal[open]", state="detached")


def capture_console(browser) -> None:
    """Drive the console with real questions, then capture the dashboard."""
    context = browser.new_context(
        viewport={"width": 1280, "height": 1400},
        device_scale_factor=2,
        color_scheme="dark",
    )
    page = context.new_page()
    try:
        page.goto(BASE_URL + "/console/", wait_until="networkidle")
        for question in CONSOLE_QUESTIONS:
            ask_console(page, question)
        page.wait_for_timeout(600)  # dashboard refresh settles after last answer

        summary = page.locator("#summary")
        summary.scroll_into_view_if_needed()
        save_screenshot(summary, SHOTS_DIR / "dashboard-cost.png")

        panel = page.locator("#dashboard-panel")
        panel.scroll_into_view_if_needed()
        save_screenshot(panel, SHOTS_DIR / "dashboard-latency.png")
    finally:
        context.close()


def capture_scorecard(browser) -> None:
    """Render the real versioned scorecard JSON as a terminal-style capture."""
    scorecard = json.loads(SCORECARD_JSON.read_text(encoding="utf-8"))
    lines = [
        "$ python evaluation/run_ragas.py",
        "",
        json.dumps(scorecard, indent=2),
        "",
        f"# written to {SCORECARD_JSON.relative_to(ROOT).as_posix()}",
    ]
    html = """<!DOCTYPE html><html><body style="margin:0;background:#0d1117;
        display:flex;justify-content:center;padding:32px">
        <pre id="t" style="background:#161b22;border:1px solid #30363d;
        border-radius:12px;padding:24px 28px;color:#c9d1d9;
        font:14px/1.6 Consolas,Menlo,monospace;margin:0"></pre>
        <script>document.getElementById("t").textContent = PAYLOAD;</script>
        </body></html>""".replace("PAYLOAD", json.dumps("\n".join(lines)))

    context = browser.new_context(
        viewport={"width": 760, "height": 480}, device_scale_factor=2
    )
    page = context.new_page()
    try:
        page.set_content(html)
        save_screenshot(page.locator("#t"), SHOTS_DIR / "ragas-scorecard.png")
    finally:
        context.close()


def maybe_make_gif() -> None:
    """Convert demo.webm → demo.gif if an ffmpeg binary is available."""
    webm = SHOTS_DIR / "demo.webm"
    gif = SHOTS_DIR / "demo.gif"
    if not webm.exists():
        return
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found — keeping demo.webm only (GIF skipped).")
        return
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(webm),
            "-vf",
            "fps=10,scale=880:-1:flags=lanczos",
            str(gif),
        ],
        check=True,
        capture_output=True,
    )
    webm.unlink()
    print("demo.gif written")


def main() -> int:
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        capture_landing(browser)
        capture_console(browser)
        capture_scorecard(browser)
        browser.close()
    maybe_make_gif()
    for path in sorted(SHOTS_DIR.iterdir()):
        if path.suffix in {".png", ".gif"}:
            print(f"  {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
