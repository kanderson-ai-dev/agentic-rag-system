"""Record the README hero GIF: ask → answer → Prompts tab → diff → A/B → promote.

Drives the same deterministic stub as ``ui_qa.py`` so the capture is
reproducible offline. Playwright records ``webm``; the ffmpeg binary bundled
with Playwright converts it to a paletted GIF, time-compressed to fit the
~15 s hero budget.

    uv run python scripts/record_demo.py
"""

import contextlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.sync_api import sync_playwright  # noqa: E402
from ui_qa import start_server  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots" / "rag-demo.gif"


def _ffmpeg() -> str:
    """A full ffmpeg build — Playwright's bundled binary is a minimal
    vp8/webm-only build (no palettegen/GIF). ``imageio-ffmpeg`` ships a static
    executable (maintainer tooling: ``uv pip install imageio-ffmpeg``)."""
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "imageio-ffmpeg missing — `uv pip install imageio-ffmpeg`"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def record() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base, server = start_server(Path(tmpdir))
        video_dir = Path(tmpdir) / "video"
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                ctx = browser.new_context(
                    viewport={"width": 1280, "height": 720},
                    color_scheme="dark",
                    record_video_dir=str(video_dir),
                    record_video_size={"width": 1280, "height": 720},
                )
                page = ctx.new_page()

                # 1. Landing: ask → grounded answer.
                page.goto(f"{base}/", wait_until="networkidle")
                page.fill("#question", "What is LangGraph?")
                page.press("#question", "Enter")
                page.wait_for_selector("#answer:not(:empty)", timeout=30_000)
                with contextlib.suppress(Exception):
                    page.wait_for_selector(
                        "#answer [role='status']", state="detached",
                        timeout=30_000,
                    )
                page.wait_for_timeout(1800)

                # 2. Console → Prompts → Answer generation → diff.
                page.goto(f"{base}/console/", wait_until="networkidle")
                page.wait_for_selector(".prompt-row", timeout=10_000)
                page.locator("#prompts-panel").scroll_into_view_if_needed()
                page.wait_for_timeout(600)
                page.locator(".prompt-row").filter(
                    has_text="Answer generation"
                ).first.click()
                page.wait_for_selector(".diff-table", timeout=10_000)
                page.locator("#prompt-diff").scroll_into_view_if_needed()
                page.wait_for_timeout(1800)

                # 3. New candidate → create v3.
                page.click("text=New candidate")
                page.wait_for_selector("#candidate-modal[open]", timeout=10_000)
                page.wait_for_timeout(700)
                page.fill("#cand-author", "demo")
                page.fill("#cand-description", "demo candidate")
                page.fill("#cand-changelog", "readme hero demo")
                page.click("#cand-submit")
                page.wait_for_selector(
                    ".prompt-version:has-text('v3')", timeout=15_000
                )
                page.wait_for_timeout(600)

                # 4. A/B run → gate verdict.
                page.click("text=Run A/B test")
                page.wait_for_selector(".ab-verdict", timeout=30_000)
                page.locator("#prompt-ab").scroll_into_view_if_needed()
                page.wait_for_timeout(2000)

                # 5. Promote → v3 becomes live.
                page.click("button:has-text('Promote')")
                page.wait_for_selector("#confirm-modal[open]", timeout=10_000)
                page.wait_for_timeout(500)
                page.click("#confirm-ok")
                page.wait_for_selector(
                    ".prompt-version:has-text('v3') .pill-accent",
                    timeout=15_000,
                )
                page.wait_for_timeout(1500)

                webm_path = Path(page.video.path())
                ctx.close()
                browser.close()
        finally:
            server.should_exit = True

        # The video file is finalized asynchronously on context close — wait
        # for it to exist with a non-trivial size before handing to ffmpeg.
        deadline = time.time() + 15
        while time.time() < deadline:
            if webm_path.exists() and webm_path.stat().st_size > 10_000:
                break
            time.sleep(0.25)
        else:
            raise RuntimeError(f"webm not finalized: {webm_path}")
        webm = webm_path

        palette = Path(tmpdir) / "palette.png"
        ffmpeg = _ffmpeg()
        subprocess.run(
            [ffmpeg, "-y", "-i", str(webm), "-vf", "palettegen", str(palette)],
            check=True, capture_output=True,
        )
        subprocess.run(
            [
                ffmpeg, "-y", "-i", str(webm), "-i", str(palette),
                "-lavfi",
                "setpts=1.15*PTS,fps=14,scale=960:-1:flags=lanczos [x]; "
                "[x][1:v] paletteuse",
                str(OUT),
            ],
            check=True, capture_output=True,
        )
        print(f"  {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    t0 = time.time()
    record()
    print(f"done in {time.time() - t0:.0f}s")
    sys.exit(0)
