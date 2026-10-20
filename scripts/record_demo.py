"""Record the README hero GIF: ask → answer → Prompts tab → diff → A/B → promote.

Two modes:

- **Fixture (default)** — the deterministic stub backend from ``ui_qa.py``,
  so the capture is reproducible offline; the A/B numbers it shows are
  fixture data, not a measurement.
- **Live (``--live``)** — the real app end-to-end (real model calls, real
  A/B engine, real promote); needs ``OPENAI_API_KEY`` and records real
  numbers into ``evaluation/results/ab/``.

    uv run python scripts/record_demo.py [--live]

Playwright records ``webm``; a full ffmpeg (via ``imageio-ffmpeg``) converts
it to a paletted GIF at near-real-time pacing — every caption stays on
screen long enough to be read.
"""

import argparse
import contextlib
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ui_qa  # noqa: E402  (import before _configure_env — it is env-inert)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots" / "rag-demo.gif"
OUT_LIVE = ROOT / "docs" / "screenshots" / "rag-demo-live.gif"


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


def _black_lead(video: Path) -> float:
    """Seconds of initial black — browser startup before the first paint."""
    out = subprocess.run(
        [
            _ffmpeg(), "-i", str(video),
            "-vf", "blackdetect=d=0.1:pix_th=0.08", "-an", "-f", "null", "-",
        ],
        check=True, capture_output=True, text=True,
    ).stderr
    m = re.search(r"black_start:0[\d.]* black_end:(\d+\.?\d*)", out)
    return float(m.group(1)) if m else 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="real app + real model calls (needs OPENAI_API_KEY, real spend)",
    )
    return parser.parse_args()


def record() -> None:
    args = _parse_args()
    ui_qa._configure_env(args.live)
    from playwright.sync_api import sync_playwright

    answer_wait = 120_000 if args.live else 30_000
    ab_wait = 900_000 if args.live else 30_000
    with tempfile.TemporaryDirectory() as tmpdir:
        base, server = ui_qa.start_server(Path(tmpdir), live=args.live)
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

                # Read beats are deliberately generous — every caption must
                # stay on screen long enough to be read comfortably.
                # 1. Landing: ask → grounded answer.
                page.goto(f"{base}/", wait_until="networkidle")
                page.fill("#question", "What is LangGraph?")
                page.wait_for_timeout(2500)
                page.press("#question", "Enter")
                page.wait_for_selector(
                    "#answer:not(:empty)", timeout=answer_wait
                )
                with contextlib.suppress(Exception):
                    page.wait_for_selector(
                        "#answer [role='status']", state="detached",
                        timeout=answer_wait,
                    )
                page.wait_for_timeout(8000)

                # 2. Console → Prompts → Answer generation → diff.
                page.goto(f"{base}/console/", wait_until="networkidle")
                page.wait_for_selector(".prompt-row", timeout=10_000)
                page.locator("#prompts-panel").scroll_into_view_if_needed()
                page.wait_for_timeout(3000)
                page.locator(".prompt-row").filter(
                    has_text="Answer generation"
                ).first.click()
                page.wait_for_selector(".diff-table", timeout=10_000)
                page.locator("#prompt-diff").scroll_into_view_if_needed()
                page.wait_for_timeout(8000)

                # 3. New candidate → create v3 (the real hypothesis in live
                # mode so the run is a genuine prompt change, not a copy).
                page.click("text=New candidate")
                page.wait_for_selector("#candidate-modal[open]", timeout=10_000)
                page.wait_for_timeout(2500)
                if args.live:
                    page.fill("#cand-system", ui_qa.LIVE_CANDIDATE["system"])
                    page.fill("#cand-human", ui_qa.LIVE_CANDIDATE["human"])
                    page.fill("#cand-author", ui_qa.LIVE_CANDIDATE["author"])
                    page.fill(
                        "#cand-description", ui_qa.LIVE_CANDIDATE["description"]
                    )
                    page.fill(
                        "#cand-changelog", ui_qa.LIVE_CANDIDATE["changelog"]
                    )
                else:
                    page.fill("#cand-author", "demo")
                    page.fill("#cand-description", "demo candidate")
                    page.fill("#cand-changelog", "readme hero demo")
                page.wait_for_timeout(3000)
                page.click("#cand-submit")
                page.wait_for_selector(
                    ".prompt-version:has-text('v3')", timeout=15_000
                )
                page.wait_for_timeout(3500)

                # 4. A/B run → gate verdict — the scorecards are the densest
                # text in the demo; give them the longest read beat.
                page.click("text=Run A/B test")
                page.wait_for_selector(".ab-verdict", timeout=ab_wait)
                page.locator("#prompt-ab").scroll_into_view_if_needed()
                page.wait_for_timeout(9000)

                # 5. Promote → v3 becomes live. In --live mode the gate
                # verdict is real: promote only when the run was allowed —
                # a blocked gate is honest evidence, not a skipped beat.
                verdict_text = (
                    page.locator(".ab-verdict").inner_text().lower()
                )
                gate_allowed = "allow" in verdict_text
                if not args.live or gate_allowed:
                    page.click("button:has-text('Promote')")
                    page.wait_for_selector(
                        "#confirm-modal[open]", timeout=10_000
                    )
                    page.wait_for_timeout(2500)
                    page.click("#confirm-ok")
                    page.wait_for_selector(
                        ".prompt-version:has-text('v3') .pill-accent",
                        timeout=15_000,
                    )
                    page.wait_for_timeout(6000)

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
        out = OUT_LIVE if args.live else OUT
        # Trim the dead black lead (browser launch before first paint) so the
        # GIF starts on the landing, not on a black screen.
        seek = ["-ss", f"{_black_lead(webm):.2f}"]
        subprocess.run(
            [
                ffmpeg, "-y", *seek, "-i", str(webm),
                "-vf", "palettegen", str(palette),
            ],
            check=True, capture_output=True,
        )
        subprocess.run(
            [
                ffmpeg, "-y", *seek, "-i", str(webm), "-i", str(palette),
                "-lavfi",
                "setpts=1.0*PTS,fps=14,scale=960:-1:flags=lanczos [x]; "
                "[x][1:v] paletteuse",
                str(out),
            ],
            check=True, capture_output=True,
        )
        print(f"  {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    t0 = time.time()
    record()
    print(f"done in {time.time() - t0:.0f}s")
    sys.exit(0)
