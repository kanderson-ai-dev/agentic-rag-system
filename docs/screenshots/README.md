# Screenshots

Drop your captures here and reference them from the root `README.md`. The README
already contains image placeholders pointing at these paths.

Expected files:

| File | What to capture |
|---|---|
| `langsmith-experiment.png` | The LangSmith experiment / run list after `python evaluation/run_ragas.py` |
| `ragas-scorecard.png` | The scorecard JSON output (terminal or `evaluation/results/ragas_scorecard.json`) |
| `dashboard-cost.png` | The frontend dashboard cards (total cost) at `http://localhost:8000/` |
| `dashboard-latency.png` | The dashboard latency cards and recent-requests table |
| `frontend.png` | The chat UI with a question and its answer |

A typical workflow to produce them:

1. Start the app: `uv run uvicorn app.main:app --reload`
2. Open `http://localhost:8000/`, sign in, ask a question, and capture the chat.
3. Open the dashboard section and capture cost/latency cards.
4. Run the evaluation (`python evaluation/run_ragas.py`) and capture the LangSmith
   experiment + the scorecard output.
