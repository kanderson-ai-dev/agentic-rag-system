# Contributing

## Setup

```bash
uv sync
cp .env.example .env   # OPENAI_API_KEY enables the RAG path; vector/graph/auth all have local fallbacks
uv run uvicorn app.main:app --reload
```

## Conventions

- Python ≥ 3.12, package manager `uv`. `requirements.txt` is generated (`uv export`),
  never hand-edited.
- `ruff check .` must pass (line-length 100, rules `E,F,I,UP,B,SIM`, ignore `B008`).
- `mypy --strict app/` must pass with 0 errors.
- Tests must be deterministic and fully offline — no network, no real credentials.
  Credential-dependent tests use the `requires_*` markers in `tests/conftest.py`
  (`requires_pinecone`, `requires_neo4j`, `requires_openai_key`,
  `requires_langsmith_key`).
- EDD gate: changes to prompts, retrieval, or graph architecture are not "done" until
  `python evaluation/run_ragas.py` still meets the versioned thresholds
  (`faithfulness`/`answer_relevancy` ≥ 0.85, `context_precision`/`context_recall`
  ≥ 0.75). A quality regression fails CI like a failing test.
- Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`).
- Never commit secrets. `.env`, `data/*.sqlite`, and internal planning docs are
  gitignored.
