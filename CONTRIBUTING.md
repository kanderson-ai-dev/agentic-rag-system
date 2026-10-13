# Contributing

## Setup

```bash
uv sync
cp .env.example .env   # OPENAI_API_KEY enables the RAG path; vector/graph/auth all have local fallbacks
uv run uvicorn app.main:app --reload
```

## Conventions

- Python ≥ 3.12, package manager `uv`. `requirements.txt` is generated, never
  hand-edited: `uv export --no-hashes --no-dev --no-emit-project -o requirements.txt`
  (the `--no-emit-project` flag keeps the `-e .` self-install out — the Docker build
  installs the file before copying the source tree).
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

## Changing a prompt

Prompts are versioned artifacts (`prompts/<name>/vN.yaml`) — never edit the
promoted file in place:

1. `POST /api/v1/prompts/{name}/candidates` (or the console's "New candidate")
   creates `v(N+1)` as a **candidate** from the promoted template.
2. `POST /api/v1/prompts/{name}/ab-test` runs the measured A/B; poll
   `GET …/ab-test/{job_id}` until it finishes — the result carries both
   scorecards and the gate verdict.
3. `POST /api/v1/prompts/{name}/promote` only succeeds when the candidate meets
   every EDD threshold and doesn't regress the live version; the change is
   appended to `prompts/AUDIT.jsonl`. `POST …/rollback` restores the previous
   promoted version instantly.

Mutations need a JWT (when auth is configured) or `PROMPT_REGISTRY_WRITE_ENABLED=true`.
Every step is also visible — and clickable — in the console's **Prompts** tab.
- Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`).
- Never commit secrets. `.env`, `data/*.sqlite`, and internal planning docs are
  gitignored.
