# Security

This document maps the security controls implemented in this project to the
OWASP Top 10 (web) and the OWASP Top 10 for LLM Applications. It serves as a
checklist for closing out each phase.

> The repository is intended to be public. Never commit secrets; they live in
> `.env` locally and in GitHub Actions Secrets in CI.

## OWASP Top 10 (Web, 2021)

| Risk | Mitigation | Where |
|---|---|---|
| A01 Broken Access Control | JWT `verify_jwt`, HITL for low-confidence answers | `app/core/dependencies.py` |
| A02 Cryptographic Failures | `SecretStr`, bcrypt, signed JWT; TLS delegated to hosting | `app/core/config.py`, `app/core/security.py` |
| A03 Injection | Input guardrails, parameterized Cypher | `app/graph/guardrails.py`, `app/services/graph_store.py` |
| A04 Insecure Design | Guardrail-first + HITL + bounded retrieval | graph architecture |
| A05 Security Misconfiguration | Security headers middleware, non-root Docker user | `app/api/middleware.py`, `Dockerfile` |
| A06 Vulnerable Components | `pip-audit` in CI | `.github/workflows/ci.yml` |
| A07 Auth Failures | Short-lived JWT, generic errors, rate-limited login | `app/api/v1/routes/auth.py` |
| A08 Software/Data Integrity | Pinned deps (`uv.lock`), `gitleaks` in CI | `pyproject.toml`, CI |
| A09 Logging/Monitoring Failures | `structlog` + request ID + Prometheus, secret redaction | `app/core/logging.py` |
| A10 SSRF | No tool accepts arbitrary URLs for direct fetch (design constraint) | — |

## OWASP Top 10 for LLM Applications

| Risk | Mitigation | Where |
|---|---|---|
| LLM01 Prompt Injection | Input guardrail | `app/graph/guardrails.py` |
| LLM02 Insecure Output Handling | Output guardrail | `app/graph/guardrails.py` |
| LLM03 Training Data Poisoning | N/A (model is not re-trained) | documented |
| LLM04 Model DoS | Input length limit, bounded `top_k` | `app/graph/guardrails.py`, `app/services/vector_store.py` |
| LLM05 Supply Chain Vulnerabilities | `pip-audit`, pinned versions | CI |
| LLM06 Sensitive Information Disclosure | Log redaction, `usage_store` without conversation content | `app/core/logging.py`, `app/services/usage_store.py` |
| LLM07 Insecure Plugin Design | Parameterized Cypher, explicit input validation | `app/services/graph_store.py` |
| LLM08 Excessive Agency | HITL before accepting low-confidence answers | `app/graph/nodes.py` |
| LLM09 Overreliance | LLM-as-judge + real evaluation | `evaluation/` |
| LLM10 Model Theft | N/A (model consumed via API) | documented |

## Fork/PR threat model

CI uses `on: pull_request` (never `pull_request_target`) and
`permissions: contents: read`, so a pull request from an external fork cannot
read repository secrets. All optional features degrade cleanly when their
secrets are absent (no insecure fallback to a hardcoded key).

## Known dependency advisories

`pip-audit` runs in CI. Two notes on currently-handled findings:

- **chromadb (1.5.9)** — the flagged advisories (PYSEC-2026-311/3813/3814/3815)
  affect the Chroma **server** (multi-tenant RBAC and pre-auth code injection).
  This project uses chromadb only as an **embedded local store** (no server, no
  network exposure), so those vectors do not apply. No fixed release exists yet,
  so they are allowlisted in the CI `pip-audit` step.
- **langgraph-checkpoint-sqlite** — SQL-injection advisories
  (PYSEC-2026-1528/1529/3636, CVE-2025-67644) were fixed by upgrading to 3.1.1.
