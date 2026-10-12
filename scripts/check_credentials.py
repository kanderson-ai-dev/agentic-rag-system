"""Credential smoke tests for the external services the app can talk to.

Runs one minimal call per configured service and reports only a status word —
`ok`, `skipped (not configured)` or `failed: <ExceptionClass>` — so it is safe
to paste the output into a report: no key value, URI or error body is ever
printed. Unconfigured services are skipped (the app degrades cleanly without
them); a configured service that fails exits with status 1.

    python scripts/check_credentials.py [--max-cost-usd 1.00]
"""

import argparse
import sys

from app.core.config import get_settings


def _check_openai(max_cost_usd: float) -> str:
    """One tiny chat completion; verifies the key and the configured model."""
    settings = get_settings()
    if not settings.openai_api_key:
        return "skipped (not configured)"
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key_value())
        response = client.chat.completions.create(
            model=settings.chat_model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        usage = response.usage
        if usage is not None:
            cost = (
                usage.prompt_tokens * settings.cost_input_price_per_1m
                + usage.completion_tokens * settings.cost_output_price_per_1m
            ) / 1_000_000
            if cost > max_cost_usd:
                return f"failed: CostCapExceeded ({cost:.6f} USD > {max_cost_usd} USD)"
        return "ok"
    except Exception as exc:  # noqa: BLE001 - report exception class only
        return f"failed: {type(exc).__name__}"


def _check_langsmith() -> str:
    """Lists one project; verifies the LangSmith API key."""
    settings = get_settings()
    if not settings.langsmith_api_key:
        return "skipped (not configured)"
    try:
        from langsmith import Client

        client = Client(api_key=settings.langsmith_api_key.get_secret_value())
        next(iter(client.list_projects(limit=1)), None)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"failed: {type(exc).__name__}"


def _check_pinecone() -> str:
    """Checks the configured index exists and returns stats."""
    settings = get_settings()
    if not settings.pinecone_api_key:
        return "skipped (not configured)"
    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=settings.pinecone_api_key_value())
        names = [index.name for index in pc.list_indexes()]
        if settings.pinecone_index_name not in names:
            return "failed: IndexNotFound"
        pc.Index(settings.pinecone_index_name).describe_index_stats()
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"failed: {type(exc).__name__}"


def _check_neo4j() -> str:
    """Verifies connectivity to the configured Neo4j instance."""
    settings = get_settings()
    if not (settings.neo4j_uri and settings.neo4j_password):
        return "skipped (not configured)"
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password_value() or ""),
            connection_timeout=5.0,
        )
        try:
            driver.verify_connectivity()
        finally:
            driver.close()
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"failed: {type(exc).__name__}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=1.00,
        help="Abort if the tiny OpenAI call would exceed this cost (default: 1.00).",
    )
    args = parser.parse_args()

    results = {
        "openai": _check_openai(args.max_cost_usd),
        "langsmith": _check_langsmith(),
        "pinecone": _check_pinecone(),
        "neo4j": _check_neo4j(),
    }
    for service, status in results.items():
        print(f"{service}: {status}")

    failed = [name for name, status in results.items() if status.startswith("failed")]
    if failed:
        print(f"Configured but unreachable: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
