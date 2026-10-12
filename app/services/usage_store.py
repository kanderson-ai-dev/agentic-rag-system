"""Persistent per-request usage store (SQLite).

Only numeric metadata is stored (tokens, cost, latency, flags) — never the
content of the question or the answer — to minimize exposure if the file leaks.
"""

import contextlib
import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class UsageStore:
    """Append-only SQLite store for request usage metadata."""

    def __init__(self, db_path: str) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # `check_same_thread=False` allows the store to be created in the
        # lifespan and used from FastAPI's threadpool request handlers.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                session_id TEXT,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                latency_ms INTEGER NOT NULL,
                blocked INTEGER NOT NULL,
                escalated INTEGER NOT NULL,
                retry_count INTEGER NOT NULL
            )
            """
        )
        # Backfill `session_id` for databases created before this column existed.
        with contextlib.suppress(sqlite3.OperationalError):
            self._conn.execute("ALTER TABLE usage ADD COLUMN session_id TEXT")
        # Backfill `prompt_versions` (JSON object {prompt_name: version}) the
        # same way for databases created before the prompt registry existed.
        with contextlib.suppress(sqlite3.OperationalError):
            self._conn.execute("ALTER TABLE usage ADD COLUMN prompt_versions TEXT")
        # Backfill `node_latencies` (JSON object {node_name: ms}) for
        # databases created before per-node timing was persisted.
        with contextlib.suppress(sqlite3.OperationalError):
            self._conn.execute("ALTER TABLE usage ADD COLUMN node_latencies TEXT")
        self._conn.commit()

    def record(
        self,
        *,
        thread_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: int,
        blocked: bool,
        escalated: bool,
        retry_count: int,
        session_id: str | None = None,
        prompt_versions: dict[str, int] | None = None,
        node_latencies: dict[str, float] | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO usage
                (timestamp, thread_id, session_id, prompt_tokens, completion_tokens,
                 cost_usd, latency_ms, blocked, escalated, retry_count,
                 prompt_versions, node_latencies)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(UTC).isoformat(),
                thread_id,
                session_id,
                prompt_tokens,
                completion_tokens,
                cost_usd,
                latency_ms,
                int(blocked),
                int(escalated),
                retry_count,
                (
                    json.dumps(prompt_versions, sort_keys=True)
                    if prompt_versions
                    else None
                ),
                (
                    json.dumps(node_latencies, sort_keys=True)
                    if node_latencies
                    else None
                ),
            ),
        )
        self._conn.commit()

    def summary(self, session_id: str | None = None) -> dict[str, Any]:
        """Aggregate usage metrics.

        With no `session_id`, aggregates across all recorded requests (useful
        for operators). Passed a `session_id`, scopes the aggregate to that
        browser session only — this is what the frontend uses so a first-time
        visitor (a fresh session, e.g. a new tab) always sees a dashboard
        starting at zero, instead of the service's lifetime totals.
        """
        where = "WHERE session_id = ?" if session_id is not None else ""
        params = (session_id,) if session_id is not None else ()
        row = self._conn.execute(
            f"""
            SELECT
                COUNT(*),
                COALESCE(SUM(prompt_tokens), 0),
                COALESCE(SUM(completion_tokens), 0),
                COALESCE(SUM(cost_usd), 0),
                COALESCE(AVG(latency_ms), 0),
                COALESCE(SUM(blocked), 0),
                COALESCE(SUM(escalated), 0)
            FROM usage {where}
            """,
            params,
        ).fetchone()
        return {
            "total_requests": row[0],
            "total_prompt_tokens": row[1],
            "total_completion_tokens": row[2],
            "total_cost_usd": row[3],
            "avg_latency_ms": row[4],
            "blocked_requests": row[5],
            "escalated_requests": row[6],
        }

    def recent(
        self, limit: int = 20, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        where = "WHERE session_id = ?" if session_id is not None else ""
        params: tuple[Any, ...] = (session_id, limit) if session_id is not None else (limit,)
        rows = self._conn.execute(
            f"""
            SELECT timestamp, thread_id, prompt_tokens, completion_tokens,
                   cost_usd, latency_ms, blocked, escalated, retry_count,
                   prompt_versions
            FROM usage {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [
            {
                "timestamp": row[0],
                "thread_id": row[1],
                "prompt_tokens": row[2],
                "completion_tokens": row[3],
                "cost_usd": row[4],
                "latency_ms": row[5],
                "blocked": bool(row[6]),
                "escalated": bool(row[7]),
                "retry_count": row[8],
                "prompt_versions": (
                    json.loads(row[9]) if row[9] else None
                ),
            }
            for row in rows
        ]

    def summary_by_prompt_version(
        self, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Requests, average cost and average latency per `generate_answer`
        prompt version — the evidence that a prompt change is measurable.

        Aggregated in Python (the usage store is demo-scale) rather than via
        SQLite's JSON1 extension so it works on every bundled sqlite3.
        Scoped to ``session_id`` like :meth:`summary` when one is given.
        """
        aggregates: dict[int, dict[str, Any]] = {}
        where = "AND session_id = ?" if session_id is not None else ""
        params = (session_id,) if session_id is not None else ()
        rows = self._conn.execute(
            f"SELECT prompt_versions, cost_usd, latency_ms FROM usage "
            f"WHERE prompt_versions IS NOT NULL {where}",
            params,
        ).fetchall()
        for versions_json, cost_usd, latency_ms in rows:
            version = json.loads(versions_json).get("generate_answer")
            if version is None:
                continue
            bucket = aggregates.setdefault(
                int(version), {"requests": 0, "cost": 0.0, "latency": 0.0}
            )
            bucket["requests"] += 1
            bucket["cost"] += cost_usd
            bucket["latency"] += latency_ms
        return [
            {
                "generate_answer_version": version,
                "requests": bucket["requests"],
                "avg_cost_usd": bucket["cost"] / bucket["requests"],
                "avg_latency_ms": bucket["latency"] / bucket["requests"],
            }
            for version, bucket in sorted(aggregates.items())
        ]

    def daily_spend_usd(self) -> float:
        """Total recorded LLM cost since UTC midnight.

        Timestamps are stored as UTC ISO-8601, so the leading 10 chars are
        the UTC day — the same clock the daily budget gate enforces on. The
        aggregate is deliberately global (never session-scoped): the budget
        protects the deployment's wallet, not one visitor.
        """
        today = datetime.now(UTC).date().isoformat()
        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM usage "
            "WHERE substr(timestamp, 1, 10) = ?",
            (today,),
        ).fetchone()
        return float(row[0])

    def budget_status(self, daily_limit_usd: float) -> dict[str, Any]:
        """Position against the daily budget: spent, remaining, alert flags.

        ``warning`` fires at >= 80% (the visible alert band) and
        ``exhausted`` at >= 100% — the deterministic block. A non-positive
        limit counts as already exhausted.
        """
        spent = self.daily_spend_usd()
        used_pct = (
            spent / daily_limit_usd * 100.0 if daily_limit_usd > 0 else 100.0
        )
        return {
            "daily_limit_usd": daily_limit_usd,
            "spent_today_usd": spent,
            "remaining_usd": max(0.0, daily_limit_usd - spent),
            "used_pct": used_pct,
            "warning": used_pct >= 80.0,
            "exhausted": spent >= daily_limit_usd,
        }

    def performance(self, session_id: str | None = None) -> dict[str, Any]:
        """Latency/cost aggregates for the performance dashboard panel.

        Returns end-to-end percentiles, per-node p50/p95 (from the persisted
        ``node_latencies`` JSON), token and cost averages, a 7-day cost trend,
        block/escalation rates, and per-prompt-version aggregates — all
        scoped to ``session_id`` like :meth:`summary`.
        """
        where = "WHERE session_id = ?" if session_id is not None else ""
        params = (session_id,) if session_id is not None else ()
        rows = self._conn.execute(
            f"""
            SELECT timestamp, prompt_tokens, completion_tokens, cost_usd,
                   latency_ms, blocked, escalated, node_latencies
            FROM usage {where}
            """,
            params,
        ).fetchall()

        latencies = [row[4] for row in rows]
        costs = [row[3] for row in rows]
        tokens = [row[1] + row[2] for row in rows]
        n = len(rows)

        node_samples: dict[str, list[float]] = {}
        for raw in (row[7] for row in rows):
            if not raw:
                continue
            for node, ms in json.loads(raw).items():
                node_samples.setdefault(node, []).append(float(ms))
        per_node = [
            {
                "node": node,
                "samples": len(samples),
                "p50_ms": percentile(samples, 50),
                "p95_ms": percentile(samples, 95),
            }
            for node, samples in sorted(node_samples.items())
        ]

        trend_rows = self._conn.execute(
            f"""
            SELECT substr(timestamp, 1, 10) AS day, SUM(cost_usd), COUNT(*)
            FROM usage {where}
            GROUP BY day
            ORDER BY day DESC
            LIMIT 7
            """,
            params,
        ).fetchall()
        daily_cost_trend = [
            {"date": day, "cost_usd": cost, "requests": count}
            for day, cost, count in reversed(trend_rows)
        ]

        return {
            "requests": n,
            "latency": {
                "p50_ms": percentile(latencies, 50),
                "p95_ms": percentile(latencies, 95),
                "target_p50_ms": 3000,
                "target_p95_ms": 6000,
            },
            "cost": {
                "total_usd": sum(costs),
                "avg_per_query_usd": sum(costs) / n if n else 0.0,
                "target_per_query_usd": 0.01,
            },
            "tokens": {"avg_per_query": sum(tokens) / n if n else 0.0},
            "rates": {
                "blocked": sum(row[5] for row in rows) / n if n else 0.0,
                "escalated": sum(row[6] for row in rows) / n if n else 0.0,
            },
            "per_node": per_node,
            "daily_cost_trend": daily_cost_trend,
            "per_prompt_version": self.summary_by_prompt_version(session_id),
        }


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile over a small sample set (0.0 when empty)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[index]
