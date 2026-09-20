"""Persistent per-request usage store (SQLite).

Only numeric metadata is stored (tokens, cost, latency, flags) — never the
content of the question or the answer — to minimize exposure if the file leaks.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path


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
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO usage
                (timestamp, thread_id, prompt_tokens, completion_tokens,
                 cost_usd, latency_ms, blocked, escalated, retry_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(UTC).isoformat(),
                thread_id,
                prompt_tokens,
                completion_tokens,
                cost_usd,
                latency_ms,
                int(blocked),
                int(escalated),
                retry_count,
            ),
        )
        self._conn.commit()

    def summary(self) -> dict:
        row = self._conn.execute(
            """
            SELECT
                COUNT(*),
                COALESCE(SUM(prompt_tokens), 0),
                COALESCE(SUM(completion_tokens), 0),
                COALESCE(SUM(cost_usd), 0),
                COALESCE(AVG(latency_ms), 0),
                COALESCE(SUM(blocked), 0),
                COALESCE(SUM(escalated), 0)
            FROM usage
            """
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

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT timestamp, thread_id, prompt_tokens, completion_tokens,
                   cost_usd, latency_ms, blocked, escalated, retry_count
            FROM usage
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
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
            }
            for row in rows
        ]
