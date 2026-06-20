import hashlib
import json
from typing import Any

from app.db.session import get_connection


PROMPT_VERSION = "paperops-rag-v1"


def make_cache_key(
    *,
    user_id: str,
    project_id: str,
    question: str,
    top_k: int,
    llm_model: str,
    max_tokens: int,
) -> str:
    """
    Creates a stable cache key for identical RAG requests.

    For a stronger version later, include document index version or file hashes.
    """
    normalized_question = " ".join(question.lower().strip().split())

    payload = {
        "user_id": user_id,
        "project_id": project_id,
        "question": normalized_question,
        "top_k": top_k,
        "llm_model": llm_model,
        "max_tokens": max_tokens,
        "prompt_version": PROMPT_VERSION,
    }

    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_response(cache_key: str) -> dict[str, Any] | None:
    conn = get_connection()

    try:
        row = conn.execute(
            """
            SELECT answer, citations_json, metrics_json
            FROM response_cache
            WHERE cache_key = ?
            LIMIT 1
            """,
            (cache_key,),
        ).fetchone()

        if row is None:
            return None

        return {
            "answer": row["answer"],
            "citations": json.loads(row["citations_json"]),
            "metrics": json.loads(row["metrics_json"]),
        }

    finally:
        conn.close()


def save_cached_response(
    *,
    cache_key: str,
    user_id: str,
    project_id: str,
    question: str,
    answer: str,
    citations: list[dict],
    metrics: dict,
) -> None:
    conn = get_connection()

    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO response_cache
            (
                cache_key,
                user_id,
                project_id,
                question,
                answer,
                citations_json,
                metrics_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                user_id,
                project_id,
                question,
                answer,
                json.dumps(citations),
                json.dumps(metrics),
            ),
        )

        conn.commit()

    finally:
        conn.close()