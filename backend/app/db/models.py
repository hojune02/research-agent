from app.db.session import get_connection


def init_db() -> None:
    """
    Initialize SQLite tables for Research Agent.
    """
    conn = get_connection()

    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                memory_item TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'project',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_user_project
            ON memories (user_id, project_id);
            """
        )

        conn.commit()

    finally:
        conn.close()