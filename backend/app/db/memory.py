from app.db.session import get_connection
from app.schemas import MemoryItem


def _row_to_memory(row) -> MemoryItem:
    return MemoryItem(
        id=int(row["id"]),
        user_id=str(row["user_id"]),
        project_id=str(row["project_id"]),
        memory_item=str(row["memory_item"]),
        memory_type=str(row["memory_type"]),
        created_at=str(row["created_at"]),
    )


def create_memory(
    user_id: str,
    project_id: str,
    memory_item: str,
    memory_type: str = "project",
) -> MemoryItem:
    """
    Save a memory item to SQLite.
    """
    conn = get_connection()

    try:
        cursor = conn.execute(
            """
            INSERT INTO memories (user_id, project_id, memory_item, memory_type)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, project_id, memory_item, memory_type),
        )

        conn.commit()
        memory_id = cursor.lastrowid

        row = conn.execute(
            """
            SELECT id, user_id, project_id, memory_item, memory_type, created_at
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        ).fetchone()

        return _row_to_memory(row)

    finally:
        conn.close()


def list_memories(
    user_id: str,
    project_id: str,
    limit: int = 50,
) -> list[MemoryItem]:
    """
    Load recent memories for a user/project.
    """
    conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT id, user_id, project_id, memory_item, memory_type, created_at
            FROM memories
            WHERE user_id = ? AND project_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, project_id, limit),
        ).fetchall()

        return [_row_to_memory(row) for row in rows]

    finally:
        conn.close()


def delete_memory(memory_id: int) -> bool:
    """
    Delete a memory item by ID.
    """
    conn = get_connection()

    try:
        cursor = conn.execute(
            """
            DELETE FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        )

        conn.commit()
        return cursor.rowcount > 0

    finally:
        conn.close()


def memory_exists(
    user_id: str,
    project_id: str,
    memory_item: str,
) -> bool:
    """
    Simple duplicate-prevention helper.
    """
    conn = get_connection()

    try:
        row = conn.execute(
            """
            SELECT id
            FROM memories
            WHERE user_id = ? AND project_id = ? AND memory_item = ?
            LIMIT 1
            """,
            (user_id, project_id, memory_item),
        ).fetchone()

        return row is not None

    finally:
        conn.close()


def create_memory_if_new(
    user_id: str,
    project_id: str,
    memory_item: str,
    memory_type: str = "project",
) -> MemoryItem | None:
    """
    Save memory only if the exact same item does not already exist.
    """
    if memory_exists(user_id, project_id, memory_item):
        return None

    return create_memory(
        user_id=user_id,
        project_id=project_id,
        memory_item=memory_item,
        memory_type=memory_type,
    )