from typing import Any, Literal, TypedDict

from app.schemas import Citation, SearchResult


TaskType = Literal["qa", "compare", "lit_review", "unknown"]


class AgentState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.

    Each node receives this state and returns partial updates.
    """

    # Inputs
    user_id: str
    project_id: str
    user_query: str
    top_k: int

    # Planning
    task_type: TaskType

    # Retrieval
    retrieved_chunks: list[SearchResult]
    retrieval_latency_ms: int

    # Generation
    draft_answer: str
    final_answer: str

    # Citations
    citations: list[Citation]

    # Metrics
    generation_latency_ms: int
    total_latency_ms: int
    llm_backend: str
    llm_model: str
    mock: bool
    estimated_tokens_per_second: float

    # Diagnostics / future memory
    warnings: list[str]
    memory_updates: list[str]

    # Optional errors
    error: str
    raw_llm_result: dict[str, Any]