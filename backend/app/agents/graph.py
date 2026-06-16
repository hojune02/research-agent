import time

from langgraph.graph import END, START, StateGraph

from app.agents.prompts import (
    build_compare_prompt,
    build_context_from_chunks,
    build_lit_review_prompt,
    build_qa_prompt,
)
from app.agents.state import AgentState, TaskType
from app.config import settings
from app.llm.client import generate_answer
# Phase 7: from app.rag.vectorstore import search_chunks
from app.tools.paper_tools import retrieve_context
from app.schemas import AskMetrics, AskResponse, Citation


def planner_node(state: AgentState) -> AgentState:
    """
    Simple rule-based planner.

    In a larger version, this could be LLM-based routing.
    For MVP, deterministic routing is better and easier to debug.
    """
    query = state["user_query"].lower()

    task_type: TaskType = "qa"

    if any(word in query for word in ["compare", "comparison", "versus", "vs"]):
        task_type = "compare"
    elif any(
        phrase in query
        for phrase in [
            "literature review",
            "lit review",
            "survey",
            "related work",
            "research directions",
        ]
    ):
        task_type = "lit_review"
    elif not query.strip():
        task_type = "unknown"

    return {
        "task_type": task_type,
        "warnings": [],
        "memory_updates": [],
    }


def retrieve_node(state: AgentState) -> AgentState:
    """
    Tool-like retrieval node.

    Calls Chroma semantic search with strict user_id/project_id filtering.
    """
    if state.get("task_type") == "unknown":
        return {
            "retrieved_chunks": [],
            "retrieval_latency_ms": 0,
            "warnings": state.get("warnings", []) + ["Empty or unknown query."],
        }

    results, retrieval_latency_ms = retrieve_context(
    user_id=state["user_id"],
    project_id=state["project_id"],
    query=state["user_query"],
    top_k=state.get("top_k", settings.TOP_K),
)

    return {
        "retrieved_chunks": results,
        "retrieval_latency_ms": retrieval_latency_ms,
    }


def synthesize_node(state: AgentState) -> AgentState:
    """
    Generation node.

    Builds context from retrieved chunks and calls the LLM client.
    """
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return {
            "draft_answer": "I do not know from the uploaded documents.",
            "final_answer": "I do not know from the uploaded documents.",
            "generation_latency_ms": 0,
            "llm_backend": "none",
            "llm_model": "none",
            "mock": settings.MOCK_LLM,
            "estimated_tokens_per_second": 0.0,
        }

    task_type = state.get("task_type", "qa")

    if task_type == "compare":
        prompt = build_compare_prompt(state["user_query"])
    elif task_type == "lit_review":
        prompt = build_lit_review_prompt(state["user_query"])
    else:
        prompt = build_qa_prompt(state["user_query"])

    context = build_context_from_chunks(chunks)

    llm_result = generate_answer(
        prompt=prompt,
        context=context,
    )

    return {
        "draft_answer": llm_result["text"],
        "raw_llm_result": llm_result,
        "generation_latency_ms": llm_result["generation_latency_ms"],
        "llm_backend": llm_result["backend"],
        "llm_model": llm_result["model"],
        "mock": llm_result["mock"],
        "estimated_tokens_per_second": llm_result["estimated_tokens_per_second"],
    }


def citation_check_node(state: AgentState) -> AgentState:
    """
    Citation checker node.

    MVP behavior:
    - Convert retrieved chunks into Citation objects.
    - If no retrieved chunks, no citations.
    - Ensure final answer exists.
    """
    chunks = state.get("retrieved_chunks", [])

    citations: list[Citation] = []

    for chunk in chunks:
        citations.append(
            Citation(
                source=chunk.source,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                text=chunk.text[:500],
            )
        )

    draft_answer = state.get("draft_answer", "")

    warnings = state.get("warnings", [])

    if chunks and not citations:
        warnings.append("Retrieved chunks existed, but no citations were created.")

    if not chunks:
        final_answer = "I do not know from the uploaded documents."
    else:
        final_answer = draft_answer or "I do not know from the uploaded documents."

    return {
        "citations": citations,
        "final_answer": final_answer,
        "warnings": warnings,
    }


def memory_stub_node(state: AgentState) -> AgentState:
    """
    Placeholder memory node.

    Real SQLite memory comes in Phase 9.
    For now, we generate memory update candidates.
    """
    memory_updates: list[str] = []

    task_type = state.get("task_type", "unknown")

    if task_type != "unknown":
        memory_updates.append(
            f"User asked a {task_type} question in project '{state['project_id']}'."
        )

    if state.get("retrieved_chunks"):
        sources = sorted({chunk.source for chunk in state["retrieved_chunks"]})
        memory_updates.append(
            f"Retrieved context from sources: {', '.join(sources)}."
        )

    return {
        "memory_updates": memory_updates,
    }


def build_agent_graph():
    """
    Build and compile the LangGraph workflow.
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("citation_check", citation_check_node)
    graph.add_node("memory_stub", memory_stub_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "citation_check")
    graph.add_edge("citation_check", "memory_stub")
    graph.add_edge("memory_stub", END)

    return graph.compile()


agent_graph = build_agent_graph()


def run_agent(
    user_id: str,
    project_id: str,
    user_query: str,
    top_k: int | None = None,
) -> AskResponse:
    """
    Public function used by FastAPI endpoints.

    Runs the full LangGraph workflow and converts final state into AskResponse.
    """
    total_start = time.time()

    initial_state: AgentState = {
        "user_id": user_id,
        "project_id": project_id,
        "user_query": user_query,
        "top_k": top_k or settings.TOP_K,
    }

    final_state = agent_graph.invoke(initial_state)

    total_latency_ms = int((time.time() - total_start) * 1000)

    return AskResponse(
        answer=final_state.get(
            "final_answer",
            "I do not know from the uploaded documents.",
        ),
        citations=final_state.get("citations", []),
        metrics=AskMetrics(
            retrieval_latency_ms=final_state.get("retrieval_latency_ms", 0),
            generation_latency_ms=final_state.get("generation_latency_ms", 0),
            total_latency_ms=total_latency_ms,
            llm_backend=final_state.get("llm_backend", "unknown"),
            llm_model=final_state.get("llm_model", "unknown"),
            mock=final_state.get("mock", settings.MOCK_LLM),
            estimated_tokens_per_second=final_state.get(
                "estimated_tokens_per_second",
                0.0,
            ),
        ),
    )

def run_agent_debug(
    user_id: str,
    project_id: str,
    user_query: str,
    top_k: int | None = None,
) -> dict:
    initial_state: AgentState = {
        "user_id": user_id,
        "project_id": project_id,
        "user_query": user_query,
        "top_k": top_k or settings.TOP_K,
    }

    final_state = agent_graph.invoke(initial_state)

    return {
        "task_type": final_state.get("task_type", "unknown"),
        "retrieved_chunks": len(final_state.get("retrieved_chunks", [])),
        "citations": len(final_state.get("citations", [])),
        "memory_updates": final_state.get("memory_updates", []),
        "warnings": final_state.get("warnings", []),
        "answer": final_state.get("final_answer", ""),
    }