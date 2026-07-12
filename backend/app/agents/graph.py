import json
import time
import re

from langgraph.graph import END, START, StateGraph

from app.agents.prompts import (
    build_compare_prompt,
    build_context_from_chunks,
    build_lit_review_prompt,
    build_qa_prompt,
)
from app.agents.state import AgentState, TaskType
from app.config import settings
from app.db.memory import list_memories
from app.db.response_cache import (
    get_cached_response,
    make_cache_key,
    save_cached_response,
)
from app.llm.client import generate_answer, stream_answer
from app.metrics.tracker import save_latest_metrics
from app.rag.citation_filter import filter_citations
from app.schemas import AskMetrics, AskResponse, Citation
from app.tools.paper_tools import retrieve_context, save_memory

REFUSAL_TEXT = "I do not know from the uploaded documents."

def _is_refusal(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized.startswith("i do not know") or "do not know from the" in normalized

def _clean_rewritten_query(raw: str, fallback: str) -> str:
    """Extract the actual query from a small model's chatty rewrite output."""
    text = " ".join(raw.split()).strip().strip('"')
    if not text:
        return fallback

    # Model wrapped the query in quotes -> take the longest quoted span.
    quoted = re.findall(r'"([^"]+)"', raw)
    if quoted:
        return max(quoted, key=len).strip()[:300]

    # Model prefixed commentary ending in a colon -> take what follows the last colon.
    if ":" in text:
        tail = text.rsplit(":", 1)[1].strip().strip('"')
        if len(tail) >= 3:
            return tail[:300]

    return text[:300]

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
        "retrieval_attempts": 0,
        "active_query": "",
        "memory_context": "",
    }


def load_memory_node(state: AgentState) -> AgentState:
    """
    Memory read path.

    Loads recent project memory from SQLite and formats it as a background
    block that synthesize_node prepends to the retrieved context.
    """
    try:
        memories = list_memories(
            user_id=state["user_id"],
            project_id=state["project_id"],
            limit=5,
        )
    except Exception:
        return {"memory_context": ""}

    if not memories:
        return {"memory_context": ""}

    # list_memories returns newest first (ORDER BY id DESC);
    # reverse so the block reads chronologically.
    lines = [f"- {memory.memory_item}" for memory in reversed(memories)]
    block = "Known project context from earlier sessions:\n" + "\n".join(lines)

    return {"memory_context": block[:400]}


def retrieve_node(state: AgentState) -> AgentState:
    """
    Tool-like retrieval node.

    Calls Chroma semantic search with strict user_id/project_id filtering.
    Uses active_query (set by rewrite_query_node on retries) when present.
    """
    attempts = state.get("retrieval_attempts", 0) + 1

    if state.get("task_type") == "unknown":
        return {
            "retrieved_chunks": [],
            "retrieval_latency_ms": 0,
            "retrieval_attempts": attempts,
            "warnings": state.get("warnings", []) + ["Empty or unknown query."],
        }

    query = state.get("active_query") or state["user_query"]

    results, retrieval_latency_ms = retrieve_context(
        user_id=state["user_id"],
        project_id=state["project_id"],
        query=query,
        top_k=state.get("top_k", settings.TOP_K),
    )

    return {
        "retrieved_chunks": results,
        "retrieval_latency_ms": retrieval_latency_ms,
        "retrieval_attempts": attempts,
    }


def route_after_retrieve(state: AgentState) -> str:
    """
    Conditional edge: decide whether retrieval was strong enough.

    Weak retrieval (no chunks, or best score below RETRIEVAL_MIN_SCORE)
    triggers one bounded query-rewrite retry; otherwise proceed to
    synthesis, where the empty-chunks branch handles final abstention.
    """
    if state.get("task_type") == "unknown":
        return "synthesize"

    chunks = state.get("retrieved_chunks", [])
    attempts = state.get("retrieval_attempts", 1)

    top_score = max((chunk.score or 0.0) for chunk in chunks) if chunks else 0.0
    weak = (not chunks) or (top_score < settings.RETRIEVAL_MIN_SCORE)

    if weak and attempts < settings.MAX_RETRIEVAL_ATTEMPTS:
        return "rewrite_query"

    return "synthesize"


def rewrite_query_node(state: AgentState) -> AgentState:
    """
    Rewrites the query once when retrieval confidence is low.
    """
    rewrite_prompt = (
        "Rewrite the following search query to fix any potential typo, maximize recall in a "
        "semantic search over academic PDF chunks. Expand abbreviations, "
        "add likely synonyms, keep it under 30 words. "
        "Return only the rewritten query.\n\n"
        f"Query: {state['user_query']}"
    )

    try:
        result = generate_answer(prompt=rewrite_prompt, context="")
        rewritten = (result.get("text") or "").strip().strip('"')
        rewritten = " ".join(rewritten.split())[:300]
    except Exception:
        rewritten = ""

    rewritten = _clean_rewritten_query(rewritten, state["user_query"])

    warnings = state.get("warnings", []) + [
        f"Low retrieval confidence; retried with rewritten query: {rewritten[:120]}"
    ]

    return {"active_query": rewritten, "warnings": warnings}


def synthesize_node(state: AgentState) -> AgentState:
    """
    Generation node.

    Builds context from retrieved chunks (plus project memory background)
    and calls the LLM client.
    """
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return {
            "draft_answer": REFUSAL_TEXT,
            "final_answer": REFUSAL_TEXT,
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

   # Memory is intentionally NOT injected into the QA grounding context.
    # Retrieved document chunks are the sole basis for factual answers;
    # mixing prior-session memory into the context primes small models to
    # refuse and consumes the limited context window. Memory is kept in
    # state for observability and future non-grounding uses only.
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
    Citation verification node.

    Parses inline [Source n] markers from the draft answer and keeps only
    the chunks the answer actually cited. Refusals carry no citations.
    """
    chunks = state.get("retrieved_chunks", [])
    draft_answer = state.get("draft_answer", "")
    warnings = state.get("warnings", [])

    if not chunks:
        return {
            "citations": [],
            "final_answer": REFUSAL_TEXT,
            "warnings": warnings,
        }

    final_answer = draft_answer or REFUSAL_TEXT

    if _is_refusal(final_answer):
        return {
            "citations": [],
            "final_answer": final_answer,
            "warnings": warnings,
        }

    citations, citation_warnings = filter_citations(final_answer, chunks)

    return {
        "citations": citations,
        "final_answer": final_answer,
        "warnings": warnings + citation_warnings,
    }


def memory_update_node(state: AgentState) -> AgentState:
    """
    Persistent memory node.

    Saves a compact Q/A record for non-refusal answers, so that
    load_memory_node has something useful to read back next session.
    """
    memory_updates: list[str] = []

    user_id = state["user_id"]
    project_id = state["project_id"]
    user_query = state.get("user_query", "").strip()
    final_answer = state.get("final_answer", "")

    refused = _is_refusal(final_answer)
    low_signal = "do not know" in final_answer.lower() or len(final_answer) < 40

    if user_query and final_answer and not refused and not low_signal:
        memory_updates.append(f"Topic asked: {user_query} (answered)")

    for item in memory_updates:
        save_memory(
            user_id=user_id,
            project_id=project_id,
            memory_item=item,
            memory_type="agent",
        )

    return {
        "memory_updates": memory_updates,
    }


def build_agent_graph():
    """
    Build and compile the LangGraph workflow.

    START -> planner -> load_memory -> retrieve
        -> (conditional) rewrite_query -> retrieve   [bounded retry]
        -> synthesize -> citation_check -> memory_update -> END
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("citation_check", citation_check_node)
    graph.add_node("memory_update", memory_update_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "load_memory")
    graph.add_edge("load_memory", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "rewrite_query": "rewrite_query",
            "synthesize": "synthesize",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("synthesize", "citation_check")
    graph.add_edge("citation_check", "memory_update")
    graph.add_edge("memory_update", END)

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

    cache_lookup_start = time.time()

    cache_key = make_cache_key(
        user_id=user_id,
        project_id=project_id,
        question=user_query,
        top_k=top_k or settings.TOP_K,
        llm_model=settings.LLM_MODEL,
        max_tokens=settings.LLM_MAX_TOKENS,
    )

    cached = get_cached_response(cache_key)
    cache_lookup_latency_ms = int((time.time() - cache_lookup_start) * 1000)

    if cached is not None:
        total_latency_ms = int((time.time() - total_start) * 1000)

        response = AskResponse(
            answer=cached["answer"],
            citations=[Citation(**citation) for citation in cached["citations"]],
            metrics=AskMetrics(
                retrieval_latency_ms=0,
                generation_latency_ms=0,
                total_latency_ms=total_latency_ms,
                llm_backend="cache",
                llm_model=settings.LLM_MODEL,
                mock=settings.MOCK_LLM,
                estimated_tokens_per_second=0.0,
                cache_hit=True,
                cache_lookup_latency_ms=cache_lookup_latency_ms,
            ),
        )

        save_latest_metrics(response.metrics.model_dump())
        return response

    final_state = agent_graph.invoke(initial_state)

    total_latency_ms = int((time.time() - total_start) * 1000)

    response = AskResponse(
        answer=final_state.get("final_answer", REFUSAL_TEXT),
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

    save_cached_response(
        cache_key=cache_key,
        user_id=user_id,
        project_id=project_id,
        question=user_query,
        answer=response.answer,
        citations=[citation.model_dump() for citation in response.citations],
        metrics=response.metrics.model_dump(),
    )

    save_latest_metrics(response.metrics.model_dump())
    return response


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
        "retrieval_attempts": final_state.get("retrieval_attempts", 0),
        "active_query": final_state.get("active_query", ""),
        "memory_context_chars": len(final_state.get("memory_context", "")),
        "citations": len(final_state.get("citations", [])),
        "memory_updates": final_state.get("memory_updates", []),
        "warnings": final_state.get("warnings", []),
        "answer": final_state.get("final_answer", ""),
    }


def run_agent_stream(
    user_id: str,
    project_id: str,
    user_query: str,
    top_k: int | None = None,
):
    """
    Streaming twin of run_agent.

    Reuses the same node functions (planner, memory, retrieval with the
    same weak-retrieval retry policy) and streams only the synthesis step.
    Yields newline-delimited JSON events:
        metadata -> token* -> citations_final -> done
    """
    state: AgentState = {
        "user_id": user_id,
        "project_id": project_id,
        "user_query": user_query,
        "top_k": top_k or settings.TOP_K,
    }

    state.update(planner_node(state))
    state.update(load_memory_node(state))
    state.update(retrieve_node(state))

    while route_after_retrieve(state) == "rewrite_query":
        state.update(rewrite_query_node(state))
        state.update(retrieve_node(state))

    chunks = state.get("retrieved_chunks", [])

    yield json.dumps(
        {
            "type": "metadata",
            "retrieval_latency_ms": state.get("retrieval_latency_ms", 0),
            "task_type": state.get("task_type", "qa"),
            "warnings": state.get("warnings", []),
            # Provisional provenance; replaced by citations_final below.
            "citations": [
                {
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text[:500],
                }
                for chunk in chunks
            ],
        }
    ) + "\n"

    if not chunks:
        yield json.dumps({"type": "token", "text": REFUSAL_TEXT}) + "\n"
        yield json.dumps(
            {"type": "citations_final", "citations": [], "warnings": []}
        ) + "\n"
        yield json.dumps({"type": "done"}) + "\n"
        return

    task_type = state.get("task_type", "qa")

    if task_type == "compare":
        prompt = build_compare_prompt(user_query)
    elif task_type == "lit_review":
        prompt = build_lit_review_prompt(user_query)
    else:
        prompt = build_qa_prompt(user_query)

    context = build_context_from_chunks(chunks)

    full_answer = ""
    for token in stream_answer(prompt=prompt, context=context):
        full_answer += token
        yield json.dumps({"type": "token", "text": token}) + "\n"

    if _is_refusal(full_answer):
        final_citations, citation_warnings = [], []
    else:
        final_citations, citation_warnings = filter_citations(full_answer, chunks)

    yield json.dumps(
        {
            "type": "citations_final",
            "citations": [citation.model_dump() for citation in final_citations],
            "warnings": citation_warnings,
        }
    ) + "\n"

    state["final_answer"] = full_answer
    memory_update_node(state)

    yield json.dumps({"type": "done"}) + "\n"