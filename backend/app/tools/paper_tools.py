import time

from app.agents.prompts import (
    build_compare_prompt,
    build_context_from_chunks,
    build_lit_review_prompt,
)
from app.config import settings
from app.llm.client import generate_answer
from app.rag.vectorstore import search_chunks
from app.schemas import AskMetrics, AskResponse, Citation, SearchResult


def retrieve_context(
    user_id: str,
    project_id: str,
    query: str,
    top_k: int | None = None,
) -> tuple[list[SearchResult], int]:
    """
    Tool: retrieve_context

    Performs user/project-isolated semantic search over Chroma.
    This is the main retrieval tool used by the research automation tools.
    """
    return search_chunks(
        user_id=user_id,
        project_id=project_id,
        query=query,
        top_k=top_k or settings.TOP_K,
    )


def make_citations(results: list[SearchResult]) -> list[Citation]:
    citations: list[Citation] = []

    for result in results:
        citations.append(
            Citation(
                source=result.source,
                page=result.page,
                chunk_id=result.chunk_id,
                text=result.text[:500],
            )
        )

    return citations


def _no_context_response(
    retrieval_latency_ms: int,
    total_start: float,
) -> AskResponse:
    total_latency_ms = int((time.time() - total_start) * 1000)

    return AskResponse(
        answer="I do not know from the uploaded documents.",
        citations=[],
        metrics=AskMetrics(
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=0,
            total_latency_ms=total_latency_ms,
            llm_backend="none",
            llm_model="none",
            mock=settings.MOCK_LLM,
            estimated_tokens_per_second=0.0,
        ),
    )


def _grounded_generate(
    *,
    user_id: str,
    project_id: str,
    retrieval_query: str,
    prompt: str,
    top_k: int | None = None,
) -> AskResponse:
    """
    Shared helper for research tools.

    Flow:
        retrieve context
        build context
        call LLM
        return answer + citations + metrics
    """
    total_start = time.time()

    results, retrieval_latency_ms = retrieve_context(
        user_id=user_id,
        project_id=project_id,
        query=retrieval_query,
        top_k=top_k,
    )

    if not results:
        return _no_context_response(
            retrieval_latency_ms=retrieval_latency_ms,
            total_start=total_start,
        )

    context = build_context_from_chunks(results)

    llm_result = generate_answer(
        prompt=prompt,
        context=context,
    )

    total_latency_ms = int((time.time() - total_start) * 1000)

    return AskResponse(
        answer=llm_result["text"],
        citations=make_citations(results),
        metrics=AskMetrics(
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=llm_result["generation_latency_ms"],
            total_latency_ms=total_latency_ms,
            llm_backend=llm_result["backend"],
            llm_model=llm_result["model"],
            mock=llm_result["mock"],
            estimated_tokens_per_second=llm_result["estimated_tokens_per_second"],
        ),
    )


def compare_documents(
    user_id: str,
    project_id: str,
    question: str,
    top_k: int | None = None,
) -> AskResponse:
    """
    Tool: compare_documents

    Compares uploaded papers based on retrieved context.
    """
    prompt = build_compare_prompt(question)

    return _grounded_generate(
        user_id=user_id,
        project_id=project_id,
        retrieval_query=question,
        prompt=prompt,
        top_k=top_k,
    )


def generate_literature_review(
    user_id: str,
    project_id: str,
    topic: str,
    top_k: int | None = None,
) -> AskResponse:
    """
    Tool: generate_literature_review

    Generates a mini literature review from uploaded papers.
    """
    prompt = build_lit_review_prompt(topic)

    retrieval_query = (
        f"{topic}. Main themes, methods, limitations, datasets, experiments, "
        f"future work, related work."
    )

    return _grounded_generate(
        user_id=user_id,
        project_id=project_id,
        retrieval_query=retrieval_query,
        prompt=prompt,
        top_k=top_k,
    )


def build_extract_insights_prompt(focus: str) -> str:
    return f"""
You are PaperOps Agent, a research automation assistant.

Extract structured research insights from the uploaded papers using only the provided context.

Focus on:
{focus}

Return the answer in this structure:

1. Methods
2. Datasets / experimental setup
3. Key findings
4. Limitations
5. Future research directions
6. Useful keywords

Rules:
- Use only the provided context.
- If a field is not supported by the context, say "Not found in retrieved context."
- Mention source filenames and pages when useful.
- Do not invent details.
""".strip()


def extract_methods_and_limitations(
    user_id: str,
    project_id: str,
    focus: str,
    top_k: int | None = None,
) -> AskResponse:
    """
    Tool: extract_methods_and_limitations

    Extracts method, dataset, limitation, and future-work information.
    """
    prompt = build_extract_insights_prompt(focus)

    retrieval_query = (
        f"{focus}. method dataset experiment result limitation future work "
        f"ablation evaluation benchmark."
    )

    return _grounded_generate(
        user_id=user_id,
        project_id=project_id,
        retrieval_query=retrieval_query,
        prompt=prompt,
        top_k=top_k,
    )


# def save_memory(
#     user_id: str,
#     project_id: str,
#     memory_item: str,
# ) -> None:
#     """
#     Tool: save_memory

#     Phase 8 stub.
#     Real SQLite-backed memory is implemented in Phase 9.
#     """
#     print(
#         f"[MEMORY_STUB] user_id={user_id}, "
#         f"project_id={project_id}, memory_item={memory_item}"
#     )

from app.db.memory import create_memory_if_new

def save_memory(
    user_id: str,
    project_id: str,
    memory_item: str,
    memory_type: str = "project",
) -> None:
    """
    Tool: save_memory

    Saves useful user/project memory to SQLite.
    """
    create_memory_if_new(
        user_id=user_id,
        project_id=project_id,
        memory_item=memory_item,
        memory_type=memory_type,
    )