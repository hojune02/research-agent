import time

from app.config import settings
from app.llm.client import generate_answer
from app.rag.vectorstore import search_chunks
from app.schemas import AskMetrics, AskResponse, Citation, SearchResult

from collections.abc import Generator
import json

from app.llm.client import stream_answer
from app.rag.vectorstore import search_chunks


def build_context(results: list[SearchResult]) -> str:
    """
    Build context string passed to the LLM.

    Each chunk is labeled with citation info so the model can cite sources.
    """
    context_parts: list[str] = []

    for idx, result in enumerate(results, start=1):
        context_parts.append(
            f"[Source {idx}] "
            f"file={result.source}, page={result.page}, chunk_id={result.chunk_id}\n"
            f"{result.text}"
        )

    return "\n\n".join(context_parts)


def build_rag_prompt(question: str) -> str:
    """
    Prompt used for cited RAG QA.
    """
    return f"""
Answer the user's question using only the provided context.

Rules:
- If the answer is not supported by the context, say: "I do not know from the uploaded documents."
- Be concise but specific.
- Mention source filenames and pages when useful.
- Do not invent citations.
- Do not use outside knowledge.

Question:
{question}
""".strip()


def make_citations(results: list[SearchResult]) -> list[Citation]:
    """
    Convert retrieved chunks into citation objects.
    """
    citations: list[Citation] = []

    for result in results:
        # Keep citation excerpt short enough for UI readability.
        excerpt = result.text[:500]

        citations.append(
            Citation(
                source=result.source,
                page=result.page,
                chunk_id=result.chunk_id,
                text=excerpt,
            )
        )

    return citations


def answer_question(
    user_id: str,
    project_id: str,
    question: str,
    top_k: int | None = None,
) -> AskResponse:
    """
    Full RAG QA pipeline:
        search Chroma
        build context
        call LLM
        return answer + citations + metrics
    """
    total_start = time.time()

    k = top_k or settings.TOP_K

    results, retrieval_latency_ms = search_chunks(
        user_id=user_id,
        project_id=project_id,
        query=question,
        top_k=k,
    )

    if not results:
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

    context = build_context(results)
    prompt = build_rag_prompt(question)

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

def stream_answer_question(
    user_id: str,
    project_id: str,
    question: str,
    top_k: int | None = None,
) -> Generator[str, None, None]:
    """
    Streams a RAG answer as Server-Sent Events-style JSON lines.

    Each yielded line is a JSON object followed by newline.
    """
    k = top_k or settings.TOP_K

    results, retrieval_latency_ms = search_chunks(
        user_id=user_id,
        project_id=project_id,
        query=question,
        top_k=k,
    )

    citations = [
        {
            "source": result.source,
            "page": result.page,
            "chunk_id": result.chunk_id,
            "text": result.text[:500],
        }
        for result in results
    ]

    # First event: metadata and citations
    yield json.dumps(
        {
            "type": "metadata",
            "retrieval_latency_ms": retrieval_latency_ms,
            "citations": citations,
        }
    ) + "\n"

    if not results:
        yield json.dumps(
            {
                "type": "token",
                "text": "I do not know from the uploaded documents.",
            }
        ) + "\n"
        yield json.dumps({"type": "done"}) + "\n"
        return

    context = build_context(results)
    prompt = build_rag_prompt(question)

    for token in stream_answer(prompt=prompt, context=context):
        yield json.dumps(
            {
                "type": "token",
                "text": token,
            }
        ) + "\n"

    yield json.dumps({"type": "done"}) + "\n"