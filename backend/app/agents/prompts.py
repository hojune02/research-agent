from app.schemas import SearchResult


def build_context_from_chunks(chunks: list[SearchResult]) -> str:
    """
    Convert retrieved chunks into a context block for the LLM.
    """
    context_parts: list[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Source {idx}]\n"
            f"file: {chunk.source}\n"
            f"page: {chunk.page}\n"
            f"chunk_id: {chunk.chunk_id}\n"
            f"text:\n{chunk.text}"
        )

    return "\n\n".join(context_parts)


def build_qa_prompt(question: str) -> str:
    return f"""
You are PaperOps Agent, a careful research assistant.

Answer the user's question using only the provided context.

Rules:
- If the answer is not supported by the context, say: "I do not know from the uploaded documents."
- Do not use outside knowledge.
- Be concise but specific.
- Mention source filenames and pages when useful.
- Do not invent citations.

User question:
{question}
""".strip()


def build_compare_prompt(question: str) -> str:
    return f"""
You are PaperOps Agent.

Compare the uploaded papers using only the retrieved context.

Focus on:
- main problem
- method
- dataset or experiment
- strengths
- limitations
- open questions

If the context is insufficient, say so clearly.

User request:
{question}
""".strip()


def build_lit_review_prompt(topic: str) -> str:
    return f"""
You are PaperOps Agent.

Write a mini literature review using only the retrieved context.

Required structure:
1. Short overview
2. Main themes
3. Method comparison
4. Limitations
5. Open research directions

If the context is insufficient, say so clearly.

Topic:
{topic}
""".strip()