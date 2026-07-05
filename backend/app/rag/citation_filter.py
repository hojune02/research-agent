"""
Post-generation citation verification.

The prompt instructs the model to cite sources inline as [Source n],
matching the labels produced by build_context_from_chunks. This module
parses those markers from the finished answer and keeps only the
retrieved chunks that were actually cited, so citations mean
"evidence the answer used", not "everything retrieval returned".
"""

import re

from app.schemas import Citation, SearchResult

_MARKER_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


def extract_cited_indices(answer: str, num_sources: int) -> list[int]:
    """Return sorted unique 1-based [Source n] indices found in the answer,
    keeping only indices that map to an actually retrieved chunk."""
    indices: set[int] = set()
    for match in _MARKER_RE.finditer(answer):
        idx = int(match.group(1))
        if 1 <= idx <= num_sources:
            indices.add(idx)
    return sorted(indices)


def filter_citations(
    answer: str,
    results: list[SearchResult],
) -> tuple[list[Citation], list[str]]:
    """
    Build citations from only the chunks the answer cited.

    Returns (citations, warnings).

    Fallback: if the model produced an answer but no valid inline markers,
    all retrieved chunks are returned as provenance with a warning, so the
    UI never silently loses traceability.
    """
    warnings: list[str] = []

    if not results:
        return [], warnings

    cited = extract_cited_indices(answer, len(results))

    if not cited:
        warnings.append(
            "Answer contained no valid inline [Source n] markers; "
            "falling back to all retrieved chunks as provenance."
        )
        selected = results
    else:
        selected = [results[i - 1] for i in cited]

    citations = [
        Citation(
            source=result.source,
            page=result.page,
            chunk_id=result.chunk_id,
            text=result.text[:500],
        )
        for result in selected
    ]
    return citations, warnings