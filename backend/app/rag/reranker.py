import math
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings
from app.schemas import SearchResult


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.RERANKER_MODEL)

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rerank_results(query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
    if not results:
        return results

    model = get_reranker()

    pairs = [(query, result.text) for result in results]
    scores = model.predict(pairs)

    scored = list(zip(results, scores))
    scored.sort(key=lambda item: float(item[1]), reverse=True)

    reranked = []

    for result, score in scored[:top_k]:
         # Normalize raw cross-encoder logits to (0, 1) so that scores are
        # comparable with the distance-derived similarity used when the
        # reranker is disabled. Sorting order is preserved (sigmoid is monotonic).
        result.score = round(_sigmoid(float(score)), 4)
        reranked.append(result)

    return reranked