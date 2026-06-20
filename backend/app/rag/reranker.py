from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings
from app.schemas import SearchResult


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.RERANKER_MODEL)


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
        result.score = round(float(score), 4)
        reranked.append(result)

    return reranked