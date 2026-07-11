"""
Cross-encoder reranking of retrieved candidates.
"""

import logging
from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_reranker_model() -> CrossEncoder:
    """
    Load the cross-encoder reranker model, cached for the process lifetime
    so it is only loaded once regardless of how many queries are served.
    """
    logger.info("Loading reranker model: %s", settings.RERANKER_MODEL_NAME)
    return CrossEncoder(settings.RERANKER_MODEL_NAME)


def rerank(query: str, candidates: List[Document], top_n: int | None = None) -> List[Document]:
    """
    Re-score retrieved candidates against the query using a cross-encoder
    and return the top_n most relevant, most-relevant-first.

    Reranking jointly scores (query, chunk) pairs, which is more accurate
    than comparing independently-computed embeddings, but has a higher
    per-pair cost — it is applied only to the retrieval shortlist, not the
    full corpus.
    """
    top_n = top_n or settings.RERANK_TOP_N
    if not candidates:
        return []

    model = _get_reranker_model()
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = model.predict(pairs)

    scored = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
    return [doc for doc, _ in scored[:top_n]]
