"""
LLM response caching.

Two strategies are supported:
  - InMemoryCache (default): exact-match caching, no external infrastructure.
  - RedisSemanticCache (optional): paraphrase-aware caching via embedding
    similarity, requires a running Redis Stack instance (with the
    RediSearch module — plain Redis is not sufficient).
"""

import logging

from langchain_core.caches import InMemoryCache
from langchain_core.embeddings import Embeddings
from langchain_core.globals import set_llm_cache

from config import settings

logger = logging.getLogger(__name__)


def configure_cache(embeddings: Embeddings | None = None) -> None:
    """
    Configure the global LangChain LLM response cache based on
    settings.USE_SEMANTIC_CACHE.

    `embeddings` is required only when semantic caching is enabled, since
    RedisSemanticCache embeds each incoming query to compare against
    previously cached queries.
    """
    if settings.USE_SEMANTIC_CACHE:
        if embeddings is None:
            raise ValueError("embeddings must be provided when USE_SEMANTIC_CACHE is enabled.")
        from langchain_community.cache import RedisSemanticCache

        set_llm_cache(
            RedisSemanticCache(
                redis_url=settings.REDIS_URL,
                embedding=embeddings,
                score_threshold=settings.REDIS_CACHE_SCORE_THRESHOLD,
            )
        )
        logger.info("LLM response cache: RedisSemanticCache (%s)", settings.REDIS_URL)
    else:
        set_llm_cache(InMemoryCache())
        logger.info("LLM response cache: InMemoryCache")
