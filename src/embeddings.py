"""
Embedding client configuration, wrapped with a disk-backed cache to avoid
redundant Azure OpenAI API calls across runs.
"""

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import AzureOpenAIEmbeddings

from config import settings


def get_cached_embeddings() -> CacheBackedEmbeddings:
    """
    Build an Azure OpenAI embeddings client wrapped with a local disk cache.

    Chunk text is hashed to form the cache key; re-embedding the same content
    (across notebook/script runs, or when only a subset of documents changes)
    is served from disk instead of calling the Azure API again.
    """
    base_embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        deployment=settings.EMBEDDING_DEPLOYMENT_NAME,
    )

    store = LocalFileStore(str(settings.EMBEDDING_CACHE_DIR))

    return CacheBackedEmbeddings.from_bytes_store(
        base_embeddings,
        store,
        namespace=settings.EMBEDDING_DEPLOYMENT_NAME,
    )
