"""
Retriever construction: sparse (BM25), dense (FAISS + MMR), and their fusion
into a single hybrid retriever. Also provides helpers for metadata-scoped
retrieval.
"""

from typing import Callable, List, Optional

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import settings


def build_bm25_retriever(chunks: List[Document]) -> BM25Retriever:
    """Build a BM25 (sparse/lexical) retriever over the given chunks."""
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = settings.BM25_TOP_K
    return retriever


def build_faiss_retriever(vectorstore: FAISS) -> BaseRetriever:
    """
    Build a dense retriever using Maximal Marginal Relevance (MMR) to reduce
    redundant/near-duplicate results in the retrieved set.
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.FAISS_TOP_K,
            "fetch_k": settings.FAISS_FETCH_K,
            "lambda_mult": settings.FAISS_MMR_LAMBDA,
        },
    )


def build_hybrid_retriever(chunks: List[Document], vectorstore: FAISS) -> EnsembleRetriever:
    """
    Fuse BM25 (sparse) and FAISS/MMR (dense) retrievers using Reciprocal
    Rank Fusion, weighted per settings.ENSEMBLE_WEIGHTS.
    """
    bm25_retriever = build_bm25_retriever(chunks)
    faiss_retriever = build_faiss_retriever(vectorstore)

    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=settings.ENSEMBLE_WEIGHTS,
    )


def build_filtered_retriever(
    vectorstore: FAISS,
    metadata_filter: dict | Callable[[dict], bool],
    k: Optional[int] = None,
) -> BaseRetriever:
    """
    Build a dense retriever scoped to chunks matching `metadata_filter`.

    `metadata_filter` may be a dict of exact-match key/value pairs (e.g.
    {"doc_type": "rba_guidance"}), or a callable taking a chunk's metadata
    dict and returning a bool (for more complex conditions, such as a range
    of question numbers).
    """
    return vectorstore.as_retriever(
        search_kwargs={
            "k": k or settings.FAISS_TOP_K,
            "filter": metadata_filter,
        }
    )
