"""
FAISS vector store construction and persistence.
"""

import logging
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import settings

logger = logging.getLogger(__name__)


def build_or_load_vectorstore(
    chunks: List[Document],
    embeddings: Embeddings,
    force_rebuild: bool = False,
) -> FAISS:
    """
    Load a persisted FAISS index if one exists at settings.FAISS_INDEX_DIR,
    otherwise build a new one from `chunks` and persist it.

    Set force_rebuild=True after changing the source document set or
    chunking parameters, since a stale index will silently reflect the old
    configuration otherwise.
    """
    index_path = str(settings.FAISS_INDEX_DIR)

    if not force_rebuild and settings.FAISS_INDEX_DIR.exists():
        logger.info("Loading existing FAISS index from %s", index_path)
        return FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )

    logger.info("Building new FAISS index from %d chunks", len(chunks))
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    logger.info("Persisted FAISS index to %s", index_path)
    return vectorstore
