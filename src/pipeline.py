"""
End-to-end RAG pipeline: retrieval (hybrid) -> reranking -> generation.

This module composes the individual stages (ingestion, embeddings,
vectorstore, retrieval, reranking, prompting, caching) into a single
reusable pipeline object.
"""

import logging
from typing import Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import AzureChatOpenAI

from config import settings
from src import cache, embeddings as embeddings_module, ingestion, reranker, retrieval, vectorstore
from src.prompts import QA_PROMPT

logger = logging.getLogger(__name__)


class RAGResult(TypedDict):
    answer: str
    source_documents: list[Document]


class RAGPipeline:
    """
    A hybrid retrieval-augmented generation pipeline over the configured
    document set (see config/settings.py).

    Usage:
        pipeline = RAGPipeline()
        pipeline.build()  # or pipeline.load() if an index already exists
        result = pipeline.ask("How often should country risk ratings be reviewed?")
    """

    def __init__(self) -> None:
        settings.validate()
        self.embeddings = embeddings_module.get_cached_embeddings()
        self.chunks: list[Document] = []
        self.vectorstore = None
        self.hybrid_retriever: Optional[BaseRetriever] = None
        self.llm: Optional[AzureChatOpenAI] = None

    def build(self, force_rebuild: bool = False) -> "RAGPipeline":
        """Load documents, chunk, embed, index, and initialize the LLM."""
        pages = ingestion.load_documents()
        self.chunks = ingestion.chunk_documents(pages)

        self.vectorstore = vectorstore.build_or_load_vectorstore(
            self.chunks, self.embeddings, force_rebuild=force_rebuild
        )
        self.hybrid_retriever = retrieval.build_hybrid_retriever(self.chunks, self.vectorstore)

        cache.configure_cache(self.embeddings if settings.USE_SEMANTIC_CACHE else None)

        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            deployment_name=settings.CHAT_DEPLOYMENT_NAME,
            # Note: gpt-5-mini is a reasoning-family model and does not
            # support a configurable `temperature` parameter.
        )

        logger.info("Pipeline ready (%d chunks indexed).", len(self.chunks))
        return self

    def ask(
        self,
        query: str,
        top_n: Optional[int] = None,
        retriever: Optional[BaseRetriever] = None,
    ) -> RAGResult:
        """
        Run the full pipeline for a single query: hybrid retrieval -> rerank
        -> grounded generation.

        Pass a custom `retriever` (e.g. from retrieval.build_filtered_retriever)
        to scope this query to a subset of the indexed documents.
        """
        if self.hybrid_retriever is None or self.llm is None:
            raise RuntimeError("Pipeline not built. Call .build() or .load() first.")

        active_retriever = retriever or self.hybrid_retriever

        candidates = active_retriever.invoke(query)
        top_docs = reranker.rerank(query, candidates, top_n=top_n)

        context = "\n\n".join(doc.page_content for doc in top_docs)
        formatted_prompt = QA_PROMPT.format(context=context, question=query)
        response = self.llm.invoke(formatted_prompt)

        return RAGResult(answer=response.content, source_documents=top_docs)
