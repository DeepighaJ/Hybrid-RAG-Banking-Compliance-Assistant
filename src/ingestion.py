"""
Document ingestion: loading source PDFs, chunking, and metadata tagging.
"""

import logging
import os
import re
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger(__name__)

_QUESTION_NUMBER_PATTERN = re.compile(r"Q(\d{1,2})\.")


def load_documents(paths: List[str] | None = None) -> List[Document]:
    """
    Load one or more PDFs, returning a flat list of page-level Documents.

    Each page is tagged with `source_file` metadata (the PDF's basename),
    which downstream chunking uses to apply document-specific tagging rules.
    """
    paths = paths or settings.document_paths()
    pages: List[Document] = []

    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Source document not found: {path}. "
                f"Place PDFs in {settings.DATA_DIR} or update config/settings.py."
            )
        loader = PyPDFLoader(path)
        doc_pages = loader.load()
        source_file = os.path.basename(path)
        for page in doc_pages:
            page.metadata["source_file"] = source_file
        pages.extend(doc_pages)
        logger.info("Loaded %d pages from %s", len(doc_pages), source_file)

    return pages


def chunk_documents(pages: List[Document]) -> List[Document]:
    """
    Split loaded pages into chunks and tag each chunk with source-specific
    metadata (doc_type, and question_number where applicable).

    A recursive splitter is used because source documents contain long,
    multi-paragraph sections; it attempts natural boundaries (question
    markers, then paragraphs, then sentences) before falling back to a hard
    split, preserving semantic coherence as much as possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=settings.CHUNK_SEPARATORS,
    )
    chunks = splitter.split_documents(pages)
    return _tag_metadata(chunks)


def _tag_metadata(chunks: List[Document]) -> List[Document]:
    """
    Attach doc_type and question_number metadata to each chunk, based on
    the document configuration in config/settings.py.

    question_number is only parsed for documents flagged with
    has_question_numbers=True; it is carried forward across chunks belonging
    to the same question (since a single Q&A answer may span multiple
    chunks) and reset only when a new question boundary is encountered.
    """
    current_question_number = None
    current_source = None

    for chunk in chunks:
        source_file = chunk.metadata.get("source_file", "")
        doc_info = settings.DOCUMENT_CONFIG.get(
            source_file, {"doc_type": "unknown", "has_question_numbers": False}
        )

        if source_file != current_source:
            current_question_number = None
            current_source = source_file

        chunk.metadata["doc_type"] = doc_info["doc_type"]

        if doc_info["has_question_numbers"]:
            match = _QUESTION_NUMBER_PATTERN.search(chunk.page_content)
            if match:
                current_question_number = int(match.group(1))
            chunk.metadata["question_number"] = current_question_number
        else:
            chunk.metadata["question_number"] = None

    return chunks


def load_and_chunk() -> List[Document]:
    """Convenience wrapper: load all configured documents and chunk them."""
    pages = load_documents()
    chunks = chunk_documents(pages)
    logger.info("Produced %d chunks from %d pages", len(chunks), len(pages))
    return chunks
