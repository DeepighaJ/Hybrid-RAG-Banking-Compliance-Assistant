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

# Zero-width split point at the start of any line beginning "Q<n>. "
_QUESTION_BOUNDARY = re.compile(r"(?m)^(?=Q\d{1,2}\.\s)")

# Section headings in Q&A documents, e.g. "Section B: Customer Identification"
_SECTION_LINE = re.compile(r"(?m)^Section [A-Z]: .+$")


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

    qa_pages, other_pages = [], []
    for page in pages:
        doc_info = settings.DOCUMENT_CONFIG.get(page.metadata.get("source_file", ""), {})
        (qa_pages if doc_info.get("has_question_numbers") else other_pages).append(page)

    chunks = _split_qa_pages(qa_pages, splitter) + splitter.split_documents(other_pages)
    return _tag_metadata(chunks)


def _split_qa_pages(pages: List[Document], splitter) -> List[Document]:
    """
    Split Q&A documents so that each chunk holds exactly one question and its answer.

    The generic recursive splitter packs several short Q&A pairs into one
    chunk, which (a) lets near-duplicate questions be retrieved together,
    weakening precision testing, and (b) means question_number only records
    the first question in the chunk. Splitting on question boundaries fixes
    both. Answers longer than CHUNK_SIZE still fall back to the recursive
    splitter; continuation chunks inherit question_number via _tag_metadata.

    Text before the first question on a page (running headers, page numbers,
    cover pages) is discarded. Section headings are removed from chunk text
    and recorded as `section` metadata for the questions that follow them.
    Assumes a question never spans a page break.
    """
    chunks: List[Document] = []
    section = None
    current_source = None

    for page in pages:
        source = page.metadata.get("source_file")
        if source != current_source:
            section, current_source = None, source

        for part in _QUESTION_BOUNDARY.split(page.page_content):
            headings = _SECTION_LINE.findall(part)
            body = _SECTION_LINE.sub("", part).strip()

            if _QUESTION_NUMBER_PATTERN.match(body):
                doc = Document(page_content=body, metadata={**page.metadata, "section": section})
                if len(body) > settings.CHUNK_SIZE:
                    chunks.extend(splitter.split_documents([doc]))
                else:
                    chunks.append(doc)

            # A heading at the end of this part labels the questions after it
            if headings:
                section = headings[-1].strip()

    return chunks


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