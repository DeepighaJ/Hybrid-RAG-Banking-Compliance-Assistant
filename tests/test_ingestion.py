"""
Tests for document ingestion: metadata tagging and Q&A-aware chunking.

Run with: pytest tests/
"""

import os

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from src.ingestion import _split_qa_pages, _tag_metadata, load_and_chunk

FAQ_FILE = "Eastmere Bank - KYC and CDD Procedures FAQ.pdf"
POLICY_FILE = "Eastmere Bank - AML Risk Assessment Policy.pdf"


# ---------------------------------------------------------------------------
# _tag_metadata
# ---------------------------------------------------------------------------


def test_tags_doc_type_for_known_source():
    chunks = [Document(page_content="Q1. What is the purpose?", metadata={"source_file": FAQ_FILE})]
    tagged = _tag_metadata(chunks)
    assert tagged[0].metadata["doc_type"] == "kyc_faq"


def test_parses_question_number_for_faq_document():
    chunks = [Document(page_content="Q3. How are customers rated?", metadata={"source_file": FAQ_FILE})]
    tagged = _tag_metadata(chunks)
    assert tagged[0].metadata["question_number"] == 3


def test_question_number_is_none_for_non_qa_document():
    chunks = [
        Document(
            page_content="RA-3.5 Risk Tier Thresholds. The weighted risk score places...",
            metadata={"source_file": POLICY_FILE},
        )
    ]
    tagged = _tag_metadata(chunks)
    assert tagged[0].metadata["question_number"] is None
    assert tagged[0].metadata["doc_type"] == "aml_risk_policy"


def test_question_number_carries_forward_across_chunks_of_same_question():
    chunks = [
        Document(page_content="Q5. What information must be collected?", metadata={"source_file": FAQ_FILE}),
        Document(page_content="...continued answer with no new Q marker...", metadata={"source_file": FAQ_FILE}),
    ]
    tagged = _tag_metadata(chunks)
    assert tagged[0].metadata["question_number"] == 5
    assert tagged[1].metadata["question_number"] == 5


def test_unknown_source_file_does_not_raise():
    chunks = [Document(page_content="some text", metadata={"source_file": "unrelated.pdf"})]
    tagged = _tag_metadata(chunks)
    assert tagged[0].metadata["doc_type"] == "unknown"
    assert tagged[0].metadata["question_number"] is None


# ---------------------------------------------------------------------------
# _split_qa_pages (synthetic pages: fast, no PDF needed)
# ---------------------------------------------------------------------------


def _splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=settings.CHUNK_SEPARATORS,
    )


def test_qa_split_gives_one_question_per_chunk():
    page = Document(
        page_content=(
            "Eastmere Bank | EMB-KYC-FAQ-001 | Version 3.1\n"
            "Page 4\n"
            "Section D: Beneficial Ownership\n"
            "Q15. What is the threshold?\nIt is 25%.\n"
            "Q16. Is a different threshold applied to High-risk customers?\nYes, 10%.\n"
        ),
        metadata={"source_file": FAQ_FILE},
    )
    chunks = _split_qa_pages([page], _splitter())

    assert len(chunks) == 2
    assert chunks[0].page_content.startswith("Q15.")
    assert chunks[1].page_content.startswith("Q16.")
    # The two near-duplicate questions must never share a chunk
    assert "Q16." not in chunks[0].page_content


def test_qa_split_drops_page_header_and_records_section():
    page = Document(
        page_content=(
            "Eastmere Bank | EMB-KYC-FAQ-001 | Version 3.1\n"
            "Page 4\n"
            "Section D: Beneficial Ownership\n"
            "Q15. What is the threshold?\nIt is 25%.\n"
        ),
        metadata={"source_file": FAQ_FILE},
    )
    chunks = _split_qa_pages([page], _splitter())

    assert "EMB-KYC-FAQ-001" not in chunks[0].page_content
    assert "Section D" not in chunks[0].page_content
    assert chunks[0].metadata["section"] == "Section D: Beneficial Ownership"


def test_qa_split_section_heading_mid_page_labels_following_questions():
    page = Document(
        page_content=(
            "Section A: Purpose and Scope\n"
            "Q4. Who owns this document?\nThe Head of Financial Crime Compliance.\n"
            "Section B: Customer Identification and Verification\n"
            "Q5. What information must be collected?\nFull legal name and more.\n"
        ),
        metadata={"source_file": FAQ_FILE},
    )
    chunks = _split_qa_pages([page], _splitter())

    assert chunks[0].metadata["section"] == "Section A: Purpose and Scope"
    assert chunks[1].metadata["section"] == "Section B: Customer Identification and Verification"
    # The next section's heading must not leak into the previous answer
    assert "Section B" not in chunks[0].page_content


def test_qa_split_long_answer_falls_back_to_recursive_splitter():
    long_answer = "This is a long answer sentence. " * 80  # well over CHUNK_SIZE
    page = Document(
        page_content=f"Q7. A question with a very long answer?\n{long_answer}\n",
        metadata={"source_file": FAQ_FILE},
    )
    chunks = _tag_metadata(_split_qa_pages([page], _splitter()))

    assert len(chunks) > 1
    assert all(len(c.page_content) <= settings.CHUNK_SIZE for c in chunks)
    assert all(c.metadata["question_number"] == 7 for c in chunks)


# ---------------------------------------------------------------------------
# End-to-end on the real documents in data/
# ---------------------------------------------------------------------------

_real_docs_present = all(os.path.isfile(p) for p in settings.document_paths())


@pytest.mark.skipif(not _real_docs_present, reason="Source PDFs not found in data/")
def test_real_faq_produces_one_chunk_per_question_in_order():
    chunks = load_and_chunk()
    faq_chunks = [c for c in chunks if c.metadata["doc_type"] == "kyc_faq"]

    assert [c.metadata["question_number"] for c in faq_chunks] == list(range(1, 30))
    assert all(c.page_content.startswith(f"Q{c.metadata['question_number']}.") for c in faq_chunks)


@pytest.mark.skipif(not _real_docs_present, reason="Source PDFs not found in data/")
def test_real_documents_have_no_unknown_doc_types():
    # Catches filename typos in DOCUMENT_CONFIG, which otherwise fail silently
    chunks = load_and_chunk()
    assert {c.metadata["doc_type"] for c in chunks} == {"kyc_faq", "aml_risk_policy"}