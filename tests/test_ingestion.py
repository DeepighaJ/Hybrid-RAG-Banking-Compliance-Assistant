"""
Minimal smoke tests for document ingestion metadata tagging.

Run with: pytest tests/
"""

from langchain_core.documents import Document

from src.ingestion import _tag_metadata


def test_tags_doc_type_for_known_source():
    chunks = [
        Document(
            page_content="Q1. What is country risk?",
            metadata={"source_file": "Wolfsberg Group Country Risk FAQs (2024).pdf"},
        )
    ]
    tagged = _tag_metadata(chunks)
    assert tagged[0].metadata["doc_type"] == "country_risk_faq"


def test_parses_question_number_for_faq_document():
    chunks = [
        Document(
            page_content="Q3. How often should ratings be reviewed?",
            metadata={"source_file": "Wolfsberg Group Country Risk FAQs (2024).pdf"},
        )
    ]
    tagged = _tag_metadata(chunks)
    assert tagged[0].metadata["question_number"] == 3


def test_question_number_is_none_for_non_qa_document():
    chunks = [
        Document(
            page_content="Proportionality: an FI should design...",
            metadata={"source_file": "Wolfsberg Group - Risk Based Approach Guidance _June2026.pdf"},
        )
    ]
    tagged = _tag_metadata(chunks)
    assert tagged[0].metadata["question_number"] is None
    assert tagged[0].metadata["doc_type"] == "rba_guidance"


def test_question_number_carries_forward_across_chunks_of_same_question():
    chunks = [
        Document(
            page_content="Q5. What methodologies are available?",
            metadata={"source_file": "Wolfsberg Group Country Risk FAQs (2024).pdf"},
        ),
        Document(
            page_content="...continued discussion with no new Q marker...",
            metadata={"source_file": "Wolfsberg Group Country Risk FAQs (2024).pdf"},
        ),
    ]
    tagged = _tag_metadata(chunks)
    assert tagged[0].metadata["question_number"] == 5
    assert tagged[1].metadata["question_number"] == 5


def test_unknown_source_file_does_not_raise():
    chunks = [Document(page_content="some text", metadata={"source_file": "unrelated.pdf"})]
    tagged = _tag_metadata(chunks)
    assert tagged[0].metadata["doc_type"] == "unknown"
    assert tagged[0].metadata["question_number"] is None
