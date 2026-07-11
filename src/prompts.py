"""
Prompt templates for answer generation.
"""

from langchain_core.prompts import PromptTemplate

QA_PROMPT = PromptTemplate.from_template(
    "You are a compliance assistant answering questions using only the provided "
    "excerpts from AML/CTF guidance documents.\n\n"
    "Rules:\n"
    "- Answer only using the context below. Do not use outside knowledge.\n"
    "- If the answer is not fully contained in the context, say: "
    "\"I don't have enough information in the provided documents to answer that.\"\n"
    "- Do not name specific countries, entities, or make risk determinations "
    "not explicitly stated in the context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)
