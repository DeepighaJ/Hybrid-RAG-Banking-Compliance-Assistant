"""
Promptfoo provider adapter for the Hybrid RAG pipeline.

Builds the pipeline once (module-level singleton) and reuses it across all
test cases, since rebuilding the FAISS index per test case would be far too
slow for a regression suite you want to run often. Retrieval, reranking, and
the prompt template all stay exactly as configured in src/pipeline.py and
src/prompts.py — this adapter does not alter pipeline behavior, it only
calls it and reports the result back to Promptfoo in the shape it expects.

Promptfoo invokes call_api() once per test case with the rendered prompt
(here just the raw query, see promptfooconfig.yaml) and the test's `vars`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import RAGPipeline  # noqa: E402

_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline().build()
    return _pipeline


def call_api(prompt: str, options: dict, context: dict) -> dict:
    query = context["vars"]["query"]
    pipeline = _get_pipeline()
    result = pipeline.ask(query)

    return {
        "output": result["answer"],
        "metadata": {
            "sources": [
                doc.metadata.get("source_file", "unknown")
                for doc in result["source_documents"]
            ]
        },
    }
