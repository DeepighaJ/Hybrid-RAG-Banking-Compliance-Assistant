"""
Centralized configuration for the banking RAG pipeline.

All environment-dependent values (Azure credentials, model names, retrieval
parameters) are defined here so the rest of the codebase never reads
`os.getenv` directly. This keeps configuration auditable in one place and
makes it straightforward to override values for testing.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_index"
EMBEDDING_CACHE_DIR = PROJECT_ROOT / "embedding_cache"

# ---------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5-mini")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# ---------------------------------------------------------------------------
# Document set
#
# Maps each source PDF (relative to DATA_DIR) to a normalized doc_type label
# and whether it uses a numbered Q&A structure (e.g. "Q1.", "Q2.") that should
# be parsed into `question_number` metadata during chunking.
# ---------------------------------------------------------------------------

DOCUMENT_CONFIG: Dict[str, Dict] = {
    "Wolfsberg Group Country Risk FAQs (2024).pdf": {
        "doc_type": "country_risk_faq",
        "has_question_numbers": True,
    },
    "Wolfsberg Group - Risk Based Approach Guidance _June2026.pdf": {
        "doc_type": "rba_guidance",
        "has_question_numbers": False,
    },
}


def document_paths() -> list[str]:
    """Return absolute paths to all configured source documents."""
    return [str(DATA_DIR / filename) for filename in DOCUMENT_CONFIG]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHUNK_SEPARATORS = ["\nQ", "\n\n", "\n", ". ", " "]

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

BM25_TOP_K = int(os.getenv("BM25_TOP_K", "5"))

FAISS_TOP_K = int(os.getenv("FAISS_TOP_K", "5"))
FAISS_FETCH_K = int(os.getenv("FAISS_FETCH_K", "15"))
FAISS_MMR_LAMBDA = float(os.getenv("FAISS_MMR_LAMBDA", "0.5"))

ENSEMBLE_WEIGHTS = [0.5, 0.5]  # [BM25, FAISS]

RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

# ---------------------------------------------------------------------------
# Evaluation (Ragas)
#
# The judge LLM is deliberately independent from both the pipeline's model
# (Azure gpt-5-mini, OpenAI) and from OpenAI's open-weight models, to avoid
# any shared-lineage bias between the model being judged and the model doing
# the judging. Groq hosts Meta's Llama 3.3 70B at no cost (free tier) and at
# high speed (purpose-built inference hardware), which also avoids the
# timeout/reasoning-quality issues seen when judging locally with a small
# CPU-bound model (e.g. llama3.2:1b via Ollama).
#
# Get a free API key (no credit card required): https://console.groq.com
#
# Embeddings for the AnswerRelevancy metric still run locally via Ollama,
# since Groq does not offer an embeddings endpoint and embedding calls are
# cheap/fast enough that local CPU inference is not a bottleneck here.
#   ollama pull nomic-embed-text
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

EVAL_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

USE_SEMANTIC_CACHE = os.getenv("USE_SEMANTIC_CACHE", "false").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_CACHE_SCORE_THRESHOLD = float(os.getenv("REDIS_CACHE_SCORE_THRESHOLD", "0.95"))


def validate() -> None:
    """Raise a clear error early if required configuration is missing."""
    missing = [
        name
        for name, value in [
            ("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY),
            ("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT),
        ]
        if not value
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            f"Copy .env.example to .env and fill in your Azure OpenAI credentials."
        )
