"""
FastAPI entry point for the Hybrid RAG Banking Compliance Assistant.

Sits alongside main.py (CLI) and wraps the existing RAGPipeline unchanged:
main.py is for the terminal, api.py is for the React frontend over HTTP.

Run locally:
    uvicorn api:app --reload --port 8000
Then open http://localhost:8000/docs to try the endpoints in the browser.

Run in a container behind Azure Container Apps ingress:
    uvicorn api:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips="*"
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src import reranker
from src.pipeline import RAGPipeline

logger = logging.getLogger("rag-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Config (from environment, with local-dev defaults)
# ---------------------------------------------------------------------------

# Browser origins allowed to call this API (comma-separated).
# Local default covers Vite (5173) and Create React App (3000).
# In Azure, set to your Static Web Apps URL.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")
    if origin.strip()
]

# Per-visitor limit (by IP), plus a global daily cap across ALL visitors.
# The global cap is the hard ceiling on Azure OpenAI spend.
PER_IP_LIMIT = os.getenv("PER_IP_LIMIT", "5/minute;30/day")
GLOBAL_DAILY_LIMIT = os.getenv("GLOBAL_DAILY_LIMIT", "300/day")

MAX_QUESTION_CHARS = 500

limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# Startup: build the pipeline and load the reranker ONCE
# ---------------------------------------------------------------------------

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Building RAG pipeline (chunking documents, loading FAISS index)...")
    pipeline = RAGPipeline().build()
    # The reranker otherwise loads lazily on the first question, making the
    # first visitor wait. Loading it here moves that cost to startup.
    logger.info("Loading reranker model...")
    reranker._get_reranker_model()
    logger.info("API ready.")
    yield


app = FastAPI(title="Hybrid RAG Banking Compliance API", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)

# ---------------------------------------------------------------------------
# Request / response shapes
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=MAX_QUESTION_CHARS)


class Source(BaseModel):
    source_file: str | None = None
    doc_type: str | None = None
    question_number: int | None = None
    section: str | None = None
    page: int | None = None
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Liveness check; the frontend can also call this to wake the server."""
    return {"status": "ok", "pipeline_loaded": pipeline is not None}


@app.post("/ask", response_model=AskResponse)
@limiter.limit(PER_IP_LIMIT)
@limiter.limit(GLOBAL_DAILY_LIMIT, key_func=lambda request: "global")
def ask(request: Request, body: AskRequest):
    # slowapi requires the `request: Request` parameter, even though it's unused here.
    if pipeline is None:
        raise HTTPException(status_code=503, detail="The assistant is still starting up.")

    try:
        result = pipeline.ask(body.question)
    except Exception:
        # Log the real error for us; never send stack traces to the browser.
        logger.exception("Pipeline error for question: %r", body.question)
        raise HTTPException(status_code=500, detail="Something went wrong answering that question.")

    sources = []
    for doc in result["source_documents"]:
        meta = doc.metadata
        page = meta.get("page")
        sources.append(
            Source(
                source_file=meta.get("source_file"),
                doc_type=meta.get("doc_type"),
                question_number=meta.get("question_number"),
                section=meta.get("section"),
                page=page + 1 if isinstance(page, int) else None,  # PyPDF pages are 0-based
                text=doc.page_content,
            )
        )

    return AskResponse(answer=result["answer"], sources=sources)