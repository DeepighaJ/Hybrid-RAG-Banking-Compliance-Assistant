# Container image for the Hybrid RAG Banking Compliance API (api.py).
#
# Build:  docker build -t rag-api .
# Run:    docker run --rm -p 8000:8000 --env-file .env rag-api
#
# The FAISS index is NOT copied in. It is built from data/*.pdf when the
# container starts (44 chunks: about a second and a tiny embedding cost),
# which guarantees the index always matches the documents in the image.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface

# Run as a non-root user (security best practice)
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
RUN chown appuser:appuser /app

# 1. CPU-only PyTorch. The default Linux torch wheel bundles several GB of
#    GPU (CUDA) libraries that are useless here, since Container Apps has no GPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# 2. Runtime dependencies (evaluation-only packages live in requirements-eval.txt)
COPY requirements.txt .
RUN pip install -r requirements.txt

USER appuser

# 3. Download the reranker model at BUILD time, so containers never download
#    it on startup (it is large, and cold starts would otherwise be very slow).
ARG RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('${RERANKER_MODEL_NAME}')"

# 4. Application code and source documents (copied last: they change most often,
#    so Docker can reuse the cached layers above on rebuilds)
COPY --chown=appuser:appuser config/ config/
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser data/ data/
COPY --chown=appuser:appuser api.py .

# Use only the model baked into the image; skip Hugging Face network checks
ENV HF_HUB_OFFLINE=1 \
    RERANKER_MODEL_NAME=${RERANKER_MODEL_NAME}

EXPOSE 8000

# --proxy-headers: behind Azure's ingress, read the visitor's real IP from
# X-Forwarded-For, so per-IP rate limits apply per visitor, not per proxy.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]