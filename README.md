# Hybrid RAG: Banking/AML Compliance Assistant

A production-structured retrieval-augmented generation (RAG) system for answering questions over
banking compliance documentation, built with a hybrid retrieval pipeline (dense + sparse search,
reranking, and metadata filtering) on top of Azure OpenAI.

This repository is a refactored, modular version of an original prototype notebook (retained under
`notebooks/` for reference), restructured into a standard Python package layout suitable for version
control, testing, and reuse outside of a single notebook.

The reference documents used are two publicly available AML/CTF industry guidance documents from the
[Wolfsberg Group](https://db.wolfsberg-group.org):

1. **Country Risk FAQ (2024)** — a structured Q&A document
2. **Guidance on the Risk-Based Approach (June 2026)** — a narrative guidance document

## Architecture

```
                        ┌─────────────┐
                        │   Query     │
                        └──────┬──────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │  LLM Response Cache Check │
                  │  (in-memory or Redis)     │──── hit ──► Answer
                  └─────────────┬─────────────┘
                              miss
                               ▼
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐        ┌──────────────────────┐
        │  BM25 Retriever   │        │   FAISS Retriever     │
        │  (sparse/lexical) │        │   (dense + MMR)        │
        └────────┬──────────┘        └───────────┬───────────┘
                 │                                │
                 └─────────────┬──────────────────┘
                                ▼
                  ┌──────────────────────────┐
                  │  Ensemble Retriever (RRF) │
                  └─────────────┬─────────────┘
                                ▼
                  ┌──────────────────────────┐
                  │  Cross-Encoder Reranker    │
                  │  (BAAI/bge-reranker-v2-m3) │
                  └─────────────┬─────────────┘
                                ▼
                  ┌──────────────────────────┐
                  │   Prompt Template + LLM    │
                  │   (Azure OpenAI)           │
                  └─────────────┬─────────────┘
                                ▼
                        ┌─────────────┐
                        │   Answer    │
                        └─────────────┘
```

## Project Structure

```
.
├── config/
│   └── settings.py          # All environment-dependent configuration, loaded once
├── src/
│   ├── ingestion.py         # PDF loading, chunking, metadata tagging
│   ├── embeddings.py        # Cached Azure OpenAI embeddings client
│   ├── vectorstore.py       # FAISS build/load/persist
│   ├── retrieval.py         # BM25, FAISS+MMR, ensemble fusion, filtered retrievers
│   ├── reranker.py          # Cross-encoder reranking
│   ├── prompts.py           # Grounding-focused prompt template
│   ├── cache.py             # LLM response caching (in-memory / optional Redis semantic)
│   └── pipeline.py          # RAGPipeline — orchestrates all stages end-to-end
├── scripts/
│   └── build_index.py       # CLI: build/rebuild the FAISS index offline
├── tests/
│   └── test_ingestion.py    # Unit tests for metadata tagging logic
├── eval/
│   ├── eval_dataset.py      # Curated questions + reference answers
│   ├── evaluate.py          # Runs the pipeline, scores with Ragas (Groq judge)
│   └── results/             # Saved per-run score reports (gitignored, except sample)
├── notebooks/
│   └── banking_hybrid_rag.ipynb   # Original prototype notebook (reference)
├── assets/                   # README screenshots
├── data/                     # Source PDFs (not committed; see Setup step 4)
├── main.py                   # CLI entrypoint for querying the pipeline
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

### 1. Prerequisites

- Python 3.10+
- An Azure OpenAI resource with the following deployed:
  - A chat model (e.g. `gpt-5-mini`)
  - An embedding model (e.g. `text-embedding-3-small`)

### 2. Install dependencies

```bash
git clone <repo-url>
cd <repo-folder>
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

![Dependency installation](assets/pip_install.png)

### 3. Configure environment variables

```bash
cp .env.example .env
```

Fill in `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` at minimum. All other variables in
`.env.example` have sensible defaults and are optional to override.

### 4. Download the source documents

The two source PDFs are **not included in this repository**. The Wolfsberg Group's publication
terms require written permission for reproduction and restrict hosting their documents on
third-party sites; per their terms, only linking to the official page (not the document itself) is
permitted. Download them manually from the Wolfsberg Group's resources pages and place them in
`data/`, using the exact filenames below (required, since `config/settings.py` matches on filename):

| File | Source page |
|---|---|
| `Wolfsberg Group Country Risk FAQs (2024).pdf` | [wolfsberg-group.org/resources/rba](https://wolfsberg-group.org/resources/rba) |
| `Wolfsberg Group - Risk Based Approach Guidance _June2026.pdf` | [wolfsberg-group.org/resources/rba](https://wolfsberg-group.org/resources/rba) |

### 5. Build the index

```bash
python scripts/build_index.py
```

This loads both PDFs from `data/`, chunks them, embeds them (using the disk-backed embedding cache),
and persists a FAISS index to `faiss_index/`. Re-run with `--force-rebuild` after changing the source
document set or chunking parameters.

### 6. Ask questions

```bash
python main.py "How often should country risk data sources be refreshed?"
```

Or start an interactive session:

```bash
python main.py --interactive
```

![Example query and response](assets/main_py_run.png)

## Using the Pipeline Programmatically

```python
from src.pipeline import RAGPipeline

pipeline = RAGPipeline().build()

result = pipeline.ask("What are the three key elements of a risk-based approach?")
print(result["answer"])

for doc in result["source_documents"]:
    print(doc.metadata, doc.page_content[:200])
```

Scoped retrieval using metadata filtering:

```python
from src.retrieval import build_filtered_retriever

rba_only = build_filtered_retriever(pipeline.vectorstore, {"doc_type": "rba_guidance"})
result = pipeline.ask("What is prioritisation?", retriever=rba_only)
```

## Adding More Documents

1. Place the new PDF in `data/`
2. Add an entry to `DOCUMENT_CONFIG` in `config/settings.py`, using the **exact filename** (including
   spaces, capitalization, and punctuation) as it appears on disk — the key must match
   `os.path.basename(path)` character-for-character, or the document will silently fall back to
   `doc_type: "unknown"`:
   ```python
   "Your Exact File Name.pdf": {
       "doc_type": "your_doc_type_label",
       "has_question_numbers": False,  # True only if it uses a "Q1.", "Q2." structure
   },
   ```
3. Rebuild the index: `python scripts/build_index.py --force-rebuild`

## Testing

```bash
pip install pytest
pytest tests/
```

## Evaluation

Answer and retrieval quality are evaluated with [Ragas](https://docs.ragas.io/), judged by **Llama
3.3 70B via Groq's free tier** rather than the pipeline's own Azure model. This choice is deliberate:

- **Avoids shared-lineage bias** — Llama 3.3 (Meta) shares no training lineage with the Azure OpenAI
  model being judged, unlike using an OpenAI model (even an open-weight one) to judge OpenAI output.
- **Fast and reliable** — an earlier local judge (`llama3.2:1b` via Ollama) produced unreliable,
  frequently-timing-out scores on the Faithfulness metric, since a 1B-parameter CPU-bound model lacks
  the reasoning capacity for Faithfulness's multi-step claim verification, and Ragas' default
  concurrency overwhelmed a single local Ollama instance. Groq's purpose-built inference hardware
  serves genuine concurrent requests quickly and reliably.
- **Free** — no cost, no credit card required for Groq's free tier.

Embeddings for the Answer Relevancy metric still run locally via Ollama, since Groq does not offer
an embeddings endpoint and this calculation is cheap enough that local CPU inference is not a
bottleneck.

**Metrics used:**

| Metric | What it checks |
|---|---|
| Faithfulness | Is the generated answer supported by the retrieved context, or does it contain unsupported claims? |
| Answer Relevancy | Does the answer actually address the question asked? |
| Context Precision | Were the retrieved chunks actually relevant, and ranked appropriately? |

**Setup (one-time):**

```bash
# Free Groq API key, no credit card: https://console.groq.com
# Add GROQ_API_KEY to .env

# Local embedding model
ollama pull nomic-embed-text
```

**Run evaluation:**

```bash
python eval/evaluate.py              # full dataset
python eval/evaluate.py --limit 5    # quick smoke test on the first 5 questions
```

![Evaluation run and per-question scores](assets/eval_scores.png)

Results are printed to the console and saved as a timestamped CSV under `eval/results/`, so scores
can be compared across runs after changing chunking, retrieval, or reranking parameters.

The evaluation dataset (`eval/eval_dataset.py`) includes single-document questions for each source
PDF, cross-document questions requiring synthesis across both, a couple of near-duplicate questions
to test retrieval precision, and one deliberately out-of-scope question to confirm the grounding
prompt declines rather than fabricates an answer.

![Evaluation dataset structure](assets/eval_dataset.png)

**Example results** (5-question smoke test; full results in [`eval/results/sample_eval_results.csv`](eval/results/sample_eval_results.csv)):

| Metric | Average Score |
|---|---|
| Faithfulness | 1.000 |
| Answer Relevancy | 0.817 |
| Context Precision | 1.000 |

Faithfulness and Context Precision scoring perfectly indicates no hallucinated claims and fully
relevant retrieval on this sample. Answer Relevancy varied (0.67–0.97) across questions; inspection
of individual rows showed lower scores generally corresponded to answers that included genuinely
useful adjacent context (e.g., both a standard cadence and its trigger-based exceptions) rather than
narrowly scoped responses — a reasonable completeness-vs-precision tradeoff for a compliance
assistant, rather than an indication of irrelevant answers.

## Notes

- `gpt-5-mini` is a reasoning-family model and does not support a configurable `temperature`
  parameter; this is intentionally omitted from the LLM configuration.
- The FAISS index and embedding cache are persisted to disk (`faiss_index/`, `embedding_cache/`) and
  excluded from version control — they are regenerated via `scripts/build_index.py`.
- The optional Redis semantic cache (`USE_SEMANTIC_CACHE=true`) requires a Redis Stack instance with
  the RediSearch module; plain Redis is not sufficient.
- This project is a portfolio/demonstration implementation and is not intended for use in an actual
  compliance decision-making process.

## License

This project is provided for educational and portfolio purposes. The source PDFs are © The Wolfsberg
Group and are used here under their publicly available terms for reference only.