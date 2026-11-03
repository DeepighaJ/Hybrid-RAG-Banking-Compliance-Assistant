# Hybrid RAG: Banking/AML Compliance Assistant

**Live demo:** https://ambitious-rock-0e6d4db1e.4.azurestaticapps.net
*(Runs on scale-to-zero hosting: the first question after a quiet period can take a minute while the server starts.)*

![Answer with a clickable policy citation highlighting its source](assets/app-answer.png)

A production-structured retrieval-augmented generation (RAG) system for answering questions over
banking compliance documentation. It combines a hybrid retrieval pipeline (dense + sparse search,
cross-encoder reranking, and document-aware chunking) on Azure OpenAI with two complementary
evaluation layers (Ragas quality scoring and Promptfoo regression testing), and is deployed as a
FastAPI service on Azure Container Apps behind a React frontend on Azure Static Web Apps.

This repository is a refactored, modular version of an original prototype notebook (retained under
`notebooks/` for reference), restructured into a standard Python package layout suitable for version
control, testing, and reuse outside of a single notebook.

## Source Documents

The knowledge base is two **fictional, self-authored** policy documents for an invented institution,
**Eastmere Bank**, written specifically for this project:

| Document | Format | Purpose in the corpus |
|---|---|---|
| `Eastmere Bank - KYC and CDD Procedures FAQ.pdf` (EMB-KYC-FAQ-001) | 29 numbered Q&As | Due diligence procedures, thresholds, review cycles |
| `Eastmere Bank - AML Risk Assessment Policy.pdf` (EMB-AML-RAP-002) | Narrative policy, `RA-x.x` sections | Risk scoring model, tier thresholds, country risk, governance |

The documents are designed as a **test corpus**, not just content:

- **Cross-references:** the FAQ points to the policy for how risk tiers are calculated, and the
  policy points back to the FAQ for review frequencies, so some questions can only be answered by
  combining both documents.
- **Near-duplicate facts:** e.g. a 25% beneficial ownership threshold for Low/Medium-risk customers
  vs 10% for High-risk; several different review frequencies (quarterly, annual, 12/24/36 months).
- **Exact policy codes** (`CDD-4.2`, `RA-3.5`, `EMB-ISR-01`) on every answer, to test keyword retrieval.
- **A scoring formula with a worked example**, to test whether the model applies stated rules correctly.

> The project originally used two Wolfsberg Group guidance documents. Their publication terms
> restrict reproduction and hosting, which prevented committing the PDFs or deploying a public demo,
> so the corpus was replaced with original documents. Eastmere Bank is not a real institution, and
> the documents are not real policies or compliance advice.

## Architecture

![Architecture diagram](assets/architecture.svg)

**Pipeline:** PDF ingestion → document-aware chunking → BM25 (sparse) + FAISS with MMR (dense) →
weighted ensemble fusion → cross-encoder reranking (`BAAI/bge-reranker-v2-m3`, top 5) →
grounded answer generation (Azure OpenAI `gpt-5-mini`) with policy-code citations.

**Deployment:** React frontend (Azure Static Web Apps) → FastAPI service in a Docker container
(Azure Container Apps) → Azure OpenAI. See [Deployment](#deployment).

**Chunking is document-aware:**

- The Q&A document is split **one question per chunk**, so near-duplicate questions never share a
  chunk; each chunk carries `question_number` and `section` metadata, and page headers are dropped.
- The narrative policy uses recursive character splitting.

**The frontend makes answers traceable:** every policy code in an answer (`CDD-5.3`, `RA-3.7`) is
clickable and highlights the retrieved passage that contains it, next to a ranked list of all
sources with document, question or section, and page.

## Project Structure

```
.
├── config/
│   └── settings.py          # All environment-dependent configuration, loaded once
├── src/
│   ├── ingestion.py         # PDF loading, Q&A-aware chunking, metadata tagging
│   ├── embeddings.py        # Cached Azure OpenAI embeddings client
│   ├── vectorstore.py       # FAISS build/load/persist
│   ├── retrieval.py         # BM25 (code-aware tokenizer), FAISS+MMR, ensemble fusion
│   ├── reranker.py          # Cross-encoder reranking
│   ├── prompts.py           # Grounding-focused prompt template
│   ├── cache.py             # LLM response caching (in-memory / optional Redis semantic)
│   └── pipeline.py          # RAGPipeline: orchestrates all stages end-to-end
├── frontend/                # React + Vite UI with clickable policy-code citations
│   ├── src/
│   │   ├── App.jsx          # Chat, citation chips, ranked sources panel
│   │   ├── api.js           # Client for the FastAPI backend (VITE_API_URL)
│   │   ├── App.css
│   │   └── index.css        # Design tokens, light and dark themes
│   └── vite.config.js       # Dev server pinned to port 5174
├── scripts/
│   └── build_index.py       # CLI: build/rebuild the FAISS index offline
├── tests/
│   └── test_ingestion.py    # Unit + end-to-end tests for chunking and metadata
├── eval/
│   ├── eval_dataset.py      # 19 curated questions + reference answers
│   ├── evaluate.py          # Runs the pipeline, scores with Ragas
│   ├── debug_faithfulness.py # Shows the judge's claim-by-claim verdicts for one answer
│   ├── promptfoo_tests.py   # Generates Promptfoo test cases from eval_dataset.py
│   ├── promptfoo_provider.py # Promptfoo adapter wrapping RAGPipeline
│   └── results/             # Saved per-run score reports
├── .github/workflows/       # GitHub Actions: builds and deploys the frontend on push to main
├── notebooks/
│   └── banking_hybrid_rag.ipynb   # Original prototype notebook (reference)
├── assets/                  # README images
├── data/                    # Source PDFs (Eastmere Bank, committed)
├── api.py                   # FastAPI service: /ask, /health, CORS, rate limits
├── main.py                  # CLI entrypoint for querying the pipeline
├── Dockerfile               # API container image
├── .dockerignore            # Keeps secrets, eval code and the frontend out of the image
├── requirements.txt         # Runtime dependencies (pipeline, CLI, API)
├── requirements-eval.txt    # Eval and test dependencies (includes requirements.txt)
├── promptfooconfig.yaml
├── .env.example
└── README.md
```

## Setup

### 1. Prerequisites

- Python 3.10+
- Node.js 22.22+ (for the frontend and Promptfoo)
- An Azure OpenAI resource with a chat model (e.g. `gpt-5-mini`) and an embedding model
  (e.g. `text-embedding-3-small`) deployed

> **Windows tip:** keep the project in a path without `&` or spaces (e.g. `D:\Projects\...`).
> npm runs scripts through `cmd.exe`, which treats `&` as a command separator and breaks the
> frontend tooling.

### 2. Install dependencies

```bash
git clone <repo-url>
cd <repo-folder>
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements-eval.txt
```

`requirements-eval.txt` installs everything, including the evaluation and test tools.
`requirements.txt` alone is enough to run the pipeline and API (this is what the Docker image uses).

### 3. Configure environment variables

```bash
cp .env.example .env
```

Fill in `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` at minimum. Use the **base** endpoint
(e.g. `https://<resource>.openai.azure.com/`), not a full API path copied from the portal, and a
valid API version (default `2024-10-21`).

### 4. Build the index

```bash
python scripts/build_index.py
```

This loads both PDFs from `data/`, chunks them, embeds them (using the disk-backed embedding cache),
and persists a FAISS index to `faiss_index/`. **Re-run with `--force-rebuild` after changing the
documents or chunking settings**; otherwise the old index is reused while BM25 is rebuilt from the
new text, and the two retrievers silently disagree.

### 5. Ask questions

From the command line:

```bash
python main.py "What is the beneficial ownership threshold for High-risk customers?"
python main.py --interactive
```

Or run the full stack (API + frontend) locally:

```bash
# Terminal 1: API on http://localhost:8000 (interactive docs at /docs)
python -m uvicorn api:app --port 8000

# Terminal 2: frontend on http://localhost:5174
cd frontend
npm install
npm run dev
```

Good demo questions:

| Question | What it demonstrates |
|---|---|
| What is the beneficial ownership threshold for High-risk customers? | Precision: 10%, not the similar 25% rule |
| What does policy CDD-5.4 say? | Exact policy-code lookup via BM25 |
| What weighted risk score makes a customer High risk, and how often must they be reviewed? | Cross-document synthesis |
| A trust connected to a Medium-rated jurisdiction uses private banking and was onboarded face-to-face. What is its risk score and tier? | Applying the scoring formula (3.25, Medium) |
| What interest rate does Eastmere Bank pay on savings accounts? | Declining instead of fabricating |

## Using the Pipeline Programmatically

```python
from src.pipeline import RAGPipeline

pipeline = RAGPipeline().build()
result = pipeline.ask("Who must approve a relationship with a politically exposed person?")
print(result["answer"])

for doc in result["source_documents"]:
    print(doc.metadata, doc.page_content[:200])
```

## Adding More Documents

1. Place the new PDF in `data/`
2. Add an entry to `DOCUMENT_CONFIG` in `config/settings.py` using the **exact filename**. The key
   must match the file name character for character, or the document silently falls back to
   `doc_type: "unknown"` (a test in `tests/test_ingestion.py` catches this):
   ```python
   "Your Exact File Name.pdf": {
       "doc_type": "your_doc_type_label",
       "has_question_numbers": False,  # True only for a "Q1.", "Q2." Q&A structure
   },
   ```
3. Rebuild the index: `python scripts/build_index.py --force-rebuild`

## Testing

```bash
python -m pytest tests/ -v
```

11 tests cover metadata tagging, one-question-per-chunk splitting (including that near-duplicate
questions never share a chunk), section tracking, long-answer fallback, and end-to-end checks on
the real PDFs.

## Evaluation

### Dataset

`eval/eval_dataset.py` holds 19 questions with hand-written reference answers, each verified
against the source PDFs:

| Query type | Count | Tests |
|---|---|---|
| `single_document_faq` / `single_document_policy` | 7 | Answerable from one document |
| `calculation` | 2 | Applying the RA-3.4 scoring formula, including the missing-data rule |
| `cross_document` | 4 | Combining facts from both documents |
| `precision_test` | 4 | Near-duplicate facts and exact policy-code lookup |
| `out_of_scope` | 2 | Must return the exact refusal phrase |

### Layer 1: Ragas quality scores

Metrics: **Faithfulness** (is every claim supported by the retrieved context?), **Answer
Relevancy** (does the answer address the question?), and **Context Precision** (were the right
chunks retrieved and ranked highly?).

**Judge:** `openai/gpt-oss-120b` via Groq's free tier, configurable with `GROQ_JUDGE_MODEL`.
The original judge (Llama 3.3 70B) was retired by Groq in August 2026, and its suggested
replacement (Qwen 3.6, a preview model) was retired a month later; gpt-oss-120b was chosen as a
Groq *production* model for stability. It shares some lineage with the pipeline's OpenAI model,
which is mitigated by cross-checking with Promptfoo and inspecting low scores claim by claim.
Answer Relevancy embeddings run locally via Ollama (`nomic-embed-text`).

```bash
# Add GROQ_API_KEY to .env (free: https://console.groq.com), then:
ollama pull nomic-embed-text
python eval/evaluate.py --limit 3    # smoke test
python eval/evaluate.py              # full run
```

**How the evals found and fixed two real bugs**

The first full run exposed two retrieval problems:

1. **Exact policy codes were invisible to BM25.** The default tokenizer splits only on spaces, so
   `(Policy reference: CDD-5.4)` became the token `CDD-5.4)` and never matched a query for
   `CDD-5.4`. Fixed with a lowercasing, code-preserving tokenizer.
2. **Multi-part questions got partial refusals.** Only the top 3 reranked chunks reached the model,
   too few for questions needing 2-3 facts, so it answered part and declined the rest. Fixed by
   raising `RERANK_TOP_N` to 5.

| Metric (17 in-scope questions) | Before fixes | After fixes |
|---|---|---|
| Faithfulness | 0.852 | **0.893** |
| Answer Relevancy | 0.602 | **0.845** |
| Context Precision | 0.868 | 0.836 |
| Questions with partial refusals | 5 | **0** |
| `CDD-5.4` lookup (faithfulness / relevancy / precision) | 0 / 0 / 0 | **1.0 / 0.63 / 1.0** |

The small Context Precision drop is the expected cost of passing 5 chunks instead of 3; complete
answers were judged worth more than a tighter context list.

**Per-question results (after fixes)**

| # | Question | Type | Faithfulness | Answer relevancy | Context precision |
|---|---|---|---|---|---|
| 1 | Can an account be opened before verification is complete? | FAQ | 0.91 | 0.81 | 0.92 |
| 2 | Who approves a PEP relationship? | FAQ | 0.60 ¹ | 0.97 | 1.00 |
| 3 | Which events trigger an out-of-cycle review? | FAQ | 1.00 | 0.72 | 1.00 |
| 4 | How long are KYC records retained? | FAQ | 1.00 | 0.86 | 1.00 |
| 5 | Risk scoring factors and weights | Policy | 1.00 | 0.80 | 0.50 |
| 6 | Which customers are automatically High risk? | Policy | 1.00 | 0.91 | 0.83 |
| 7 | Relationships in a Prohibited jurisdiction | Policy | 0.83 | 0.79 | 1.00 |
| 8 | Trust: risk score and tier (3.25, Medium) | Calculation | 1.00 | 0.74 | 0.58 |
| 9 | Missing geography data: score and tier (2.35, Medium) | Calculation | 0.50 ³ | 0.93 | 0.95 |
| 10 | High-risk score threshold and review frequency | Cross-document | 1.00 | 0.99 | 0.33 |
| 11 | Worked-example company: ownership threshold | Cross-document | 1.00 | 0.83 | 0.70 |
| 12 | Country rated High: what happens, and how quickly | Cross-document | 1.00 ² | 0.72 | 1.00 |
| 13 | PEP: tier, approver, and review frequency | Cross-document | 1.00 | 0.79 | 0.81 |
| 14 | Ownership threshold for Medium-risk customers | Precision | 1.00 | 1.00 | 1.00 |
| 15 | How often is the Country Risk List reviewed? | Precision | 1.00 | 0.91 | 1.00 |
| 16 | Who approves AML policy changes? | Precision | 0.33 ³ | 0.98 | 0.58 |
| 17 | What does policy CDD-5.4 say? | Precision | 1.00 | 0.63 | 1.00 |
| | **Average (17 in-scope questions)** | | **0.893** | **0.845** | **0.836** |

¹ A Ragas claim-splitting false negative: the "Head **or** delegated deputy" rule was split into two
"must approve" claims. The answer is verbatim-correct, and Promptfoo passes it.
² Faithful but incomplete: the answer correctly declines the "how quickly" part (the multi-hop
limitation). Faithfulness measures unsupported claims, not completeness; Promptfoo catches this one.
³ Not yet diagnosed claim by claim with `eval/debug_faithfulness.py`.

The two out-of-scope questions are excluded from the averages, since Ragas scores a correct
refusal as 0 answer relevancy. Promptfoo verifies them with an exact-phrase check instead.

### Layer 2: Promptfoo regression tests

Test cases are generated from the same `eval_dataset.py` (one source of truth). Out-of-scope
questions must contain the exact refusal string; all others are graded by an `llm-rubric` check
against the reference answer, using the same Groq judge.

```bash
npx promptfoo eval -j 1   # -j 1 avoids Groq free-tier rate limits
npx promptfoo view
```

**Result: 18 of 19 passing.** The one failure is a genuine limitation (see below), kept failing
deliberately rather than hidden by editing the reference answer.

![Promptfoo results summary: 18 of 19 passing](assets/promptfoo-results.png)

![Promptfoo per-test results grid](assets/promptfoo-results1.png)

### Known limitations (found by the evals)

- **Multi-hop retrieval.** "If a country's rating is raised to High, what must happen, and how
  quickly?" needs RA-4.4, which only says the review follows `CDD-6.3`, and then the FAQ answer
  that defines the 5-business-day timeline. Single-pass retrieval matches the question's wording,
  which resembles RA-4.4 but not the FAQ answer, so the timeline is missed. The model correctly
  answers what it can and **declines the rest instead of guessing**. A planned fix is
  citation-following retrieval: pulling in chunks for policy codes referenced by the top results.
- **Ragas faithfulness and "or" rules.** For "approved by the Head of Financial Crime Compliance
  **or** a delegated deputy", Ragas' claim-extraction step produced "the Head *must* approve" and
  "the deputy *must* approve", then correctly rejected both, scoring a verbatim-correct answer
  0.4-0.6. This was diagnosed with `eval/debug_faithfulness.py`, reproduced with two different
  judge models, and cross-checked with Promptfoo, which passes the same answer.
- **Reference answers must match the question's scope.** Four early Promptfoo failures were
  correct answers penalized for omitting side facts included in the references. The references
  were trimmed to what each question asks.
- **Narrative chunks can span two sections.** The policy document is split by size, so a chunk can
  start mid-section and run into the next one; its source label then names the later heading. A
  planned fix is splitting the policy at its `RA-x.x` headings, as the FAQ is split by question.

## Deployment

| Component | Service | Notes |
|---|---|---|
| Frontend (React + Vite) | Azure Static Web Apps (Free) | Built and deployed by GitHub Actions on every push to `main` |
| API (FastAPI) | Azure Container Apps | 2 vCPU / 4 GiB, scale-to-zero (min 0, max 1 replica) |
| Container image | GitHub Container Registry | Public image; no secrets inside |
| LLM + embeddings | Azure OpenAI | `gpt-5-mini`, `text-embedding-3-small` |

**API** (`api.py`): `POST /ask` and `GET /health`, with CORS restricted to the frontend's origin,
per-IP and global daily rate limits (a hard ceiling on Azure OpenAI spend), input length
validation, and structured source metadata for the citation panel. The reranker is loaded at
startup so the first visitor doesn't wait for it.

**Docker image:** CPU-only PyTorch (avoids several GB of unused GPU libraries), the reranker model
downloaded at build time so containers never fetch it on startup, a non-root user, and the FAISS
index rebuilt from the PDFs at startup so it always matches the documents in the image.

**Configuration and secrets:** the Azure OpenAI key is stored as an encrypted Container Apps secret
and referenced by an environment variable; `.env` is excluded from both git and the Docker build.
The frontend's API address is supplied at build time through `VITE_API_URL` in the GitHub Actions
workflow, and the API's allowed browser origins through `ALLOWED_ORIGINS`.

**Cost controls:** scale-to-zero (no compute cost while idle), a single-replica cap, the API's
global daily request limit, and an Azure budget alert.

### Run the API in Docker locally

```bash
docker build -t rag-api .
docker run --rm -p 8000:8000 --env-file .env rag-api
```

`--env-file` passes your `.env` values in at run time (values must not be wrapped in quotes).

## Notes

- `gpt-5-mini` is a reasoning-family model and does not support a configurable `temperature`.
- The FAISS index and embedding cache are persisted to disk and excluded from version control.
- The optional Redis semantic cache (`USE_SEMANTIC_CACHE=true`) requires Redis Stack with RediSearch.
- This is a portfolio/demonstration project and is not intended for real compliance decisions.

## License

This project is provided for educational and portfolio purposes. The Eastmere Bank documents in
`data/` are original works created for this project and describe a fictional institution.
