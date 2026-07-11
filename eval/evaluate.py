"""
Runs the RAG pipeline against the curated evaluation dataset and scores the
results using Ragas, with Llama 3.3 70B (via Groq) as the judge LLM and a
local Ollama embedding model for the AnswerRelevancy metric's similarity
calculation.

An independent judge (rather than the pipeline's own Azure gpt-5-mini) is
used for two reasons:
  1. Avoids shared-lineage bias — Llama 3.3 (Meta, via Groq) has no shared
     training lineage with the Azure OpenAI model being judged, unlike an
     OpenAI open-weight model would.
  2. Keeps evaluation cost off the Azure quota, and avoids the slow,
     unreliable judgments seen when using a small, local, CPU-bound model
     (e.g. llama3.2:1b via Ollama), which lacks the reasoning capacity for
     metrics like Faithfulness and times out under concurrent load.

Requires:
  - A free Groq API key (https://console.groq.com) set as GROQ_API_KEY in .env
  - Ollama running locally with the embedding model pulled:
      ollama pull nomic-embed-text

Usage:
    python eval/evaluate.py
    python eval/evaluate.py --limit 5      # quick smoke test on first 5 questions
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness
from ragas.run_config import RunConfig

from config import settings
from eval.eval_dataset import EVAL_DATASET
from src.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline_on_dataset(pipeline: RAGPipeline, dataset: list[dict]) -> list[dict]:
    """
    Run every question in `dataset` through the real pipeline and collect
    the fields Ragas needs: user_input, response, retrieved_contexts,
    and the hand-written reference answer carried through unchanged.
    """
    rows = []
    for i, item in enumerate(dataset, 1):
        logger.info("Running question %d/%d: %s", i, len(dataset), item["question"])
        result = pipeline.ask(item["question"])
        rows.append({
            "user_input": item["question"],
            "response": result["answer"],
            "retrieved_contexts": [doc.page_content for doc in result["source_documents"]],
            "reference": item["reference"],
        })
    return rows


def get_judge_llm() -> LangchainLLMWrapper:
    """
    Judge LLM: Llama 3.3 70B via Groq's free tier.

    Deliberately independent from the pipeline's Azure OpenAI model to avoid
    any shared-lineage bias between the model being judged and the model
    doing the judging, and fast enough (purpose-built inference hardware)
    to avoid the timeout issues seen with a local CPU-bound judge.
    """
    chat = ChatGroq(model=settings.GROQ_JUDGE_MODEL, api_key=settings.GROQ_API_KEY)
    return LangchainLLMWrapper(chat)


def get_judge_embeddings() -> LangchainEmbeddingsWrapper:
    embeddings = OllamaEmbeddings(
        model=settings.OLLAMA_EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL
    )
    return LangchainEmbeddingsWrapper(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with Ragas.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N questions (useful for a quick smoke test).",
    )
    args = parser.parse_args()

    dataset = EVAL_DATASET[: args.limit] if args.limit else EVAL_DATASET

    logger.info("Building pipeline...")
    pipeline = RAGPipeline().build()

    logger.info("Running %d evaluation questions through the pipeline...", len(dataset))
    rows = run_pipeline_on_dataset(pipeline, dataset)

    logger.info("Preparing Ragas judge (Groq: %s)...", settings.GROQ_JUDGE_MODEL)
    judge_llm = get_judge_llm()
    judge_embeddings = get_judge_embeddings()

    ragas_dataset = EvaluationDataset.from_list(rows)

    metrics = [
        Faithfulness(),
        # strictness=1 avoids requesting multiple completions (n>1) per call;
        # Groq's API only supports n=1, unlike OpenAI's API which this metric
        # was originally designed against. Ragas' default strictness=3 samples
        # 3 completions per question to average out variance, so this trades
        # a small amount of score robustness for compatibility with Groq.
        AnswerRelevancy(strictness=1),
        ContextPrecision(),
    ]

    logger.info("Scoring with Ragas (Groq judge + local Ollama embeddings)...")

    # max_workers is shared across all metrics: Faithfulness/ContextPrecision use
    # the fast Groq judge, but AnswerRelevancy also calls the local Ollama
    # embedding model, which cannot reliably serve high concurrency.
    # Additionally, Groq's free tier has a per-minute rate limit; a run at
    # max_workers=3 was observed hitting frequent 429 Too Many Requests
    # errors with long retry backoffs, making the run slower overall than a
    # lower, steadier concurrency that stays under the rate limit.
    judge_run_config = RunConfig(timeout=300, max_workers=2)

    result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=judge_run_config,
    )

    df = result.to_pandas()

    settings.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = settings.EVAL_RESULTS_DIR / f"eval_{timestamp}.csv"
    df.to_csv(out_path, index=False)

    print("\n=== Per-question scores ===")
    print(df[["user_input", "faithfulness", "answer_relevancy", "context_precision"]]
          .to_string(index=False))

    print("\n=== Average scores ===")
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        print(f"{metric}: {df[metric].mean():.3f}")

    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()