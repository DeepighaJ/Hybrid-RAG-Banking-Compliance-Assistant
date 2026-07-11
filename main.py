"""
CLI entrypoint for the banking/AML compliance RAG assistant.

Usage:
    python main.py "How often should country risk data sources be refreshed?"
    python main.py --interactive
"""

import argparse
import logging

from src.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def print_result(result: dict) -> None:
    print("\nAnswer:", result["answer"])
    print("\n--- Sources used ---")
    for i, doc in enumerate(result["source_documents"], 1):
        source_file = doc.metadata.get("source_file", "unknown")
        doc_type = doc.metadata.get("doc_type", "unknown")
        page = doc.metadata.get("page")
        q_num = doc.metadata.get("question_number")
        label = f"Q{q_num}" if q_num is not None else "N/A"

        print(f"\nSource {i} | {source_file} | doc_type: {doc_type} | question: {label} | page {page}")
        print(doc.page_content[:400])


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the banking/AML compliance RAG assistant.")
    parser.add_argument("query", nargs="?", help="Question to ask. Omit with --interactive for a REPL.")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive query loop.")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild the FAISS index even if a cached one exists on disk.",
    )
    args = parser.parse_args()

    logger.info("Building pipeline...")
    pipeline = RAGPipeline().build(force_rebuild=args.force_rebuild)

    if args.interactive:
        print("Banking/AML Compliance RAG Assistant — type 'exit' to quit.")
        while True:
            query = input("\nQuestion: ").strip()
            if query.lower() in {"exit", "quit"}:
                break
            if not query:
                continue
            result = pipeline.ask(query)
            print_result(result)
    elif args.query:
        result = pipeline.ask(args.query)
        print_result(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
