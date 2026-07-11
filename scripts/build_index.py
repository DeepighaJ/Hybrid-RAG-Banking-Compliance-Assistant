"""
Standalone script to build (or rebuild) the FAISS index from the configured
source documents, without running a query.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --force-rebuild
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from src import embeddings as embeddings_module, ingestion, vectorstore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or rebuild the FAISS index.")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild the index even if one already exists on disk.",
    )
    args = parser.parse_args()

    settings.validate()

    logger.info("Loading and chunking source documents...")
    pages = ingestion.load_documents()
    chunks = ingestion.chunk_documents(pages)

    logger.info("Preparing embeddings client...")
    cached_embeddings = embeddings_module.get_cached_embeddings()

    logger.info("Building/loading FAISS index...")
    vectorstore.build_or_load_vectorstore(
        chunks, cached_embeddings, force_rebuild=args.force_rebuild
    )

    logger.info("Done. Index available at: %s", settings.FAISS_INDEX_DIR)


if __name__ == "__main__":
    main()
