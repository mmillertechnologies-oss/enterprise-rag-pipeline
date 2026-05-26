#!/usr/bin/env python3
"""CLI for ingesting documents into the RAG vector store."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.chunker import chunk_document
from src.ingestion.loader import load_directory, load_file
from src.retrieval.retriever import VectorRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", help="Directory of documents to ingest")
    group.add_argument("--file", help="Single file to ingest")
    args = parser.parse_args()

    retriever = VectorRetriever()

    if args.dir:
        docs = list(load_directory(args.dir))
    else:
        docs = [load_file(args.file)]

    total_chunks = 0
    for doc in docs:
        chunks = list(chunk_document(doc))
        if not chunks:
            logger.warning("No chunks from %s", doc["source"])
            continue
        indexed = retriever.index(chunks)
        total_chunks += indexed
        logger.info("%s — %d chunks", doc["filename"], indexed)

    logger.info("Done. %d total chunks indexed across %d documents.", total_chunks, len(docs))
    logger.info("Index size: %d chunks", retriever.document_count)


if __name__ == "__main__":
    main()
