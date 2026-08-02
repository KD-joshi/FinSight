"""Ingestion CLI for FinSight.

Processes downloaded SEC filings and uploads them to the Pinecone vector database.

Usage:
    python -m src.finsight.ingestion.ingest
    python -m src.finsight.ingestion.ingest --tickers AAPL TSLA
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console
from rich.logging import RichHandler

from config.settings import settings
from src.finsight.ingestion.document_processor import process_all_tickers
from src.finsight.ingestion.vector_store import ingest_documents

console = Console()
logger = logging.getLogger("finsight.ingest")


def _setup_logging(verbose: bool = False) -> None:
    """Configure structured logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_path=False,
            )
        ],
    )
    # Quiet down external packages
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("pinecone").setLevel(logging.WARNING)


def run_ingestion(
    *,
    tickers: Optional[list[str]] = None,
    filing_types: Optional[list[str]] = None,
) -> None:
    """Run the document processing and vector ingestion pipeline."""
    target_index = settings.pinecone_index_name
    console.print(
        f"[bold cyan]🚀 FinSight Ingestion Pipeline — Index: {target_index} (Namespace: finsight)[/bold cyan]\n"
    )

    # 1. Process documents
    console.print("[info]📄 Processing documents from raw directory...[/info]")
    documents = process_all_tickers(
        raw_dir=settings.raw_data_dir,
        tickers=tickers,
        filing_types=filing_types,
    )

    if not documents:
        console.print(
            "[error]❌ No documents processed. Please verify that files exist in raw directory.[/error]"
        )
        sys.exit(1)

    console.print(
        f"[success]✓ Processed {len(documents)} document chunks successfully.[/success]"
    )

    # 2. Ingest documents into Pinecone
    console.print(f"[info]📤 Uploading {len(documents)} chunks to Pinecone Cloud...[/info]")
    try:
        total_ingested = ingest_documents(documents)
        console.print(
            f"[success]✓ Ingestion complete! {total_ingested} chunks uploaded to '{target_index}'.[/success]"
        )
    except Exception as e:
        console.print(f"[error]❌ Ingestion failed: {e}[/error]")
        sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process and ingest SEC filings into Pinecone Cloud.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Optional filter: ticker symbols to process (e.g. AAPL TSLA).",
    )
    parser.add_argument(
        "--filings",
        nargs="+",
        help="Optional filter: filing types to process (e.g. 10-K 10-Q).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    _setup_logging(verbose=args.verbose)

    run_ingestion(
        tickers=args.tickers,
        filing_types=args.filings,
    )
