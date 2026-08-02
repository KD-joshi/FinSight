"""Process financial documents: parse → chunk → attach metadata.

Pipeline
--------
1. **Parse** — Convert PDF / HTML filings to Markdown using
   `docling <https://github.com/DS4SD/docling>`_, which handles
   complex tables, multi-column layouts, and nested headers.
2. **Chunk** — Split the Markdown output into overlapping chunks via
   LangChain's ``RecursiveCharacterTextSplitter``.
3. **Enrich** — Attach structured metadata (ticker, filing type, fiscal
   year, section) to each ``Document`` object.

Usage::

    from src.finsight.ingestion.document_processor import process_filing_directory

    docs = process_filing_directory(Path("data/raw/sec-edgar-filings/AAPL/10-K"))
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
_SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".htm", ".html", ".txt"}

# Regex to extract fiscal year from filing text (e.g. "fiscal year ended December 31, 2024")
_FISCAL_YEAR_PATTERN = re.compile(
    r"(?:fiscal\s+year|year)\s+ended?\s+\w+\s+\d{1,2},?\s+(\d{4})",
    re.IGNORECASE,
)


# ======================================================================
# 1. Document Parsing
# ======================================================================

def _extract_primary_doc(file_path: Path) -> Path:
    """Extract the primary 10-K/10-Q HTML/text from full-submission.txt.

    Saves it to a temporary file in the cache directory and returns the path.
    If it's not a full-submission.txt, returns the original path.
    """
    if file_path.name != "full-submission.txt":
        return file_path

    logger.info("Extracting primary document from SEC full submission: %s", file_path.name)
    content = file_path.read_text(encoding="utf-8", errors="ignore")

    # Find all <DOCUMENT> blocks
    documents = re.findall(r"<DOCUMENT>([\s\S]*?)</DOCUMENT>", content)

    primary_doc_text = ""
    for doc in documents:
        doc_type_match = re.search(r"<TYPE>(\S+)", doc)
        if doc_type_match:
            doc_type = doc_type_match.group(1).upper()
            if doc_type in ("10-K", "10-Q"):
                # Extract text between <TEXT> and </TEXT>
                text_match = re.search(r"<TEXT>([\s\S]*?)</TEXT>", doc, re.IGNORECASE)
                if text_match:
                    primary_doc_text = text_match.group(1).strip()
                    logger.info("Found primary %s document block (size: %d chars)", doc_type, len(primary_doc_text))
                    break

    if not primary_doc_text:
        logger.warning("Could not find primary 10-K/10-Q block in %s, falling back to full file.", file_path.name)
        return file_path

    # Write to a temporary file in cache directory
    cache_dir = settings.cache_dir
    # Create a unique filename based on parent folder names to avoid collisions
    unique_name = f"{file_path.parents[2].name}_{file_path.parents[1].name}_{file_path.parent.name}_primary.html"
    temp_path = cache_dir / unique_name
    temp_path.write_text(primary_doc_text, encoding="utf-8")
    return temp_path


def parse_document(file_path: Path) -> str:
    """Parse a financial document (PDF/HTML) into Markdown text.

    Uses ``docling`` for high-fidelity conversion that preserves table
    structure and section hierarchy.

    Parameters
    ----------
    file_path:
        Path to the document file.

    Returns
    -------
    str
        The document content as Markdown.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file type is unsupported.
    RuntimeError
        If parsing fails.
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    # Intercept full-submission.txt and extract primary document to improve speed and quality
    working_file_path = file_path
    if file_path.name == "full-submission.txt":
        try:
            working_file_path = _extract_primary_doc(file_path)
        except Exception as exc:
            logger.warning("Failed to extract primary document, using original: %s", exc)

    if working_file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{working_file_path.suffix}'. "
            f"Supported: {_SUPPORTED_EXTENSIONS}"
        )

    logger.info("Parsing document: %s", working_file_path)

    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(working_file_path))
        markdown_text = result.document.export_to_markdown()

        logger.info(
            "Parsed %s → %d characters of Markdown.",
            working_file_path.name,
            len(markdown_text),
        )
        return markdown_text

    except ImportError:
        logger.warning(
            "docling not installed — falling back to plain text extraction "
            "for %s. Install docling for high-fidelity parsing.",
            working_file_path.name,
        )
        return _fallback_parse(working_file_path)

    except Exception as exc:
        logger.error("Failed to parse %s: %s", working_file_path, exc)
        raise RuntimeError(f"Document parsing failed for {working_file_path}: {exc}") from exc


def _fallback_parse(file_path: Path) -> str:
    """Simple fallback: read file as plain text."""
    try:
        return file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise RuntimeError(f"Fallback parsing failed for {file_path}: {exc}") from exc


# ======================================================================
# 2. Chunking
# ======================================================================

def chunk_document(
    text: str,
    metadata: Optional[dict[str, Any]] = None,
    *,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Document]:
    """Split text into overlapping chunks with attached metadata.

    Parameters
    ----------
    text:
        The full document text (typically Markdown).
    metadata:
        Base metadata dict to attach to every chunk.  Each chunk also
        gets ``chunk_index`` and ``total_chunks`` fields.
    chunk_size:
        Target chunk size in characters (default from settings).
    chunk_overlap:
        Overlap between chunks (default from settings).

    Returns
    -------
    list[Document]
        LangChain ``Document`` objects with ``.page_content`` and
        ``.metadata``.
    """
    if not text or not text.strip():
        logger.warning("chunk_document called with empty text — returning [].")
        return []

    metadata = metadata or {}
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ",   # Markdown H2
            "\n### ",  # Markdown H3
            "\n#### ",
            "\n\n",    # Paragraph break
            "\n",      # Line break
            ". ",      # Sentence
            " ",       # Word
            "",        # Character
        ],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_text(text)

    documents: list[Document] = []
    for idx, chunk in enumerate(chunks):
        chunk_metadata = {
            **metadata,
            "chunk_index": idx,
            "total_chunks": len(chunks),
        }
        documents.append(Document(page_content=chunk, metadata=chunk_metadata))

    logger.info(
        "Chunked document into %d pieces (size=%d, overlap=%d). "
        "Metadata keys: %s",
        len(documents),
        chunk_size,
        chunk_overlap,
        list(metadata.keys()),
    )
    return documents


# ======================================================================
# 3. Metadata extraction helpers
# ======================================================================

def _extract_metadata_from_path(file_path: Path) -> dict[str, Any]:
    """Infer metadata from the filing directory structure.

    Expects the sec-edgar-downloader layout::

        data/raw/sec-edgar-filings/<TICKER>/<FILING_TYPE>/<ACCESSION>/…

    Parameters
    ----------
    file_path:
        Path to the filing document.

    Returns
    -------
    dict
        Metadata dict with ``ticker``, ``filing_type``, ``source_file``,
        and ``accession_number`` when inferrable.
    """
    parts = file_path.resolve().parts
    metadata: dict[str, Any] = {"source_file": str(file_path)}

    # Walk up the path looking for "sec-edgar-filings"
    try:
        idx = parts.index("sec-edgar-filings")
        if idx + 1 < len(parts):
            metadata["ticker"] = parts[idx + 1]
        if idx + 2 < len(parts):
            metadata["filing_type"] = parts[idx + 2]
        if idx + 3 < len(parts):
            metadata["accession_number"] = parts[idx + 3]
    except ValueError:
        # Not in the expected directory structure — that's okay
        logger.debug(
            "Could not infer SEC metadata from path: %s", file_path
        )

    return metadata


def _extract_fiscal_year(text: str) -> Optional[str]:
    """Try to extract fiscal year from document text."""
    match = _FISCAL_YEAR_PATTERN.search(text[:5000])  # Search near the top
    if match:
        return match.group(1)
    return None


# ======================================================================
# 4. Directory-level processing
# ======================================================================

def process_filing_directory(
    dir_path: Path,
    *,
    extra_metadata: Optional[dict[str, Any]] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Document]:
    """Process all supported files in a directory into chunked Documents.

    Parameters
    ----------
    dir_path:
        Path to a directory containing filing files (e.g.
        ``data/raw/sec-edgar-filings/AAPL/10-K``).
    extra_metadata:
        Additional metadata to merge into every chunk.
    chunk_size:
        Override default chunk size.
    chunk_overlap:
        Override default chunk overlap.

    Returns
    -------
    list[Document]
        All chunked documents from every supported file in the
        directory.
    """
    dir_path = Path(dir_path).resolve()

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    # Discover supported files
    files = sorted(
        f
        for f in dir_path.rglob("*")
        if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS
    )

    if not files:
        logger.warning("No supported files found in %s", dir_path)
        return []

    logger.info(
        "Processing %d file(s) from %s", len(files), dir_path
    )

    all_documents: list[Document] = []

    for file_path in files:
        try:
            # Parse
            text = parse_document(file_path)

            # Build metadata
            metadata = _extract_metadata_from_path(file_path)
            fiscal_year = _extract_fiscal_year(text)
            if fiscal_year:
                metadata["fiscal_year"] = fiscal_year
            if extra_metadata:
                metadata.update(extra_metadata)

            # Chunk
            docs = chunk_document(
                text,
                metadata=metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            all_documents.extend(docs)

            logger.info(
                "  ✓ %s → %d chunks", file_path.name, len(docs)
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("  ✗ Failed to process %s: %s", file_path, exc)

    logger.info(
        "Directory processing complete: %d total chunks from %d file(s).",
        len(all_documents),
        len(files),
    )
    return all_documents


def process_all_tickers(
    raw_dir: Optional[Path] = None,
    *,
    tickers: Optional[list[str]] = None,
    filing_types: Optional[list[str]] = None,
) -> list[Document]:
    """Process filings for all (or specified) tickers in the raw data dir.

    Parameters
    ----------
    raw_dir:
        Root directory with SEC filings (default: ``data/raw/``).
    tickers:
        Optional filter — only process these tickers.
    filing_types:
        Optional filter — only process these filing types.

    Returns
    -------
    list[Document]
        Chunked documents across all matching filings.
    """
    raw_dir = raw_dir or settings.raw_data_dir
    sec_root = raw_dir / "sec-edgar-filings"

    if not sec_root.exists():
        logger.warning("No SEC filings directory found at %s", sec_root)
        return []

    all_docs: list[Document] = []

    for ticker_dir in sorted(sec_root.iterdir()):
        if not ticker_dir.is_dir():
            continue
        if tickers and ticker_dir.name not in [t.upper() for t in tickers]:
            continue

        for type_dir in sorted(ticker_dir.iterdir()):
            if not type_dir.is_dir():
                continue
            if filing_types and type_dir.name not in filing_types:
                continue

            docs = process_filing_directory(type_dir)
            all_docs.extend(docs)

    logger.info(
        "Processed all tickers: %d total chunks.", len(all_docs)
    )
    return all_docs
