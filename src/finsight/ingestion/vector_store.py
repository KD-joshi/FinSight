"""Pinecone Cloud vector store for document embeddings.

Manages the full lifecycle: similarity search with metadata filtering,
and document ingestion using namespaces to prevent collision.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from config.settings import settings
from src.finsight.utils.embeddings import get_embeddings

logger = logging.getLogger(__name__)

# Batch size for upsert operations
_UPSERT_BATCH_SIZE: int = 64

# ======================================================================
# LangChain Vector Store wrapper
# ======================================================================

def get_vector_store(namespace: str) -> PineconeVectorStore:
    """Return a LangChain PineconeVectorStore configured for a specific namespace.

    Parameters
    ----------
    namespace:
        The Pinecone namespace (usually the session/thread ID).

    Returns
    -------
    PineconeVectorStore
        A LangChain-compatible vector store.
    """
    if not settings.has_pinecone_config():
        raise ValueError(
            "Pinecone Cloud is not configured. "
            "Set PINECONE_API_KEY in your .env file."
        )

    embeddings = get_embeddings()
    index_name = settings.pinecone_index_name

    logger.debug("Initializing PineconeVectorStore for index: '%s', namespace: '%s'", index_name, namespace)

    store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
        pinecone_api_key=settings.pinecone_api_key
    )

    return store


# ======================================================================
# Ingestion
# ======================================================================

def ingest_documents(
    documents: list[Document],
    namespace: str,
    *,
    batch_size: int = _UPSERT_BATCH_SIZE,
) -> int:
    """Embed documents and upsert them into Pinecone.

    Parameters
    ----------
    documents:
        LangChain ``Document`` objects to ingest.
    batch_size:
        Number of documents per upsert batch.

    Returns
    -------
    int
        Total number of documents ingested.
    """
    if not documents:
        logger.warning("ingest_documents called with empty list — skipping.")
        return 0

    store = get_vector_store(namespace)

    logger.info(
        "Ingesting %d document(s) into Pinecone namespace '%s' (batch_size=%d)…",
        len(documents),
        namespace,
        batch_size,
    )

    total_ingested = 0

    try:
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            store.add_documents(batch)
            total_ingested += len(batch)

            logger.debug(
                "  Batch %d–%d ingested (%d/%d).",
                start,
                start + len(batch) - 1,
                total_ingested,
                len(documents),
            )

    except Exception as exc:
        logger.error(
            "Ingestion failed after %d/%d documents: %s",
            total_ingested,
            len(documents),
            exc,
        )
        raise RuntimeError(
            f"Document ingestion failed: {exc}"
        ) from exc

    logger.info("✓ Ingested %d document(s) into Pinecone.", total_ingested)
    return total_ingested


# ======================================================================
# Search
# ======================================================================

def similarity_search(
    query: str,
    namespace: str,
    *,
    k: int = 0,
    filter_metadata: Optional[dict[str, Any]] = None,
) -> list[Document]:
    """Search for documents similar to the query.

    Parameters
    ----------
    query:
        Natural-language query string.
    k:
        Number of results to return.
    filter_metadata:
        Optional metadata filter dict. Pinecone accepts standard dicts like:
        {"ticker": "AAPL", "filing_type": "10-K"}

    Returns
    -------
    list[Document]
        Top-*k* similar documents with scores in metadata.
    """
    k = k or settings.retrieval_top_k
    store = get_vector_store(namespace)

    logger.info(
        "Similarity search (Pinecone %s): query=%.80s…, k=%d, filter=%s",
        namespace,
        query,
        k,
        filter_metadata,
    )

    try:
        results = store.similarity_search_with_score(
            query, 
            k=k, 
            filter=filter_metadata
        )

        documents = []
        for doc, score in results:
            doc.metadata["score"] = score
            documents.append(doc)

        logger.info("Found %d result(s).", len(documents))
        return documents

    except Exception as exc:
        logger.error("Similarity search failed: %s", exc)
        raise RuntimeError(f"Search failed: {exc}") from exc
