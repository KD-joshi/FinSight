"""Embedding provider using HuggingFace models.

Uses the sentence-transformers library via ``langchain_community``.
This module exposes a thin wrapper that returns a configured
``HuggingFaceEmbeddings`` instance, ready for use with LangChain
vector stores and retrieval chains.

Usage::

    from src.finsight.utils.embeddings import get_embeddings

    embeddings = get_embeddings()
    vector = embeddings.embed_query("What was Apple's revenue in 2024?")
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Module-level cache ───────────────────────────────────────────────
_cached_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings(
    *,
    model: Optional[str] = None,
) -> HuggingFaceEmbeddings:
    """Return a configured HuggingFace embedding model.

    The instance is cached at module level so repeated calls reuse the
    same underlying client connection.

    Parameters
    ----------
    model:
        Override the default embedding model name.  Defaults to the
        value of ``EMBEDDING_MODEL`` in settings
        (``all-MiniLM-L6-v2``).

    Returns
    -------
    HuggingFaceEmbeddings
        A LangChain-compatible embedding model.
    """
    global _cached_embeddings  # noqa: PLW0603

    if _cached_embeddings is not None and model is None:
        return _cached_embeddings

    resolved_model = model or settings.embedding_model

    logger.info(
        "Initialising embedding model: %s (dim=%d)",
        resolved_model,
        settings.embedding_dimension,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=resolved_model,
    )

    if model is None:
        _cached_embeddings = embeddings

    return embeddings


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts and return their vector representations."""
    if not texts:
        logger.warning("embed_texts called with empty list — returning [].")
        return []

    embeddings = get_embeddings()

    try:
        logger.debug("Embedding %d text(s)…", len(texts))
        vectors = embeddings.embed_documents(texts)
        logger.info(
            "Embedded %d text(s) → vectors of dim %d.",
            len(vectors),
            len(vectors[0]) if vectors else 0,
        )
        return vectors
    except Exception as exc:
        logger.error("Embedding request failed: %s", exc)
        raise RuntimeError(f"Failed to embed {len(texts)} text(s): {exc}") from exc


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    embeddings = get_embeddings()

    try:
        logger.debug("Embedding query: %.80s…", query)
        vector = embeddings.embed_query(query)
        logger.debug("Query embedded → dim %d.", len(vector))
        return vector
    except Exception as exc:
        logger.error("Query embedding failed: %s", exc)
        raise RuntimeError(f"Failed to embed query: {exc}") from exc
