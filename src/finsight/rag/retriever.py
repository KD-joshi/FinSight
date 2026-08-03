"""Hybrid retriever combining dense (Pinecone) and sparse (BM25) search.

Uses Reciprocal Rank Fusion (RRF) to merge results from both retrievers,
providing robust retrieval that captures both semantic similarity and
keyword relevance — critical for financial documents where exact terms
(ticker symbols, metric names, dollar amounts) matter as much as meaning.

Note:
    BM25 sparse retrieval is currently disabled when running against
    Pinecone Serverless (initialised with ``documents=[]``). Only
    dense vector search is active.

References:
    - RRF Paper: Cormack, Clarke, Buettcher (2009)
      "Reciprocal Rank Fusion outperforms Condorcet and individual
       Rank Learning Methods"
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines vector similarity search with BM25 keyword search.

    Uses Reciprocal Rank Fusion (RRF) to merge ranked results from two
    fundamentally different retrieval strategies:

    - **Dense retrieval** (Pinecone): Captures semantic similarity via embeddings.
      Good for paraphrased questions, conceptual queries.
    - **Sparse retrieval** (BM25): Captures lexical overlap via term frequency.
      Good for exact matches, ticker symbols, specific metric names.

    RRF score for document d across N rankings:
        score(d) = Σ_{r ∈ rankings} 1 / (k + rank(d, r))

    where k is a constant (default 60) that dampens the influence of
    high-ranking documents, making the fusion more robust.

    Attributes:
        documents: All ingested documents (used for BM25 index).
        k: RRF fusion constant. Higher values = more equal weighting.
        bm25_index: Pre-built BM25Okapi index over document corpus.
        tokenized_docs: Tokenized document texts for BM25 lookup.
    """

    def __init__(
        self,
        documents: list[Document],
        k: int = 60,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            documents: List of all documents to index for BM25 search.
            k: RRF fusion constant. Default 60 per the original paper.
        """
        self.documents = documents
        self.k = k

        # Build BM25 index at init time
        self.tokenized_docs: list[list[str]] = []
        self.bm25_index: BM25Okapi | None = None
        self._build_bm25_index(documents)

        logger.info(
            "HybridRetriever initialized with %d documents, RRF k=%d",
            len(documents),
            k,
        )

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_bm25_index(self, documents: list[Document]) -> None:
        """Build BM25 index from document texts.

        Tokenizes each document using a simple whitespace + punctuation
        tokenizer with lowercasing. Financial terms like ``10-K``,
        ``P/E``, and ticker symbols are preserved.

        Args:
            documents: List of LangChain Document objects to index.
        """
        if not documents:
            logger.warning("No documents provided for BM25 index — sparse retrieval disabled.")
            return

        self.tokenized_docs = [
            self._tokenize(doc.page_content) for doc in documents
        ]
        self.bm25_index = BM25Okapi(self.tokenized_docs)

        logger.info("BM25 index built over %d documents.", len(documents))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25.

        Uses a regex tokenizer that preserves financial terms like
        ``10-K``, ``10-Q``, ``$1.5B``, ``P/E``, ``EPS`` while
        lowercasing for case-insensitive matching.

        Args:
            text: Raw document text.

        Returns:
            List of lowercase tokens.
        """
        # Match word characters, digits, hyphens, slashes, dollar signs,
        # and periods (for numbers like 1.5B) as single tokens
        tokens = re.findall(r"[\w$][\w\-/.$]*", text.lower())
        return tokens

    # ------------------------------------------------------------------
    # Individual retrieval strategies
    # ------------------------------------------------------------------

    def _vector_search(
        self,
        query: str,
        namespace: str,
        top_k: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Dense retrieval using the Pinecone vector store.

        Args:
            query: User question or search query.
            namespace: Session-specific Pinecone namespace.
            top_k: Maximum number of documents to return.
            filter_dict: Optional metadata filter passed to Pinecone.
                Example: ``{"company": "AAPL", "filing_type": "10-K"}``

        Returns:
            Ranked list of Document objects, most relevant first.
        """
        from finsight.ingestion.vector_store import get_vector_store
        try:
            vector_store = get_vector_store(namespace)
            kwargs: dict[str, Any] = {"k": top_k}
            if filter_dict:
                kwargs["filter"] = filter_dict

            results = vector_store.similarity_search(query, **kwargs)
            logger.debug(
                "Vector search returned %d results for query: %.80s",
                len(results),
                query,
            )
            return results

        except Exception:
            logger.exception("Vector search failed — returning empty results.")
            return []

    def _bm25_search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Sparse retrieval using BM25 keyword matching.

        Args:
            query: User question or search query.
            top_k: Maximum number of documents to return.
            filter_dict: Optional metadata filter applied post-retrieval.

        Returns:
            Ranked list of Document objects, most relevant first.
        """
        if self.bm25_index is None:
            logger.warning("BM25 index not available — skipping sparse retrieval.")
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Get BM25 scores for all documents
        scores: np.ndarray = self.bm25_index.get_scores(query_tokens)

        # Rank by score (descending)
        ranked_indices = np.argsort(scores)[::-1]

        results: list[Document] = []
        for idx in ranked_indices:
            if len(results) >= top_k:
                break

            doc = self.documents[idx]

            # Apply metadata filter if provided
            if filter_dict and not self._matches_filter(doc, filter_dict):
                continue

            # Skip zero-score documents (no term overlap at all)
            if scores[idx] <= 0:
                break

            results.append(doc)

        logger.debug(
            "BM25 search returned %d results for query: %.80s",
            len(results),
            query,
        )
        return results

    @staticmethod
    def _matches_filter(doc: Document, filter_dict: dict[str, Any]) -> bool:
        """Check if a document's metadata matches the given filter.

        Supports simple equality matching on metadata fields.

        Args:
            doc: Document to check.
            filter_dict: Key-value pairs to match against metadata.

        Returns:
            True if all filter conditions are satisfied.
        """
        # Handle empty filters
        if not filter_dict:
            return True
            
        def _evaluate_condition(condition: dict) -> bool:
            for k, v in condition.items():
                if k == "$or":
                    if not isinstance(v, list):
                        return False
                    # Return True if ANY condition in the $or list is True
                    if not any(_evaluate_condition(sub_cond) for sub_cond in v):
                        return False
                elif k == "$and":
                    if not isinstance(v, list):
                        return False
                    # Return True only if ALL conditions in the $and list are True
                    if not all(_evaluate_condition(sub_cond) for sub_cond in v):
                        return False
                else:
                    doc_value = doc.metadata.get(k)
                    
                    # Handle nested operators like {"$exists": False}
                    if isinstance(v, dict):
                        for op, op_val in v.items():
                            if op == "$exists":
                                exists = doc_value is not None
                                if exists != op_val:
                                    return False
                            elif op == "$eq":
                                if doc_value != op_val:
                                    return False
                            elif op == "$ne":
                                if doc_value == op_val:
                                    return False
                            elif op == "$in":
                                if doc_value not in op_val:
                                    return False
                            else:
                                # Unsupported operator, be safe and ignore or fail?
                                # Let's just assume it doesn't match
                                return False
                    else:
                        # Direct equality match
                        if doc_value is None or doc_value != v:
                            return False
            return True
            
        return _evaluate_condition(filter_dict)

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        rankings: list[list[Document]],
        k: int = 60,
    ) -> list[Document]:
        """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

        For each document that appears in any ranking, compute:
            RRF_score(d) = Σ_{r ∈ rankings} 1 / (k + rank(d, r))

        where rank(d, r) is the 1-based position of document d in ranking r
        (documents not present in a ranking contribute 0 to the sum).

        Args:
            rankings: List of ranked document lists from different retrievers.
                Each inner list is ordered from most to least relevant.
            k: Fusion constant (default 60). Controls the impact of rank
                position on the final score.

        Returns:
            Single merged list of Document objects, sorted by descending
            RRF score. Duplicates across rankings are unified.
        """
        # Map document content hash → (Document, cumulative RRF score)
        rrf_scores: dict[str, tuple[Document, float]] = {}

        for ranking in rankings:
            for rank_position, doc in enumerate(ranking, start=1):
                # Use page_content + source as dedup key to handle docs
                # that appear in multiple rankings
                doc_key = self._doc_key(doc)
                rrf_contribution = 1.0 / (k + rank_position)

                if doc_key in rrf_scores:
                    existing_doc, existing_score = rrf_scores[doc_key]
                    rrf_scores[doc_key] = (existing_doc, existing_score + rrf_contribution)
                else:
                    rrf_scores[doc_key] = (doc, rrf_contribution)

        # Sort by RRF score descending
        sorted_docs = sorted(
            rrf_scores.values(),
            key=lambda item: item[1],
            reverse=True,
        )

        # Attach RRF score to metadata for downstream inspection
        result: list[Document] = []
        for doc, score in sorted_docs:
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rrf_score": round(score, 6)},
            )
            result.append(doc_copy)

        return result

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """Generate a stable deduplication key for a document.

        Uses content hash + source metadata to identify unique documents.

        Args:
            doc: Document to generate key for.

        Returns:
            String key for deduplication.
        """
        source = doc.metadata.get("source", "")
        chunk_id = doc.metadata.get("chunk_id", "")
        # Use first 200 chars of content + source for a stable key
        content_prefix = doc.page_content[:200]
        return f"{source}::{chunk_id}::{hash(content_prefix)}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        namespace: str = "finsight",
        top_k: int = 5,
        filter: dict[str, Any] | None = None,  # noqa: A002
        vector_weight: int = 10,
        bm25_weight: int = 10,
    ) -> list[Document]:
        """Hybrid retrieval with Reciprocal Rank Fusion.

        Runs both dense (vector) and sparse (BM25) retrieval in sequence,
        then fuses results using RRF. The ``vector_weight`` and
        ``bm25_weight`` control how many candidates each retriever
        contributes to the fusion pool.

        Args:
            query: User question or search query.
            top_k: Number of final documents to return after fusion.
            filter: Optional metadata filter.
                Example: ``{"company": "AAPL", "year": "2024"}``
            vector_weight: Number of candidates from vector search
                (fetched before fusion). Default 10.
            bm25_weight: Number of candidates from BM25 search
                (fetched before fusion). Default 10.

        Returns:
            Ranked list of ``top_k`` documents after RRF fusion.
            Each document's metadata includes an ``rrf_score`` field.

        Example:
            >>> retriever = HybridRetriever(vector_store, documents)
            >>> results = retriever.retrieve(
            ...     "What was Apple's revenue in 2024?",
            ...     top_k=5,
            ...     filter={"company": "AAPL"},
            ... )
            >>> for doc in results:
            ...     print(doc.metadata["rrf_score"], doc.page_content[:80])
        """
        logger.info("Hybrid retrieval for query: %.120s", query)

        # 1. Run both retrieval strategies
        vector_results = self._vector_search(query, namespace, top_k=vector_weight, filter_dict=filter)
        bm25_results = self._bm25_search(query, top_k=bm25_weight, filter_dict=filter)

        logger.info(
            "Retrieved %d vector + %d BM25 candidates",
            len(vector_results),
            len(bm25_results),
        )

        # 2. Handle edge cases
        if not vector_results and not bm25_results:
            logger.warning("Both retrievers returned empty results.")
            return []

        if not vector_results:
            return bm25_results[:top_k]

        if not bm25_results:
            return vector_results[:top_k]

        # 3. Fuse with RRF
        fused = self._reciprocal_rank_fusion(
            rankings=[vector_results, bm25_results],
            k=self.k,
        )

        result = fused[:top_k]
        logger.info(
            "RRF fusion produced %d results (returning top %d)",
            len(fused),
            len(result),
        )

        return result

    def update_documents(self, new_documents: list[Document]) -> None:
        """Update the BM25 index with new documents.

        Call this after ingesting new documents into the vector store
        to keep the BM25 index in sync.

        Args:
            new_documents: Complete list of documents (replaces existing index).
        """
        self.documents = new_documents
        self._build_bm25_index(new_documents)
        logger.info("BM25 index rebuilt with %d documents.", len(new_documents))
