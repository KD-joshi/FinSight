"""LangGraph-based Agentic RAG pipeline for financial document analysis.

Implements Corrective RAG (CRAG) with self-grading, query rewriting,
and optional query decomposition for complex multi-hop questions.

Graph Flow::

    START
      │
      ▼
    route_query ──► complexity?
      │                       │
      │ simple                │ complex
      ▼                       ▼
    retrieve              plan_query
      │                       │
      │                       ▼
      │                   retrieve (per sub-query)
      │                       │
      ▼◄──────────────────────┘
    grade_documents
      │
      ▼
    enough relevant docs?
      │              │
      │ yes          │ no (retry ≤ 3)
      ▼              ▼
    generate     rewrite_query ──► retrieve ──► grade_documents
      │
      ▼
     END

Key design decisions:
- Max 3 retrieval retries to avoid infinite loops
- Documents below relevance threshold are filtered out
- RRF scores from hybrid retriever are preserved in metadata
- Each node is a pure function over AgentState for testability
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from finsight.rag.chains import (
    build_generator_chain,
    build_planner_chain,
    build_rewriter_chain,
    build_router_chain,
    build_query_analyzer_chain,
)
from finsight.rag.retriever import HybridRetriever
from finsight.utils.query_cache import query_cache

logger = logging.getLogger(__name__)


# ======================================================================
# Agent State
# ======================================================================

class AgentState(TypedDict, total=False):
    """State maintained across the agentic RAG loop.

    All node functions read from and write to this state dict.
    Fields use ``total=False`` so nodes only need to set the fields
    they modify.

    Attributes:
        question: Original user question.
        sub_queries: Decomposed sub-queries (populated for complex queries).
        current_query: The query currently being processed (may be rewritten).
        documents: Retrieved document chunks after grading/filtering.
        all_documents: Accumulated documents across sub-queries.
        generation: Final generated answer text.
        relevance_scores: Grading results per document from the grader.
        retry_count: Number of retrieval retries so far.
        max_retries: Maximum allowed retries (default 3).
        route: Query classification — ``"simple"`` or ``"complex"``.
        error: Error message if something went wrong.
    """

    question: str
    sub_queries: list[str]
    current_query: str
    documents: list[Document]
    all_documents: list[Document]
    generation: str
    relevance_scores: list[dict[str, Any]]
    retry_count: int
    max_retries: int
    route: str
    error: str


# ======================================================================
# Node Functions
# ======================================================================

def _make_route_query(router_chain):
    """Create the route_query node function.

    Classifies the query as simple or complex using the router chain.
    """

    def route_query(state: AgentState) -> AgentState:
        """Classify query complexity and set the routing decision."""
        question = state["question"]
        logger.info("Routing query: %.120s", question)
        
        # Check cache first
        cached_answer = query_cache.get(question)
        if cached_answer:
            logger.info("Found cached answer for query. Skipping pipeline.")
            return {
                "route": "cached",
                "current_query": question,
                "generation": cached_answer,
                "retry_count": 0,
            }

        try:
            result = router_chain.invoke({"question": question})
            route = result.get("complexity", "simple")
            logger.info(
                "Query routed as '%s': %s",
                route,
                result.get("reasoning", ""),
            )
        except Exception:
            logger.exception("Router chain failed — defaulting to 'simple'.")
            route = "simple"

        return {
            "route": route,
            "current_query": question,
            "retry_count": 0,
            "max_retries": state.get("max_retries", 3),
            "all_documents": [],
            "documents": [],
            "generation": "",
        }

    return route_query


def _make_plan_query(planner_chain):
    """Create the plan_query node function.

    Decomposes complex queries into sub-queries.
    """

    def plan_query(state: AgentState) -> AgentState:
        """Decompose a complex query into simpler sub-queries."""
        question = state["question"]
        logger.info("Planning sub-queries for: %.120s", question)

        try:
            result = planner_chain.invoke({"question": question})
            sub_queries = result.get("sub_queries", [question])

            if not sub_queries:
                sub_queries = [question]

            logger.info("Decomposed into %d sub-queries.", len(sub_queries))
            for i, sq in enumerate(sub_queries, 1):
                logger.debug("  Sub-query %d: %s", i, sq)

        except Exception:
            logger.exception("Planner chain failed — using original query.")
            sub_queries = [question]

        return {
            "sub_queries": sub_queries,
            "current_query": sub_queries[0],
            "retry_count": 0,
        }

    return plan_query


def _make_retrieve(retriever: HybridRetriever, analyzer_chain):
    """Create the retrieve node function.

    For simple queries: retrieves documents for current_query.
    For complex queries: retrieves documents for each sub-query
    and accumulates results.
    """

    def retrieve(state: AgentState) -> AgentState:
        """Retrieve documents using hybrid search."""
        route = state.get("route", "simple")
        all_documents = list(state.get("all_documents", []))

        def _extract_filters(q: str) -> dict[str, Any] | None:
            try:
                res = analyzer_chain.invoke({"question": q})
                filters = {k: v for k, v in res.items() if v is not None}
                return filters if filters else None
            except Exception:
                return None

        if route == "complex" and state.get("sub_queries"):
            # Complex path: retrieve for all sub-queries
            sub_queries = state["sub_queries"]
            documents: list[Document] = []

            for sq in sub_queries:
                filters = _extract_filters(sq)
                logger.info("Retrieving for sub-query: %.120s with filters %s", sq, filters)
                results = retriever.retrieve(sq, top_k=5, filter=filters)
                documents.extend(results)

            # Deduplicate across sub-queries
            documents = _deduplicate_documents(documents)
            all_documents.extend(documents)

            logger.info(
                "Complex retrieval: %d total documents across %d sub-queries.",
                len(documents),
                len(sub_queries),
            )
        else:
            # Simple path or retry: retrieve for current_query
            current_query = state.get("current_query", state["question"])
            filters = _extract_filters(current_query)
            logger.info("Retrieving for query: %.120s with filters %s", current_query, filters)
            documents = retriever.retrieve(current_query, top_k=5, filter=filters)

        return {
            "documents": documents,
            "all_documents": all_documents if route == "complex" else documents,
        }

    return retrieve


def _make_rerank_documents():
    """Create the rerank_documents node function.

    Reranks retrieved documents using a local cross-encoder (flashrank).
    """
    from flashrank import Ranker, RerankRequest
    
    # Initialize ranker once (it caches the model)
    ranker = Ranker()

    def rerank_documents(state: AgentState) -> AgentState:
        """Rerank retrieved documents and keep the top 4."""
        question = state.get("current_query", state["question"])
        documents = state.get("documents", [])

        if not documents:
            return {"documents": [], "relevance_scores": []}
            
        logger.info("Reranking %d documents.", len(documents))
        
        passages = []
        for i, doc in enumerate(documents):
            passages.append({
                "id": i,
                "text": doc.page_content,
                "meta": doc.metadata
            })
            
        request = RerankRequest(query=question, passages=passages)
        
        try:
            results = ranker.rerank(request)
        except Exception:
            logger.exception("Flashrank failed. Falling back to original order.")
            results = [{"id": i, "score": 1.0} for i in range(len(documents))]
        
        relevant_docs: list[Document] = []
        relevance_scores: list[dict[str, Any]] = []
        
        # Take Top 4
        top_n = min(4, len(results))
        for res in results[:top_n]:
            idx = res.get("id")
            score = res.get("score", 0.0)
            if idx is not None and 0 <= idx < len(documents):
                doc = documents[idx]
                relevant_docs.append(doc)
                relevance_scores.append({
                    "doc_index": idx,
                    "relevant": True,
                    "reasoning": f"Flashrank score: {float(score):.4f}",
                    "source": doc.metadata.get("source", "unknown"),
                })
                logger.debug("  Doc %d: Score %.4f", idx, score)
                
        logger.info(
            "Reranking complete: kept top %d documents.",
            len(relevant_docs)
        )

        return {
            "documents": relevant_docs,
            "relevance_scores": relevance_scores,
        }

    return rerank_documents


def _make_rewrite_query(rewriter_chain):
    """Create the rewrite_query node function.

    Rewrites the current query to improve retrieval on the next attempt.
    """

    def rewrite_query(state: AgentState) -> AgentState:
        """Rewrite the query to improve retrieval."""
        current_query = state.get("current_query", state["question"])
        retry_count = state.get("retry_count", 0)

        # Build a summary of the failed context
        failed_docs = state.get("documents", [])
        failed_context = "\n---\n".join(
            doc.page_content[:500] for doc in failed_docs[:3]
        ) if failed_docs else "No documents were retrieved."

        logger.info(
            "Rewriting query (attempt %d): %.120s",
            retry_count + 1,
            current_query,
        )

        try:
            rewritten = rewriter_chain.invoke({
                "question": current_query,
                "failed_context": failed_context,
            })
            rewritten = rewritten.strip()
            logger.info("Rewritten query: %.120s", rewritten)
        except Exception:
            logger.exception("Rewriter chain failed — using original query.")
            rewritten = current_query

        return {
            "current_query": rewritten,
            "retry_count": retry_count + 1,
        }

    return rewrite_query


def _make_generate(generator_chain):
    """Create the generate node function.

    Generates the final answer from relevant documents.
    """

    def generate(state: AgentState) -> AgentState:
        """Generate a cited answer from relevant documents."""
        question = state["question"]
        documents = state.get("documents", [])
        all_documents = state.get("all_documents", documents)

        # Use all accumulated documents for complex queries
        docs_for_generation = all_documents if all_documents else documents

        if not docs_for_generation:
            logger.warning("No documents available for generation.")
            return {
                "generation": (
                    "I don't have enough information to answer this question. "
                    "No relevant documents were found in the knowledge base. "
                    "Please try rephrasing your question or ensure the relevant "
                    "SEC filings have been ingested."
                ),
            }

        # Format documents with source metadata for citation
        context_parts: list[str] = []
        for i, doc in enumerate(docs_for_generation, 1):
            source = doc.metadata.get("source", f"Document {i}")
            filing_type = doc.metadata.get("filing_type", "")
            company = doc.metadata.get("company", "")
            section = doc.metadata.get("section", "")

            header_parts = [f"[Document {i}]"]
            if company:
                header_parts.append(f"Company: {company}")
            if filing_type:
                header_parts.append(f"Filing: {filing_type}")
            if section:
                header_parts.append(f"Section: {section}")
            header_parts.append(f"Source: {source}")

            header = " | ".join(header_parts)
            context_parts.append(f"{header}\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)

        logger.info(
            "Generating answer from %d documents for: %.120s",
            len(docs_for_generation),
            question,
        )

        try:
            generation = generator_chain.invoke({
                "question": question,
                "context": context,
            })
            # Save successful generation to cache
            query_cache.set(question, generation)
        except Exception:
            logger.exception("Generator chain failed.")
            generation = (
                "I encountered an error while generating the answer. "
                "Please try again."
            )

        return {"generation": generation}

    return generate


# ======================================================================
# Edge / Conditional Functions
# ======================================================================

def _should_retrieve_or_plan(state: AgentState) -> Literal["retrieve", "plan_query", "END"]:
    """Route based on query complexity classification or cache hit.

    Returns:
        ``"plan_query"`` for complex queries, ``"retrieve"`` for simple ones, or ``"END"`` if cached.
    """
    if state.get("generation"):
        logger.info("Answer retrieved from cache. Ending pipeline.")
        return "END"
        
    route = state.get("route", "simple")
    if route == "complex":
        logger.info("Complex query → planning sub-queries.")
        return "plan_query"
    logger.info("Simple query → direct retrieval.")
    return "retrieve"


def _should_generate_or_rewrite(state: AgentState) -> Literal["generate", "rewrite_query"]:
    """Decide whether to generate an answer or rewrite the query.

    Generates if:
    - We have at least 1 relevant document, OR
    - We've exhausted our retry budget.

    Rewrites if:
    - No relevant documents found AND retries remain.

    Returns:
        ``"generate"`` or ``"rewrite_query"``.
    """
    documents = state.get("documents", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if documents:
        logger.info(
            "Found %d relevant documents → generating answer.",
            len(documents),
        )
        return "generate"

    if retry_count >= max_retries:
        logger.warning(
            "No relevant docs after %d retries → generating with best effort.",
            retry_count,
        )
        return "generate"

    logger.info(
        "No relevant docs (retry %d/%d) → rewriting query.",
        retry_count,
        max_retries,
    )
    return "rewrite_query"


# ======================================================================
# Utility functions
# ======================================================================

def _deduplicate_documents(documents: list[Document]) -> list[Document]:
    """Remove duplicate documents based on content hash.

    Args:
        documents: List of documents, potentially with duplicates.

    Returns:
        Deduplicated list preserving original order.
    """
    seen: set[int] = set()
    unique: list[Document] = []

    for doc in documents:
        content_hash = hash(doc.page_content[:300])
        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(doc)

    if len(unique) < len(documents):
        logger.debug(
            "Deduplicated %d → %d documents.",
            len(documents),
            len(unique),
        )

    return unique


# ======================================================================
# Graph Builder
# ======================================================================

def build_rag_agent(
    retriever: HybridRetriever,
    llm: BaseChatModel,
    fallback_llm: BaseChatModel | None = None,
    max_retries: int = 3,
) -> StateGraph:
    """Build and compile the Corrective RAG agent graph.

    Creates a LangGraph ``StateGraph`` with the following nodes:
    - ``route_query``: Classify query complexity
    - ``plan_query``: Decompose complex queries into sub-queries
    - ``retrieve``: Hybrid retrieval (vector + BM25)
    - ``rerank_documents``: Rerank documents using Flashrank
    - ``rewrite_query``: Rewrite query on failed retrieval
    - ``generate``: Generate a cited answer

    Args:
        retriever: HybridRetriever instance for document retrieval.
        llm: Primary LLM (e.g., Groq Llama 3.1 70B).
        fallback_llm: Optional fallback LLM (e.g., Gemini Flash).
            Used for grading/routing to reduce primary LLM load.
            If None, the primary LLM is used for everything.
        max_retries: Maximum number of query rewrite + re-retrieve cycles.

    Returns:
        Compiled LangGraph ``StateGraph`` ready for invocation.

    Example:
        >>> graph = build_rag_agent(retriever, llm)
        >>> result = graph.invoke({
        ...     "question": "What was Apple's revenue in 2024?",
        ...     "max_retries": 3,
        ... })
        >>> print(result["generation"])
    """
    # Use fallback LLM for cheaper operations (grading, routing)
    # and primary LLM for generation/planning
    grading_llm = fallback_llm or llm
    routing_llm = fallback_llm or llm

    # Build individual chains
    router_chain = build_router_chain(routing_llm)
    planner_chain = build_planner_chain(llm)
    rewriter_chain = build_rewriter_chain(llm)
    generator_chain = build_generator_chain(llm)
    analyzer_chain = build_query_analyzer_chain(routing_llm)

    # Create node functions (closures over chains + retriever)
    route_query = _make_route_query(router_chain)
    plan_query = _make_plan_query(planner_chain)
    retrieve = _make_retrieve(retriever, analyzer_chain)
    rerank_documents = _make_rerank_documents()
    rewrite_query = _make_rewrite_query(rewriter_chain)
    generate = _make_generate(generator_chain)

    # ---- Build the graph ----
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("route_query", route_query)
    workflow.add_node("plan_query", plan_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank_documents", rerank_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)

    # Set entry point
    workflow.set_entry_point("route_query")

    # Add conditional edge: route_query → retrieve OR plan_query OR END
    workflow.add_conditional_edges(
        "route_query",
        _should_retrieve_or_plan,
        {
            "retrieve": "retrieve",
            "plan_query": "plan_query",
            "END": END,
        },
    )

    # plan_query → retrieve
    workflow.add_edge("plan_query", "retrieve")

    # retrieve → rerank_documents
    workflow.add_edge("retrieve", "rerank_documents")

    # Conditional edge: rerank_documents → generate OR rewrite_query
    workflow.add_conditional_edges(
        "rerank_documents",
        _should_generate_or_rewrite,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )

    # rewrite_query → retrieve (retry loop)
    workflow.add_edge("rewrite_query", "retrieve")

    # generate → END
    workflow.add_edge("generate", END)

    # Compile with in-memory checkpointer for conversation state
    checkpointer = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=checkpointer)

    logger.info("RAG agent graph compiled successfully (max_retries=%d).", max_retries)

    return compiled_graph
