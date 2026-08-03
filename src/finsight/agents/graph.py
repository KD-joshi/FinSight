"""LangGraph-based Agentic RAG pipeline for financial document analysis.

Implements Corrective RAG (CRAG) with Flashrank reranking, query rewriting,
optional query decomposition, and human-in-the-loop web search.

Graph Flow (9 Nodes)::

    START
      │
      ▼
    condense_question ──► route_query ──► complexity?
                            │         │            │
                            │ cached  │ simple     │ complex
                            │ / ood   ▼            ▼
                            ▼       retrieve    plan_query
                           END        │            │
                                      │            ▼
                                      │        retrieve (per sub-query)
                                      │            │
                                      ▼◄───────────┘
                                  rerank_documents (Flashrank)
                                      │
                                      ▼
                                  relevant docs?
                                    │       │          │
                                    │ yes   │ no       │ no (exhausted retries)
                                    ▼       ▼          ▼
                                 generate  rewrite   ask_human_consent
                                    │       │          │
                                    ▼       ▼          ▼
                                   END   retrieve   surf_and_ingest_node
                                                       │
                                                       ▼
                                                    retrieve (loop back)

Key design decisions:
- Max 3 retrieval retries to avoid infinite loops
- Flashrank cross-encoder reranking with 0.80 relevance threshold
- Web search results are chunked and stored in Pinecone before retrieval
- Each node is a pure function over AgentState for testability
- Split-Brain LLM: small_llm (256 tokens) for internal, primary_llm (4096) for generation
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from finsight.rag.chains import (
    build_generator_chain,
    build_planner_chain,
    build_rewriter_chain,
    build_router_chain,
    build_query_analyzer_chain,
    build_condense_question_chain,
    build_conversational_chain,
)
from finsight.rag.retriever import HybridRetriever
from finsight.utils.query_cache import query_cache

logger = logging.getLogger(__name__)


# ======================================================================
# Agent State
# ======================================================================

from langgraph.graph.message import add_messages
from typing import Annotated

class AgentState(TypedDict, total=False):
    """State maintained across the agentic RAG loop.

    All node functions read from and write to this state dict.
    Fields use ``total=False`` so nodes only need to set the fields
    they modify.

    Attributes:
        chat_history: Conversational memory appended automatically.
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
        active_filters: Metadata filters applied during retrieval.
    """
    
    chat_history: Annotated[list[Any], add_messages]
    session_id: str
    question: str
    sub_queries: list[str]
    current_query: str
    documents: list[Document]
    all_documents: list[Document]
    generation: str
    relevance_scores: list[dict[str, Any]]
    retry_count: int
    max_retries: int
    route: Literal["simple", "complex", "cached", "out_of_domain"]
    error: str
    active_filters: dict[str, Any]
    human_consent: bool
    web_search_attempted: bool


# ======================================================================
# Node Functions
# ======================================================================

def _make_route_query(router_chain, conversational_chain=None):
    """Create the route_query node function.

    Classifies the query as simple or complex using the router chain.
    """

    def route_query(state: AgentState) -> AgentState:
        """Classify query complexity and set the routing decision."""
        question = state["question"]
        logger.info("Routing query: %.120s", question)
        
        from langchain_core.messages import AIMessage
        # Check cache first
        cached_answer = query_cache.get(question)
        if cached_answer:
            logger.info("Found cached answer for query. Skipping pipeline.")
            return {
                "route": "cached",
                "current_query": question,
                "generation": cached_answer,
                "chat_history": [AIMessage(content=cached_answer)],
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

        if route == "out_of_domain":
            if conversational_chain:
                try:
                    err_msg = conversational_chain.invoke({"question": question})
                except Exception:
                    err_msg = "I am FinSight, an AI Financial Analyst. I can help with corporate strategy, SEC filings, and financial performance."
            else:
                err_msg = "I am an AI Financial Analyst. I can help with corporate strategy, SEC filings, and financial performance. Or, you can ask me about documents you have explicitly uploaded."
                
            return {
                "route": route,
                "current_query": question,
                "generation": err_msg,
                "chat_history": [AIMessage(content=err_msg)],
                "retry_count": 0,
                "max_retries": state.get("max_retries", 3),
            }

        return {
            "route": route,
            "current_query": question,
            "retry_count": 0,
            "max_retries": state.get("max_retries", 3),
            "all_documents": [],
            "documents": [],
            "generation": "",
            "web_search_attempted": False,
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
        session_id = state.get("session_id", "finsight")

        def _extract_filters(q: str) -> dict[str, Any] | None:
            try:
                res = analyzer_chain.invoke({"question": q})
                filters = {}
                
                company_name = res.get("company_name")
                doc_type = res.get("type")
                
                if company_name:
                    filters["$or"] = [{"company_name": company_name}, {"company_name": {"$exists": False}}]
                
                if doc_type:
                    filters["type"] = doc_type
                    
                return filters if filters else None
            except Exception:
                return None

        if route == "complex" and state.get("sub_queries"):
            # Complex path: retrieve for all sub-queries
            sub_queries = state["sub_queries"]
            documents: list[Document] = []
            merged_filters = {}

            for sq in sub_queries:
                filters = _extract_filters(sq)
                if filters:
                    merged_filters.update(filters)
                logger.info("Retrieving for sub-query: %.120s with filters %s", sq, filters)
                results = retriever.retrieve(sq, namespace=session_id, top_k=40, filter=filters, vector_weight=40)
                documents.extend(results)

            # Deduplicate across sub-queries
            documents = _deduplicate_documents(documents)
            all_documents.extend(documents)
            filters = merged_filters

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
            documents = retriever.retrieve(current_query, namespace=session_id, top_k=40, filter=filters, vector_weight=40)

        return {
            "documents": documents,
            "all_documents": all_documents if route == "complex" else documents,
            "active_filters": filters or {},
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
        
        # Take Top 4 and apply a strict relevance threshold
        # Flashrank scores are typically probabilities [0, 1].
        # We use 0.50 as a cutoff to ensure only relevant docs are passed.
        # If no docs pass this threshold, it will trigger the query rewrite/web search fallback.
        active_filters = state.get("active_filters", {})
        is_upload_query = active_filters.get("type") == "user_upload"

        top_n = min(20, len(results))
        for res in results[:top_n]:
            idx = res.get("id")
            score = res.get("score", 0.0)
            
            if idx is None or not (0 <= idx < len(documents)):
                continue
                
            doc = documents[idx]
                
            # Flashrank scores are typically probabilities [0, 1].
            # We use 0.10 as a cutoff to allow tangentially relevant chunks (like data tables) 
            # while blocking completely irrelevant documents (which score < 0.01).
            if float(score) < 0.10:
                doc_type = doc.metadata.get("type")
                if is_upload_query and doc_type == "user_upload":
                    pass # Keep it
                else:
                    logger.info("  Doc %d: Score %.4f (Below threshold, discarding)", idx, score)
                    continue
                
            relevant_docs.append(doc)
            relevance_scores.append({
                "doc_index": idx,
                "relevant": True,
                "reasoning": f"Flashrank score: {float(score):.4f}",
                "source": doc.metadata.get("source", "unknown"),
            })
            logger.info("  Doc %d: Score %.4f (Kept)", idx, score)
            logger.info("  Doc %d Content snippet: %.200s...", idx, doc.page_content.replace('\n', ' '))
        logger.info(
            "Reranking complete: kept top %d documents.",
            len(relevant_docs)
        )

        return {
            "documents": relevant_docs,
            "all_documents": relevant_docs,
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
            if not rewritten:
                rewritten = current_query
                logger.warning("Rewriter returned empty string, falling back to original query.")
            else:
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

        from langchain_core.messages import AIMessage
        return {
            "generation": generation,
            "chat_history": [AIMessage(content=generation)]
        }

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


def _should_generate_or_rewrite(state: AgentState) -> Literal["generate", "rewrite_query", "ask_human_consent"]:
    """Decide whether to generate an answer or rewrite the query.

    Generates if:
    - We have at least 1 relevant document, OR
    - We've exhausted our retry budget.

    Rewrites if:
    - No relevant documents found AND retries remain.

    Returns:
        ``"generate"``, ``"rewrite_query"``, or ``"ask_human_consent"``.
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
        if state.get("web_search_attempted"):
            logger.warning("Already attempted web search. Giving up and generating partial answer.")
            return "generate"
        logger.warning(
            "No relevant docs after %d retries → asking human consent to web search.",
            retry_count,
        )
        return "ask_human_consent"

    logger.info(
        "No relevant docs (retry %d/%d) → rewriting query.",
        retry_count,
        max_retries,
    )
    return "rewrite_query"


# ======================================================================
# Human-in-the-Loop & Web Search
# ======================================================================

def ask_human_consent(state: AgentState) -> AgentState:
    from langgraph.types import interrupt
    logger.info("Pausing graph execution to ask for human consent...")
    
    # Interrupt execution and wait for the user to resume
    consent = interrupt("For this question I would have to surf the web and collate info. Should I proceed?")
    
    if consent is True or consent == "proceed":
        logger.info("Human granted consent for web search.")
        return {"route": "surf"}
    else:
        logger.info("Human denied consent.")
        return {"route": "cancel", "generation": "Search cancelled by user."}


def surf_and_ingest_node(state: AgentState) -> AgentState:
    from finsight.tools.web_surfer import surf_and_ingest
    
    # Use the original question rather than a heavily rewritten query
    # which might exceed Tavily's 400 character limit.
    original_query = state["question"]
    # Truncate just to be completely safe against Tavily 400 char limit
    search_query = original_query[:350]
    session_id = state.get("session_id", "finsight")
    # Extract company name using the analyzer chain so we can tag the docs
    from finsight.rag.chains import build_query_analyzer_chain
    from langchain_groq import ChatGroq
    from config.settings import settings
    
    # We create a lightweight analyzer here to keep it simple, or we could pass it in.
    llm = ChatGroq(model=settings.groq_fallback_model, api_key=settings.groq_api_key)
    analyzer = build_query_analyzer_chain(llm)
    try:
        analysis = analyzer.invoke({"question": search_query})
        company_name = analysis.get("company_name")
        optimized_query = analysis.get("optimized_search_query")
        if optimized_query:
            search_query = optimized_query[:350]
    except Exception:
        company_name = None
        
    extra_metadata = {"company_name": company_name} if company_name else {}

    logger.info("Executing surf_and_ingest node for: %s (Company: %s)", search_query, company_name)
    
    # This downloads, chunks, and puts the results into Pinecone
    ingested_docs = surf_and_ingest(search_query, namespace=session_id, extra_metadata=extra_metadata)
    logger.info("Successfully ingested %d chunks to Pinecone.", len(ingested_docs))
    
    # We do NOT pass the ingested docs directly to generate!
    # Instead, we clear the documents list and reset retry count, 
    # and route back to the `retrieve` node to fetch the top 4 chunks cleanly.
    return {
        "documents": [],
        "retry_count": 0,
        "current_query": search_query, # Use this for retrieval
        "web_search_attempted": True,
    }


def _should_surf_or_end(state: AgentState) -> Literal["surf_and_ingest_node", "END"]:
    if state.get("route") == "surf":
        logger.info("Human consent granted → initiating web search.")
        return "surf_and_ingest_node"
    logger.info("Human consent denied → terminating pipeline.")
    return "END"

def _make_check_hallucinations_and_answer(answer_grader):
    def _check_hallucinations_and_answer(state: AgentState) -> Literal["rewrite_query", "ask_human_consent", "END"]:
        """Self-reflection: Grades if the generation answers the question."""
        generation = state.get("generation", "")
        question = state["question"]
        
        # Hardcoded heuristic check first for obvious failures
        lacks_info = (
            "don't have enough information" in generation.lower() or 
            "don’t have enough information" in generation.lower() or 
            "do not have enough information" in generation.lower() or
            "not have enough information" in generation.lower()
        )
        
        if lacks_info:
            score = "no"
            logger.warning("Generation failed heuristic check (lacks info).")
        else:
            try:
                # Ask LLM grader if it answered the question
                res = answer_grader.invoke({
                    "question": question,
                    "generation": generation
                })
                score = res.get("score", "yes")
                logger.info("Answer grader score: %s", score)
            except Exception:
                logger.exception("Answer grader failed, defaulting to 'yes'.")
                score = "yes"
                
        if score == "yes":
            return "END"
            
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        
        # If we still have retries, try rewriting and fetching better context
        if retry_count < max_retries:
            logger.warning("Answer graded as inadequate. Triggering retry loop (%d/%d).", retry_count + 1, max_retries)
            return "rewrite_query"
            
        # If we are out of retries and haven't tried web search yet
        if not state.get("web_search_attempted"):
            logger.warning("Answer graded as inadequate and retries exhausted. Triggering web search.")
            return "ask_human_consent"
            
        logger.warning("Answer graded as inadequate, but all fallbacks exhausted. Ending.")
        return "END"
        
    return _check_hallucinations_and_answer

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
    small_llm: BaseChatModel | None = None,
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
        small_llm: Smaller LLM for routing and planning tasks.
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
    grading_llm = small_llm or fallback_llm or llm
    routing_llm = small_llm or fallback_llm or llm
    planning_llm = small_llm or llm

    # Build individual chains
    router_chain = build_router_chain(routing_llm)
    planner_chain = build_planner_chain(planning_llm)
    rewriter_chain = build_rewriter_chain(planning_llm)
    generator_chain = build_generator_chain(llm)
    analyzer_chain = build_query_analyzer_chain(routing_llm)
    condenser_chain = build_condense_question_chain(routing_llm)
    conversational_chain = build_conversational_chain(routing_llm)
    from finsight.rag.chains import build_answer_grader_chain
    answer_grader_chain = build_answer_grader_chain(routing_llm)

    # Create node functions (closures over chains + retriever)
    route_query = _make_route_query(router_chain, conversational_chain)
    plan_query = _make_plan_query(planner_chain)
    retrieve = _make_retrieve(retriever, analyzer_chain)
    rerank_documents = _make_rerank_documents()
    rewrite_query = _make_rewrite_query(rewriter_chain)
    generate = _make_generate(generator_chain)

    def _make_condense_question(condenser_chain):
        def condense_question(state: AgentState) -> AgentState:
            history = state.get("chat_history", [])
            question = state["question"]
            
            # If history only has the current message or is empty, skip condensation
            if not history or len(history) <= 1:
                return {"question": question}
            
            logger.info("Condensing query using chat history...")
            # We pass the history excluding the latest HumanMessage (which is the current question)
            past_history = history[:-1]
            history_str = "\n".join([f"{'Human' if msg.type == 'human' else 'AI'}: {msg.content}" for msg in past_history[-4:]]) # only keep last 4 to prevent context bloat
            
            condensed = condenser_chain.invoke({
                "chat_history": history_str,
                "question": question
            })
            logger.info("Condensed query: %s", condensed)
            return {"question": condensed}
        return condense_question

    condense_question_node = _make_condense_question(condenser_chain)
    
    # ---- Build the graph ----
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("condense_question", condense_question_node)
    workflow.add_node("route_query", route_query)
    workflow.add_node("plan_query", plan_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank_documents", rerank_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    
    # HITL Nodes
    workflow.add_node("ask_human_consent", ask_human_consent)
    workflow.add_node("surf_and_ingest_node", surf_and_ingest_node)

    # Set entry point
    workflow.set_entry_point("condense_question")
    workflow.add_edge("condense_question", "route_query")

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

    # Conditional edge: rerank_documents → generate OR rewrite_query OR ask_human_consent
    workflow.add_conditional_edges(
        "rerank_documents",
        _should_generate_or_rewrite,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "ask_human_consent": "ask_human_consent",
        },
    )

    # rewrite_query → retrieve (retry loop)
    workflow.add_edge("rewrite_query", "retrieve")

    # ask_human_consent → conditional to surf or END
    workflow.add_conditional_edges(
        "ask_human_consent",
        _should_surf_or_end,
        {
            "surf_and_ingest_node": "surf_and_ingest_node",
            "END": END,
        }
    )

    # surf_and_ingest_node → retrieve
    # (After ingesting to Pinecone, we loop back to retrieve the top 4 chunks)
    workflow.add_edge("surf_and_ingest_node", "retrieve")

    # generate → END, ask_human_consent, or rewrite_query (Self-Reflection Loop)
    check_hallucinations = _make_check_hallucinations_and_answer(answer_grader_chain)
    workflow.add_conditional_edges(
        "generate",
        check_hallucinations,
        {
            "ask_human_consent": "ask_human_consent",
            "rewrite_query": "rewrite_query",
            "END": END,
        }
    )

    # Compile with persistent Sqlite checkpointer for session state
    from langgraph.checkpoint.sqlite import SqliteSaver
    import sqlite3
    
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    compiled_graph = workflow.compile(checkpointer=checkpointer)

    logger.info("RAG agent graph compiled successfully (max_retries=%d).", max_retries)

    return compiled_graph
