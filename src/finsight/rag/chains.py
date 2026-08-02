"""Core RAG chains for the agentic pipeline.

Each chain is a focused, composable LangChain Expression Language (LCEL)
component designed to be called as a node in the LangGraph agent graph.

Chains:
    1. GraderChain      — Evaluates document relevance to a query
    2. RewriterChain    — Rewrites queries to improve retrieval
    3. GeneratorChain   — Generates grounded, cited answers
    4. RouterChain      — Classifies query complexity
    5. PlannerChain     — Decomposes complex queries into sub-queries
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable

logger = logging.getLogger(__name__)




# ======================================================================
# 2. QUERY REWRITER
# ======================================================================

REWRITER_SYSTEM_PROMPT = """\
You are a query optimization specialist for a financial document retrieval system
that searches SEC filings (10-K, 10-Q, 8-K), earnings reports, and corporate
financial disclosures.

Your task is to rewrite a user's question to improve retrieval from a vector
database. The original query failed to retrieve sufficiently relevant documents.

Rewriting strategies:
1. **Add specificity**: Include the full company name alongside ticker symbols
   (e.g., "AAPL" → "Apple Inc. (AAPL)")
2. **Expand financial terms**: Spell out abbreviations
   (e.g., "EPS" → "earnings per share (EPS)")
3. **Add temporal context**: Include fiscal year or quarter references
4. **Use SEC filing language**: Match the formal language used in 10-K/10-Q filings
   (e.g., "revenue" → "total net revenue" or "net sales")
5. **Decompose compound queries**: If the query asks multiple things, focus on the
   most critical information need
6. **Add section references**: Reference specific SEC filing sections
   (e.g., "Item 7 — Management's Discussion and Analysis")

Respond with ONLY the rewritten query, nothing else."""

REWRITER_USER_PROMPT = """\
Original question: {question}

Previously retrieved context that was NOT relevant:
---
{failed_context}
---

Rewrite the question to improve retrieval. Respond with only the rewritten query."""


def build_rewriter_chain(llm: BaseChatModel) -> RunnableSerializable[dict[str, Any], str]:
    """Build the query rewriter chain.

    Rewrites a query that failed to retrieve relevant documents,
    using the failed context to understand what went wrong and
    craft a more targeted query.

    Args:
        llm: Language model instance.

    Returns:
        LCEL chain that accepts ``{"question": str, "failed_context": str}``
        and returns a rewritten query string.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITER_SYSTEM_PROMPT),
        ("human", REWRITER_USER_PROMPT),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain


# ======================================================================
# 3. ANSWER GENERATOR
# ======================================================================

GENERATOR_SYSTEM_PROMPT = """\
You are FinSight, an expert financial analyst assistant. You answer questions
about companies' financial performance, SEC filings, and corporate disclosures
using ONLY the provided context documents.

STRICT RULES — you MUST follow these without exception:

1. **Ground every claim in the provided context.** Only use information
   explicitly stated in the documents below. Do NOT use prior knowledge.

2. **Cite your sources.** For every factual claim, include an inline citation
   in the format [Source: <source_identifier>] where source_identifier comes
   from the document's metadata (filename, filing type, section, page, etc.).

3. **Cite financial numbers precisely.** When reporting any financial metric
   (revenue, net income, EPS, margins, etc.), always include:
   - The exact number as stated in the source
   - The time period (fiscal year, quarter)
   - The source citation

4. **Acknowledge gaps honestly.** If the provided context does not contain
   enough information to fully answer the question, explicitly state:
   "Based on the available documents, I don't have enough information to
   answer this completely." Then provide whatever partial answer IS supported.

5. **Never fabricate numbers.** If you cannot find a specific metric in the
   context, say so. Do not estimate, interpolate, or guess.

6. **Structure your response clearly.** Use headers, bullet points, and
   tables where appropriate to present financial data clearly.

7. **Compare carefully.** When asked to compare metrics across periods or
   companies, clearly label which number belongs to which entity/period."""

GENERATOR_USER_PROMPT = """\
Question: {question}

Context Documents:
{context}

Provide a comprehensive, well-cited answer based ONLY on the context above."""


def build_generator_chain(llm: BaseChatModel) -> RunnableSerializable[dict[str, Any], str]:
    """Build the answer generator chain.

    Generates grounded, cited answers from retrieved context documents.
    Enforces strict citation requirements and prohibits hallucination.

    Args:
        llm: Language model instance.

    Returns:
        LCEL chain that accepts ``{"question": str, "context": str}``
        and returns a cited answer string.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", GENERATOR_SYSTEM_PROMPT),
        ("human", GENERATOR_USER_PROMPT),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain


# ======================================================================
# 4. QUERY ROUTER
# ======================================================================

ROUTER_SYSTEM_PROMPT = """\
You are a query complexity classifier for a financial document Q&A system.

Classify the user's question into one of two categories:

**simple** — The question can be answered with a single retrieval pass:
- Single-company, single-metric queries ("What was Apple's revenue in 2024?")
- Direct factual lookups ("Who is Tesla's CEO?")
- Single time period questions ("What risks did Google disclose in their 2024 10-K?")

**complex** — The question requires multi-hop reasoning or multiple retrievals:
- Cross-company comparisons ("Compare Apple and Google's revenue growth")
- Multi-period trend analysis ("How has Tesla's gross margin changed from 2022 to 2024?")
- Questions requiring aggregation across multiple documents
- Cause-and-effect questions spanning multiple filing sections
- Questions involving calculations from multiple data points

Respond with a JSON object containing exactly two fields:
- "complexity": "simple" or "complex"
- "reasoning": string (1 sentence explaining the classification)

Do NOT include any text outside the JSON object."""

ROUTER_USER_PROMPT = """\
Question: {question}

Classify this question's complexity. Respond with JSON only."""


def build_router_chain(llm: BaseChatModel) -> RunnableSerializable[dict[str, Any], dict[str, Any]]:
    """Build the query router/classifier chain.

    Classifies incoming questions as ``simple`` (single retrieval pass)
    or ``complex`` (needs decomposition into sub-queries).

    Args:
        llm: Language model instance.

    Returns:
        LCEL chain that accepts ``{"question": str}`` and returns
        ``{"complexity": "simple" | "complex", "reasoning": str}``.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_PROMPT),
        ("human", ROUTER_USER_PROMPT),
    ])
    chain = prompt | llm | JsonOutputParser()
    return chain


# ======================================================================
# 5. QUERY PLANNER / DECOMPOSER
# ======================================================================

PLANNER_SYSTEM_PROMPT = """\
You are a query decomposition specialist for a financial document retrieval system.

Given a complex financial question, break it down into a sequence of simpler
sub-queries that can each be answered with a single retrieval pass against
a database of SEC filings (10-K, 10-Q, 8-K annual reports).

Decomposition guidelines:
1. Each sub-query should target a SINGLE company, metric, and time period.
2. Order sub-queries logically — gather data first, then compare/analyze.
3. For comparison questions, create one sub-query per entity being compared.
4. For trend analysis, create one sub-query per time period.
5. Include the company name/ticker in each sub-query for retrieval accuracy.
6. Use specific SEC filing terminology (e.g., "total net revenue" instead of
   just "revenue").
7. Limit to a maximum of 5 sub-queries to avoid excessive API calls.

Respond with a JSON object containing exactly one field:
- "sub_queries": list of strings, each being a self-contained sub-query

Example input: "Compare Apple and Tesla's revenue growth from 2023 to 2024"
Example output:
{{
  "sub_queries": [
    "What was Apple Inc. (AAPL) total net revenue for fiscal year 2023?",
    "What was Apple Inc. (AAPL) total net revenue for fiscal year 2024?",
    "What was Tesla Inc. (TSLA) total automotive revenue for fiscal year 2023?",
    "What was Tesla Inc. (TSLA) total automotive revenue for fiscal year 2024?"
  ]
}}

Do NOT include any text outside the JSON object."""

PLANNER_USER_PROMPT = """\
Complex question: {question}

Decompose this into simpler sub-queries. Respond with JSON only."""


def build_planner_chain(llm: BaseChatModel) -> RunnableSerializable[dict[str, Any], dict[str, Any]]:
    """Build the query planner/decomposition chain.

    Decomposes complex multi-hop questions into a sequence of simpler
    sub-queries that can each be answered independently.

    Args:
        llm: Language model instance.

    Returns:
        LCEL chain that accepts ``{"question": str}`` and returns
        ``{"sub_queries": list[str]}``.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM_PROMPT),
        ("human", PLANNER_USER_PROMPT),
    ])
    chain = prompt | llm | JsonOutputParser()
    return chain


# ======================================================================
# 6. QUERY ANALYZER
# ======================================================================

QUERY_ANALYZER_SYSTEM_PROMPT = """\
You are an expert financial query analyzer. Your job is to extract metadata filters from a user's question to optimize database retrieval.

Extract two optional fields:
1. "ticker": The stock ticker symbol (e.g., AAPL, TSLA, GOOGL, MSFT) if mentioned. If a company name is used instead, output its standard ticker.
2. "fiscal_year": The 4-digit year (e.g., "2024", "2023") if mentioned.

If a field is not mentioned, return null for that field.
Respond with a JSON object containing exactly two fields: "ticker" and "fiscal_year".

Example input: "What was Apple's revenue in 2024?"
Example output:
{
  "ticker": "AAPL",
  "fiscal_year": "2024"
}
"""

QUERY_ANALYZER_USER_PROMPT = """\
Question: {question}

Extract metadata filters. Respond with JSON only."""


def build_query_analyzer_chain(llm: BaseChatModel) -> RunnableSerializable[dict[str, Any], dict[str, Any]]:
    """Build the query analyzer chain.

    Extracts metadata filters (ticker, fiscal_year) from the query
    to optimize retrieval.

    Args:
        llm: Language model instance.

    Returns:
        LCEL chain that accepts ``{"question": str}`` and returns
        ``{"ticker": str | None, "fiscal_year": str | None}``.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_ANALYZER_SYSTEM_PROMPT),
        ("human", QUERY_ANALYZER_USER_PROMPT),
    ])
    chain = prompt | llm | JsonOutputParser()
    return chain


# ======================================================================
# Convenience: Build all chains at once
# ======================================================================

def build_all_chains(
    llm: BaseChatModel,
) -> dict[str, RunnableSerializable]:
    """Build all RAG chains with the given LLM.

    Convenience function that constructs every chain needed by the
    LangGraph agentic pipeline.

    Args:
        llm: Primary language model instance (e.g., Groq Llama 3.1 70B).

    Returns:
        Dictionary mapping chain names to LCEL chain instances::

            {
                "grader": ...,
                "rewriter": ...,
                "generator": ...,
                "router": ...,
                "planner": ...,
                "analyzer": ...,
            }
    """
    chains = {
        "grader": build_grader_chain(llm),
        "rewriter": build_rewriter_chain(llm),
        "generator": build_generator_chain(llm),
        "router": build_router_chain(llm),
        "planner": build_planner_chain(llm),
        "analyzer": build_query_analyzer_chain(llm),
    }
    logger.info("Built %d RAG chains: %s", len(chains), list(chains.keys()))
    return chains

