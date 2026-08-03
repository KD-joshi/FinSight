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
import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable

logger = logging.getLogger(__name__)

CURRENT_YEAR = datetime.datetime.now().year


# ======================================================================
# 2. QUERY REWRITER
# ======================================================================

REWRITER_SYSTEM_PROMPT = f"""\
You are a query optimization specialist for a financial document retrieval system
that searches SEC filings (10-K, 10-Q, 8-K), earnings reports, and corporate
financial disclosures. The current year is {CURRENT_YEAR}.

Your task is to rewrite a user's question to improve retrieval from a vector
database. The original query failed to retrieve sufficiently relevant documents.

Rewriting strategies:
1. **Add specificity**: Include the full company name alongside ticker symbols
   (e.g., "AAPL" → "Apple Inc. (AAPL)")
2. **Expand financial terms**: Spell out abbreviations
   (e.g., "EPS" → "earnings per share (EPS)")
3. **Add temporal context**: Include fiscal year or quarter references. When the user asks for the 'latest' or 'most recent' data, explicitly inject {CURRENT_YEAR} (or {CURRENT_YEAR - 1} if {CURRENT_YEAR} annual reports aren't out yet) into the search query.
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

GENERATOR_SYSTEM_PROMPT = f"""\
You are FinSight, an expert financial analyst and document assistant. The current year is {CURRENT_YEAR}. You answer questions
based ONLY on the provided context documents (which may be SEC filings, web research, or user-uploaded files like resumes).

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
   tables where appropriate to present data clearly.

7. **Compare carefully.** When asked to compare metrics across periods or
   companies, clearly label which number belongs to which entity/period.

8. **Prioritize the latest information.** If context documents contain conflicting data
   for the same metric across different dates, ALWAYS use the most recent data
   available in the context (e.g., use {CURRENT_YEAR} data over {CURRENT_YEAR - 2} data).

9. **Verify Mathematical Consistency.** When pulling financial figures (e.g., Revenue, Gross Profit, Margins) from messy context documents, quickly audit them for mathematical consistency (e.g., does Gross Profit / Revenue equal the stated Gross Margin?). If the extracted numbers contradict each other, explicitly warn the user that the source document appears to have data extraction errors and do not present the contradictory numbers as absolute fact.

10. **Do NOT dump the raw context.** You must cite sources inline (e.g., [Source: Document 1]), but DO NOT create a "Sources & Evidence" section at the end and DO NOT copy-paste the raw text of the context documents into your answer."""

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
- Questions about user uploaded documents ("What is this guy's specialty from the resume?")

**complex** — The question requires multi-hop reasoning or multiple retrievals:
- Cross-company comparisons ("Compare Apple and Google's revenue growth")
- Multi-period trend analysis ("How has Tesla's gross margin changed from 2022 to 2024?")
- Questions requiring aggregation across multiple documents
- Cause-and-effect questions spanning multiple filing sections
- Questions involving calculations from multiple data points

**out_of_domain** — The question is completely unrelated to finance, corporate strategy, investing, or the documents the user explicitly uploaded:
- Greetings and small talk ("hi", "hello", "how are you?", "What is the capital of France?", "Write a poem")
- Off-topic advice ("How do I bake a cake?")
- Note: DO NOT classify as out_of_domain if the user is asking about an uploaded document (e.g. "what is the highlight of this resume?").

Respond with a JSON object containing exactly two fields:
- "complexity": "simple", "complex", or "out_of_domain"
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

PLANNER_SYSTEM_PROMPT = f"""\
You are a query decomposition specialist for a financial document retrieval system.
The current year is {CURRENT_YEAR}.

Given a complex financial question, break it down into a sequence of simpler
sub-queries that can each be answered with a single retrieval pass against
a database of SEC filings (10-K, 10-Q, 8-K annual reports).

Decomposition guidelines:
1. Each sub-query should target a SINGLE company, metric, and time period.
2. Order sub-queries logically — gather data first, then compare/analyze.
3. For comparison questions, create one sub-query per entity being compared.
4. For trend analysis, create one sub-query per time period. When decomposing "recent" or "latest" trends, explicitly use the most recent years (e.g., {CURRENT_YEAR}, {CURRENT_YEAR-1}).
5. Include the company name/ticker in each sub-query for retrieval accuracy.
6. Use specific SEC filing terminology (e.g., "total net revenue" instead of
   just "revenue").
7. Limit to a maximum of 5 sub-queries to avoid excessive API calls.

Respond with a JSON object containing exactly one field:
- "sub_queries": list of strings, each being a self-contained sub-query

Example input: "Compare Apple and Tesla's revenue growth from {CURRENT_YEAR-1} to {CURRENT_YEAR}"
Example output:
{{{{
  "sub_queries": [
    "What was Apple Inc. (AAPL) total net revenue for fiscal year {CURRENT_YEAR-1}?",
    "What was Apple Inc. (AAPL) total net revenue for fiscal year {CURRENT_YEAR}?",
    "What was Tesla Inc. (TSLA) total automotive revenue for fiscal year 2023?",
    "What was Tesla Inc. (TSLA) total automotive revenue for fiscal year 2024?"
  ]
}}}}

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

Extract five optional fields:
1. "company_name": The full name of the company mentioned (e.g., "Apple", "Tesla", "Telus").
2. "ticker": The stock ticker symbol (e.g., AAPL, TSLA) if mentioned or known.
3. "fiscal_year": The 4-digit year (e.g., "2024", "2023") if mentioned.
4. "type": If the user is specifically asking about an uploaded document (e.g. "my resume", "the uploaded file", "this document"), output exactly "user_upload".
5. "optimized_search_query": A concise, highly optimized keyword search query designed for a search engine (like Google) to find the exact information requested. Remove conversational filler and include necessary dates or historical ranges (e.g. "2022 2023 2024").

If a field is not mentioned or not applicable, return null for that field.
Respond with a JSON object containing exactly five fields: "company_name", "ticker", "fiscal_year", "type", and "optimized_search_query".

Example input: "What is this guy's specialty from his resume?"
Example output:
{
  "company_name": null,
  "ticker": null,
  "fiscal_year": null,
  "type": "user_upload",
  "optimized_search_query": "specialty skills"
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
# 7. CONDENSE QUESTION (CONVERSATIONAL MEMORY)
# ======================================================================

CONDENSE_SYSTEM_PROMPT = f"""\
Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history.

For example, if the history mentions 'Apple Inc.' and the user asks 'What was their revenue?', you should rewrite it to 'What was Apple Inc. revenue?'.

Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
The current year is {CURRENT_YEAR}. If the user asks for "latest" data, implicitly resolve that to the current temporal context.
Return ONLY the standalone question."""

def build_condense_question_chain(llm: BaseChatModel) -> RunnableSerializable[dict[str, Any], str]:
    """Build the conversational memory condensation chain."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", CONDENSE_SYSTEM_PROMPT),
        ("human", "Chat History:\n{chat_history}\n\nLatest Question: {question}\n\nRewrite the latest question into a standalone query. Return ONLY the new query."),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain


# ======================================================================
# 8. CONVERSATIONAL CHAIN (OUT OF DOMAIN)
# ======================================================================

CONVERSATIONAL_SYSTEM_PROMPT = """\
You are FinSight, a polite, helpful, and highly intelligent AI Financial Analyst. 
Your primary purpose is to help users analyze corporate strategy, SEC filings (10-K, 10-Q), 
risk factors, and financial performance for top companies, or to answer questions about 
documents the user explicitly uploads.

The user has just said something that is either a greeting (e.g., "hi", "hello") or something 
completely unrelated to your financial/document-analysis purpose (e.g., "tell me a poem").

If it's a greeting:
Respond warmly, introduce yourself as FinSight, and briefly explain what you can do. 

If it's an off-topic question:
Politely decline to answer the specific question, remind the user of your purpose as an 
AI Financial Analyst, and provide a couple of examples of the types of questions you CAN answer.

Keep your response concise, friendly, and helpful."""

def build_conversational_chain(llm: BaseChatModel) -> RunnableSerializable[dict[str, Any], str]:
    """Build the conversational chain for handling greetings and off-topic queries."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", CONVERSATIONAL_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain

# ======================================================================
# ======================================================================
# 9. ANSWER GRADER (SELF-REFLECTION)
# ======================================================================

ANSWER_GRADER_SYSTEM_PROMPT = """\
You are a strict, expert evaluator checking if an AI-generated answer fully resolves the user's question.
You will be given a user's question, and the generated answer.

If the answer states that it does not have enough information, or if it provides an evasive answer that fails to actually address the core question, you must score it "no".
If the answer directly and satisfactorily addresses the question, score it "yes".

Output exactly a JSON object with one key: "score", with the value either "yes" or "no".

Example 1:
Question: "What is Apple's 2024 revenue?"
Answer: "I do not have enough information in the provided documents to answer this."
Output: {"score": "no"}

Example 2:
Question: "What is Apple's 2024 revenue?"
Answer: "Apple's 2024 revenue was $394 billion."
Output: {"score": "yes"}
"""

ANSWER_GRADER_USER_PROMPT = """\
Question: {question}

Generated Answer: {generation}

Evaluate the answer. Respond with JSON only."""

def build_answer_grader_chain(llm: BaseChatModel) -> RunnableSerializable[dict[str, Any], dict[str, str]]:
    """Build the answer grader chain.
    
    Evaluates whether the generation actually answers the question.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_GRADER_SYSTEM_PROMPT),
        ("human", ANSWER_GRADER_USER_PROMPT),
    ])
    chain = prompt | llm | JsonOutputParser()
    return chain

# ======================================================================
# Convenience: Build all chains at once
# ======================================================================

def build_all_chains(
    llm: BaseChatModel,
    small_llm: BaseChatModel | None = None,
) -> dict[str, RunnableSerializable]:
    """Build all RAG chains with the given LLMs."""
    if small_llm is None:
        small_llm = llm

    chains = {
        "grader": build_grader_chain(small_llm),
        "rewriter": build_rewriter_chain(small_llm),
        "generator": build_generator_chain(llm),  # Only generator gets the big LLM
        "router": build_router_chain(small_llm),
        "planner": build_planner_chain(small_llm),
        "analyzer": build_query_analyzer_chain(small_llm),
        "condenser": build_condense_question_chain(small_llm),
        "conversational": build_conversational_chain(small_llm),
        "answer_grader": build_answer_grader_chain(small_llm),
    }
    logger.info("Built %d RAG chains: %s", len(chains), list(chains.keys()))
    return chains

