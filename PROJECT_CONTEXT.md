# FinSight — Project Context & Handoff Document

**Project**: FinSight — Agentic RAG for Financial Intelligence

## 1. Project Overview
FinSight is a production-grade Agentic Retrieval-Augmented Generation (RAG) system specifically designed for SEC filings and financial documents. It leverages advanced orchestration patterns like Corrective RAG (CRAG) using LangGraph and LangChain, enabling sophisticated multi-hop reasoning over financial data with human-in-the-loop web search.

## 2. Current State & Completed Work

### 2.1 Backend Pipeline (Complete)
- **Ingestion**: Tools to download and process SEC filings (`sec_downloader.py`, `document_processor.py`), chunking them with Docling and embedding them with HuggingFace (`all-MiniLM-L6-v2`).
- **Vector Store**: Pinecone Serverless with namespace-per-session isolation.
- **Retrieval**: Dense vector search via Pinecone. BM25 sparse search is implemented but currently disabled (Pinecone Serverless doesn't support scanning all docs for local BM25 index). Reranking performed using a local cross-encoder (Flashrank, threshold 0.80).
- **Agentic Loop (LangGraph)**: `agents/graph.py` implements a 9-node state graph:
  - **Condenser**: Resolves conversational references ("their revenue" → "Apple's revenue") using chat history.
  - **Router**: Classifies queries as `simple`, `complex`, `out_of_domain`, or `cached`.
  - **Planner**: Decomposes complex queries into ≤5 sub-queries.
  - **Retriever**: Executes Pinecone vector search (namespace = session_id).
  - **Reranker**: Flashrank cross-encoder scoring with 0.80 threshold.
  - **Rewriter**: Rewrites the query if documents are not relevant, with max 3 retries.
  - **Generator**: Uses primary LLM (Groq `openai/gpt-oss-120b`) to generate a cited answer.
  - **HITL Consent**: Pauses execution and asks the user for permission to search the web.
  - **Web Surfer**: Tavily web search → LlamaParse (PDFs) → chunk → ingest to Pinecone → loop back to retriever.
- **API (FastAPI)**: REST endpoints for chat, session management, document upload, and HITL consent resume.
- **Frontend (Next.js)**: Chat interface with session management, consent UI, document upload, and session history with human-readable titles.
- **Observability**: LangSmith integration for native tracing (configured via LANGCHAIN_TRACING_V2).
- **Evaluation**: Synthetic dataset generator and evaluator (`langsmith_eval.py`) using LangSmith.

### 2.2 Tech Stack
- **Primary LLM**: Groq API (`openai/gpt-oss-120b`, 4096 max tokens) with Gemini 3.8 Flash + Cohere fallbacks.
- **Small LLM**: Same model capped at 256 max tokens for internal agents (Router, Planner, Rewriter, Condenser).
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`, 384 dimensions, local).
- **Vector Database**: Pinecone Serverless.
- **Reranking**: Flashrank (local cross-encoder).
- **Web Search**: Tavily API.
- **Document Parsing**: Docling (local PDFs), LlamaParse (uploads + web PDFs).
- **Orchestration**: LangGraph & LangChain.
- **Backend API**: FastAPI (Python).
- **Frontend**: Next.js 15, React, TypeScript.
- **Session Persistence**: SQLite (`checkpoints.sqlite`) via LangGraph `SqliteSaver`.
- **Query Cache**: `query_cache.json` (exact-match, persistent).

### 2.3 Token Optimization (Split-Brain Architecture)
- **`small_llm` (256 tokens)**: Router, Planner, Rewriter, Analyzer, Condenser.
- **`primary_llm` (4096 tokens)**: Generator only.
- **Context cap**: Generator context hard-capped at 16,000 characters (~4K tokens).
- **Fallback chain**: `Groq → Gemini → Cohere` via LangChain `.with_fallbacks()`.

## 3. What's Next (Pending Work)

1. **Refinement & Polish**
   - Clean up BM25 sparse retrieval or remove it entirely.
   - Dockerize the application (`docker-compose.yaml`).
   - Add rate limit handling improvements.

2. **Advanced Fine-Tuning**
   - Fine-tune models specifically for financial Q&A using SFT (Supervised Fine-Tuning), DPO (Direct Preference Optimization), and GRPO on custom datasets via Colab.

## 4. Codebase Navigation Guide
- `/src/finsight/agents/graph.py`: The heart of the agentic RAG loop (9-node LangGraph state machine).
- `/src/finsight/rag/chains.py`: LangChain prompts and LCEL chain builders for each agent.
- `/src/finsight/rag/retriever.py`: Hybrid search logic (Pinecone + BM25 + RRF).
- `/src/finsight/tools/web_surfer.py`: Tavily web search + LlamaParse + Pinecone ingestion.
- `/src/finsight/api/routes.py`: FastAPI endpoints (chat, upload, sessions, HITL resume).
- `/src/finsight/api/dependencies.py`: LLM and agent initialization factory.
- `/src/finsight/utils/llm_provider.py`: Multi-tier LLM fallback logic.
- `/src/finsight/utils/embeddings.py`: HuggingFace embedding provider.
- `/src/finsight/ingestion/vector_store.py`: Pinecone vector store wrapper.
- `/config/settings.py`: Centralized app configuration (pydantic-settings).
- `/frontend/`: Next.js web application.

## 5. Instructions for LLM Agents
When picking up a task on this project:
1. **Understand the Goal**: We are aiming for a highly polished, production-grade AI financial tool. Follow the "Teacher Mode" guidelines specified in `.agents/AGENTS.md` (break down the What, Why, and How before writing complex code).
2. **Consult docs/architecture_diagram.md**: Full architecture documentation with Mermaid diagrams.
3. **Frontend Aesthetic**: The UI must be visually excellent, employing modern design systems, clean typography, and smooth micro-animations.

---
*Last updated: 2 Aug 2026*
