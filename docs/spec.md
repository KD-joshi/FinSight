# FinSight Technical Specification

This document details the exact technical stack, tools, concepts, algorithms, and models used in the FinSight architecture.

## 1. Core Tech Stack & Libraries

### Orchestration & AI Framework
*   **LangGraph** (`langgraph v1.2.6`): Used for orchestrating the multi-agent state machine. Enables loops, conditional routing, and human-in-the-loop (HITL) checkpoints.
*   **LangChain** (`langchain-core v1.4.8`): Provides the core abstractions for LLMs, prompts (LCEL), and document handling.

### Backend API
*   **FastAPI** (`fastapi v0.138.0`): High-performance async web framework for the backend REST API.

### Frontend
*   **Next.js 15 / React**: The frontend framework (implemented in TypeScript) managing the chat interface and session state.

## 2. Models and LLM Configuration

The system employs a "Split-Brain" architecture and a resilient fallback mechanism.

### Primary Generator LLM
*   **Provider**: Groq Cloud
*   **Model**: `openai/gpt-oss-120b`
*   **Configurable Parameters**:
    *   `max_tokens`: 4096 (Hard limit for extensive generation)
    *   `temperature`: Adjustable via `settings.py` (Default: 0.0 for deterministic financial answers)
*   **Role**: Only used by the final `Generator` node to produce the cited, grounded answer.

### Internal Agent LLM ("Small LLM")
*   **Provider**: Groq Cloud
*   **Model**: `openai/gpt-oss-120b` (Identical model, strictly constrained)
*   **Configurable Parameters**:
    *   `max_tokens`: 256
*   **Role**: Used by internal agents (Router, Planner, Rewriter, Condenser, Analyzer). Constraining the tokens drastically reduces the Tokens Per Minute (TPM) budget consumed on the Groq free tier.

### Fallback LLMs (Waterfall Strategy)
If Groq hits a `429 Rate Limit` or `500 Server Error`, the system automatically falls back using LangChain's `.with_fallbacks()` mechanism:
1.  **Google Gemini**: `gemini-3.6-flash` (via `langchain-google-genai v4.2.5`)
2.  **Cohere**: `command-a-03-2025` (via `langchain-cohere v0.6.0`)

## 3. Embedding and Retrieval Specifications

### Embedding Model
*   **Provider**: HuggingFace (Local)
*   **Model**: `all-MiniLM-L6-v2` (via `langchain-huggingface v1.2.2`)
*   **Dimension**: 384
*   **Role**: Embeds document chunks and user queries for semantic search.

### Vector Database
*   **Provider**: Pinecone Serverless (via `langchain-pinecone v0.2.13`)
*   **Architecture**: Namespace-based isolation. Each chat thread receives a unique `session_id` (UUID), which maps directly to a Pinecone namespace. This ensures 100% data isolation between chat sessions.

### Retrieval Algorithms
*   **Dense Search**: Cosine similarity search within the Pinecone namespace.
*   **Sparse Search**: BM25 (`rank-bm25 v0.2.2`). *Note: Currently disabled due to Pinecone Serverless constraints on scanning full document lists locally.*
*   **Reciprocal Rank Fusion (RRF)**: Implemented to merge dense and sparse results, though currently operates on dense results only.

### Reranking Algorithm
*   **Provider**: FlashRank (`FlashRank v0.2.10`)
*   **Model**: Local Cross-Encoder
*   **Threshold**: `0.80` strict cutoff.
*   **Role**: Takes the top retrieved documents and reranks them. Documents scoring below 0.80 are discarded unless explicitly uploaded by the user. If 0 documents pass, it triggers a query rewrite or web search.

## 4. Document Processing & Web Search

### Parsing Tools
*   **LlamaParse** (`llama-parse v0.6.94`): Used to parse unstructured user-uploaded PDFs and PDFs found during web search into clean Markdown.
*   **Docling** (`docling v2.107.0`): Used for high-fidelity conversion of local SEC filings (HTML/PDF), preserving complex table structures.

### Chunking Strategy
*   **Algorithm**: LangChain `RecursiveCharacterTextSplitter`
*   **Chunk Size**: 1024 characters
*   **Overlap**: 200 characters

### Web Search
*   **Provider**: Tavily (`tavily-python v0.7.27`)
*   **Parameters**: `search_depth="advanced"`, `max_results=3`
*   **Workflow**: When local retrieval fails (3 retries exhausted), the system pauses for human consent. Upon approval, Tavily scrapes the live web. The raw text/PDFs are chunked (1024 chars) and ingested into Pinecone under the active `session_id`. The workflow then loops back to Pinecone retrieval.

## 5. Session Memory and Checkpointing

*   **Technology**: SQLite (`checkpoints.sqlite`)
*   **Concept**: LangGraph `SqliteSaver` captures the complete `AgentState` at every node transition.
*   **Role**: Provides conversational memory (chat history) and enables the execution graph to pause (for Human-in-the-Loop web search) and resume flawlessly.
*   **Query Cache**: A simple JSON-based exact-match cache (`query_cache.json`) intercepts repeated queries at the Router node to bypass execution entirely.
