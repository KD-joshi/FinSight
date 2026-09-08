# FinSight Technical Specification

This document details the exact technologies, models, and third-party services utilized in the production deployment of FinSight.

## 1. Large Language Models (LLMs)
The system utilizes a 4-tier waterfall fallback strategy to ensure 100% uptime during high-volume document reasoning or when hitting strict free-tier API rate limits.

1. **Primary LLM**: `groq/compound`
   - **Provider**: Groq Cloud
   - **Purpose**: The primary workhorse for all planning, routing, grading, and answering. Selected for its massive 70,000 Tokens-Per-Minute limit to survive heavy RAG evaluation loops.
2. **Fallback 1 (High Intelligence)**: `openai/gpt-oss-120b` (Groq Hosted)
   - **Provider**: Groq Cloud
   - **Purpose**: Triggers if the primary model fails. Provides 120-billion parameter reasoning capabilities for highly complex financial comparisons.
3. **Fallback 2**: `gemini-3.8-flash`
   - **Provider**: Google AI Studio
   - **Purpose**: Triggers if the entire Groq cloud is rate-limited.
4. **Fallback 3**: `command-a-03-2025`
   - **Provider**: Cohere
   - **Purpose**: The tertiary safety net.

## 2. Retrieval & Embeddings
- **Dense Embedding Model**: `all-MiniLM-L6-v2` (HuggingFace)
  - **Dimension**: 384
  - **Purpose**: Generates fast, lightweight vector embeddings for all ingested text chunks.
- **Sparse Embeddings**: BM25
  - **Purpose**: Used for exact keyword matching (highly important for financial tickers and specific accounting terms).
- **Vector Database**: Pinecone Serverless
  - **Purpose**: Stores the hybrid index (dense + sparse) for cloud-based scalability.
- **Cross-Encoder Reranker**: `Flashrank`
  - **Purpose**: Runs locally to syntactically score the top 40 vector results against the exact user query, filtering them down to a highly precise top 7 chunks.

## 3. Orchestration & Pipeline
- **Orchestration Engine**: `LangGraph`
  - **Purpose**: Defines the cyclical state machine. Allows the agent to route conditionally, self-reflect on its answers, and loop back if it hallucinates.
- **RAG Utilities**: `LangChain`
  - **Purpose**: Provides standard wrappers for ChatModels, PromptTemplates, and RecursiveCharacterTextSplitters.

## 4. Autonomous Data Ingestion
- **Web Search Engine**: Tavily API
  - **Purpose**: Actively queries the internet (with `search_depth="advanced"`) to find relevant SEC filings and HTML articles when the local Pinecone DB lacks the answer.
- **Document Parser**: LlamaParse
  - **Purpose**: Uses vision-language models to parse complex SEC `.pdf` documents directly from the web, converting dense balance sheets into clean Markdown tables before chunking.

## 5. Backend & Evaluation
- **API Framework**: FastAPI
  - **Purpose**: Serves the asynchronous backend endpoints for the chat interface.
- **Evaluation Suite**: RAGAS
  - **Purpose**: Runs synthetic datasets through the pipeline to automatically calculate Answer Relevancy, Faithfulness, and Context Precision using LLM-as-a-judge techniques.
- **Observability**: Langfuse
  - **Purpose**: Traces every LangGraph node execution, LLM token usage, and latency metric for debugging in production.
