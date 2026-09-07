# FinSight — Agentic RAG for Financial Intelligence

> Production-grade agentic RAG system for SEC filings and financial documents, powered by LangGraph + LangChain.

### Overview & Capabilities
- **What it does**: FinSight is a specialized Agentic RAG (Retrieval-Augmented Generation) search engine that answers complex finance-related questions for any company by leveraging advanced AI search capabilities.
- **Accurate Insights**: To ensure precision and prevent hallucinations, it exclusively answers queries using real, scraped financial data securely stored in a Pinecone vector database.
- **Custom Document Analysis**: Users can easily upload their own personal or proprietary financial documents to securely query and extract insights from their private data.

### Architecture
- **Agentic Orchestration**: Uses LangGraph to dynamically route user queries, formulate multi-step plans, and conditionally loop through retrieval, grading, and rewriting phases to guarantee accurate answers.
- **Hybrid Retrieval System**: Combines a robust retrieval architecture backed by Pinecone serverless vector DB with a 3-tier LLM fallback system (Groq, Gemini, Cohere) for highly available, cited answer generation.

```text
User Query → Router → Planner → Hybrid Retriever (Pinecone + BM25) → Grader → [Rewriter loop] → Generator → Cited Answer
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **LLM Inference** | Groq (`gpt-oss-120b`), Gemini Flash, Cohere (3-tier fallback) |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) |
| **Vector DB** | Pinecone Serverless |
| **Orchestration** | LangGraph |
| **RAG Pipeline** | LangChain |
| **Document Parsing** | LlamaParse |
| **Web Search** | Tavily API |
| **Observability** | Langfuse |
| **API Backend** | FastAPI |
| **Frontend UI** | Next.js (React, TypeScript) |

## 📦 Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Fill in your API keys in the `.env` file
```

## 🔑 Required API Keys

1. **Groq**: https://console.groq.com
2. **Google AI Studio**: https://aistudio.google.com
3. **Cohere**: https://dashboard.cohere.com
4. **Pinecone**: https://app.pinecone.io
5. **Tavily**: https://tavily.com
6. **LlamaParse**: https://cloud.llamaindex.ai

## 🚀 Quick Start

```bash
# Ingest SEC filings
python -m finsight.ingestion.sec_downloader --tickers AAPL TSLA GOOGL

# Run the RAG system
python -m finsight.main

# Run evaluation
python -m finsight.evaluation.run_eval
```

## 📝 License

MIT
