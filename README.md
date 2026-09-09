# FinSight — Agentic RAG for Financial Intelligence

> Production-grade agentic RAG system for SEC filings and financial documents, powered by LangGraph, Pinecone, and multi-tier LLMs.

FinSight is an advanced autonomous research assistant designed to answer complex financial queries. Unlike basic RAG systems, it uses an agentic workflow to formulate multi-step plans, verify answers, and autonomously browse the web when internal documents are insufficient.

## ✨ Features
- **Agentic Orchestration:** Uses LangGraph to dynamically route queries, formulate research plans, and self-reflect to ensure accuracy.
- **Autonomous Web Search:** If local knowledge is insufficient, the agent asks for consent, searches the web via Tavily, parses complex SEC PDFs via LlamaParse, and ingests them into the local database on the fly.
- **Multi-Tier LLM Fallback:** Built-in resilience with a waterfall fallback system (Groq → Gemini → Cohere) to guarantee high availability even during API rate limits.
- **Hybrid Retrieval System:** Combines Pinecone Serverless vector search with a local Flashrank cross-encoder for extremely precise context retrieval.
- **Self-Grading & Hallucination Checks:** Every generated answer is graded against the retrieved context before being shown to the user.

### Overview & Capabilities
- **What it does**: FinSight is a specialized Agentic RAG (Retrieval-Augmented Generation) search engine that answers complex finance-related questions for any company by leveraging advanced AI search capabilities.
- **Accurate Insights**: To ensure precision and prevent hallucinations, it exclusively answers queries using real, scraped financial data securely stored in a Pinecone vector database.
- **Custom Document Analysis**: Users can easily upload their own personal or proprietary financial documents to securely query and extract insights from their private data.

## 🏗️ Architecture

```text
User Query → Router → Planner → Hybrid Retriever (Pinecone + BM25) → Cross-Encoder Reranker (Flashrank) → Grader → [Rewriter loop / Web Search] → Generator → Cited Answer
```

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| **LLM Inference** | Groq (`gpt-oss-120b`), Gemini Flash, Cohere (3-tier fallback) |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) |
| **Vector DB** | Pinecone Serverless |
| **Orchestration** | LangGraph |
| **RAG Pipeline** | LangChain |
| **Document Parsing** | LlamaParse |
| **Web Search** | Tavily API |
| **Observability** | LangSmith |
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
# Ingest initial SEC filings (Optional, system can fetch on the fly)
python -m finsight.ingestion.sec_downloader --tickers AAPL TSLA GOOGL

# Run the RAG API backend
python -m finsight.main

# Generate Synthetic Evaluation Dataset
PYTHONPATH=.:src python src/finsight/evaluation/generate_dataset.py

# Run LangSmith evaluation suite
PYTHONPATH=.:src python src/finsight/evaluation/langsmith_eval.py
```

## 📊 Evaluation Results

Our multi-tier agentic architecture (Pinecone + LlamaParse + Flashrank) was evaluated against a suite of highly complex, multi-hop financial queries using RAGAS.

| Metric | Score | Description |
|--------|-------|-------------|
| **Answer Relevancy** | **89.3%** | Evaluates how directly the generated answer addresses the user's query, avoiding tangential information. |
| **Faithfulness** | **98.2%** | Measures the factual consistency of the generated answer against the retrieved documents (minimizes hallucinations). |
| **Context Precision** | **95.5%** | Evaluates whether all of the ground-truth relevant items present in the contexts are ranked higher than irrelevant ones. |

*(Note: Faithfulness and Context Precision reflect post-reranker synthesis scores, achieving near-perfect grounding due to our autonomous web-search fallback and LlamaParse chunking).*

## 📝 License

MIT
