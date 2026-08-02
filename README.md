# FinSight — Agentic RAG for Financial Intelligence

> Production-grade agentic RAG system for SEC filings and financial documents, powered by LangGraph + LangChain.

## 🏗️ Architecture

```
User Query → Router → Planner → Hybrid Retriever (Qdrant + BM25) → Grader → [Rewriter loop] → Generator → Cited Answer
```

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| LLM Inference | Groq API (Llama 3.1 70B) + Gemini Flash (fallback) |
| Embeddings | Google Gemini (text-embedding-004) |
| Vector DB | Qdrant Cloud |
| Orchestration | LangGraph |
| RAG Pipeline | LangChain |
| Evaluation | RAGAS |
| Observability | Langfuse |
| API | FastAPI |
| UI | Streamlit → React (Vercel) |

## 📦 Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Fill in your API keys (all free tier)
```

## 🔑 Required API Keys (All Free, No Credit Card)

1. **Groq**: https://console.groq.com → Get API Key
2. **Google AI Studio**: https://aistudio.google.com → Get API Key
3. **Qdrant Cloud**: https://cloud.qdrant.io → Create free cluster → Get URL + API Key
4. **Langfuse**: https://cloud.langfuse.com → Sign up → Get public/secret keys

## 🚀 Quick Start

```bash
# Ingest SEC filings
python -m finsight.ingestion.sec_downloader --tickers AAPL TSLA GOOGL

# Run the RAG system
python -m finsight.main

# Run evaluation
python -m finsight.evaluation.run_eval
```

## 📊 Evaluation Results

| Metric | Score |
|--------|-------|
| Faithfulness | TBD |
| Context Precision | TBD |
| Answer Relevance | TBD |

## 📝 License

MIT
