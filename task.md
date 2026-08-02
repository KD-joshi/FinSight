# 📋 FinSight — Task Tracker

## Week 1: Foundation — Data Pipeline + Basic RAG
- [x] Set up project structure (Python packaging, venv, Git, .env)
- [x] Sign up for free accounts: Groq, Google AI Studio, Qdrant Cloud, Langfuse *(USER action)*
- [x] Build SEC filing downloader (`ingestion/sec_downloader.py`)
- [x] Build document parser — PDF → structured markdown (`ingestion/document_processor.py`)
- [x] Implement chunking with metadata (company, filing type, fiscal year)
- [x] Connect to Gemini Embedding API (`utils/embeddings.py`)
- [x] Build Qdrant Cloud vector store manager (`ingestion/vector_store.py`)
- [x] Build LLM provider with Groq → Gemini fallback (`utils/llm_provider.py`)
- [x] Build hybrid retriever — Vector + BM25 + RRF (`rag/retriever.py`)
- [x] Build RAG chains — grader, rewriter, generator, router, planner (`rag/chains.py`)
- [x] Build LangGraph agentic pipeline — full Corrective RAG (`agents/graph.py`)
- [x] Build CLI entry point with Rich UI (`main.py`)
- [x] Install dependencies and test with sample SEC filings *(next step)*

## Week 2: Hybrid Search + Agentic Loop (LangGraph)
- [x] Add BM25 sparse retriever
- [x] Implement Reciprocal Rank Fusion (RRF)
- [x] Build LangGraph StateGraph (Router → Planner → Retriever → Grader → Rewriter → Generator)
- [x] Add conditional edges with retry limits
- [x] Add human-in-the-loop
- [x] Test with multi-hop financial questions

## Week 3: Evaluation + Observability
- [x] Create gold-standard eval set (20-30 Q&A pairs)
- [x] Integrate RAGAS evaluation
- [x] Integrate Langfuse tracing
- [x] Add statement-level attribution
- [x] Add metadata filtering
- [x] Build automated eval pipeline

## Week 4: API + UI + Deployment
- `[x]` Build FastAPI backend
  - `[x]` Define Pydantic schemas (`src/finsight/api/schemas.py`)
  - `[x]` Setup Dependency Injection (`src/finsight/api/dependencies.py`)
  - `[x]` Implement Chat Route (`src/finsight/api/routes.py`)
  - `[x]` Initialize FastAPI App (`src/finsight/api/main.py`)
  - `[x]` Test Backend Endpoint
- [ ] Build Streamlit frontend
- [ ] Add Yahoo Finance stock data integration
- [ ] Add conversation memory
- [ ] Write README with architecture diagram
- [ ] Deploy to HuggingFace Spaces

## Week 5: Polish
- [ ] Docker-compose setup
- [ ] Rate limit handling + Gemini fallback
- [ ] Blog post
- [ ] Demo video
- [ ] Clean up repo

## Week 6: Fine-tuning SFT + DPO (Colab)
- [ ] Curate financial dataset
- [ ] SFT on financial Q&A
- [ ] Create preference pairs
- [ ] DPO training
- [ ] Benchmark comparisons

## Week 7: GRPO + Integration
- [ ] GRPO training
- [ ] Full comparison: Base → SFT → DPO → GRPO
- [ ] Upload to HuggingFace Hub
- [ ] Swap into FinSight, run RAGAS before/after
- [ ] Blog post on fine-tuning

## Final: Public Frontend
- [ ] Build polished React/Next.js frontend
- [ ] Deploy on Vercel (free tier)
- [ ] Connect to FastAPI backend
