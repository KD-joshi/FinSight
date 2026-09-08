# FinSight Reading Guide

If you are a new developer or looking to understand how FinSight operates under the hood, we recommend reading the codebase in the following chronological order. This mimics the actual lifecycle of a user request passing through the system.

## 1. Core Configuration
* **`config/settings.py`**: Start here. This file uses `pydantic-settings` to load all environment variables, API keys, model definitions, and system defaults. Understanding this file gives you the overarching context of what external services the app uses.

## 2. API Entrypoint
* **`src/finsight/main.py`**: This is the FastAPI backend application. Look at the `/chat` or `/query` endpoint to see how a user's prompt is received and how the LangGraph pipeline is triggered.

## 3. The Brain (LangGraph Orchestration)
* **`src/finsight/agents/graph.py`**: This is the most important file in the project. It defines the state machine (`AgentState`) and wires together the directed acyclic graph (DAG) using LangGraph. Study the `add_edge` and `add_conditional_edges` functions to understand how the agent loops, self-reflects, and routes traffic.

## 4. The Agent Actions
* **`src/finsight/agents/nodes.py`**: While `graph.py` defines the *structure* of the flow, this file contains the actual Python functions (nodes) that execute at each step (e.g., generating the plan, rewriting the query, drafting the answer).

## 5. Retrieval & Reranking
* **`src/finsight/rag/retriever.py`**: See how documents are fetched from the Pinecone vector database using dense embeddings and sparse BM25 keyword matching.
* Look closely at how the retrieved chunks (usually 40) are passed to the local `Flashrank` cross-encoder to be aggressively filtered down to the top 7.

## 6. Autonomous Web Fallback
* **`src/finsight/tools/web_surfer.py`**: If the agent decides it lacks context, it calls this script. Read this to understand how Tavily fetches live URLs, how raw text is ingested, and how complex SEC PDFs are bypassed through `LlamaParse` for markdown table extraction.

## 7. The LLM Waterfall
* **`src/finsight/utils/llm_provider.py`**: Understand the multi-tier LLM fallback logic. This file wraps the LLM initializations and sets up a robust failover chain (Groq → Gemini → Cohere) to guarantee high availability during API rate limits.

## 8. Evaluation
* **`src/finsight/evaluation/evaluator.py`**: Finally, review how the system's performance is autonomously graded. This file bypasses the human-in-the-loop web search interrupts and runs the synthetic datasets through the RAGAS framework to compute Answer Relevancy, Faithfulness, and Context Precision.
