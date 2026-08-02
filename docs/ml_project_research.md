# 🔬 ML Project Research Report — June 2026

> Deep research into the best ML projects for interviews, skill demonstration, and real-world value.  
> Compiled from 11+ targeted web searches — all sourced from **2026 publications and trends**.

---

## 📊 What The Market Actually Demands (2026)

### Top In-Demand ML Skills — June 2026

| Rank | Skill Area | 2026 Context |
|------|-----------|--------------|
| 1 | **Generative AI & LLMs** | RAG pipelines, fine-tuning (GRPO/DPO), agentic workflows, prompt engineering |
| 2 | **Agentic AI & Multi-Agent Systems** | MCP + A2A protocols, LangGraph orchestration, tool use, guardrails |
| 3 | **MLOps & LLMOps** | Langfuse/LangSmith observability, Evidently drift monitoring, CI/CD for ML |
| 4 | **Deep Learning & Foundation Models** | PyTorch, Transformers, vision foundation models (SAM 2, Florence-2) |
| 5 | **Responsible AI & Governance** | AI safety, hallucination reduction, RBAC, compliance, evaluation frameworks |

### Critical Hiring Shifts in 2026

> [!IMPORTANT]
> **The "Application Gap"** — The biggest employer challenge in 2026 is NOT finding people who understand AI theory, but finding those who can **apply** it. End-to-end projects with deployment + monitoring + evaluation are now **non-negotiable**.

- **Skills-first hiring** is now standard — portfolios of real-world projects outweigh credentials
- Market has fragmented into **specialized roles**: GenAI Infrastructure Engineer, NLP Engineer, MLOps Specialist, Agent Developer
- Employers screen heavily for **soft skills** (communication, trade-off reasoning, systems thinking)
- ML from "prediction" to **"action"** — systems that trigger real-world consequences, not just classify

---

## 🏢 What FAANG & Top Companies Want in 2026 Portfolios

### The Non-Negotiables

1. **End-to-End Execution** — Data → Training → Evaluation → **Deployment → Monitoring**. Jupyter notebooks alone = instant reject.
2. **Production Mindset** — Docker, FastAPI/Streamlit, cloud platforms, observability tooling
3. **Business Impact Framing** — "My model reduced X by Y%" not "My model got Z% accuracy"
4. **Trade-off Documentation** — Why you chose this model over that. Cost vs performance. Latency budget.
5. **Quality > Quantity** — 3–5 polished projects >>> 15 unfinished notebooks
6. **Live Demos** — Recruiters spend <10 seconds. A clickable link on HuggingFace Spaces wins.

### ❌ Avoid vs ✅ Include

| Feature | ❌ 2026 Red Flags | ✅ 2026 Green Flags |
|---------|------------------|-------------------|
| **Data** | Pre-cleaned Kaggle (Titanic, MNIST) | Self-scraped, messy, real-world data |
| **Dev** | Jupyter-only, tutorial clones | Modular Python, clean repo structure |
| **Deploy** | "Works on my machine" | Docker + FastAPI + cloud + live URL |
| **Eval** | "It works" | RAGAS scores, LLM-as-a-judge, A/B metrics |
| **Monitoring** | None | Langfuse traces, Evidently drift reports |

---

## 🏆 Top 5 Projects — Ranked by 2026 Interview Impact

---

### 🥇 #1: Production-Grade Agentic RAG System

> **Category**: LLM + RAG + Agents · **Time**: 4–6 weeks · **Impact**: ★★★★★

The RAG landscape in 2026 has evolved from linear "retrieve-then-generate" to **cyclic agentic systems** that plan, retrieve, grade, and self-correct.

**What you build**: A document intelligence system that ingests PDFs/docs, uses **hybrid retrieval** (vector + BM25 with Reciprocal Rank Fusion), implements **Corrective RAG** (auto-regrading and re-querying on low relevance), and adds an **agentic layer** using LangGraph for multi-step reasoning.

**2026-Accurate Tech Stack**:
- `LangGraph` — agentic orchestration (industry standard for stateful production agents)
- `LlamaIndex` or `LangChain` — RAG pipeline
- `Qdrant` / `pgvector` / `Pinecone` — vector database
- Hybrid Search: Dense (embeddings) + Sparse (BM25) + `Cohere Rerank`
- `RAGAS` — RAG evaluation (Faithfulness, Context Precision, Answer Relevance)
- `Langfuse` — LLM observability, tracing, prompt versioning, cost tracking
- `FastAPI` + `Docker` — production serving
- `Streamlit` / `Gradio` — demo UI

**2026 RAG State-of-the-Art Patterns**:

| Feature | 2023–2025 Era | 2026 State of the Art |
|---------|---------------|----------------------|
| **Workflow** | Linear (Query → Retrieve → Generate) | Cyclic (Agentic Loop: Plan → Retrieve → Grade → Reflect) |
| **Reasoning** | Single-hop | Multi-hop via Graph RAG |
| **Adaptability** | Fixed retrieval | Self-correcting (Corrective RAG, Adaptive RAG) |
| **Search** | Naive vector search | Hybrid Search + Reranking + RRF |
| **Evaluation** | "It looks right" | RAGAS metrics + LLM-as-a-judge in production |
| **Attribution** | Generic citations | Statement-level attribution |

**Why it dominates 2026 interviews**:
- **#1 most deployed enterprise use case** — literally what every company is building
- Shows you understand the **full 2026 RAG stack**: agentic loops, hybrid search, reranking, evaluation, observability
- Corrective RAG and self-grading retrieval are **cutting-edge patterns** most candidates don't know
- Langfuse integration shows production-grade LLMOps thinking

**Interview talking points**:
- Agentic RAG vs naive RAG — when the agent decides NOT to retrieve
- Corrective RAG: grading retrieved documents and triggering re-query
- Graph RAG for multi-hop reasoning over interconnected knowledge
- Hybrid search (dense + sparse + RRF) vs pure vector search
- RAGAS evaluation: Faithfulness, Context Precision, Answer Relevance
- Statement-level attribution vs generic citations
- Semantic caching for cost optimization
- Langfuse tracing for debugging non-deterministic agent loops

---

### 🥈 #2: Multi-Agent System with MCP & A2A Protocols

> **Category**: AI Agents · **Time**: 4–5 weeks · **Impact**: ★★★★★

In 2026, the agent ecosystem has converged on **two standardized protocols** — this is the defining trend of the year.

**What you build**: A multi-agent research & task automation system where specialized agents (Researcher, Analyzer, Writer) coordinate via the **A2A protocol**, each connecting to external tools via **MCP servers**. The host agent orchestrates, manages memory, and implements human-in-the-loop approvals.

**2026-Accurate Tech Stack**:
- `LangGraph` — stateful graph-based agent orchestration (checkpointing, human-in-the-loop)
- **MCP (Model Context Protocol)** — universal agent-to-tool connectivity ("USB-C for AI")
- **A2A (Agent-to-Agent Protocol)** — inter-agent communication (adopted by 150+ orgs, Linux Foundation)
- `Langfuse` — agent observability, trace replay, cost tracking
- OpenAI / Anthropic / local LLMs via Ollama
- `Streamlit` / `Gradio` — interactive demo
- `Docker` — deployment

**2026 Agent Framework Landscape**:

| Framework | Best For | Key Characteristic |
|-----------|---------|-------------------|
| **LangGraph** | Stateful production agents | Graph-based, checkpointing, human-in-the-loop |
| **CrewAI** | Multi-agent prototypes | Intuitive role-based teams |
| **OpenAI Agents SDK** | Platform-native agents | Streamlined OpenAI-specific experience |
| **smolagents** | Fast code-first builds | Lightweight, zero-bloat (HuggingFace) |
| **AutoGen / AG2** | Research-style collaboration | Multi-agent conversational loops |

**Why it dominates 2026 interviews**:
- **AI agents are THE defining trend of 2026** — every company is exploring agent deployment
- MCP and A2A are **new industry standards** that most candidates haven't built with yet
- Shows multi-agent orchestration, guardrails, memory management
- Demonstrates understanding of **when agents are overkill vs when they add value**

**Interview talking points**:
- MCP vs A2A — tool connectivity vs agent coordination (complementary protocols)
- Deterministic guardrails vs LLM-only reasoning for mission-critical steps
- Human-in-the-loop patterns and escalation boundaries
- Agent memory architectures (conversation buffer, summary, vector-backed)
- Trace replay for debugging non-deterministic agent loops
- Cost management strategies for agent loops (token budgets, early stopping)

---

### 🥉 #3: Real-Time Multi-Stage Recommendation Engine

> **Category**: RecSys + System Design · **Time**: 5–7 weeks · **Impact**: ★★★★★

The Two-Tower architecture remains the **2026 industry standard** for large-scale candidate retrieval, now enhanced with late-interaction mechanisms and real-time feature stores.

**What you build**: A full FAANG-style recommendation pipeline — Two-Tower model for candidate retrieval → HNSW index for sub-millisecond ANN search → Deep ranking model → Real-time feature store for live user behavior.

**2026-Accurate Tech Stack**:
- `PyTorch` — Two-Tower model training (user tower + item tower)
- `FAISS` / `Milvus` / `Redis` — HNSW vector index for retrieval stage
- `XGBoost` / deep ranker — secondary ranking model
- `Feast` — real-time feature store (latest user clicks/behavior)
- `Kafka` — event streaming
- `FastAPI` — serving layer
- `Docker` + `Redis` (caching) — production infrastructure

**2026 Two-Tower Innovations**:
- **"Fully Interacted" Two-Tower**: Late-interaction mechanisms and meta-query modules for more expressive feature crossing without sacrificing retrieval speed
- **AQR-HNSW**: Density-aware quantization reducing memory by 75% while maintaining recall
- **Online Feature Stores**: Sub-second feature freshness ensuring user embeddings reflect real-time behavior

**Why it dominates 2026 interviews**:
- The **bread and butter of FAANG companies** — most tech company revenue depends on recommendations
- **ML System Design interviews at Google/Meta/Netflix** literally ask you to design this
- Shows multi-stage architecture thinking at scale
- Demonstrates awareness of latency budgets, ANN trade-offs, cold-start solutions

**Interview talking points**:
- Two-Tower: why decouple user and item towers? Inference speed vs expressiveness
- HNSW vs IVF vs PQ — ANN algorithm trade-offs (recall vs speed vs memory)
- Cold-start: content-based fallback, explore/exploit, contextual bandits
- Latency budget allocation: 50ms for retrieval, 30ms for ranking, etc.
- Online vs offline features — why feature freshness matters
- Business metrics: CTR, diversity, serendipity vs pure NDCG/RMSE

---

### 4️⃣ #4: Edge-Deployed Computer Vision with Foundation Models

> **Category**: Computer Vision + Edge · **Time**: 4–6 weeks · **Impact**: ★★★★☆

The 2026 CV landscape is defined by **foundation models** (SAM 2, Florence-2) replacing task-specific pipelines, and the push toward **Visual General Intelligence (VGI)**.

**What you build**: A multi-task vision system using Florence-2 as a unified model (detection + OCR + captioning via natural language prompts), with SAM 2 for video object tracking. Deploy an optimized version to edge with quantization.

**2026-Accurate Tech Stack**:
- `Florence-2` — unified vision-language model (prompt-based multi-task: detect, OCR, caption)
- `SAM 2` — real-time video segmentation with temporal memory
- `YOLOv8/v11` — real-time object detection (if needed)
- `TensorRT` / `ONNX Runtime` / `TensorFlow Lite` — model optimization
- `OpenCV` — video processing
- `Docker` — containerization
- `Streamlit` — interactive demo

**2026 Computer Vision State of the Art**:

| Trend | Description |
|-------|-------------|
| **Foundation Models** | Florence-2, SAM 2 replace building custom CNNs from scratch |
| **Agentic Vision** | Systems that reason about and act on visual environments |
| **World Models** | V-JEPA, NVIDIA Cosmos — predicting consequences of actions in physical spaces |
| **Continuous Learning** | Pipelines that learn from new data streams, preventing drift |

**Why it stands out in 2026**:
- **Extremely rare in portfolios** — most candidates still only train basic classifiers
- Florence-2 as a unified "Vision Agent" is cutting-edge
- SAM 2's temporal video understanding is the new standard
- Shows hardware-aware optimization skills (quantization, pruning)

**Interview talking points**:
- Florence-2: treating vision tasks as language problems — paradigm shift
- SAM 2: streaming memory module for temporal consistency across video frames
- INT8 vs FP16 quantization: accuracy impact and when each is appropriate
- World Models: moving from "what is in the image?" to "what will happen next?"
- Closed-loop systems: human-in-the-loop for low-confidence predictions → continuous retraining

---

### 5️⃣ #5: Foundation Model Time Series Forecasting

> **Category**: Time Series · **Time**: 3–5 weeks · **Impact**: ★★★★☆

Time series foundation models have moved from experimental to **production-viable** in 2026, with Chronos-2, TimesFM 2.5, and Moirai-2 leading the field.

**What you build**: A forecasting system that uses **zero-shot and few-shot** foundation models on cold-start problems (no historical training data), benchmarks against classical methods (ARIMA, Prophet, XGBoost), and adds **conformal prediction** for uncertainty quantification.

**2026-Accurate Tech Stack**:
- `Amazon Chronos-2` — T5-based, 300+ forecasts/sec, multivariate + covariate support
- `Google TimesFM 2.5` — 16k context window, in-context fine-tuning without weight updates
- `Salesforce Moirai-2` — MoE architecture, high efficiency
- `MAPIE` — conformal prediction for uncertainty intervals
- `MLflow` — experiment tracking
- `Streamlit` — interactive dashboard
- `Docker` — deployment

**2026 Time Series Foundation Model Landscape**:

| Model | Architecture | Best For |
|-------|-------------|---------|
| **Chronos-2** | T5 (Encoder-Decoder) | Production-scale, multivariate, covariate-rich tasks |
| **TimesFM 2.5** | Decoder-only | Enterprise (BigQuery), few-shot adaptability |
| **Moirai-2** | Mixture-of-Experts | High efficiency, lower compute overhead |

**Why it stands out in 2026**:
- Foundation models for time series are **production-ready in 2026** — most candidates still use only ARIMA/Prophet
- In-context fine-tuning (TimesFM 2.5) is a genuinely new concept
- Conformal prediction for uncertainty is next-level
- Massive business value: demand forecasting, energy, inventory

**Interview talking points**:
- Zero-shot vs few-shot vs in-context fine-tuning — when to use each
- Why Chronos-2 treats forecasting as a language modeling task
- Conformal prediction vs Bayesian uncertainty — practical trade-offs
- When to use foundation models vs classical approaches (well-behaved data → classical may win)
- Concept drift detection and online learning for production
- Quantile-based objectives for probabilistic forecasting

---

## 📋 More Strong Project Ideas (2026)

### LLM Fine-Tuning with GRPO/DPO Alignment
- Fine-tune Llama 3/4 or Mistral using **GRPO** (for reasoning) or **DPO** (for preferences) + LoRA
- Show the full **hybrid pipeline**: SFT → DPO → GRPO
- **2026 context**: GRPO reduces memory by ~50% vs PPO. DPO replaces traditional RLHF for most use cases.
- **Stack**: PyTorch, HuggingFace TRL, PEFT/LoRA, W&B, vLLM for serving
- **Time**: 3–4 weeks

### End-to-End MLOps Pipeline (Churn/Fraud)
- Full lifecycle: data versioning → experiment tracking → training → containerized API → CI/CD → drift monitoring
- **Stack**: DVC, MLflow, FastAPI, Docker, GitHub Actions, Evidently AI
- **Time**: 3–4 weeks
- **Why**: This is literally what ML engineers do at work every day

### Autonomous DevOps / IT Monitoring Agent
- Agent monitors logs/infra, detects anomalies, triggers diagnostic sub-agent, routes to remediation or human escalation
- **Stack**: LangGraph, MCP servers for cloud APIs, anomaly detection model, Slack/PagerDuty
- **Time**: 4–5 weeks
- **Why**: Combines classical ML (anomaly detection) with agentic AI — very practical

### AI-Powered Job Search Assistant (Agentic)
- Agent reads CV, searches live job boards, checks company pages, generates ranked fit report
- **Stack**: LangGraph, MCP for web browsing tools, A2A for multi-agent coordination
- **Time**: 3–4 weeks
- **Why**: Extremely relatable. Proves you can identify a pain point and build a complete product.

### Real-Time Fraud Detection System
- Real-time with strict latency SLAs, imbalanced data handling, streaming inference
- **Stack**: XGBoost/LightGBM, SMOTE, Kafka, FastAPI, SHAP explainability, Docker
- **Time**: 3–4 weeks
- **Why**: Every fintech needs this. Shows you handle class imbalance + latency constraints.

### Smart Surveillance / Safety Gear Detection
- Monitor footage for safety compliance (hard hats, vests). Human-in-the-loop for low-confidence predictions.
- **Stack**: YOLOv8/v11, OpenCV, Streamlit dashboard, ONNX/TensorRT for optimization
- **Time**: 3–4 weeks
- **Why**: Direct industrial application with clear business value

### Predictive Maintenance for IoT (Edge)
- Predict equipment failure from sensor data. Remaining Useful Life estimation. Edge-optimized.
- **Stack**: LSTM/Transformer, Scikit-learn, TF Lite for edge, Streamlit dashboard
- **Time**: 3–4 weeks
- **Why**: Classic "real-world" problem. Deploy to edge = major differentiator.

### RBAC-Enabled Enterprise RAG System
- RAG with role-based access control — different users see different documents based on permissions
- **Stack**: LangChain, pgvector, JWT auth, metadata filtering, FastAPI, audit logging
- **Time**: 4–5 weeks
- **Why**: Addresses #1 enterprise concern with RAG — data security and governance

---

## 🔌 2026 Protocol Landscape (NEW — Know This)

> [!IMPORTANT]
> These protocols are **defining the 2026 agent ecosystem**. Knowing them = major interview advantage.

| Protocol | Purpose | Analogy |
|----------|---------|---------|
| **MCP** (Model Context Protocol) | Agent ↔ Tool connectivity | "USB-C for AI" — universal tool integration |
| **A2A** (Agent-to-Agent Protocol) | Agent ↔ Agent coordination | "HTTP for agents" — cross-framework communication |
| **AG-UI / A2UI** | Agent ↔ User Interface | Runtime events between agents and dashboards |

- **MCP**: Adopted by Anthropic, now industry standard. Eliminates bespoke integration code.
- **A2A**: Adopted by 150+ orgs, donated to Linux Foundation. Agents discover each other's capabilities.
- Using these in a project = instant "this person knows what's current" signal.

---

## 🧬 2026 LLM Alignment Landscape (NEW — Know This)

| Method | Best For | Complexity | Resource Cost |
|--------|---------|-----------|--------------|
| **DPO** | General preference alignment | Low | Low (60–80% cheaper than PPO) |
| **GRPO** | Reasoning & verifiable tasks | Medium | Medium (~50% less memory than PPO) |
| **RLHF (PPO)** | Safety-critical / frontier models | High | High |
| **SimPO** | Maximum cost efficiency | Low | Very Low (no reference model) |
| **KTO** | Binary good/bad signals only | Low | Low |

> [!TIP]
> The 2026 consensus: **SFT → DPO → GRPO** hybrid pipeline. Data quality >>> data quantity.

---

## 💎 The "Mega Project" Strategy

> [!TIP]
> The absolute best strategy is to build **ONE project that combines multiple categories**.

### Intelligent Document Intelligence Platform
**Combines**: Agentic RAG (#1) + Multi-Agent (#2) + CV (Florence-2)

- **Agentic RAG pipeline**: Corrective RAG with self-grading, hybrid search + reranking
- **Multi-agent orchestration**: Specialized agents for parsing, retrieval, analysis via LangGraph
- **Multimodal**: Florence-2 for chart/image understanding in PDFs
- **MCP + A2A**: Agents connect to tools via MCP, coordinate via A2A
- **Evaluated**: RAGAS metrics in CI/CD pipeline
- **Observable**: Langfuse tracing, cost tracking, trace replay
- **Deployed**: FastAPI + Docker + live demo on HuggingFace Spaces

**This single project lets you discuss**: LLMs, agentic RAG, MCP/A2A protocols, multi-agent systems, multimodal AI, vector databases, hybrid search, evaluation, observability, deployment — covering **nearly every 2026 interview topic**.

**Estimated time**: 6–8 weeks for a polished, portfolio-ready version.

---

## 🎯 Recommended Portfolio Composition (Pick 4–5)

| Slot | Category | Example Projects | Purpose |
|------|---------|-----------------|--------|
| 1 | **GenAI / Agentic RAG** | Agentic RAG, Multi-Agent System | Shows you're current with 2026 |
| 2 | **MLOps / E2E** | Full pipeline with deployment + monitoring | Shows you can ship |
| 3 | **Classical ML** | Fraud Detection, Churn with SHAP | Shows fundamentals |
| 4 | **CV or Domain-Specific** | Florence-2 vision agent, Edge deployment | Shows depth |
| 5 | **Cutting-Edge** | Time Series Foundation Models, GRPO fine-tuning | Shows you push boundaries |

---

## 📝 How to Make Any Project "Hirable" in 2026

### README Formula
> **Problem → Approach → Architecture Diagram → Tech Stack (with rationale) → Results → How to Run → Live Demo Link**

### Resume Line Formula
> "Built [WHAT] using [TECH STACK] that [RESULT/IMPACT]"
> 
> *Example: "Built an agentic RAG system using LangGraph and Langfuse, achieving 94% faithfulness on RAGAS evaluation with sub-2s latency across 10K+ documents."*

### Key Advice from 2026 Hiring Managers
1. **Recruiters spend <10 seconds** scanning your profile → make READMEs scannable
2. **Solve the "unsexy" problems** → data cleaning, edge cases, drift monitoring impress more than fancy models
3. **Host live demos** on HuggingFace Spaces, Streamlit Cloud, or Vercel (all free)
4. **Include architecture diagrams** in every README
5. **Document trade-offs and failures** → interviewers love hearing what went wrong
6. **Write blog posts** explaining your projects → demonstrates communication skills
7. **Don't neglect DSA and ML theory** → projects get you interviews, fundamentals pass them
8. **3–5 polished projects** >>> 15 unfinished notebooks. Archive the rest.
9. **Align with your target industry** → fintech? build fraud detection. retail? build demand forecasting.

---

## 🛠️ Essential Tech Stack Reference (2026)

| Category | Must-Know Tools |
|----------|----------------|
| **Model Dev** | PyTorch, HuggingFace, Scikit-learn, XGBoost |
| **LLM/GenAI** | LangChain, LangGraph, LlamaIndex, OpenAI API |
| **Agents** | MCP, A2A protocol, LangGraph, CrewAI, smolagents |
| **Vector DBs** | Qdrant, pgvector, Pinecone, FAISS, Milvus |
| **Fine-Tuning** | HuggingFace TRL, PEFT/LoRA, Unsloth, GRPO/DPO |
| **Deployment** | FastAPI, Docker, Streamlit, HuggingFace Spaces |
| **LLMOps** | Langfuse, LangSmith, RAGAS, TruLens |
| **MLOps** | MLflow, DVC, GitHub Actions, Evidently AI |
| **Cloud** | AWS SageMaker, GCP Vertex AI, Azure ML |
| **CV (2026)** | Florence-2, SAM 2, YOLOv8/v11, OpenCV |
| **Time Series** | Chronos-2, TimesFM 2.5, Moirai-2, MAPIE |

---

## ⚡ Quick Decision Matrix

| Project | Build Time | Interview Impact | Uniqueness | Difficulty |
|---------|-----------|-----------------|------------|------------|
| Agentic RAG System | 4–6 wks | ★★★★★ | ★★★★☆ | Medium-Hard |
| Multi-Agent (MCP+A2A) | 4–5 wks | ★★★★★ | ★★★★★ | Hard |
| RecSys (Two-Tower) | 5–7 wks | ★★★★★ | ★★★★☆ | Hard |
| Edge CV (Florence-2/SAM2) | 4–6 wks | ★★★★☆ | ★★★★★ | Hard |
| Time Series Foundation | 3–5 wks | ★★★★☆ | ★★★★★ | Medium-Hard |
| LLM Fine-Tuning (GRPO) | 3–4 wks | ★★★★☆ | ★★★★☆ | Medium |
| MLOps E2E Pipeline | 3–4 wks | ★★★★☆ | ★★★☆☆ | Medium |
| Fraud Detection (RT) | 3–4 wks | ★★★★☆ | ★★★☆☆ | Medium |
| Predictive Maintenance | 3–4 wks | ★★★★☆ | ★★★☆☆ | Medium |
| Mega Combo Project | 6–8 wks | ★★★★★ | ★★★★★ | Hard |

---

*Sources: Scaler, DataCamp, Towards Data Science, ML Mastery, DataQuest, KDnuggets, CodeBasics, Medium, Reddit, HackerNews, Anthropic (MCP), Google (A2A), Databricks, Meta (SAM 2), Microsoft (Florence-2), DeepLearning.AI, FutureAGI — all from 2025–2026 publications.*
