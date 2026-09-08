# FinSight: Functional Specification & Architecture Blueprint

This document outlines the precise operations, state management, and lifecycle of a request as it passes through the FinSight Agentic RAG architecture.

## 1. State Definition (`AgentState`)
The core data structure passed between all nodes in the LangGraph pipeline is the `AgentState` typed dictionary. It holds the mutable memory of the current request:
- `question`: The original user query.
- `current_query`: The active search query (which may be rewritten by the agent).
- `plan`: A multi-step breakdown of how to solve the query.
- `documents`: The top N chunks retrieved from Pinecone/Flashrank.
- `all_documents`: An accumulation of all documents fetched across multiple search attempts.
- `generation`: The drafted LLM response.
- `retry_count`: The number of times the agent has hallucinated or rewritten the query.
- `web_search_attempted`: Boolean flag indicating if Tavily was called.
- `route`: The decision from the human-in-the-loop (e.g., "surf" or "END").

## 2. Pipeline Execution Flow (Node Lifecycle)

### Phase 1: Planning and Routing
1. **API Trigger**: The user submits a query to the FastAPI endpoint.
2. **`plan_query`**: The primary LLM analyzes the user's question and writes a step-by-step strategy for how to research the answer. The query is placed into the `current_query` state variable.

### Phase 2: Retrieval and Aggressive Reranking
3. **`retrieve`**: The agent queries the **Pinecone Serverless** database using dense embeddings (`all-MiniLM-L6-v2`). It pulls a wide net of 40 potential chunks.
4. **`rerank_documents`**: Because vector search can be noisy, the 40 chunks are passed to a local **Flashrank** cross-encoder. Flashrank evaluates the semantic relationship between the exact `current_query` and each chunk, discarding irrelevant text and keeping only the top 7 highly precise chunks.

### Phase 3: Generation and Self-Reflection
5. **`generate`**: The primary LLM is given the user's question and the top 7 chunks. It drafts an initial answer with citations.
6. **`check_hallucinations_and_answer`**: This is a conditional routing node powered by an LLM "Grader". It asks two questions:
   - *Is the generated answer grounded in the retrieved documents?* (Faithfulness)
   - *Does the answer actually address the user's question?* (Answer Relevancy)

### Phase 4: Conditional Routing
Based on the Grader's output from Phase 3, the graph routes to one of three paths:
- **Path A (Success)**: If the answer is grounded and relevant, the pipeline reaches `END` and returns the answer to the user.
- **Path B (Hallucination)**: If the answer hallucinates facts not in the context, the pipeline routes to **`rewrite_query`**. The LLM refines the search query and loops back to **Phase 2 (Retrieval)**. This loop repeats until the retry threshold is hit.
- **Path C (Missing Context)**: If the documents simply do not contain the answer, the pipeline routes to **`ask_human_consent`**.

### Phase 5: Autonomous Web Search (Fallback)
7. **`ask_human_consent`**: A human-in-the-loop interrupt. The agent asks the user for permission to browse the web for the missing information.
8. **`surf_and_ingest_node`**: If consent is granted, the agent uses **Tavily** to find relevant URLs.
   - For standard HTML pages, Tavily extracts the raw text.
   - For `.pdf` files (like SEC 10-Ks), the agent downloads the raw file and passes it to **LlamaParse** to extract complex tables into structured Markdown.
   - The text is chunked using LangChain and ingested live into Pinecone.
9. **Loop Back**: The graph routes back to **`retrieve`** (Phase 2). Because the new web documents are now in Pinecone, the agent retrieves them, reranks them, and successfully generates the answer.
