import sys
import logging

logging.basicConfig(level=logging.INFO)

from finsight.utils.llm_provider import get_llm
from finsight.rag.retriever import HybridRetriever
from finsight.agents.graph import build_rag_agent

# Initialize LLM and retriever
llm = get_llm()
retriever = HybridRetriever()

# Build agent graph
graph = build_rag_agent(retriever=retriever, llm=llm, fallback_llm=llm)

query = "what all data do u have in your backend"
print(f"Testing query: {query}")

result = graph.invoke({
    "question": query,
    "max_retries": 1
}, config={"configurable": {"thread_id": "test_1"}})

print("\n==== Final Answer ====\n")
print(result.get("generation"))

print("\n==== Test Cache Hit ====\n")
result2 = graph.invoke({
    "question": query,
    "max_retries": 1
}, config={"configurable": {"thread_id": "test_2"}})
print(result2.get("generation"))
