import logging
from typing import Optional
from fastapi import Request

from finsight.utils.llm_provider import get_llm_with_fallback, get_fallback_llm
from finsight.ingestion.vector_store import get_vector_store
from finsight.rag.retriever import HybridRetriever
from finsight.agents.graph import build_rag_agent
from config.settings import settings

logger = logging.getLogger(__name__)

# Global cache for the agent
_agent_graph = None

def get_rag_agent():
    """FastAPI dependency to get the initialized LangGraph agent.
    
    Initializes the LLM, retriever, and graph exactly once and caches it.
    """
    global _agent_graph
    
    if _agent_graph is None:
        logger.info("Initializing LangGraph agent for the first time...")
        
        # Initialize LLM and fallback
        primary_llm = get_llm_with_fallback()
        fallback_llm = get_fallback_llm()
        
        # Initialize retriever
        store = get_vector_store()
        retriever = HybridRetriever(
            vector_store=store,
            documents=[], # BM25 sparse disabled for Pinecone serverless since we can't scroll all docs easily
            k=60
        )
        
        # Build the graph
        _agent_graph = build_rag_agent(retriever, primary_llm, fallback_llm)
        logger.info("LangGraph agent initialized and cached.")
        
    return _agent_graph
