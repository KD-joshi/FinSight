import logging
from typing import Optional
from fastapi import Request

from finsight.utils.llm_provider import get_llm_with_fallback

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
        # primary_llm already has .with_fallbacks() (Groq → Gemini → Cohere)
        primary_llm = get_llm_with_fallback()
        small_llm = get_llm_with_fallback(max_tokens=256)
        
        # Initialize retriever
        retriever = HybridRetriever(
            documents=[], # BM25 sparse disabled for Pinecone serverless since we can't scroll all docs easily
            k=60
        )
        
        # Build the graph
        _agent_graph = build_rag_agent(retriever, primary_llm, small_llm)
        logger.info("LangGraph agent initialized and cached.")
        
    return _agent_graph
