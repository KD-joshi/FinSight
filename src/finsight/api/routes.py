import time
import logging
from fastapi import APIRouter, Depends, HTTPException

from finsight.api.schemas import ChatRequest, ChatResponse, SourceDocument
from finsight.api.dependencies import get_rag_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, agent=Depends(get_rag_agent)):
    """
    Process a financial question using the FinSight RAG pipeline.
    """
    import uuid
    start_time = time.time()
    
    thread_id = request.thread_id or str(uuid.uuid4())
    
    try:
        # Invoke the LangGraph agent
        result = agent.invoke(
            {
                "question": request.query,
                "max_retries": request.max_retries
            },
            config={"configurable": {"thread_id": thread_id}}
        )
        
        # Extract components from the agent state
        generation = result.get("generation", "No answer generated.")
        route = result.get("route", "unknown")
        
        # Format the source documents
        raw_docs = result.get("documents", [])
        sources = []
        for doc in raw_docs:
            sources.append(
                SourceDocument(
                    content=doc.page_content,
                    metadata=doc.metadata
                )
            )
            
        execution_time = (time.time() - start_time) * 1000
        
        return ChatResponse(
            answer=generation,
            sources=sources,
            route=route,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        logger.exception("Error processing chat request")
        raise HTTPException(status_code=500, detail=str(e))
