import time
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

from finsight.api.schemas import ChatRequest, ChatResponse, SourceDocument, ConsentRequest, SessionInfo, SessionHistoryResponse, MessageInfo
from finsight.api.dependencies import get_rag_agent
from langgraph.types import Command
import sqlite3

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])

def _format_chat_response(result: dict, start_time: float, requires_consent: bool = False, interrupt_payload: str = None) -> ChatResponse:
    if requires_consent:
        return ChatResponse(
            answer=interrupt_payload or "For this question I would have to surf the web and collate info. Should I proceed?",
            sources=[],
            route="ask_human_consent",
            execution_time_ms=(time.time() - start_time) * 1000,
            requires_consent=True
        )

    generation = result.get("generation", "No answer generated.")
    route = result.get("route", "unknown")
    raw_docs = result.get("documents", [])
    sources = [SourceDocument(content=doc.page_content, metadata=doc.metadata) for doc in raw_docs]
    
    return ChatResponse(
        answer=generation,
        sources=sources,
        route=route,
        execution_time_ms=(time.time() - start_time) * 1000,
        requires_consent=False
    )

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, agent=Depends(get_rag_agent)):
    """Process a financial question using the FinSight RAG pipeline."""
    import uuid
    start_time = time.time()
    
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        from langchain_core.messages import HumanMessage
        result = agent.invoke(
            {
                "question": request.query, 
                "session_id": thread_id,
                "max_retries": request.max_retries,
                "chat_history": [HumanMessage(content=request.query)]
            },
            config=config
        )
        
        # Check if the graph was interrupted
        state = agent.get_state(config)
        if state.next:
            interrupts = [t.interrupts for t in state.tasks if t.interrupts]
            payload = interrupts[0][0].value if interrupts and interrupts[0] else None
            return _format_chat_response(result, start_time, requires_consent=True, interrupt_payload=payload)
            
        return _format_chat_response(result, start_time)
        
    except Exception as e:
        logger.exception("Error processing chat request")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/resume", response_model=ChatResponse)
async def chat_resume_endpoint(request: ConsentRequest, agent=Depends(get_rag_agent)):
    """Resume a paused LangGraph session after human consent."""
    start_time = time.time()
    config = {"configurable": {"thread_id": request.thread_id}}
    
    try:
        state = agent.get_state(config)
        if not state.next:
            raise HTTPException(status_code=400, detail="No active paused session found for this thread.")
            
        result = agent.invoke(Command(resume=request.proceed), config=config)
        
        new_state = agent.get_state(config)
        if new_state.next:
            interrupts = [t.interrupts for t in new_state.tasks if t.interrupts]
            payload = interrupts[0][0].value if interrupts and interrupts[0] else None
            return _format_chat_response(result, start_time, requires_consent=True, interrupt_payload=payload)
            
        return _format_chat_response(result, start_time)
        
    except Exception as e:
        logger.exception("Error resuming chat request")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_document(file: UploadFile = File(...), thread_id: str = Form(...)):
    """Upload a document, parse with LlamaParse, chunk, and ingest into Pinecone."""
    import tempfile
    import os
    from finsight.ingestion.document_processor import chunk_document
    from finsight.ingestion.vector_store import get_vector_store
    from llama_parse import LlamaParse
    from config.settings import settings
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are currently supported for upload.")
        
    if not settings.llama_parse_api_key:
        raise HTTPException(status_code=500, detail="LlamaParse API key is not configured.")
        
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
            
        parser = LlamaParse(api_key=settings.llama_parse_api_key, result_type="markdown")
        parsed_docs = parser.load_data(tmp_path)
        os.remove(tmp_path)
        
        if not parsed_docs:
            raise HTTPException(status_code=400, detail="Failed to extract any text from the document.")
            
        markdown_text = "\n".join([d.text for d in parsed_docs])
        metadata = {"source": file.filename, "type": "user_upload"}
        
        chunks = chunk_document(markdown_text, metadata=metadata)
        from finsight.ingestion.vector_store import ingest_documents
        ingest_documents(chunks, namespace=thread_id)
        
        return {"status": "success", "chunks_ingested": len(chunks), "filename": file.filename}
        
    except Exception as e:
        logger.exception("Error during document upload")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions(agent=Depends(get_rag_agent)):
    """List all available chat sessions from the local SQLite checkpointer."""
    try:
        conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        cursor = conn.cursor()
        # Query distinct threads, ordering by the latest rowid (newest first)
        cursor.execute("SELECT thread_id, MAX(rowid) as max_rowid FROM checkpoints GROUP BY thread_id ORDER BY max_rowid DESC")
        threads = cursor.fetchall()
        
        sessions = []
        for row in threads:
            thread_id = row[0]
            session_name = "New Session"
            message_count = 0
            
            # Extract first human message from the state
            config = {"configurable": {"thread_id": thread_id}}
            try:
                state = agent.get_state(config)
                if state and hasattr(state, 'values') and 'chat_history' in state.values:
                    messages = state.values['chat_history']
                    message_count = len(messages)
                    for msg in messages:
                        if msg.type == "human":
                            content = msg.content
                            session_name = content[:30] + "..." if len(content) > 30 else content
                            break
            except Exception:
                pass
                
            sessions.append(SessionInfo(
                thread_id=thread_id, 
                last_updated="Unknown", 
                message_count=message_count,
                session_name=session_name
            ))
            
        return sessions
    except Exception as e:
        logger.exception("Error fetching sessions")
        raise HTTPException(status_code=500, detail="Failed to fetch sessions")

@router.get("/sessions/{thread_id}", response_model=SessionHistoryResponse)
async def get_session_history(thread_id: str, agent=Depends(get_rag_agent)):
    """Fetch the chat history for a given session."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent.get_state(config)
        if not state or not hasattr(state, 'values') or 'chat_history' not in state.values:
            return SessionHistoryResponse(thread_id=thread_id, messages=[])
            
        messages = state.values['chat_history']
        
        formatted_messages = []
        for msg in messages:
            role = "user" if msg.type == "human" else "assistant"
            formatted_messages.append(MessageInfo(role=role, content=msg.content))
            
        return SessionHistoryResponse(thread_id=thread_id, messages=formatted_messages)
    except Exception as e:
        logger.exception("Error fetching session history")
        raise HTTPException(status_code=500, detail="Failed to fetch session history")
