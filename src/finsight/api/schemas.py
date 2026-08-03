from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """Request payload for the chat endpoint."""
    query: str = Field(..., description="The user's financial question", example="What was Apple's revenue in 2024?")
    thread_id: Optional[str] = Field(None, description="Optional session ID for conversation memory", example="user_session_123")
    max_retries: int = Field(1, description="Maximum number of times to retry retrieval if answer is missing", example=1)

class SourceDocument(BaseModel):
    """Represents a source document chunk returned in the response."""
    content: str = Field(..., description="The text content of the document chunk")
    metadata: Dict[str, Any] = Field(..., description="Metadata such as ticker, year, filing type, and source")

class ChatResponse(BaseModel):
    """Response payload for the chat endpoint."""
    answer: str = Field(..., description="The generated answer from the LangGraph agent")
    sources: List[SourceDocument] = Field(default_factory=list, description="List of source documents used to generate the answer")
    route: Optional[str] = Field(None, description="The reasoning route taken (e.g. 'simple', 'complex', 'cached')")
    execution_time_ms: Optional[float] = Field(None, description="Execution time in milliseconds")
    requires_consent: bool = Field(default=False, description="True if the agent paused to ask for human consent")

class ConsentRequest(BaseModel):
    """Request payload for resuming a paused session."""
    thread_id: str = Field(..., description="The session ID to resume")
    proceed: bool = Field(..., description="True to grant consent, False to cancel")

class SessionInfo(BaseModel):
    thread_id: str
    last_updated: str
    message_count: int
    session_name: Optional[str] = None

class MessageInfo(BaseModel):
    role: str
    content: str

class SessionHistoryResponse(BaseModel):
    thread_id: str
    messages: List[MessageInfo]
