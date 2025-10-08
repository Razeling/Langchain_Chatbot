"""
Pydantic models for chat-related data structures
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class VehicleInfo(BaseModel):
    """Vehicle information model."""

    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = Field(None, ge=1900, le=2030)
    mileage: Optional[int] = Field(None, ge=0)
    engine_type: Optional[str] = None
    transmission: Optional[str] = None


class FunctionCall(BaseModel):
    """Function call execution model."""

    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None
    status: Literal["pending", "completed", "error"] = "pending"
    execution_time: Optional[float] = None
    error_message: Optional[str] = None


class SourceCitation(BaseModel):
    """Source citation from RAG retrieval."""

    title: str
    content: str
    similarity: float = Field(ge=0.0, le=1.0)
    source: str
    page: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """Chat message model."""

    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    function_calls: Optional[List[FunctionCall]] = None
    sources: Optional[List[SourceCitation]] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """Chat request from frontend."""

    message: str = Field(..., min_length=1, max_length=2000)
    messages: List[ChatMessage] = Field(default_factory=list)
    vehicle_info: Optional[VehicleInfo] = None
    country: str = Field(default="LT", description="European country code (LT, DE, FR, etc.)")
    stream: bool = Field(default=False)
    include_sources: bool = Field(default=True)
    max_tokens: Optional[int] = Field(default=1000, ge=100, le=4000)


class ChatResponse(BaseModel):
    """Response model for chat API."""

    chat_id: str = Field(..., description="Unique chat identifier")
    response: str = Field(..., description="The chatbot's response message")
    sources: List[SourceCitation] = Field(default=[], description="All sources combined (legacy)")

    # Separate source categories to prevent duplication
    pure_knowledge_sources: List[SourceCitation] = Field(
        default=[], description="📚 Internal Knowledge Base - Pure knowledge base content"
    )
    web_learned_sources: List[SourceCitation] = Field(
        default=[], description="🧠 Previously Learned from Web - Previously learned web content"
    )
    web_sources: List[SourceCitation] = Field(
        default=[], description="🌐 Web Sources - Fresh web search results"
    )

    # Legacy fields for backward compatibility
    knowledge_sources: List[SourceCitation] = Field(
        default=[], description="Legacy: Pure knowledge base sources"
    )

    learned_documents: List[Dict[str, Any]] = Field(
        default=[], description="Newly learned documents"
    )
    function_calls: List[FunctionCall] = Field(
        default=[], description="Function calls made during response"
    )
    vehicle_info: Optional[VehicleInfo] = Field(
        default=None, description="Vehicle information used"
    )
    country: str = Field(default="LT", description="Country context")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class StreamChunk(BaseModel):
    """Streaming response chunk."""

    id: str
    content: str
    finished: bool = False
    function_calls: Optional[List[FunctionCall]] = None
    sources: Optional[List[SourceCitation]] = None


class ChatSession(BaseModel):
    """Chat session model."""

    id: str
    messages: List[ChatMessage] = Field(default_factory=list)
    vehicle_info: Optional[VehicleInfo] = None
    start_time: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
