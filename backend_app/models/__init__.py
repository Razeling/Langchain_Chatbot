"""
Pydantic models package
"""

from .chat_models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatSession,
    ErrorResponse,
    FunctionCall,
    SourceCitation,
    StreamChunk,
    VehicleInfo,
)
from .vehicle_models import (
    CarDiagnostic,
    CostEstimate,
    DiagnosticCode,
    MaintenanceItem,
    MaintenanceSchedule,
    RepairEstimate,
    SymptomAnalysis,
    VehicleHistory,
)

__all__ = [
    # Chat models
    "VehicleInfo",
    "FunctionCall",
    "SourceCitation",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "StreamChunk",
    "ChatSession",
    "ErrorResponse",
    # Vehicle models
    "CarDiagnostic",
    "CostEstimate",
    "RepairEstimate",
    "MaintenanceItem",
    "MaintenanceSchedule",
    "DiagnosticCode",
    "VehicleHistory",
    "SymptomAnalysis",
]
