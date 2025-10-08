"""
Pydantic models for vehicle diagnostics and maintenance
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class CarDiagnostic(BaseModel):
    """Car diagnostic result model."""

    problem: str
    severity: Literal["low", "medium", "high", "critical"]
    description: str
    possible_causes: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    estimated_cost: Optional["CostEstimate"] = None
    urgency: str
    safety_notes: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CostEstimate(BaseModel):
    """Cost estimation model."""

    min_cost: float = Field(ge=0)
    max_cost: float = Field(ge=0)
    currency: str = Field(default="USD")
    breakdown: Optional[Dict[str, float]] = Field(default_factory=dict)
    labor_hours: Optional[float] = Field(None, ge=0)
    labor_rate: Optional[float] = Field(None, ge=0)


class RepairEstimate(BaseModel):
    """Repair cost estimate model."""

    part_name: str
    labor_hours: float = Field(ge=0)
    part_cost: float = Field(ge=0)
    labor_rate: float = Field(ge=0)
    total_cost: float = Field(ge=0)
    description: str
    difficulty: Literal["easy", "moderate", "difficult", "expert"]
    estimated_time: Optional[str] = None  # e.g., "2-3 hours"
    tools_required: Optional[List[str]] = Field(default_factory=list)
    warranty: Optional[str] = None


class MaintenanceItem(BaseModel):
    """Individual maintenance item."""

    item: str
    interval_miles: int = Field(ge=0)
    interval_months: int = Field(ge=0)
    last_service: Optional[datetime] = None
    next_due: datetime
    priority: Literal["low", "medium", "high"]
    description: str
    estimated_cost: Optional[CostEstimate] = None
    overdue: bool = Field(default=False)


class MaintenanceSchedule(BaseModel):
    """Complete maintenance schedule."""

    vehicle_info: Optional[Dict[str, Any]] = Field(default_factory=dict)
    current_mileage: Optional[int] = Field(None, ge=0)
    items: List[MaintenanceItem] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)
    next_service_due: Optional[datetime] = None
    total_estimated_cost: Optional[float] = Field(None, ge=0)


class DiagnosticCode(BaseModel):
    """OBD-II diagnostic trouble code."""

    code: str  # e.g., "P0300"
    description: str
    severity: Literal["informational", "moderate", "severe"]
    system: str  # e.g., "Engine", "Transmission", "Emissions"
    possible_causes: List[str] = Field(default_factory=list)
    recommended_tests: List[str] = Field(default_factory=list)


class VehicleHistory(BaseModel):
    """Vehicle service history."""

    vehicle_info: Dict[str, Any]
    service_records: List[Dict[str, Any]] = Field(default_factory=list)
    known_issues: List[str] = Field(default_factory=list)
    recalls: List[Dict[str, Any]] = Field(default_factory=list)
    warranty_info: Optional[Dict[str, Any]] = None


class SymptomAnalysis(BaseModel):
    """Analysis of reported symptoms."""

    symptoms: List[str]
    primary_system: str  # e.g., "Engine", "Brakes", "Electrical"
    confidence: float = Field(ge=0.0, le=1.0)
    related_symptoms: List[str] = Field(default_factory=list)
    questions_to_ask: List[str] = Field(default_factory=list)
    immediate_concerns: List[str] = Field(default_factory=list)


# Update CarDiagnostic to include forward reference
CarDiagnostic.model_rebuild()
