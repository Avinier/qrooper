"""Minimal Qrooper schemas

Only the schemas actually used in the codebase.
Clean, simple, maintainable.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from datetime import datetime


# ===== Enums =====

class AnalysisStatus(str, Enum):
    """Status of analysis request"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryType(str, Enum):
    """Schema for query types"""
    GENERAL = "general"
    ARCHITECTURE = "architecture"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    SECURITY = "security"
    PERFORMANCE = "performance"


class QueryComplexity(str, Enum):
    """Schema for query complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


# ===== Reconnaissance Schemas =====

class CodebaseFingerprint(BaseModel):
    """Fingerprint of a codebase from lightning scan"""
    path: str
    name: str
    timestamp: str
    languages: Dict[str, int] = Field(default_factory=dict)
    frameworks: List[str] = Field(default_factory=list)
    build_tools: List[str] = Field(default_factory=list)
    total_files: int = 0
    size_estimate: str = "Unknown"
    top_level_structure: Dict[str, Any] = Field(default_factory=dict)
    entry_points: List[str] = Field(default_factory=list)
    dependencies: Dict[str, Any] = Field(default_factory=dict)  # New field for dependency detection
    scan_time: float = 0.0


class ExplorationPlan(BaseModel):
    """Minimal plan for codebase exploration"""
    steps: List[str] = Field(..., description="Actionable steps with specific paths and actions")


class ReconnaissanceResult(BaseModel):
    """Complete reconnaissance analysis result"""
    query: str
    fingerprint: CodebaseFingerprint
    architecture: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float
    phases_executed: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error information if LLM call failed")
    tool_errors: Optional[List[str]] = Field(default=None, description="List of tool execution errors if any")
    termination_reason: Optional[str] = Field(default=None, description="Reason for loop termination (done tool, max iterations, auto-terminate, or error)")


# ===== Request/Response Schemas =====

class QrooperAnalysisRequest(BaseModel):
    """Request schema for Qrooper analysis"""
    query: str = Field(..., description="Query about the codebase")
    max_files: Optional[int] = Field(None, description="Maximum number of files to analyze")


class QrooperAnalysisResponse(BaseModel):
    """Response schema for Qrooper analysis"""
    status: str = Field(default="success")
    query: str
    result: Optional[str] = None
    files_analyzed: List[str] = Field(default_factory=list)
    used_ast: bool = Field(default=False)
    execution_time: Optional[float] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    detail: Optional[str] = None
    status_code: int = Field(default=500)


# ===== Export =====

__all__ = [
    # Enums
    'AnalysisStatus',
    'QueryType',
    'QueryComplexity',

    # Reconnaissance schemas
    'CodebaseFingerprint',
    'ExplorationPlan',
    'ReconnaissanceResult',

    # Request/Response schemas
    'QrooperAnalysisRequest',
    'QrooperAnalysisResponse',
    'ErrorResponse',
]