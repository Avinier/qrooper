"""qrooper package initializer.

This module ensures environment variables are loaded once at package import time
for all qrooper modules and scripts, without each script needing custom code.
"""

from typing import Optional

__version__ = "0.3.0"  # Updated for LLM-Powered 3-Pass Architecture


_QROOPER_ENV_LOADED: bool = False


def _load_env_once() -> None:
    """Load environment variables once at package import time."""
    global _QROOPER_ENV_LOADED
    if _QROOPER_ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv  # type: ignore
        from pathlib import Path as _Path

        # Candidate locations: package repo root, current working directory
        # repo root is 4 parents up from this file: qrooper/__init__.py -> qrooper -> src -> packages -> repo root
        repo_root = _Path(__file__).resolve().parents[4]
        candidates = [
            repo_root / ".env.development",
            repo_root / ".env",
            _Path.cwd() / ".env.development",
            _Path.cwd() / ".env",
        ]
        for _p in candidates:
            if _p.exists():
                load_dotenv(_p)
                _QROOPER_ENV_LOADED = True
                break
        if not _QROOPER_ENV_LOADED:
            load_dotenv()
            _QROOPER_ENV_LOADED = True
    except Exception:
        # Optional dependency; skip silently if unavailable
        _QROOPER_ENV_LOADED = True


_load_env_once()

# Main Engine - Primary interface for codebase analysis
from .qrooper_engine import QrooperEngine, QrooperAnalysisResult

# Convenience functions
async def analyze_codebase(codebase_path: str, query: str, **kwargs) -> QrooperAnalysisResult:
    """Convenience function for codebase analysis"""
    engine = QrooperEngine(codebase_path, **kwargs)
    return await engine.analyze(query)

async def debug_issue(codebase_path: str, issue: str, **kwargs) -> QrooperAnalysisResult:
    """Convenience function for debugging issues"""
    engine = QrooperEngine(codebase_path, **kwargs)
    return await engine.debug(issue)

# Agent exports (now default)
from .agents.reconnaissance import ReconnaissanceAgent, ReconnaissanceResult
from .agents.pattern_recognition import PatternRecognitionAgent, PatternRecognitionResult
from .agents.deep_analysis import DeepAnalysisAgent, AnalysisResult as DeepAnalysisResult

# LLM provider
from .agents.llm_calls import QrooperLLM

# Decider
from .agents.decider import DeciderAgent

# Legacy agents (for backward compatibility)
# Note: Legacy agents have been removed - use the V2 versions above instead

# Tools
from .tools import FilesystemUtils, ASTParsing, FileResult, GrepResult, CommandResult

# Schemas
from .schemas import (
    QrooperAnalysisRequest,
    QrooperAnalysisResponse,
    QueryType,
    QueryComplexity
)


# Main exports (plug-and-play)
__all__ = [
    # Main Engine
    'QrooperEngine',
    'QrooperAnalysisResult',
    'analyze_codebase',
    'debug_issue',

    # Agents (for advanced usage)
    'ReconnaissanceAgent',
    'PatternRecognitionAgent',
    'DeepAnalysisAgent',
    'ReconnaissanceResult',
    'PatternRecognitionResult',
    'DeepAnalysisResult',

    # Tools
    'FilesystemUtils',
    'ASTParsing',
    'FileResult',
    'GrepResult',
    'CommandResult',
    'QueryClassifier',
    'QueryType',
    'QueryComplexity',
    'ContextBuilder',

    # LLM provider
    'QrooperLLM',

    # Decider
    'DeciderAgent',

    # Schemas
    'QrooperAnalysisRequest',
    'QrooperAnalysisResponse',
    'QueryType',
    'QueryComplexity',
]