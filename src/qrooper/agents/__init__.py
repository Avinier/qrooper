"""
LLM-powered agents for code analysis

This module provides the LLM-powered 3-pass analysis agents:
- Scout (Reconnaissance) - Maps codebase structure
- Connect (Pattern Recognition) - Finds relationships and flows
- DeepDive (Deep Analysis) - Provides detailed answers
"""

# LLM-powered 3-pass analysis agents
from .reconnaissance import ReconnaissanceAgent
from .pattern_recognition import PatternRecognitionAgent
from .deep_analysis import DeepAnalysisAgent

# Decider Architecture
from .decider import DeciderAgent

# LLM provider
from .llm_calls import QrooperLLM

__all__ = [
    # LLM-powered 3-pass agents
    'ReconnaissanceAgent',
    'PatternRecognitionAgent',
    'DeepAnalysisAgent',

    # Decider Architecture
    'DeciderAgent',

    # LLM provider
    'QrooperLLM'
]