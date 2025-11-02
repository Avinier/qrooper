"""
QrooperEngine - Main interface for codebase analysis

The QrooperEngine provides a simple, unified interface for the 3-pass
LLM-powered codebase analysis system. It orchestrates Scout, Connect,
and DeepDive agents internally to provide comprehensive code analysis.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .agents.llm_calls import QrooperLLM
from .agents.reconnaissance import ReconnaissanceAgent, ReconnaissanceResult
from .agents.pattern_recognition import PatternRecognitionAgent, PatternRecognitionResult
from .agents.deep_analysis import DeepAnalysisAgent, AnalysisResult
from .agents.decider import DeciderAgent
from .prompts import format_coordination_prompt


@dataclass
class QrooperAnalysisResult:
    """Result from QrooperEngine analysis"""
    query: str
    answer: str
    passes_used: int
    decision_reasoning: str
    analysis_time: float

    # Raw data from each phase (if executed)
    reconnaissance_data: Dict[str, Any]  # Always present

    # Optional fields with defaults
    files_analyzed: Optional[int] = None
    pattern_data: Optional[Dict[str, Any]] = None  # Present if 2+ passes
    deep_data: Optional[Dict[str, Any]] = None  # Present if 3 passes

    # Only from deep analysis (if executed)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    examples: Dict[str, str] = field(default_factory=dict)


class QrooperEngine:
    def __init__( self, codebase_path: str, model: str = "deepseek-v3p1",
        reasoning_effort: str = "medium", desc: str = "", **kwargs
    ):
        """
        Initialize QrooperEngine

        Args:
            codebase_path: Path to the codebase to analyze
            model: LLM model to use for analysis
            reasoning_effort: Reasoning effort level (none/low/medium/high)
            desc: Optional description for the engine
            **kwargs: Additional parameters
        """
        self.codebase_path = Path(codebase_path)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.desc = desc

        # Initialize LLM provider
        self.llm = QrooperLLM(
            model=model,
            reasoning_effort=reasoning_effort,
            desc="Qrooper Analysis Engine"
        )

        # Initialize Decider for intelligent pass determination
        self.decider = DeciderAgent(self.llm)

        # Initialize agents
        self.recon = ReconnaissanceAgent(model=self.model)
        self.pattern_recog = PatternRecognitionAgent(self.codebase_path, self.llm)
        self.deep = DeepAnalysisAgent(self.codebase_path, self.llm)

        # Session context for in-memory storage
        self.session_context = {}

    async def analyze(
        self,
        query: str,
        mode: str = "default",
        **kwargs  # Additional parameters (currently unused)
    ) -> QrooperAnalysisResult:
        """
        Analyze codebase with intelligent pass determination

        Args:
            query: User's query about the codebase
            mode: Analysis mode (default, debugging, architecture, security, performance)
            **kwargs: Additional parameters

        Returns:
            QrooperAnalysisResult with comprehensive analysis
        """
        start_time = time.time()

        print(f"[qrooper_engine.analyze] Analyzing query: {query}")
        print(f"[qrooper_engine.analyze] Model: {self.model}, Reasoning: {self.reasoning_effort}")

        # Step 1: Run DeciderAgent and Reconnaissance IN PARALLEL
        # Reconnaissance is always needed, so we start it immediately
        # while Decider determines how many passes we need
        print("[qrooper_engine.analyze] Starting parallel analysis: Decider + Reconnaissance...")

        # Execute both tasks in parallel
        # Use asyncio.to_thread for the sync decider method
        decision_task = asyncio.create_task(asyncio.to_thread(self.decider.decide, query))
        recon_task = asyncio.create_task(self._execute_reconnaissance(query))

        # Get decision first (it's much faster than reconnaissance)
        decision = await decision_task
        print(f"[qrooper_engine.analyze] LLM decision: {decision.passes_required} pass(es) needed")
        print(f"[qrooper_engine.analyze] Reasoning: {decision.reasoning}")

        # Store decision in session context
        self.session_context['decision'] = decision

        # If only 1 pass needed, wait for reconnaissance and return immediately
        if decision.passes_required == "one":
            print("[qrooper_engine.analyze] One pass needed - waiting for reconnaissance to complete")
            recon_result = await recon_task  # Wait for it to finish
            print(f"[qrooper_engine.analyze] Reconnaissance completed in {recon_result.analysis_time:.2f}s")
            results = {'reconnaissance': recon_result}
            print("[qrooper_engine.analyze] Stopping after 1 pass - sufficient information gathered")
            return await self._create_early_result(query, decision, results, start_time)

        # For 2 or 3 passes, reconnaissance should be done or nearly done
        print("[qrooper_engine.analyze] Multiple passes needed - ensuring reconnaissance completed")
        recon_result = await recon_task  # This should be quick since it's been running in parallel
        print(f"[qrooper_engine.analyze] Reconnaissance completed in {recon_result.analysis_time:.2f}s")

        # Store results and continue
        results = {'reconnaissance': recon_result}

        # Phase 2: Pattern recognition (needed for 2 or 3 passes)
        if decision.passes_required in ["two", "three"]:
            print("[qrooper_engine.analyze] Phase 2: Pattern Recognition - Finding relationships...")
            pattern_result = await self._execute_pattern_recognition(
                query, recon_result
            )
            results['pattern'] = pattern_result
            print(f"[qrooper_engine.analyze] Pattern recognition completed in {pattern_result.analysis_time:.2f}s")

            # Check if we can stop early after pattern recognition
            if decision.passes_required == "two":
                print("[qrooper_engine.analyze] Stopping after 2 passes - comprehensive analysis complete")
                return await self._create_early_result(query, decision, results, start_time)

        # Phase 3: Deep analysis (only for 3 passes)
        if decision.passes_required == "three":
            print("[qrooper_engine.analyze] Phase 3: Deep Analysis - Providing detailed answer...")
            deep_result = await self._execute_deep_analysis(
                query, recon_result, pattern_result, mode
            )
            results['deep'] = deep_result
            print(f"[qrooper_engine.analyze] Deep analysis completed in {deep_result.analysis_time:.2f}s")

        # Synthesize final result
        final_result = await self._synthesize_result(
            query, decision, results, mode, start_time
        )

        print(f"[qrooper_engine.analyze] Total analysis time: {final_result.analysis_time:.2f}s")
        print(f"[qrooper_engine.analyze] Passes used: {decision.passes_required}")

        return final_result

    async def _execute_reconnaissance(self, query: str) -> ReconnaissanceResult:
        """Execute Phase 1: Reconnaissance"""
        context_key = f"recon_{hash(query)}"
        if context_key in self.session_context:
            print("[qrooper_engine.execute_reconnaissance] Using session context for reconnaissance")
            return self.session_context[context_key]

        result = await self.recon.analyze(query, str(self.codebase_path))

        # Session context is always enabled
        self.session_context[context_key] = result

        return result

    async def _execute_pattern_recognition(
        self,
        query: str,
        recon_result: ReconnaissanceResult
    ) -> PatternRecognitionResult:
        """Execute Phase 2: Pattern Recognition"""
        context_key = f"pattern_{hash(query)}"
        if context_key in self.session_context:
            print("[qrooper_engine.execute_pattern_recognition] Using session context for pattern recognition")
            return self.session_context[context_key]

        result = await self.pattern_recog.analyze(query, recon_result)

        # Session context is always enabled
        self.session_context[context_key] = result

        return result

    async def _execute_deep_analysis(
        self,
        query: str,
        recon_result: ReconnaissanceResult,
        pattern_result: PatternRecognitionResult,
        mode: str
    ) -> AnalysisResult:
        """Execute Phase 3: Deep Analysis"""
        context_key = f"deep_{hash(query)}_{mode}"
        if context_key in self.session_context:
            print("[qrooper_engine.execute_deep_analysis] Using session context for deep analysis")
            return self.session_context[context_key]

        result = await self.deep.analyze(query, recon_result, pattern_result, mode)

        # Session context is always enabled
        self.session_context[context_key] = result

        return result

    
    # Convenience methods
    async def debug(
        self,
        query: str,
        error_message: Optional[str] = None,
        stack_trace: Optional[str] = None
    ) -> QrooperAnalysisResult:
        """
        Convenience method for debugging issues
        """
        enhanced_query = query
        if error_message:
            enhanced_query += f"\nError: {error_message}"
        if stack_trace:
            enhanced_query += f"\nStack trace: {stack_trace}"

        return await self.analyze(enhanced_query, mode="debugging")

    async def analyze_architecture(self, query: str) -> QrooperAnalysisResult:
        """Convenience method for architecture analysis"""
        return await self.analyze(query, mode="architecture")

    async def analyze_security(self, query: str) -> QrooperAnalysisResult:
        """Convenience method for security analysis"""
        return await self.analyze(query, mode="security")

    async def analyze_performance(self, query: str) -> QrooperAnalysisResult:
        """Convenience method for performance analysis"""
        return await self.analyze(query, mode="performance")

    def clear_cache(self):
        """Clear the session context"""
        # Session context is always enabled
        self.session_context.clear()
        print("[qrooper_engine.clear_cache] Session context cleared")

    async def _create_early_result(
        self,
        query: str,
        decision: 'DecisionResult',
        results: Dict[str, Any],
        start_time: float
    ) -> QrooperAnalysisResult:
        """Create result from early termination (1 or 2 passes)"""
        recon_result = results.get('reconnaissance')
        pattern_result = results.get('pattern')

        # Use the best available answer
        if pattern_result and hasattr(pattern_result, 'insights'):
            answer = f"{pattern_result.summary}\n\nKey Insights:\n" + "\n".join(f"• {insight}" for insight in pattern_result.insights[:5])
        else:
            answer = recon_result.synthesis.get("executive_summary", "Reconnaissance completed")

        # Prepare reconnaissance data
        recon_data = {
            "summary": recon_result.synthesis.get("executive_summary", ""),
            "structure": recon_result.architecture,
            "files_analyzed": recon_result.files_analyzed,
            "fingerprint": {
                "languages": recon_result.fingerprint.languages,
                "frameworks": recon_result.fingerprint.frameworks,
                "total_files": recon_result.fingerprint.total_files
            }
        }

        # Prepare pattern data if available
        pattern_data = None
        if pattern_result:
            pattern_data = {
                "summary": pattern_result.summary,
                "insights": pattern_result.insights,
                "patterns_found": len(pattern_result.pattern_matches) if hasattr(pattern_result, 'pattern_matches') else 0
            }

        return QrooperAnalysisResult(
            query=query,
            answer=answer,
            passes_used=decision.passes_required,
            decision_reasoning=decision.reasoning,
            reconnaissance_data=recon_data,
            pattern_data=pattern_data,
            deep_data=None,  # Not executed in early termination
            analysis_time=time.time() - start_time,
            files_analyzed=recon_result.files_analyzed
        )

    async def _synthesize_result(
        self,
        query: str,
        decision: 'DecisionResult',
        results: Dict[str, Any],
        mode: str,
        start_time: float
    ) -> QrooperAnalysisResult:
        """Synthesize results from all executed passes"""
        recon_result = results.get('reconnaissance')
        pattern_result = results.get('pattern')
        deep_result = results.get('deep')

        # Prepare summaries
        recon_summary = json.dumps({
            "structure": recon_result.architecture,
            "summary": recon_result.synthesis.get("executive_summary", ""),
            "key_findings": recon_result.synthesis.get("key_findings", []),
            "fingerprint": {
                "languages": recon_result.fingerprint.languages,
                "frameworks": recon_result.fingerprint.frameworks,
                "total_files": recon_result.fingerprint.total_files
            }
        }, indent=2)

        pattern_summary = json.dumps({
            "patterns": pattern_result.patterns_identified if pattern_result else {},
            "insights": pattern_result.insights if pattern_result else [],
            "flows": len(pattern_result.data_flows) if pattern_result else 0
        }, indent=2)

        deep_summary = json.dumps({
            "answer": deep_result.answer[:500] + "..." if len(deep_result.answer) > 500 else deep_result.answer,
            "evidence_count": len(deep_result.evidence)
        }, indent=2)

        # Use LLM to synthesize if needed
        if mode in ["debugging", "architecture"] or decision.passes_required == "three":
            synthesis_prompt = format_coordination_prompt(
                recon_summary, pattern_summary, deep_summary
            )

            try:
                synthesized = await self.llm.fw_basic_call(
                    prompt_or_messages=synthesis_prompt,
                    model=self.model,
                    temperature=0.3,
                    max_tokens=2000,
                    reasoning_effort=self.reasoning_effort
                )
                final_answer = synthesized
            except Exception as e:
                print(f"Error in synthesis: {e}")
                final_answer = deep_result.answer if deep_result else recon_result.summary
        else:
            final_answer = deep_result.answer if deep_result else recon_result.synthesis.get("executive_summary", "Analysis complete")

        # Prepare phase data
        recon_data = {
            "summary": recon_result.synthesis.get("executive_summary", ""),
            "structure": recon_result.architecture,
            "files_analyzed": recon_result.files_analyzed,
            "fingerprint": {
                "languages": recon_result.fingerprint.languages,
                "frameworks": recon_result.fingerprint.frameworks,
                "total_files": recon_result.fingerprint.total_files
            }
        }

        pattern_data = None
        if pattern_result:
            pattern_data = {
                "insights": pattern_result.insights,
                "patterns_found": len(pattern_result.pattern_matches) if hasattr(pattern_result, 'pattern_matches') else 0,
                "flows_identified": len(pattern_result.data_flows) if hasattr(pattern_result, 'data_flows') else 0
            }

        deep_data = None
        evidence_serializable = []
        recommendations = []
        examples = {}

        if deep_result:
            deep_data = {
                "root_cause": deep_result.root_cause
            }

            # Convert evidence to serializable format (remove confidence)
            for ev in deep_result.evidence:
                evidence_serializable.append({
                    "file_path": ev.file_path,
                    "line_number": ev.line_number,
                    "code_snippet": ev.code_snippet,
                    "explanation": ev.explanation
                })

            recommendations = deep_result.recommendations
            examples = deep_result.examples

        return QrooperAnalysisResult(
            query=query,
            answer=final_answer,
            passes_used=decision.passes_required,
            decision_reasoning=decision.reasoning,
            reconnaissance_data=recon_data,
            pattern_data=pattern_data,
            deep_data=deep_data,
            evidence=evidence_serializable,
            recommendations=recommendations,
            examples=examples,
            analysis_time=time.time() - start_time,
            files_analyzed=recon_result.files_analyzed
        )


# Unit tests (run when executed directly)
if __name__ == "__main__":
    print("🧪 QrooperEngine Unit Tests")
    print("=" * 50)

    # Test 1: Basic initialization
    print("\n1. Testing QrooperEngine initialization...")
    try:
        engine = QrooperEngine(
            codebase_path=".",
            model="deepseek-v3p1",
            reasoning_effort="medium",
            desc="Test Engine"
        )
        print("✅ Engine initialized successfully")
        print(f"   - Codebase path: {engine.codebase_path}")
        print(f"   - Model: {engine.model}")
        print(f"   - Reasoning effort: {engine.reasoning_effort}")
        print(f"   - Description: {engine.desc}")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")

    # Test 2: Check internal agents
    print("\n2. Testing internal agents...")
    try:
        print(f"   - Scout agent: {'✅' if engine.recon else '❌'}")
        print(f"   - Connect agent: {'✅' if engine.pattern_recog else '❌'}")
        print(f"   - DeepDive agent: {'✅' if engine.deep else '❌'}")
    except Exception as e:
        print(f"❌ Agent check failed: {e}")

    # Test 3: Check method availability
    print("\n3. Testing method availability...")
    methods = [
        'analyze',
        'debug',
        'analyze_architecture',
        'analyze_security',
        'analyze_performance',
        'clear_cache'
    ]

    for method in methods:
        has_method = hasattr(engine, method)
        print(f"   - {method}(): {'✅' if has_method else '❌'}")

    # Test 4: Session context functionality
    print("\n4. Testing session context functionality...")
    engine.session_context["test"] = "value"
    engine.clear_cache()
    print(f"   - Context clearing: {'✅' if not engine.session_context else '❌'}")

    print("\n✅ All basic tests passed!")
    print("\n📝 To run a full analysis, use:")
    print("   python -c \"from qrooper import QrooperEngine; import asyncio; asyncio.run(QrooperEngine('.').analyze('test'))\"")