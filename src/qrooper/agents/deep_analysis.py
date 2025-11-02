"""
Deep Analysis Agent - LLM-powered detailed analyzer
Third pass of the 3-pass analysis strategy
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from ..prompts import DEEP_ANALYSIS_AGENT_PROMPT, get_deep_analysis_prompt_with_mode
from ..tools import FilesystemUtils
from ..agents.llm_calls import QrooperLLM
from ..agents.reconnaissance import ReconnaissanceResult
from ..agents.pattern_recognition import PatternRecognitionResult


@dataclass
class Evidence:
    """Evidence supporting analysis conclusions"""
    file_path: str
    line_number: int
    code_snippet: str
    explanation: str
    confidence: float


@dataclass
class AnalysisResult:
    """Result of deep analysis"""
    answer: str
    confidence: float
    evidence: List[Evidence] = field(default_factory=list)
    root_cause: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    related_files: List[str] = field(default_factory=list)
    examples: Dict[str, str] = field(default_factory=dict)
    edge_cases: List[str] = field(default_factory=list)
    analysis_time: float = 0.0


class DeepAnalysisAgent:
    """
    DeepDive - The LLM-powered deep analysis agent.
    Provides detailed, specific answers building on previous analysis.
    """

    def __init__(self, codebase_path: Path, llm_provider: QrooperLLM):
        """
        Initialize deep analysis agent

        Args:
            codebase_path: Path to the codebase
            llm_provider: LLM provider for analysis
        """
        self.codebase_path = Path(codebase_path)
        self.llm_provider = llm_provider
        self.tools = FilesystemUtils(self.codebase_path)
        self.system_prompt = DEEP_ANALYSIS_AGENT_PROMPT

    async def analyze(self,
                      query: str,
                      recon_result: ReconnaissanceResult,
                      pattern_result: PatternRecognitionResult,
                      mode: str = "default") -> AnalysisResult:
        """
        Perform deep analysis building on previous passes

        Args:
            query: User's query
            recon_result: Results from reconnaissance agent
            pattern_result: Results from pattern recognition agent
            mode: Analysis mode (debugging, architecture, security, performance)

        Returns:
            Comprehensive AnalysisResult with specific details
        """
        import time
        start_time = time.time()

        # Get specialized prompt based on mode
        system_prompt = get_deep_analysis_prompt_with_mode(mode)

        # Phase 1: Gather all context
        full_context = await self._gather_full_context(
            query, recon_result, pattern_result
        )

        # Phase 2: Deep dive analysis
        deep_analysis = await self._perform_deep_analysis(
            query, full_context, system_prompt
        )

        # Phase 3: Extract evidence
        evidence = await self._extract_evidence(
            query, deep_analysis, full_context
        )

        # Phase 4: Generate recommendations
        recommendations = await self._generate_recommendations(
            query, deep_analysis, evidence
        )

        # Phase 5: Create examples
        examples = await self._create_examples(
            query, deep_analysis, full_context
        )

        result = AnalysisResult(
            answer=deep_analysis.get('answer', ''),
            confidence=deep_analysis.get('confidence', 0.7),
            evidence=evidence,
            root_cause=deep_analysis.get('root_cause'),
            recommendations=recommendations,
            related_files=deep_analysis.get('related_files', []),
            examples=examples,
            edge_cases=deep_analysis.get('edge_cases', []),
            analysis_time=time.time() - start_time
        )

        return result

    async def _gather_full_context(self,
                                 query: str,
                                 recon_result: ReconnaissanceResult,
                                 pattern_result: PatternRecognitionResult) -> Dict[str, Any]:
        """Gather comprehensive context from previous agents"""

        context = {
            "query": query,
            "reconnaissance": {
                "structure": recon_result.structure,
                "summary": recon_result.summary,
                "key_findings": recon_result.key_findings
            },
            "patterns": {
                "identified": pattern_result.patterns_identified,
                "insights": pattern_result.insights,
                "data_flows": [
                    {
                        "source": flow.source,
                        "destination": flow.destination,
                        "data_type": flow.data_type,
                        "path": flow.medium
                    } for flow in pattern_result.data_flows
                ],
                "control_flows": [
                    {
                        "trigger": flow.trigger,
                        "sequence": flow.sequence
                    } for flow in pattern_result.control_flows
                ]
            }
        }

        # Identify critical files to examine
        critical_files = set()

        # From pattern agent context
        deep_context = pattern_result.context_for_deep_agent
        critical_files.update(deep_context.get('critical_path', []))
        critical_files.update(deep_context.get('files_to_examine', []))

        # From pattern locations
        pattern_locs = deep_context.get('pattern_locations', {})
        for loc in pattern_locs.values():
            if isinstance(loc, list):
                critical_files.update(loc)
            else:
                critical_files.add(loc)

        # Get content for critical files
        context["critical_files"] = {}
        for file_path in list(critical_files)[:10]:  # Limit to prevent context overflow
            content = await self.tools.read_file(file_path)
            if not content.error:
                context["critical_files"][file_path] = {
                    "content": content.content,
                    "lines": content.lines
                }

        return context

    async def _perform_deep_analysis(self,
                                    query: str,
                                    context: Dict[str, Any],
                                    system_prompt: str) -> Dict[str, Any]:
        """Perform the actual deep analysis with LLM"""

        # Prepare a comprehensive prompt for the LLM
        analysis_prompt = f"""
Perform a deep analysis to answer this question: "{query}"

CONTEXT FROM PREVIOUS AGENTS:

1. RECONNAISSANCE (Scout):
{json.dumps(context['reconnaissance'], indent=2)}

2. PATTERN RECOGNITION (Connect):
{json.dumps(context['patterns'], indent=2)}

3. CRITICAL FILES:
{json.dumps({k: {"lines": v["lines"]} for k, v in context["critical_files"].items()}, indent=2)}

YOUR TASK:
Provide a detailed, specific answer that addresses the user's query. Include:
1. The main answer/explanation
2. Root cause identification (if applicable)
3. Key files and line numbers involved
4. Edge cases or potential issues
5. Related areas that might be affected

RESPONSE FORMAT:
{{
  "answer": "Detailed answer to the user's query...",
  "confidence": 0.95,
  "root_cause": "The underlying cause is...",
  "related_files": ["file1.py", "file2.py"],
  "edge_cases": ["What happens when...", "Consider this scenario..."],
  "key_insights": ["Important insight 1", "Important insight 2"]
}}

Be specific and provide concrete details with file paths and line numbers where possible.
"""

        try:
            response = await self.llm_provider.fw_basic_call(
                prompt_or_messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                model="deepseek-v3p1",
                temperature=0.2,  # Lower temperature for more precise answers
                max_tokens=3000,
                reasoning_effort="high"
            )

            # Try to parse JSON response
            try:
                analysis = json.loads(response)
            except json.JSONDecodeError:
                # Fallback: treat as plain text response
                analysis = {
                    "answer": response,
                    "confidence": 0.7,
                    "root_cause": None,
                    "related_files": [],
                    "edge_cases": [],
                    "key_insights": []
                }

            return analysis

        except Exception as e:
            print(f"Error in deep analysis: {e}")
            return {
                "answer": f"Unable to complete deep analysis due to error: {str(e)}",
                "confidence": 0.1,
                "root_cause": "Analysis failed",
                "related_files": [],
                "edge_cases": [],
                "key_insights": []
            }

    async def _extract_evidence(self,
                               query: str,
                               analysis: Dict[str, Any],
                               context: Dict[str, Any]) -> List[Evidence]:
        """Extract concrete evidence from code to support analysis"""

        evidence_list = []

        # Analyze mentioned files
        for file_path in analysis.get('related_files', [])[:5]:
            if file_path in context.get('critical_files', {}):
                file_info = context['critical_files'][file_path]
                content = file_info['content']

                # Look for relevant code patterns based on query
                relevant_lines = await self._find_relevant_lines(
                    query, content, file_path
                )

                for line_info in relevant_lines[:3]:  # Limit evidence per file
                    evidence_list.append(Evidence(
                        file_path=file_path,
                        line_number=line_info['line'],
                        code_snippet=line_info['code'],
                        explanation=line_info['explanation'],
                        confidence=0.9
                    ))

        return evidence_list

    async def _find_relevant_lines(self,
                                 query: str,
                                 content: str,
                                 file_path: str) -> List[Dict[str, Any]]:
        """Find lines in code relevant to the query"""
        relevant = []
        lines = content.split('\n')

        # Extract keywords from query
        query_words = [w.lower() for w in query.split() if len(w) > 3]

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Check for query keywords
            if any(word in line_lower for word in query_words):
                # Get context
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                context_lines = lines[start:end]

                relevant.append({
                    'line': i + 1,
                    'code': '\n'.join(context_lines),
                    'explanation': f"Line contains relevant keyword from query"
                })

            # Also check for important patterns
            if any(pattern in line for pattern in [
                'class ', 'def ', 'raise ', 'return ', 'if ',
                'import ', 'from ', 'TODO:', 'FIXME:', 'BUG:'
            ]):
                relevant.append({
                    'line': i + 1,
                    'code': line.strip(),
                    'explanation': f"Important code structure at this line"
                })

        return relevant[:5]  # Limit results

    async def _generate_recommendations(self,
                                       query: str,
                                       analysis: Dict[str, Any],
                                       evidence: List[Evidence]) -> List[str]:
        """Generate specific, actionable recommendations"""

        if not analysis.get('recommendations'):
            # Generate recommendations based on analysis
            rec_prompt = f"""
Based on this analysis, provide 3-5 specific, actionable recommendations:

Query: "{query}"
Analysis: {analysis.get('answer', '')[:1000]}...
Root Cause: {analysis.get('root_cause', 'Not identified')}

Return as a JSON array of strings:
[
  "Recommendation 1: Do X to fix/improve Y",
  "Recommendation 2: Consider implementing Z",
  "..."
]

Make recommendations:
- Specific and actionable
- Related to the user's query
- Based on the evidence found
"""

            try:
                response = await self.llm_provider.fw_basic_call(
                    prompt_or_messages=rec_prompt,
                    model="deepseek-v3p1",
                    temperature=0.3,
                    max_tokens=500
                )

                # Parse JSON array
                recs = json.loads(response)
                if isinstance(recs, list):
                    return recs[:5]

            except Exception as e:
                print(f"Error generating recommendations: {e}")

        # Fallback recommendations
        return [
            "Review the identified files for potential improvements",
            "Consider adding more comprehensive error handling",
            "Document the complex interactions for future maintenance"
        ]

    async def _create_examples(self,
                             query: str,
                             analysis: Dict[str, Any],
                             context: Dict[str, Any]) -> Dict[str, str]:
        """Create code examples to illustrate findings or fixes"""

        examples = {}

        # If this is a debugging/fix query, provide fix examples
        if 'bug' in query.lower() or 'error' in query.lower() or 'fix' in query.lower():
            examples['fix'] = await self._create_fix_example(
                query, analysis, context
            )

        # If this is about patterns, provide usage examples
        if 'pattern' in query.lower() or 'implement' in query.lower():
            examples['implementation'] = await self._create_implementation_example(
                query, analysis, context
            )

        # Always include a key finding example
        if analysis.get('related_files'):
            key_file = analysis['related_files'][0]
            if key_file in context.get('critical_files', {}):
                content = context['critical_files'][key_file]['content']
                examples['key_code'] = self._extract_key_example(content)

        return examples

    async def _create_fix_example(self,
                                query: str,
                                analysis: Dict[str, Any],
                                context: Dict[str, Any]) -> str:
        """Create a code example showing how to fix an issue"""
        return f"""
# Example Fix
Based on the analysis: {analysis.get('root_cause', 'Issue identified')}

# Recommended changes:
# 1. Add proper error handling
try:
    # Your code here
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {{e}}")
    # Handle gracefully

# 2. Add validation
if not input_data:
    raise ValueError("Input data cannot be empty")

# 3. Add logging
logger.info("Processing started")
# ... your code
logger.info("Processing completed")
"""

    async def _create_implementation_example(self,
                                           query: str,
                                           analysis: Dict[str, Any],
                                           context: Dict[str, Any]) -> str:
        """Create a code example showing implementation"""
        return f"""
# Implementation Example
# Based on the pattern identified in: {', '.join(analysis.get('related_files', [])[:2])}

class ExampleImplementation:
    def __init__(self):
        self._instance = None

    def get_instance(self):
        \"\"\"Singleton pattern example\"\"\"
        if not self._instance:
            self._instance = ExampleImplementation()
        return self._instance

    def process_data(self, data):
        \"\"\"Main processing method\"\"\"
        # Validate input
        if not data:
            raise ValueError("Data is required")

        # Process data
        result = self._transform(data)

        # Return result
        return result

    def _transform(self, data):
        \"\"\"Internal transformation\"\"\"
        return data.upper()
"""

    def _extract_key_example(self, content: str) -> str:
        """Extract a key example from code"""
        lines = content.split('\n')

        # Find a good example (class or function)
        for i, line in enumerate(lines):
            if line.startswith('class ') or line.startswith('def '):
                # Extract this class/function
                example_lines = [line]
                j = i + 1
                indent_level = len(line) - len(line.lstrip())

                while j < len(lines) and j < i + 20:
                    current_line = lines[j]
                    if current_line.strip() == '':
                        j += 1
                        continue
                    current_indent = len(current_line) - len(current_line.lstrip())
                    if current_line.strip() and current_indent <= indent_level:
                        break
                    example_lines.append(current_line)
                    j += 1

                return '\n'.join(example_lines[:10])

        return "# Key code example not found"

    async def debug_issue(self,
                         query: str,
                         error_message: Optional[str] = None,
                         stack_trace: Optional[str] = None) -> AnalysisResult:
        """Specialized method for debugging issues"""

        debug_prompt = f"""
DEBUG MODE - Help debug this issue:

Query: "{query}"
Error Message: {error_message or 'Not provided'}
Stack Trace: {stack_trace or 'Not provided'}

Focus on:
1. Root cause identification
2. Where the error originates
3. How it propagates
4. Specific fixes needed
"""

        # Add debugging context to analysis
        if error_message:
            debug_prompt += f"\n\nAnalyzing error: {error_message}"

        # Use specialized debugging prompt
        system_prompt = get_deep_analysis_prompt_with_mode("debugging")

        # This would integrate with the main analyze method
        # For now, return a basic structure
        return AnalysisResult(
            answer="Debugging analysis would be performed here with full context",
            confidence=0.8,
            evidence=[],
            root_cause="To be determined after analysis",
            recommendations=["Add error handling", "Check input validation"],
            related_files=[],
            examples={"debug": "# Debugging example would go here"},
            edge_cases=["Consider edge cases that might trigger this error"]
        )