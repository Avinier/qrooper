"""
Pattern Recognition Agent - LLM-powered pattern and flow analyzer
Second pass of the 3-pass analysis strategy
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from ..prompts import PATTERN_RECOGNITION_AGENT_PROMPT
from ..tools import FilesystemUtils
from ..agents.llm_calls import QrooperLLM
from ..agents.reconnaissance import ReconnaissanceResult


@dataclass
class DataFlow:
    """Represents data flow through the system"""
    source: str
    destination: str
    data_type: str
    transformation: str
    medium: str  # function call, API, database, etc.
    file_path: str
    confidence: float = 0.8


@dataclass
class ControlFlow:
    """Represents control flow in the system"""
    trigger: str
    sequence: List[str]
    conditions: List[str]
    exceptions: List[str]
    file_path: str


@dataclass
class PatternMatch:
    """Represents a matched design pattern"""
    pattern_name: str
    confidence: float
    locations: List[Dict[str, Any]]
    description: str


@dataclass
class PatternRecognitionResult:
    """Result of pattern recognition analysis"""
    patterns_identified: Dict[str, Any]
    insights: List[str]
    context_for_deep_agent: Dict[str, Any]
    data_flows: List[DataFlow]
    control_flows: List[ControlFlow]
    pattern_matches: List[PatternMatch]
    analysis_time: float
    confidence: float


class PatternRecognitionAgent:
    """
    Connect - The LLM-powered pattern recognition agent.
    Understands relationships, flows, and patterns in code.
    """

    def __init__(self, codebase_path: Path, llm_provider: QrooperLLM):
        """
        Initialize pattern recognition agent

        Args:
            codebase_path: Path to the codebase
            llm_provider: LLM provider for analysis
        """
        self.codebase_path = Path(codebase_path)
        self.llm_provider = llm_provider
        self.tools = FilesystemUtils(self.codebase_path)
        self.system_prompt = PATTERN_RECOGNITION_AGENT_PROMPT

        # Design pattern definitions for LLM
        self.pattern_definitions = {
            "Singleton": ["__new__", "_instance", "getInstance", "instance", "self._instance"],
            "Factory": ["create", "build", "factory", "Factory", "make_"],
            "Observer": ["observe", "notify", "subscribe", "listener", "event", "emit"],
            "Strategy": ["strategy", "Strategy", "algorithm", "StrategyPattern"],
            "Adapter": ["adapter", "Adapter", "adapt", "wrap", "wrapper"],
            "Decorator": ["decorator", "Decorator", "@", "wrapper"],
            "Repository": ["repository", "Repository", "save", "find", "delete", "update"],
            "Service": ["service", "Service", "business", "logic"],
            "Controller": ["controller", "Controller", "handle", "request", "endpoint"],
            "Model": ["model", "Model", "data", "entity", "schema"],
            "MVC": ["Model", "View", "Controller"],
            "Dependency Injection": ["inject", "provider", "container", "DI"],
            "Builder": ["builder", "Builder", "build_", "with_"],
            "Command": ["command", "Command", "execute", "invoker"],
            "Facade": ["facade", "Facade", "simplify", "interface"],
            "Proxy": ["proxy", "Proxy", "delegate", "surrogate"]
        }

    async def analyze(self,
                      query: str,
                      recon_result: ReconnaissanceResult) -> PatternRecognitionResult:
        """
        Perform pattern recognition analysis

        Args:
            query: User's query
            recon_result: Results from reconnaissance agent

        Returns:
            Comprehensive PatternRecognitionResult
        """
        import time
        start_time = time.time()

        # Phase 1: Extract files to analyze
        files_to_analyze = await self._determine_files_to_analyze(
            recon_result
        )

        # Phase 2: LLM-guided pattern discovery
        pattern_discovery = await self._llm_pattern_discovery(
            query, recon_result, files_to_analyze
        )

        # Phase 3: Flow analysis
        flow_analysis = await self._analyze_flows(
            query, recon_result, files_to_analyze, pattern_discovery
        )

        # Phase 4: Synthesize insights
        result = await self._synthesize_patterns(
            query, recon_result, pattern_discovery, flow_analysis
        )

        result.analysis_time = time.time() - start_time

        return result

    async def _determine_files_to_analyze(self,
                                        recon_result: ReconnaissanceResult) -> List[str]:
        """Determine which files to analyze for patterns"""

        files_to_analyze = set()

        # Start with files focused during reconnaissance (these have full paths)
        files_to_analyze.update(recon_result.files_focused)

        # Add files from reconnaissance context
        context = recon_result.context_for_next_agent
        files_to_analyze.update(context.get('files_to_examine', []))

        # Add files from focus areas
        focus_areas = context.get('focus_areas', [])
        for area in focus_areas:
            if area.endswith('/'):
                # It's a directory
                area_files = await self.tools.list_directory(area, recursive=False)
                files_to_analyze.update([f"{area}{f}" for f in area_files[:5]])
            else:
                files_to_analyze.add(area)

        # Ensure we have relevant file types
        all_files = list(files_to_analyze)
        filtered_files = []
        for file_path in all_files:
            if any(file_path.endswith(ext) for ext in ['.py', '.js', '.ts', '.java', '.go', '.rs']):
                filtered_files.append(file_path)

        return filtered_files[:15]  # Limit to prevent overwhelming the LLM

    async def _llm_pattern_discovery(self,
                                     query: str,
                                     recon_result: ReconnaissanceResult,
                                     files_to_analyze: List[str]) -> Dict[str, Any]:
        """Use LLM to discover patterns in the code"""

        # First, gather code snippets from key files
        code_snippets = {}
        for file_path in files_to_analyze[:10]:  # Limit files
            content = await self.tools.read_file(file_path)
            if not content.error:
                # Extract key parts (classes, functions, imports)
                snippet = await self._extract_key_snippets(content.content)
                code_snippets[file_path] = snippet

        # Ask LLM to identify patterns
        pattern_prompt = f"""
Analyze these code snippets for design patterns and relationships.

User Query: "{query}"

Reconnaissance Summary: {recon_result.summary}

Code Snippets:
{json.dumps(code_snippets, indent=2)}

Look for these design patterns: {', '.join(self.pattern_definitions.keys())}

Also identify:
1. Data flows between components
2. Control flow patterns
3. Component relationships
4. Architectural decisions

Return JSON with:
{{
  "design_patterns": [
    {{
      "name": "Repository",
      "confidence": 0.9,
      "locations": ["repositories/user.py"],
      "evidence": "UserRepository class with save/find methods"
    }}
  ],
  "component_relationships": [
    {{
      "component": "UserController",
      "depends_on": ["UserService", "UserModel"],
      "relationship_type": "composition"
    }}
  ],
  "architectural_insights": [
    "Clean architecture with separation of concerns",
    "Dependency injection pattern used throughout"
  ],
  "data_flow_hints": [
    "Request -> Controller -> Service -> Repository -> Database"
  ]
}}
"""

        try:
            response = await self.llm_provider.fw_basic_call(
                prompt_or_messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": pattern_prompt}
                ],
                model="deepseek-v3p1",
                temperature=0.3,
                max_tokens=2000
            )

            return json.loads(response)

        except Exception as e:
            print(f"Error in LLM pattern discovery: {e}")
            return {"design_patterns": [], "component_relationships": [], "architectural_insights": []}

    async def _extract_key_snippets(self, content: str) -> Dict[str, Any]:
        """Extract key parts of code for analysis"""
        lines = content.split('\n')
        snippets = {
            "imports": [],
            "classes": [],
            "functions": [],
            "key_lines": []
        }

        for i, line in enumerate(lines[:100]):  # First 100 lines
            stripped = line.strip()

            if stripped.startswith(('import ', 'from ')):
                snippets["imports"].append(stripped)
            elif stripped.startswith('class '):
                # Get class definition
                class_end = min(i + 20, len(lines))
                snippets["classes"].append({
                    "definition": stripped,
                    "context": '\n'.join(lines[i:class_end])
                })
            elif stripped.startswith('def '):
                # Get function definition
                func_end = min(i + 10, len(lines))
                snippets["functions"].append({
                    "definition": stripped,
                    "context": '\n'.join(lines[i:func_end])
                })

            # Also capture some key lines with patterns
            for pattern_name, keywords in self.pattern_definitions.items():
                if any(keyword in stripped for keyword in keywords[:3]):
                    snippets["key_lines"].append({
                        "line": stripped,
                        "line_number": i + 1,
                        "potential_pattern": pattern_name
                    })

        return snippets

    async def _analyze_flows(self,
                            query: str,
                            recon_result: ReconnaissanceResult,
                            files_to_analyze: List[str],
                            pattern_discovery: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data and control flows"""

        flow_analysis = {
            "data_flows": [],
            "control_flows": [],
            "sequence_diagrams": []
        }

        # Use grep to trace specific patterns
        for file_path in files_to_analyze[:5]:
            # Find function calls
            func_calls = await self.tools.grep(
                pattern=r"\w+\(",
                path=file_path,
                max_results=20
            )

            # Find data assignments/transformations
            data_assign = await self.tools.grep(
                pattern=r"(data|result|response|output)\s*=",
                path=file_path,
                max_results=10
            )

            # Find database operations
            db_ops = await self.tools.grep(
                pattern=r"(select|insert|update|delete|create|read)",
                path=file_path,
                ignore_case=True,
                max_results=10
            )

            flow_analysis["data_flows"].extend([
                {
                    "file": file_path,
                    "function_calls": func_calls.matches[:5],
                    "data_operations": data_assign.matches[:5],
                    "db_operations": db_ops.matches[:5]
                }
            ])

        # Ask LLM to interpret flows
        flow_prompt = f"""
Based on this trace information, map the data and control flows.

Query: "{query}"

Flow Traces:
{json.dumps(flow_analysis, indent=2)}

Identify and return:
1. Main data flows (what data moves where)
2. Control flows (what triggers what)
3. Entry points to exit points paths

JSON format:
{{
  "data_flows": [
    {{
      "source": "API endpoint",
      "destination": "Service layer",
      "data_type": "User request",
      "path": "api/user.py -> services/user_service.py"
    }}
  ],
  "control_flows": [
    {{
      "trigger": "POST /users",
      "sequence": ["validation", "service", "database", "response"]
    }}
  ]
}}
"""

        try:
            response = await self.llm_provider.fw_basic_call(
                prompt_or_messages=flow_prompt,
                model="deepseek-v3p1",
                temperature=0.3,
                max_tokens=1500
            )

            return json.loads(response)

        except Exception as e:
            print(f"Error in flow analysis: {e}")
            return flow_analysis

    async def _synthesize_patterns(self,
                                 query: str,
                                 recon_result: ReconnaissanceResult,
                                 pattern_discovery: Dict[str, Any],
                                 flow_analysis: Dict[str, Any]) -> PatternRecognitionResult:
        """Synthesize all pattern and flow analysis"""

        synthesis_prompt = f"""
You are Connect, the pattern recognition agent. Synthesize all findings about patterns and flows.

User Query: "{query}"

Reconnaissance Context: {json.dumps(recon_result.context_for_next_agent, indent=2)}

Pattern Discovery: {json.dumps(pattern_discovery, indent=2)}

Flow Analysis: {json.dumps(flow_analysis, indent=2)}

Provide comprehensive analysis in JSON format:
{{
  "patterns_identified": {{
    "design_patterns": ["Repository", "Factory"],
    "data_flows": [
      {{
        "from": "API endpoint",
        "to": "Service layer",
        "data": "User request",
        "path": "api/user.py -> services/user_service.py"
      }}
    ],
    "control_flows": [
      {{
        "trigger": "POST /users",
        "sequence": ["validation", "service", "database", "response"]
      }}
    ],
    "relationships": {{
      "UserController": ["depends_on": ["UserService", "UserModel"]]
    }}
  }},
  "insights": [
    "Clean separation of concerns with service layer",
    "Dependency injection used throughout"
  ],
  "context_for_deep_agent": {{
    "critical_path": ["api/user.py", "services/user_service.py"],
    "pattern_locations": {{
      "Repository pattern": "repositories/user_repository.py"
    }},
    "complex_interactions": "Description of complex flows"
  }}
}}

Focus on patterns that help the deep analysis agent solve the query.
"""

        try:
            response = await self.llm_provider.fw_basic_call(
                prompt_or_messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": synthesis_prompt}
                ],
                model="deepseek-v3p1",
                temperature=0.3,
                max_tokens=2000
            )

            result_data = json.loads(response)

            # Convert to dataclasses
            data_flows = []
            for flow in result_data.get('patterns_identified', {}).get('data_flows', []):
                data_flows.append(DataFlow(
                    source=flow.get('from', ''),
                    destination=flow.get('to', ''),
                    data_type=flow.get('data', ''),
                    transformation='',
                    medium=flow.get('path', ''),
                    file_path=flow.get('path', ''),
                    confidence=0.8
                ))

            control_flows = []
            for flow in result_data.get('patterns_identified', {}).get('control_flows', []):
                control_flows.append(ControlFlow(
                    trigger=flow.get('trigger', ''),
                    sequence=flow.get('sequence', []),
                    conditions=[],
                    exceptions=[],
                    file_path=''
                ))

            pattern_matches = []
            for pattern in result_data.get('patterns_identified', {}).get('design_patterns', []):
                pattern_matches.append(PatternMatch(
                    pattern_name=pattern,
                    confidence=0.8,
                    locations=[],
                    description=f"Identified {pattern} pattern"
                ))

            return PatternRecognitionResult(
                patterns_identified=result_data.get('patterns_identified', {}),
                insights=result_data.get('insights', []),
                context_for_deep_agent=result_data.get('context_for_deep_agent', {}),
                data_flows=data_flows,
                control_flows=control_flows,
                pattern_matches=pattern_matches,
                analysis_time=0,  # Will be set by caller
                confidence=0.85
            )

        except Exception as e:
            print(f"Error in synthesis: {e}")
            return PatternRecognitionResult(
                patterns_identified={},
                insights=["Pattern analysis completed with basic findings"],
                context_for_deep_agent={
                    "critical_path": recon_result.context_for_next_agent.get('files_to_examine', []),
                    "pattern_locations": {},
                    "complex_interactions": "Further analysis needed"
                },
                data_flows=[],
                control_flows=[],
                pattern_matches=[],
                analysis_time=0,
                confidence=0.5
            )

    async def trace_data_flow(self, start_file: str, end_file: str) -> List[DataFlow]:
        """Trace data flow between two points"""
        flows = []

        # Find connections in start file
        start_content = await self.tools.read_file(start_file)
        if not start_content.error:
            # Look for function calls or imports that might lead to end_file
            connections = await self._find_connections(
                start_content.content, end_file
            )
            for conn in connections:
                flows.append(DataFlow(
                    source=start_file,
                    destination=conn['target'],
                    data_type=conn['data_type'],
                    transformation=conn['operation'],
                    medium='function_call',
                    file_path=conn['file_path']
                ))

        return flows

    async def _find_connections(self, content: str, target_file: str) -> List[Dict[str, Any]]:
        """Find connections in code that lead to target file"""
        connections = []

        # Simple pattern matching for connections
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'import' in line and target_file.replace('.py', '') in line:
                connections.append({
                    'target': target_file,
                    'data_type': 'module',
                    'operation': 'import',
                    'file_path': target_file,
                    'line': i + 1
                })

        return connections