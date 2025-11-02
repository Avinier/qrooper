# Qrooper - LLM-Powered 3-Pass Codebase Analysis System

## Overview

Qrooper is an LLM-powered codebase analysis system featuring agents with distinct personalities and specialized capabilities. The system follows a 3-pass analysis strategy where each agent builds upon the work of the previous one.

## Architecture

### Intelligent Pass Determination

QrooperEngine uses a **DeciderAgent** that runs once at the beginning to determine the optimal number of analysis passes:

1. **1 Pass (Reconnaissance only)**: For simple questions
   - "What files are in src/?"
   - "What technologies are used?"
   - "Where is the main entry point?"

2. **2 Passes (Reconnaissance + Pattern)**: For relationship questions
   - "How does authentication flow?"
   - "What design patterns are used?"
   - "How are services connected?"

3. **3 Passes (Full Analysis)**: For complex debugging/detailed questions
   - "Why is authentication failing?"
   - "How is user data validated and stored?"
   - "Find the root cause of this performance issue"

### The Adaptive Analysis Flow

```
User Query
     ↓
┌─────────────────┐
│  DeciderAgent    │  Analyzes query once
│  (Smart Decision) │  • Determines complexity
│                 │  • Selects 1/2/3 passes
│                 │  • Identifies focus areas
└─────────────────┘
     ↓
┌─────────────────┐  Only if needed
│  SCOUT          │  Pass 1: Always runs
│  (Reconnaissance)│  • Maps codebase structure
│                 │  • Identifies key files
└─────────┬───────┘
         │
         ↓ (If 2+ passes)
┌─────────────────┐
│  CONNECT        │  Pass 2: Relationships
│  (Pattern       │  • Data flows
│   Recognition)   │  • Design patterns
└─────────┬───────┘
         │
         ↓ (If 3 passes)
┌─────────────────┐
│  DEEPDIVE       │  Pass 3: Details
│  (Deep Analysis) │  • Evidence
│                 │  • Root causes
└─────────────────┘
     ↓
Optimized Answer
```

### Efficiency Benefits

- **Faster responses**: Simple queries only need 1 pass
- **Reduced LLM calls**: No redundant analysis passes
- **Smart resource usage**: Only runs necessary agents
- **Early termination**: Stops when sufficient information is gathered

### Agent Personalities

#### 1. Scout (Reconnaissance Agent)
- **Personality**: Curious, systematic, highly observant
- **Role**: Maps the codebase structure and identifies key areas
- **Tools**: File system navigation, grep, basic code analysis
- **Output**: Structural overview with focus areas for next agent

#### 2. Connect (Pattern Recognition Agent)
- **Personality**: Analytical, insightful, pattern-obsessed
- **Role**: Understands relationships, flows, and patterns
- **Tools**: Advanced grep, dependency analysis, flow tracing
- **Output**: Pattern maps, relationship diagrams, flow charts

#### 3. DeepDive (Deep Analysis Agent)
- **Personality**: Meticulous, analytical, detail-oriented
- **Role**: Provides specific, detailed answers with evidence
- **Tools**: Full code analysis, debugging, architectural review
- **Output**: Detailed answer with evidence, recommendations, examples

## Quick Start

### Basic Usage

```python
from qrooper import analyze_codebase

# Simple analysis
result = await analyze_codebase(
    codebase_path="./my_project",
    query="How does authentication work?",
    mode="default"
)

print(result.answer)
print(f"Confidence: {result.confidence}")
```

### Advanced Usage

```python
from qrooper import QrooperOrchestrator

# Create orchestrator
orchestrator = QrooperOrchestrator("./my_project")

# Analyze with specific mode
result = await orchestrator.analyze(
    query="What's the root cause of this bug?",
    mode="debugging",
    files_to_focus=["auth.py", "models.py"]
)

# Get detailed evidence
for evidence in result.evidence:
    print(f"{evidence.file_path}:{evidence.line_number}")
    print(f"  {evidence.code_snippet}")
    print(f"  → {evidence.explanation}")
```

## Analysis Modes

The system supports specialized analysis modes:

### Default Mode
Standard analysis for general questions about code structure and behavior

### Debugging Mode
```python
result = await orchestrator.debug(
    query="User authentication is failing",
    error_message="Invalid token error",
    stack_trace="..."
)
```

### Architecture Mode
```python
result = await orchestrator.analyze_architecture(
    "What design patterns are used in this service layer?"
)
```

### Security Mode
```python
result = await orchestrator.analyze_security(
    "Are there any SQL injection vulnerabilities?"
)
```

### Performance Mode
```python
result = await orchestrator.analyze_performance(
    "Why is this database query so slow?"
)
```

## Key Features

### 1. **LLM-Powered Analysis**
All agents use LLM for intelligent decision-making and pattern recognition

### 2. **Context Passing**
Each agent receives and builds upon context from previous agents

### 3. **Evidence-Based Answers**
DeepDive provides concrete evidence with file paths and line numbers

### 4. **Caching**
Results are cached for performance (configurable)

### 5. **Specialized Prompts**
Each agent has a personality-driven system prompt for consistent behavior

### 6. **Read-Only Operations**
All analysis is read-only - no code modifications

## API Reference

### Core Classes

#### `QrooperOrchestrator`
Main orchestrator for the 3-pass analysis system

```python
orchestrator = QrooperOrchestrator(
    codebase_path: Path,
    llm_provider: Optional[QrooperLLM] = None,
    enable_caching: bool = True
)
```

#### `QrooperV2AnalysisResult`
Result object containing the complete analysis

```python
@dataclass
class QrooperAnalysisResult:
    query: str
    answer: str
    confidence: float
    agent_results: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    recommendations: List[str]
    examples: Dict[str, str]
    analysis_time: float
    files_analyzed: int
    analysis_mode: str
```

### Convenience Functions

#### `analyze_codebase()`
One-off analysis function

```python
result = await analyze_codebase(
    codebase_path: str,
    query: str,
    mode: str = "default",
    files_to_focus: Optional[List[str]] = None
)
```

#### `debug_issue()`
Specialized debugging function

```python
result = await debug_issue(
    codebase_path: str,
    issue_description: str,
    error_message: Optional[str] = None,
    stack_trace: Optional[str] = None
)
```

## Agent Details

All agent prompts are centrally located in `src/qrooper/prompts.py` for easy maintenance and consistency.

### Scout (ReconnaissanceAgent)

**Capabilities:**
- Directory traversal and file discovery
- Language detection
- Entry point identification
- Configuration file discovery
- LLM-guided targeted exploration

**Output Structure:**
```json
{
  "structure": {
    "directories": [...],
    "key_files": [...],
    "languages": {"python": 45},
    "tech_stack": [...],
    "entry_points": [...]
  },
  "summary": "Overview of codebase...",
  "key_findings": [...],
  "context_for_next_agent": {...}
}
```

### Connect (PatternRecognitionAgent)

**Capabilities:**
- Design pattern recognition
- Data flow tracing
- Control flow analysis
- Dependency mapping
- Relationship identification

**Output Structure:**
```json
{
  "patterns_identified": {
    "design_patterns": [...],
    "data_flows": [...],
    "control_flows": [...],
    "relationships": {...}
  },
  "insights": [...],
  "context_for_deep_agent": {...}
}
```

### DeepDive (DeepAnalysisAgent)

**Capabilities:**
- Detailed code analysis
- Root cause identification
- Evidence extraction
- Recommendation generation
- Example creation

**Output Structure:**
```json
{
  "answer": "Detailed answer...",
  "confidence": 0.95,
  "evidence": [...],
  "root_cause": "...",
  "recommendations": [...],
  "examples": {...}
}
```

## Configuration

### Environment Variables

```bash
# Required for LLM operations
FIREWORKS_API_KEY=your_api_key_here
GOOGLE_API_KEY=your_google_key_here  # Optional
GLM_API_KEY=your_glm_key_here      # Optional
```

### Custom LLM Provider

```python
from qrooper import QrooperLLM, QrooperOrchestrator

# Create custom LLM provider
llm = QrooperLLM(model="deepseek-v3p1")

# Use with orchestrator
orchestrator = QrooperOrchestrator("./my_project", llm_provider=llm)
```

## Examples

### Example 1: Understanding a Codebase

```python
# Ask about architecture
result = await analyze_codebase(
    "./fastapi-project",
    "How is the authentication system structured?"
)

print(f"Architecture: {result.agent_results['pattern_recognition']['patterns_found']} patterns found")
print(f"Answer: {result.answer}")
```

### Example 2: Debugging an Issue

```python
# Debug a specific issue
result = await debug_issue(
    "./my-project",
    "Users can't log in with valid credentials",
    error_message="401 Unauthorized",
    stack_trace="auth.py:45"
)

print(f"Root cause: {result.agent_results['deep_analysis']['root_cause']}")
for rec in result.recommendations:
    print(f"Fix: {rec}")
```

### Example 3: Performance Analysis

```python
# Analyze performance
result = await orchestrator.analyze_performance(
    "Why are database queries slow in the user service?"
)

# Get performance recommendations
print("Performance Issues:")
for rec in result.recommendations:
    print(f"• {rec}")
```


## Best Practices

1. **Be Specific in Queries**: The more specific your question, the better the analysis
2. **Use Appropriate Modes**: Choose debugging, architecture, security, or performance modes when relevant
3. **Provide Context**: Include error messages or stack traces when debugging
4. **Limit Scope**: For large codebases, specify focus files to improve accuracy
5. **Review Evidence**: Always check the provided evidence for validity

## Limitations

- Read-only analysis (no code modifications)
- Requires API keys for LLM operations
- Large codebases may hit context limits
- Analysis quality depends on LLM provider performance

## Testing

Run the test suite to verify functionality:

```bash
cd packages/qrooper
python test_qrooper_v2.py
```

## Contributing

When extending the system:
1. Maintain agent personality consistency
2. Follow the 3-pass pattern
3. Add evidence for all claims
4. Test with various codebases
5. Document new capabilities

## License

This project is part of the SuperServerAI agentic systems suite.