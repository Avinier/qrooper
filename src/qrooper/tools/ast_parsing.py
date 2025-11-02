"""
AST and code structure parsing utilities extracted from AnalysisTools.
Tree-sitter is optional; functions fall back to simple regex heuristics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tree_sitter  # noqa: F401
    from tree_sitter import Language, Parser
    import tree_sitter_python
    import tree_sitter_javascript
    import tree_sitter_typescript
    import tree_sitter_json
    import tree_sitter_yaml
    import tree_sitter_java
    import tree_sitter_go
    import tree_sitter_php
    import tree_sitter_ruby
    import tree_sitter_c
    import tree_sitter_cpp
    import tree_sitter_rust
except Exception:
    Language = None  # type: ignore
    Parser = None  # type: ignore
    tree_sitter_python = None
    tree_sitter_javascript = None
    tree_sitter_typescript = None
    tree_sitter_json = None
    tree_sitter_yaml = None
    tree_sitter_java = None
    tree_sitter_go = None
    tree_sitter_php = None
    tree_sitter_ruby = None
    tree_sitter_c = None
    tree_sitter_cpp = None
    tree_sitter_rust = None


# Standard tool calling structure for AST and code structure analysis
oai_compatible_asttools = [
    {
        "name": "analyze_imports",
            "description": "Analyze imports and dependencies across multiple code files",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to analyze for imports"
                    }
                },
                "required": ["files"]
            }
        },
        {
            "name": "analyze_code_structure",
            "description": "Analyze the structure of a single code file to extract functions, classes, and other definitions",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the code file to analyze"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content of the file to analyze (optional, will read file if not provided)"
                    }
                },
                "required": ["file_path"]
            }
        },
        {
            "name": "extract_functions",
            "description": "Extract function definitions and their signatures from code files",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to extract functions from"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["python", "javascript", "typescript", "java", "go", "rust", "cpp", "c", "php", "ruby"],
                        "description": "Programming language of the files (optional, auto-detected if not provided)"
                    }
                },
                "required": ["files"]
            }
        },
        {
            "name": "extract_classes",
            "description": "Extract class definitions and their methods from code files",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to extract classes from"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["python", "javascript", "typescript", "java", "go", "rust", "cpp", "c", "php", "ruby"],
                        "description": "Programming language of the files (optional, auto-detected if not provided)"
                    }
                },
                "required": ["files"]
            }
        },
        {
            "name": "get_call_graph",
            "description": "Generate a call graph showing function relationships and dependencies",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to analyze for call relationships"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Maximum depth of call graph to generate (default: 3)"
                    }
                },
                "required": ["files"]
            }
        },
        {
            "name": "detect_patterns",
            "description": "Detect common code patterns, anti-patterns, and architectural elements",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to analyze for patterns"
                    },
                    "pattern_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["design_patterns", "anti_patterns", "code_smells", "architectural_patterns", "security_patterns"]
                        },
                        "description": "Types of patterns to detect (default: all)"
                    }
                },
                "required": ["files"]
            }
        }
    ]


class ASTParsing:
    """Language-aware parsers for imports and structure analysis."""

    def __init__(self) -> None:
        self.parsers: Dict[str, Any] = {}
        self._init_parsers()

    def _init_parsers(self) -> None:
        if not Parser or not Language:
            return
        try:
            self.parsers.clear()
            if tree_sitter_python:
                PY_LANGUAGE = Language(tree_sitter_python.language())
                self.parsers['python'] = Parser(PY_LANGUAGE)
            if tree_sitter_javascript:
                JS_LANGUAGE = Language(tree_sitter_javascript.language())
                self.parsers['javascript'] = Parser(JS_LANGUAGE)
            if tree_sitter_typescript:
                TS_LANGUAGE = Language(tree_sitter_typescript.language_typescript())
                self.parsers['typescript'] = Parser(TS_LANGUAGE)
            if tree_sitter_json:
                JSON_LANGUAGE = Language(tree_sitter_json.language())
                self.parsers['json'] = Parser(JSON_LANGUAGE)
            if tree_sitter_yaml:
                YAML_LANGUAGE = Language(tree_sitter_yaml.language())
                self.parsers['yaml'] = Parser(YAML_LANGUAGE)
            if tree_sitter_java:
                JAVA_LANGUAGE = Language(tree_sitter_java.language())
                self.parsers['java'] = Parser(JAVA_LANGUAGE)
            if tree_sitter_go:
                GO_LANGUAGE = Language(tree_sitter_go.language())
                self.parsers['go'] = Parser(GO_LANGUAGE)
            if tree_sitter_php:
                PHP_LANGUAGE = Language(tree_sitter_php.language())
                self.parsers['php'] = Parser(PHP_LANGUAGE)
            if tree_sitter_ruby:
                RUBY_LANGUAGE = Language(tree_sitter_ruby.language())
                self.parsers['ruby'] = Parser(RUBY_LANGUAGE)
            if tree_sitter_c:
                C_LANGUAGE = Language(tree_sitter_c.language())
                self.parsers['c'] = Parser(C_LANGUAGE)
            if tree_sitter_cpp:
                CPP_LANGUAGE = Language(tree_sitter_cpp.language())
                self.parsers['cpp'] = Parser(CPP_LANGUAGE)
            if tree_sitter_rust:
                RUST_LANGUAGE = Language(tree_sitter_rust.language())
                self.parsers['rust'] = Parser(RUST_LANGUAGE)
        except Exception as e:
            print(f"Warning: Failed to initialize parsers: {e}")

    async def analyze_imports(self, files: List[str], read_file_func) -> Dict[str, Any]:
        """Analyze imports for Python and JS/TS; read_file_func(path)->FileResult."""
        imports: Dict[str, int] = {}
        modules: List[str] = []
        dependencies: List[str] = []
        for file_path in files:
            file_result = await read_file_func(file_path)
            if getattr(file_result, 'error', None):
                continue
            for imp in self._extract_imports_regex(file_result.content):
                imports[imp] = imports.get(imp, 0) + 1
                if imp not in modules:
                    modules.append(imp)
                top = imp.split('.')[0].lstrip('.')
                if top and top not in dependencies and top not in ['os','sys','json','pathlib','typing','asyncio','dataclasses','collections','itertools','functools','datetime','time','re','math','random']:
                    dependencies.append(top)
        return {
            "imports": imports,
            "modules": sorted(modules),
            "dependencies": sorted(dependencies),
            "total_files": len(files),
        }

    def _extract_imports_regex(self, content: str) -> List[str]:
        imports: List[str] = []
        for line in content.split('\n'):
            stripped = line.strip()
            if stripped.startswith('import '):
                module = stripped[7:].split(' as ')[0].split(',')[0].strip()
                if module:
                    imports.append(module)
            elif stripped.startswith('from '):
                parts = stripped[5:].split(' import ', 1)
                if len(parts) == 2:
                    module = parts[0].strip()
                    if module:
                        imports.append(module)
        return imports

    async def analyze_code_structure(self, file_path: str, content: str) -> Dict[str, Any]:
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python', '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript',
            '.ts': 'typescript', '.tsx': 'typescript', '.java': 'java', '.go': 'go',
            '.rs': 'rust', '.c': 'c', '.h': 'c', '.cpp': 'cpp', '.cxx': 'cpp', '.cc': 'cpp',
            '.hpp': 'cpp', '.hxx': 'cpp', '.c++': 'cpp', '.h++': 'cpp', '.php': 'php',
            '.phtml': 'php', '.php3': 'php', '.php4': 'php', '.php5': 'php', '.phps': 'php',
            '.rb': 'ruby', '.rbw': 'ruby', '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml',
        }
        language = language_map.get(ext, 'text')
        return {
            "path": file_path,
            "language": language,
            "functions": [],
            "classes": [],
            "imports": [],
            "exports": [],
            "definitions": [],
        }



# uv run python -m qrooper.tools.ast_parsing
if __name__ == "__main__":
    import asyncio
    import sys

    async def _run_demo() -> None:
        # Codebase root to test against (absolute path from repo root)
        current_file = Path(__file__).resolve()
        # Walk up until we find the repository root that contains 'packages'
        repo_root = current_file
        for _ in range(8):
            if (repo_root / 'packages').is_dir():
                break
            repo_root = repo_root.parent

        # Ensure we can import qrooper utilities when running this file directly
        qrooper_src = (repo_root / 'packages' / 'qrooper' / 'src').resolve()
        if str(qrooper_src) not in sys.path:
            sys.path.insert(0, str(qrooper_src))
        from qrooper.tools.filesystem_utils import FilesystemUtils  # type: ignore

        cb_path = (repo_root / 'packages' / 'eva' / 'src' / 'eva').resolve()
        if not cb_path.exists():
            raise SystemExit(f"Demo error: codebase path does not exist: {cb_path}")
        utils = FilesystemUtils(cb_path)

        print(f"Codebase root: {utils.codebase_path}")

        # Discover files (absolute paths)
        py_files = await utils.find_files('*.py', '.', absolute=True)
        print(f"Discovered {len(py_files)} Python files.")

        astp = ASTParsing()

        # Exercise analyze_imports using FilesystemUtils.read_file (absolute paths)
        print("\n[analyze_imports] Running...")
        imports_result = await astp.analyze_imports(py_files, utils.read_file)
        print(f"Total files scanned: {imports_result['total_files']}")
        print(f"Unique modules found: {len(imports_result['modules'])}")
        top_imports = sorted(
            imports_result["imports"].items(), key=lambda kv: kv[1], reverse=True
        )[:10]
        if top_imports:
            print("Top imports:")
            for name, count in top_imports:
                print(f"  {name}: {count}")
        else:
            print("No imports detected.")

        # Exercise analyze_code_structure on a limited subset for brevity
        print("\n[analyze_code_structure] Running...")
        struct_limit = 20
        struct_count = 0
        language_counts: Dict[str, int] = {}
        for fp in py_files[:struct_limit]:
            fr = await utils.read_file(fp)
            if getattr(fr, 'error', None):
                continue
            result = await astp.analyze_code_structure(fp, fr.content)
            language = result.get("language", "unknown")
            language_counts[language] = language_counts.get(language, 0) + 1
            struct_count += 1
        print(f"Structures analyzed: {struct_count}")
        if language_counts:
            print("Language distribution:")
            for lang, cnt in sorted(language_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {lang}: {cnt}")

        # Exercise private regex import extractor on a sample snippet
        print("\n[_extract_imports_regex] Sample run...")
        sample_snippet = (
            "import os\n"
            "from pathlib import Path\n"
            "import numpy as np, sys\n"
            "from mypkg.sub.mod import thing, other as alias\n"
        )
        extracted = astp._extract_imports_regex(sample_snippet)
        print(f"Extracted from sample: {extracted}")

    asyncio.run(_run_demo())
