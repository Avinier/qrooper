"""
Filesystem and grep utilities extracted from AnalysisTools.
Provides safe, read-only operations for exploring codebases.
"""

import os
import subprocess
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import ripgrepy
except Exception:  # ripgrepy is optional; fallback will be used
    ripgrepy = None  # type: ignore


# Standard tool calling structure for filesystem operations
oai_compatible_filesystemtools = [
    {
        "name": "list_directory",
            "description": "List files and directories in a given path with various options",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (default: current directory)"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list files recursively"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth for recursive listing (default: 3)"
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Whether to show hidden files and directories"
                    },
                    "absolute": {
                        "type": "boolean",
                        "description": "Whether to return absolute paths (default: true)"
                    }
                },
                "required": []
            }
        },
        {
            "name": "read_file",
            "description": "Read contents of a file with optional line range and context",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line number (1-based, default: 1)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Ending line number (optional)"
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines to include around specified range"
                    }
                },
                "required": ["path"]
            }
        },
        {
            "name": "find_files",
            "description": "Find files matching a pattern or criteria",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Pattern to search for (supports wildcards)"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory path to search in (default: current directory)"
                    },
                    "file_type": {
                        "type": "string",
                        "enum": ["name", "path", "extension"],
                        "description": "Type of pattern matching: 'name' for filename, 'path' for path contains, 'extension' for file extension"
                    },
                    "exclude_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of patterns to exclude from results"
                    },
                    "absolute": {
                        "type": "boolean",
                        "description": "Whether to return absolute paths (default: true)"
                    }
                },
                "required": ["pattern"]
            }
        },
        {
            "name": "grep",
            "description": "Search for text patterns within files using ripgrep",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory path to search in (default: current directory)"
                    },
                    "file_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File patterns to include in search (e.g., ['*.py', '*.js'])"
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Whether to ignore case in search"
                    },
                    "line_numbers": {
                        "type": "boolean",
                        "description": "Whether to include line numbers in results"
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines around matches"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 100)"
                    },
                    "absolute": {
                        "type": "boolean",
                        "description": "Whether to return absolute file paths (default: true)"
                    }
                },
                "required": ["pattern"]
            }
        },
        {
            "name": "get_file_tree",
            "description": "Get a structured tree representation of files and directories",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to analyze (default: current directory)"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth to traverse (default: 3)"
                    }
                },
                "required": []
            }
        },
        {
            "name": "detect_languages",
            "description": "Detect programming languages used in the codebase by file extensions",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to analyze (default: current directory)"
                    }
                },
                "required": []
            }
        }
    ]


@dataclass
class FileResult:
    """Result from file reading operations"""
    path: str
    content: str
    lines: int
    encoding: str = "utf-8"
    error: Optional[str] = None


@dataclass
class GrepResult:
    """Result from grep operations"""
    pattern: str
    matches: List[Dict[str, Any]]
    total_matches: int
    files_searched: int


@dataclass
class CommandResult:
    """Result from bash command execution"""
    command: str
    stdout: str
    stderr: str
    returncode: int
    success: bool




class FilesystemUtils:
    """
    Filesystem and search helpers for codebase analysis.
    """

    def __init__(self, codebase_path: Path):
        self.codebase_path = Path(codebase_path).resolve()
        if not self.codebase_path.exists():
            raise ValueError(f"Codebase path does not exist: {self.codebase_path}")

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _get_rg_path(self) -> Optional[str]:
        """Find a usable ripgrep binary path, respecting common install locations."""
        system = platform.system().lower()
        machine = platform.machine().lower()

        # Platform-specific paths
        common_paths = []

        if system == "darwin":  # macOS
            # Standard macOS installations
            common_paths.extend([
                "/opt/homebrew/bin/rg",  # Apple Silicon
                "/usr/local/bin/rg",     # Intel
                "/usr/bin/rg",
            ])
        elif system == "linux":
            # Linux Docker paths
            common_paths.extend([
                "/usr/local/bin/rg",     # Common installation
                "/usr/bin/rg",           # Package manager installation
                "/bin/rg",               # Some minimal installations
                "/app/venv/bin/rg",      # Virtual environment in Docker
                "/venv/bin/rg",          # Another venv pattern
                "/root/.cargo/bin/rg",   # Rust/Cargo installation
                "/usr/local/cargo/bin/rg",  # System cargo
            ])

            # Architecture-specific paths if needed
            if machine in ["x86_64", "amd64"]:
                common_paths.extend([
                    "/usr/local/bin/rg-x86_64-unknown-linux-gnu",
                    "/opt/ripgrep/rg",
                ])
            elif machine in ["aarch64", "arm64"]:
                common_paths.extend([
                    "/usr/local/bin/rg-aarch64-unknown-linux-gnu",
                    "/opt/ripgrep/rg",
                ])
        else:
            # Fallback for other systems
            common_paths.extend([
                "/usr/local/bin/rg",
                "/usr/bin/rg",
                "/bin/rg",
            ])

        # Check all paths
        for p in common_paths:
            if os.path.exists(p) and os.access(p, os.X_OK):  # Also check if executable
                return p

        # Try to find via PATH
        try:
            import shutil
            found = shutil.which('rg')
            if found and os.path.exists(found) and os.access(found, os.X_OK):
                return found
        except Exception:
            pass

        # Log that we couldn't find ripgrep (could use logging in production)
        return None

    async def _run_command(self, command: str, timeout: int = 30) -> CommandResult:
        try:
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.codebase_path,
            )
            return CommandResult(
                command=command,
                stdout=process.stdout,
                stderr=process.stderr,
                returncode=process.returncode,
                success=process.returncode == 0,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                command=command,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                returncode=-1,
                success=False,
            )
        except Exception as e:
            return CommandResult(
                command=command,
                stdout="",
                stderr=f"Error running command: {str(e)}",
                returncode=-1,
                success=False,
            )

    # ---------------------------------------------------------------------
    # Filesystem operations
    # ---------------------------------------------------------------------
    async def list_directory(self, path: str = ".", recursive: bool = False,
                             max_depth: int = 3, show_hidden: bool = False,
                             absolute: bool = True) -> List[str]:
        full_path = self.codebase_path / path
        if not full_path.exists():
            return []

        items: List[str] = []
        if recursive:
            rg_path = self._get_rg_path()
            if rg_path:
                cmd_parts = ["\"" + rg_path + "\"", "--files"]
                if show_hidden:
                    cmd_parts.append("--hidden")
                cmd_parts.append(str(full_path))
                cmd = " ".join(cmd_parts)
                result = await self._run_command(cmd)
                file_paths: List[str] = []
                if result.success and result.stdout.strip():
                    for line in result.stdout.strip().split('\n'):
                        if not line:
                            continue
                        try:
                            p = Path(line)
                            if not p.is_absolute():
                                p = (self.codebase_path / p).resolve()
                            rel = str(p.relative_to(self.codebase_path))
                            file_paths.append(rel)
                        except Exception:
                            continue
                dir_set = set()
                for f in file_paths:
                    if not show_hidden and any(part.startswith('.') for part in Path(f).parts):
                        continue
                    items.append(str((self.codebase_path / f).resolve()) if absolute else f)
                    rel_to_base = Path(f).relative_to(path) if path != "." else Path(f)
                    parts = rel_to_base.parts
                    for i in range(1, min(len(parts), max_depth) + 1):
                        d = Path(path) / Path(*parts[:i]) if path != "." else Path(*parts[:i])
                        if (self.codebase_path / d).is_dir():
                            dir_set.add(str(d))
                # Add directories, respecting absolute flag
                if absolute:
                    items.extend([str((self.codebase_path / d).resolve()) for d in sorted(dir_set)])
                else:
                    items.extend(sorted(dir_set))
            else:
                for root, dirs, files in os.walk(full_path):
                    depth = Path(root).relative_to(full_path).parts
                    if len(depth) >= max_depth:
                        dirs.clear()
                        continue
                    for item in dirs + files:
                        if not show_hidden and item.startswith('.'):
                            continue
                        rel = str(Path(root).relative_to(self.codebase_path) / item)
                        items.append(str((self.codebase_path / rel).resolve()) if absolute else rel)
        else:
            for item in full_path.iterdir():
                if not show_hidden and item.name.startswith('.'):
                    continue
                items.append(str(item.resolve()) if absolute else str(item.relative_to(self.codebase_path)))

        return sorted(items)

    async def read_file(self, path: str, start_line: int = 1,
                        end_line: Optional[int] = None,
                        context_lines: int = 0) -> FileResult:
        full_path = self.codebase_path / path
        if not full_path.exists():
            return FileResult(path=path, content="", lines=0, error=f"File not found: {path}")
        if not full_path.is_file():
            return FileResult(path=path, content="", lines=0, error=f"Path is not a file: {path}")

        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                all_lines = f.readlines()
            if context_lines > 0:
                start_line = max(1, start_line - context_lines)
                if end_line:
                    end_line = end_line + context_lines
            start_idx = start_line - 1
            end_idx = end_line if end_line is None else end_line
            selected_lines = all_lines[start_idx:end_idx]
            content = ''.join(selected_lines)
            return FileResult(path=path, content=content, lines=len(selected_lines))
        except Exception as e:
            return FileResult(path=path, content="", lines=0, error=f"Error reading file: {str(e)}")

    async def find_files(self, pattern: str, path: str = ".",
                         file_type: str = "name",
                         exclude_patterns: Optional[List[str]] = None,
                         absolute: bool = True) -> List[str]:
        full_path = self.codebase_path / path
        matches: List[str] = []
        exclude_patterns = exclude_patterns or []
        try:
            rg_path = self._get_rg_path()
            if rg_path:
                cmd_parts = ["\"" + rg_path + "\"", "--files"]
                globs: List[str] = []
                if file_type == "name":
                    globs.append(pattern)
                elif file_type == "path":
                    globs.append(f"*{pattern}*")
                elif file_type == "extension":
                    globs.append(f"*.{pattern}")
                else:
                    globs.append(pattern)
                for g in globs:
                    cmd_parts.extend(["--glob", g])
                for ex in (exclude_patterns or []):
                    cmd_parts.extend(["--glob", f"!{ex}"])
                cmd_parts.append(str(full_path))
                cmd = " ".join(cmd_parts)
                result = await self._run_command(cmd)
                if result.success:
                    for line in result.stdout.strip().split('\n'):
                        if line and os.path.isfile(line):
                            p = Path(line)
                            if not p.is_absolute():
                                p = (self.codebase_path / p).resolve()
                            rel_path = str(p.relative_to(self.codebase_path))
                            if not any(pat in rel_path for pat in (exclude_patterns or [])):
                                matches.append(str(p) if absolute else rel_path)
            else:
                if file_type == "name":
                    cmd = f"find {full_path} -name '{pattern}' -type f"
                elif file_type == "path":
                    cmd = f"find {full_path} -path '*{pattern}*' -type f"
                elif file_type == "extension":
                    cmd = f"find {full_path} -name '*.{pattern}' -type f"
                else:
                    cmd = f"find {full_path} -name '{pattern}' -type f"
                result = await self._run_command(cmd)
                if result.success:
                    for line in result.stdout.strip().split('\n'):
                        if line and os.path.isfile(line):
                            p = Path(line)
                            if not p.is_absolute():
                                p = (self.codebase_path / p).resolve()
                            rel_path = str(p.relative_to(self.codebase_path))
                            if not any(pat in rel_path for pat in (exclude_patterns or [])):
                                matches.append(str(p) if absolute else rel_path)
        except Exception as e:
            print(f"Error in find_files: {e}")
        return sorted(matches)

    # ---------------------------------------------------------------------
    # Grep operations
    # ---------------------------------------------------------------------
    async def grep(self, pattern: str, path: str = ".",
                   file_patterns: Optional[List[str]] = None,
                   ignore_case: bool = False,
                   line_numbers: bool = True,
                   context_lines: int = 0,
                   max_results: int = 100,
                   absolute: bool = True) -> GrepResult:
        full_path = self.codebase_path / path
        try:
            rg_path = self._get_rg_path()
            if not rg_path or ripgrepy is None:
                raise Exception("ripgrep not available; using fallback")
            rg = ripgrepy.Ripgrepy(pattern, str(full_path), rg_path=rg_path)
            if ignore_case:
                rg = rg.ignore_case()
            if line_numbers:
                rg = rg.line_number()
            if context_lines > 0:
                rg = rg.context(context_lines)
            if file_patterns:
                for fp in file_patterns:
                    rg = rg.glob(fp)
            exclude_patterns = [
                "*.pyc", "*.pyo", "__pycache__", ".git", ".svn",
                "node_modules", ".vscode", ".idea", "*.min.js",
                "dist", "build", "*.log", "*.tmp",
            ]
            for ex in exclude_patterns:
                rg = rg.glob(f"!{ex}")
            rg_json = rg.json()
            result = rg_json.run()
            matches: List[Dict[str, Any]] = []
            files_searched = set()
            if hasattr(result, 'as_dict'):
                data = result.as_dict
                if isinstance(data, list):
                    for item in data[:max_results]:
                        if item.get('type') == 'match' and 'data' in item:
                            d = item['data']
                            if 'path' in d and 'lines' in d:
                                filepath = d['path']['text']
                                line_data = d['lines']
                                line_number = d.get('line_number', 1)
                                content = line_data.get('text', '')
                                p = Path(filepath)
                                if not p.is_absolute():
                                    p = (self.codebase_path / p).resolve()
                                rel_path = str(p.relative_to(self.codebase_path))
                                matches.append({
                                    "file": str(p) if absolute else rel_path,
                                    "line": line_number,
                                    "content": content.strip(),
                                    "match": f"{rel_path}:{line_number}:{content}",
                                })
                                files_searched.add(rel_path)
            return GrepResult(pattern=pattern, matches=matches,
                              total_matches=len(matches), files_searched=len(files_searched))
        except Exception:
            return await self._grep_fallback(pattern, path, file_patterns, ignore_case, line_numbers, context_lines, max_results, absolute)

    async def _grep_fallback(self, pattern: str, path: str = ".",
                             file_patterns: Optional[List[str]] = None,
                             ignore_case: bool = False,
                             line_numbers: bool = True,
                             context_lines: int = 0,
                             max_results: int = 100,
                             absolute: bool = True) -> GrepResult:
        full_path = self.codebase_path / path
        cmd_parts = ["grep", "-r"]
        if ignore_case:
            cmd_parts.append("-i")
        if line_numbers:
            cmd_parts.append("-n")
        if context_lines > 0:
            cmd_parts.extend(["-C", str(context_lines)])
        if file_patterns:
            for fp in file_patterns:
                cmd_parts.extend(["--include", fp])
        exclude_patterns = [
            "*.pyc", "*.pyo", "__pycache__", ".git", ".svn",
            "node_modules", ".vscode", ".idea", "*.min.js",
            "dist", "build", "*.log",
        ]
        for ex in exclude_patterns:
            cmd_parts.extend(["--exclude", ex])
            cmd_parts.extend(["--exclude-dir", ex])
        cmd_parts.extend([pattern, str(full_path)])
        cmd = " ".join(cmd_parts)
        result = await self._run_command(cmd)
        matches: List[Dict[str, Any]] = []
        files_searched = set()
        if result.success:
            lines = result.stdout.strip().split('\n')
            for line in lines[:max_results]:
                if not line:
                    continue
                try:
                    if ':' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            filepath = parts[0]
                            linenumber = int(parts[1])
                            content = parts[2]
                            p = Path(filepath)
                            if not p.is_absolute():
                                p = (self.codebase_path / p).resolve()
                            rel_path = str(p.relative_to(self.codebase_path))
                            matches.append({
                                "file": str(p) if absolute else rel_path,
                                "line": linenumber,
                                "content": content.strip(),
                                "match": line,
                            })
                            files_searched.add(rel_path)
                except Exception:
                    continue
        return GrepResult(pattern=pattern, matches=matches,
                          total_matches=len(matches), files_searched=len(files_searched))

    # ---------------------------------------------------------------------
    # Structured file tree and language detection
    # ---------------------------------------------------------------------
    async def get_file_tree(self, path: str = ".", max_depth: int = 3) -> Dict[str, Any]:
        # Handle the case where path might be a file instead of directory
        full_path = self.codebase_path / path
        if full_path.is_file():
            # If it's a file, get the tree for its parent directory
            path = str(full_path.parent.relative_to(self.codebase_path))
            full_path = full_path.parent

        rg_path = self._get_rg_path()
        files: List[str] = []
        if rg_path:
            cmd = f"\"{rg_path}\" --files {full_path}"
            result = await self._run_command(cmd)
            if result.success and result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue
                    try:
                        p = Path(line)
                        if not p.is_absolute():
                            p = (self.codebase_path / p).resolve()
                        rel = str(p.relative_to(self.codebase_path))
                        files.append(rel)
                    except Exception:
                        continue
        else:
            for root, _, filenames in os.walk(full_path):
                for fn in filenames:
                    p = Path(root) / fn
                    try:
                        files.append(str(p.relative_to(self.codebase_path)))
                    except Exception:
                        continue

        tree: Dict[str, Any] = {"_type": "directory", "_children": {}}
        for f in files:
            rel = Path(f)
            # Use the base path for relative calculations
            if path == ".":
                rel_to_base = rel
            else:
                # Try to make it relative to the requested path
                try:
                    rel_to_base = rel.relative_to(path)
                except ValueError:
                    # If that fails, just use the relative path from root
                    rel_to_base = rel

            parts = rel_to_base.parts
            if not parts:
                continue
            cursor = tree["_children"]
            walk_depth = min(len(parts), max_depth)
            for i in range(walk_depth):
                name = parts[i]
                is_last = i == walk_depth - 1
                if is_last and i == len(parts) - 1 and (self.codebase_path / rel).is_file():
                    cursor[name] = {
                        "_type": "file",
                        "path": f,
                        "size": (self.codebase_path / f).stat().st_size if (self.codebase_path / f).exists() else 0,
                    }
                else:
                    node = cursor.get(name)
                    if not node or node.get("_type") != "directory":
                        cursor[name] = {"_type": "directory", "_children": {}}
                    cursor = cursor[name]["_children"]
        return tree

    async def detect_languages(self, path: str = ".") -> Dict[str, int]:
        extensions: Dict[str, List[str]] = {
            "python": [".py"],
            "javascript": [".js", ".mjs"],
            "typescript": [".ts", ".tsx"],
            "java": [".java"],
            "go": [".go"],
            "rust": [".rs"],
            "c": [".c", ".h"],
            "cpp": [".cpp", ".cxx", ".cc", ".hpp", ".hxx", ".c++", ".h++"],
            "php": [".php", ".phtml", ".php3", ".php4", ".php5", ".phps"],
            "ruby": [".rb", ".rbw"],
            "html": [".html", ".htm"],
            "css": [".css", ".scss", ".sass", ".less"],
            "json": [".json"],
            "yaml": [".yaml", ".yml"],
            "markdown": [".md", ".markdown"],
            "docker": ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"],
            "shell": [".sh", ".bash", ".zsh", ".fish", ".ksh"],
            "sql": [".sql"],
            "xml": [".xml", ".xsl", ".xslt"],
            "toml": [".toml"],
            "ini": [".ini", ".cfg", ".conf"],
        }

        language_counts: Dict[str, int] = {}
        rg_path = self._get_rg_path()
        all_files: List[str] = []
        if rg_path:
            cmd = f"\"{rg_path}\" --files {self.codebase_path / path}"
            result = await self._run_command(cmd)
            if result.success and result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue
                    try:
                        p = Path(line)
                        if not p.is_absolute():
                            p = (self.codebase_path / p).resolve()
                        rel = str(p.relative_to(self.codebase_path))
                        all_files.append(rel)
                    except Exception:
                        continue
        else:
            all_files = await self.list_directory(path, recursive=True, max_depth=4)

        for file_path in all_files:
            ext = Path(file_path).suffix.lower()
            matched = False
            for language, lang_exts in extensions.items():
                if ext in lang_exts:
                    language_counts[language] = language_counts.get(language, 0) + 1
                    matched = True
                    break
            if not matched and ext:
                key = f"other_{ext.lstrip('.')}"
                language_counts[key] = language_counts.get(key, 0) + 1
        return language_counts



#uv run python -m qrooper.tools.filesystem_utils
if __name__ == "__main__":
    import asyncio
    import json

    async def _run_demo() -> None:
        # Codebase root to test against (absolute path from repo root)
        current_file = Path(__file__).resolve()
        # Walk up until we find the repository root that contains 'packages'
        repo_root = current_file
        for _ in range(8):
            if (repo_root / 'packages').is_dir():
                break
            repo_root = repo_root.parent
        # Try different package paths, with preference for current one
        possible_paths = [
            'packages/qrooper/src/qrooper',  # Current package
            'packages/eva/src/eva',          # Alternative
        ]

        cb_path = None
        for rel_path in possible_paths:
            candidate = (repo_root / rel_path).resolve()
            if candidate.exists():
                cb_path = candidate
                break

        if not cb_path:
            raise SystemExit(f"Demo error: no valid codebase path found. Tried: {', '.join(possible_paths)}")
        utils = FilesystemUtils(cb_path)

        print(f"Codebase root: {utils.codebase_path}")

        # 1) List directory (non-recursive)
        entries = await utils.list_directory('.', recursive=False, show_hidden=False)
        print(f"Top-level entries (first 10): {entries[:10]}")

        # 2) Detect languages
        langs = await utils.detect_languages('.')
        print("Languages detected:")
        print(json.dumps(langs, indent=2))

        # 3) Find files by pattern
        py_files = await utils.find_files('*.py', '.')
        print(f"Python files found: {len(py_files)} (first 5): {py_files[:5]}")

        # 4) Read a file
        target_file = py_files[0] if py_files else '__init__.py'
        file_result = await utils.read_file(target_file, start_line=1, end_line=50)
        print(f"Read file: {target_file} -> lines={file_result.lines}, error={file_result.error}")

        # 5) Grep for a simple pattern across Python files
        grep_res = await utils.grep(r"class\s+\w+", path='.', file_patterns=["*.py"], max_results=10)
        print(f"Grep: total_matches={grep_res.total_matches}, files_searched={grep_res.files_searched}")
        for m in grep_res.matches[:3]:
            print(f"  {m.get('match', '')[:120]}")

        # 6) Build a shallow file tree
        tree = await utils.get_file_tree('.', max_depth=2)
        root_children = list(tree.get('_children', {}).keys())
        print(f"File tree root entries (up to depth 2): {root_children[:10]}")

    asyncio.run(_run_demo())
