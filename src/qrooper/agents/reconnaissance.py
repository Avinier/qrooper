"""
Reconnaissance Agent - Adaptive Layered Codebase Analysis

Implements a 4-phase reconnaissance strategy:
1. Lightning Scan - Quick orientation without reading files
2. Structural Mapping - Architecture and conventions
3. Query-Specific Deep Dive - Focused exploration
4. Synthesis & Context Building - Ranked insights

Follows the principle: "lazy and smart" - minimum effort for maximum insight.
"""

import os
import re
import json
import asyncio
import subprocess
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import time

from ..schemas import ReconnaissanceResult, CodebaseFingerprint, ExplorationPlan
from ..tools.filesystem_utils import FilesystemUtils, oai_compatible_filesystemtools
from ..tools.ast_parsing import oai_compatible_asttools

# Import LLM provider
from ..agents.llm_calls import QrooperLLM

# Import context manager
from .context_manager import ContextManagerAgent

# Import prompts
from ..prompts import RECONNAISSANCE_AGENT_PROMPT, RECONNAISSANCE_PLANNING_PROMPT, RECONNAISSANCE_SYNTHESIS_PROMPT



class FileCache:
    """High-performance file cache for reconnaissance operations"""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self._files_by_name = {}
        self._files_by_extension = {}
        self._directories = set()
        self._files_in_subdir = {}
        self._initialized = False

    def _initialize(self):
        """Build cache indexes on first access"""
        if self._initialized:
            return

        file_count = 0
        dir_count = 0
        try:
            for file_path in self.root_path.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(self.root_path)
                    name = file_path.name
                    ext = file_path.suffix.lower()
                    parent = str(rel_path.parent)

                    # Index by name
                    if name not in self._files_by_name:
                        self._files_by_name[name] = []
                    self._files_by_name[name].append(str(rel_path))

                    # Index by extension
                    if ext:
                        if ext not in self._files_by_extension:
                            self._files_by_extension[ext] = []
                        self._files_by_extension[ext].append(str(rel_path))

                    # Index by parent directory
                    if parent not in self._files_in_subdir:
                        self._files_in_subdir[parent] = []
                    self._files_in_subdir[parent].append(str(rel_path))

                    file_count += 1

                elif file_path.is_dir():
                    rel_path = str(file_path.relative_to(self.root_path))
                    if not rel_path.startswith('.'):
                        self._directories.add(rel_path)
                        dir_count += 1

            self._initialized = True

        except Exception as e:
            # Only log major errors
            print(f"❌ FileCache indexing failed: {str(e)[:100]}")
            raise

    def get_files_by_name(self, name: str) -> List[str]:
        """Get files by exact name match"""
        self._initialize()
        return self._files_by_name.get(name, [])

    def get_files_by_extension(self, ext: str) -> List[str]:
        """Get files by extension (with or without leading dot)"""
        self._initialize()
        if not ext.startswith('.'):
            ext = '.' + ext
        return self._files_by_extension.get(ext.lower(), [])

    def get_files_by_pattern(self, pattern: str) -> List[str]:
        """Get files matching a glob pattern"""
        self._initialize()
        import fnmatch
        matches = []
        for file_list in self._files_by_name.values():
            for file_path in file_list:
                if fnmatch.fnmatch(file_path, pattern):
                    matches.append(file_path)
        return matches[:20]  # Limit results

    def get_directories(self) -> List[str]:
        """Get all non-hidden directories"""
        self._initialize()
        return list(self._directories)

    def get_files_in_subdir(self, subdir: str) -> List[str]:
        """Get all files in a subdirectory"""
        self._initialize()
        return self._files_in_subdir.get(subdir, [])

    def has_file(self, file_path: str) -> bool:
        """Check if file exists in cache"""
        self._initialize()
        return self.root_path.joinpath(file_path).exists()

    def count_files(self) -> int:
        """Count total files in cache"""
        self._initialize()
        total = 0
        for file_list in self._files_by_name.values():
            total += len(file_list)
        return total


@dataclass
class ReconConfig:
    """Configuration for reconnaissance operations"""
    max_files_analyzed: int = 50
    max_directory_depth: int = 3
    timeout_per_phase: int = 120  # seconds
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*.pyc", "__pycache__", "node_modules", ".git", "dist",
        "build", ".pytest_cache", ".venv", "venv", "env"
    ])
    include_patterns: List[str] = field(default_factory=lambda: [
        "*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.go", "*.java",
        "*.rs", "*.rb", "*.php", "*.cpp", "*.h", "*.cs"
    ])
    max_file_size: int = 1024 * 1024  # 1MB


@dataclass
class ReconPhase:
    """Tracks reconnaissance phase execution"""
    name: str
    duration: float
    findings: Dict[str, Any]
    artifacts: List[str] = field(default_factory=list)


class LightningScanner:
    """Phase 1.1: Quick codebase fingerprinting without reading file contents"""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.fs_utils = FilesystemUtils(self.root_path)
        self.cache = FileCache(self.root_path)
        # Simple logger for critical errors only
        self.logger = logging.getLogger("LightningScanner")

    async def scan(self) -> CodebaseFingerprint:
        """Perform ultra-fast codebase assessment - Optimized with parallel execution"""
        start = time.time()

        # Project fingerprinting
        fingerprint = CodebaseFingerprint(
            path=str(self.root_path),
            name=self.root_path.name,
            timestamp=datetime.now().isoformat()
        )

        # Run independent scans in parallel for maximum performance
        try:
            tasks = await asyncio.gather(
                self._detect_languages_optimized(),
                self._detect_frameworks_optimized(),
                self._detect_build_tools_optimized(),
                self._detect_dependencies_optimized(),
                self._estimate_size_optimized(),
                self._find_entry_points_optimized(),
                return_exceptions=True
            )

            # Extract results, handling any exceptions gracefully
            if isinstance(tasks[0], Exception):
                print(f"⚠️ Language detection failed")
                fingerprint.languages = {}
            else:
                fingerprint.languages = tasks[0]

            if isinstance(tasks[1], Exception):
                print(f"⚠️ Framework detection failed")
                fingerprint.frameworks = []
            else:
                fingerprint.frameworks = tasks[1]

            if isinstance(tasks[2], Exception):
                print(f"⚠️ Build tools detection failed")
                fingerprint.build_tools = []
            else:
                fingerprint.build_tools = tasks[2]

            if isinstance(tasks[3], Exception):
                print(f"⚠️ Dependencies detection failed")
                fingerprint.dependencies = {}
            else:
                fingerprint.dependencies = tasks[3]

            if isinstance(tasks[4], Exception):
                fingerprint.size_estimate = 'Unknown'
            else:
                fingerprint.size_estimate = tasks[4]

            if isinstance(tasks[5], Exception):
                print(f"⚠️ Entry point detection failed")
                fingerprint.entry_points = []
            else:
                fingerprint.entry_points = tasks[5]

        except Exception as e:
            print(f"❌ Scan execution failed: {str(e)[:100]}")
            raise

        # Use cached file count and structure
        try:
            fingerprint.total_files = await self._count_files()
            fingerprint.top_level_structure = await self._get_top_level_structure()
            self.logger.debug(f"Total files: {fingerprint.total_files}")
        except Exception as e:
            self.logger.warning(f"File counting failed: {str(e)}")
            fingerprint.total_files = 0
            fingerprint.top_level_structure = []

        fingerprint.scan_time = time.time() - start
        self.logger.info(
            f"✅ Lightning Scan completed in {fingerprint.scan_time:.2f}s - "
            f"Found {len(fingerprint.languages)} languages, {len(fingerprint.frameworks)} frameworks"
        )

        return fingerprint

    async def _detect_languages_optimized(self) -> Dict[str, int]:
        """Count files by extension using high-performance cache - Optimized version"""
        self.logger.debug("Starting optimized language detection")

        # Map extensions to language names
        language_mapping = {
            'py': 'Python',
            'js': 'JavaScript',
            'ts': 'TypeScript',
            'java': 'Java',
            'go': 'Go',
            'rs': 'Rust',
            'c': 'C',
            'cpp': 'C++',
            'cc': 'C++',
            'cxx': 'C++',
            'h': 'C/C++ Header',
            'hpp': 'C++ Header',
            'php': 'PHP',
            'rb': 'Ruby',
            'html': 'HTML',
            'htm': 'HTML',
            'css': 'CSS',
            'json': 'JSON',
            'yaml': 'YAML',
            'yml': 'YAML',
            'md': 'Markdown',
            'Dockerfile': 'Docker',
            'sh': 'Shell',
            'bash': 'Shell',
            'zsh': 'Zsh',
            'fish': 'Fish',
            'ps1': 'PowerShell',
            'sql': 'SQL',
            'xml': 'XML',
            'toml': 'TOML',
            'ini': 'INI',
            'pyi': 'Python (Stubs)',
            'pyx': 'Cython',
            'jsx': 'React/JavaScript',
            'tsx': 'React/TypeScript',
            'mjs': 'JavaScript (ESM)',
            'cjs': 'JavaScript (CommonJS)',
            'kt': 'Kotlin',
            'scala': 'Scala',
            'cs': 'C#',
            'dart': 'Dart',
            'zig': 'Zig',
            'nim': 'Nim',
            'pl': 'Perl',
            'swift': 'Swift',
            'r': 'R',
            'jl': 'Julia',
            'ipynb': 'Jupyter Notebook',
            'vue': 'Vue.js',
            'svelte': 'Svelte',
            'astro': 'Astro',
            'mdx': 'MDX',
            'scss': 'Sass',
            'sass': 'Sass',
            'less': 'Less',
            'tf': 'Terraform',
            'hcl': 'HCL',
            'prisma': 'Prisma Schema',
            'graphql': 'GraphQL',
            'gql': 'GraphQL',
            'lua': 'Lua',
            'ex': 'Elixir',
            'exs': 'Elixir',
            'erl': 'Erlang',
            'hs': 'Haskell',
            'ml': 'OCaml'
        }

        try:
            # Count files by extension using cache
            extension_counts = {}
            detected_count = 0

            # Get all common extensions from cache
            for ext, lang_name in language_mapping.items():
                files = self.cache.get_files_by_extension(ext)
                if files:
                    extension_counts[lang_name] = len(files)
                    detected_count += 1
                    self.logger.debug(f"  {lang_name}: {len(files)} files")

            # Sort by count and log top languages
            sorted_langs = sorted(extension_counts.items(), key=lambda x: x[1], reverse=True)
            top_langs = sorted_langs[:5]
            if top_langs:
                self.logger.info(f"Top languages: {', '.join([f'{lang} ({count})' for lang, count in top_langs])}")

            self.logger.debug(f"Language detection complete: {detected_count} languages found")
            return extension_counts

        except Exception as e:
            error_msg = f"Language detection failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}\n{traceback.format_exc()}")
            return {}

    async def _detect_languages(self) -> Dict[str, int]:
        """Count files by extension without reading content - Comprehensive tech stack coverage"""
        # Use the enhanced language detection from FilesystemUtils
        # but with our extended language mapping
        fs_lang_counts = await self.fs_utils.detect_languages('.')

        # Extended mapping from FilesystemUtils to our comprehensive language names
        language_mapping = {
            'python': 'Python',
            'javascript': 'JavaScript',
            'typescript': 'TypeScript',
            'java': 'Java',
            'go': 'Go',
            'rust': 'Rust',
            'c': 'C',
            'cpp': 'C++',
            'php': 'PHP',
            'ruby': 'Ruby',
            'html': 'HTML',
            'css': 'CSS',
            'json': 'JSON',
            'yaml': 'YAML',
            'markdown': 'Markdown',
            'docker': 'Docker',
            'shell': 'Shell',
            'sql': 'SQL',
            'xml': 'XML',
            'toml': 'TOML',
            'ini': 'INI'
        }

        # Additional file extensions to check that FilesystemUtils might not include
        additional_extensions = {
            '.pyi': 'Python (Stubs)',
            '.pyx': 'Cython',
            '.pyd': 'Python',
            '.jsx': 'React/JavaScript',
            '.tsx': 'React/TypeScript',
            '.mjs': 'JavaScript (ESM)',
            '.cjs': 'JavaScript (CommonJS)',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.cs': 'C#',
            '.vb': 'VB.NET',
            '.fs': 'F#',
            '.dart': 'Dart',
            '.cc': 'C++',
            '.cxx': 'C++',
            '.h': 'C/C++ Header',
            '.hpp': 'C++ Header',
            '.zig': 'Zig',
            '.nim': 'Nim',
            '.pl': 'Perl',
            '.raku': 'Raku',
            '.swift': 'Swift',
            '.obj-c': 'Objective-C',
            '.objc': 'Objective-C',
            '.r': 'R',
            '.R': 'R',
            '.jl': 'Julia',
            '.m': 'MATLAB/Octave',
            '.ipynb': 'Jupyter Notebook',
            '.xsl': 'XSLT',
            '.vue': 'Vue.js',
            '.svelte': 'Svelte',
            '.astro': 'Astro',
            '.mdx': 'MDX',
            '.scss': 'Sass',
            '.sass': 'Sass',
            '.less': 'Less',
            '.styl': 'Stylus',
            '.zsh': 'Zsh',
            '.fish': 'Fish',
            '.ps1': 'PowerShell',
            '.tf': 'Terraform',
            '.hcl': 'HCL',
            '.prisma': 'Prisma Schema',
            '.graphql': 'GraphQL',
            '.gql': 'GraphQL',
            '.lua': 'Lua',
            '.ex': 'Elixir',
            '.exs': 'Elixir',
            '.erl': 'Erlang',
            '.hs': 'Haskell',
            '.ml': 'OCaml'
        }

        # Map FilesystemUtils results to our naming convention
        lang_counts = {}
        for fs_lang, count in fs_lang_counts.items():
            # Skip other_* entries
            if fs_lang.startswith('other_'):
                # Check if it's in our additional extensions
                ext = '.' + fs_lang.replace('other_', '')
                if ext in additional_extensions:
                    lang = additional_extensions[ext]
                    lang_counts[lang] = lang_counts.get(lang, 0) + count
            elif fs_lang in language_mapping:
                lang = language_mapping[fs_lang]
                lang_counts[lang] = lang_counts.get(lang, 0) + count

        # For additional extensions, do a targeted search if needed
        for ext, lang in additional_extensions.items():
            if lang not in lang_counts:  # Only search if not already found
                files = await self.fs_utils.find_files(f"*{ext}", '.', file_type="name")
                if files:
                    lang_counts[lang] = len(files)

        return lang_counts

    async def _detect_frameworks(self) -> List[str]:
        """Detect frameworks from config files and directory names - Comprehensive coverage"""
        frameworks = []

        # Python Frameworks & Libraries
        python_indicators = {
            'requirements.txt': ['Python/Basic'],
            'Pipfile': ['Python/Pipenv'],
            'poetry.lock': ['Python/Poetry'],
            'pyproject.toml': ['Python/Poetry/Modern'],
            'conda.yaml': ['Python/Conda'],
            'environment.yml': ['Python/Conda'],
            'setup.py': ['Python/Setuptools'],
            'setup.cfg': ['Python/Setuptools'],
            # Django indicators
            'manage.py': ['Django'],
            'wsgi.py': ['Django/WSGI'],
            'asgi.py': ['Django/ASGI'],
            'django.conf': ['Django'],
            # Flask indicators
            'app.py': ['Flask (Possible)'],
            'Flaskfile': ['Flask'],
            # FastAPI indicators
            'main.py': ['FastAPI (Possible)'],
            'api/__init__.py': ['FastAPI (Possible)'],
            # SQLAlchemy/Databases
            'alembic.ini': ['SQLAlchemy/Alembic'],
            'migrations/': ['Database/Migrations'],
            # Other Python frameworks
            'pyramid.ini': ['Pyramid'],
            'tornado.web': ['Tornado'],
            'sanic/__init__.py': ['Sanic'],
            'aiohttp/web.py': ['Aiohttp'],
            'streamlit_app.py': ['Streamlit'],
            'gradio_app.py': ['Gradio'],
            'langchain_helper.py': ['LangChain'],
            'huggingface_hub.py': ['HuggingFace']
        }

        # JavaScript/TypeScript/React Frameworks
        js_indicators = {
            'package.json': ['Node.js'],
            'yarn.lock': ['Yarn'],
            'pnpm-lock.yaml': ['pnpm'],
            'package-lock.json': ['npm'],
            'bun.lockb': ['Bun'],
            # React ecosystem
            'next.config.js': ['Next.js'],
            'next.config.mjs': ['Next.js'],
            'next.config.ts': ['Next.js'],
            'nuxt.config.js': ['Nuxt.js'],
            'nuxt.config.ts': ['Nuxt.js'],
            'gatsby-config.js': ['Gatsby'],
            'gatsby-config.ts': ['Gatsby'],
            'remix.config.js': ['Remix'],
            'svelte.config.js': ['SvelteKit'],
            'astro.config.mjs': ['Astro'],
            'vite.config.js': ['Vite'],
            'vite.config.ts': ['Vite'],
            # Vue ecosystem
            'vue.config.js': ['Vue CLI'],
            'quasar.config.js': ['Quasar'],
            # Angular ecosystem
            'angular.json': ['Angular'],
            'nx.json': ['Nx'],
            'nest-cli.json': ['NestJS'],
            # Other JS frameworks
            'electron-builder.json': ['Electron'],
            'tauri.conf.json': ['Tauri'],
            'deno.json': ['Deno'],
            'webpack.mix.js': ['Laravel Mix'],
            'rollup.config.js': ['Rollup'],
            'parcel.config.js': ['Parcel'],
            'esbuild.config.js': ['esbuild'],
            'turbo.json': ['Turborepo'],
            'rush.json': ['Rush']
        }

        # Enterprise Java Frameworks
        java_indicators = {
            'pom.xml': ['Java/Maven'],
            'build.gradle': ['Java/Gradle'],
            'build.gradle.kts': ['Java/Gradle/Kotlin'],
            'gradle.properties': ['Gradle'],
            'settings.gradle': ['Gradle'],
            'maven.config': ['Maven'],
            # Spring ecosystem
            'application.properties': ['Spring Boot'],
            'application.yml': ['Spring Boot'],
            'application.yaml': ['Spring Boot'],
            'bootstrap.properties': ['Spring'],
            'spring.factories': ['Spring Framework'],
            # Jakarta/Java EE
            'persistence.xml': ['JPA/Jakarta EE'],
            'faces-config.xml': ['JSF'],
            'web.xml': ['Java EE/Web'],
            'ejb-jar.xml': ['EJB'],
            # Quarkus
            'application.properties': ['Quarkus (Possible)'],
            'quarkus.properties': ['Quarkus'],
            # Micronaut
            'application.yml': ['Micronaut (Possible)'],
            'micronaut-cli.yml': ['Micronaut'],
            # Other Java frameworks
            'play.conf': ['Play Framework'],
            'akka.conf': ['Akka'],
            'vertx.json': ['Vert.x'],
            'dropwizard.yaml': ['Dropwizard'],
            'config.groovy': ['Grails'],
            'BuildConfig.groovy': ['Grails']
        }

        # .NET Ecosystem
        dotnet_indicators = {
            '*.csproj': ['.NET/Core'],
            '*.vbproj': ['.NET/VB'],
            '*.fsproj': ['.NET/F#'],
            'project.json': ['.NET/Core'],
            'packages.config': ['.NET/NuGet'],
            'appsettings.json': ['.NET/Core'],
            'appsettings.Development.json': ['.NET/Core'],
            'Startup.cs': ['.NET/Core'],
            'Program.cs': ['.NET/Core'],
            'global.json': ['.NET CLI'],
            'Directory.Build.props': ['.NET/MSBuild'],
            'nuget.config': ['NuGet'],
            # ASP.NET specific
            'web.config': ['ASP.NET'],
            'bundleconfig.json': ['ASP.NET/Bundling']
        }

        # Go Ecosystem
        go_indicators = {
            'go.mod': ['Go Modules'],
            'go.sum': ['Go Modules'],
            'Gopkg.toml': ['Go/dep'],
            'Gopkg.lock': ['Go/dep'],
            'glide.yaml': ['Go/Glide'],
            'vendor.json': ['Go/govend'],
            # Go frameworks
            'main.go': ['Go (Possible)'],
            'go.uber.org/zap': ['Go/Uber Zap'],
            'gin-gonic/gin': ['Go/Gin'],
            'echo': ['Go/Echo'],
            'fiber': ['Go/Fiber'],
            'chi': ['Go/Chi'],
            'mux': ['Go/gorilla/mux'],
            'kit': ['Go/kit'],
            'micro': ['Go/Micro']
        }

        # Rust Ecosystem
        rust_indicators = {
            'Cargo.toml': ['Rust/Cargo'],
            'Cargo.lock': ['Rust/Cargo'],
            'rust-toolchain': ['Rust'],
            'rust-toolchain.toml': ['Rust'],
            # Rust frameworks
            'axum': ['Rust/Axum'],
            'actix-web': ['Rust/Actix'],
            'rocket': ['Rust/Rocket'],
            'tokio': ['Rust/Tokio'],
            'warp': ['Rust/Warp'],
            'serde': ['Rust/Serde'],
            'clap': ['Rust/Clap']
        }

        # Ruby Ecosystem
        ruby_indicators = {
            'Gemfile': ['Ruby/Bundler'],
            'Gemfile.lock': ['Ruby/Bundler'],
            'gems.rb': ['Ruby/Bundler'],
            'gems.locked': ['Ruby/Bundler'],
            'Rakefile': ['Ruby/Rake'],
            'config.ru': ['Rack'],
            # Ruby frameworks
            'config/application.rb': ['Ruby on Rails'],
            'config/environment.rb': ['Ruby on Rails'],
            'config/routes.rb': ['Ruby on Rails'],
            'app/controllers': ['Ruby on Rails'],
            'app/models': ['Ruby on Rails'],
            'app/views': ['Ruby on Rails'],
            'sinatra/base.rb': ['Sinatra'],
            'puma.rb': ['Puma'],
            'sidekiq.yml': ['Sidekiq']
        }

        # PHP Ecosystem
        php_indicators = {
            'composer.json': ['PHP/Composer'],
            'composer.lock': ['PHP/Composer'],
            # PHP frameworks
            'artisan': ['Laravel'],
            'app/Http/Controllers': ['Laravel'],
            'app/Models': ['Laravel'],
            'resources/views': ['Laravel'],
            'config/app.php': ['Laravel'],
            'symfony.lock': ['Symfony'],
            'src/Controller': ['Symfony'],
            'bin/console': ['Symfony'],
            'config/bundles.php': ['Symfony'],
            'index.php': ['WordPress (Possible)'],
            'wp-config.php': ['WordPress'],
            'wp-content': ['WordPress'],
            'Magento': ['Magento'],
            'drush': ['Drupal'],
            'sites/default': ['Drupal'],
            'craft': ['Craft CMS'],
            'system/src': ['Craft CMS']
        }

        # Database & Infrastructure
        infra_indicators = {
            'docker-compose.yml': ['Docker Compose'],
            'docker-compose.yaml': ['Docker Compose'],
            'docker-compose.override.yml': ['Docker Compose'],
            'Dockerfile': ['Docker'],
            'Dockerfile.dev': ['Docker'],
            'Dockerfile.prod': ['Docker'],
            'k8s/': ['Kubernetes'],
            'kubernetes/': ['Kubernetes'],
            'helm/': ['Helm'],
            'Chart.yaml': ['Helm'],
            'values.yaml': ['Helm'],
            'values-dev.yaml': ['Helm'],
            'Vagrantfile': ['Vagrant'],
            'Terraform': ['Terraform'],
            'main.tf': ['Terraform'],
            'terraform.tfvars': ['Terraform'],
            '.terraform': ['Terraform'],
            'ansible.cfg': ['Ansible'],
            'playbook.yml': ['Ansible'],
            'requirements.yml': ['Ansible'],
            'Jenkinsfile': ['Jenkins'],
            '.gitlab-ci.yml': ['GitLab CI'],
            '.github/workflows': ['GitHub Actions'],
            'azure-pipelines.yml': ['Azure Pipelines'],
            'bitbucket-pipelines.yml': ['Bitbucket Pipelines'],
            'circle.yml': ['CircleCI'],
            '.circleci': ['CircleCI'],
            'prow.yaml': ['Prow'],
            'tekton/': ['Tekton'],
            'argo/': ['Argo CD'],
            'fleet.yaml': ['Fleet']
        }

        # Database specific
        db_indicators = {
            'schema.sql': ['SQL Schema'],
            'migrations/': ['Database/Migrations'],
            'seeds/': ['Database/Seeds'],
            'prisma/schema.prisma': ['Prisma ORM'],
            'drizzle.config.ts': ['Drizzle ORM'],
            'typeorm.config.ts': ['TypeORM'],
            'sequelize.config.js': ['Sequelize'],
            'knexfile.js': ['Knex.js'],
            'alembic.ini': ['Alembic (SQLAlchemy)'],
            'models/index.js': ['Sequelize/Node'],
            'database.yml': ['Rails DB'],
            'my.cnf': ['MySQL'],
            'postgresql.conf': ['PostgreSQL'],
            'redis.conf': ['Redis'],
            'mongod.conf': ['MongoDB'],
            'neo4j.conf': ['Neo4j'],
            'cassandra.yaml': ['Cassandra'],
            'elasticsearch.yml': ['Elasticsearch'],
            'solr/': ['Apache Solr'],
            'influxdb.conf': ['InfluxDB']
        }

        # Frontend & Build Tools
        frontend_indicators = {
            'tsconfig.json': ['TypeScript'],
            'tsconfig.build.json': ['TypeScript'],
            'tsconfig.app.json': ['TypeScript'],
            'jsconfig.json': ['JavaScript/Config'],
            'babel.config.js': ['Babel'],
            'babel.config.json': ['Babel'],
            '.babelrc': ['Babel'],
            '.babelrc.js': ['Babel'],
            'postcss.config.js': ['PostCSS'],
            'postcss.config.json': ['PostCSS'],
            'tailwind.config.js': ['Tailwind CSS'],
            'tailwind.config.ts': ['Tailwind CSS'],
            'bootstrap.config.js': ['Bootstrap'],
            'bulma.config.js': ['Bulma'],
            'foundation.config.js': ['Foundation'],
            'material-ui.config.js': ['Material-UI'],
            'antd.config.js': ['Ant Design'],
            'chakra.config.js': ['Chakra UI'],
            'mui.config.js': ['MUI'],
            'styled.d.ts': ['Styled Components'],
            'emotion.d.ts': ['Emotion'],
            ' stitches.config.ts': ['Stitches'],
            'windi.config.ts': ['Windi CSS'],
            'uno.config.ts': ['UnoCSS']
        }

        # Testing frameworks
        test_indicators = {
            'jest.config.js': ['Jest'],
            'jest.config.json': ['Jest'],
            'jest.config.ts': ['Jest'],
            'jest.config.mjs': ['Jest'],
            'vitest.config.ts': ['Vitest'],
            'vitest.config.js': ['Vitest'],
            'mocha.opts': ['Mocha'],
            '.mocharc.json': ['Mocha'],
            '.mocharc.js': ['Mocha'],
            'karma.conf.js': ['Karma'],
            'jasmine.json': ['Jasmine'],
            'cypress.json': ['Cypress'],
            'cypress.config.js': ['Cypress'],
            'playwright.config.js': ['Playwright'],
            'playwright.config.ts': ['Playwright'],
            'testcafe.config.js': ['TestCafe'],
            'wdio.conf.js': ['WebdriverIO'],
            'nightwatch.conf.js': ['Nightwatch'],
            'protractor.conf.js': ['Protractor'],
            'robotframework': ['Robot Framework'],
            'behave.ini': ['Behave (Python)'],
            'pytest.ini': ['Pytest'],
            'pyproject.toml': ['Pytest (Possible)'],
            'tox.ini': ['Tox'],
            '.coveragerc': ['Coverage.py'],
            'nyc.config.js': ['NYC/Istanbul'],
            'coverage.xml': ['Coverage Reports']
        }

        # Combine all indicators
        all_indicators = {
            **python_indicators,
            **js_indicators,
            **java_indicators,
            **dotnet_indicators,
            **go_indicators,
            **rust_indicators,
            **ruby_indicators,
            **php_indicators,
            **infra_indicators,
            **db_indicators,
            **frontend_indicators,
            **test_indicators
        }

        # Check for each indicator
        for config_file, detected_frameworks in all_indicators.items():
            # Handle directory patterns ending with /
            if config_file.endswith('/'):
                config_dir = self.root_path / config_file.rstrip('/')
                if config_dir.exists() and config_dir.is_dir():
                    frameworks.extend(detected_frameworks)
            # Handle glob patterns
            elif '*' in config_file:
                for file_path in self.root_path.glob(config_file):
                    if file_path.exists():
                        frameworks.extend(detected_frameworks)
                        break
            # Handle exact file matches
            else:
                if (self.root_path / config_file).exists():
                    frameworks.extend(detected_frameworks)

        # Categorize and clean up frameworks
        categorized = {
            'web_frameworks': [],
            'frontend': [],
            'backend': [],
            'mobile': [],
            'desktop': [],
            'devops': [],
            'testing': [],
            'databases': [],
            'package_managers': [],
            'other': []
        }

        for fw in frameworks:
            fw_lower = fw.lower()
            if any(x in fw_lower for x in ['django', 'flask', 'fastapi', 'express', 'spring', 'rails', 'laravel', 'symfony', 'nestjs']):
                categorized['web_frameworks'].append(fw)
            elif any(x in fw_lower for x in ['react', 'vue', 'angular', 'svelte', 'next.js', 'nuxt.js']):
                categorized['frontend'].append(fw)
            elif any(x in fw_lower for x in ['npm', 'yarn', 'pnpm', 'pip', 'poetry', 'maven', 'gradle', 'cargo']):
                categorized['package_managers'].append(fw)
            elif any(x in fw_lower for x in ['github actions', 'circleci', 'gitlab ci', 'jenkins', 'docker', 'kubernetes', 'terraform']):
                categorized['devops'].append(fw)
            elif any(x in fw_lower for x in ['jest', 'vitest', 'mocha', 'cypress', 'pytest', 'junit']):
                categorized['testing'].append(fw)
            elif any(x in fw_lower for x in ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch']):
                categorized['databases'].append(fw)
            elif any(x in fw_lower for x in ['react native', 'flutter', 'cordova', 'ionic']):
                categorized['mobile'].append(fw)
            elif any(x in fw_lower for x in ['electron', 'tauri', 'desktop']):
                categorized['desktop'].append(fw)
            else:
                categorized['other'].append(fw)

        # Return prioritized list - actual frameworks first, then tools
        final_frameworks = (
            categorized['web_frameworks'][:3] +
            categorized['frontend'][:3] +
            categorized['backend'][:2] +
            categorized['mobile'][:2] +
            categorized['desktop'][:2] +
            categorized['testing'][:2] +
            categorized['databases'][:2] +
            categorized['devops'][:3] +
            categorized['other'][:3]
        )

        return list(dict.fromkeys(final_frameworks))  # Remove duplicates while preserving order

    async def _detect_frameworks_optimized(self) -> List[str]:
        """Detect frameworks using high-performance cache - Optimized version"""
        frameworks = []

        # Check for framework files using cache
        framework_files = {
            'Django': ['manage.py', 'wsgi.py', 'asgi.py'],
            'Flask': ['app.py', 'Flaskfile'],
            'FastAPI': ['main.py'],
            'Poetry': ['pyproject.toml', 'poetry.lock'],
            'Pipenv': ['Pipfile', 'Pipfile.lock'],
            'Conda': ['environment.yml', 'conda.yaml'],
            'Next.js': ['next.config.js', 'next.config.ts', 'next.config.mjs'],
            'Nuxt.js': ['nuxt.config.js', 'nuxt.config.ts'],
            'Gatsby': ['gatsby-config.js', 'gatsby-config.ts'],
            'Angular': ['angular.json', '.angular-cli.json'],
            'NestJS': ['nest-cli.json'],
            'Vue CLI': ['vue.config.js', 'quasar.config.js'],
            'SvelteKit': ['svelte.config.js'],
            'Astro': ['astro.config.mjs'],
            'Vite': ['vite.config.js', 'vite.config.ts'],
            'Webpack': ['webpack.config.js', 'webpack.config.ts'],
            'Rollup': ['rollup.config.js', 'rollup.config.ts'],
            'TypeScript': ['tsconfig.json'],
            'Babel': ['babel.config.js', 'babel.config.json'],
            'Maven': ['pom.xml', 'maven.config'],
            'Gradle': ['build.gradle', 'gradle.properties'],
            'Spring Boot': ['application.properties', 'application.yml'],
            'Go Modules': ['go.mod', 'go.sum'],
            'Cargo': ['Cargo.toml', 'Cargo.lock'],
            'Laravel': ['artisan'],
            'Symfony': ['symfony.lock'],
            'Composer': ['composer.json', 'composer.lock'],
            'Ruby on Rails': ['config/application.rb'],
            'Bundler': ['Gemfile', 'Gemfile.lock'],
            'GitHub Actions': ['.github/workflows'],
            'CircleCI': ['.circleci', 'circle.yml'],
            'GitLab CI': ['.gitlab-ci.yml'],
            'Jenkins': ['Jenkinsfile'],
            'Docker': ['Dockerfile', 'docker-compose.yml', 'docker-compose.yaml'],
            'Jest': ['jest.config.js', 'jest.config.json'],
            'Mocha': ['.mocharc.json', 'mocha.opts'],
            'Pytest': ['pytest.ini'],
            'Cypress': ['cypress.config.js', 'cypress.json'],
            'Playwright': ['playwright.config.js', 'playwright.config.ts'],
            'Vitest': ['vitest.config.js', 'vitest.config.ts']
        }

        # Check for frameworks using cache
        for framework, files in framework_files.items():
            for file_name in files:
                if self.cache.has_file(file_name) or self.cache.get_files_by_name(file_name):
                    frameworks.append(framework)
                    break

        # Check for framework directories
        directories = self.cache.get_directories()
        framework_dirs = {
            '.github': 'GitHub Actions',
            '.circleci': 'CircleCI',
            'node_modules': 'Node.js',
            'app': 'Application Directory',
            'src': 'Source Directory',
            'tests': 'Test Directory',
            '__tests__': 'Jest Tests',
            'spec': 'Spec Tests'
        }

        for dir_name, framework in framework_dirs.items():
            if dir_name in directories:
                frameworks.append(framework)

        return frameworks

    async def _detect_dependencies(self) -> Dict[str, Any]:
        """Detect dependencies and package management files - Enhanced with comprehensive scanning"""
        dependencies = {
            'package_managers': [],
            'dependency_files': [],
            'lock_files': [],
            'python_deps': {},
            'javascript_deps': {},
            'go_deps': {},
            'rust_deps': {},
            'docker_compose': [],
            'databases': [],
            'blockchain_libs': [],
            'build_systems': [],
            'cloud_infra': []
        }

        # Enhanced Python dependencies with more patterns
        python_files = {
            # Standard Python packaging
            'requirements.txt': 'Production dependencies',
            'requirements-dev.txt': 'Development dependencies',
            'requirements-test.txt': 'Test dependencies',
            'requirements/base.txt': 'Base Python dependencies',
            'requirements/local.txt': 'Local development dependencies',
            'requirements/production.txt': 'Production dependencies',
            'pyproject.toml': 'Modern Python packaging (Poetry/PDM/setuptools)',
            'poetry.lock': 'Poetry lock file',
            'pdm.lock': 'PDM lock file',
            'Pipfile': 'Pipenv dependencies',
            'Pipfile.lock': 'Pipenv lock file',
            'setup.py': 'Traditional Python setup',
            'setup.cfg': 'Python configuration',
            'setup.cfg': 'Setuptools configuration',
            'conda.yaml': 'Conda environment',
            'conda.yml': 'Conda environment',
            'environment.yml': 'Conda environment',
            'environment-dev.yml': 'Conda dev environment',
            'Pipfile.lock': 'Pipenv lock file',
            'poetry.lock': 'Poetry lock file',
            'pdm.lock': 'PDM lock file',
            'requirements.in': 'pip-tools dependencies',
            'requirements-dev.in': 'pip-tools dev dependencies',
            # Pipenv
            'Pipfile': 'Pipenv dependencies',
            'Pipfile.lock': 'Pipenv lock file'
        }

        # Enhanced JavaScript/Node.js dependencies
        js_files = {
            'package.json': 'Node.js dependencies',
            'package-lock.json': 'npm lock file',
            'yarn.lock': 'Yarn lock file',
            'pnpm-lock.yaml': 'pnpm lock file',
            'bun.lockb': 'Bun lock file',
            'npm-shrinkwrap.json': 'npm shrinkwrap',
            'tsconfig.json': 'TypeScript configuration',
            'jsconfig.json': 'JavaScript configuration',
            'tsconfig.build.json': 'TypeScript build config',
            'tsconfig.app.json': 'TypeScript app config',
            'vite.config.js': 'Vite build tool',
            'vite.config.ts': 'Vite build tool (TS)',
            'webpack.config.js': 'Webpack bundler',
            'webpack.config.ts': 'Webpack bundler (TS)',
            'rollup.config.js': 'Rollup bundler',
            'babel.config.js': 'Babel transpiler',
            'babel.config.json': 'Babel transpiler config',
            '.babelrc': 'Babel configuration',
            '.babelrc.js': 'Babel configuration',
            'next.config.js': 'Next.js framework',
            'nuxt.config.js': 'Nuxt.js framework',
            'svelte.config.js': 'Svelte/SvelteKit',
            'tailwind.config.js': 'Tailwind CSS',
            'postcss.config.js': 'PostCSS'
        }

        # Enhanced Go dependencies
        go_files = {
            'go.mod': 'Go module definition',
            'go.sum': 'Go module checksums',
            'go.work': 'Go workspace',
            'go.work.sum': 'Go workspace checksums',
            'Gopkg.toml': 'Dep dependency management',
            'Gopkg.lock': 'Dep lock file',
            'glide.yaml': 'Glide dependency management',
            'glide.lock': 'Glide lock file',
            'vendor.json': 'Govend dependency management',
            'go.vendor': 'Go vendor directory'
        }

        # Rust dependencies
        rust_files = {
            'Cargo.toml': 'Rust package definition',
            'Cargo.lock': 'Rust lock file',
            'rust-toolchain': 'Rust toolchain',
            'rust-toolchain.toml': 'Rust toolchain config'
        }

        # Enhanced Docker and containers with comprehensive patterns
        docker_files = {
            # Docker Compose - all variants
            'docker-compose.yml': 'Docker Compose configuration',
            'docker-compose.yaml': 'Docker Compose configuration',
            'docker-compose.override.yml': 'Docker Compose override',
            'docker-compose.prod.yml': 'Production Docker Compose',
            'docker-compose.production.yml': 'Production Docker Compose',
            'docker-compose.dev.yml': 'Development Docker Compose',
            'docker-compose.development.yml': 'Development Docker Compose',
            'docker-compose.test.yml': 'Test Docker Compose',
            'docker-compose.testing.yml': 'Test Docker Compose',
            'docker-compose.ci.yml': 'CI Docker Compose',
            'docker-compose.local.yml': 'Local Docker Compose',
            'docker-compose.localprod.yml': 'Local production Docker Compose',
            'docker-compose.staging.yml': 'Staging Docker Compose',
            # Dockerfiles - all variants
            'Dockerfile': 'Docker image',
            'Dockerfile.prod': 'Production Docker image',
            'Dockerfile.production': 'Production Docker image',
            'Dockerfile.dev': 'Development Docker image',
            'Dockerfile.development': 'Development Docker image',
            'Dockerfile.test': 'Test Docker image',
            'Dockerfile.testing': 'Test Docker image',
            'Dockerfile.ci': 'CI Docker image',
            'Dockerfile.local': 'Local Docker image',
            'Dockerfile.base': 'Base Docker image',
            'Dockerfile.builder': 'Builder Docker image',
            'Dockerfile.runtime': 'Runtime Docker image',
            # Docker configuration
            '.dockerignore': 'Docker ignore file',
            'docker-compose.yml.dist': 'Distribution Docker Compose',
            'docker-compose.yaml.dist': 'Distribution Docker Compose',
            '.docker': 'Docker configuration directory',
            '.dockerenv': 'Docker environment file'
        }

        # Enhanced Database files
        db_files = {
            # SQL databases
            'migrations/': 'Database migrations',
            'migrate/': 'Database migrations',
            'db/migrate/': 'Database migrations',
            'db/migrations/': 'Database migrations',
            'sql/': 'SQL scripts directory',
            'schema.sql': 'Database schema',
            'schema.sql': 'Database schema',
            'database.sql': 'Database schema',
            'init.sql': 'Database initialization',
            'seed.sql': 'Database seed data',
            'seeds/': 'Database seeds',
            'db/seeds/': 'Database seeds',
            'fixtures/': 'Database fixtures',
            # ORMs
            'schema.prisma': 'Prisma schema',
            'prisma/schema.prisma': 'Prisma schema',
            'drizzle.config.ts': 'Drizzle ORM',
            'typeorm.config.ts': 'TypeORM',
            'sequelize.config.js': 'Sequelize ORM',
            'knexfile.js': 'Knex.js query builder',
            'alembic.ini': 'Alembic (SQLAlchemy) migrations',
            'alembic/': 'Alembic migrations directory',
            # Database configurations
            'database.yml': 'Database configuration',
            'database.yaml': 'Database configuration',
            'database.yml': 'Rails database config',
            'mongoid.yml': 'MongoDB configuration',
            'redis.conf': 'Redis configuration',
            'redis/redis.conf': 'Redis configuration',
            'elasticsearch.yml': 'Elasticsearch configuration',
            'neo4j.conf': 'Neo4j configuration',
            'cassandra.yaml': 'Cassandra configuration',
            'influxdb.conf': 'InfluxDB configuration'
        }

        # Cloud infrastructure files
        cloud_files = {
            'terraform/': 'Terraform infrastructure',
            'terragrunt.hcl': 'Terragrunt configuration',
            '.terraform': 'Terraform state',
            'k8s/': 'Kubernetes manifests',
            'kubernetes/': 'Kubernetes manifests',
            'helm/': 'Helm charts',
            'Chart.yaml': 'Helm chart',
            'values.yaml': 'Helm values',
            'values-dev.yaml': 'Helm dev values',
            'values-prod.yaml': 'Helm prod values',
            '.kube/': 'Kubernetes configuration',
            'serverless.yml': 'Serverless framework',
            'serverless.yaml': 'Serverless framework',
            'template.yaml': 'AWS SAM template',
            'aws-cloudformation/': 'CloudFormation templates',
            'pulumi/': 'Pulumi infrastructure',
            'ansible/': 'Ansible playbooks',
            'docker-swarm.yml': 'Docker Swarm',
            'docker-stack.yml': 'Docker Stack'
        }

        # Build system files
        build_files = {
            'Makefile': 'Make build system',
            'makefile': 'Make build system',
            'CMakeLists.txt': 'CMake build system',
            'CMakeCache.txt': 'CMake cache',
            'configure': 'Autoconf script',
            'configure.ac': 'Autoconf configuration',
            'Makefile.am': 'Automake makefile',
            'build.gradle': 'Gradle build',
            'build.gradle.kts': 'Gradle build (Kotlin)',
            'gradle.properties': 'Gradle properties',
            'pom.xml': 'Maven build',
            'build.xml': 'Ant build',
            'Gruntfile.js': 'Grunt task runner',
            'gulpfile.js': 'Gulp task runner',
            'bsconfig.json': 'BuckleScript config',
            'bsconfig.json': 'ReScript config',
            'justfile': 'Just task runner',
            'Taskfile.yml': 'Task task runner',
            'tasks.py': 'Invoke tasks',
            'pyproject.toml': 'Python build (setuptools/poetry)',
            'setup.cfg': 'Python setup config',
            'tox.ini': 'Tox test automation',
            'noxfile.py': 'Nox session management',
            'bazel/': 'Bazel build system',
            'BUILD': 'Bazel build file',
            'WORKSPACE': 'Bazel workspace',
            'build.bzl': 'Bazel starlark',
            'pants.toml': 'Pants build system',
            'BUCK': 'Buck build system',
            'targets.bzl': 'Buck targets'
        }

        # Combine all file types for systematic checking
        all_files = {
            **python_files,
            **js_files,
            **go_files,
            **rust_files,
            **docker_files,
            **db_files,
            **cloud_files,
            **build_files
        }

        # Enhanced scanning with subdirectory support
        async def scan_file_or_dir(file_path: str, description: str, category: str):
            """Scan a file or directory and add to dependencies if found"""
            found_locations = []

            if file_path.endswith('/'):
                # Directory pattern - check root and subdirectories
                dir_names = [file_path.rstrip('/')]
                # Also check common subdirectory locations
                for subdir in ['backend/', 'src/', 'app/', 'server/']:
                    full_dir_path = subdir + file_path.rstrip('/')
                    if (self.root_path / full_dir_path).exists():
                        dir_names.append(full_dir_path)

                for dir_path in dir_names:
                    full_path = self.root_path / dir_path
                    if full_path.exists() and full_path.is_dir():
                        found_locations.append(f"{dir_path} ({description})")

                        # Add category-specific information
                        if category == 'docker':
                            dependencies['docker_compose'].append(dir_path)
                        elif category == 'database':
                            if 'migrations' in file_path:
                                if 'SQL Database' not in dependencies['databases']:
                                    dependencies['databases'].append('SQL Database')
                            elif 'seeds' in file_path:
                                if 'SQL Database with seeds' not in dependencies['databases']:
                                    dependencies['databases'].append('SQL Database with seeds')
                        elif category == 'cloud':
                            if 'Terraform' in description and 'Terraform' not in dependencies['cloud_infra']:
                                dependencies['cloud_infra'].append('Terraform')
                            elif 'Kubernetes' in description and 'Kubernetes' not in dependencies['cloud_infra']:
                                dependencies['cloud_infra'].append('Kubernetes')
                            elif 'Helm' in description and 'Helm' not in dependencies['cloud_infra']:
                                dependencies['cloud_infra'].append('Helm')
            else:
                # File pattern - check root and common subdirectories
                file_locations = [file_path]
                # Check subdirectories for common dependency files
                if any(x in file_path for x in ['requirements', 'package.json', 'go.mod', 'Cargo.toml', 'docker-compose']):
                    for subdir in ['backend/', 'src/', 'app/', 'server/', 'frontend/', 'api/']:
                        full_file_path = subdir + file_path
                        if (self.root_path / full_file_path).exists():
                            file_locations.append(full_file_path)

                for file_loc in file_locations:
                    full_path = self.root_path / file_loc
                    if full_path.exists() and full_path.is_file():
                        found_locations.append(f"{file_loc} ({description})")

                        # Categorize and update package managers
                        if file_loc in python_files:
                            dependencies['python_deps'][file_loc] = description
                            if 'Poetry' in description and 'Poetry' not in dependencies['package_managers']:
                                dependencies['package_managers'].append('Poetry')
                            elif 'Pipenv' in description and 'Pipenv' not in dependencies['package_managers']:
                                dependencies['package_managers'].append('Pipenv')
                            elif 'Conda' in description and 'Conda' not in dependencies['package_managers']:
                                dependencies['package_managers'].append('Conda')
                            elif 'pip' not in dependencies['package_managers']:
                                dependencies['package_managers'].append('pip')

                        elif file_loc in js_files:
                            dependencies['javascript_deps'][file_loc] = description
                            if 'Yarn' in description and 'Yarn' not in dependencies['package_managers']:
                                dependencies['package_managers'].append('Yarn')
                            elif 'pnpm' in description and 'pnpm' not in dependencies['package_managers']:
                                dependencies['package_managers'].append('pnpm')
                            elif 'Bun' in description and 'Bun' not in dependencies['package_managers']:
                                dependencies['package_managers'].append('Bun')
                            elif 'npm' not in dependencies['package_managers']:
                                dependencies['package_managers'].append('npm')

                        elif file_loc in go_files:
                            dependencies['go_deps'][file_loc] = description
                            if 'Go Modules' not in dependencies['package_managers']:
                                dependencies['package_managers'].append('Go Modules')

                        elif file_loc in rust_files:
                            dependencies['rust_deps'][file_loc] = description
                            if 'Cargo' not in dependencies['package_managers']:
                                dependencies['package_managers'].append('Cargo')

                        elif file_loc in docker_files:
                            dependencies['docker_compose'].append(file_loc)

                        elif file_loc in db_files:
                            if 'MongoDB' in description and 'MongoDB' not in dependencies['databases']:
                                dependencies['databases'].append('MongoDB')
                            elif 'Redis' in description and 'Redis' not in dependencies['databases']:
                                dependencies['databases'].append('Redis')
                            elif 'Elasticsearch' in description and 'Elasticsearch' not in dependencies['databases']:
                                dependencies['databases'].append('Elasticsearch')
                            elif 'PostgreSQL' in description and 'PostgreSQL' not in dependencies['databases']:
                                dependencies['databases'].append('PostgreSQL')
                            elif 'MySQL' in description and 'MySQL' not in dependencies['databases']:
                                dependencies['databases'].append('MySQL')
                            elif 'schema' in description.lower() and 'SQL Database' not in dependencies['databases']:
                                dependencies['databases'].append('SQL Database')

                        elif file_loc in cloud_files:
                            if 'Terraform' in description and 'Terraform' not in dependencies['cloud_infra']:
                                dependencies['cloud_infra'].append('Terraform')
                            elif 'Kubernetes' in description and 'Kubernetes' not in dependencies['cloud_infra']:
                                dependencies['cloud_infra'].append('Kubernetes')
                            elif 'Helm' in description and 'Helm' not in dependencies['cloud_infra']:
                                dependencies['cloud_infra'].append('Helm')

                        elif file_loc in build_files:
                            build_name = description.split()[0]  # Get first word as build system name
                            if build_name not in dependencies['build_systems']:
                                dependencies['build_systems'].append(build_name)

            return found_locations

        # Scan all file types
        for file_path, description in all_files.items():
            # Determine category for categorization
            if file_path in docker_files:
                category = 'docker'
            elif file_path in db_files:
                category = 'database'
            elif file_path in cloud_files:
                category = 'cloud'
            elif file_path in build_files:
                category = 'build'
            else:
                category = 'package'

            found = await scan_file_or_dir(file_path, description, category)
            dependencies['dependency_files'].extend(found)

        # Enhanced library detection with multiple methods
        async def detect_libraries():
            """Detect various libraries and frameworks"""
            # Detect Python libraries more comprehensively
            python_libs = {
                'web': ['django', 'flask', 'fastapi', 'starlette', 'aiohttp', 'tornado', 'sanic'],
                'orm': ['sqlalchemy', 'django.db', 'peewee', 'pony', 'databases'],
                'async': ['asyncio', 'aiofiles', 'aioredis', 'aiomysql', 'asyncpg'],
                'testing': ['pytest', 'unittest', 'nose', 'doctest', 'hypothesis'],
                'data': ['pandas', 'numpy', 'scipy', 'matplotlib', 'plotly'],
                'web3': ['web3', 'ethers', 'web3.py', 'brownie', 'ape', 'hardhat', 'alchemy', 'infura'],
                'api': ['rest_framework', 'graphene', 'strawberry', 'tartiflette'],
                'ml': ['tensorflow', 'torch', 'sklearn', 'keras', 'xgboost'],
                'celery': ['celery', 'kombu', 'billiard'],
                'redis': ['redis', 'aioredis', 'hiredis'],
                'http': ['requests', 'httpx', 'aiohttp', 'urllib3']
            }

            detected_libs = []

            # More efficient library detection using ripgrep if available
            for category, libs in python_libs.items():
                try:
                    # Search for import statements
                    lib_pattern = '|'.join(libs)
                    result = await asyncio.create_subprocess_exec(
                        'rg', '--type', 'py', '-l', f'^(import|from)\\s*({"|".join(libs)})',
                        str(self.root_path),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                    stdout, _ = await result.communicate()
                    if stdout.decode() and stdout.decode().strip():
                        detected_libs.extend(libs)
                except Exception:
                    # Fallback to grep if ripgrep not available
                    for lib in libs:
                        try:
                            result = await asyncio.create_subprocess_exec(
                                'grep', '-r', '--include=*.py', '-l', f'^import {lib}\\|^from {lib}',
                                str(self.root_path),
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.DEVNULL
                            )
                            stdout, _ = await result.communicate()
                            if stdout.decode() and stdout.decode().strip():
                                detected_libs.append(lib)
                        except Exception:
                            pass

            return detected_libs

        # Detect libraries
        detected_libs = await detect_libraries()
        if detected_libs:
            dependencies['python_libraries'] = list(set(detected_libs))

        # Remove duplicates from dependency files
        dependencies['dependency_files'] = list(dict.fromkeys(dependencies['dependency_files']))

        # Remove duplicates from package managers
        dependencies['package_managers'] = list(dict.fromkeys(dependencies['package_managers']))

        return dependencies

    async def _detect_dependencies_optimized(self) -> Dict[str, Any]:
        """Detect dependencies using high-performance cache - Optimized version"""
        dependencies = {
            'package_managers': [],
            'dependency_files': [],
            'docker_compose': [],
            'databases': []
        }

        # Key dependency files to check
        dep_files = {
            'requirements.txt': 'pip',
            'pyproject.toml': 'Poetry',
            'poetry.lock': 'Poetry',
            'Pipfile': 'Pipenv',
            'Pipfile.lock': 'Pipenv',
            'environment.yml': 'Conda',
            'package.json': 'npm',
            'package-lock.json': 'npm',
            'yarn.lock': 'Yarn',
            'pnpm-lock.yaml': 'pnpm',
            'go.mod': 'Go Modules',
            'go.sum': 'Go Modules',
            'go.work': 'Go Workspace',
            'Cargo.toml': 'Cargo',
            'Cargo.lock': 'Cargo',
            'docker-compose.yml': 'Docker Compose',
            'docker-compose.yaml': 'Docker Compose',
            'Dockerfile': 'Docker',
            'pom.xml': 'Maven',
            'build.gradle': 'Gradle',
            'composer.json': 'Composer',
            'Gemfile': 'Bundler'
        }

        # Check files using cache
        for file_name, manager in dep_files.items():
            if self.cache.has_file(file_name) or self.cache.get_files_by_name(file_name):
                dependencies['dependency_files'].append(file_name)
                if manager not in dependencies['package_managers']:
                    dependencies['package_managers'].append(manager)

        return dependencies

    async def _estimate_size_optimized(self) -> str:
        """Optimized size estimation using cached file count"""
        self.logger.debug("Estimating project size (optimized)...")
        try:
            # Quick estimation based on file count from cache
            total_files = self.cache.count_files()

            if total_files < 100:
                size_estimate = "Very Small (< 1MB)"
            elif total_files < 1000:
                size_estimate = "Small (1-10MB)"
            elif total_files < 5000:
                size_estimate = "Medium (10-50MB)"
            elif total_files < 20000:
                size_estimate = "Large (50-200MB)"
            else:
                size_estimate = "Very Large (> 200MB)"

            return size_estimate
        except Exception:
            return "Unknown"

    async def _find_entry_points_optimized(self) -> List[str]:
        """Optimized entry point detection using FileCache"""
        entry_points: List[str] = []

        try:
            # Common entry point file patterns
            entry_patterns = [
                'main.py', 'app.py', 'index.js', 'server.js', 'app.js',
                'main.js', 'index.ts', 'server.ts', 'app.ts',
                'main.go', 'main.rs', 'lib.rs', 'mod.rs',
                'index.html', 'index.php', 'main.java',
                'Application.java', 'App.java', 'Program.cs',
                'startup.py', 'run.py', 'start.py', 'boot.py',
                'manage.py', 'wsgi.py'
            ]

            # Check for entry point files using cache (O(1) lookups)
            found_files = 0
            for pattern in entry_patterns:
                matching_files = self.cache.get_files_by_name(pattern)
                for file_path in matching_files:
                    entry_points.append(file_path)
                    found_files += 1
                    if len(entry_points) >= 20:  # Limit results
                        break
                if len(entry_points) >= 20:
                    break

            self.logger.debug(f"  Found {found_files} entry point files")

            # Look for common entry directories using cache
            entry_dirs = ['src', 'app', 'server', 'main', 'cmd', 'bin']
            found_dirs = 0
            for dir_name in entry_dirs:
                if dir_name in self.cache.get_directories():
                    entry_points.append(f"{dir_name}/ (directory)")
                    found_dirs += 1

            self.logger.debug(f"  Found {found_dirs} entry directories")

            # Check for package.json with main field (limited checks)
            package_jsons = self.cache.get_files_by_name('package.json')
            node_entries = 0
            for package_json in list(package_jsons)[:5]:  # Limit checks
                try:
                    package_path = self.root_path / package_json
                    if package_path.exists():
                        content = package_path.read_text(encoding='utf-8', errors='ignore')
                        if '"main"' in content or '"start"' in content:
                            entry_points.append(f"{package_json} (Node.js entry)")
                            node_entries += 1
                except Exception:
                    pass

            # Check for setup.py with entry points (limited checks)
            setup_files = self.cache.get_files_by_name('setup.py')
            for setup_file in list(setup_files)[:3]:  # Limit checks
                try:
                    setup_path = self.root_path / setup_file
                    if setup_path.exists():
                        content = setup_path.read_text(encoding='utf-8', errors='ignore')
                        if 'entry_points' in content or 'py_modules' in content:
                            entry_points.append(f"{setup_file} (Python entry)")
                except Exception:
                    pass

            return entry_points[:20]

        except Exception as e:
            # Only log major errors
            print(f"❌ Entry point detection failed: {str(e)[:100]}")
            return []

    async def _detect_build_tools(self) -> List[str]:
        """Detect build and dependency management tools - Enhanced with comprehensive coverage"""
        tools = []

        # Comprehensive build tools mapping
        build_files = {
            # Traditional build systems
            'Makefile': 'Make',
            'makefile': 'Make',
            'GNUmakefile': 'Make (GNU)',
            'CMakeLists.txt': 'CMake',
            'CMakeCache.txt': 'CMake Cache',
            'configure': 'Autoconf',
            'configure.ac': 'Autoconf',
            'configure.in': 'Autoconf',
            'Makefile.am': 'Automake',
            'Makefile.in': 'Autotools Template',
            'autogen.sh': 'Autogen script',
            'bootstrap': 'Bootstrap script',
            'build.sh': 'Shell build script',
            'build.bat': 'Windows build script',
            'build.ps1': 'PowerShell build script',

            # JavaScript/TypeScript build tools
            'Gruntfile.js': 'Grunt',
            'Gruntfile.coffee': 'Grunt (CoffeeScript)',
            'Gruntfile.ts': 'Grunt (TypeScript)',
            'gulpfile.js': 'Gulp',
            'gulpfile.ts': 'Gulp (TypeScript)',
            'gulpfile.babel.js': 'Gulp (Babel)',
            'webpack.config.js': 'Webpack',
            'webpack.config.ts': 'Webpack (TypeScript)',
            'webpack.config.babel.js': 'Webpack (Babel)',
            'webpack.mix.js': 'Laravel Mix (Webpack)',
            'webpackfile.js': 'Webpack',
            'rollup.config.js': 'Rollup',
            'rollup.config.ts': 'Rollup (TypeScript)',
            'rollup.config.mjs': 'Rollup (ESM)',
            'vite.config.js': 'Vite',
            'vite.config.ts': 'Vite (TypeScript)',
            'vite.config.mjs': 'Vite (ESM)',
            'parcel.config.js': 'Parcel',
            'parcel.config.json': 'Parcel',
            'esbuild.config.js': 'esbuild',
            'esbuild.config.mjs': 'esbuild (ESM)',
            'esbuild.js': 'esbuild',
            'snowpack.config.js': 'Snowpack',
            'snowpack.config.mjs': 'Snowpack (ESM)',
            'turbo.json': 'Turborepo',
            'nx.json': 'Nx',
            'rush.json': 'Rush',
            'lerna.json': 'Lerna',
            'workspace.json': 'Angular CLI',
            'angular.json': 'Angular CLI',
            'nest-cli.json': 'NestJS CLI',
            '.angular-cli.json': 'Angular CLI (Legacy)',

            # TypeScript configuration
            'tsconfig.json': 'TypeScript',
            'tsconfig.build.json': 'TypeScript (Build)',
            'tsconfig.app.json': 'TypeScript (App)',
            'tsconfig.spec.json': 'TypeScript (Spec)',
            'tsconfig.lib.json': 'TypeScript (Library)',
            'tsconfig.base.json': 'TypeScript (Base)',
            'tsconfig.json': 'JavaScript (Config)',
            'tsconfig.eslint.json': 'TypeScript ESLint',

            # Babel configuration
            'babel.config.js': 'Babel',
            'babel.config.json': 'Babel',
            'babel.config.mjs': 'Babel (ESM)',
            '.babelrc': 'Babel',
            '.babelrc.js': 'Babel',
            '.babelrc.json': 'Babel',
            '.babelignore': 'Babel Ignore',

            # CSS/SCSS preprocessors
            'postcss.config.js': 'PostCSS',
            'postcss.config.json': 'PostCSS',
            'postcss.config.ts': 'PostCSS (TS)',
            '.postcssrc': 'PostCSS',
            '.postcssrc.js': 'PostCSS',
            '.postcssrc.json': 'PostCSS',
            'tailwind.config.js': 'Tailwind CSS',
            'tailwind.config.ts': 'Tailwind CSS (TS)',
            'tailwind.config.mjs': 'Tailwind CSS (ESM)',
            'sass.config.js': 'Sass',
            'stylus.config.js': 'Stylus',
            'less.config.js': 'Less',
            'stylelint.config.js': 'Stylelint',
            '.stylelintrc': 'Stylelint',
            '.stylelintrc.json': 'Stylelint',
            '.stylelintrc.js': 'Stylelint',
            '.stylelintrc.yaml': 'Stylelint',
            '.stylelintrc.yml': 'Stylelint',

            # Testing frameworks
            'jest.config.js': 'Jest',
            'jest.config.json': 'Jest',
            'jest.config.ts': 'Jest (TS)',
            'jest.config.mjs': 'Jest (ESM)',
            'jest.config.base.js': 'Jest (Base)',
            'jest.config.common.js': 'Jest (Common)',
            'jest.config.e2e.js': 'Jest (E2E)',
            'jest.setup.js': 'Jest (Setup)',
            'jest.setup.ts': 'Jest (Setup TS)',
            'jest.config.babel.js': 'Jest (Babel)',
            'vitest.config.js': 'Vitest',
            'vitest.config.ts': 'Vitest (TS)',
            'vitest.config.mjs': 'Vitest (ESM)',
            'cypress.config.js': 'Cypress',
            'cypress.config.ts': 'Cypress (TS)',
            'cypress.json': 'Cypress (Legacy)',
            'playwright.config.js': 'Playwright',
            'playwright.config.ts': 'Playwright (TS)',
            'testcafe.config.js': 'TestCafe',
            'wdio.conf.js': 'WebdriverIO',
            'wdio.conf.ts': 'WebdriverIO (TS)',
            'nightwatch.conf.js': 'Nightwatch',
            'nightwatch.conf.ts': 'Nightwatch (TS)',
            'protractor.conf.js': 'Protractor',
            'karma.conf.js': 'Karma',
            'karma.conf.ts': 'Karma (TS)',
            '.karma.conf.js': 'Karma',
            'jasmine.json': 'Jasmine',
            'mocha.opts': 'Mocha',
            '.mocharc.json': 'Mocha',
            '.mocharc.js': 'Mocha',
            '.mocharc.yml': 'Mocha',
            '.mocharc.yaml': 'Mocha',
            'mocha.setup.js': 'Mocha Setup',
            'ava.config.js': 'AVA',
            'ava.config.cjs': 'AVA (CJS)',
            'tap-config.js': 'TAP',
            'tape.config.js': 'Tape',

            # Python testing and quality
            'pytest.ini': 'Pytest',
            'pyproject.toml': 'Pytest (Possible)',
            'tox.ini': 'Tox',
            '.coveragerc': 'Coverage.py',
            'pyproject.toml': 'Coverage.py (Possible)',
            'nose.cfg': 'Nose',
            '.noserc': 'Nose',
            'setup.cfg': 'Python Testing (Possible)',
            'mypy.ini': 'MyPy',
            '.mypy.ini': 'MyPy',
            'pyproject.toml': 'MyPy (Possible)',
            'ruff.toml': 'Ruff',
            '.ruff.toml': 'Ruff',
            'pyproject.toml': 'Ruff (Possible)',
            'black.toml': 'Black',
            'pyproject.toml': 'Black (Possible)',
            'isort.cfg': 'isort',
            '.isort.cfg': 'isort',
            'pyproject.toml': 'isort (Possible)',
            'flake8.cfg': 'Flake8',
            '.flake8': 'Flake8',
            'setup.cfg': 'Flake8 (Possible)',
            'pyproject.toml': 'Flake8 (Possible)',
            'pylintrc': 'Pylint',
            '.pylintrc': 'Pylint',
            'pyproject.toml': 'Pylint (Possible)',
            'bandit.yaml': 'Bandit',
            '.bandit': 'Bandit',

            # Containerization and virtualization
            'Dockerfile': 'Docker',
            'Dockerfile.prod': 'Docker (Production)',
            'Dockerfile.production': 'Docker (Production)',
            'Dockerfile.dev': 'Docker (Development)',
            'Dockerfile.development': 'Docker (Development)',
            'Dockerfile.test': 'Docker (Testing)',
            'Dockerfile.testing': 'Docker (Testing)',
            'Dockerfile.ci': 'Docker (CI)',
            'Dockerfile.local': 'Docker (Local)',
            'Dockerfile.base': 'Docker (Base)',
            'Dockerfile.builder': 'Docker (Builder)',
            'Dockerfile.runtime': 'Docker (Runtime)',
            'docker-compose.yml': 'Docker Compose',
            'docker-compose.yaml': 'Docker Compose',
            'docker-compose.override.yml': 'Docker Compose (Override)',
            'docker-compose.prod.yml': 'Docker Compose (Production)',
            'docker-compose.dev.yml': 'Docker Compose (Development)',
            'docker-compose.test.yml': 'Docker Compose (Testing)',
            'docker-compose.ci.yml': 'Docker Compose (CI)',
            'docker-compose.local.yml': 'Docker Compose (Local)',
            'docker-compose.localprod.yml': 'Docker Compose (Local Production)',
            'docker-compose.staging.yml': 'Docker Compose (Staging)',
            'docker-compose.yml.dist': 'Docker Compose (Distribution)',
            'docker-compose.yaml.dist': 'Docker Compose (Distribution)',
            '.dockerignore': 'Docker Ignore',
            'Containerfile': 'Podman Container',
            'buildah': 'Buildah',
            'Vagrantfile': 'Vagrant',

            # CI/CD platforms
            '.github/workflows': 'GitHub Actions',
            '.github/workflows/': 'GitHub Actions',
            '.gitlab-ci.yml': 'GitLab CI',
            '.gitlab-ci.yaml': 'GitLab CI',
            'gitlab-ci.yml': 'GitLab CI',
            'gitlab-ci.yaml': 'GitLab CI',
            '.gitlab-ci': 'GitLab CI',
            '.travis.yml': 'Travis CI',
            'travis.yml': 'Travis CI',
            'appveyor.yml': 'AppVeyor',
            '.appveyor.yml': 'AppVeyor',
            'circle.yml': 'CircleCI',
            '.circleci': 'CircleCI',
            '.circleci/config.yml': 'CircleCI',
            'circle.yml': 'CircleCI',
            'codeship-services.yml': 'Codeship',
            'codeship-steps.yml': 'Codeship',
            'azure-pipelines.yml': 'Azure Pipelines',
            'azure-pipelines.yaml': 'Azure Pipelines',
            'bitbucket-pipelines.yml': 'Bitbucket Pipelines',
            'bamboo-specs/': 'Bamboo',
            'buildkite.yml': 'Buildkite',
            'buildkite.yaml': 'Buildkite',
            '.buildkite': 'Buildkite',
            'drone.yml': 'Drone CI',
            'drone.yaml': 'Drone CI',
            '.drone.yml': 'Drone CI',
            'semaphore.yml': 'Semaphore',
            'semaphore.yaml': 'Semaphore',
            'snapcraft.yml': 'Snapcraft',
            'snapcraft.yaml': 'Snapcraft',
            'fastlane/Fastfile': 'Fastlane',
            'fastlane/Appfile': 'Fastlane',
            'Jenkinsfile': 'Jenkins',
            'Jenkinsfile.groovy': 'Jenkins',
            'jenkins.yaml': 'Jenkins Configuration',
            'jenkins.yml': 'Jenkins Configuration',
            '.jenkins': 'Jenkins',
            'tekton/': 'Tekton',
            'tekton.yaml': 'Tekton',
            'tekton.yml': 'Tekton',
            'argo/': 'Argo CD',
            'argo.yaml': 'Argo CD',
            'argo.yml': 'Argo CD',
            'prow.yaml': 'Prow',
            'fleet.yaml': 'Fleet',
            'github-actions/': 'GitHub Actions (Alt)',
            'actions/': 'GitHub Actions (Alt)',

            # Java build tools
            'build.gradle': 'Gradle',
            'build.gradle.kts': 'Gradle (Kotlin DSL)',
            'settings.gradle': 'Gradle Settings',
            'settings.gradle.kts': 'Gradle Settings (Kotlin)',
            'gradle.properties': 'Gradle Properties',
            'gradle-wrapper.properties': 'Gradle Wrapper',
            'gradlew': 'Gradle Wrapper',
            'gradlew.bat': 'Gradle Wrapper (Windows)',
            'pom.xml': 'Maven',
            'pom.xml': 'Maven POM',
            'maven.config': 'Maven Config',
            'project.xml': 'Maven Project',
            'build.xml': 'Apache Ant',
            'ant.xml': 'Apache Ant',
            'ant.properties': 'Apache Ant',
            'build.properties': 'Apache Ant',
            'ivy.xml': 'Apache Ivy',
            'ivysettings.xml': 'Apache Ivy',
            'project.clj': 'Leiningen',
            'profile.clj': 'Leiningen Profile',
            'boot.properties': 'Clojure CLI',
            'deps.edn': 'Clojure CLI',
            'shadow-cljs.edn': 'Shadow CLJS',
            'bb.edn': 'Babashka',

            # .NET build tools
            'project.json': '.NET Core',
            'global.json': '.NET CLI',
            'Directory.Build.props': 'MSBuild',
            'Directory.Build.targets': 'MSBuild',
            'nuget.config': 'NuGet',
            'packages.config': 'NuGet (Legacy)',
            'csproj': 'MSBuild Project',
            'vbproj': 'MSBuild Project (VB)',
            'fsproj': 'MSBuild Project (F#)',
            '.csproj': 'MSBuild Project',
            '.vbproj': 'MSBuild Project (VB)',
            '.fsproj': 'MSBuild Project (F#)',
            'dotnet-tools.json': '.NET Tools',
            'project.assets.json': '.NET Project Assets',

            # Ruby build tools
            'Gemfile': 'Bundler',
            'Gemfile.lock': 'Bundler Lock',
            'gems.rb': 'Bundler',
            'gems.locked': 'Bundler Lock',
            'Rakefile': 'Rake',
            'Rakefile.rb': 'Rake',
            'rake.rb': 'Rake',
            'Capfile': 'Capistrano',
            'capfile': 'Capistrano',
            'Berksfile': 'Berkshelf',
            'Berksfile.lock': 'Berkshelf Lock',
            'Cheffile': 'Chef',
            'metadata.rb': 'Chef Cookbook',
            'Policyfile.rb': 'Chef Policy',
            'Thorfile': 'Thor',
            'Guardfile': 'Guard',
            'config.ru': 'Rack',
            'puma.rb': 'Puma',
            'unicorn.rb': 'Unicorn',
            'sidekiq.yml': 'Sidekiq',
            'delayed_job_active_record.gemspec': 'Delayed Job',

            # PHP build tools
            'composer.json': 'Composer',
            'composer.lock': 'Composer Lock',
            'package.json': 'NPM (PHP)',
            'package-lock.json': 'NPM (PHP)',
            'yarn.lock': 'Yarn (PHP)',
            'pnpm-lock.yaml': 'pnpm (PHP)',
            'webpack.mix.js': 'Laravel Mix',
            'vite.config.js': 'Vite (PHP)',
            'postcss.config.js': 'PostCSS (PHP)',
            'tailwind.config.js': 'Tailwind (PHP)',
            'build.properties': 'Magento Build',
            'composer.phar': 'Composer Phar',
            'phing/build.xml': 'Phing',
            'phinx.yml': 'Phinx',
            'phinx.yaml': 'Phinx',
            'phinx.php': 'Phinx',
            'doctrine/migrations': 'Doctrine Migrations',
            'migrations.yml': 'Symfony Migrations',
            'doctrine.yaml': 'Doctrine Configuration',

            # Go build tools
            'go.mod': 'Go Modules',
            'go.sum': 'Go Modules Checksum',
            'go.work': 'Go Workspace',
            'go.work.sum': 'Go Workspace Checksum',
            'Gopkg.toml': 'Dep',
            'Gopkg.lock': 'Dep Lock',
            'glide.yaml': 'Glide',
            'glide.lock': 'Glide Lock',
            'vendor.conf': 'Govend',
            'vendor.json': 'Govend JSON',
            'golangci.yml': 'golangci-lint',
            'golangci.yaml': 'golangci-lint',
            '.golangci.yml': 'golangci-lint',
            '.golangci.yaml': 'golangci-lint',
            '.golangci.yml': 'golangci-lint',
            'go.renovate.json': 'Go Renovate',
            'goreleaser.yml': 'GoReleaser',
            'goreleaser.yaml': 'GoReleaser',
            '.goreleaser.yml': 'GoReleaser',
            '.goreleaser.yaml': 'GoReleaser',

            # Rust build tools
            'Cargo.toml': 'Cargo',
            'Cargo.lock': 'Cargo Lock',
            'rust-toolchain': 'Rust Toolchain',
            'rust-toolchain.toml': 'Rust Toolchain',
            'rustfmt.toml': 'Rustfmt',
            '.rustfmt.toml': 'Rustfmt',
            'clippy.toml': 'Clippy',
            '.clippy.toml': 'Clippy',
            'justfile': 'Just',
            'justfile': 'Just Task Runner',
            'Cargo.toml': 'Cargo (Possible Audit)',
            'deny.toml': 'Cargo Deny',
            'audit.toml': 'Cargo Audit',
            'out-of-tree.toml': 'Out of Tree',

            # C/C++ build tools
            'CMakeLists.txt': 'CMake',
            'CMakeCache.txt': 'CMake Cache',
            'cmake_install.cmake': 'CMake Install',
            'Makefile': 'Make',
            'makefile': 'Make',
            'configure.ac': 'Autotools',
            'configure.in': 'Autotools',
            'Makefile.am': 'Automake',
            'Makefile.in': 'Autotools Template',
            'autogen.sh': 'Autogen',
            'bootstrap': 'Bootstrap',
            'build.ninja': 'Ninja',
            'rules.ninja': 'Ninja Rules',
            '.ninja_log': 'Ninja Log',
            'ninja.build': 'Ninja Build',
            'conanfile.txt': 'Conan',
            'conanfile.py': 'Conan',
            'CMakeLists.txt': 'Conan (Possible)',
            'vcpkg.json': 'vcpkg',
            'vcpkg.json': 'vcpkg Package Manager',

            # Swift build tools
            'Package.swift': 'Swift Package Manager',
            'Package.resolved': 'SPM Resolved',
            'Podfile': 'CocoaPods',
            'Podfile.lock': 'CocoaPods Lock',
            'Cartfile': 'Carthage',
            'Cartfile.resolved': 'Carthage Resolved',
            'Brewfile': 'Homebrew',
            'Mintfile': 'Mint',

            # Task runners and automation
            'justfile': 'Just',
            'Taskfile.yml': 'Task',
            'Taskfile.yaml': 'Task',
            'tasks.py': 'Invoke',
            'invoke.yaml': 'Invoke',
            'noxfile.py': 'Nox',
            'tox.ini': 'Tox',
            'hatch.toml': 'Hatch',
            'pyproject.toml': 'Hatch (Possible)',
            'pdm.lock': 'PDM',
            'Pipfile': 'Pipenv',
            'Pipfile.lock': 'Pipenv Lock',
            'poetry.lock': 'Poetry Lock',
            'pyproject.toml': 'Poetry (Possible)',
            'pnpm-workspace.yaml': 'pnpm Workspace',
            'lerna.json': 'Lerna',
            'rush.json': 'Rush',
            'pnpm-workspace.yaml': 'pnpm Workspace',
            'yarn.lock': 'Yarn Workspace (Possible)',
            'package.json': 'Nx Workspace (Possible)',
            'workspace.json': 'Angular Workspace',
            'angular.json': 'Angular Workspace',
            'project.json': 'Angular Project',
            'nest-cli.json': 'NestJS Project',

            # Documentation generators
            'mkdocs.yml': 'MkDocs',
            'mkdocs.yaml': 'MkDocs',
            'docusaurus.config.js': 'Docusaurus',
            'docusaurus.config.ts': 'Docusaurus (TS)',
            'vuepress.config.js': 'VuePress',
            'vuepress.config.ts': 'VuePress (TS)',
            'vite.config.js': 'VitePress (Possible)',
            'vitepress.config.js': 'VitePress',
            'vitepress.config.ts': 'VitePress (TS)',
            'gridsome.config.js': 'Gridsome',
            'gridsome.config.ts': 'Gridsome (TS)',
            'gatsby-config.js': 'Gatsby',
            'gatsby-config.ts': 'Gatsby (TS)',
            'next.config.js': 'Next.js (Possible)',
            'nuxt.config.js': 'Nuxt.js (Possible)',
            'svelte.config.js': 'SvelteKit (Possible)',
            'storybook/': 'Storybook',
            '.storybook/': 'Storybook',
            'stencil.config.ts': 'Stencil',
            'stencil.config.js': 'Stencil',

            # Miscellaneous tools
            '.editorconfig': 'EditorConfig',
            '.gitattributes': 'Git Attributes',
            '.gitignore': 'Git Ignore',
            '.gitignore-global': 'Git Ignore (Global)',
            '.gitignore': 'Git Ignore (Global)',
            '.prettierignore': 'Prettier Ignore',
            '.eslintignore': 'ESLint Ignore',
            '.dockerignore': 'Docker Ignore',
            '.nodemonignore': 'Nodemon Ignore',
            '.stylelintignore': 'Stylelint Ignore',
            '.nvmrc': 'NVM Config',
            '.node-version': 'Node Version',
            '.python-version': 'Python Version',
            '.ruby-version': 'Ruby Version',
            '.go-version': 'Go Version',
            '.java-version': 'Java Version',
            'phpunit.xml': 'PHPUnit',
            'phpunit.xml.dist': 'PHPUnit',
            'phpcs.xml': 'PHP_CodeSniffer',
            'phpstan.neon': 'PHPStan',
            'psalm.xml': 'Psalm',
            'infection.json': 'Infection',
            'humbug.json': 'Humbug',
            'behat.yml': 'Behat',
            'codeception.yml': 'Codeception',
            '.php-cs-fixer.php': 'PHP CS Fixer',
            '.php_cs': 'PHP CS Fixer',
            'phpmd.xml': 'PHPMD',
            'phpdox.xml': 'PHPDocX',
            'phpunit.xml': 'PHPUnit (Alt)',
            'phpunit.xml': 'PHPUnit (Alt)',
            'robocode.xml': 'Robo',
            'phinx.yml': 'Phinx',
            'propel.ini': 'Propel',
            'doctrine.dbal.xml': 'Doctrine DBAL',
            'doctrine.orm.xml': 'Doctrine ORM',
            'sami.php': 'Sami',
            'api-skeleton': 'API Skeleton',
            'swagger.yaml': 'Swagger',
            'swagger.yml': 'Swagger',
            'swagger.json': 'Swagger',
            'openapi.yaml': 'OpenAPI',
            'openapi.yml': 'OpenAPI',
            'openapi.json': 'OpenAPI',
            'raml.yaml': 'RAML',
            'raml.yml': 'RAML',
            'api-blueprint.md': 'API Blueprint',
            'postman.json': 'Postman',
            'postman_collection.json': 'Postman',
            'insomnia.json': 'Insomnia',
            'http-client.env.json': 'HTTP Client',
            'http-client.private.env.json': 'HTTP Client (Private)',
            '.http': 'HTTP File',
            'rest-client.env.json': 'REST Client',
            'thunder-tests': 'Thunder Client',
            'bruno.json': 'Bruno',
            'bruno.toml': 'Bruno'
        }

        # Check for each build tool
        for file_path, tool in build_files.items():
            # Handle directory patterns ending with /
            if file_path.endswith('/'):
                dir_path = self.root_path / file_path.rstrip('/')
                if dir_path.exists() and dir_path.is_dir():
                    tools.append(tool)
            # Handle glob patterns
            elif '*' in file_path:
                for found_path in self.root_path.glob(file_path):
                    if found_path.exists():
                        tools.append(tool)
                        break
            # Handle exact file matches
            else:
                if (self.root_path / file_path).exists():
                    tools.append(tool)

        # Remove duplicates while preserving order
        return list(dict.fromkeys(tools))

    async def _detect_build_tools_optimized(self) -> List[str]:
        """Detect build tools using high-performance cache - Optimized version"""
        tools = []

        # Key build tool files to check
        build_tool_files = {
            'Makefile': 'Make',
            'package.json': 'Node.js',
            'requirements.txt': 'Python',
            'go.mod': 'Go Modules',
            'Cargo.toml': 'Cargo',
            'pom.xml': 'Maven',
            'build.gradle': 'Gradle',
            'Dockerfile': 'Docker',
            'docker-compose.yml': 'Docker Compose',
            'docker-compose.yaml': 'Docker Compose',
            'tsconfig.json': 'TypeScript',
            'webpack.config.js': 'Webpack',
            'vite.config.js': 'Vite',
            'jest.config.js': 'Jest',
            '.github/workflows': 'GitHub Actions',
            '.gitlab-ci.yml': 'GitLab CI',
            'Jenkinsfile': 'Jenkins',
            'circle.yml': 'CircleCI',
            'cypress.config.js': 'Cypress',
            'playwright.config.js': 'Playwright',
            'pytest.ini': 'Pytest',
            'tox.ini': 'Tox',
            'CMakeLists.txt': 'CMake',
            'babel.config.js': 'Babel',
            'rollup.config.js': 'Rollup',
            'yarn.lock': 'Yarn',
            'pnpm-lock.yaml': 'pnpm',
            'poetry.lock': 'Poetry',
            'composer.lock': 'Composer',
            'Gemfile.lock': 'Bundler',
            '.eslintrc.js': 'ESLint',
            'angular.json': 'Angular CLI'
        }

        # Check for build tools using cache
        for file_name, tool_name in build_tool_files.items():
            if self.cache.has_file(file_name) or self.cache.get_files_by_name(file_name):
                tools.append(tool_name)

        # Check for directories
        directories = self.cache.get_directories()
        build_directories = {
            '.github': 'GitHub Actions',
            '.circleci': 'CircleCI',
            'node_modules': 'Node.js',
            'target': 'Maven/Gradle',
            'build': 'Build Directory',
            'dist': 'Distribution',
            'coverage': 'Coverage Reports'
        }

        for dir_name, dir_desc in build_directories.items():
            if dir_name in directories:
                tools.append(dir_desc)

        return tools

    async def _count_files(self) -> int:
        """Count total files efficiently using FilesystemUtils"""
        try:
            # Use FilesystemUtils which leverages ripgrep for faster file listing
            files = await self.fs_utils.list_directory('.', recursive=True, show_hidden=False)
            return len([f for f in files if (self.root_path / f).is_file()])
        except Exception:
            return 0

    async def _estimate_size(self) -> str:
        """Get rough size estimate"""
        try:
            result = await asyncio.create_subprocess_exec(
                'du', '-sh', str(self.root_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            return stdout.decode().split()[0] if stdout.decode() else 'Unknown'
        except Exception:
            return 'Unknown'

    async def _get_top_level_structure(self) -> Dict[str, Any]:
        """Get immediate directory structure"""
        structure = {}

        for item in self.root_path.iterdir():
            if item.name.startswith('.'):
                continue

            if item.is_dir():
                structure[item.name] = {
                    'type': 'directory',
                    'item_count': len(list(item.iterdir())) if item.exists() else 0
                }
            else:
                structure[item.name] = {
                    'type': 'file',
                    'size': item.stat().st_size
                }

        return structure

    async def _find_entry_points(self) -> List[str]:
        """Find likely entry points - more robust filtering"""
        entry_points = []

        # Common entry point patterns by priority
        patterns = [
            # Python
            'main.py', 'app.py', 'run.py', 'serve.py', 'start.py',
            'manage.py',  # Django
            'wsgi.py',    # WSGI applications
            # JavaScript/TypeScript
            'index.js', 'index.ts', 'server.js', 'app.js',
            'src/index.js', 'src/index.ts', 'src/main.js', 'src/main.ts',
            # Go
            'main.go', 'cmd/main.go', 'cmd/server/main.go',
            # Java
            'main.java', 'Main.java', 'Application.java',
            'src/main/java/Main.java',
            # C#
            'Program.cs', 'Main.cs', 'Startup.cs',
            # Shell scripts
            'start.sh', 'run.sh', 'deploy.sh'
        ]

        for pattern in patterns:
            if (self.root_path / pattern).exists():
                entry_points.append(pattern)

        # Look for main functions in Python (limit to important ones)
        try:
            result = await asyncio.create_subprocess_exec(
                'grep', '-rl', '--include=*.py', "if __name__ == '__main__'",
                str(self.root_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            for line in stdout.decode().splitlines():
                full_path = line.replace(str(self.root_path) + '/', '')
                # Filter out virtual environment and dependency paths
                if not any(exclude in full_path.lower() for exclude in [
                    'site-packages', 'venv/', 'env/', '.venv/', 'node_modules/',
                    'dist/', 'build/', '__pycache__/', 'pip/_vendor'
                ]):
                    entry_points.append(full_path)
        except Exception as e:
            pass  # Silently continue if grep fails

        # Look for executables with shebang
        try:
            result = await asyncio.create_subprocess_exec(
                'find', str(self.root_path), '-type', 'f', '-executable',
                '-exec', 'grep', '-l', '^#!', '{}', ';',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            for line in stdout.decode().splitlines()[:5]:  # Limit to 5
                full_path = line.replace(str(self.root_path) + '/', '')
                if not any(exclude in full_path.lower() for exclude in [
                    'site-packages', 'venv/', 'env/', '.venv/', 'node_modules/'
                ]):
                    entry_points.append(full_path)
        except Exception:
            pass

        # Remove duplicates and filter for relevance
        seen = set()
        filtered = []
        for ep in entry_points:
            if ep not in seen and len(ep) < 200:  # Reasonable path length
                seen.add(ep)
                filtered.append(ep)

        return filtered[:10]  # Return top 10 entry points


class StructuralMapper:
    """Phase 1.2: Map architecture and conventions - Optimized"""

    def __init__(self, root_path: str, cache: Optional[FileCache] = None):
        self.root_path = Path(root_path)
        self.fs_utils = FilesystemUtils(self.root_path)
        # Use provided cache or create new one
        self.cache = cache or FileCache(self.root_path)

    async def map_architecture(self, fingerprint: CodebaseFingerprint) -> Dict[str, Any]:
        """Analyze codebase architecture and patterns"""
        architecture = {
            'patterns': [],
            'conventions': {},
            'modules': {},
            'tests': {},
            'configuration': {},
            'documentation': {}
        }

        # Detect architectural patterns
        architecture['patterns'] = await self._detect_patterns()

        # Analyze conventions
        architecture['conventions'] = await self._analyze_conventions()

        # Map modules
        architecture['modules'] = await self._map_modules(fingerprint)

        # Analyze testing setup
        architecture['tests'] = await self._analyze_testing()

        # Find configuration
        architecture['configuration'] = await self._find_configuration()

        # Check documentation
        architecture['documentation'] = await self._find_documentation()

        return architecture

    async def _detect_patterns(self) -> List[str]:
        """Detect architectural patterns from structure - Comprehensive coverage"""
        patterns = []

        # Multi-language architecture detection
        language_dirs = set()
        for item in self.root_path.iterdir():
            if item.is_dir():
                # Check for language-specific directories
                if item.name.lower() in ['gobase', 'go-code', 'golang', 'go-src']:
                    language_dirs.add('Go')
                elif any(x in item.name.lower() for x in ['backend', 'api', 'server', 'service']):
                    # Check if Python backend
                    if any((item / f).exists() for f in ['main.py', 'app.py', 'wsgi.py', '__init__.py']):
                        language_dirs.add('Python')
                elif 'frontend' in item.name.lower() or 'web' in item.name.lower():
                    language_dirs.add('Frontend')

        if len(language_dirs) > 1:
            patterns.append(f"Multi-Language Architecture ({', '.join(sorted(language_dirs))})")

        # Microservices detection
        service_dirs = []
        for item in self.root_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Common microservice patterns
                if any(x in item.name.lower() for x in [
                    'service', 'microservice', 'api-', '-api', 'srv-',
                    'auth', 'user', 'payment', 'notification', 'email',
                    'gateway', 'proxy', 'loadbalancer', 'lb',
                    'relayer', 'bridge', 'crawler', 'parser', 'processor'
                ]):
                    service_dirs.append(item.name)

        if len(service_dirs) >= 2:
            patterns.append("Microservices Architecture")

        # Blockchain-specific architecture
        blockchain_indicators = []
        for item in self.root_path.iterdir():
            if item.is_dir():
                if any(x in item.name.lower() for x in [
                    'aurora', 'bridge', 'relayer', 'blockchain', 'web3',
                    'contract', 'defi', 'dex', 'nft', 'dao', 'token',
                    'ethereum', 'evm', 'solana', 'polygon'
                ]):
                    blockchain_indicators.append(item.name)

        if blockchain_indicators:
            if len(blockchain_indicators) >= 2:
                patterns.append("Blockchain Multi-Protocol Architecture")
            else:
                patterns.append("Blockchain-Focused Architecture")

        # Data Processing / Analytics architecture
        data_indicators = []
        for item in self.root_path.iterdir():
            if item.is_dir():
                if any(x in item.name.lower() for x in [
                    'analytics', 'insights', 'data', 'etl', 'pipeline',
                    'indexer', 'aggregator', 'processor', 'crawler'
                ]):
                    data_indicators.append(item.name)

        if data_indicators:
            if len(data_indicators) >= 2:
                patterns.append("Data Processing Pipeline Architecture")
            else:
                patterns.append("Analytics-Focused Architecture")

        # MVC and its variations across frameworks
        mvc_patterns = {
            'MVC (Classic)': ['controllers', 'models', 'views'],
            'MVC (Django)': ['views.py', 'models.py', 'forms.py', 'templates/'],
            'MVC (Rails)': ['app/controllers', 'app/models', 'app/views'],
            'MVC (ASP.NET)': ['Controllers/', 'Models/', 'Views/'],
            'MVC (Spring)': ['src/main/java/.../controller', 'src/main/java/.../model', 'src/main/java/.../view'],
            'MVC (Laravel)': ['app/Http/Controllers', 'app/Models', 'resources/views'],
            'MVC (Symfony)': ['src/Controller', 'src/Model', 'templates/'],
            'MVVM': ['ViewModels/', 'Views/', 'Models/'],
            'MVP': ['Presenters/', 'Views/', 'Models/'],
            'MVC+Service': ['controllers', 'services', 'models', 'views']
        }

        # Modern JavaScript/React patterns
        js_patterns = {
            'React Components': ['components/', 'hooks/', 'context/'],
            'Next.js App Router': ['app/', 'app/api/', 'app/(pages)/'],
            'Next.js Pages Router': ['pages/', 'pages/api/'],
            'Create React App': ['src/App.js', 'src/index.js', 'public/'],
            'Vite React': ['vite.config.js', 'src/main.jsx', 'src/App.jsx'],
            'Redux Pattern': ['store/', 'reducers/', 'actions/', 'middleware/'],
            'Redux Toolkit': ['store/', 'slices/', 'api/'],
            'Context API': ['context/', 'providers/'],
            'Zustand': ['store/', 'hooks/'],
            'Recoil': ['atoms/', 'selectors/'],
            'MobX': ['store/', 'actions/', 'stores/'],
            'Vue.js Composition': ['composables/', 'components/'],
            'Vue.js Options': ['components/', 'mixins/'],
            'Nuxt.js': ['pages/', 'components/', 'layouts/', 'middleware/'],
            'Angular': ['src/app/', 'components/', 'services/', 'modules/', 'pipes/', 'guards/'],
            'Svelte': ['src/routes/', 'src/lib/', 'src/components/'],
            'SvelteKit': ['src/routes/', 'src/lib/', 'src/hooks/'],
            'Astro': ['src/pages/', 'src/layouts/', 'src/components/']
        }

        # Enterprise patterns (Java, .NET)
        enterprise_patterns = {
            'Spring Boot': ['src/main/java', 'src/main/resources', 'src/test/java', 'application.properties'],
            'Spring MVC': ['src/main/java/.../controller', 'src/main/java/.../service', 'src/main/java/.../repository'],
            'Spring Cloud': ['config/', 'gateway/', 'eureka/', 'hystrix/'],
            'Jakarta EE': ['src/main/java', 'WEB-INF/', 'META-INF/', 'resources/'],
            'Java EE': ['ejb/', 'war/', 'ear/', 'jpa/'],
            'ASP.NET Core': ['Controllers/', 'Models/', 'Views/', 'Program.cs', 'Startup.cs'],
            'ASP.NET MVC': ['Controllers/', 'Models/', 'Views/', 'App_Start/'],
            'ASP.NET Web API': ['Controllers/', 'Models/', 'App_Start/'],
            'Entity Framework': ['Migrations/', 'Models/', 'DbContext.cs'],
            '.NET MAUI': ['MauiProgram.cs', 'Platforms/', 'Resources/'],
            'Xamarin': ['MainActivity.cs', 'Resources/', 'Droid/'],
            'Blazor': ['Pages/', 'Shared/', 'Program.cs', 'App.razor'],
            'WPF': ['Views/', 'ViewModels/', 'Models/', 'App.xaml'],
            'WinForms': ['Forms/', 'Models/', 'Program.cs']
        }

        # Microservices and distributed systems
        microservices_patterns = {
            'Microservices': ['services/', 'apps/', 'microservices/', 'src/services/'],
            'Event-Driven': ['events/', 'handlers/', 'publishers/', 'subscribers/'],
            'CQRS': ['commands/', 'queries/', 'handlers/'],
            'Event Sourcing': ['events/', 'snapshots/', 'aggregates/'],
            'Saga Pattern': ['sagas/', 'orchestration/', 'choreography/'],
            'API Gateway': ['gateway/', 'routes/', 'middleware/'],
            'Service Mesh': ['mesh/', 'istio/', 'linkerd/'],
            'Distributed Tracing': ['tracing/', 'opentracing/', 'jaeger/'],
            'Circuit Breaker': ['circuitbreaker/', 'resilience/', 'hystrix/']
        }

        # Domain-Driven Design patterns
        ddd_patterns = {
            'DDD (Classic)': ['domain/', 'application/', 'infrastructure/', 'presentation/'],
            'DDD (Hexagonal)': ['domain/', 'application/', 'ports/', 'adapters/'],
            'DDD (Onion)': ['domain/', 'application/', 'infrastructure/', 'core/'],
            'DDD Entities': ['domain/entities/', 'domain/valueobjects/'],
            'DDD Aggregates': ['domain/aggregates/', 'domain/repositories/'],
            'DDD Services': ['domain/services/', 'application/services/'],
            'DDD Events': ['domain/events/', 'application/events/']
        }

        # Clean Architecture patterns
        clean_arch_patterns = {
            'Clean Architecture': ['entities/', 'usecases/', 'interfaces/', 'frameworks/'],
            'Clean Architecture (Python)': ['src/domain/', 'src/usecases/', 'src/interface_adapters/', 'src/frameworks/'],
            'Clean Architecture (Java)': ['src/main/java/com/.../domain/', 'src/main/java/com/.../usecase/', 'src/main/java/com/.../interface/'],
            'Ports and Adapters': ['ports/', 'adapters/', 'application/', 'domain/'],
            'Hexagonal Architecture': ['ports/', 'adapters/', 'application/', 'domain/']
        }

        # Data Layer patterns
        data_patterns = {
            'Repository Pattern': ['repositories/', 'repository/', 'Repository.php', 'Repository.java'],
            'Data Mapper': ['mappers/', 'mappers/', 'DataMapper.php'],
            'Active Record': ['models/', 'entity/', 'Entity.php'],
            'Unit of Work': ['UnitOfWork.php', 'UnitOfWork.cs', 'work/'],
            'Query Object': ['query/', 'Query.php', 'criteria/'],
            'CQRS': ['commands/', 'queries/', 'CommandHandler.php'],
            'Event Sourcing': ['events/', 'snapshots/', 'EventStore.php'],
            'Database First': ['db/', 'database/', 'sql/', 'migrations/'],
            'Code First': ['models/', 'entities/', 'DbContext.cs']
        }

        # Frontend patterns
        frontend_patterns = {
            'Atomic Design': ['atoms/', 'molecules/', 'organisms/', 'templates/', 'pages/'],
            'Feature-First': ['features/', 'modules/', 'pages/'],
            'Domain-Driven (UI)': ['features/', 'shared/', 'entities/'],
            'Component Library': ['lib/', 'components/', 'stories/', 'dist/'],
            'Design System': ['tokens/', 'components/', 'patterns/', 'docs/'],
            'Monorepo (Nx)': ['apps/', 'libs/', 'tools/', 'nx.json'],
            'Monorepo (Lerna)': ['packages/', 'lerna.json'],
            'Monorepo (Rush)': ['projects/', 'rush.json'],
            'Static Site Generator': ['src/pages/', 'static/', 'public/', 'build/'],
            'SPA (React)': ['src/', 'public/', 'build/', 'index.html'],
            'MPA (Traditional)': ['views/', 'templates/', 'static/', 'public/']
        }

        # Testing patterns
        test_patterns = {
            'Test Pyramid': ['unit/', 'integration/', 'e2e/'],
            'BDD': ['features/', 'step_definitions/', 'support/'],
            'TDD': ['tests/', 'spec/', '__tests__/'],
            'Contract Testing': ['contracts/', 'pacts/', 'specifications/'],
            'Property-Based Testing': ['properties/', 'generators/'],
            'Visual Testing': ['visual/', 'screenshots/', 'regression/'],
            'Performance Testing': ['performance/', 'load/', 'stress/'],
            'Security Testing': ['security/', 'penetration/', 'vulnerability/']
        }

        # DevOps and Infrastructure patterns
        devops_patterns = {
            'GitOps': ['manifests/', 'k8s/', 'helm/', 'argocd/'],
            'IaC (Terraform)': ['terraform/', 'infra/', 'modules/'],
            'IaC (CloudFormation)': ['cloudformation/', 'templates/', 'stacks/'],
            'IaC (ARM)': ['arm/', 'bicep/', 'templates/'],
            'Docker Compose': ['docker-compose.yml', 'docker-compose.override.yml', 'services/'],
            'Kubernetes': ['k8s/', 'kubernetes/', 'deployments/', 'services/', 'ingress/'],
            'Helm Charts': ['helm/', 'Chart.yaml', 'values/', 'templates/'],
            'Serverless': ['serverless.yml', 'functions/', 'layers/'],
            'Lambda': ['lambda/', 'functions/', 'sam.yml'],
            'Azure Functions': ['function.json', 'Functions/']
        }

        # Database patterns
        db_patterns = {
            'Relational DB': ['sql/', 'schema.sql', 'migrations/', 'seeds/'],
            'NoSQL (Document)': ['collections/', 'documents/', 'nosql/'],
            'NoSQL (Key-Value)': ['keyvalue/', 'cache/', 'redis/'],
            'NoSQL (Graph)': ['nodes/', 'edges/', 'graph/', 'neo4j/'],
            'NoSQL (Column)': ['columnfamily/', 'cassandra/', 'widecolumn/'],
            'Search Engine': ['elasticsearch/', 'solr/', 'algolia/', 'indices/'],
            'Time Series': ['timeseries/', 'influx/', 'prometheus/'],
            'Data Lake': ['lake/', 'data/', 'raw/', 'processed/'],
            'Data Warehouse': ['warehouse/', 'dw/', 'etl/', 'bi/']
        }

        # Combine all pattern indicators
        all_patterns = {
            **mvc_patterns,
            **js_patterns,
            **enterprise_patterns,
            **microservices_patterns,
            **ddd_patterns,
            **clean_arch_patterns,
            **data_patterns,
            **frontend_patterns,
            **test_patterns,
            **devops_patterns,
            **db_patterns
        }

        # Check for each pattern
        for pattern, indicators in all_patterns.items():
            matches = 0
            total_indicators = len(indicators)

            for indicator in indicators:
                # Check for exact file match
                if (self.root_path / indicator).exists():
                    matches += 1
                # Check for directory match
                elif (self.root_path / indicator).is_dir():
                    matches += 1
                # Check for pattern in file tree
                else:
                    try:
                        if any(indicator in str(p) for p in self.root_path.rglob('*')):
                            matches += 1
                            break  # Found at least one match for this indicator
                    except Exception as e:
                        print(f"[Reconnaissance] Error checking pattern '{indicator}': {e}")
                        pass

            # Only add pattern if we have clear evidence - no fallbacks or guesses
            if total_indicators <= 2:
                # For small indicator sets (1-2), require ALL indicators to match
                if matches == total_indicators and matches > 0:
                    patterns.append(pattern)
            elif total_indicators <= 4:
                # For medium indicator sets (3-4), require at least 75% matches
                if matches / total_indicators >= 0.75:
                    patterns.append(pattern)
            else:
                # For large indicator sets (5+), require at least 60% matches
                if matches / total_indicators >= 0.6:
                    patterns.append(pattern)

        return patterns

    async def _analyze_conventions(self) -> Dict[str, Any]:
        """Analyze coding conventions"""
        conventions = {
            'naming': {},
            'structure': {},
            'imports': {}
        }

        # Analyze Python naming conventions
        if any(self.root_path.rglob('*.py')):
            conventions['naming']['python'] = await self._analyze_python_naming()

        # Analyze JavaScript/TypeScript conventions
        if any(self.root_path.rglob('*.js')) or any(self.root_path.rglob('*.ts')):
            conventions['naming']['javascript'] = await self._analyze_js_naming()

        return conventions

    async def _analyze_python_naming(self) -> Dict[str, List[str]]:
        """Analyze Python naming patterns using FilesystemUtils"""
        patterns = {
            'snake_case': [],
            'PascalCase': [],
            'camelCase': [],
            'UPPER_CASE': []
        }

        # Use FilesystemUtils to efficiently find Python files
        py_files = await self.fs_utils.find_files('*.py', '.', exclude_patterns=[
            '*.pyc', '__pycache__', '.git', 'node_modules', '.venv', 'venv'
        ])

        # Sample the first 20 Python files for analysis
        for file_path in py_files[:20]:
            try:
                # Use FilesystemUtils to read the file
                rel_path = str(Path(file_path).relative_to(self.root_path))
                file_result = await self.fs_utils.read_file(rel_path)

                if file_result.error:
                    continue

                content = file_result.content

                # Find class names
                classes = re.findall(r'class\s+([A-Z][a-zA-Z0-9]*)', content)
                patterns['PascalCase'].extend(classes)

                # Find function names
                functions = re.findall(r'def\s+([a-z_][a-z0-9_]*)', content)
                patterns['snake_case'].extend(functions)

                # Find constants
                constants = re.findall(r'([A-Z_][A-Z0-9_]*)\s*=', content)
                patterns['UPPER_CASE'].extend(constants)

            except Exception:
                pass

        return patterns

    async def _analyze_js_naming(self) -> Dict[str, List[str]]:
        """Analyze JavaScript/TypeScript naming patterns using FilesystemUtils"""
        patterns = {
            'camelCase': [],
            'PascalCase': [],
            'snake_case': [],
            'kebab-case': []
        }

        # Use FilesystemUtils to efficiently find JS/TS files
        js_files = await self.fs_utils.find_files('*.js', '.', exclude_patterns=[
            '*.min.js', '.git', 'node_modules', 'dist', 'build'
        ])
        ts_files = await self.fs_utils.find_files('*.ts', '.', exclude_patterns=[
            '*.min.js', '.git', 'node_modules', 'dist', 'build'
        ])
        jsx_files = await self.fs_utils.find_files('*.jsx', '.', exclude_patterns=[
            '*.min.js', '.git', 'node_modules', 'dist', 'build'
        ])
        tsx_files = await self.fs_utils.find_files('*.tsx', '.', exclude_patterns=[
            '*.min.js', '.git', 'node_modules', 'dist', 'build'
        ])

        # Combine all JS/TS related files
        all_js_files = (js_files + ts_files + jsx_files + tsx_files)[:40]  # Sample first 40

        for file_path in all_js_files:
            try:
                # Use FilesystemUtils to read the file
                rel_path = str(Path(file_path).relative_to(self.root_path))
                file_result = await self.fs_utils.read_file(rel_path)

                if file_result.error:
                    continue

                content = file_result.content

                # Find function names
                functions = re.findall(r'(?:function|const|let|var)\s+([a-z][a-zA-Z0-9]*)', content)
                patterns['camelCase'].extend(functions)

                # Find class names
                classes = re.findall(r'class\s+([A-Z][a-zA-Z0-9]*)', content)
                patterns['PascalCase'].extend(classes)

            except Exception:
                pass

        return patterns

    async def _map_modules(self, fingerprint: CodebaseFingerprint) -> Dict[str, Any]:
        """Map module organization"""
        modules = {
            'directories': {},
            'dependencies': {},
            'circular_imports': []
        }

        # Analyze key directories
        for item in self.root_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                modules['directories'][item.name] = {
                    'file_count': len(list(item.rglob('*.*'))),
                    'subdirs': [d.name for d in item.iterdir() if d.is_dir()],
                    'purpose': self._infer_directory_purpose(item.name)
                }

        return modules

    def _infer_directory_purpose(self, dirname: str) -> str:
        """Infer directory purpose from name - Comprehensive tech stack coverage"""
        purposes = {
            # Core Application Structure
            'src': 'Source code',
            'lib': 'Library code',
            'app': 'Application code',
            'apps': 'Multiple applications',
            'packages': 'Package modules (Monorepo)',
            'projects': 'Project modules (Monorepo)',
            'modules': 'Application modules',

            # Layered Architecture
            'api': 'API endpoints',
            'apis': 'Multiple APIs',
            'services': 'Business services',
            'service': 'Business service',
            'business': 'Business logic',
            'domain': 'Domain logic (DDD)',
            'core': 'Core domain logic',
            'entities': 'Domain entities',
            'usecases': 'Use cases (Clean Arch)',
            'use_cases': 'Use cases (Clean Arch)',
            'interactors': 'Interactors (Clean Arch)',
            'application': 'Application layer',
            'presentation': 'Presentation layer',
            'infrastructure': 'Infrastructure layer',
            'interfaces': 'Interface definitions',
            'adapters': 'Adapters (Hexagonal)',
            'ports': 'Ports (Hexagonal)',

            # MVC/MVVM Patterns
            'models': 'Data models',
            'model': 'Data model',
            'views': 'View logic',
            'view': 'View logic',
            'controllers': 'Request handlers',
            'controller': 'Request handler',
            'viewmodels': 'View models (MVVM)',
            'viewmodel': 'View model (MVVM)',
            'presenters': 'Presenters (MVP)',
            'presenter': 'Presenter (MVP)',

            # React/JavaScript Specific
            'components': 'React/Vue components',
            'component': 'Component',
            'pages': 'Page components (Next/Nuxt)',
            'page': 'Page component',
            'routes': 'Route definitions',
            'route': 'Route definition',
            'hooks': 'Custom React hooks',
            'hook': 'React hook',
            'context': 'React context',
            'store': 'State management (Redux/Vuex)',
            'stores': 'State stores',
            'state': 'Application state',
            'atoms': 'Recoil atoms',
            'selectors': 'Recoil selectors',
            'reducers': 'Redux reducers',
            'actions': 'Redux actions',
            'middleware': 'Express/Redux middleware',
            'sagas': 'Redux sagas',
            'slices': 'Redux Toolkit slices',
            'composables': 'Vue composables',
            'directives': 'Vue/Angular directives',
            'pipes': 'Angular pipes',
            'guards': 'Angular/NestJS guards',
            'interceptors': 'Angular interceptors',
            'providers': 'NestJS providers',
            'decorators': 'Decorators',

            # Enterprise Java/.NET
            'java': 'Java source',
            'kotlin': 'Kotlin source',
            'scala': 'Scala source',
            'groovy': 'Groovy source',
            'csharp': 'C# source',
            'vb': 'VB.NET source',
            'fsharp': 'F# source',
            'main': 'Main source code',
            'test': 'Test code',

            # Blockchain & Web3 Specific
            'contracts': 'Smart contracts',
            'smart-contracts': 'Smart contracts',
            'solidity': 'Solidity contracts',
            'hardhat': 'Hardhat development',
            'truffle': 'Truffle development',
            'foundry': 'Foundry development',
            'web3': 'Web3 integration',
            'ethers': 'Ethers.js integration',
            'web3py': 'Web3.py integration',
            'blockchain': 'Blockchain utilities',
            'aurora': 'Aurora bridge/relayer',
            'relayer': 'Bridge relayer',
            'bridge': 'Blockchain bridge',
            'ethereum': 'Ethereum integration',
            'evm': 'EVM utilities',
            'defi': 'DeFi protocols',
            'dex': 'DEX integration',
            'nft': 'NFT utilities',
            'dao': 'DAO contracts',
            'token': 'Token contracts',
            'wallet': 'Wallet integration',
            'transactions': 'Transaction processing',
            'events': 'Event listeners',
            'logs': 'Event logs',
            'abis': 'ABI definitions',
            'artifacts': 'Build artifacts',

            # Data Processing & Analytics
            'analytics': 'Data analytics',
            'insights': 'Data insights',
            'data': 'Data processing',
            'etl': 'ETL pipelines',
            'pipeline': 'Data pipeline',
            'processor': 'Data processor',
            'parser': 'Data parser',
            'scraper': 'Web scraper',
            'crawler': 'Web crawler',
            'indexer': 'Data indexer',
            'aggregator': 'Data aggregator',
            'transformer': 'Data transformer',
            'normalizer': 'Data normalizer',
            'validator': 'Data validator',

            # Go-specific
            'gobase': 'Go codebase',
            'cmd': 'Command line tools',
            'bin': 'Binary executables',
            'pkg': 'Go packages',
            'internal': 'Internal packages',
            'proto': 'Protocol buffers',
            'grpc': 'gRPC services',

            # Infrastructure & DevOps
            'conf': 'Configuration',
            'config': 'Configuration',
            'configs': 'Configuration files',
            'deploy': 'Deployment scripts',
            'docker': 'Docker files',
            'k8s': 'Kubernetes configs',
            'terraform': 'Terraform configs',
            'infra': 'Infrastructure code',
            'monitoring': 'Monitoring setup',
            'logging': 'Logging setup',
            'metrics': 'Metrics collection',

            # MCP & Integrations
            'mcp': 'MCP (Model Context Protocol)',
            'twitter': 'Twitter integration',
            'social': 'Social media integration',
            'api-client': 'API client',
            'sdk': 'Software Development Kit',
            'client': 'Client library',
            'clients': 'Client libraries',
            'connector': 'System connector',
            'adapter': 'System adapter',
            'socket': 'Socket programming',
            'websocket': 'WebSocket handling',

            # Experiments & R&D
            'experiments': 'Experimental features',
            'lab': 'R&D projects',
            'sandbox': 'Sandbox environment',
            'prototype': 'Prototype code',
            'poc': 'Proof of concept',
            'research': 'Research code',
            'demo': 'Demo projects',
            'examples': 'Example code',
            'samples': 'Sample code',
            'tutorial': 'Tutorial code',

            # Load Balancing & Infrastructure
            'loadbalancer': 'Load balancer',
            'lb': 'Load balancer',
            'proxy': 'Proxy server',
            'gateway': 'API gateway',
            'nginx': 'Nginx config',
            'haproxy': 'HAProxy config',
            'traefik': 'Traefik config',

            # Common patterns
            'backend': 'Backend services',
            'frontend': 'Frontend code',
            'fullstack': 'Full-stack application',
            'microservice': 'Microservice',
            'server': 'Server code',
            'worker': 'Background workers',
            'jobs': 'Job processing',
            'queue': 'Message queue',
            'cache': 'Caching layer',
            'search': 'Search functionality',
            'storage': 'Storage layer',
            'auth': 'Authentication',
            'security': 'Security utilities',
            'utils': 'Utility functions',
            'helpers': 'Helper functions',
            'common': 'Common utilities',
            'shared': 'Shared code',
            'types': 'Type definitions',
            'interfaces': 'Type interfaces',
            'schemas': 'Data schemas',
            'migrations': 'Database migrations',
            'seeds': 'Database seeds',
            'fixtures': 'Test fixtures',
            'mocks': 'Mock objects',
            'stubs': 'Test stubs',
            'docs': 'Documentation',
            'scripts': 'Build/deploy scripts',
            'tools': 'Development tools',
            'assets': 'Static assets',
            'static': 'Static files',
            'public': 'Public files',
            'temp': 'Temporary files',
            'tmp': 'Temporary files',
            'logs': 'Log files',
            'backup': 'Backup files',
            'archive': 'Archived code',
            'tests': 'Test code',
            'spec': 'Test specifications',
            'specs': 'Test specifications',
            'resources': 'Application resources',
            'staticresources': 'Static resources (.NET)',
            'webapp': 'Web application (.NET)',
            'WEB-INF': 'Web-INF (Java EE)',
            'META-INF': 'META-INF (Java)',
            'assembly': 'Assembly (.NET)',
            'bin': 'Compiled binaries',
            'obj': 'Build objects',
            'out': 'Build output',
            'target': 'Maven/Gradle target',
            'build': 'Build output',
            'dist': 'Distribution',
            'release': 'Release build',
            'debug': 'Debug build',

            # Data Layer
            'database': 'Database related',
            'db': 'Database related',
            'data': 'Data access layer',
            'dal': 'Data Access Layer',
            'repository': 'Repository pattern',
            'repositories': 'Repository pattern',
            'mappers': 'Data mappers',
            'mapper': 'Data mapper',
            'migrations': 'Database migrations',
            'migration': 'Database migration',
            'seeds': 'Seed data',
            'seed': 'Seed data',
            'fixtures': 'Test fixtures',
            'fixture': 'Test fixture',
            'schema': 'Database schema',
            'schemas': 'Database schemas',
            'sql': 'SQL scripts',
            'queries': 'Database queries',
            'query': 'Database query',
            'entity': 'Entity classes',
            'entities': 'Entity classes',
            'dto': 'Data Transfer Objects',
            'dtos': 'Data Transfer Objects',
            'vo': 'Value Objects',
            'valueobjects': 'Value Objects',

            # Configuration
            'config': 'Configuration',
            'configuration': 'Configuration',
            'settings': 'Application settings',
            'properties': 'Property files',
            'env': 'Environment files',
            'environment': 'Environment files',
            'conf': 'Configuration',
            'cfg': 'Configuration',
            'ini': 'INI configuration',

            # Utilities & Helpers
            'utils': 'Utility functions',
            'util': 'Utility function',
            'helpers': 'Helper functions',
            'helper': 'Helper function',
            'common': 'Common functionality',
            'shared': 'Shared components',
            'base': 'Base classes',
            'abstract': 'Abstract classes',
            'interfaces': 'Interface definitions',
            'contracts': 'Contract definitions',
            'traits': 'Traits (PHP/Rust)',
            'mixins': 'Mixins',
            'extensions': 'Extensions',
            'plugins': 'Plugin system',
            'plugin': 'Plugin',
            'modules': 'Application modules',
            'module': 'Application module',

            # Frontend Assets
            'assets': 'Static assets',
            'asset': 'Static asset',
            'static': 'Static files',
            'public': 'Public files',
            'styles': 'CSS/Style files',
            'style': 'CSS/Style file',
            'css': 'CSS files',
            'scss': 'Sass/SCSS files',
            'sass': 'Sass files',
            'less': 'Less files',
            'scripts': 'JavaScript files',
            'js': 'JavaScript files',
            'ts': 'TypeScript files',
            'images': 'Image assets',
            'img': 'Image assets',
            'fonts': 'Font files',
            'icons': 'Icon files',
            'media': 'Media files',
            'themes': 'UI themes',
            'theme': 'UI theme',

            # Documentation
            'docs': 'Documentation',
            'doc': 'Documentation',
            'documentation': 'Documentation',
            'wiki': 'Wiki documentation',
            'guide': 'User guide',
            'guides': 'User guides',
            'examples': 'Example code',
            'example': 'Example code',
            'samples': 'Sample code',
            'sample': 'Sample code',
            'tutorials': 'Tutorial content',
            'tutorial': 'Tutorial content',

            # Build & Deployment
            'scripts': 'Build/deploy scripts',
            'tools': 'Build tools',
            'tool': 'Build tool',
            'tasks': 'Build tasks',
            'task': 'Build task',
            'gulp': 'Gulp tasks',
            'grunt': 'Grunt tasks',
            'webpack': 'Webpack config',
            'rollup': 'Rollup config',
            'vite': 'Vite config',
            'babel': 'Babel config',
            'postcss': 'PostCSS config',
            'eslint': 'ESLint config',
            'prettier': 'Prettier config',
            'jest': 'Jest test config',
            'cypress': 'Cypress test config',
            'playwright': 'Playwright test config',

            # DevOps & Infrastructure
            'deploy': 'Deployment scripts',
            'deployment': 'Deployment configuration',
            'docker': 'Docker files',
            'k8s': 'Kubernetes configs',
            'kubernetes': 'Kubernetes configs',
            'helm': 'Helm charts',
            'terraform': 'Terraform configs',
            'ansible': 'Ansible playbooks',
            'ci': 'Continuous Integration',
            'cd': 'Continuous Deployment',
            'pipeline': 'CI/CD pipeline',
            'workflows': 'GitHub Actions workflows',
            'actions': 'GitHub Actions',

            # Language Specific
            'python': 'Python packages',
            'node_modules': 'Node.js dependencies',
            'vendor': 'Third-party code (PHP/Go)',
            'third_party': 'Third-party code',
            'external': 'External libraries',
            'deps': 'Dependencies',
            'dependencies': 'Dependencies',
            'include': 'Header files (C/C++)',
            'inc': 'Header files (C/C++)',
            'srcs': 'Source files (Android)',
            'res': 'Resources (Android)',
            'layout': 'Layout files (Android)',
            'drawable': 'Drawable resources (Android)',
            'mipmap': 'Mipmap resources (Android)',

            # Framework Specific
            'artisan': 'Laravel artisan',
            'blade': 'Blade templates (Laravel)',
            'twig': 'Twig templates (Symfony)',
            'jinja': 'Jinja2 templates (Python)',
            'handlebars': 'Handlebars templates',
            'mustache': 'Mustache templates',
            'erb': 'ERB templates (Ruby)',
            'ejs': 'EJS templates (Node)',
            'pug': 'Pug templates (Node)',
            'hbs': 'Handlebars templates',

            # Command Line
            'cmd': 'Command-line applications',
            'cli': 'CLI tools',
            'bin': 'Executables',
            'command': 'Command definitions',
            'commands': 'Command definitions',

            # Testing
            'unit': 'Unit tests',
            'integration': 'Integration tests',
            'e2e': 'End-to-end tests',
            'functional': 'Functional tests',
            'performance': 'Performance tests',
            'load': 'Load tests',
            'stress': 'Stress tests',
            'security': 'Security tests',
            'acceptance': 'Acceptance tests',
            'regression': 'Regression tests',
            'smoke': 'Smoke tests',

            # Legacy/Special
            'legacy': 'Legacy code',
            'deprecated': 'Deprecated code',
            'temp': 'Temporary files',
            'tmp': 'Temporary files',
            'cache': 'Cache files',
            'log': 'Log files',
            'logs': 'Log files',
            'backup': 'Backup files',
            'archive': 'Archived files'
        }

        return purposes.get(dirname.lower(), 'Unknown')

    async def _analyze_testing(self) -> Dict[str, Any]:
        """Analyze testing setup using FilesystemUtils"""
        testing = {
            'frameworks': [],
            'directories': [],
            'coverage': None
        }

        # Find test directories using FilesystemUtils
        test_dirs = ['tests', 'test', '__tests__', 'spec', 'specs']
        for test_dir in test_dirs:
            if (self.root_path / test_dir).exists() and (self.root_path / test_dir).is_dir():
                testing['directories'].append(test_dir)

        # Detect test frameworks by searching for test files efficiently
        test_patterns = ['*test*.py', '*_test.py', 'test_*.py', '*.test.js', '*.spec.js',
                        '*.test.ts', '*.spec.ts', '*.test.jsx', '*.spec.jsx']

        test_files = []
        for pattern in test_patterns:
            found = await self.fs_utils.find_files(pattern, '.', exclude_patterns=[
                '.git', 'node_modules', '__pycache__', '.pytest_cache', 'coverage', '.nyc_output'
            ])
            test_files.extend(found)

        if test_files:
            # Sample test files to detect framework
            sample = test_files[:5]
            for test_file_path in sample:
                try:
                    # Use FilesystemUtils to read the file
                    rel_path = str(Path(test_file_path).relative_to(self.root_path))
                    file_result = await self.fs_utils.read_file(rel_path)

                    if file_result.error:
                        continue

                    content = file_result.content.lower()

                    if 'pytest' in content or 'from pytest' in content:
                        testing['frameworks'].append('pytest')
                    elif 'unittest' in content or 'from unittest' in content:
                        testing['frameworks'].append('unittest')
                    elif 'jest' in content or 'describe(' in content:
                        testing['frameworks'].append('jest')
                    elif 'mocha' in content:
                        testing['frameworks'].append('mocha')
                    elif 'jasmine' in content:
                        testing['frameworks'].append('jasmine')
                    elif 'cypress' in content:
                        testing['frameworks'].append('cypress')
                    elif 'playwright' in content:
                        testing['frameworks'].append('playwright')
                    elif 'vitest' in content:
                        testing['frameworks'].append('vitest')
                except Exception:
                    pass

        # Check for coverage configuration
        coverage_files = ['.coveragerc', 'coverage.xml', 'lcov.info', 'nyc.config.js',
                          'coverage.json', '.nycrc', '.nycrc.json']
        for cov_file in coverage_files:
            if (self.root_path / cov_file).exists():
                testing['coverage'] = cov_file
                break

        testing['frameworks'] = list(set(testing['frameworks']))
        return testing

    async def _find_configuration(self) -> Dict[str, List[str]]:
        """Find configuration files - Enhanced with comprehensive coverage"""
        config = {
            'runtime': [],
            'build': [],
            'linting': [],
            'formatting': [],
            'environment': [],
            'testing': [],
            'ci_cd': [],
            'docker': [],
            'database': [],
            'security': [],
            'documentation': []
        }

        # Enhanced configuration files mapping
        config_files = {
            # Runtime configuration
            'runtime': [
                'package.json', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
                'requirements.txt', 'requirements-dev.txt', 'requirements-test.txt',
                'requirements/base.txt', 'requirements/local.txt', 'requirements/production.txt',
                'Pipfile', 'Pipfile.lock', 'pyproject.toml', 'poetry.lock', 'pdm.lock',
                'go.mod', 'go.sum', 'go.work', 'go.work.sum',
                'Cargo.toml', 'Cargo.lock', 'rust-toolchain', 'rust-toolchain.toml',
                'pom.xml', 'build.gradle', 'build.gradle.kts', 'gradle.properties',
                'Gemfile', 'Gemfile.lock', 'composer.json', 'composer.lock'
            ],

            # Build configuration
            'build': [
                'Makefile', 'makefile', 'CMakeLists.txt', 'CMakeCache.txt',
                'configure', 'configure.ac', 'Makefile.am', 'Makefile.in',
                'webpack.config.js', 'webpack.config.ts', 'webpack.config.babel.js',
                'vite.config.js', 'vite.config.ts', 'vite.config.mjs',
                'rollup.config.js', 'rollup.config.ts', 'rollup.config.mjs',
                'tsconfig.json', 'tsconfig.build.json', 'tsconfig.app.json',
                'babel.config.js', 'babel.config.json', 'babel.config.mjs',
                '.babelrc', '.babelrc.js', '.babelrc.json',
                'postcss.config.js', 'postcss.config.json', 'postcss.config.ts',
                'tailwind.config.js', 'tailwind.config.ts', 'tailwind.config.mjs',
                'sass.config.js', 'stylus.config.js', 'less.config.js',
                'Gruntfile.js', 'gulpfile.js', 'gulpfile.ts',
                'parcel.config.js', 'parcel.config.json',
                'esbuild.config.js', 'esbuild.config.mjs',
                'snowpack.config.js', 'snowpack.config.mjs',
                'turbo.json', 'nx.json', 'lerna.json', 'rush.json',
                'angular.json', 'nest-cli.json', 'workspace.json',
                'next.config.js', 'nuxt.config.js', 'svelte.config.js',
                'gatsby-config.js', 'gatsby-config.ts',
                'build.xml', 'ant.properties', 'ivy.xml', 'ivysettings.xml'
            ],

            # Linting and code quality
            'linting': [
                '.eslintrc.js', '.eslintrc.json', '.eslintrc.yml', '.eslintrc.yaml',
                'eslint.config.js', 'eslint.config.mjs', 'eslint.config.ts',
                '.eslintrc', '.eslintrc.cjs',
                'pylintrc', '.pylintrc', 'setup.cfg', 'pyproject.toml',
                'ruff.toml', '.ruff.toml', 'ruff.py', '.ruff.py',
                'flake8.cfg', '.flake8', 'tox.ini',
                '.jshintrc', '.jshintignore',
                '.jscsrc', '.jscsrc.json',
                'tslint.json', 'tslint.yaml', 'tslint.yml',
                '.pylintrc', '.pylintrc', 'setup.cfg',
                'phpcs.xml', '.phpcs.xml', 'phpcs.xml.dist', '.phpcs.xml.dist',
                'phpstan.neon', 'phpstan.neon.dist',
                'psalm.xml', 'psalm.xml.dist',
                'golangci.yml', 'golangci.yaml', '.golangci.yml', '.golangci.yaml',
                'clippy.toml', '.clippy.toml',
                'rustfmt.toml', '.rustfmt.toml',
                '.editorconfig', '.editorconfig'
            ],

            # Code formatting
            'formatting': [
                '.prettierrc', '.prettierrc.json', '.prettierrc.yml', '.prettierrc.yaml',
                '.prettierrc.js', '.prettierrc.cjs', '.prettierrc.mjs',
                'prettier.config.js', 'prettier.config.cjs', 'prettierrc.config.js',
                'pyproject.toml', 'setup.cfg', '.flake8',
                '.rustfmt.toml', 'rustfmt.toml',
                '.clang-format', '_clang-format',
                '.dotnet-format.json', '.dotnet-format',
                'format.xml', 'format.xml',
                '.stylelintrc', '.stylelintrc.json', '.stylelintrc.js',
                '.stylelintrc.yml', '.stylelintrc.yaml',
                'stylelint.config.js', 'stylelint.config.cjs', 'stylelint.config.mjs',
                'editorconfig', '.editorconfig'
            ],

            # Environment configuration
            'environment': [
                '.env', '.env.example', '.env.template', '.env.sample', '.env.dist',
                '.env.local', '.env.dev', '.env.development', '.env.test', '.env.testing',
                '.env.staging', '.env.stage', '.env.prod', '.env.production',
                '.env.defaults', '.env.shared',
                'environment.yml', 'environment-dev.yml', 'environment-prod.yml',
                'conda.yaml', 'conda-dev.yaml', 'conda-prod.yaml',
                '.envrc', '.envrc.local', '.envrc.private',
                '.python-version', '.node-version', '.nvmrc', '.ruby-version',
                '.go-version', '.java-version', '.php-version',
                'config/.env', 'config/.env.example',
                'conf/.env', 'conf/.env.example',
                '.env.template', '.env.template.example'
            ],

            # Testing configuration
            'testing': [
                'jest.config.js', 'jest.config.json', 'jest.config.ts', 'jest.config.mjs',
                'jest.config.base.js', 'jest.config.common.js', 'jest.config.e2e.js',
                'jest.setup.js', 'jest.setup.ts',
                'vitest.config.js', 'vitest.config.ts', 'vitest.config.mjs',
                'cypress.config.js', 'cypress.config.ts', 'cypress.json',
                'playwright.config.js', 'playwright.config.ts',
                'testcafe.config.js',
                'wdio.conf.js', 'wdio.conf.ts',
                'nightwatch.conf.js', 'nightwatch.conf.ts',
                'protractor.conf.js',
                'karma.conf.js', 'karma.conf.ts', '.karma.conf.js',
                'jasmine.json', 'jasmine.json',
                'mocha.opts', '.mocharc.json', '.mocharc.js', '.mocharc.yml', '.mocharc.yaml',
                'mocha.setup.js',
                'ava.config.js', 'ava.config.cjs',
                'tap-config.js', 'tape.config.js',
                'pytest.ini', 'pyproject.toml', 'tox.ini', 'setup.cfg',
                'nose.cfg', '.noserc',
                'conftest.py', 'pytest.ini', 'pyproject.toml',
                '.coveragerc', 'coverage.xml', 'lcov.info', 'nyc.config.js', 'nyc.config.json',
                'phpunit.xml', 'phpunit.xml.dist',
                'phpunit.xml', 'phpunit.xml.dist',
                'phpunit.xml', 'phpunit.xml.dist',
                'phpunit.xml', 'phpunit.xml.dist',
                'phpunit.xml', 'phpunit.xml.dist',
                'behat.yml', 'codeception.yml',
                'go.mod', 'go.sum', 'go.work', 'go.work.sum'
            ],

            # CI/CD configuration
            'ci_cd': [
                '.github/workflows', '.github/workflows/',
                '.gitlab-ci.yml', '.gitlab-ci.yaml', 'gitlab-ci.yml', 'gitlab-ci.yaml',
                '.gitlab-ci',
                '.travis.yml', 'travis.yml',
                'appveyor.yml', '.appveyor.yml',
                'circle.yml', '.circleci', '.circleci/config.yml',
                'codeship-services.yml', 'codeship-steps.yml',
                'azure-pipelines.yml', 'azure-pipelines.yaml',
                'bitbucket-pipelines.yml',
                'buildkite.yml', 'buildkite.yaml', '.buildkite',
                'drone.yml', 'drone.yaml', '.drone.yml',
                'semaphore.yml', 'semaphore.yaml',
                'snapcraft.yml', 'snapcraft.yaml',
                'fastlane/Fastfile', 'fastlane/Appfile',
                'Jenkinsfile', 'Jenkinsfile.groovy', 'jenkins.yaml', 'jenkins.yml', '.jenkins',
                'tekton/', 'tekton.yaml', 'tekton.yml',
                'argo/', 'argo.yaml', 'argo.yml',
                'prow.yaml', 'fleet.yaml',
                'cloudbuild.yaml', 'cloudbuild.yml',
                '.buildkite', 'buildkite.yml', 'buildkite.yaml',
                'cirrus.yml', '.cirrus.yml',
                'appveyor.yml', '.appveyor.yml',
                'wercker.yml', '.wercker.yml',
                'codeship-steps.yml', 'codeship-services.yml',
                'buddy.yml', 'buddy.yaml',
                'shippable.yml', '.shippable.yml',
                'screwdriver.yaml',
                'teamcity-settings.kts',
                'bitrise.yml', '.bitrise.yml',
                'scratch.yml', '.scratch.yml'
            ],

            # Docker configuration
            'docker': [
                'Dockerfile', 'Dockerfile.prod', 'Dockerfile.production', 'Dockerfile.dev',
                'Dockerfile.development', 'Dockerfile.test', 'Dockerfile.testing',
                'Dockerfile.ci', 'Dockerfile.local', 'Dockerfile.base', 'Dockerfile.builder',
                'Dockerfile.runtime', 'docker-compose.yml', 'docker-compose.yaml',
                'docker-compose.override.yml', 'docker-compose.prod.yml', 'docker-compose.production.yml',
                'docker-compose.dev.yml', 'docker-compose.development.yml', 'docker-compose.test.yml',
                'docker-compose.testing.yml', 'docker-compose.ci.yml', 'docker-compose.local.yml',
                'docker-compose.localprod.yml', 'docker-compose.staging.yml', '.dockerignore',
                'Containerfile', 'docker-compose.yml.dist', 'docker-compose.yaml.dist',
                '.docker', '.dockerenv'
            ],

            # Database configuration
            'database': [
                'migrations/', 'migrate/', 'db/migrate/', 'db/migrations/', 'sql/',
                'schema.sql', 'database.sql', 'init.sql', 'seed.sql', 'seeds/', 'db/seeds/', 'fixtures/',
                'schema.prisma', 'prisma/schema.prisma',
                'drizzle.config.ts', 'typeorm.config.ts', 'sequelize.config.js', 'knexfile.js',
                'alembic.ini', 'alembic/', 'database.yml', 'database.yaml',
                'mongoid.yml', 'redis.conf', 'redis/redis.conf', 'elasticsearch.yml',
                'neo4j.conf', 'cassandra.yaml', 'influxdb.conf',
                'my.cnf', 'postgresql.conf', 'pg_hba.conf'
            ],

            # Security configuration
            'security': [
                '.htaccess', '.htpasswd', 'web.config',
                'security.yml', 'security.yaml', 'security.xml',
                'auth.json', '.auth.json',
                'ssh_config', 'sshd_config',
                'openssl.cnf',
                'cert.pem', 'key.pem', 'ca.pem',
                'secrets.yml', 'secrets.yaml',
                'vault.yml', 'vault.yaml',
                'ansible-vault',
                'k8s/secrets/', 'kubernetes/secrets/',
                'security/', '.security/',
                'bandit.yaml', '.bandit',
                'audit.toml', 'deny.toml',
                'snyk.json', '.snyk',
                'sonar-project.properties',
                'codeql.yml', '.github/codeql'
            ],

            # Documentation configuration
            'documentation': [
                'mkdocs.yml', 'mkdocs.yaml',
                'docusaurus.config.js', 'docusaurus.config.ts',
                'vuepress.config.js', 'vuepress.config.ts',
                'vitepress.config.js', 'vitepress.config.ts',
                'gridsome.config.js', 'gridsome.config.ts',
                'gatsby-config.js', 'gatsby-config.ts',
                'storybook/', '.storybook/',
                'stencil.config.ts', 'stencil.config.js',
                'typedoc.json', 'typedoc.js',
                'jsdoc.conf.json', 'jsdoc.json',
                'javadoc.yml', 'javadoc.xml',
                'pydoc.toml', 'pyproject.toml',
                'sphinx.conf.py', 'docs/conf.py',
                'hugo.toml', 'hugo.yaml', 'hugo.yml', 'config.toml', 'config.yaml', 'config.yml',
                'pelicanconf.py',
                'jekyll.rb', '_config.yml',
                'config.rb', 'nanoc.yaml',
                'docsy/config.yaml',
                'antora.yml', 'antora.yaml',
                'docsify/_sidebar.md',
                'gitbook.yaml',
                'vue.json',
                'swagger.yaml', 'swagger.yml', 'swagger.json',
                'openapi.yaml', 'openapi.yml', 'openapi.json',
                'raml.yaml', 'raml.yml',
                'api-blueprint.md'
            ]
        }

        # Scan for configuration files with subdirectory support
        async def scan_config_files(files: List[str], category: str) -> List[str]:
            """Scan for configuration files in root and common subdirectories"""
            found = []

            # Common subdirectories to check
            subdirs = ['', 'config/', 'conf/', 'configs/', '.config/', 'etc/', 'settings/']

            for file_name in files:
                # Check root and subdirectories
                for subdir in subdirs:
                    full_path = self.root_path / subdir / file_name

                    if file_name.endswith('/'):
                        # Directory pattern
                        if full_path.exists() and full_path.is_dir():
                            found.append(f"{subdir}{file_name}")
                    else:
                        # File pattern
                        if full_path.exists() and full_path.is_file():
                            found.append(f"{subdir}{file_name}")
                            break  # Found it, no need to check other subdirs

            return found

        # Scan each category
        for category, files in config_files.items():
            config[category] = await scan_config_files(files, category)

        return config

    async def _find_documentation(self) -> Dict[str, Any]:
        """Find documentation files - Enhanced with comprehensive coverage"""
        docs = {
            'files': [],
            'directories': [],
            'apis': [],
            'guides': [],
            'changelogs': [],
            'licenses': [],
            'contributing': [],
            'architecture': [],
            'deployment': []
        }

        # Enhanced documentation files mapping
        doc_files = {
            # General documentation
            'README': ['README.md', 'README.rst', 'README.txt', 'README', 'readme.md', 'Readme.md', 'README.MD'],
            'OVERVIEW': ['OVERVIEW.md', 'OVERVIEW.rst', 'overview.md', 'ABOUT.md', 'about.md'],
            'INTRODUCTION': ['INTRODUCTION.md', 'INTRO.md', 'introduction.md', 'intro.md', 'GETTING_STARTED.md', 'GETTING-STARTED.md'],

            # Changelog and releases
            'CHANGELOG': ['CHANGELOG.md', 'CHANGELOG.rst', 'CHANGELOG.txt', 'CHANGELOG', 'changelog.md', 'Changelog.md', 'CHANGES.md', 'CHANGES.rst', 'CHANGES.txt', 'HISTORY.md', 'HISTORY.rst', 'RELEASES.md', 'RELEASES.rst', 'RELEASE-NOTES.md', 'RELEASE-NOTES.txt', 'RELEasenotes.md', 'RELEASENOTES.md', 'NEWS.md', 'news.txt'],
            'VERSION': ['VERSION.md', 'version.md', 'VERSION.txt', 'VERSION', 'version.txt'],

            # Contributing and community
            'CONTRIBUTING': ['CONTRIBUTING.md', 'CONTRIBUTING.rst', 'CONTRIBUTING.txt', 'CONTRIBUTING', 'contributing.md', 'Contributing.md', 'CONTRIBUTE.md', 'CONTRIBUTE.rst', 'CONTRIBUTING-guide.md', 'CONTRIBUTING_GUIDE.md'],
            'CODE_OF_CONDUCT': ['CODE_OF_CONDUCT.md', 'CODE_OF_CONDUCT.rst', 'CODE_OF_CONDUCT.txt', 'CODE_OF_CONDUCT', 'code_of_conduct.md', 'CONDUCT.md', 'conduct.md', 'CODE_CONDUCT.md'],
            'GOVERNANCE': ['GOVERNANCE.md', 'GOVERNANCE.rst', 'governance.md'],
            'SUPPORT': ['SUPPORT.md', 'support.md', 'HELP.md', 'help.md', 'TROUBLESHOOTING.md', 'troubleshooting.md', 'FAQ.md', 'faq.md', 'FAQS.md', 'faqs.md'],
            'CONTACT': ['CONTACT.md', 'contact.md', 'CONTACT.rst', 'CONTACT.txt'],

            # Architecture and design
            'ARCHITECTURE': ['ARCHITECTURE.md', 'ARCHITECTURE.rst', 'ARCHITECTURE.txt', 'ARCHITECTURE', 'architecture.md', 'Architecture.md', 'DESIGN.md', 'design.md', 'DESIGN.md', 'DESIGN.rst', 'SYSTEM_DESIGN.md', 'system-design.md', 'SYSTEM_DESIGN.rst', 'TECHNICAL_DESIGN.md', 'technical-design.md'],
            'DESIGN_DOCUMENTS': ['DESIGN.md', 'DESIGN.rst', 'DESIGN_DOCS.md', 'design-docs.md', 'DESIGN_DOCUMENTS.md', 'design-documents.md', 'ADR.md', 'ARCHITECTURE_DECISION_RECORDS.md', 'adrs/', 'adr/', 'docs/adr/', 'docs/adrs/'],
            'API_DESIGN': ['API_DESIGN.md', 'api-design.md', 'API_SPECIFICATION.md', 'api-specification.md', 'API Blueprint.md', 'api-blueprint.md'],

            # API documentation
            'API': ['API.md', 'API.rst', 'API.txt', 'API', 'api.md', 'Api.md', 'API_GUIDE.md', 'api-guide.md', 'API_REFERENCE.md', 'api-reference.md', 'API_DOCS.md', 'api-docs.md', 'REST_API.md', 'rest-api.md', 'GRAPHQL_API.md', 'graphql-api.md', 'OPENAPI.md', 'openapi.md', 'SWAGGER.md', 'swagger.md'],

            # Deployment and operations
            'DEPLOYMENT': ['DEPLOYMENT.md', 'DEPLOYMENT.rst', 'DEPLOYMENT.txt', 'DEPLOYMENT', 'deployment.md', 'Deployment.md', 'DEPLOY.md', 'deploy.md', 'INSTALL.md', 'INSTALL.rst', 'INSTALL.txt', 'INSTALL', 'install.md', 'INSTALLATION.md', 'installation.md', 'SETUP.md', 'setup.md', 'PROVISIONING.md', 'provisioning.md'],
            'PRODUCTION': ['PRODUCTION.md', 'production.md', 'PRODUCTION_GUIDE.md', 'production-guide.md', 'LIVE.md', 'live.md', 'STAGING.md', 'staging.md'],
            'INFRASTRUCTURE': ['INFRASTRUCTURE.md', 'infrastructure.md', 'INFRA.md', 'infra.md', 'OPS.md', 'ops.md', 'OPERATIONS.md', 'operations.md', 'DEVOPS.md', 'devops.md'],
            'MONITORING': ['MONITORING.md', 'monitoring.md', 'OBSERVABILITY.md', 'observability.md', 'LOGGING.md', 'logging.md', 'ALERTING.md', 'alerting.md'],

            # Development guides
            'DEVELOPMENT': ['DEVELOPMENT.md', 'development.md', 'DEV_GUIDE.md', 'dev-guide.md', 'DEVELOPER_GUIDE.md', 'developer-guide.md', 'HACKING.md', 'hacking.md', 'DEVELOPERS.md', 'developers.md'],
            'LOCAL_DEVELOPMENT': ['LOCAL_DEVELOPMENT.md', 'local-development.md', 'LOCAL_SETUP.md', 'local-setup.md', 'DEVELOPMENT_ENVIRONMENT.md', 'development-environment.md'],
            'DEBUGGING': ['DEBUGGING.md', 'debugging.md', 'TROUBLESHOOTING.md', 'troubleshooting.md', 'DEBUG.md', 'debug.md'],
            'TESTING': ['TESTING.md', 'testing.md', 'TESTS.md', 'tests.md', 'TEST_GUIDE.md', 'test-guide.md', 'TEST_COVERAGE.md', 'test-coverage.md'],

            # Configuration and environment
            'CONFIGURATION': ['CONFIGURATION.md', 'configuration.md', 'CONFIG.md', 'config.md', 'SETTINGS.md', 'settings.md', 'ENVIRONMENT.md', 'environment.md', 'ENV_SETUP.md', 'env-setup.md'],
            'ENVIRONMENT_VARIABLES': ['ENVIRONMENT_VARIABLES.md', 'environment-variables.md', 'ENV_VARS.md', 'env-vars.md', 'ENVIRONMENT_VARIABLES.txt', 'environment-variables.txt'],

            # Security
            'SECURITY': ['SECURITY.md', 'security.md', 'SECURITY.rst', 'SECURITY.txt', 'SECURITY', 'security.txt', 'SECURITY_POLICY.md', 'security-policy.md', 'VULNERABILITY.md', 'vulnerability.md', 'VULNERABILITIES.md', 'vulnerabilities.md'],
            'AUTHENTICATION': ['AUTHENTICATION.md', 'authentication.md', 'AUTH.md', 'auth.md', 'AUTHORIZATION.md', 'authorization.md'],

            # Performance and optimization
            'PERFORMANCE': ['PERFORMANCE.md', 'performance.md', 'OPTIMIZATION.md', 'optimization.md', 'PERFORMANCE_GUIDE.md', 'performance-guide.md', 'BENCHMARKS.md', 'benchmarks.md'],
            'SCALING': ['SCALING.md', 'scaling.md', 'SCALE.md', 'scale.md', 'SCALABILITY.md', 'scalability.md'],

            # Legal and licensing
            'LICENSE': ['LICENSE', 'LICENSE.md', 'LICENSE.txt', 'LICENCE', 'LICENCE.md', 'LICENCE.txt', 'COPYING', 'COPYING.md', 'COPYRIGHT.md', 'copyright.md', 'PATENTS.md', 'patents.md', 'TRADEMARKS.md', 'trademarks.md'],
            'LEGAL': ['LEGAL.md', 'legal.md', 'LEGAL_NOTICES.md', 'legal-notices.md', 'TERMS.md', 'terms.md', 'TERMS_OF_SERVICE.md', 'terms-of-service.md', 'PRIVACY.md', 'privacy.md', 'PRIVACY_POLICY.md', 'privacy-policy.md'],

            # Tutorials and examples
            'TUTORIAL': ['TUTORIAL.md', 'tutorial.md', 'TUTORIALS.md', 'tutorials.md', 'EXAMPLES.md', 'examples.md', 'EXAMPLE.md', 'example.md', 'SAMPLES.md', 'samples.md', 'SAMPLE.md', 'sample.md', 'COOKBOOK.md', 'cookbook.md', 'RECIPES.md', 'recipes.md'],
            'WALKTHROUGH': ['WALKTHROUGH.md', 'walkthrough.md', 'WALK_THROUGH.md', 'walk-through.md', 'GUIDE.md', 'guide.md', 'GUIDES.md', 'guides.md'],
            'QUICK_START': ['QUICK_START.md', 'quick-start.md', 'QUICKSTART.md', 'quickstart.md', 'QUICK-START.md', 'QUICK-START.rst', 'QUICK_GUIDE.md', 'quick-guide.md'],
            'GETTING_STARTED': ['GETTING_STARTED.md', 'getting-started.md', 'GETTING-STARTED.md', 'GETTING-STARTED.rst'],

            # Reference documentation
            'REFERENCE': ['REFERENCE.md', 'reference.md', 'REFERENCE_GUIDE.md', 'reference-guide.md', 'MANUAL.md', 'manual.md', 'DOCUMENTATION.md', 'documentation.md', 'DOCS.md', 'docs.md'],
            'GLOSSARY': ['GLOSSARY.md', 'glossary.md', 'TERMINOLOGY.md', 'terminology.md'],
            'FAQ': ['FAQ.md', 'faq.md', 'FAQS.md', 'faqs.md', 'FAQ.txt', 'faq.txt', 'FREQUENTLY_ASKED_QUESTIONS.md', 'frequently-asked-questions.md'],

            # Project management
            'ROADMAP': ['ROADMAP.md', 'roadmap.md', 'ROADMAP.rst', 'ROADMAP.txt', 'MILESTONES.md', 'milestones.md', 'TIMELINE.md', 'timeline.md'],
            'PROJECT_PLAN': ['PROJECT_PLAN.md', 'project-plan.md', 'PROJECT.md', 'project.md'],
            'REQUIREMENTS': ['REQUIREMENTS.md', 'requirements.md', 'REQUIREMENTS.rst', 'SPECIFICATIONS.md', 'specifications.md', 'SPECS.md', 'specs.md'],
            'CHANGELOG': ['CHANGELOG.md', 'CHANGELOG.rst', 'CHANGELOG.txt', 'CHANGELOG', 'changelog.md', 'Changelog.md', 'CHANGES.md', 'CHANGES.rst', 'CHANGES.txt', 'HISTORY.md', 'HISTORY.rst', 'RELEASES.md', 'RELEASES.rst', 'RELEASE-NOTES.md', 'RELEASE-NOTES.txt', 'RELEasenotes.md', 'RELEASENOTES.md', 'NEWS.md', 'news.txt'],

            # Documentation directories
            'docs_directories': ['docs/', 'doc/', 'documentation/', 'guides/', 'tutorials/', 'examples/', 'reference/', 'api/', 'wiki/', '_docs/', '.docs/']
        }

        # Enhanced documentation subdirectories to check
        doc_subdirs = [
            '', 'docs/', 'doc/', 'documentation/', 'guides/', 'tutorials/', 'examples/',
            'reference/', 'api/', 'wiki/', '_docs/', '.docs/', 'source/', 'content/',
            'assets/', 'static/', 'media/', 'images/', 'img/', 'assets/docs/'
        ]

        # Scan for documentation files
        async def scan_doc_files(files: List[str], category: str) -> List[str]:
            """Scan for documentation files in root and subdirectories"""
            found = []

            for file_name in files:
                # Check root and subdirectories
                for subdir in doc_subdirs:
                    full_path = self.root_path / subdir / file_name

                    if full_path.exists() and full_path.is_file():
                        # Add with subdirectory prefix if not root
                        full_doc_path = f"{subdir}{file_name}" if subdir else file_name
                        found.append(full_doc_path)
                        break  # Found it, no need to check other subdirs

            return found

        # Scan each category
        for category, files in doc_files.items():
            if category == 'docs_directories':
                # Handle directories separately
                for dir_name in files:
                    full_path = self.root_path / dir_name.rstrip('/')
                    if full_path.exists() and full_path.is_dir():
                        docs['directories'].append(dir_name)
                        # Scan for files in this directory
                        try:
                            for ext in ['*.md', '*.rst', '*.txt', '*.adoc', '*.asciidoc']:
                                for file_path in full_path.glob(ext):
                                    if file_path.is_file():
                                        relative_path = f"{dir_name}{file_path.name}"
                                        docs['files'].append(relative_path)
                        except Exception:
                            pass
            else:
                # Scan files for this category
                found_files = await scan_doc_files(files, category)
                docs['files'].extend(found_files)

                # Categorize files for easier access
                if category in ['README', 'OVERVIEW', 'INTRODUCTION']:
                    docs['guides'].extend(found_files)
                elif category in ['API', 'API_DESIGN', 'API_GUIDE', 'API_REFERENCE', 'API_DOCS']:
                    docs['apis'].extend(found_files)
                elif category in ['CHANGELOG', 'VERSION', 'HISTORY', 'RELEASES']:
                    docs['changelogs'].extend(found_files)
                elif category in ['LICENSE', 'LEGAL', 'COPYING', 'COPYRIGHT']:
                    docs['licenses'].extend(found_files)
                elif category in ['CONTRIBUTING', 'CODE_OF_CONDUCT', 'GOVERNANCE']:
                    docs['contributing'].extend(found_files)
                elif category in ['ARCHITECTURE', 'DESIGN_DOCUMENTS', 'SYSTEM_DESIGN']:
                    docs['architecture'].extend(found_files)
                elif category in ['DEPLOYMENT', 'INSTALLATION', 'SETUP', 'PRODUCTION']:
                    docs['deployment'].extend(found_files)

        # Remove duplicates while preserving order
        for key in ['files', 'directories', 'apis', 'guides', 'changelogs', 'licenses', 'contributing', 'architecture', 'deployment']:
            if key in docs:
                docs[key] = list(dict.fromkeys(docs[key]))

        # Limit the number of files returned to avoid overwhelming output
        for key in ['files', 'apis', 'guides']:
            if key in docs and len(docs[key]) > 20:
                docs[key] = docs[key][:20]

        return docs

    # ===== OPTIMIZED METHODS USING FILECACHE =====

    async def map_architecture_optimized(self, fingerprint: CodebaseFingerprint) -> Dict[str, Any]:
        """Optimized architecture mapping using FileCache and parallel execution"""
        print("[StructuralMapper] Starting optimized architecture mapping...")

        # Run all analyses in parallel for maximum performance
        tasks = await asyncio.gather(
            self._detect_patterns_optimized(),
            self._analyze_conventions_optimized(),
            self._map_modules_optimized(fingerprint),
            self._analyze_testing_optimized(),
            self._find_configuration_optimized(),
            self._find_documentation_optimized(),
            return_exceptions=True
        )

        # Handle results and exceptions
        architecture = {
            'patterns': tasks[0] if not isinstance(tasks[0], Exception) else [],
            'conventions': tasks[1] if not isinstance(tasks[1], Exception) else {},
            'modules': tasks[2] if not isinstance(tasks[2], Exception) else {},
            'tests': tasks[3] if not isinstance(tasks[3], Exception) else {},
            'configuration': tasks[4] if not isinstance(tasks[4], Exception) else {},
            'documentation': tasks[5] if not isinstance(tasks[5], Exception) else {}
        }

        print("[StructuralMapper] Optimized architecture mapping completed")
        return architecture

    async def _detect_patterns_optimized(self) -> List[str]:
        """Optimized pattern detection using FileCache"""
        print("[StructuralMapper] Detecting patterns (optimized)...")
        patterns = []

        # Multi-language architecture detection using cache
        directories = self.cache.get_directories()
        language_dirs = set()
        for item in directories:
            # Check for language-specific directories
            if item.lower() in ['gobase', 'go-code', 'golang', 'go-src',
                                'py-src', 'python-src', 'pycode',
                                'js-src', 'javascript', 'typescript',
                                'java-src', 'jsource', 'kotlin-src',
                                'rs-src', 'rust-src', 'csharp-src']:
                language_dirs.add(item)

        if language_dirs:
            patterns.append("Multi-language Architecture")

        # Microservices patterns detection
        microservices_indicators = ['services/', 'microservices/', 'apis/', 'src/services/']
        for indicator in microservices_indicators:
            if indicator in directories:
                patterns.append("Microservices Architecture")
                break

        # Layered architecture detection
        layered_indicators = ['controllers/', 'models/', 'views/', 'services/', 'repositories/']
        found_layers = sum(1 for indicator in layered_indicators if indicator in directories)
        if found_layers >= 2:
            patterns.append("Layered/N-tier Architecture")

        # Common patterns detection using cache
        pattern_files = [
            ('src/main/java', 'Java/Maven Structure'),
            ('src/app/', 'Flutter/Dart Structure'),
            ('app/Http/', 'Laravel Structure'),
            ('src/components/', 'React/Vue Component Structure'),
            ('lib/', 'Python Package Structure'),
            ('cmd/', 'Go Command Structure'),
            ('internal/', 'Go Internal Structure'),
            ('pkg/', 'Go Package Structure'),
            ('static/', 'Web Static Assets'),
            ('public/', 'Web Public Assets'),
            ('config/', 'Configuration Management'),
            ('configs/', 'Configuration Management'),
            ('infrastructure/', 'Infrastructure as Code'),
            ('deploy/', 'Deployment Configuration'),
            ('scripts/', 'Build/Deploy Scripts')
        ]

        for path, pattern in pattern_files:
            if path in directories or any(d.startswith(path) for d in directories):
                patterns.append(pattern)

        print(f"[StructuralMapper] Found {len(patterns)} patterns")
        return patterns[:15]  # Limit results

    async def _analyze_conventions_optimized(self) -> Dict[str, str]:
        """Optimized convention analysis using FileCache"""
        print("[StructuralMapper] Analyzing conventions (optimized)...")
        conventions = {}

        # Analyze directory naming conventions using cache
        directories = self.cache.get_directories()
        kebab_case = sum(1 for d in directories if '-' in d)
        snake_case = sum(1 for d in directories if '_' in d)
        camel_case = sum(1 for d in directories if any(c.isupper() for c in d[1:]))

        if kebab_case > snake_case and kebab_case > camel_case:
            conventions['directory_naming'] = 'kebab-case'
        elif snake_case > kebab_case and snake_case > camel_case:
            conventions['directory_naming'] = 'snake_case'
        elif camel_case > 0:
            conventions['directory_naming'] = 'camelCase/PascalCase'
        else:
            conventions['directory_naming'] = 'mixed'

        # Analyze file naming using cache
        all_files = [f for d in directories for f in self.cache.get_files_in_subdir(d)]
        file_extensions = {}
        for file_path in all_files:
            if '.' in file_path:
                ext = file_path.split('.')[-1].lower()
                file_extensions[ext] = file_extensions.get(ext, 0) + 1

        if file_extensions:
            most_common_ext = max(file_extensions, key=file_extensions.get)
            conventions['primary_language'] = most_common_ext

        # Check for common convention files using cache
        convention_files = {
            '.editorconfig': 'Editor Config',
            '.eslintrc.js': 'ESLint',
            '.prettierrc': 'Prettier',
            'pyproject.toml': 'Python Project Config',
            'rustfmt.toml': 'Rust Format'
        }

        found_conventions = []
        for file_name, convention in convention_files.items():
            if self.cache.has_file(file_name):
                found_conventions.append(convention)

        if found_conventions:
            conventions['tools'] = ', '.join(found_conventions)

        print("[StructuralMapper] Convention analysis completed")
        return conventions

    async def _map_modules_optimized(self, fingerprint: CodebaseFingerprint) -> Dict[str, Any]:
        """Optimized module mapping using FileCache"""
        print("[StructuralMapper] Mapping modules (optimized)...")
        modules = {}

        # Patterns to exclude (dependencies, cache, generated files)
        exclude_patterns = [
            '__pycache__', 'node_modules', 'site-packages', '.git',
            'dist', 'build', '.pytest_cache', '.coverage', 'htmlcov',
            '.tox', '.venv', 'venv', 'env', '.env', '.idea', '.vscode',
            '.mypy_cache', 'migrations'
        ]

        # Check if directory should be excluded
        def should_exclude(directory: str) -> bool:
            dir_lower = directory.lower()
            for pattern in exclude_patterns:
                if pattern in dir_lower or dir_lower.endswith('/__pycache__'):
                    return True
            # Exclude paths containing 'lib/python' (virtual env packages)
            if 'lib/python' in dir_lower and 'site-packages' in dir_lower:
                return True
            return False

        # Map modules using cache
        directories = self.cache.get_directories()
        for directory in sorted(directories):
            if not directory.startswith('.') and not should_exclude(directory):
                files_in_dir = self.cache.get_files_in_subdir(directory)
                file_count = len(files_in_dir)

                if file_count > 0:  # Only include non-empty directories
                    modules[directory] = {
                        'file_count': file_count,
                        'has_tests': any('test' in f.lower() for f in files_in_dir),
                        'has_config': any(f.endswith(('.json', '.yaml', '.yml', '.toml', '.ini', '.conf'))
                                       for f in files_in_dir)
                    }

        # Limit to top modules to avoid overwhelming output
        if len(modules) > 50:
            # Sort by file count and keep top 50
            modules = dict(sorted(modules.items(),
                                key=lambda x: x[1]['file_count'], reverse=True)[:50])

        print(f"[StructuralMapper] Mapped {len(modules)} modules")
        return modules

    async def _analyze_testing_optimized(self) -> Dict[str, Any]:
        """Optimized testing analysis using FileCache"""
        print("[StructuralMapper] Analyzing testing setup (optimized)...")
        testing = {
            'frameworks': [],
            'directories': [],
            'files': []
        }

        # Find test directories using cache
        test_dirs = ['tests/', 'test/', '__tests__/', 'spec/', 'specs/']
        for test_dir in test_dirs:
            if test_dir in self.cache.get_directories():
                testing['directories'].append(test_dir)

        # Find test files using cache
        test_patterns = ['test_*.py', '*_test.py', '*.test.js', '*.spec.js', '*_test.go', '*_test.rs']
        for pattern in test_patterns:
            matching_files = self.cache.get_files_by_pattern(pattern)
            testing['files'].extend(matching_files[:10])  # Limit files

        # Detect test frameworks using cache
        test_framework_files = {
            'pytest.ini': 'Pytest',
            'jest.config.js': 'Jest',
            'karma.conf.js': 'Karma',
            'testcafe.config.js': 'TestCafe'
        }

        for file_name, framework in test_framework_files.items():
            if self.cache.has_file(file_name):
                testing['frameworks'].append(framework)

        # Remove duplicates and limit results
        testing['files'] = list(dict.fromkeys(testing['files']))[:20]
        testing['directories'] = list(dict.fromkeys(testing['directories']))

        print(f"[StructuralMapper] Found {len(testing['frameworks'])} test frameworks")
        return testing

    async def _find_configuration_optimized(self) -> Dict[str, Any]:
        """Optimized configuration detection using FileCache"""
        print("[StructuralMapper] Finding configuration (optimized)...")
        configuration = {
            'files': [],
            'formats': {}
        }

        # Important configuration files (prioritized)
        important_config_files = [
            'package.json', 'tsconfig.json', 'pyproject.toml', 'Cargo.toml',
            'go.mod', 'pom.xml', 'build.gradle', 'composer.json',
            'docker-compose.yml', 'docker-compose.yaml', 'Dockerfile',
            '.env.example', '.env', 'config.yaml', 'config.yml',
            'settings.py', 'settings.js', 'app.config.js', 'app.config.ts'
        ]

        # Directories to exclude
        exclude_dirs = [
            'node_modules', 'site-packages', '__pycache__', '.git',
            'dist', 'build', 'coverage', '.pytest_cache', 'tests',
            'test', 'spec', 'specs', 'mocks', 'fixtures'
        ]

        # Check for important config files first
        for config_file in important_config_files:
            files = self.cache.get_files_by_name(config_file)
            for file_path in files:
                if not any(exclude in file_path.lower() for exclude in exclude_dirs):
                    configuration['files'].append(file_path)
                    ext = file_path.split('.')[-1].lower()
                    configuration['formats'][ext] = configuration['formats'].get(ext, 0) + 1

        # Add other config files but be selective
        config_extensions = {'.yaml', '.yml', '.toml', '.ini', '.conf', '.cfg', '.env'}
        for ext in config_extensions:
            files = self.cache.get_files_by_extension(ext)
            for file_path in files[:20]:  # Limit per extension
                file_name = file_path.split('/')[-1].lower()
                # Exclude test configs, example configs, and dependency directories
                if (not any(exclude in file_path.lower() for exclude in exclude_dirs) and
                    not file_name.startswith('test') and
                    not file_name.startswith('example') and
                    not file_name.startswith('sample') and
                    not 'license' in file_name and
                    not 'changelog' in file_name):
                    configuration['files'].append(file_path)
                    configuration['formats'][ext.lstrip('.')] = configuration['formats'].get(ext.lstrip('.'), 0) + 1

        # Limit total results to avoid bloating
        configuration['files'] = configuration['files'][:30]

        print(f"[StructuralMapper] Found {len(configuration['files'])} configuration files")
        return configuration

    async def _find_documentation_optimized(self) -> Dict[str, Any]:
        """Optimized documentation detection using FileCache"""
        print("[StructuralMapper] Finding documentation (optimized)...")
        docs = {
            'files': [],
            'directories': [],
            'changelogs': [],
            'contributing': [],
            'architecture': [],
            'deployment': []
        }

        # Documentation files using cache
        doc_extensions = ['.md', '.rst', '.txt', '.adoc', '.asciidoc']
        doc_files = []

        for ext in doc_extensions:
            files = self.cache.get_files_by_extension(ext)
            doc_files.extend(files)

        # Filter and categorize documentation files
        for file_path in doc_files[:100]:  # Limit to 100 files
            file_name = file_path.split('/')[-1].lower()

            # Skip license files and dependency docs
            if ('license' in file_name or 'licence' in file_name or
                'site-packages' in file_path or 'node_modules' in file_path):
                continue

            docs['files'].append(file_path)

            # Categorize by filename
            if 'contributing' in file_name:
                docs['contributing'].append(file_path)
            elif 'changelog' in file_name or 'history' in file_name:
                docs['changelogs'].append(file_path)
            elif 'architecture' in file_name or 'design' in file_name:
                docs['architecture'].append(file_path)
            elif 'deploy' in file_name or 'install' in file_name:
                docs['deployment'].append(file_path)

        # Documentation directories using cache
        doc_directories = ['docs/', 'doc/', 'documentation/']
        for doc_dir in doc_directories:
            if doc_dir in self.cache.get_directories():
                docs['directories'].append(doc_dir)

        # Remove duplicates and limit results
        for key in ['files', 'changelogs', 'contributing', 'architecture', 'deployment']:
            if key in docs and docs[key]:
                docs[key] = list(dict.fromkeys(docs[key]))[:20]

        print(f"[StructuralMapper] Found {len(docs['files'])} documentation files")
        return docs


class ReconnaissanceAgent:
    """
    Advanced reconnaissance agent implementing adaptive layered analysis with intelligent optimization
    """

    def __init__(self, model: str = "gemini-2.5-flash", reasoning_effort: str = "medium", fingerprint: Optional[CodebaseFingerprint] = None, architecture: Optional[Dict[str, Any]] = None):
        # Use provided model or default
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.config = ReconConfig()

        print(f"🤖 Initializing Enhanced ReconnaissanceAgent with model: {self.model}")

        # Initialize LLM provider
        self.llm = QrooperLLM(desc="ReconnaissanceAgent", model=self.model, reasoning_effort=self.reasoning_effort)

        # Simple logger for critical errors only
        self.logger = logging.getLogger("ReconnaissanceAgent")

        # Smart context management - track visited files/directories to prevent redundancy
        self.visited_files = set()
        self.visited_directories = set()
        self.visited_patterns = set()

        
        # Initialize Context Manager for intelligent context compression
        self.context_manager = ContextManagerAgent(
            desc="ReconnaissanceContextManager",
            model=self.model,
            reasoning_effort=self.reasoning_effort
        )

        # Context management settings
        self.compressed_context = None
        self.last_compression_iteration = 0

        # Dynamic exploration strategy settings - simplified

        # Define the completed() tool specifically for reconnaissance, to terminate loop
        completed_tool = {
            "name": "completed",
            "description": """Mark this step as completed with a summary.

CALL THIS WHEN:
- You have gathered sufficient information for this step
- You've read relevant files and understand the structure
- Further exploration would be redundant
- You can answer the step's question

DO NOT explore endlessly. Call completed() to progress to the next step.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A concise summary of findings for this step (2-4 sentences)"
                    }
                },
                "required": ["summary"]
            }
        }

        # Combine available tools with completed() tool
        self.available_tools = oai_compatible_filesystemtools + oai_compatible_asttools + [completed_tool]
        # Max iterations for exploration loop
        self.max_iterations = 20

        # Store pre-computed fingerprint and architecture if provided
        self.fingerprint = fingerprint
        self.architecture = architecture


    async def analyze(self, query: str, root_path: str) -> ReconnaissanceResult:
        """Perform comprehensive reconnaissance analysis with looped LLM+tools architecture"""

        self.logger.info("RECON ANALYZE STARTING...")
        self.logger.debug(f"Analyzing path: {root_path}")
        self.logger.debug(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")

        start_time = time.time()
        phases = []

        # Check if fingerprint and architecture are pre-computed
        if self.fingerprint is None or self.architecture is None:
            raise ValueError("Fingerprint and architecture must be provided during initialization")

        fingerprint = self.fingerprint
        architecture = self.architecture

        # Add pre-computed phases to tracking
        phases.append(ReconPhase(
            name="Lightning Scan (Pre-computed)",
            duration=0.0,
            findings={"fingerprint": fingerprint.model_dump()}
        ))

        phases.append(ReconPhase(
            name="Structural Mapping (Pre-computed)",
            duration=0.0,
            findings=architecture
        ))

        # Phase 1.3: Planning Phase
        self.logger.info("Creating exploration plan...")
        plan_start = time.time()
        exploration_plan = await self._create_exploration_plan(query, fingerprint, architecture)
        phases.append(ReconPhase(
            name="Exploration Planning",
            duration=time.time() - plan_start,
            findings={"plan": exploration_plan.model_dump()}
        ))

        print(f"EXPLORATION PLANNNNNNN: {exploration_plan}")

        # Phase 1.4+: Intelligent Exploration Loop
        self.logger.info("Running intelligent exploration loop...")
        phase_start = time.time()
        exploration_context = await self._run_exploration_loop(query, fingerprint, architecture, root_path, exploration_plan)

        phases.append(ReconPhase(
            name="Intelligent Exploration",
            duration=time.time() - phase_start,
            findings=exploration_context
        ))

        # Phase 2: Synthesis
        self.logger.info("Running synthesis phase...")
        synthesis_start = time.time()
        final_synthesis = exploration_context.get("final_synthesis", "No synthesis available")
        phases.append(ReconPhase(
            name="Synthesis",
            duration=time.time() - synthesis_start,
            findings=final_synthesis
        ))

        # Determine termination reason
        termination_reason = "normal"
        if exploration_context.get("llm_error"):
            termination_reason = "llm_error"
        elif exploration_context.get("tool_errors"):
            termination_reason = "tool_errors"
        elif len(exploration_context.get("findings", [])) > 50:
            termination_reason = "auto_terminate_too_many_findings"
        elif exploration_context.get("iterations", 0) >= self.max_iterations:
            termination_reason = "max_iterations_reached"
        else:
            termination_reason = "done_tool_called"

        # Build final result
        result = ReconnaissanceResult(
            query=query,
            fingerprint=fingerprint,
            architecture=architecture,
            execution_time=time.time() - start_time,
            phases_executed=[{
                "name": phase.name,
                "duration": phase.duration,
                "findings": phase.findings,
                "artifacts": phase.artifacts
            } for phase in phases],
            # Include error information if an LLM error occurred
            error=exploration_context.get("llm_error"),
            # Include tool errors if any
            tool_errors=exploration_context.get("tool_errors"),
            # Include termination reason
            termination_reason=termination_reason
        )

        return result

    async def _create_exploration_plan(self, query: str, fingerprint: CodebaseFingerprint,
                                     architecture: Dict[str, Any]) -> ExplorationPlan:
        """Create a structured exploration plan using LLM analysis"""
        self.logger.info("Creating exploration plan...")

        # Build the planning prompt
        planning_prompt = RECONNAISSANCE_PLANNING_PROMPT.format(
            query=query,
            fingerprint=json.dumps(fingerprint.model_dump(), indent=2),
            architecture=json.dumps(architecture, indent=2)
        )

        # Call LLM to generate plan
        try:
            response = self.llm.call(
                prompt_or_messages=[
                    {"role": "system", "content": planning_prompt},
                    {"role": "user", "content": "Create an exploration plan to answer the user's query based on the provided context."}
                ],
                temperature=0.5,  # Lower temperature for more structured planning
                response_format={"type": "json_object"}
            )

            # Parse the JSON response
            # Response is a string directly, not a dict
            self.logger.debug(f"Raw LLM response: {response}")

            # Extract JSON from response (handle markdown code blocks)
            json_str = response
            if response and "```" in response:
                # Extract JSON from markdown code blocks
                import re
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
                if match:
                    json_str = match.group(1)
                    self.logger.debug(f"Extracted JSON from markdown: {json_str}")

            # Try to parse JSON, with better error handling
            try:
                if json_str and json_str.strip():
                    plan_data = json.loads(json_str)
                else:
                    plan_data = {}
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON response: {e}")
                self.logger.error(f"Raw response: {repr(response)}")
                plan_data = {}

            # Create ExplorationPlan object (minimal)
            plan = ExplorationPlan(
                steps=plan_data.get("steps", [])
            )

            self.logger.info(f"Generated exploration plan with {len(plan.steps)} actionable steps")

            # Log first few steps for debugging
            for i, step in enumerate(plan.steps[:3]):
                self.logger.info(f"Step {i+1}: {step}")

            return plan

        except Exception as e:
            self.logger.error(f"Failed to create exploration plan: {str(e)}")
            # Exploration planning failed - cannot proceed
            self.logger.error("Exploration planning failed - cannot create a valid plan")
            raise RuntimeError(f"Failed to create exploration plan: {str(e)}")


    async def _run_exploration_loop(self, query: str, fingerprint: CodebaseFingerprint,
                                  architecture: Dict[str, Any], root_path: str,
                                  exploration_plan: Optional[ExplorationPlan] = None) -> Dict[str, Any]:
        """
        Execute exploration using the provided exploration plan.
        
        Args:
            query: The user's query
            fingerprint: Codebase fingerprint from phase 1.1
            architecture: Architecture mapping from phase 1.2
            root_path: Root path of the codebase
            exploration_plan: Required exploration plan with steps
            
        Returns:
            Dictionary containing exploration results
            
        Raises:
            ValueError: If no exploration plan is provided
        """
        if not exploration_plan or not exploration_plan.steps:
            raise ValueError("Exploration plan is required but not provided or empty")

        # Import filesystem utils for tool execution
        fs_utils = FilesystemUtils(Path(root_path))

        # Initialize global context to accumulate results from all steps
        global_context = {
            "query": query,
            "fingerprint": fingerprint.model_dump(),
            "architecture": architecture,
            "exploration_plan": exploration_plan.model_dump() if exploration_plan else None,
            "step_contexts": [],  # Array to store results from each step
            "total_steps": len(exploration_plan.steps) if exploration_plan else 0,
            "completed_steps": 0,
            "start_time": time.time()
        }

        self.logger.info(f"🚀 Starting nested exploration with {global_context['total_steps']} steps")

        # Build base system prompt
        base_system_prompt = RECONNAISSANCE_AGENT_PROMPT.format(
            fingerprint=json.dumps(fingerprint.model_dump(), indent=2),
            architecture=json.dumps(architecture, indent=2)
        )

        # OUTER LOOP: Iterate through each step in the exploration plan
        for step_idx, step in enumerate(exploration_plan.steps):
            step_num = step_idx + 1
            total_steps = len(exploration_plan.steps)

            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"📍 STEP {step_num}/{total_steps}: {step}")
            self.logger.info(f"{'='*80}")

            # Initialize step-specific context
            step_context = {
                "step_number": step_num,
                "step_description": step,
                "tool_results": [],
                "findings": [],
                "search_history": [],
                "iterations": 0,
                "completed": False,
                "summary": "",
                "start_time": time.time()
            }

            # Initialize conversation history for this step with proper message accumulation
            conversation_history = [
                {
                    "role": "user",
                    "content": f"Step {step_num}/{total_steps}: {step}\n\nPlease explore this step and call completed() when you have sufficient information."
                }
            ]

            # INNER LOOP: Run LLM+tool interactions for this step until completed()
            max_step_iterations = min(10, self.max_iterations)  # Limit iterations per step
            step_completed = False

            # Loop detection variables
            last_tool_calls_signature = None
            iterations_without_progress = 0

            for iteration in range(max_step_iterations):
                self.logger.info(f"[Step {step_num}] Iteration {iteration + 1}/{max_step_iterations}")
                step_context["iterations"] = iteration + 1

                # Check if context compression should be triggered
                current_context_size = len(str(step_context.get("findings", []))) + len(str(step_context.get("tool_results", [])))
                if self.context_manager.should_trigger_context_compression(iteration + 1, current_context_size):
                    self.logger.info(f"🔄 Context compression triggered at step {step_num}, iteration {iteration + 1}")
                    self.compressed_context = self.context_manager.compress_accumulated_context(
                        accumulated_findings=step_context.get("findings", []),
                        files_explored=len(self.visited_files),
                        directories_explored=len(self.visited_directories),
                        current_iteration=iteration + 1,
                        max_iterations=max_step_iterations
                    )
                    step_context["compressed_context"] = self.compressed_context

                # Add iteration prompt if not the first iteration
                if iteration > 0:
                    conversation_history.append({
                        "role": "user",
                        "content": f"Iteration {iteration + 1}. Continue exploring or call completed() if you have gathered sufficient information."
                    })

                # LLM decides next action for this step
                try:
                    self.logger.debug(f"Making LLM call for step {step_num}, iteration {iteration + 1}")

                    # Log the full prompt being sent to LLM
                    print(f"\n{'='*80}")
                    print(f"🔸 LLM INPUT - Step {step_num}/{total_steps}, Iteration {iteration + 1}")
                    print(f"{'='*80}")
                    print(f"Current Step: {step}")
                    print(f"Conversation history length: {len(conversation_history)} messages")

                    response = self.llm.call(
                        prompt_or_messages=conversation_history,
                        tools=self.available_tools,
                        system_prompt=base_system_prompt,
                        model=self.model,
                        temperature=0.3,
                        reasoning_effort=self.reasoning_effort
                    )

                    # Log the full response from LLM
                    print(f"{'='*80}")
                    print(f"🔸 LLM RESPONSE - Step {step_num}, Iteration {iteration + 1}")
                    print(f"{'='*80}")
                    if response.get("tool_calls"):
                        print(f"LLM decided to use tools:")
                        for i, tc in enumerate(response.get("tool_calls", []), 1):
                            tool_name = tc.get("function", {}).get("name", "unknown")
                            tool_args = tc.get("function", {}).get("arguments", "{}")
                            print(f"  {i}. {tool_name}({tool_args})")
                        tool_names = [tc.get("function", {}).get("name", "unknown") for tc in response.get("tool_calls", [])]
                        self.logger.info(f"✅ LLM called tools: {', '.join(tool_names)}")
                    else:
                        content = response.get("content", "")
                        print(f"LLM Direct Response:\n{content}")
                        self.logger.info(f"✅ LLM provided direct response")
                    print(f"{'='*80}\n")

                except Exception as e:
                    error_msg = f"LLM call failed in step {step_num}, iteration {iteration + 1}"
                    self.logger.error(f"❌ [Reconnaissance] {error_msg}: {e}")
                    step_context["llm_error"] = {
                        "error": str(e),
                        "step": step_num,
                        "iteration": iteration + 1,
                        "model": self.model,
                        "timestamp": time.time()
                    }
                    break

                # Add assistant response to conversation history
                assistant_message = {
                    "role": "assistant",
                    "content": response.get("content", "")
                }
                if response.get("tool_calls"):
                    assistant_message["tool_calls"] = response["tool_calls"]
                conversation_history.append(assistant_message)

                # Check if LLM provided final answer (without tool calls)
                if not response.get("tool_calls"):
                    step_context["final_answer"] = response.get("content", "")
                    step_context["summary"] = response.get("content", "")
                    step_context["completed"] = True
                    step_completed = True
                    self.logger.info(f"✅ Step {step_num} completed with direct answer")
                    break

                # Execute tool calls for this step
                tool_results = []
                step_iteration_completed = False
                tool_errors = []

                for tool_call in response.get("tool_calls", []):
                    tool_name = tool_call.get("function", {}).get("name")
                    self.logger.info(f"   Executing tool: {tool_name}")

                    try:
                        result = await self._execute_tool(tool_call, fs_utils)

                        # Apply context compression if needed
                        content = result.get("content", "")
                        if isinstance(content, str) and len(content) > 3000:
                            try:
                                compressed_result = self.context_manager.compress_tool_interaction(
                                    llm_response=f"Step {step_num}: Used {tool_name}",
                                    tool_use=f"Tool: {tool_name}",
                                    tool_output=content
                                )
                                result["content"] = compressed_result
                                result["compressed"] = True
                                self.logger.info(f"   🔄 Compressed large output from '{tool_name}'")
                            except Exception as e:
                                self.logger.warning(f"   ⚠️ Context compression failed for '{tool_name}': {e}")
                                result["content"] = content[:2000] + "\n... (truncated)"

                        tool_results.append(result)

                        # Log tool execution
                        print(f"\n--- Tool Result: {tool_name} (Step {step_num}) ---")
                        print(f"Success: {result.get('success')}")
                        if result.get("success"):
                            content = result.get("content", "")
                            if isinstance(content, str) and len(content) > 1000:
                                print(f"Content (truncated):\n{content[:1000]}...")
                            elif isinstance(content, str):
                                print(f"Content:\n{content}")
                            elif isinstance(content, list):
                                print(f"Content: Found {len(content)} items")
                            else:
                                print(f"Content: {content}")
                        else:
                            error_msg = result.get('content', 'Unknown error')
                            print(f"Error: {error_msg}")
                            tool_errors.append(f"{tool_name}: {error_msg}")
                        print(f"--- End Tool Result ---\n")

                        # Check if completed() tool was called
                        if isinstance(result, dict) and result.get("task_completed"):
                            summary = result.get("content", "").replace("Reconnaissance task completed. Summary: ", "")
                            step_context["summary"] = summary
                            step_iteration_completed = True
                            self.logger.info(f"   ✅ Step {step_num} completed via completed() tool!")

                    except Exception as e:
                        error_msg = f"Tool execution exception: {str(e)}"
                        self.logger.error(f"   ❌ CRITICAL: Tool '{tool_name}' crashed: {error_msg}")
                        tool_errors.append(error_msg)
                        import traceback
                        self.logger.error(f"   Full traceback: {traceback.format_exc()}")

                # CRITICAL FIX: Add tool results to conversation history so LLM sees them
                tool_results_for_llm = []
                for tool_call in response.get("tool_calls", []):
                    # Find the corresponding result
                    tool_name = tool_call.get("function", {}).get("name")
                    matching_result = None

                    for result in tool_results:
                        if result.get("tool") == tool_name:
                            matching_result = result
                            break

                    if matching_result:
                        tool_results_for_llm.append({
                            "tool_call_id": tool_call.get("id", ""),
                            "role": "tool",
                            "name": tool_name,
                            "content": matching_result.get("content", "")
                        })

                # Add tool results to conversation history
                conversation_history.extend(tool_results_for_llm)

                # Update step context with tool results
                step_context["tool_results"].extend(tool_results)

                if tool_errors:
                    if "tool_errors" not in step_context:
                        step_context["tool_errors"] = []
                    step_context["tool_errors"].extend(tool_errors)

                # Record iteration in search history
                step_context["search_history"].append({
                    "iteration": iteration + 1,
                    "tool_calls": response.get("tool_calls", []),
                    "results": tool_results,
                    "llm_thought": response.get("content", ""),
                    "errors": tool_errors
                })

                # Extract findings from tool results
                for tool_result in tool_results:
                    if isinstance(tool_result, dict) and "content" in tool_result:
                        step_context["findings"].append(tool_result["content"])
                    elif isinstance(tool_result, str):
                        step_context["findings"].append(tool_result)

                # LOOP DETECTION: Check for redundant tool calls
                if response.get("tool_calls"):
                    current_signature = str(sorted([
                        f"{tc['function']['name']}:{json.dumps(tc['function']['arguments'])}"
                        for tc in response["tool_calls"]
                    ]))

                    if current_signature == last_tool_calls_signature:
                        iterations_without_progress += 1
                        self.logger.warning(f"⚠️ Detected identical tool calls for {iterations_without_progress} consecutive iterations")

                        if iterations_without_progress >= 3:
                            self.logger.warning(f"🔄 Auto-completing step {step_num} due to redundant loop")
                            step_context["summary"] = f"Auto-completed: detected redundant loop of identical tool calls"
                            step_context["completed"] = True
                            step_completed = True
                            break
                    else:
                        iterations_without_progress = 0
                        last_tool_calls_signature = current_signature

                # Check if step was completed in this iteration
                if step_iteration_completed:
                    step_context["completed"] = True
                    step_completed = True
                    self.logger.info(f"✅ Step {step_num} completed after {iteration + 1} iterations")
                    break
                else:
                    self.logger.debug(f"   Continuing to next iteration for step {step_num}")

                # Safeguard: Auto-terminate if too many findings without completion
                if len(step_context.get("findings", [])) > 30:
                    self.logger.warning(f"Auto-terminating step {step_num}: Too many findings without completion")
                    step_context["summary"] = f"Explored step with {len(step_context['findings'])} findings"
                    step_context["completed"] = True
                    step_completed = True
                    break

            # End of inner loop for this step

            # Ensure step is marked as completed even if max iterations reached
            if not step_completed:
                self.logger.warning(f"Step {step_num} reached max iterations without completion")
                step_context["completed"] = True
                if not step_context.get("summary"):
                    step_context["summary"] = f"Completed step after {max_step_iterations} iterations"

            # Calculate step duration
            step_context["duration"] = time.time() - step_context["start_time"]

            # Add completed step context to global context
            global_context["step_contexts"].append(step_context)
            global_context["completed_steps"] += 1

            self.logger.info(f"✅ Step {step_num}/{total_steps} finished in {step_context['duration']:.2f}s")

            # Brief pause between steps for clarity
            await asyncio.sleep(0.1)

        # All steps completed - now synthesize final result
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🎯 ALL STEPS COMPLETED - Running final synthesis")
        self.logger.info(f"{'='*80}")

        # Final synthesis using all step contexts
        final_synthesis = await self._aggregate_all_steps_output(query, global_context)

        # Prepare final context
        final_context = {
            "query": query,
            "global_context": global_context,
            "final_synthesis": final_synthesis,
            "total_duration": time.time() - global_context["start_time"],
            "total_steps": global_context["total_steps"],
            "completed_steps": global_context["completed_steps"],
            "step_summaries": [ctx.get("summary", "") for ctx in global_context["step_contexts"]],
            "all_findings": [finding for ctx in global_context["step_contexts"] for finding in ctx.get("findings", [])]
        }

        return final_context

    def _build_base_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build a clear, contextual prompt that guides the LLM's next exploration step"""

        iteration = context.get('iterations', 0)

        # Calculate progress metrics
        files_explored = len(self.visited_files)
        dirs_explored = len(self.visited_directories)
        progress_pct = min(100, (files_explored / 10) * 100)  # Rough progress estimate

        # Summarize recent findings
        findings_summary = ""

        # Identify information gaps
        info_gaps = self._identify_information_gaps(query, context)
        gaps_section = ""
        if info_gaps:
            gaps_section = f"\nINFORMATION GAPS TO ADDRESS:\n"
            for gap in info_gaps[:3]:  # Show top 3 gaps
                gaps_section += f"- {gap}\n"

        return f"""
CURRENT OBJECTIVE:
Answer the query: "{query}"

EXPLORATION PROGRESS:
- Files analyzed: {files_explored}
- Directories explored: {dirs_explored}
- Iteration: {iteration + 1}/{self.max_iterations}
- Estimated completion: {progress_pct:.0f}%

{findings_summary}

{gaps_section}

NEXT STEPS:
Based on current findings, focus on:
1. Files most likely to contain answers to the query
2. Configuration files that explain the architecture
3. Entry points and main application files
4. Documentation files (README, docs, comments)

STOP CONDITION:
Call completed() as soon as you have sufficient information to provide a comprehensive answer about:
- The codebase structure relevant to the query
- Key components and their relationships
- Technologies and patterns used
- Specific locations and implementations that answer the user's question
        """

    def _identify_information_gaps(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Identify what information is still needed to answer the query"""
        gaps = []

        # Check if we have basic project understanding
        if not any('README' in f or 'readme' in f for f in self.visited_files):
            gaps.append("Project documentation and README files")

        # Check if we understand the architecture
        if not context.get('architecture_detected'):
            gaps.append("Main application entry points and architecture")

        # Check for configuration files
        config_extensions = ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg']
        has_config = any(f.endswith(tuple(config_extensions)) for f in self.visited_files)
        if not has_config:
            gaps.append("Configuration files and dependencies")

        # Check for main source files
        if len(self.visited_files) < 5:
            gaps.append("More source code files to understand the implementation")

        # Query-specific gaps
        query_lower = query.lower()
        if 'test' in query_lower and not any('test' in f for f in self.visited_files):
            gaps.append("Test files and testing infrastructure")
        if 'api' in query_lower and not any('api' in f or 'endpoint' in f for f in self.visited_files):
            gaps.append("API endpoints and routing files")
        if 'database' in query_lower and not any('db' in f or 'sql' in f or 'model' in f for f in self.visited_files):
            gaps.append("Database models and schema files")

        return gaps

    def _build_exploration_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build a comprehensive exploration prompt that guides the LLM effectively"""

        # Start with the improved base prompt
        prompt = self._build_base_prompt(query, context)

        # Add strategic context if available from context manager
        if context.get("compressed_context"):
            compressed = context["compressed_context"]

            strategic_section = f"""

STRATEGIC ANALYSIS SUMMARY:
Architecture Pattern: {compressed.get('architecture_summary', 'Still analyzing...')}
Key Discoveries: {'; '.join(compressed.get('key_insights', ['None yet']))}
Remaining Questions: {'; '.join(compressed.get('information_gaps', ['Many']))}
Suggested Next Steps: {'; '.join(compressed.get('next_priorities', ['Continue exploring']))}

Completion Progress: {compressed.get('completion_assessment', 0)}%
"""
            prompt += strategic_section

        # Add current session context
        if context.get('findings'):
            findings = context['findings']
            key_patterns = findings.get('patterns', [])[:3]
            tech_stack = findings.get('technologies', [])

            if key_patterns or tech_stack:
                context_section = f"""

CURRENT SESSION INSIGHTS:
"""
                if tech_stack:
                    context_section += f"Technologies Detected: {', '.join(tech_stack[:5])}\n"
                if key_patterns:
                    context_section += f"Patterns Observed: {', '.join(key_patterns)}\n"

                prompt += context_section

        # Add clear exploration guidance
        guidance_section = f"""

            EXPLORATION STRATEGY:
            1. Focus on files most relevant to: "{query}"
            2. Look for main entry points (main.py, app.py, index.js, etc.)
            3. Check configuration files (package.json, requirements.txt, pyproject.toml)
            4. Examine documentation (README.md, docs/, comments)
            5. Follow import/dependency chains to understand connections

            REMEMBER:
            - Each tool call should bring you closer to answering the query
            - Don't read the same file twice
            - Use grep to search for specific patterns across multiple files
            - Call completed() when you have enough information to answer comprehensively
        """

        prompt += guidance_section

        return prompt

    def _get_current_context_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of current exploration context for optimization decisions"""
        return {
            "visited_files_count": len(self.visited_files),
            "visited_directories_count": len(self.visited_directories),
            "findings_count": len(context.get("findings", [])),
            "unexplored_directories": len(context.get("search_history", [])) < 3,
            "should_search_patterns": len(context.get("findings", [])) > 5,
            "tools_used": len(context.get("tool_calls", []))
        }

    def _build_assistant_step_prompt(self, step: str, step_num: int, total_steps: int, global_context: Dict[str, Any]) -> str:
        """Build assistant's step context message"""

        step_context_msg = f"""
            I'm currently on Step {step_num} of {total_steps} in a structured exploration plan.

            CURRENT STEP: {step}

            STEP CONTEXT:
            - This is step {step_num} out of {total_steps} total steps
            - I need to focus on completing this specific step before moving to the next
            - I'll use the completed() tool when I have finished this step's objective
            - Previous step results will inform my current exploration

            PREVIOUS STEPS COMPLETED: {step_num - 1}
        """

        # Add summaries from previous steps if available
        if step_num > 1 and global_context.get("step_contexts"):
            step_context_msg += "\nPREVIOUS STEP SUMMARIES:\n"
            for i, prev_ctx in enumerate(global_context["step_contexts"][:step_num-1], 1):
                summary = prev_ctx.get("summary", "No summary available")
                step_context_msg += f"Step {i}: {summary}\n"

        # Add step-specific guidance
        step_context_msg += """

            STEP EXECUTION GUIDELINES:
            1. Focus specifically on the current step's objective
            2. Use tools efficiently to gather information for this step
            3. Call completed() once I have sufficient information for this step
            4. My findings will be combined with other steps for a comprehensive answer
            5. The user will provide specific iteration instructions as needed
            6. I should track my progress and findings within each step

            Remember: Each step builds toward answering the user's original query. Complete this step thoroughly but efficiently.
        """

        return step_context_msg.strip()


    async def _aggregate_all_steps_output(self, query: str, global_context: Dict[str, Any]) -> str:
        """Synthesize results from all completed steps into a comprehensive answer"""

        step_summaries = []
        all_findings = []

        # Collect summaries and findings from all steps
        for step_ctx in global_context.get("step_contexts", []):
            step_num = step_ctx.get("step_number", 0)
            step_desc = step_ctx.get("step_description", "Unknown step")
            summary = step_ctx.get("summary", "No summary available")
            findings = step_ctx.get("findings", [])

            step_summaries.append({
                "step": step_num,
                "description": step_desc,
                "summary": summary,
                "findings_count": len(findings)
            })

            all_findings.extend(findings)

        # Build messages for synthesis
        messages = [
            {
                "role": "user",
                "content": f"""Please synthesize the results from this multi-step exploration:

                    ORIGINAL QUERY: {query}

                    EXPLORATION STEPS COMPLETED:
                """
            }
        ]

        # Add step summaries to user message
        steps_content = ""
        for step_info in step_summaries:
            steps_content += f"""
                Step {step_info['step']}: {step_info['description']}
                - Summary: {step_info['summary']}
                - Findings: {step_info['findings_count']} items
            """

            steps_content += f"""
                TOTAL FINDINGS ACROSS ALL STEPS: {len(all_findings)}

                Please provide a comprehensive synthesis that addresses the original query.
            """

        # Update the user message with all content
        messages[0]["content"] += steps_content

        try:
            # Make LLM call for synthesis
            self.logger.info("🔄 Running final synthesis of all steps...")

            # Format synthesis prompt with fingerprint and architecture
            synthesis_system_prompt = RECONNAISSANCE_SYNTHESIS_PROMPT.format(
                fingerprint=self.fingerprint,
                architecture=self.architecture
            )

            response = self.llm.call(
                prompt_or_messages=messages,
                system_prompt=synthesis_system_prompt,
                model=self.model,
                temperature=0.3,
                reasoning_effort=self.reasoning_effort
            )

            synthesis = response.get("content", "Failed to synthesize results")
            self.logger.info("✅ FINAL SYNTHESIS COMPLETEDDDDDD")

            return synthesis

        except Exception as e:
            self.logger.error(f"❌ Synthesis failed: {e}")
            # Fallback: combine summaries manually
            fallback_synthesis = f"Based on {len(step_summaries)} exploration steps:\n\n"
            for step_info in step_summaries:
                fallback_synthesis += f"Step {step_info['step']}: {step_info['summary']}\n\n"
            return fallback_synthesis

    async def _execute_tool(self, tool_call: Dict, fs_utils: FilesystemUtils) -> Dict[str, Any]:
        """Execute a single tool call"""

        tool_name = tool_call.get("function", {}).get("name")
        arguments = tool_call.get("function", {}).get("arguments", "{}")

        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            args = {}

        result = {"tool": tool_name, "args": args, "success": False, "content": ""}

        try:
            # Route to appropriate tool handler
            if tool_name in ["list_directory", "read_file", "find_files", "grep", "get_file_tree", "detect_languages"]:
                result = await self._execute_filesystem_tool(tool_name, args, fs_utils)
            elif tool_name in ["analyze_imports", "analyze_code_structure", "extract_functions", "extract_classes"]:
                result = await self._execute_ast_tool(tool_name, args, fs_utils)
            elif tool_name == "completed":
                result = await self._execute_completed_tool(tool_name, args)
            else:
                result["content"] = f"Unknown tool: {tool_name}"
                result["success"] = False
        except Exception as e:
            result["content"] = f"Tool execution error: {str(e)}"
            result["success"] = False

        return result

    async def _execute_filesystem_tool(self, tool_name: str, args: Dict, fs_utils: FilesystemUtils) -> Dict[str, Any]:
        """Execute filesystem tools"""

        result = {"tool": tool_name, "args": args, "success": True, "content": ""}

        try:
            if tool_name == "list_directory":
                path = args.get("path", ".")
                recursive = args.get("recursive", False)
                max_depth = args.get("max_depth", 3)
                show_hidden = args.get("show_hidden", False)

                # Create unique key for this specific directory operation
                dir_key = f"{path}:{recursive}:{max_depth}:{show_hidden}"

                # Check for redundancy and skip if already visited
                if dir_key in self.visited_directories:
                    result["success"] = True
                    result["content"] = f"""Directory '{path}' was already explored with these parameters.

Exploration history:
- Files analyzed so far: {len(self.visited_files)}
- Directories explored: {len(self.visited_directories)}
- This specific listing was already done

Recommendations:
- Try reading specific files from this directory
- Use find_files with different patterns
- Use grep to search for content within files
- Explore subdirectories individually"""
                    return result

                # Track this specific directory operation
                self.visited_directories.add(dir_key)

                files = await fs_utils.list_directory(
                    path, recursive=recursive, max_depth=max_depth, show_hidden=show_hidden
                )
                result["content"] = f"Found {len(files)} items in {path}:\n" + "\n".join(files[:20])

            elif tool_name == "read_file":
                path = args.get("path")
                if not path:
                    result["content"] = "Error: path is required"
                    result["success"] = False
                else:
                    # Check for redundancy and skip if already visited
                    if path in self.visited_files:
                        # Return a brief summary of what we know about this file
                        filename = Path(path).name
                        result["success"] = True
                        result["content"] = f"""File '{filename}' was already read in a previous iteration.

Key points about this file:
- Path: {path}
- Status: Previously analyzed and content collected
- Recommendation: Try exploring related files or use grep to search for specific patterns

Instead of re-reading this file, consider:
- Reading related files in the same directory
- Using grep to search for specific patterns across files
- Calling completed() if you have enough information"""
                        return result

                    # Track visited files to avoid redundancy
                    self.visited_files.add(path)

                    file_result = await fs_utils.read_file(path)
                    if file_result.error:
                        result["content"] = f"Error reading {path}: {file_result.error}"
                        result["success"] = False
                    else:
                        content = file_result.content
                        # Truncate very long files
                        if len(content) > 5000:
                            content = content[:5000] + "\n... (truncated)"
                        result["content"] = f"Content of {path}:\n{content}"

            elif tool_name == "find_files":
                pattern = args.get("pattern", "*")
                path = args.get("path", ".")
                file_type = args.get("file_type", "name")
                exclude_patterns = args.get("exclude_patterns", [])

                # Track visited patterns to avoid redundant searches
                pattern_key = f"{pattern}:{path}:{file_type}"
                if pattern_key not in self.visited_patterns:
                    self.visited_patterns.add(pattern_key)

                files = await fs_utils.find_files(pattern, path, file_type, exclude_patterns)
                result["content"] = f"Found {len(files)} files matching '{pattern}' in {path}:\n" + "\n".join(files[:30])

            elif tool_name == "grep":
                pattern = args.get("pattern", "")
                path = args.get("path", ".")
                file_patterns = args.get("file_patterns", ["*"])
                context_lines = args.get("context_lines", 2)

                matches = await fs_utils.grep(pattern, path, file_patterns, context_lines)
                result["content"] = f"Found {len(matches)} matches for '{pattern}':\n"
                for match in matches[:20]:
                    result["content"] += f"  {match}\n"

            elif tool_name == "get_file_tree":
                path = args.get("path", ".")
                max_depth = args.get("max_depth", 3)

                tree = await fs_utils.get_file_tree(path, max_depth)
                # Format the tree for display
                def format_tree(node, indent=0):
                    lines = []
                    for name, child in node.get("_children", {}).items():
                        prefix = "  " * indent
                        if child.get("_type") == "file":
                            lines.append(f"{prefix}{name} (file)")
                        else:
                            lines.append(f"{prefix}{name}/")
                            lines.extend(format_tree(child, indent + 1))
                    return lines

                formatted_tree = format_tree(tree)
                result["content"] = f"File tree for {path} (max depth {max_depth}):\n" + "\n".join(formatted_tree[:50])

            elif tool_name == "detect_languages":
                path = args.get("path", ".")

                languages = await fs_utils.detect_languages(path)
                result["content"] = f"Programming languages detected in {path}:\n"
                for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                    result["content"] += f"  {lang}: {count} files\n"

        except Exception as e:
            result["content"] = f"Error executing {tool_name}: {str(e)}"
            result["success"] = False

        return result

    async def _execute_completed_tool(self, tool_name: str, args: Dict) -> Dict[str, Any]:
        """Execute the completed tool to mark task completion"""

        summary = args.get("summary", "Task completed")

        # Print the completion message
        self.logger.info("✅ QROOPER QUERY COMPLETED SUCCESSFULLY")

        result = {
            "tool": tool_name,
            "args": args,
            "success": True,
            "content": f"Reconnaissance task completed. Summary: {summary}",
            "task_completed": True  # Special flag to indicate completion
        }

        return result

    async def _execute_ast_tool(self, tool_name: str, args: Dict, fs_utils: FilesystemUtils) -> Dict[str, Any]:
        """Execute AST parsing tools"""

        result = {"tool": tool_name, "args": args, "success": True, "content": ""}

        # For now, provide a placeholder implementation
        # In a full implementation, we would import and use the actual AST parsing functions
        result["content"] = f"AST tool {tool_name} executed with args: {args}"
        result["success"] = True

        return result


  

# uv run python -m qrooper.agents.reconnaissance
if __name__ == "__main__":
    """
    Main entry point for running the reconnaissance demo.

    This script demonstrates the capabilities of the LightningScanner and StructureMapper
    agents by analyzing the current codebase structure and providing detailed insights.

    Usage:
        python reconnaissance.py

    The demo will:
        1. Perform a quick lightning scan to fingerprint the codebase
        2. Map the architecture and conventions
        3. Display comprehensive analysis results
    """
    import asyncio
    import sys

    async def _run_demo():
        """Run a demonstration of the full ReconnaissanceAgent implementation"""
        from pathlib import Path

        print("=" * 80)
        print("QROOPER Reconnaissance Agent Demo - Full Implementation")
        print("=" * 80)

        # Use the eth-insights codebase for testing
        root_path = Path("/Users/avinier/0xPPL/eth-insights")

        if not root_path.exists():
            raise SystemExit(f"Demo error: eth-insights codebase not found at {root_path}")

        print(f"\nAnalyzing project at: {root_path}")
        print(f"Project name: {root_path.name}")

        # Configure model and reasoning effort
        model = "glm-4.5-air"  # Use the model from original demo
        reasoning_effort = "none"

        print(f"\nModel: {model}")
        print(f"Reasoning Effort: {reasoning_effort}")

        print("\n" + "=" * 80)
        print("Starting Full Reconnaissance Analysis...")
        print("=" * 80)

        # Phase 1.1: Lightning Scan (moved from agent) - Optimized
        print("\nPhase 1.1: Lightning Scan - Quick codebase fingerprinting (OPTIMIZED)...")
        phase_start = time.time()
        scanner = LightningScanner(root_path)  # Creates shared FileCache internally
        fingerprint = await scanner.scan()  # Uses optimized parallel methods
        lightning_duration = time.time() - phase_start
        print(f"Lightning scan completed in {lightning_duration:.2f}s (OPTIMIZED)")

        # Print raw fingerprint data
        # print("\nCODE FINGERPRINT RAWRESPONSE:")
        # print("=" * 50)
        # print(fingerprint.model_dump_json(indent=2))
        # print("=" * 50)

        # Phase 1.2: Structural Mapping (moved from agent) - Optimized
        print("\nPhase Phase 1.2: Structural Mapping - Architecture analysis (OPTIMIZED)...")
        phase_start = time.time()
        # Reuse the same cache from LightningScanner for maximum performance
        mapper = StructuralMapper(root_path, cache=scanner.cache)
        architecture = await mapper.map_architecture_optimized(fingerprint)  # Use optimized version
        struct_duration = time.time() - phase_start
        print(f"Structural mapping completed in {struct_duration:.2f}s (OPTIMIZED)")

        # Print raw architecture data
        # print("\nCODE ARCHITECTURE RAWRESPONSE:")
        # print("=" * 50)
        # print(json.dumps(architecture, indent=2, default=str))
        # print("=" * 50)

        # Initialize the full ReconnaissanceAgent with pre-computed data
        print("\nAgent Initializing ReconnaissanceAgent with pre-computed analysis...")
        agent = ReconnaissanceAgent(
            model=model,
            reasoning_effort=reasoning_effort,
            fingerprint=fingerprint,
            architecture=architecture
        )

        # Define a test query
        query = "Analyze this codebase and provide insights about its architecture, technologies used, and main components. Focus on understanding the overall structure and purpose of this project."

        print(f"\nQuery: {query}")

        # Phase 1.3+: Intelligent Exploration Loop
        print("\n Phase 1.3+: Intelligent Exploration Loop - LLM-guided analysis...")
        print("=" * 80)

        # Track the exploration phase start time
        exploration_start = time.time()

        try:
            # Execute the reconnaissance (only exploration phase now)
            result = await agent.analyze(query, str(root_path))

            # Calculate total execution time (all phases)
            execution_time = lightning_duration + struct_duration + (time.time() - exploration_start)

            # Get references to results for easier access
            fp = result.fingerprint
            arch = result.architecture

            # Display results
            print("\n" + "=" * 80)
            print(" RECONNAISSANCE RESULTS")
            print("=" * 80)

            print(f"\n⏱️ Total execution time: {execution_time:.2f} seconds")
            print(f" Files analyzed: {result.files_analyzed}")
            print(f" Phases executed: {len(result.phases_executed)}")

            # Show phase details
            print("\nPhases Phases Executed:")
            for phase in result.phases_executed:
                print(f"  • {phase['name']}: {phase['duration']:.2f}s")

            # Fingerprint results
            print("\n" + "-" * 50)
            print("🔍 LIGHTNING SCAN RESULTS")
            print("-" * 50)

            fp = result.fingerprint
            print(f"\nProject: {fp.name}")
            print(f"Total Files: {fp.total_files}")
            print(f"Size Estimate: {fp.size_estimate}")
            print(f"Scan Time: {fp.scan_time:.3f}s")

            if fp.languages:
                print(f"\n📈 Languages (top 10):")
                total = sum(fp.languages.values())
                for lang, count in sorted(fp.languages.items(), key=lambda x: x[1], reverse=True)[:10]:
                    pct = (count / total) * 100
                    print(f"  • {lang:<20} {count:>4} files ({pct:5.1f}%)")

            if fp.frameworks:
                print(f"\nPhase Frameworks Detected:")
                for fw in fp.frameworks[:10]:
                    print(f"  ✓ {fw}")

            if fp.dependencies:
                deps = fp.dependencies
                if deps.get('package_managers'):
                    print(f"\n📦 Package Managers:")
                    for pm in deps['package_managers']:
                        print(f"  ✓ {pm}")

                if deps.get('dependency_files'):
                    print(f"\nPhases Dependency Files ({len(deps['dependency_files'])}):")
                    for df in deps['dependency_files'][:5]:
                        print(f"  • {df}")
                    if len(deps['dependency_files']) > 5:
                        print(f"  ... and {len(deps['dependency_files']) - 5} more")

            # Architecture results
            print("\n" + "-" * 50)
            print("🏛️ STRUCTURAL MAPPING RESULTS")
            print("-" * 50)

            arch = result.architecture

            if arch.get('patterns'):
                print(f"\n Architectural Patterns:")
                for pattern in arch['patterns']:
                    print(f"  ✓ {pattern}")

            if arch.get('modules', {}).get('directories'):
                print(f"\n📂 Key Modules (top 15):")
                modules = arch['modules']['directories']
                for name, info in sorted(modules.items(), key=lambda x: x[1].get('file_count', 0), reverse=True)[:15]:
                    purpose = info.get('purpose', 'Unknown')
                    file_count = info.get('file_count', 0)
                    print(f"   {name:<25} {purpose:<25} ({file_count} files)")

            # File analyses
            print("\n" + "-" * 50)
            print(" FILES ANALYZED")
            print("-" * 50)

            if result.file_analyses:
                print(f"\nAnalyzed {len(result.file_analyses)} files:")
                for analysis in result.file_analyses[:10]:
                    print(f"  • {analysis.path} ({analysis.language})")
                    if analysis.summary:
                        print(f"    {analysis.summary[:100]}...")
                if len(result.file_analyses) > 10:
                    print(f"  ... and {len(result.file_analyses) - 10} more files")
            else:
                print("\nNo detailed file analyses performed (exploration may have been sufficient)")

            #Synthesis results
            print("\n" + "-" * 50)
            print(" SYNTHESIS & INSIGHTS")
            print("-" * 50)

            print("No synthesis generated (LLM call failed due to API issues)")

            # Raw data for debugging
            print("\n" + "-" * 50)
            print("RAW RESPONSE DATA")
            print("-" * 50)

            import pprint

            print(f"\nRAW_FINGERPRINT OBJECT:")
            print("=" * 30)
            # Convert fingerprint to dict for pretty printing
            fp_dict = {
                "name": fp.name,
                "path": fp.path,
                "timestamp": fp.timestamp,
                "total_files": fp.total_files,
                "size_estimate": fp.size_estimate,
                "scan_time": fp.scan_time,
                "languages": dict(fp.languages),
                "frameworks": fp.frameworks,
                "build_tools": fp.build_tools,
                "dependencies": fp.dependencies,
                "entry_points": fp.entry_points,
                "top_level_structure": fp.top_level_structure
            }
            pprint.pprint(fp_dict, width=120, depth=3)

            print(f"\nRAW_ARCHITECTURE OBJECT:")
            print("=" * 30)
            # Display key parts of architecture
            arch_display = {
                "patterns": arch.get('patterns', []),
                "conventions": {
                    "naming": arch.get('conventions', {}).get('naming', {}),
                    "structure": arch.get('conventions', {}).get('structure', {})
                },
                "modules": {
                    "directories": {k: {"purpose": v.get('purpose'), "file_count": v.get('file_count')}
                                   for k, v in arch.get('modules', {}).get('directories', {}).items()}
                },
                "tests": arch.get('tests', {}),
                "configuration": arch.get('configuration', {}),
                "documentation": arch.get('documentation', {})
            }
            pprint.pprint(arch_display, width=120, depth=3)

            print(f"\nRAW_FILE_ANALYSES OBJECT:")
            print("=" * 30)
            if result.file_analyses:
                analyses_list = []
                for fa in result.file_analyses[:5]:  # Show first 5
                    analyses_list.append({
                        "path": fa.path,
                        "language": fa.language,
                        "size_bytes": fa.size_bytes,
                        "summary": fa.summary[:100] + "..." if fa.summary and len(fa.summary) > 100 else fa.summary,
                        "key_elements_count": len(fa.key_elements),
                        "dependencies_count": len(fa.dependencies),
                        "exports_count": len(fa.exports)
                    })
                print(f"Showing {len(analyses_list)} of {len(result.file_analyses)} file analyses:")
                pprint.pprint(analyses_list, width=120)
            else:
                print("No file analyses")

            print(f"\nRAW_SYNTHESIS OBJECT:")
            print("=" * 30)
            print("No synthesis data (LLM call failed)")

            print(f"\nRAW_RECONNAISSANCERESULT OBJECT:")
            print("=" * 30)
            result_summary = {
                "query": result.query,
                "execution_time": result.execution_time,
                "files_analyzed": result.files_analyzed,
                "phases_executed": len(result.phases_executed),
                "phase_names": [p['name'] for p in result.phases_executed],
                "fingerprint_name": result.fingerprint.name,
                "has_architecture": bool(result.architecture),
                "has_synthesis": False  # synthesis field not in schema
            }
            pprint.pprint(result_summary, width=120)

            # Also save complete raw result to JSON for debugging
            print(f"\nCOMPLETE RAW RESULT (JSON format):")
            print("=" * 50)

            print("\n" + "=" * 80)
            print("Full Reconnaissance Demo Completed Successfully!")
            print("=" * 80)

        except Exception as e:
            print(f"\nError during reconnaissance: {e}")
            import traceback
            traceback.print_exc()
            print("\n" + "=" * 80)
            print("Demo completed with errors")
            print("=" * 80)

    try:
        # Run the async demo
        asyncio.run(_run_demo())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running demo: {e}")
        sys.exit(1)


