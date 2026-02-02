# AI Pipeline Core

A high-performance async framework for building type-safe AI pipelines with LLMs, document processing, and workflow orchestration.

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Basedpyright](https://img.shields.io/badge/type%20checked-basedpyright-blue)](https://github.com/DetachHead/basedpyright)

## Overview

AI Pipeline Core is a production-ready framework that combines document processing, LLM integration, and workflow orchestration into a unified system. Built with strong typing (Pydantic), automatic retries, cost tracking, and distributed tracing, it enforces best practices while keeping application code minimal and straightforward.

### Key Features

- **Document System**: Single `Document` base class with immutable content, SHA256-based identity, automatic MIME type detection, provenance tracking, and multi-part attachments
- **Document Store**: Pluggable storage backends (ClickHouse production, local filesystem CLI/debug, in-memory testing) with automatic deduplication
- **LLM Integration**: Unified interface to any model via LiteLLM proxy with context caching (default 300s TTL)
- **Structured Output**: Type-safe generation with Pydantic model validation via `generate_structured()`
- **Workflow Orchestration**: Prefect-based flows and tasks with annotation-driven document types
- **Auto-Persistence**: `@pipeline_task` saves returned documents to `DocumentStore` automatically (configurable via `persist` parameter)
- **Image Processing**: Automatic image tiling/splitting for LLM vision models with model-specific presets
- **Observability**: Built-in distributed tracing via Laminar (LMNR) with cost tracking, local trace debugging, and ClickHouse-based tracking
- **Deployment**: Unified pipeline execution for local, CLI, and production environments with per-flow resume

## Installation

```bash
pip install ai-pipeline-core
```

### Requirements

- Python 3.12 or higher
- Linux/macOS (Windows via WSL2)

### Development Installation

```bash
git clone https://github.com/bbarwik/ai-pipeline-core.git
cd ai-pipeline-core
pip install -e ".[dev]"
pipx install semgrep  # Installed separately due to dependency conflicts
make install-dev  # Installs pre-commit hooks
```

## Quick Start

### Basic Pipeline

```python
from typing import ClassVar

from pydantic import BaseModel, Field

from ai_pipeline_core import (
    Document,
    DeploymentResult,
    FlowOptions,
    PipelineDeployment,
    pipeline_flow,
    pipeline_task,
    setup_logging,
    get_pipeline_logger,
)

setup_logging(level="INFO")
logger = get_pipeline_logger(__name__)


# 1. Define document types (subclass Document)
class InputDocument(Document):
    """Pipeline input."""

class AnalysisDocument(Document):
    """Per-document analysis result."""

class ReportDocument(Document):
    """Final compiled report."""


# 2. Structured output model
class AnalysisSummary(BaseModel):
    word_count: int
    top_keywords: list[str] = Field(default_factory=list)


# 3. Pipeline task -- auto-saves returned documents to DocumentStore
@pipeline_task
async def analyze_document(document: InputDocument) -> AnalysisDocument:
    return AnalysisDocument.create(
        name=f"analysis_{document.sha256[:12]}.json",
        content=AnalysisSummary(word_count=42, top_keywords=["ai", "pipeline"]),
        sources=(document.sha256,),
    )


# 4. Pipeline flow -- type contract is in the annotations
@pipeline_flow(estimated_minutes=5)
async def analysis_flow(
    project_name: str,
    documents: list[InputDocument],
    flow_options: FlowOptions,
) -> list[AnalysisDocument]:
    results: list[AnalysisDocument] = []
    for doc in documents:
        results.append(await analyze_document(doc))
    return results


@pipeline_flow(estimated_minutes=2)
async def report_flow(
    project_name: str,
    documents: list[AnalysisDocument],
    flow_options: FlowOptions,
) -> list[ReportDocument]:
    report = ReportDocument.create(
        name="report.md",
        content="# Report\n\nAnalysis complete.",
        sources=tuple(doc.sha256 for doc in documents),
    )
    return [report]


# 5. Deployment -- ties flows together with type chain validation
class MyResult(DeploymentResult):
    report_count: int = 0


class MyPipeline(PipelineDeployment[FlowOptions, MyResult]):
    flows: ClassVar = [analysis_flow, report_flow]

    @staticmethod
    def build_result(
        project_name: str,
        documents: list[Document],
        options: FlowOptions,
    ) -> MyResult:
        reports = [d for d in documents if isinstance(d, ReportDocument)]
        return MyResult(success=True, report_count=len(reports))


# 6. CLI initializer provides project name and initial documents
def initialize(options: FlowOptions) -> tuple[str, list[Document]]:
    docs: list[Document] = [
        InputDocument.create(name="input.txt", content="Sample data"),
    ]
    return "my-project", docs


# Run from CLI (requires positional working_directory arg: python script.py ./output)
pipeline = MyPipeline()
pipeline.run_cli(initializer=initialize, trace_name="my-pipeline")
```

### Structured Output

```python
from pydantic import BaseModel
from ai_pipeline_core import llm

class Analysis(BaseModel):
    summary: str
    sentiment: float
    key_points: list[str]

# Generate structured output
response = await llm.generate_structured(
    model="gemini-3-pro",
    response_format=Analysis,
    messages="Analyze this product review: ..."
)

# Access parsed result with type safety
analysis = response.parsed
print(f"Sentiment: {analysis.sentiment}")
for point in analysis.key_points:
    print(f"- {point}")
```

### Document Handling

```python
from ai_pipeline_core import Document

class MyDocument(Document):
    """Custom document type -- must subclass Document."""

# Create documents with automatic conversion
doc = MyDocument.create(
    name="data.json",
    content={"key": "value"}  # Automatically converted to JSON bytes
)

# Parse back to original type
data = doc.parse(dict)  # Returns {"key": "value"}

# Document provenance tracking
source_doc = MyDocument.create(name="source.txt", content="original data")
plan_doc = MyDocument.create(name="plan.txt", content="research plan", sources=(source_doc.sha256,))
derived = MyDocument.create(
    name="derived.json",
    content={"result": "processed"},
    sources=("https://api.example.com/data",),  # Content came from this URL
    origins=(plan_doc.sha256,),  # Created because of this plan (causal, not content)
)

# Check provenance
for hash in derived.source_documents:
    print(f"Derived from document: {hash}")
for ref in derived.source_references:
    print(f"External source: {ref}")
```

## Core Concepts

### Documents

Documents are immutable Pydantic models that wrap binary content with metadata. There is a single `Document` base class -- subclass it to define your document types:

```python
class MyDocument(Document):
    """All documents subclass Document directly."""

# Use create() for automatic conversion
doc = MyDocument.create(
    name="data.json",
    content={"key": "value"}  # Auto-converts to JSON
)

# Access content
if doc.is_text:
    print(doc.text)

# Parse structured data
data = doc.as_json()  # or as_yaml()
model = doc.as_pydantic_model(MyModel)  # Requires model_type argument

# Convert between document types
other = doc.model_convert(OtherDocType)

# Content-addressed identity
print(doc.sha256)  # Full SHA256 hash (base32)
print(doc.id)      # Short 6-char identifier
```

**Document fields:**
- `name`: Filename (validated for security -- no path traversal)
- `description`: Optional human-readable description
- `content`: Raw bytes (auto-converted from str, dict, list, BaseModel via `create()`)
- `sources`: Content provenance — SHA256 hashes of source documents or external references (URLs, file paths). A SHA256 must not appear in both sources and origins.
- `origins`: Causal provenance — SHA256 hashes of documents that caused this document to be created without contributing to its content.
- `attachments`: Tuple of `Attachment` objects for multi-part content

Documents support:
- Automatic content serialization based on file extension: `.json` → JSON, `.yaml`/`.yml` → YAML, others → UTF-8 text. Structured data (dict, list, BaseModel) requires `.json` or `.yaml` extension.
- MIME type detection via `mime_type` cached property, with `is_text`/`is_image`/`is_pdf` helpers
- SHA256-based identity and deduplication
- Source provenance tracking (`sources` for references, `origins` for parent lineage)
- `FILES` enum for filename restrictions (definition-time validation)
- `model_convert()` for type conversion between document subclasses
- `canonical_name()` for standardized snake_case class identification
- Token count estimation via `approximate_tokens_count`

### Document Store

Documents are automatically persisted by `@pipeline_task` to a `DocumentStore`. The store is a protocol with three implementations:

- **ClickHouseDocumentStore**: Production backend (selected when `CLICKHOUSE_HOST` is configured). Requires `clickhouse-connect` (included in dependencies).
- **LocalDocumentStore**: CLI/debug mode (filesystem-based, browsable files on disk)
- **MemoryDocumentStore**: Testing (in-memory, zero I/O)

**Store selection depends on the execution mode:**
- `run_cli()`: Always uses `LocalDocumentStore` (files saved to the working directory)
- `run_local()`: Always uses `MemoryDocumentStore` (in-memory, no persistence)
- `as_prefect_flow()`: Auto-selects based on settings -- `ClickHouseDocumentStore` when `CLICKHOUSE_HOST` is set, `LocalDocumentStore` otherwise

**Store protocol methods:**
- `save(document, run_scope)` -- Save a single document (idempotent)
- `save_batch(documents, run_scope)` -- Save multiple documents
- `load(run_scope, document_types)` -- Load documents by type
- `has_documents(run_scope, document_type)` -- Check if documents exist for a type
- `check_existing(sha256s)` -- Check which SHA256 hashes exist in the store
- `update_summary(run_scope, document_sha256, summary)` -- Update summary for a stored document
- `load_summaries(run_scope, document_sha256s)` -- Load summaries by SHA256
- `flush()` -- Block until all pending background work (summaries) is processed
- `shutdown()` -- Flush pending work and stop background workers

**Document summaries:** When a `SummaryGenerator` callable is provided, stores automatically generate LLM-powered summaries in the background after each new document is saved (including empty and binary documents). Summaries are best-effort (failures are logged and skipped) and stored as store-level metadata (not on the `Document` model). Configure via `DOC_SUMMARY_ENABLED` and `DOC_SUMMARY_MODEL` environment variables.

**Note:** Store implementations are not exported from the top-level package. Import from submodules:

```python
from ai_pipeline_core.document_store.local import LocalDocumentStore
from ai_pipeline_core.document_store.memory import MemoryDocumentStore
from ai_pipeline_core.document_store.clickhouse import ClickHouseDocumentStore
```

### LLM Integration

The framework provides a unified interface for LLM interactions with context caching:

```python
from ai_pipeline_core import llm, AIMessages, ModelOptions

# Simple generation
response = await llm.generate(
    model="gemini-3-pro",
    messages="Explain quantum computing"
)
print(response.content)

# With context caching (saves 50-90% tokens on repeated calls)
static_context = AIMessages([large_document])

# First call: caches context (default TTL is 300s / 5 minutes)
r1 = await llm.generate(
    model="gemini-3-pro",
    context=static_context,
    messages="Summarize"
)

# Second call: reuses cache
r2 = await llm.generate(
    model="gemini-3-pro",
    context=static_context,
    messages="Key points?"
)

# Multi-turn conversation
messages = AIMessages([
    "What is Python?",
    r1,  # ModelResponse from previous call
    "Can you give an example?"
])
response = await llm.generate("gemini-3-pro", messages=messages)

# Observability: purpose labels traces, expected_cost tracks budget
response = await llm.generate(
    model="gemini-3-pro",
    messages="Analyze this",
    purpose="source-verification",
    expected_cost=0.05,
)
```

**`generate()` signature:**
```python
async def generate(
    model: ModelName,
    *,
    context: AIMessages | None = None,   # Static cacheable content
    messages: AIMessages | str,           # Dynamic query
    options: ModelOptions | None = None,  # Usually omit (defaults are optimal)
    purpose: str | None = None,           # Span name for tracing
    expected_cost: float | None = None,   # Cost tracking attribute
) -> ModelResponse
```

**`generate_structured()` signature:**
```python
async def generate_structured(
    model: ModelName,
    response_format: type[T],             # Pydantic model class
    *,
    context: AIMessages | None = None,
    messages: AIMessages | str,
    options: ModelOptions | None = None,
    purpose: str | None = None,
    expected_cost: float | None = None,
) -> StructuredModelResponse[T]
```

**`ModelOptions` key fields (all optional with sensible defaults):**
- `cache_ttl`: Context cache TTL (default `"300s"`, set `None` to disable)
- `system_prompt`: System-level instructions
- `reasoning_effort`: `"low" | "medium" | "high"` for models with explicit reasoning
- `search_context_size`: `"low" | "medium" | "high"` for search-enabled models
- `retries`: Retry attempts (default `3`)
- `retry_delay_seconds`: Delay between retries (default `20`)
- `timeout`: Max wait seconds (default `600`)
- `service_tier`: `"auto" | "default" | "flex" | "scale" | "priority"` (OpenAI only)
- `max_completion_tokens`: Max output tokens
- `temperature`: Generation randomness (usually omit -- use provider defaults)

**ModelName predefined values:** `"gemini-3-pro"`, `"gpt-5.1"`, `"gemini-3-flash"`, `"gpt-5-mini"`, `"grok-4.1-fast"`, `"gemini-3-flash-search"`, `"sonar-pro-search"` (also accepts any string for custom models).

### Pipeline Decorators

#### `@pipeline_task`

Decorates async functions as traced Prefect tasks with automatic document persistence:

```python
from ai_pipeline_core import pipeline_task

@pipeline_task  # No parameters needed for most cases
async def process_chunk(document: InputDocument) -> OutputDocument:
    return OutputDocument.create(
        name="result.json",
        content={"processed": True},
        sources=(document.sha256,),
    )

@pipeline_task(retries=3, estimated_minutes=5)
async def expensive_task(data: str) -> OutputDocument:
    # Retries, tracing, and document auto-save handled automatically
    ...

@pipeline_task(persist=False)  # Disable auto-save for this task
async def transient_task(data: str) -> OutputDocument:
    ...
```

Key parameters:
- `persist`: Auto-save returned documents to store (default `True`)
- `retries`: Retry attempts on failure (default `0` -- no retries unless specified)
- `estimated_minutes`: Duration estimate for progress tracking (default `1`, must be >= 1)
- `timeout_seconds`: Task execution timeout
- `trace_level`: `"always" | "debug" | "off"` (default `"always"`)
- `user_summary`: Enable LLM-generated span summaries (default `False`)
- `expected_cost`: Expected cost budget for cost tracking

Key features:
- Async-only enforcement (raises `TypeError` if not `async def`)
- Laminar tracing (automatic)
- Document auto-save to DocumentStore (returned documents are extracted and persisted)
- Source validation (warns if referenced SHA256s don't exist in store)

#### `@pipeline_flow`

Decorates async flow functions with annotation-driven document type extraction. Always requires parentheses:

```python
from ai_pipeline_core import pipeline_flow, FlowOptions

@pipeline_flow(estimated_minutes=10, retries=2, timeout_seconds=1200)
async def my_flow(
    project_name: str,
    documents: list[InputDoc],       # Input types extracted from annotation
    flow_options: MyFlowOptions,     # Must be FlowOptions or subclass
) -> list[OutputDoc]:                # Output types extracted from annotation
    ...
```

The flow's `documents` parameter annotation determines input types, and the return annotation determines output types. The function must have exactly 3 parameters: `(str, list[...], FlowOptions)`. No separate config class needed -- the type contract is in the function signature.

**FlowOptions** is a base `BaseSettings` class for pipeline configuration. Subclass it to add flow-specific parameters:

```python
class ResearchOptions(FlowOptions):
    analysis_model: ModelName = "gemini-3-pro"
    verification_model: ModelName = "grok-4.1-fast"
    synthesis_model: ModelName = "gemini-3-pro"
    max_sources: int = 10
```

#### `PipelineDeployment`

Orchestrates multi-flow pipelines with resume, uploads, and webhooks:

```python
class MyPipeline(PipelineDeployment[MyOptions, MyResult]):
    flows: ClassVar = [flow_1, flow_2, flow_3]

    @staticmethod
    def build_result(
        project_name: str,
        documents: list[Document],
        options: MyOptions,
    ) -> MyResult:
        ...
```

**Execution modes:**

```python
pipeline = MyPipeline()

# CLI mode: parses sys.argv, requires positional working_directory argument
# Usage: python script.py ./output [--start N] [--end N] [--max-keywords 8]
pipeline.run_cli(initializer=init_fn, trace_name="my-pipeline")

# Local mode: in-memory store, returns result directly (synchronous)
result = pipeline.run_local(
    project_name="test",
    documents=input_docs,
    options=MyOptions(),
)

# Production: generates a Prefect flow for deployment
prefect_flow = pipeline.as_prefect_flow()
```

Features:
- **Per-flow resume**: Skips flows whose output documents already exist in the store
- **Type chain validation**: At class definition time, validates that each flow's input types are producible by preceding flows
- **Per-flow uploads**: Upload documents after each flow completes
- **CLI mode**: `--start N` / `--end N` for step control, automatic `LocalDocumentStore`

### Image Processing

The `images` module provides image splitting and compression for LLM vision models:

```python
from ai_pipeline_core.images import process_image, process_image_to_documents, ImagePreset

# Process an image with model-specific presets
result = process_image(screenshot_bytes, preset=ImagePreset.GEMINI)
for part in result:
    print(part.label, len(part.data))

# Convert to Document objects for AIMessages
image_docs = process_image_to_documents(screenshot_bytes, name_prefix="screenshot")
```

Available presets: `GEMINI` (3000px, 9M pixels), `CLAUDE` (1568px, 1.15M pixels), `GPT4V` (2048px, 4M pixels).

The LLM client automatically splits oversized images at the model boundary -- you typically don't need to call these functions directly.

### Prompt Manager

Jinja2 template management for structured prompts:

```python
from ai_pipeline_core import PromptManager

# Module-level initialization (uses __file__ for relative template discovery)
prompts = PromptManager(__file__, prompts_dir="templates")

# Render a template
prompt = prompts.get("analyze.jinja2", source_id="example.com", task="summarize")
```

Globals available in all templates: `current_date` (formatted as "01 February 2026").

### Local Trace Debugging

When running via `run_cli()`, trace spans are automatically saved to `<working_dir>/.trace/` for
LLM-assisted debugging. Disable with `--no-trace`.

The directory structure mirrors the execution flow:

```
.trace/
  20260128_152932_abc12345_my_flow/
  |-- _trace.yaml           # Trace metadata
  |-- _tree.yaml            # Lightweight tree structure
  |-- _llm_calls.yaml       # LLM-specific details (tokens, cost, purpose)
  |-- _errors.yaml          # Failed spans only (written only if errors exist)
  |-- _summary.md           # Static execution summary (always generated)
  |-- artifacts/            # Deduplicated content storage
  |   +-- sha256/
  |       +-- ab/cd/        # Sharded by hash prefix
  |           +-- abcdef...1234.txt
  +-- 0001_my_flow/         # Root span (numbered for execution order)
      |-- _span.yaml        # Span metadata (timing, status, attributes, I/O refs)
      |-- input.yaml
      |-- output.yaml
      |-- events.yaml       # OTel span events (log records, etc.)
      +-- 0002_task_1/
          +-- 0003_llm_call/
              |-- _span.yaml
              |-- input.yaml
              +-- output.yaml
```

Up to 20 traces are kept (oldest are automatically cleaned up).

## Configuration

### Environment Variables

```bash
# LLM Configuration (via LiteLLM proxy)
OPENAI_BASE_URL=http://localhost:4000
OPENAI_API_KEY=your-api-key

# Optional: Observability
LMNR_PROJECT_API_KEY=your-lmnr-key
LMNR_DEBUG=true  # Enable debug traces

# Optional: Orchestration
PREFECT_API_URL=http://localhost:4200/api
PREFECT_API_KEY=your-prefect-key
PREFECT_API_AUTH_STRING=your-auth-string
PREFECT_WORK_POOL_NAME=default
PREFECT_WORK_QUEUE_NAME=default
PREFECT_GCS_BUCKET=your-gcs-bucket

# Optional: GCS (for remote storage)
GCS_SERVICE_ACCOUNT_FILE=/path/to/service-account.json

# Optional: Document Store & Tracking (ClickHouse -- omit for local filesystem store)
CLICKHOUSE_HOST=your-clickhouse-host
CLICKHOUSE_PORT=8443
CLICKHOUSE_DATABASE=default
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your-password
CLICKHOUSE_SECURE=true
TRACKING_ENABLED=true
TRACKING_SUMMARY_MODEL=gemini-3-flash

# Optional: Document Summaries (store-level, LLM-generated)
DOC_SUMMARY_ENABLED=true
DOC_SUMMARY_MODEL=gemini-3-flash
```

### Settings Management

Create custom settings by inheriting from the base Settings class:

```python
from ai_pipeline_core import Settings

class ProjectSettings(Settings):
    """Project-specific configuration."""
    app_name: str = "my-app"
    max_retries: int = 3

# Create singleton instance
settings = ProjectSettings()

# Access configuration (all env vars above are available)
print(settings.openai_base_url)
print(settings.app_name)
```

## Best Practices

### Framework Rules

1. **Decorators**: Use `@pipeline_task` without parameters for most cases, `@pipeline_flow(estimated_minutes=N)` with annotations (always requires parentheses)
2. **Logging**: Use `get_pipeline_logger(__name__)` -- never `print()` or `logging` module directly
3. **LLM calls**: Use `AIMessages` or `str` for messages. Wrap Documents in `AIMessages`
4. **Options**: Omit `ModelOptions` unless specifically needed (defaults are production-optimized)
5. **Documents**: Create with just `name` and `content` -- skip `description`. Always subclass `Document`
6. **Flow annotations**: Input/output types are in the function signature -- `list[InputDoc]` and `-> list[OutputDoc]`
7. **Initialization**: `PromptManager` and logger at module scope, not in functions
8. **Document lists**: Use plain `list[Document]` -- no wrapper class needed

### Import Convention

Always import from the top-level package when possible:

```python
# CORRECT - top-level imports
from ai_pipeline_core import Document, pipeline_flow, pipeline_task, llm, AIMessages

# ALSO CORRECT - store implementations are NOT exported from top-level
from ai_pipeline_core.document_store.local import LocalDocumentStore
from ai_pipeline_core.document_store.memory import MemoryDocumentStore
```

## Development

### Running Tests

```bash
make test              # Run all tests
make test-cov          # Run with coverage report
make test-clickhouse   # ClickHouse integration tests (requires Docker)
```

### Code Quality

```bash
make check             # Run ALL checks (lint, typecheck, deadcode, semgrep, docstrings, tests)
make lint              # Ruff linting (27 rule sets)
make format            # Auto-format and auto-fix code with ruff
make typecheck         # Type checking with basedpyright (strict mode)
make deadcode          # Dead code detection with vulture
make semgrep           # Project-specific AST pattern checks (.semgrep/ rules)
make docstrings-cover  # Docstring coverage (100% required)
```

**Static analysis tools:**
- **Ruff** — 27 rule sets including bugbear, security (bandit), complexity, async enforcement, exception patterns
- **Basedpyright** — strict mode with `reportUnusedCoroutine`, `reportUnreachable`, `reportImplicitStringConcatenation`
- **Vulture** — dead code detection with framework-aware whitelist
- **Semgrep** — custom rules in `.semgrep/` for frozen model mutable fields, async enforcement, docstring quality, architecture constraints
- **Interrogate** — 100% docstring coverage enforcement

### AI Documentation

```bash
make docs-ai-build  # Generate .ai-docs/ from source code
make docs-ai-check  # Validate .ai-docs/ freshness and completeness
```

## Examples

The `examples/` directory contains:

- **`showcase.py`** -- Full pipeline demonstrating Document types, `@pipeline_task` auto-save, `@pipeline_flow` annotations, `PipelineDeployment`, and CLI mode
- **`showcase_document_store.py`** -- DocumentStore usage patterns: MemoryDocumentStore, LocalDocumentStore, RunContext scoping, pipeline tasks with auto-save, and `run_local()` execution

Run examples:
```bash
# CLI mode with output directory
python examples/showcase.py ./output

# With custom options
python examples/showcase.py ./output --max-keywords 8

# Document store showcase (no arguments needed)
python examples/showcase_document_store.py
```

## Project Structure

```
ai-pipeline-core/
|-- ai_pipeline_core/
|   |-- deployment/       # Pipeline deployment, deploy script, progress, remote
|   |-- docs_generator/   # AI-focused documentation generator
|   |-- document_store/   # Store protocol and backends (ClickHouse, local, memory)
|   |-- documents/        # Document system (Document base class, attachments, context)
|   |-- images/           # Image processing for LLM vision models
|   |-- llm/              # LLM client, AIMessages, ModelOptions, ModelResponse
|   |-- logging/          # Logging infrastructure
|   |-- observability/    # Tracing, tracking, and debug trace writer
|   |-- pipeline/         # Pipeline decorators and FlowOptions
|   |-- prompt_manager.py # Jinja2 template management
|   |-- settings.py       # Configuration management (Pydantic BaseSettings)
|   |-- testing.py        # Prefect test harness re-exports
|   +-- exceptions.py     # Framework exceptions (LLMError, DocumentNameError, etc.)
|-- tests/                # Comprehensive test suite
|-- examples/             # Usage examples
|-- .specification/       # Framework requirements and documentation spec
+-- pyproject.toml        # Project configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes following the project's style guide
4. Run all checks (`make check`)
5. Commit your changes
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/bbarwik/ai-pipeline-core/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bbarwik/ai-pipeline-core/discussions)

## Acknowledgments

- Built on [Prefect](https://www.prefect.io/) for workflow orchestration
- Uses [LiteLLM](https://github.com/BerriAI/litellm) for LLM provider abstraction
- Integrates [Laminar (LMNR)](https://www.lmnr.ai/) for observability
- Type checking with [Pydantic](https://pydantic.dev/) and [basedpyright](https://github.com/DetachHead/basedpyright)
