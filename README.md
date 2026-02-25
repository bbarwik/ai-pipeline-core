# AI Pipeline Core

A high-performance async framework for building type-safe AI pipelines with LLMs, document processing, and workflow orchestration.

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Basedpyright](https://img.shields.io/badge/type%20checked-basedpyright-blue)](https://github.com/DetachHead/basedpyright)

## Overview

AI Pipeline Core is a production-ready framework that combines document processing, LLM integration, and workflow orchestration into a unified system. Built with strong typing (Pydantic), automatic retries, cost tracking, and distributed tracing, it enforces best practices while keeping application code minimal and straightforward.

### Key Features

- **Document System**: Single `Document` base class with immutable content, SHA256-based identity, automatic MIME type detection, provenance tracking, multi-part attachments, and optional typed content via `Document[ModelType]`
- **Document Store**: Pluggable storage backends (ClickHouse production, local filesystem CLI/debug, in-memory testing) with automatic deduplication
- **Conversation Class**: Immutable, stateful multi-turn LLM conversations with context caching, automatic URL/address shortening, and eager response restoration
- **LLM Integration**: Unified interface to any model via LiteLLM proxy (OpenRouter compatible) with context caching (default 300s TTL)
- **Structured Output**: Type-safe generation with Pydantic model validation via `Conversation.send_structured()`
- **Workflow Orchestration**: Prefect-based flows and tasks with annotation-driven document types and decoration-time input/output validation
- **Auto-Persistence**: `@pipeline_task` saves returned documents to `DocumentStore` automatically
- **Image Processing**: Automatic image tiling/splitting for LLM vision models with model-specific presets
- **Observability**: Built-in distributed tracing via Laminar (LMNR) with cost tracking, local trace debugging, and ClickHouse-based tracking
- **Prompt Compiler**: Type-safe prompt specifications replacing Jinja2 templates — typed Python classes for roles, rules, guides, and output formats with definition-time validation and a CLI tool for inspection
- **Deployment**: Unified pipeline execution for local, CLI, and production environments with per-flow resume and dual-store support

## Installation

```bash
pip install ai-pipeline-core
```

This installs two CLI commands:
- `ai-prompt-compiler` — discover, inspect, render, and compile prompt specifications
- `ai-pipeline-deploy` — build and deploy pipelines to Prefect Cloud

### Requirements

- Python 3.12 or higher
- Linux/macOS (Windows via WSL2)
- [uv](https://astral.sh/uv) (recommended)

### Development Installation

```bash
git clone https://github.com/bbarwik/ai-pipeline-core.git
cd ai-pipeline-core
make install-dev     # Initializes uv environment and installs pre-commit hooks
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
        derived_from=(document.sha256,),
    )


# 4. Pipeline flow -- type contract is in the annotations
@pipeline_flow(estimated_minutes=5)
async def analysis_flow(
    run_id: str,
    documents: list[InputDocument],
    flow_options: FlowOptions,
) -> list[AnalysisDocument]:
    results: list[AnalysisDocument] = []
    for doc in documents:
        results.append(await analyze_document(doc))
    return results


@pipeline_flow(estimated_minutes=2)
async def report_flow(
    run_id: str,
    documents: list[AnalysisDocument],
    flow_options: FlowOptions,
) -> list[ReportDocument]:
    report = ReportDocument.create(
        name="report.md",
        content="# Report\n\nAnalysis complete.",
        derived_from=tuple(doc.sha256 for doc in documents),
    )
    return [report]


# 5. Deployment -- ties flows together with type chain validation
class MyResult(DeploymentResult):
    report_count: int = 0


class MyPipeline(PipelineDeployment[FlowOptions, MyResult]):
    flows: ClassVar = [analysis_flow, report_flow]

    @staticmethod
    def build_result(
        run_id: str,
        documents: list[Document],
        options: FlowOptions,
    ) -> MyResult:
        reports = [d for d in documents if isinstance(d, ReportDocument)]
        return MyResult(success=True, report_count=len(reports))


# 6. CLI initializer provides run ID and initial documents
def initialize(options: FlowOptions) -> tuple[str, list[Document]]:
    docs: list[Document] = [
        InputDocument.create_root(name="input.txt", content="Sample data", reason="CLI input"),
    ]
    return "my-project", docs


# Run from CLI (requires positional working_directory arg: python script.py ./output)
pipeline = MyPipeline()
pipeline.run_cli(initializer=initialize, trace_name="my-pipeline")
```

### Conversation (Multi-Turn LLM)

```python
from ai_pipeline_core.llm import Conversation, ModelOptions

# Create a conversation with model and optional context
conv = Conversation(model="gemini-3-pro")

# Add documents to cacheable context prefix (shared across forks)
conv = conv.with_context(doc1, doc2, doc3)

# Add a document to dynamic messages suffix (NOT cached, per-fork content)
conv = conv.with_document(my_document)

# Send a message (returns NEW immutable Conversation instance — always capture!)
conv = await conv.send("Analyze the document")
print(conv.content)  # Response text

# Structured output
conv = await conv.send_structured("Extract key points", response_format=KeyPoints)
print(conv.parsed)  # KeyPoints instance

# Multi-turn: each send() appends to conversation history
conv = await conv.send("Now summarize the key points")
print(conv.content)

# Access response properties
print(conv.reasoning_content)  # Thinking/reasoning text (if available)
print(conv.usage)              # Token usage with input/output counts
print(conv.cost)               # Estimated cost
print(conv.citations)          # Citation objects (for search models)
```

### Structured Output

```python
from pydantic import BaseModel
from ai_pipeline_core import Conversation

class Analysis(BaseModel):
    summary: str
    sentiment: float
    key_points: list[str]

# Generate structured output via Conversation
conv = Conversation(model="gemini-3-pro")
conv = await conv.send_structured(
    "Analyze this product review: ...",
    response_format=Analysis,
)

# Access parsed result with type safety
analysis = conv.parsed
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
    content={"key": "value"},  # Automatically converted to JSON bytes
    derived_from=("https://api.example.com/data",),
)

# Parse back to original type
data = doc.parse(dict)  # Returns {"key": "value"}

# Document provenance tracking
source_doc = MyDocument.create_root(name="source.txt", content="original data", reason="user upload")
plan_doc = MyDocument.create(name="plan.txt", content="research plan", derived_from=(source_doc.sha256,))
derived = MyDocument.create(
    name="derived.json",
    content={"result": "processed"},
    derived_from=("https://api.example.com/data",),  # Content came from this URL
    triggered_by=(plan_doc.sha256,),  # Created because of this plan (causal, not content)
)

# Check provenance
for hash in derived.content_documents:
    print(f"Derived from document: {hash}")
for ref in derived.content_references:
    print(f"External source: {ref}")
```

### Typed Content (Document[T])

Declare a Pydantic BaseModel as the content schema for a Document subclass. Content is validated at creation time and accessible via `.parsed`:

```python
from pydantic import BaseModel
from ai_pipeline_core import Document

class ResearchDefinition(BaseModel, frozen=True):
    topic: str
    max_sources: int = 10

class ResearchPlanDocument(Document[ResearchDefinition]):
    """Plan document with typed content schema."""

# Content is validated against the schema at creation time
plan = ResearchPlanDocument.create(
    name="plan.json",
    content=ResearchDefinition(topic="AI safety"),
    derived_from=(input_doc.sha256,),
)

# Zero-boilerplate typed access (cached, returns ResearchDefinition)
definition = plan.parsed
print(definition.topic)       # "AI safety"
print(definition.max_sources)  # 10

# Wrong schema type is rejected at creation time
class WrongModel(BaseModel, frozen=True):
    x: int

plan = ResearchPlanDocument.create(
    name="plan.json",
    content=WrongModel(x=1),   # TypeError: Expected content of type ResearchDefinition
    derived_from=(input_doc.sha256,),
)

# Introspection
ResearchPlanDocument.get_content_type()  # ResearchDefinition
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
    content={"key": "value"},  # Auto-converts to JSON
    derived_from=(source.sha256,),
)

# Access content
if doc.is_text:
    print(doc.text)

# Parse structured data
data = doc.as_json()  # or as_yaml()
model = doc.as_pydantic_model(MyModel)  # Requires model_type argument

# Convert between document types
other = doc.retype(OtherDocType, preserve_provenance=True)

# Content-addressed identity
print(doc.sha256)  # Full SHA256 hash (base32)
print(doc.id)      # Short 6-char identifier
```

**Typed content** — declare a content schema via generic parameter for automatic validation and typed access:

```python
class PlanDocument(Document[PlanModel]):
    """Content is validated against PlanModel at creation time."""

plan = PlanDocument.create(name="plan.json", content=PlanModel(...), derived_from=(...,))
plan.parsed  # → PlanModel (cached, typed)
PlanDocument.get_content_type()  # → PlanModel
```

**Document fields:**
- `name`: Filename (validated for security -- no path traversal)
- `description`: Optional human-readable description
- `content`: Raw bytes (auto-converted from str, dict, list, BaseModel via `create()`)
- `derived_from`: Content provenance — SHA256 hashes of source documents or URI-style references (must contain `://`). A SHA256 must not appear in both derived_from and triggered_by.
- `triggered_by`: Causal provenance — SHA256 hashes of documents that caused this document to be created without contributing to its content.
- `attachments`: Tuple of `Attachment` objects for multi-part content

Documents support:
- Automatic content serialization based on file extension: `.json` → JSON, `.yaml`/`.yml` → YAML, others → UTF-8 text. Structured data (dict, list, BaseModel) requires `.json` or `.yaml` extension.
- Optional typed content schema via `Document[ModelType]` with creation-time validation and `.parsed` access
- MIME type detection via `mime_type` cached property, with `is_text`/`is_image`/`is_pdf` helpers
- SHA256-based identity and deduplication
- Provenance tracking (`derived_from` for content sources, `triggered_by` for causal lineage)
- `FILES` enum for filename restrictions (definition-time validation)
- `retype(new_type, preserve_provenance=True|False)` for type conversion between document subclasses
- `derive(from_documents=..., name=..., content=...)` convenience method for creating documents from other documents (extracts SHA256 hashes automatically)
- Token count estimation via `approximate_tokens_count`

### Document Store

Documents are automatically persisted by `@pipeline_task` to a document store. The framework provides four backend implementations (ClickHouse, local filesystem, in-memory, dual), all managed internally — application code interacts only through the read-only `DocumentReader` protocol.

**Backend implementations** (internal, auto-selected by execution mode):
- **ClickHouse**: Production backend (selected when `CLICKHOUSE_HOST` is configured)
- **Local filesystem**: CLI/debug mode (browsable files on disk, collision-safe filenames)
- **In-memory**: Testing (zero I/O, used by `run_local()`)
- **Dual**: Wraps primary (ClickHouse) + secondary (local). Saves to both, reads from primary only

**Store selection depends on the execution mode:**
- `run_cli()`: Uses dual store (ClickHouse + local) when `CLICKHOUSE_HOST` is configured, local otherwise
- `run_local()`: Always uses in-memory store (no persistence)
- `as_prefect_flow()`: Auto-selects based on settings

**Public API — `DocumentReader`** (read-only, available via `get_document_store()`):
- `load(run_scope, document_types)` -- Load documents by type from a run scope
- `has_documents(run_scope, document_type, *, max_age=None)` -- Check if documents of a type exist
- `check_existing(sha256s)` -- Check which SHA256 hashes exist globally
- `find_by_source(source_values, document_type, *, max_age=None)` -- Find most recent document per source value
- `load_summaries(document_sha256s)` -- Load summaries by SHA256 (global)
- `load_by_sha256s(sha256s, document_type, run_scope=None)` -- Batch-load full documents by SHA256
- `load_nodes_by_sha256s(sha256s)` -- Batch-load lightweight `DocumentNode` metadata (global, cross-scope)
- `load_scope_metadata(run_scope)` -- Load `DocumentNode` metadata for all documents in a scope
- `get_flow_completion(run_scope, flow_name, *, max_age=None)` -- Check flow completion record (returns `FlowCompletion | None`)

Write operations (`save`, `save_batch`, `flush`, `shutdown`) are framework-internal — `@pipeline_task` handles persistence automatically.

**Document summaries:** When enabled, stores automatically generate LLM-powered summaries in the background after each new document is saved. Summaries are best-effort and stored as store-level metadata (not on the `Document` model). Configure via `DOC_SUMMARY_ENABLED` and `DOC_SUMMARY_MODEL` environment variables.

### LLM Integration

The primary interface is the **`Conversation`** class for multi-turn interactions.

#### Conversation (Recommended)

The `Conversation` class provides immutable, stateful conversation management:

```python
from ai_pipeline_core.llm import Conversation, ModelOptions

# Create with model and optional configuration
conv = Conversation(model="gemini-3-pro")

# Add documents to cacheable context prefix (shared across forks)
conv = conv.with_context(doc1, doc2, doc3)

# Add a document to dynamic messages suffix (NOT cached, per-fork content)
conv = conv.with_document(my_document)

# Configure model options
conv = conv.with_model_options(ModelOptions(
    system_prompt="You are a research analyst.",
    reasoning_effort="high",
))

# Send a message (returns NEW Conversation instance)
conv = await conv.send("Analyze the document")
print(conv.content)  # Response text

# Multi-turn: conversation history is preserved
conv = await conv.send("Now summarize the key points")
print(conv.content)

# Structured output
conv = await conv.send_structured("Extract entities", response_format=Entities)
print(conv.parsed)  # Entities instance

# Add multiple documents at once
conv = conv.with_documents([doc1, doc2, doc3])

# Inject prior assistant output (e.g., from another conversation)
conv = conv.with_assistant_message("Previous analysis result...")

# Warmup + fork pattern for parallel calls with shared cache
import asyncio
base = await conv.send("Acknowledge the context")  # Warmup
# Fork: create parallel conversations from the same base
results = await asyncio.gather(
    base.send("Analyze source 1"),
    base.send("Analyze source 2"),
    base.send("Analyze source 3"),
)

# Approximate token count for all context and messages
print(conv.approximate_tokens_count)
```

**`send_spec()`** — sends a `PromptSpec` to the LLM. Handles document placement, stop sequences, and auto-extraction of `<result>` tags. For structured specs (`PromptSpec[SomeModel]`), dispatches to `send_structured()` automatically.

**Content protection (automatic):** URLs, blockchain addresses, and high-entropy strings in context documents are automatically shortened to `prefix...suffix` forms to save tokens. Both `.content` and `.parsed` are eagerly restored after every `send()`/`send_structured()` call — no manual restoration needed. A fuzzy fallback handles LLM-mangled forms (dropped suffix, prefix/suffix truncated by 1-2 chars).

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
- `stop`: Stop sequences (tuple of strings, used internally by `send_spec` for `</result>` tags)

**ModelName predefined values:** `"gemini-3-pro"`, `"gpt-5.1"`, `"gemini-3-flash"`, `"gpt-5-mini"`, `"grok-4.1-fast"`, `"gemini-3-flash-search"`, `"gpt-5-mini-search"`, `"grok-4.1-fast-search"`, `"sonar-pro-search"` (also accepts any string for custom models).

### Image Processing

Image processing for LLM vision models is available from the `llm._images` module:

```python
from ai_pipeline_core.llm._images import process_image, ImagePreset

# Process an image with model-specific presets
result = process_image(screenshot_bytes, preset=ImagePreset.GEMINI)
for part in result:
    print(part.label, len(part.data))
```

Available presets: `GEMINI` (3000px, 9M pixels), `CLAUDE` (1568px, 1.15M pixels), `GPT4V` (2048px, 4M pixels).

**Token cost:** A single image consumes **1080 tokens** regardless of pixel dimensions.

The `Conversation` class automatically splits oversized images when documents are added to context — you typically don't need to call `process_image` directly.

### Exceptions

The framework re-exports key exceptions at the top level for convenient catching:

```python
from ai_pipeline_core import PipelineCoreError, LLMError, DocumentValidationError, DocumentSizeError, DocumentNameError
```

- `PipelineCoreError` — Base for all framework exceptions
- `LLMError` — LLM generation failures (retries exhausted, timeouts, degeneration)
- `DocumentValidationError` — Document validation failures
- `DocumentSizeError` — Document exceeds size limits
- `DocumentNameError` — Invalid document name (path traversal, etc.)

Output degeneration (token repetition loops) is detected automatically and raises `LLMError` after retry exhaustion.

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
        derived_from=(document.sha256,),
    )

@pipeline_task(retries=3, estimated_minutes=5)
async def expensive_task(data: str) -> OutputDocument:
    # Retries, tracing, and document auto-save handled automatically
    ...

```

Key parameters:
- `retries`: Retry attempts on failure (default `0` -- no retries unless specified)
- `estimated_minutes`: Duration estimate for progress tracking (default `1`, must be >= 1)
- `timeout_seconds`: Task execution timeout
- `trace_level`: `"always" | "debug" | "off"` (default `"always"`)
- `expected_cost`: Expected cost budget for cost tracking

Key features:
- Async-only enforcement (raises `TypeError` if not `async def`)
- Decoration-time return type validation (must return Document types or None)
- Decoration-time input type validation (all parameters must be annotated with serializable types)
- Laminar tracing (automatic)
- Document auto-save to DocumentStore (returned documents are extracted and persisted)
- Source validation (warns if referenced SHA256s don't exist in store)

**Input type validation** — all `@pipeline_task` parameters must be annotated with serializable types. Allowed: `str`, `int`, `float`, `bool`, `Path`, `UUID`, `None`, `Any`, Document subclasses, FlowOptions subclasses, frozen BaseModel, Enum, `NewType` wrappers of valid types, and `list`/`tuple`/`dict[str, ...]`/`Union` containers thereof. Non-serializable types (mutable models, plain classes) are rejected at decoration time.

#### `@pipeline_flow`

Decorates async flow functions with annotation-driven document type extraction. Always requires parentheses:

```python
from ai_pipeline_core import pipeline_flow, FlowOptions

@pipeline_flow(estimated_minutes=10, retries=2, timeout_seconds=1200)
async def my_flow(
    run_id: str,                     # Must be named 'run_id'
    documents: list[InputDoc],       # Input types extracted from annotation
    flow_options: MyFlowOptions,     # Must be FlowOptions or subclass
) -> list[OutputDoc]:                # Output types extracted from annotation
    ...
```

The flow's `documents` parameter annotation determines input types, and the return annotation determines output types. The function must have exactly 3 parameters. The first must be named `run_id: str`, followed by `documents: list[...]` and `flow_options: FlowOptions`. No separate config class needed -- the type contract is in the function signature.

**FlowOptions** is a base `BaseSettings` class for pipeline configuration. Subclass it to add flow-specific parameters:

```python
class ResearchOptions(FlowOptions):
    analysis_model: ModelName = "gemini-3-pro"
    verification_model: ModelName = "grok-4.1-fast"
    synthesis_model: ModelName = "gemini-3-pro"
    max_sources: int = 10
```

#### `PipelineDeployment`

Orchestrates multi-flow pipelines with resume, per-flow uploads, and event publishing:

```python
class MyPipeline(PipelineDeployment[MyOptions, MyResult]):
    flows: ClassVar = [flow_1, flow_2, flow_3]

    @staticmethod
    def build_result(
        run_id: str,
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
    run_id="test",
    documents=input_docs,
    options=MyOptions(),
)

# Production: generates a Prefect flow for deployment
prefect_flow = pipeline.as_prefect_flow()
```

Features:
- **Per-flow resume**: Skips flows with a `FlowCompletion` record in the store (explicit completion tracking, not document-presence inference). Configurable `cache_ttl` (default 24h)
- **Type chain validation**: At class definition time, validates that at least one of each flow's declared input types is producible by preceding flows (union semantics)
- **Event publishing**: Per-flow start/completion notifications via Pub/Sub (`pubsub_project_id`, `pubsub_topic_id`)
- **Concurrency limits**: Cross-run enforcement via Prefect global concurrency limits
- **CLI mode**: `--start N` / `--end N` for step control, `DualDocumentStore` when ClickHouse is configured

#### Concurrency Limits

Declare cross-run concurrency and rate limits on `PipelineDeployment` to prevent exceeding external API quotas across all concurrent pipeline runs:

```python
from ai_pipeline_core import LimitKind, PipelineLimit, PipelineDeployment, pipeline_concurrency

class MyPipeline(PipelineDeployment[MyOptions, MyResult]):
    flows = [my_flow]
    concurrency_limits = {
        "provider-a": PipelineLimit(500, LimitKind.CONCURRENT),       # max 500 simultaneous
        "provider-b": PipelineLimit(15, LimitKind.PER_MINUTE, timeout=300),  # 15/min token bucket
    }
    ...
```

Use `pipeline_concurrency()` at call sites to acquire slots:

```python
from ai_pipeline_core import pipeline_concurrency

async def fetch_data(url: str) -> Data:
    async with pipeline_concurrency("provider-a"):
        return await provider.fetch(url)
```

**Limit kinds:**
- `CONCURRENT` — Lease-based slots held during operation, released on exit
- `PER_MINUTE` — Token bucket with `limit/60` decay per second (allows bursting)
- `PER_HOUR` — Token bucket with `limit/3600` decay per second

**Behavior:**
- Limits are auto-created in Prefect server at pipeline start (idempotent upsert)
- Timeout raises `AcquireConcurrencySlotTimeoutError` (limit doing its job)
- When Prefect is unavailable, limits proceed unthrottled (logged as warning)
- Limit names are validated at class definition time (alphanumeric, dashes, underscores)

#### Parallel Execution

`safe_gather` and `safe_gather_indexed` run coroutines in parallel with fault tolerance:

```python
from ai_pipeline_core import safe_gather, safe_gather_indexed

# safe_gather: returns successes only, filters out failures
results = await safe_gather(
    process(doc1), process(doc2), process(doc3),
    label="processing",
)  # Returns list of successful results (order may shift)

# safe_gather_indexed: preserves positional correspondence (None for failures)
results = await safe_gather_indexed(
    process(doc1), process(doc2), process(doc3),
    label="processing",
)  # Returns [result1, None, result3] if doc2 failed
```

Both raise if all coroutines fail (configurable via `raise_if_all_fail=False`).

#### Deploying to Prefect Cloud

The framework includes a deploy script that builds a fully bundled deployment (project wheel + all dependency wheels), uploads to GCS, and creates a Prefect deployment. The worker installs fully offline with `--no-index` — no PyPI contact, no stale cache issues.

```bash
# From your project root (where pyproject.toml lives)
ai-pipeline-deploy

# Also available as module:
python -m ai_pipeline_core.deployment.deploy
```

**Requirements:**
- `uv` (dependency resolution and wheel download) on the deploy machine
- `PREFECT_API_URL`, `PREFECT_GCS_BUCKET` configured
- `uv` on the worker (for offline install)

#### Remote Deployment Client

`RemoteDeployment` is a typed client for calling a remote `PipelineDeployment` via Prefect. Name the client class identically to the server's deployment class so the auto-derived deployment name matches:

```python
from ai_pipeline_core import RemoteDeployment, DeploymentResult, FlowOptions, Document

class RemoteInputDocument(Document):
    """Mirror type -- class_name must match the remote pipeline's document type."""

class RemoteResult(DeploymentResult):
    """Result type matching the remote pipeline's result."""
    report_count: int = 0

class MyPipeline(RemoteDeployment[RemoteInputDocument, FlowOptions, RemoteResult]):
    """Client for the remote MyPipeline deployment."""

client = MyPipeline()
result = await client.run(
    run_id="test",
    documents=input_docs,
    options=FlowOptions(),
    on_progress=my_progress_callback,  # Optional: async (fraction, message) -> None
)
```

The client defines local Document subclasses ("mirror types") whose `class_name` must match the remote pipeline's document types exactly. `run_remote_deployment()` is also available as a lower-level function.

### Prompt Compiler

Type-safe prompt specifications that replace Jinja2 templates. Every piece of prompt content is a class or class attribute, validated at definition time (import time).

**Components** — define once, reuse across specs:

```python
from ai_pipeline_core import Role, Rule, OutputRule, Guide

class ResearchAnalyst(Role):
    """Analyst role for research pipelines."""
    text = "experienced research analyst with expertise in data synthesis"

class CiteEvidence(Rule):
    """Citation rule."""
    text = "Always cite specific evidence from the source documents.\nInclude document IDs when referencing."

class DontUseMarkdownTables(OutputRule):
    """Table formatting rule."""
    text = "Do not use markdown tables in the output."

class RiskFrameworkGuide(Guide):
    """Risk assessment framework guide."""
    template = "guides/risk_framework.md"  # Relative to module file, loaded at import time
```

**Specs** — typed prompt definitions with full validation:

```python
from ai_pipeline_core import PromptSpec, Document
from pydantic import Field

class SourceDocument(Document):
    """Source material for analysis."""

class AnalysisSpec(PromptSpec):
    """Analyze source documents for key findings."""
    role = ResearchAnalyst
    input_documents = (SourceDocument,)
    task = "Analyze the provided documents and identify key findings."
    rules = (CiteEvidence,)
    guides = (RiskFrameworkGuide,)
    output_structure = "## Key Findings\n## Evidence\n## Gaps"
    output_rules = (DontUseMarkdownTables,)

    # Dynamic fields — become template variables
    project_name: str = Field(description="Project name")
```

**Multi-line fields** — use `MultiLineField` for long or multiline content (e.g., review feedback, website content). All multi-line fields are combined into a single XML-tagged user message sent before the main prompt, not inlined in the Context section. Regular `Field()` values must be short, single-line strings (up to 500 chars) — exceeding this raises `ValueError` at render time:

```python
from ai_pipeline_core import PromptSpec, MultiLineField
from pydantic import Field

class ReviewSpec(PromptSpec):
    """Analyze a review."""
    role = ResearchAnalyst
    input_documents = (SourceDocument,)
    task = "Analyze the review and identify key themes."

    project_name: str = Field(description="Project name")          # Short, inline in prompt
    review: str = MultiLineField(description="Review text")        # Sent as <review>...</review> message
```

**Rendering and sending:**

```python
from ai_pipeline_core import Conversation, render_text, render_preview

# Create spec instance with dynamic field values
spec = AnalysisSpec(project_name="ACME")

# Render prompt text (for inspection/debugging)
prompt = render_text(spec, documents=[source_doc])

# Preview with placeholder values (for debugging)
preview = render_preview(AnalysisSpec)

# Send to LLM via Conversation
conv = await Conversation(model="gemini-3-flash").send_spec(spec, documents=[source_doc])
print(conv.content)  # <result> tags auto-extracted by send_spec()
```

**Structured output** — `output_structure` automatically enables `<result>` tag wrapping, sets a stop sequence at `</result>`, and auto-extracts the content in `Conversation.send_spec()`. `conv.content` returns clean text without tags. Structured output (`PromptSpec[SomeModel]`) uses `send_structured()` automatically.

**Follow-up specs** — use `follows=ParentSpec` to declare follow-up specs. Follow-up specs inherit context from the parent conversation and don't require `role` or `input_documents`.

**CLI tool** for discovery, inspection, rendering, and compilation:

```bash
# Inspect a spec's anatomy (role, docs, fields, rules, output config, token estimate)
ai-prompt-compiler inspect AnalysisSpec

# Render a prompt preview
ai-prompt-compiler render AnalysisSpec

# Discover, list, and compile all specs to .prompts/ directory as markdown files
ai-prompt-compiler compile

# Explicit module:class reference
ai-prompt-compiler render my_package.specs:AnalysisSpec

# Also available as module:
python -m ai_pipeline_core.prompt_compiler inspect AnalysisSpec
```

### Local Trace Debugging

When running via `run_cli()`, trace spans are automatically saved to `<working_dir>/.trace/` for
LLM-assisted debugging. Disable with `--no-trace`.

The directory structure mirrors the execution flow. Each new run overwrites the previous trace:

```
.trace/
  summary.md              # Execution tree, timing, LLM stats
  llm_calls.yaml          # All LLM calls with model, tokens, cost
  errors.yaml             # Failed spans with parent chain (only if errors exist)
  costs.md                # Cost aggregation by task
  001_my_flow/            # Root span (3-digit prefix, max 20 char name)
      span.yaml           # Span metadata (timing, status, type, I/O refs)
      input.yaml          # Span input (block scalar YAML for multiline)
      output.yaml         # Span output
      events.yaml         # OTel span events (log records)
      002_task_1/
          span.yaml
          input.yaml
          output.yaml
          003_analyze/
              span.yaml
              input.yaml
              output.yaml
```

Duplicate LLM spans are filtered by default (every `Conversation` call creates both a DEFAULT and an inner LLM span). Use `verbose=True` in `TraceDebugConfig` to include all spans.

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

# Optional: Document Summaries (store-level, LLM-generated)
DOC_SUMMARY_ENABLED=true
DOC_SUMMARY_MODEL=gemini-3-flash

# Optional: Pub/Sub event delivery (deployment progress/status)
PUBSUB_PROJECT_ID=your-gcp-project
PUBSUB_TOPIC_ID=pipeline-events
SERVICE_TYPE=your-service-name
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
3. **LLM calls**: Use `Conversation` for all LLM interactions (multi-turn and single-shot)
4. **Options**: Omit `ModelOptions` unless specifically needed (defaults are production-optimized)
5. **Documents**: Use `create_root()` for pipeline inputs (no provenance), `create()` or `derive()` for derived documents. Always subclass `Document`
6. **Flow annotations**: Input/output types are in the function signature -- `list[InputDoc]` and `-> list[OutputDoc]`
7. **Initialization**: Logger at module scope, not in functions
8. **Document lists**: Use plain `list[Document]` -- no wrapper class needed

### Import Convention

Always import from the top-level package when possible:

```python
# Top-level imports (preferred)
from ai_pipeline_core import Document, pipeline_flow, pipeline_task, Conversation

# Sub-package imports for symbols not at top level
from ai_pipeline_core.deployment import CompletedRun, DeploymentResultData
from ai_pipeline_core.llm import ModelOptions
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
make lint              # Ruff linting (28 rule sets)
make format            # Auto-format and auto-fix code with ruff
make typecheck         # Type checking with basedpyright (strict mode)
make deadcode          # Dead code detection with vulture
make semgrep           # Project-specific AST pattern checks (.semgrep/ rules)
make docstrings-cover  # Docstring coverage (100% required)
```

**Static analysis tools:**
- **Ruff** — 28 rule sets including bugbear, security (bandit), complexity, async enforcement, exception patterns
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

- **`showcase.py`** -- Full 3-stage pipeline: Conversation API, multi-turn LLM analysis, structured extraction, PipelineDeployment with CLI, resume/skip, progress tracking, image processing
- **`showcase_document_store.py`** -- Document store usage: pipeline tasks with auto-save, flow execution via `run_local()`, document provenance tracking
- **`showcase_prompt_compiler.py`** -- Prompt compiler features: Role, Rule, OutputRule, Guide, PromptSpec, rendering, output_structure, follow-up specs, definition-time validation

Run examples:
```bash
# Full pipeline showcase (requires OPENAI_BASE_URL and OPENAI_API_KEY)
python examples/showcase.py ./output

# Document store showcase (no arguments needed)
python examples/showcase_document_store.py

# Prompt compiler showcase (no arguments needed)
python examples/showcase_prompt_compiler.py
```

## Project Structure

```
ai-pipeline-core/
|-- ai_pipeline_core/
|   |-- _llm_core/        # Internal LLM client, model types, and response handling
|   |-- deployment/        # Pipeline deployment, deploy script, CLI bootstrap, progress, remote
|   |-- docs_generator/    # AI-focused documentation generator
|   |-- document_store/    # Store protocol and backends (ClickHouse, local, memory)
|   |-- documents/         # Document system (Document base class, attachments, context)
|   |-- llm/               # Conversation class, URLSubstitutor, image processing, validation
|   |-- logging/           # Logging infrastructure
|   |-- observability/     # Tracing, tracking, and debug trace writer
|   |-- pipeline/          # Pipeline decorators, FlowOptions, and concurrency limits
|   |-- prompt_compiler/   # Type-safe prompt specs, rendering, and CLI tool
|   |-- settings.py        # Configuration management (Pydantic BaseSettings)
|   +-- exceptions.py      # Framework exceptions (LLMError, DocumentNameError, etc.)
|-- tests/                 # Comprehensive test suite
|-- examples/              # Usage examples
+-- pyproject.toml         # Project configuration
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
- Uses [LiteLLM](https://github.com/BerriAI/litellm) for LLM provider abstraction (also compatible with [OpenRouter](https://openrouter.ai/))
- Integrates [Laminar (LMNR)](https://www.lmnr.ai/) for observability
- Type checking with [Pydantic](https://pydantic.dev/) and [basedpyright](https://github.com/DetachHead/basedpyright)
