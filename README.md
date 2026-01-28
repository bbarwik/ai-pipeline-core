# AI Pipeline Core

A high-performance async framework for building type-safe AI pipelines with LLMs, document processing, and workflow orchestration.

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Basedpyright](https://img.shields.io/badge/type%20checked-basedpyright-blue)](https://github.com/DetachHead/basedpyright)

## Overview

AI Pipeline Core is a production-ready framework that combines document processing, LLM integration, and workflow orchestration into a unified system. Built with strong typing (Pydantic), automatic retries, cost tracking, and distributed tracing, it enforces best practices while maintaining high performance through fully async operations.

### Key Features

- **Document Processing**: Type-safe handling of text, JSON, YAML, PDFs, and images with automatic MIME type detection and provenance tracking
- **LLM Integration**: Unified interface to any model via LiteLLM proxy with configurable context caching
- **Structured Output**: Type-safe generation with Pydantic model validation
- **Workflow Orchestration**: Prefect-based flows and tasks with automatic retries
- **Observability**: Built-in distributed tracing via Laminar (LMNR) with cost tracking for debugging and monitoring
- **Deployment**: Unified pipeline execution for local, CLI, and production environments

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
make install-dev  # Installs pre-commit hooks
```

## Quick Start

### Basic Pipeline

```python
from ai_pipeline_core import (
    pipeline_flow,
    FlowDocument,
    DocumentList,
    FlowOptions,
    FlowConfig,
    llm,
    AIMessages
)

# Define document types
class InputDoc(FlowDocument):
    """Input document for processing."""

class OutputDoc(FlowDocument):
    """Analysis result document."""

# Define flow configuration
class AnalysisConfig(FlowConfig):
    INPUT_DOCUMENT_TYPES = [InputDoc]
    OUTPUT_DOCUMENT_TYPE = OutputDoc

# Create pipeline flow with required config
@pipeline_flow(config=AnalysisConfig)
async def analyze_flow(
    project_name: str,
    documents: DocumentList,
    flow_options: FlowOptions
) -> DocumentList:
    # Process documents
    outputs = []
    for doc in documents:
        # Use AIMessages for LLM interaction
        response = await llm.generate(
            model="gpt-5.1",
            messages=AIMessages([doc])
        )

        output = OutputDoc.create(
            name=f"analysis_{doc.name}",
            content=response.content
        )
        outputs.append(output)

    # RECOMMENDED: Always validate output
    return AnalysisConfig.create_and_validate_output(outputs)
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
    model="gpt-5.1",
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
from ai_pipeline_core import FlowDocument, TemporaryDocument

# Create documents with automatic conversion
doc = MyDocument.create(
    name="data.json",
    content={"key": "value"}  # Automatically converted to JSON bytes
)

# Parse back to original type
data = doc.parse(dict)  # Returns {"key": "value"}

# Document provenance tracking
doc_with_sources = MyDocument.create(
    name="derived.json",
    content={"result": "processed"},
    sources=[source_doc.sha256, "https://api.example.com/data"]
)

# Check provenance
for hash in doc_with_sources.get_source_documents():
    print(f"Derived from document: {hash}")
for ref in doc_with_sources.get_source_references():
    print(f"External source: {ref}")

# Temporary documents (never persisted)
temp = TemporaryDocument.create(
    name="api_response.json",
    content={"status": "ok"}
)
```

## Core Concepts

### Documents

Documents are immutable Pydantic models that wrap binary content with metadata:

- **FlowDocument**: Persists across flow runs, saved to filesystem
- **TaskDocument**: Temporary within task execution, not persisted
- **TemporaryDocument**: Never persisted, useful for sensitive data

```python
class MyDocument(FlowDocument):
    """Custom document type."""

# Use create() for automatic conversion
doc = MyDocument.create(
    name="data.json",
    content={"key": "value"}  # Auto-converts to JSON
)

# Access content
if doc.is_text:
    print(doc.text)

# Parse structured data
data = doc.as_json()  # or as_yaml(), as_pydantic_model()

# Convert between document types
task_doc = flow_doc.model_convert(TaskDocument)  # Convert FlowDocument to TaskDocument
new_doc = doc.model_convert(OtherDocType, content={"new": "data"})  # With content update

# Enhanced filtering
filtered = documents.filter_by([Doc1, Doc2, Doc3])  # Multiple types
named = documents.filter_by(["file1.txt", "file2.txt"])  # Multiple names

# Immutable collections
frozen_docs = DocumentList(docs, frozen=True)  # Immutable document list
frozen_msgs = AIMessages(messages, frozen=True)  # Immutable message list
```

### LLM Integration

The framework provides a unified interface for LLM interactions with smart caching:

```python
from ai_pipeline_core import llm, AIMessages, ModelOptions

# Simple generation
response = await llm.generate(
    model="gpt-5.1",
    messages="Explain quantum computing"
)
print(response.content)

# With context caching (saves 50-90% tokens)
static_context = AIMessages([large_document])

# First call: caches context
r1 = await llm.generate(
    model="gpt-5.1",
    context=static_context,  # Cached for 120 seconds by default
    messages="Summarize"     # Dynamic query
)

# Second call: reuses cache
r2 = await llm.generate(
    model="gpt-5.1",
    context=static_context,  # Reused from cache!
    messages="Key points?"   # Different query
)

# Custom cache TTL
response = await llm.generate(
    model="gpt-5.1",
    context=static_context,
    messages="Analyze",
    options=ModelOptions(cache_ttl="300s")  # Cache for 5 minutes
)

# Disable caching for dynamic contexts
response = await llm.generate(
    model="gpt-5.1",
    context=dynamic_context,
    messages="Process",
    options=ModelOptions(cache_ttl=None)  # No caching
)
```

### Flow Configuration

Type-safe flow configuration ensures proper document flow:

```python
from ai_pipeline_core import FlowConfig

class ProcessingConfig(FlowConfig):
    INPUT_DOCUMENT_TYPES = [RawDataDocument]
    OUTPUT_DOCUMENT_TYPE = ProcessedDocument  # Must be different!

# Use in flows for validation
@pipeline_flow(config=ProcessingConfig)
async def process(
    project_name: str,
    documents: DocumentList,
    flow_options: FlowOptions
) -> DocumentList:
    # ... processing logic ...
    return ProcessingConfig.create_and_validate_output(outputs)
```

### Pipeline Decorators

Enhanced decorators with built-in tracing and monitoring:

```python
from ai_pipeline_core import pipeline_flow, pipeline_task, set_trace_cost

@pipeline_task  # Automatic retry, tracing, and monitoring
async def process_chunk(data: str) -> str:
    result = await transform(data)
    set_trace_cost(0.05)  # Track costs
    return result

@pipeline_flow(
    config=MyFlowConfig,
    trace_trim_documents=True  # Trim large documents in traces
)
async def main_flow(
    project_name: str,
    documents: DocumentList,
    flow_options: FlowOptions
) -> DocumentList:
    # Your pipeline logic
    # Large documents are automatically trimmed to 100 chars in traces
    # for better observability without overwhelming the tracing UI
    return DocumentList(results)
```

### Local Trace Debugging

Save all trace spans to the local filesystem for LLM-assisted debugging:

```bash
export TRACE_DEBUG_PATH=/path/to/debug/output
```

This creates a hierarchical directory structure that mirrors the execution flow with automatic deduplication:

```
20260128_152932_abc12345_my_flow/
├── _trace.yaml           # Trace metadata
├── _index.yaml           # Span ID → path mapping
├── _summary.md           # Unified summary for human inspection and LLM debugging
├── artifacts/            # Deduplicated content storage
│   └── sha256/
│       └── ab/cd/        # Sharded by hash prefix
│           └── abcdef...1234.txt  # Large content (>10KB)
└── 0001_my_flow/         # Root span (numbered for execution order)
    ├── _span.yaml        # Span metadata (timing, status, I/O refs)
    ├── input.yaml        # Structured inputs (inline or refs)
    ├── output.yaml       # Structured outputs (inline or refs)
    ├── 0002_task_1/      # Child spans nested inside parent
    │   ├── _span.yaml
    │   ├── input.yaml
    │   ├── output.yaml
    │   └── 0003_llm_call/
    │       ├── _span.yaml
    │       ├── input.yaml   # LLM messages with inline/external content
    │       └── output.yaml
    └── 0004_task_2/
        └── ...
```

**Key Features:**
- **Automatic Deduplication**: Identical content (e.g., system prompts) stored once in `artifacts/`
- **Smart Externalization**: Large content (>10KB) externalized with 2KB inline previews
- **AI-Friendly**: Files capped at 50KB for easy LLM processing
- **Lossless**: Full content reconstruction via `content_ref` pointers

Example `input.yaml` with externalization:
```yaml
format_version: 3
type: llm_messages
messages:
  - role: system
    parts:
      - type: text
        size_bytes: 28500
        content_ref:  # Large content → artifact
          hash: sha256:a1b2c3d4...
          path: artifacts/sha256/a1/b2/a1b2c3d4...txt
        excerpt: "You are a helpful assistant...\n[TRUNCATED]"
  - role: user
    parts:
      - type: text
        content: "Hello!"  # Small content stays inline
```

Run `tree` on the output directory to visualize the entire execution hierarchy. Feed `_summary.md` to an LLM for debugging assistance - it combines high-level overview with detailed navigation for comprehensive trace analysis.

## Configuration

### Environment Variables

```bash
# LLM Configuration (via LiteLLM proxy)
OPENAI_BASE_URL=http://localhost:4000
OPENAI_API_KEY=your-api-key

# Optional: Observability
LMNR_PROJECT_API_KEY=your-lmnr-key
LMNR_DEBUG=true  # Enable debug traces

# Optional: Local Trace Debugging
TRACE_DEBUG_PATH=/path/to/trace/output  # Save traces locally for LLM-assisted debugging

# Optional: Orchestration
PREFECT_API_URL=http://localhost:4200/api
PREFECT_API_KEY=your-prefect-key

# Optional: Storage (for Google Cloud Storage)
GCS_SERVICE_ACCOUNT_FILE=/path/to/service-account.json  # GCS auth file
```

### Settings Management

Create custom settings by inheriting from the base Settings class:

```python
from ai_pipeline_core import Settings

class ProjectSettings(Settings):
    """Project-specific configuration."""
    app_name: str = "my-app"
    max_retries: int = 3
    enable_cache: bool = True

# Create singleton instance
settings = ProjectSettings()

# Access configuration
print(settings.openai_base_url)
print(settings.app_name)
```

## Best Practices

### Framework Rules (90% Use Cases)

1. **Decorators**: Use `@pipeline_task` WITHOUT parameters, `@pipeline_flow` WITH config
2. **Logging**: Use `get_pipeline_logger(__name__)` - NEVER `print()` or `logging` module
3. **LLM calls**: Use `AIMessages` or `str`. Wrap Documents in `AIMessages`
4. **Options**: Omit `ModelOptions` unless specifically needed (defaults are optimal)
5. **Documents**: Create with just `name` and `content` - skip `description`
6. **FlowConfig**: `OUTPUT_DOCUMENT_TYPE` must differ from all `INPUT_DOCUMENT_TYPES`
7. **Initialization**: `PromptManager` and logger at module scope, not in functions
8. **DocumentList**: Use default constructor - no validation flags needed
9. **setup_logging()**: Only in application `main()`, never at import time

### Import Convention

Always import from the top-level package:

```python
# CORRECT
from ai_pipeline_core import llm, pipeline_flow, FlowDocument

# WRONG - Never import from submodules
from ai_pipeline_core.llm import generate  # NO!
from ai_pipeline_core.documents import FlowDocument  # NO!
```

## Development

### Running Tests

```bash
make test           # Run all tests
make test-cov      # Run with coverage report
make test-showcase # Test showcase example
```

### Code Quality

```bash
make lint      # Run linting
make format    # Auto-format code
make typecheck # Type checking with basedpyright
```

### Building Documentation

```bash
make docs-build  # Generate API.md
make docs-check  # Verify documentation is up-to-date
```

## Examples

The `examples/` directory contains:

- `showcase.py` - Comprehensive example demonstrating all major features
- Run with: `cd examples && python showcase.py /path/to/documents`

## API Reference

See [API.md](API.md) for complete API documentation.

### Navigation Tips

For humans:
```bash
grep -n '^##' API.md   # List all main sections
grep -n '^###' API.md  # List all classes and functions
```

For AI assistants:
- Use pattern `^##` to find module sections
- Use pattern `^###` for classes and functions
- Use pattern `^####` for methods and properties

## Project Structure

```
ai-pipeline-core/
├── ai_pipeline_core/
│   ├── deployment/      # Pipeline deployment and execution
│   ├── documents/       # Document abstraction system
│   ├── flow/            # Flow configuration and options
│   ├── llm/             # LLM client and response handling
│   ├── logging/         # Logging infrastructure
│   ├── prompt_builder/  # Document-aware prompt construction
│   ├── pipeline.py      # Pipeline decorators
│   ├── progress.py      # Intra-flow progress tracking
│   ├── prompt_manager.py # Jinja2 template management
│   ├── settings.py      # Configuration management
│   └── tracing.py       # Distributed tracing
├── tests/               # Comprehensive test suite
├── examples/            # Usage examples
├── API.md               # Complete API reference
└── pyproject.toml       # Project configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes following the project's style guide
4. Run tests and linting (`make test lint typecheck`)
5. Commit your changes
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/bbarwik/ai-pipeline-core/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bbarwik/ai-pipeline-core/discussions)
- **Documentation**: [API Reference](API.md)

## Acknowledgments

- Built on [Prefect](https://www.prefect.io/) for workflow orchestration
- Uses [LiteLLM](https://github.com/BerriAI/litellm) for LLM provider abstraction
- Integrates [Laminar (LMNR)](https://www.lmnr.ai/) for observability
- Type checking with [Pydantic](https://pydantic.dev/) and [basedpyright](https://github.com/DetachHead/basedpyright)
