# AI Pipeline Core

A high-performance, type-safe Python library for building AI-powered data processing pipelines with Prefect orchestration and LMNR observability.

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Basedpyright](https://img.shields.io/badge/type%20checked-basedpyright-blue)](https://github.com/DetachHead/basedpyright)
[![Test Coverage](https://img.shields.io/badge/coverage-80%25-green)](https://github.com/bbarwik/ai-pipeline-core)
[![Status: Beta](https://img.shields.io/badge/status-beta-yellow)](https://github.com/bbarwik/ai-pipeline-core)
[![PyPI version](https://img.shields.io/pypi/v/ai-pipeline-core.svg)](https://pypi.org/project/ai-pipeline-core/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ai-pipeline-core.svg)](https://pypi.org/project/ai-pipeline-core/)

> [!NOTE]
> **Beta Release**
>
> This library is in beta. While actively used in production systems, the API may still evolve. We follow semantic versioning for releases.

## Overview

AI Pipeline Core provides a robust foundation for building production-grade AI pipelines with a focus on:

- **100% Async Architecture** - Built for high-throughput, non-blocking operations
- **Type Safety** - Comprehensive type hints with Pydantic models throughout
- **Minimal Design** - Every line of code justified, no unnecessary abstractions
- **Production Ready** - Built-in retry logic, caching, monitoring, and error handling
- **LLM Optimization** - Smart context/message splitting for efficient token usage

## Key Features

### 🚀 Performance First
- Fully asynchronous I/O operations
- Intelligent caching for LLM context
- Streaming support for large documents
- Automatic retry with exponential backoff

### 🔒 Type Safety
- Pydantic models for all data structures
- Strict type checking with basedpyright
- Runtime validation for all inputs
- Immutable configurations by default

### 📊 Observability
- LMNR (Laminar) tracing integration
- Structured logging with Prefect
- Cost tracking for LLM operations
- Performance metrics out of the box

### 🎯 Developer Experience
- Self-documenting code for experienced developers
- Consistent patterns throughout
- Comprehensive error messages
- Smart defaults with override capabilities

### 🤖 Advanced LLM Features
- Search-enabled models (Perplexity Sonar, Gemini Flash Search)
- Reasoning models support (O1 series)
- Structured output with Pydantic models
- Dynamic model selection based on task

## Installation

```bash
pip install ai-pipeline-core
```

### Development Installation

For contributors and development:

```bash
git clone https://github.com/bbarwik/ai-pipeline-core.git
cd ai-pipeline-core
pip install -e ".[dev]"
make install-dev  # Installs pre-commit hooks
```

### Requirements
- Python 3.12 or higher
- Linux/macOS (Windows via WSL2)

## Quick Start

### Basic Document Processing
```python
from ai_pipeline_core.documents import Document, FlowDocument
from ai_pipeline_core.llm import generate_structured, AIMessages, ModelOptions
from pydantic import BaseModel

class InputDocument(FlowDocument):
    """Custom document type for your flow"""
    def get_type(self) -> str:
        return "input"

class AnalysisResult(BaseModel):
    """Example Pydantic model for structured output"""
    summary: str
    key_points: list[str]

async def process_document(doc: Document):
    # Generate AI response with structured output
    response = await generate_structured(
        model="gemini-2.5-pro",  # Model is required first parameter
        response_format=AnalysisResult,  # Pydantic model class
        context=AIMessages([doc]),  # Cached context
        messages=AIMessages(["Analyze this document"]),  # Dynamic messages
        options=ModelOptions(max_completion_tokens=5000)  # Optional options
    )
    return response.parsed
```

### Enhanced Pipeline Decorators (New in v0.1.7)
```python
from ai_pipeline_core import pipeline_flow, pipeline_task
from ai_pipeline_core.flow import FlowOptions
from ai_pipeline_core.documents import DocumentList, FlowDocument

class CustomFlowOptions(FlowOptions):
    """Extend base options with your custom fields"""
    batch_size: int = 100
    temperature: float = 0.7

@pipeline_task(trace_level="always", retries=3)
async def process_task(doc: Document) -> Document:
    # Task with automatic tracing and retries
    result = await process_document(doc)
    return OutputDocument(name="result", content=result.encode())

@pipeline_flow(trace_level="always")
async def my_pipeline(
    project_name: str,
    documents: DocumentList,
    flow_options: CustomFlowOptions  # Type-safe custom options
) -> DocumentList:
    # Pipeline flow with enforced signature and tracing
    results = []
    for doc in documents:
        result = await process_task(doc)
        results.append(result)
    return DocumentList(results)
```

### Simple Runner Utility (New in v0.1.7)
```python
from ai_pipeline_core.simple_runner import run_cli, run_pipeline
from ai_pipeline_core.flow import FlowOptions

# CLI-based pipeline execution
if __name__ == "__main__":
    run_cli(
        flows=[my_pipeline],
        flow_configs=[MyFlowConfig],
        options_cls=CustomFlowOptions
    )

# Or programmatic execution
async def main():
    result = await run_pipeline(
        project_name="my-project",
        output_dir=Path("./output"),
        flow=my_pipeline,
        flow_config=MyFlowConfig,
        flow_options=CustomFlowOptions(batch_size=50)
    )
```

### Clean Prefect Decorators (New in v0.1.7)
```python
# Import clean Prefect decorators without tracing
from ai_pipeline_core.prefect import flow, task

# Or use pipeline decorators with tracing
from ai_pipeline_core import pipeline_flow, pipeline_task

@task  # Clean Prefect task
def compute(x: int) -> int:
    return x * 2

@pipeline_task(trace_level="always")  # With tracing
def compute_traced(x: int) -> int:
    return x * 2
```

## Core Modules

### Documents System
The foundation for all data handling. Documents are immutable, type-safe wrappers around content with automatic MIME type detection.

```python
from ai_pipeline_core.documents import Document, DocumentList

# Documents handle encoding/decoding automatically
doc = MyDocument(
    name="report.pdf",
    content=pdf_bytes,
    description="Q3 Financial Report"
)

# Type-safe document collections
docs = DocumentList([doc1, doc2])
```

### LLM Module
Managed AI interactions with built-in retry logic, cost tracking, and structured outputs.

**Supported Models** (via LiteLLM proxy):
- OpenAI: GPT-4, GPT-5 series
- Anthropic: Claude 3 series
- Google: Gemini 2.5 series
- xAI: Grok models
- Perplexity: Sonar models (with search capabilities)
- And many more through LiteLLM compatibility

```python
from ai_pipeline_core.llm import generate_structured, AIMessages, ModelOptions
from pydantic import BaseModel

class YourPydanticModel(BaseModel):
    field1: str
    field2: int

# Get structured Pydantic model responses
result = await generate_structured(
    model="gemini-2.5-pro",  # Model is required first parameter
    response_format=YourPydanticModel,  # Pydantic model class for structured output
    context=AIMessages(),  # Optional context (cached)
    messages=AIMessages(["Your prompt here"]),  # Required messages
    options=ModelOptions(
        retries=3,
        timeout=30,
        max_completion_tokens=10000
    )
)
# Access the parsed result
model_instance = result.parsed  # Type: YourPydanticModel
```

### Prompt Management
Flexible Jinja2-based prompt system with smart path resolution.

```python
from ai_pipeline_core import PromptManager

pm = PromptManager(__file__)
prompt = pm.get("analyze_document.jinja2",
                 document=doc,
                 instructions=instructions)
```

### Tracing & Monitoring
Automatic observability with LMNR integration.

```python
from ai_pipeline_core.tracing import trace

@trace(metadata={"workflow": "analysis"})
async def analyze_data(data: InputData) -> OutputData:
    # Automatic tracing with performance metrics
    ...
```

## Architecture Principles

### 1. Async-First Design
Every I/O operation is asynchronous. No blocking calls, no synchronous fallbacks.

### 2. Type Safety Throughout
Complete type annotations with runtime validation. If it compiles, it works.

### 3. Minimal Surface Area
Less code is better code. Every line must justify its existence.

### 4. Configuration as Code
All configurations are Pydantic models - validated, typed, and immutable.

## Project Structure

```
ai_pipeline_core/
├── documents/          # Document handling system
│   ├── document.py     # Base document class
│   ├── flow_document.py # Prefect flow documents
│   └── task_document.py # Prefect task documents
├── llm/               # LLM interaction layer
│   ├── client.py      # Async client implementation
│   └── model_options.py # Configuration models
├── flow/              # Prefect flow utilities
│   ├── config.py      # Type-safe flow configuration
│   └── options.py     # FlowOptions base class (v0.1.7)
├── simple_runner/     # Pipeline execution utilities (v0.1.7)
│   ├── cli.py         # CLI interface
│   └── simple_runner.py # Core runner logic
├── logging/           # Structured logging
├── pipeline.py        # Enhanced decorators (v0.1.7)
├── prefect.py         # Clean Prefect exports (v0.1.7)
├── tracing.py         # Observability decorators
└── settings.py        # Centralized configuration
```

## Development

### Running Tests
```bash
make test           # Run all tests
make test-cov      # Run with coverage report
pytest tests/test_documents.py::TestDocument::test_creation  # Single test
```

### Code Quality
```bash
make lint          # Run linting checks
make format        # Auto-format code
make typecheck     # Run type checking
make pre-commit    # Run all pre-commit hooks
```

### Development Workflow
1. Create feature branch
2. Write tests first (TDD)
3. Implement minimal solution
4. Run `make format` and `make typecheck`
5. Ensure >80% test coverage
6. Submit PR with clear description

## Best Practices

### DO ✅
- Use async/await for all I/O operations
- Define Pydantic models for all data structures
- Keep functions under 20 lines
- Use type hints for everything
- Let Documents handle serialization

### DON'T ❌
- Import `logging` directly (use pipeline logger)
- Use raw dictionaries for configuration
- Write defensive code for unlikely scenarios
- Add comments explaining what (code should be clear)
- Use `requests` or other blocking libraries

## Configuration

### Environment Variables
```bash
# Required for LLM operations
OPENAI_API_KEY=sk-...  # Your OpenAI or LiteLLM proxy key
OPENAI_BASE_URL=http://your-proxy:8000  # LiteLLM proxy endpoint

# Optional - for observability
LMNR_PROJECT_API_KEY=lmnr_...  # LMNR tracing

# Optional - for orchestration
PREFECT_API_URL=http://localhost:4200/api
AI_PIPELINE_LOG_LEVEL=INFO
```

### Settings Management
```python
from ai_pipeline_core.settings import settings

# All settings are validated Pydantic models
api_key = settings.openai_api_key
base_url = settings.openai_base_url  # LiteLLM proxy endpoint
```

## Integration Examples

### With Prefect Cloud
```python
from prefect import flow
from ai_pipeline_core.flow import FlowConfig

@flow(name="document-processor")
async def process_documents(docs: DocumentList):
    # Automatic Prefect Cloud integration
    ...
```

### With Custom LLM Providers
```python
from ai_pipeline_core.settings import settings

# Configure LiteLLM proxy endpoint via environment variables
# OPENAI_BASE_URL=http://your-litellm-proxy:8000
# OPENAI_API_KEY=your-proxy-key

# Access in code (settings are immutable)
base_url = settings.openai_base_url
```

## Performance Considerations

- **Context Caching**: The LLM module automatically caches context to reduce token usage
- **Document Streaming**: Large documents are streamed rather than loaded entirely into memory
- **Batch Processing**: Use Prefect's `.map()` for parallel task execution
- **Connection Pooling**: HTTP clients use connection pooling by default

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure Python 3.12+ is installed
2. **Async Warnings**: All I/O operations must use `await`
3. **Type Errors**: Run `make typecheck` to identify issues
4. **MIME Detection**: Install `python-magic` system dependencies

### Debug Mode
```python
from ai_pipeline_core.logging import setup_logging, LoggingConfig

# Setup logging with DEBUG level
setup_logging(LoggingConfig(level="DEBUG"))
```

## Release Process

See [RELEASE.md](RELEASE.md) for detailed release procedures.

**Important**: All releases require:
- ✅ Zero errors from `make typecheck`
- ✅ All unit tests passing with >80% coverage
- ✅ **Integration tests passing** (with configured API keys)

## Contributing

> [!NOTE]
> As this is a preview repository used internally, we are not actively accepting external contributions. The codebase may change significantly without notice.
>
> **Recommended approach:**
> 1. Fork the repository
> 2. Make changes in your fork
> 3. Share your improvements with the community through your fork

If you've found a critical security issue, please report it via the GitHub Security tab.

For learning purposes, see [CLAUDE.md](CLAUDE.md) for our comprehensive coding standards and architecture guide.

## Documentation

- [CLAUDE.md](CLAUDE.md) - Detailed coding standards and architecture guide

### dependencies_docs/ Directory
> [!NOTE]
> The `dependencies_docs/` directory contains guides for AI assistants (like Claude Code) on how to interact with the project's external dependencies and tooling, NOT user documentation for ai-pipeline-core itself. These files are excluded from repository listings to avoid confusion.

**AI Assistant Dependency Guides:**
- [Prefect Integration](dependencies_docs/prefect.md) - Prefect patterns and best practices for AI assistants
- [Deployment Guide](dependencies_docs/prefect_deployment.md) - Production deployment guide for AI assistants
- [Prefect Logging](dependencies_docs/prefect_logging.md) - Logging configuration guide for AI assistants

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

> [!CAUTION]
> This is a preview repository with no guaranteed support. Issues and discussions may not be actively monitored.

- **For Learning**: Review the code, documentation, and examples
- **For Usage**: Fork the repository and maintain your own version
- **Security Issues**: Report via GitHub Security tab

## Acknowledgments

Built with:
- [Prefect](https://www.prefect.io/) - Workflow orchestration
- [LMNR](https://www.lmnr.ai/) - LLM observability
- [LiteLLM](https://litellm.ai/) - LLM proxy
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation

## What's New in v0.1.7

### Major Additions
- **Enhanced Pipeline Decorators**: New `pipeline_flow` and `pipeline_task` decorators combining Prefect functionality with automatic LMNR tracing
- **FlowOptions Base Class**: Extensible configuration system for flows with type-safe inheritance
- **Simple Runner Module**: CLI and programmatic utilities for easy pipeline execution
- **Clean Prefect Exports**: Separate imports for Prefect decorators with and without tracing
- **Expanded Exports**: All major components now accessible from top-level package import

### API Improvements
- Better type inference for document flows with custom options
- Support for custom FlowOptions inheritance in pipeline flows
- Improved error messages for invalid flow signatures
- Enhanced document utility functions (`canonical_name_key`, `sanitize_url`)

### Developer Experience
- Simplified imports - most components available from `ai_pipeline_core` directly
- Better separation of concerns between clean Prefect and traced pipeline decorators
- More intuitive flow configuration with `FlowOptions` inheritance

## Stability Notice

**Current Version**: 0.1.7
**Status**: Internal Preview
**API Stability**: Unstable - Breaking changes expected
**Recommended Use**: Learning and reference only

For production use, please fork this repository and maintain your own stable version.
