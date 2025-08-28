# AI Pipeline Core - Project Explainer

## What is AI Pipeline Core?

AI Pipeline Core is a high-performance, type-safe Python library designed for building production-grade AI-powered data processing pipelines. It provides a robust foundation for orchestrating complex AI workflows with a focus on asynchronous operations, strong typing, and minimal design principles.

## Core Philosophy

The library follows several strict principles:
1. **Minimalism**: Every line of code must justify its existence
2. **Async-First**: All I/O operations are asynchronous
3. **Type Safety**: Complete type hints with Pydantic models throughout
4. **Production Ready**: Built-in retry logic, caching, monitoring, and error handling

## Problem It Solves

Building AI pipelines involves many challenges:
- Managing documents and data flow between processing steps
- Handling LLM interactions with proper retry logic and cost tracking
- Orchestrating complex multi-step workflows
- Ensuring type safety and data validation
- Providing observability and monitoring
- Managing configuration and secrets

AI Pipeline Core provides a unified, opinionated solution to these challenges.

## Architecture Overview

### 1. Document System (Foundation Layer)
The document system is the foundation for all data handling:
- **Document**: Abstract base class for all documents with automatic MIME type detection
- **FlowDocument**: Documents that persist across Prefect flow runs
- **TaskDocument**: Temporary documents within task execution
- **TemporaryDocument**: Documents that are never persisted
- **DocumentList**: Type-validated container for documents

Documents handle:
- Content encoding/decoding
- MIME type detection
- Serialization/deserialization
- Validation and size limits
- Conversion between formats (JSON, YAML, Markdown)

### 2. LLM Module (AI Interaction Layer)
Manages all AI model interactions through LiteLLM proxy:
- **generate()**: Main entry for text generation with context/message splitting
- **generate_structured()**: Returns Pydantic model responses
- **AIMessages**: Type-safe message container with automatic document conversion
- **ModelOptions**: Configuration for retries, timeouts, and model parameters
- **ModelResponse**: Wrapped responses with cost tracking and metadata

Key features:
- Smart context caching for token efficiency
- Automatic retry with exponential backoff
- Cost tracking and usage monitoring
- Support for search-enabled and reasoning models

### 3. Flow System (Orchestration Layer)
Type-safe workflow definitions:
- **FlowConfig**: Defines input/output document types for flows
- **FlowOptions**: Runtime configuration for flows
- **pipeline_flow/pipeline_task**: Enhanced decorators with tracing

Ensures:
- Type safety between pipeline steps
- Validation of inputs and outputs
- Configuration management
- Integration with Prefect for orchestration

### 4. Tracing System (Observability Layer)
LMNR integration for monitoring:
- **@trace**: Decorator for automatic observability
- **TraceInfo**: Metadata container for tracing context
- Automatic cost tracking and performance metrics
- Test environment detection

### 5. Logging System (Diagnostic Layer)
Unified logging with Prefect integration:
- **LoggerMixin**: Consistent logging across components
- **StructuredLoggerMixin**: Structured logging with events and metrics
- **PrefectLoggerMixin**: Flow and task-specific logging
- Automatic context switching between flow/task and regular logging

### 6. Simple Runner (Execution Layer)
CLI and programmatic execution:
- **run_cli()**: Parse CLI arguments and execute pipelines
- **run_pipeline()**: Execute single pipeline with validation
- **load/save_documents_from_directory()**: File system persistence
- Automatic test harness setup when needed

### 7. Configuration (Settings Layer)
- **Settings**: Central configuration using Pydantic
- Environment variable loading
- Validation of all configuration values

## How It All Works Together

1. **Document Creation**: Users create document subclasses for their specific data types
2. **Flow Definition**: Define flows with specific input/output document types using FlowConfig
3. **LLM Processing**: Within flows/tasks, use the LLM module to process documents
4. **Orchestration**: Prefect handles the execution, retries, and scheduling
5. **Observability**: Automatic tracing and logging throughout execution
6. **Execution**: Simple runner provides CLI interface for running pipelines

## Typical Workflow

```python
# 1. Define document types
class InputDocument(FlowDocument):
    def get_type(self) -> str:
        return "input"

class OutputDocument(FlowDocument):
    def get_type(self) -> str:
        return "output"

# 2. Define flow configuration
class MyFlowConfig(FlowConfig):
    INPUT_DOCUMENT_TYPES = [InputDocument]
    OUTPUT_DOCUMENT_TYPE = OutputDocument

# 3. Create flow with LLM processing
@pipeline_flow
async def process_flow(
    project_name: str,
    documents: DocumentList,
    flow_options: FlowOptions
) -> DocumentList:
    input_doc = documents[0]

    # Use LLM to process
    response = await generate_structured(
        model="gpt-5",
        response_format=MyModel,
        messages=AIMessages([input_doc])
    )

    # Return processed documents
    return DocumentList([
        OutputDocument("output.json", response.parsed)
    ])

# 4. Run via CLI
run_cli(
    flows=[process_flow],
    flow_configs=[MyFlowConfig],
    options_cls=FlowOptions
)
```

## Key Design Decisions

1. **Async Everything**: No blocking operations allowed, ensuring high throughput
2. **Document Abstraction**: All data flows through typed Document objects
3. **Context/Message Split**: LLM context is cached separately from dynamic messages
4. **Immutable Configuration**: All configs use Pydantic with frozen=True
5. **No Raw Dicts**: Everything is a Pydantic model for type safety
6. **Decorator-Based**: Clean API through decorators for flows, tasks, and tracing
7. **File System Persistence**: Documents saved in canonical directories

## Integration Points

- **LiteLLM**: Universal interface for all LLM providers
- **Prefect**: Workflow orchestration and scheduling
- **LMNR**: Observability and monitoring
- **Pydantic**: Data validation and serialization
- **Jinja2**: Template rendering for prompts

## Error Handling

The library uses a hierarchical exception system:
- **PipelineCoreError**: Base for all exceptions
- **DocumentError**: Document-related issues
- **LLMError**: AI model interaction failures
- **PromptError**: Template rendering problems

All errors include detailed context for debugging.

## Performance Optimizations

1. **Context Caching**: Reuse expensive LLM context across calls
2. **Connection Pooling**: HTTP clients use persistent connections
3. **Streaming**: Large documents are streamed, not loaded entirely
4. **Async I/O**: Non-blocking operations throughout
5. **Batch Processing**: Support for parallel execution via Prefect

## Testing Strategy

- Async test fixtures with prefect_test_harness
- Automatic test environment detection
- Mock external services (OpenAI, LMNR)
- Type checking with basedpyright
- Coverage target >80%

This architecture enables building robust, scalable, and maintainable AI pipelines with strong guarantees about type safety and error handling.
