#!/usr/bin/env python3
"""Complete showcase of ai_pipeline_core features (v0.1.14)

This example demonstrates ALL exports from ai_pipeline_core.__init__, including:
  • Settings configuration with environment variables and .env files
  • Logging system (LoggerMixin, StructuredLoggerMixin) with Prefect integration
  • Document system with immutable, type-safe documents:
    - Document: Abstract base with MIME detection and content validation
    - FlowDocument: Persisted across flow runs
    - TaskDocument: Temporary within task execution
    - TemporaryDocument: Never persisted (NEW in v0.1.9+)
    - DocumentList: Type-safe container with validation
  • Flow configuration (FlowConfig, FlowOptions) for type-safe pipelines
  • Pipeline decorators (pipeline_flow, pipeline_task) with LMNR tracing
    - NEW: Cost tracking via trace_cost parameter (v0.1.14+)
  • Prefect utilities (flow, task, prefect_test_harness, disable_run_logger)
  • LLM module with smart caching and structured outputs:
    - generate/generate_structured with context caching
    - AIMessages supporting mixed content types
    - ModelOptions with retry and timeout configuration
    - ModelResponse with cost tracking metadata
  • Tracing (trace, TraceLevel, TraceInfo) for observability with cost tracking
  • PromptManager for Jinja2 templates with smart path resolution
  • Simple runner module (run_cli, run_pipeline, load/save documents)

Prerequisites:
  - OPENAI_API_KEY and OPENAI_BASE_URL configured (can be set in .env)
  - Optional: LMNR_PROJECT_API_KEY for tracing
  - Optional: PREFECT_API_URL for remote orchestration

Usage:
  # Basic usage with defaults
  python examples/showcase.py ./output

  # With custom options
  python examples/showcase.py ./output --temperature 0.7 --batch-size 5

  # Skip first stage
  python examples/showcase.py ./output --start 2

Tip: Set PREFECT_LOGGING_LEVEL=INFO for richer logs and LMNR_DEBUG=true for more detailed tracing
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

# Import ALL exports from ai_pipeline_core.__init__
# Note: These are the public exports as of v0.1.12
from ai_pipeline_core import (
    AIMessages,
    Document,
    DocumentList,
    # Flow configuration
    FlowConfig,
    FlowDocument,
    FlowOptions,
    # Logging
    LoggingConfig,
    ModelName,
    ModelOptions,
    ModelResponse,
    # Utilities
    PromptManager,
    StructuredLoggerMixin,
    TaskDocument,
    TemporaryDocument,
    canonical_name_key,
    # Prefect utilities
    get_logger,
    get_pipeline_logger,
    # LLM components
    llm,
    pipeline_flow,
    pipeline_task,
    sanitize_url,
    set_trace_cost,
    setup_logging,
    trace,
)

# Clean Prefect imports (no tracing)
from ai_pipeline_core.prefect import flow, task

# Import simple_runner features
from ai_pipeline_core.simple_runner import run_cli

# -----------------------------------------------------------------------------
# Setup logging with all features
# -----------------------------------------------------------------------------
setup_logging(level="INFO")  # Can also pass config_path for YAML config
logger = get_pipeline_logger(__name__)

# Demonstrate both get_logger and get_pipeline_logger (they're aliases)
alt_logger = get_logger("showcase.alternative")


# -----------------------------------------------------------------------------
# Demonstrate LoggerMixin and StructuredLoggerMixin
# -----------------------------------------------------------------------------
class DataProcessor(StructuredLoggerMixin):
    """Example class using StructuredLoggerMixin for advanced logging."""

    def process(self, data: str) -> dict[str, Any]:
        # Use inherited logging methods
        self.log_info(f"Processing data of length {len(data)}")

        # Log structured event
        self.log_event("data_processing_started", data_length=len(data), processor="showcase")

        # Log metrics
        self.log_metric("input_size", len(data), "characters", source="showcase")

        # Use context manager for timed operations
        with self.log_operation("text_analysis", data_size=len(data)):
            # Simulate processing
            result = {"length": len(data), "words": len(data.split())}

        # Log span
        self.log_span("processing_complete", 100.5, status="success")

        return result


# -----------------------------------------------------------------------------
# Document types demonstrating all document features
# -----------------------------------------------------------------------------
class AllowedInputFiles(StrEnum):
    """Demonstrate FILES enum for document validation."""

    CONFIG = "config.yaml"
    DATA = "data.json"
    TEXT = "input.txt"


class InputDocument(FlowDocument):
    """Flow document with filename restrictions."""

    FILES: ClassVar[type[AllowedInputFiles]] = AllowedInputFiles


class AnalysisDocument(FlowDocument):
    """Output document demonstrating create methods."""

    pass


class EnhancedDocument(FlowDocument):
    """Enhanced document for stage 2 output."""

    pass


class TempProcessingDocument(TaskDocument):
    """Temporary task document (not persisted)."""

    pass


# -----------------------------------------------------------------------------
# Custom FlowOptions demonstrating extension
# -----------------------------------------------------------------------------
class ShowcaseFlowOptions(FlowOptions):
    """Extended flow options with custom fields."""

    temperature: float = Field(default=0.7, ge=0, le=2, description="LLM temperature")
    batch_size: int = Field(default=10, ge=1, description="Processing batch size")
    enable_structured: bool = Field(default=True, description="Use structured output")
    reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="medium", description="Reasoning level"
    )


# -----------------------------------------------------------------------------
# Flow configurations
# -----------------------------------------------------------------------------
class Stage1Config(FlowConfig):
    """First stage: process input documents."""

    INPUT_DOCUMENT_TYPES = [InputDocument]
    OUTPUT_DOCUMENT_TYPE = AnalysisDocument


class Stage2Config(FlowConfig):
    """Second stage: enhance analysis documents."""

    INPUT_DOCUMENT_TYPES = [AnalysisDocument]
    OUTPUT_DOCUMENT_TYPE = EnhancedDocument


# -----------------------------------------------------------------------------
# Structured response schemas
# -----------------------------------------------------------------------------
class TextAnalysis(BaseModel):
    """Structured output for text analysis."""

    summary: str
    key_points: list[str]
    sentiment: str
    confidence: float = Field(ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# Tasks demonstrating various decorators and features
# -----------------------------------------------------------------------------


# Task with custom tracing parameters including cost tracking
@pipeline_task(
    name="analyze_with_tracing",
    trace_level="always",
    trace_ignore_inputs=["sensitive_data"],
    trace_cost=0.002,  # Track cost for this task (e.g., $0.002 per call)
    retries=3,
    timeout_seconds=120,
)
async def analyze_with_advanced_tracing(
    doc: Document,
    model: ModelName,
    sensitive_data: str = "secret",
) -> TextAnalysis:
    # Use AIMessages with different content types
    # NEW: AIMessages supports Documents, strings, and ModelResponse objects
    messages = AIMessages([
        doc,  # Document automatically converted to prompt format
        "Analyze this document thoroughly",
    ])

    response = await llm.generate_structured(
        model=model,
        response_format=TextAnalysis,
        messages=messages,
        # Context parameter now defaults to None instead of empty AIMessages()
        context=None,  # Optional: Add static context for caching
        options=ModelOptions(
            system_prompt="You are an expert analyst",
            max_completion_tokens=2000,
            reasoning_effort="high",
            service_tier="default",
            # Retry with fixed delay (not exponential backoff)
            retries=3,
            retry_delay_seconds=10,
        ),
    )

    return response.parsed


# Clean Prefect task (no tracing)
@task
async def simple_transform(text: str) -> str:
    """Clean Prefect task without tracing."""
    return text.upper()


# Demonstrate @trace decorator alone with cost tracking
@trace(
    level="always",
    name="custom_operation",
    span_type="PROCESSING",
    metadata={"version": "1.0"},
)
async def traced_operation(data: str) -> dict[str, Any]:
    """Function with only tracing, no Prefect."""
    processor = DataProcessor()
    set_trace_cost(0.0005)
    return processor.process(data)


# -----------------------------------------------------------------------------
# Flows demonstrating different features
# -----------------------------------------------------------------------------


@pipeline_flow(
    name="stage1_analysis",
    trace_level="always",
    trace_cost=0.01,  # Track total cost for this flow (e.g., $0.01 per run)
    retries=2,
    timeout_seconds=600,
)
async def stage1_flow(
    project_name: str, documents: DocumentList, flow_options: ShowcaseFlowOptions
) -> DocumentList:
    """First stage: analyze input documents."""
    # Validate inputs using FlowConfig
    config = Stage1Config()
    input_docs = config.get_input_documents(documents)

    # Use PromptManager for templates
    prompts = PromptManager(__file__)

    outputs = DocumentList(validate_same_type=True)

    for doc in input_docs:
        # Demonstrate canonical_name_key utility
        canonical = canonical_name_key(doc.__class__)
        logger.info(f"Processing {canonical}: {doc.name}")

        # Load prompt template
        prompt = prompts.get("showcase.jinja2", text=doc.text, temperature=flow_options.temperature)

        # Demonstrate different generate modes
        if flow_options.enable_structured:
            # Structured generation
            analysis = await analyze_with_advanced_tracing(
                doc, flow_options.core_model, "sensitive_value"
            )

            # Dynamically set cost based on document size
            # NEW: set_trace_cost for dynamic cost tracking within traced functions
            doc_size_kb = doc.size / 1024
            dynamic_cost = doc_size_kb * 0.0001  # $0.0001 per KB
            set_trace_cost(dynamic_cost)

            # Create document using smart factory with Pydantic model
            # NEW in v0.1.14: Track document sources for provenance
            output = AnalysisDocument.create(
                name=f"analysis_{doc.id}.json",
                description="Structured analysis result",
                content=analysis,  # Pydantic model auto-serialized to JSON
                sources=[doc.sha256],  # Track source document
            )
        else:
            # Raw text generation with context caching
            # NEW: Context caching saves 50-90% tokens on repeated calls
            response: ModelResponse = await llm.generate(
                flow_options.small_model,
                context=AIMessages([doc]),  # Static context cached for 120 seconds
                messages=prompt,  # Dynamic messages not cached
                # Options now defaults to None if not provided
                options=ModelOptions(
                    max_completion_tokens=1000,
                    reasoning_effort=flow_options.reasoning_effort,
                    temperature=flow_options.temperature,
                ),
            )

            # Access response metadata
            logger.info(f"Model used: {response.model}")
            logger.info(f"Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
            # Note: Cost tracking via trace_cost parameter adds metadata to LMNR traces
            # The cost is stored in gen_ai.usage.output_cost, gen_ai.usage.cost, and cost fields

            # Create text document with source tracking
            output = AnalysisDocument.create(
                name=f"analysis_{doc.id}.txt",
                description="Text analysis result",
                content=response.content,
                sources=[doc.sha256],  # Track source
            )

        outputs.append(output)

    # Use create_and_validate_output method
    return config.create_and_validate_output(outputs)


@pipeline_flow(name="stage2_enhancement")
async def stage2_flow(
    project_name: str, documents: DocumentList, flow_options: ShowcaseFlowOptions
) -> DocumentList:
    """Second stage: enhance analysis documents."""
    config = Stage2Config()
    input_docs = config.get_input_documents(documents)

    outputs = DocumentList()

    for doc in input_docs:
        # Create temporary task document (demonstrates task documents are allowed but not persisted)
        temp_task_doc = TempProcessingDocument.create(
            name="temp.txt",
            description="Temporary processing data for demonstration",
            content="This document exists only during task execution",
        )
        logger.debug(f"Created temporary task document: {temp_task_doc.id}")

        # Demonstrate various document methods
        if doc.is_text:
            content = doc.text
        elif doc.mime_type.startswith("application/json"):
            content = str(doc.as_json())
        else:
            content = f"Binary document: {doc.size} bytes"

        # Simple transformation
        enhanced = await simple_transform(content)

        # Demonstrate sanitize_url utility
        safe_name = sanitize_url(f"https://example.com/doc/{doc.id}")

        # Create enhanced document with source tracking chain
        if doc.name.endswith(".json"):
            # Try to parse and enhance JSON
            try:
                data = doc.as_json()
                data["enhanced"] = True
                data["safe_name"] = safe_name

                # NEW: Create document with multiple sources (document + reference)
                output = EnhancedDocument.create(
                    name=doc.name.replace(".", "_enhanced."),
                    description="Enhanced analysis",
                    content=data,  # Will be auto-serialized to JSON
                    sources=[
                        doc.sha256,  # Previous stage doc
                        f"stage2_flow:{project_name}",  # Process reference
                    ],
                )
            except Exception:
                output = EnhancedDocument.create(
                    name=doc.name.replace(".", "_enhanced."),
                    description="Enhanced text",
                    content=enhanced,
                    sources=[doc.sha256],
                )
        else:
            # Demonstrate source tracking with multiple sources
            output = EnhancedDocument.create(
                name=doc.name.replace(".", "_enhanced."),
                description="Enhanced text",
                content=enhanced,
                sources=[
                    doc.sha256,
                    "enhancement_process",
                ],
            )

        # Log source tracking information
        logger.info(f"Document {output.name} has {len(output.sources)} sources")
        if output.sources:
            doc_sources = output.get_source_documents()
            ref_sources = output.get_source_references()
            if doc_sources:
                logger.debug(f"  Document sources: {len(doc_sources)} documents")
            if ref_sources:
                logger.debug(f"  Reference sources: {ref_sources}")

        outputs.append(output)

    # Use create_and_validate_output method
    return config.create_and_validate_output(outputs)


# -----------------------------------------------------------------------------
# Clean Prefect flow (no tracing)
# -----------------------------------------------------------------------------
@flow(name="cleanup_flow")
async def cleanup_flow(project_name: str) -> None:
    """Demonstrate clean Prefect flow without tracing."""
    logger.info(f"Cleaning up project: {project_name}")
    # Could perform cleanup tasks here


# -----------------------------------------------------------------------------
# Initializer for run_cli
# -----------------------------------------------------------------------------
def initialize_showcase(options: FlowOptions) -> tuple[str, DocumentList]:
    """Initialize with sample documents for CLI mode."""
    logger.info("Initializing showcase with sample data")

    # Note: OPENAI_API_KEY should be set via environment variable or .env file

    # NEW in v0.1.14: Demonstrate source tracking from external references
    external_source = "https://example.com/data-source"

    # Create sample documents including list[BaseModel] demonstration
    sample_models = [
        TextAnalysis(
            summary="First sample",
            key_points=["point 1", "point 2"],
            sentiment="positive",
            confidence=0.9,
        ),
        TextAnalysis(
            summary="Second sample",
            key_points=["point A", "point B"],
            sentiment="neutral",
            confidence=0.85,
        ),
    ]

    # Create temporary document to demonstrate TemporaryDocument usage
    temp_doc = TemporaryDocument.create(
        name="temp_demo.txt",
        description="Demo of TemporaryDocument",
        content="This demonstrates the new TemporaryDocument class",
    )
    logger.info(f"Created TemporaryDocument: {temp_doc.name} (type: {temp_doc.base_type})")

    docs = DocumentList([
        InputDocument.create(
            name="input.txt",
            description="Sample input document",
            content="""AI Pipeline Core is a powerful async library for building
            production-grade AI pipelines with strong typing, observability,
            and Prefect integration.""",
            sources=[external_source],  # Track external data source
        ),
        InputDocument.create(
            name="data.json",
            description="Sample JSON data",
            content={
                "project": "ai-pipeline-core",
                "version": "0.1.12",
                "features": ["async", "typed", "observable"],
                "models": [m.model_dump() for m in sample_models],  # Include list demo
            },
        ),
        InputDocument.create(
            name="config.yaml",
            description="Sample configuration",
            content={"model": "gpt-5-mini", "temperature": 0.7, "max_tokens": 2000},
        ),
    ])

    return "showcase-project", docs


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main():
    """Main entry point - CLI mode with run_cli."""
    # LoggingConfig can be used for custom configuration
    logging_config = LoggingConfig()
    logging_config.apply()

    # Run in CLI mode with all features
    run_cli(
        flows=[stage1_flow, stage2_flow],
        flow_configs=[Stage1Config, Stage2Config],
        options_cls=ShowcaseFlowOptions,
        initializer=initialize_showcase,
        trace_name="showcase",
    )


if __name__ == "__main__":
    main()
