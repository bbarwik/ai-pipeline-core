#!/usr/bin/env python3
"""
Complete showcase of ai_pipeline_core features (v0.1.8)

This example demonstrates ALL exports from ai_pipeline_core.__init__, including:
  • Settings configuration
  • Logging system (LoggerMixin, StructuredLoggerMixin)
  • Document system (Document, FlowDocument, TaskDocument, DocumentList)
  • Flow configuration (FlowConfig, FlowOptions)
  • Pipeline decorators (pipeline_flow, pipeline_task)
  • Prefect utilities (flow, task, prefect_test_harness, disable_run_logger)
  • LLM module (generate, generate_structured, AIMessages, ModelOptions, etc.)
  • Tracing (trace, TraceLevel, TraceInfo)
  • PromptManager for Jinja2 templates
  • Simple runner module (run_cli, run_pipeline, load/save documents)

Prerequisites:
  - OPENAI_API_KEY and OPENAI_BASE_URL configured (can be set in .env)
  - Optional: LMNR_PROJECT_API_KEY for tracing

Usage:
  python examples/showcase.py ./output --temperature 0.7 --batch-size 5

Tip: Set PREFECT_LOGGING_LEVEL=INFO for richer logs and LMNR_DEBUG=true for more detailed tracing
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# Import ALL exports from ai_pipeline_core.__init__
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
    canonical_name_key,
    # Prefect utilities
    get_logger,
    get_pipeline_logger,
    # LLM components
    llm,
    pipeline_flow,
    pipeline_task,
    sanitize_url,
    # Settings
    settings,
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

    FILES = AllowedInputFiles


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


# Task with custom tracing parameters
@pipeline_task(
    name="analyze_with_tracing",
    trace_level="always",
    trace_ignore_inputs=["sensitive_data"],
    retries=3,
    timeout_seconds=120,
)
async def analyze_with_advanced_tracing(
    doc: Document,
    model: ModelName | str,
    sensitive_data: str = "secret",
) -> TextAnalysis:
    # Use AIMessages with different content types
    messages = AIMessages(
        [
            doc,  # Document automatically converted
            "Analyze this document thoroughly",
        ]
    )

    response = await llm.generate_structured(
        model=model,
        response_format=TextAnalysis,
        messages=messages,
        options=ModelOptions(
            system_prompt="You are an expert analyst",
            max_completion_tokens=2000,
            reasoning_effort="high",
            service_tier="default",
        ),
    )

    return response.parsed


# Clean Prefect task (no tracing)
@task
async def simple_transform(text: str) -> str:
    """Clean Prefect task without tracing."""
    return text.upper()


# Demonstrate @trace decorator alone
@trace(level="always", name="custom_operation", span_type="PROCESSING", metadata={"version": "1.0"})
async def traced_operation(data: str) -> dict[str, Any]:
    """Function with only tracing, no Prefect."""
    processor = DataProcessor()
    return processor.process(data)


# -----------------------------------------------------------------------------
# Flows demonstrating different features
# -----------------------------------------------------------------------------


@pipeline_flow(name="stage1_analysis", trace_level="always", retries=2, timeout_seconds=600)
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
        prompt = prompts.get(
            "showcase.jinja2", text=doc.as_text(), temperature=flow_options.temperature
        )

        # Demonstrate different generate modes
        if flow_options.enable_structured:
            # Structured generation
            analysis = await analyze_with_advanced_tracing(
                doc, flow_options.core_model, "sensitive_value"
            )

            # Create document using smart factory with Pydantic model
            output = AnalysisDocument.create(
                f"analysis_{doc.id}.json",
                "Structured analysis result",
                analysis,  # Pydantic model auto-serialized to JSON
            )
        else:
            # Raw text generation
            response: ModelResponse = await llm.generate(
                flow_options.small_model,
                context=AIMessages([doc]),  # Context for caching
                messages=prompt,
                options=ModelOptions(
                    max_completion_tokens=1000,
                    reasoning_effort=flow_options.reasoning_effort,
                    temperature=flow_options.temperature,
                ),
            )

            # Access response metadata
            logger.info(f"Model used: {response.model}")
            logger.info(f"Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")

            # Create text document
            output = AnalysisDocument.create(
                f"analysis_{doc.id}.txt", "Text analysis result", response.content
            )

        outputs.append(output)

    # Validate outputs
    config.validate_output_documents(outputs)
    return outputs


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
        TempProcessingDocument(name="temp.txt", content=b"temporary processing data")

        # Demonstrate various document methods
        if doc.is_text:
            content = doc.as_text()
        elif doc.mime_type.startswith("application/json"):
            content = str(doc.as_json())
        else:
            content = f"Binary document: {doc.size} bytes"

        # Simple transformation
        enhanced = await simple_transform(content)

        # Demonstrate sanitize_url utility
        safe_name = sanitize_url(f"https://example.com/doc/{doc.id}")

        # Create enhanced document
        if doc.name.endswith(".json"):
            # Try to parse and enhance JSON
            try:
                data = doc.as_json()
                data["enhanced"] = True
                data["safe_name"] = safe_name
                output = EnhancedDocument.create_as_json(doc.name, "Enhanced analysis", data)
            except Exception:
                output = EnhancedDocument.create(doc.name, "Enhanced text", enhanced)
        else:
            output = EnhancedDocument.create(doc.name, "Enhanced text", enhanced)

        outputs.append(output)

    config.validate_output_documents(outputs)
    return outputs


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

    # Check settings
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not set")

    # Create sample documents
    docs = DocumentList(
        [
            InputDocument.create(
                "input.txt",
                "Sample input document",
                """AI Pipeline Core is a powerful async library for building
            production-grade AI pipelines with strong typing, observability,
            and Prefect integration.""",
            ),
            InputDocument.create_as_json(
                "data.json",
                "Sample JSON data",
                {
                    "project": "ai-pipeline-core",
                    "version": "0.1.8",
                    "features": ["async", "typed", "observable"],
                },
            ),
            InputDocument.create_as_yaml(
                "config.yaml",
                "Sample configuration",
                {"model": "gpt-5-mini", "temperature": 0.7, "max_tokens": 2000},
            ),
        ]
    )

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
