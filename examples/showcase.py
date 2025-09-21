#!/usr/bin/env python3
"""Complete showcase of ai_pipeline_core features (v0.2.1)

This example demonstrates ALL exports from ai_pipeline_core.__init__, including:
  • Settings configuration with environment variables and .env files
  • Logging system (LoggerMixin, StructuredLoggerMixin) with Prefect integration
  • Document system with immutable, type-safe documents:
    - Document: Abstract base with MIME detection and content validation
    - FlowDocument: Persisted across flow runs
    - TaskDocument: Temporary within task execution
    - TemporaryDocument: Never persisted (NEW in v0.1.9+)
    - DocumentList: Type-safe container with validation
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
    DocumentList,
    FlowConfig,
    FlowDocument,
    FlowOptions,
    LoggingConfig,
    ModelName,
    ModelResponse,
    PromptManager,
    StructuredLoggerMixin,
    TaskDocument,
    canonical_name_key,
    get_logger,
    get_pipeline_logger,
    llm,
    pipeline_flow,
    pipeline_task,
    sanitize_url,
    set_trace_cost,
    setup_logging,
    trace,
)
from ai_pipeline_core.prefect import flow, task
from ai_pipeline_core.simple_runner import run_cli

setup_logging(level="INFO")  # Can also pass config_path for YAML config
logger = get_pipeline_logger(__name__)

alt_logger = get_logger("showcase.alternative")


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


class ResearchTaskDocument(TaskDocument):
    """Task document for research data (temporary, can be trimmed in traces)."""

    pass


class ResearchFlowDocument(FlowDocument):
    """Flow document for research results (persistent, not trimmed in traces)."""

    pass


class ShowcaseFlowOptions(FlowOptions):
    """Extended flow options with custom fields."""

    temperature: float = Field(default=0.7, ge=0, le=2, description="LLM temperature")
    batch_size: int = Field(default=10, ge=1, description="Processing batch size")
    enable_structured: bool = Field(default=True, description="Use structured output")
    reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="medium", description="Reasoning level"
    )


class Stage1Config(FlowConfig):
    """First stage: process input documents."""

    INPUT_DOCUMENT_TYPES = [InputDocument]
    OUTPUT_DOCUMENT_TYPE = AnalysisDocument


class Stage2Config(FlowConfig):
    """Second stage: enhance analysis documents."""

    INPUT_DOCUMENT_TYPES = [AnalysisDocument]
    OUTPUT_DOCUMENT_TYPE = EnhancedDocument


class TextAnalysis(BaseModel):
    """Structured output for text analysis."""

    summary: str
    key_points: list[str] = Field(default_factory=list)
    sentiment: str
    confidence: float = Field(ge=0, le=1)


class InputForTask(BaseModel):
    """Pydantic model containing both task and flow documents for batch processing."""

    task_doc: ResearchTaskDocument = Field(description="Temporary research data")
    flow_doc: ResearchFlowDocument = Field(description="Persistent research results")
    processing_params: dict[str, Any] = Field(default_factory=dict)


@pipeline_task
async def analyze_with_advanced_tracing(
    doc: FlowDocument,
    model: ModelName,
) -> TextAnalysis:
    messages = AIMessages([
        doc,
        "Analyze this document thoroughly",
    ])

    response = await llm.generate_structured(
        model=model, response_format=TextAnalysis, messages=messages
    )

    return response.parsed


@pipeline_task(trace_trim_documents=True)  # Trim documents in traces for this task
async def process_research_batch(
    research_inputs: list[InputForTask],
    model: ModelName = "gpt-5-mini",
) -> DocumentList:
    """Process batch of research inputs containing both task and flow documents.

    This task demonstrates:
    - Processing list[Pydantic models] with embedded Documents
    - LLM interactions for document analysis
    - Returning DocumentList with mixed document types
    - Document trimming in traces (task docs trimmed, flow docs preserved)
    """
    logger.info(f"Processing batch of {len(research_inputs)} research items")

    outputs = DocumentList(validate_same_type=False)

    for i, input_item in enumerate(research_inputs):
        task_doc = input_item.task_doc
        flow_doc = input_item.flow_doc
        params = input_item.processing_params

        logger.debug(f"Task doc size: {task_doc.size} bytes, Flow doc size: {flow_doc.size} bytes")
        messages = AIMessages([
            "Analyze this temporary research data:",
            task_doc,
            "Compare with this permanent research result:",
            flow_doc,
            f"Using parameters: {params}",
        ])

        response = await llm.generate(model=model, messages=messages)

        total_kb = (task_doc.size + flow_doc.size) / 1024
        set_trace_cost(total_kb * 0.001)
        task_preview = task_doc.text[:200]
        enhanced_task_content = (
            f"BATCH {i + 1} ANALYSIS:\n{response.content}\n\nOriginal: {task_preview}..."
        )
        enhanced_task_doc = ResearchTaskDocument.create(
            name=f"enhanced_task_{i}.txt",
            content=enhanced_task_content,
            sources=[task_doc.sha256],
        )
        flow_preview = flow_doc.text[:200]
        enhanced_flow_content = (
            f"PERMANENT RESULT {i + 1}:\n{response.content}\n\nBased on: {flow_preview}..."
        )
        enhanced_flow_doc = ResearchFlowDocument.create(
            name=f"enhanced_flow_{i}.txt",
            content=enhanced_flow_content,
            sources=[flow_doc.sha256, task_doc.sha256],
        )

        outputs.append(enhanced_task_doc)
        outputs.append(enhanced_flow_doc)

        logger.info(f"Processed item {i + 1}: created 2 documents (1 task, 1 flow)")

    logger.info(f"Batch processing complete: {len(outputs)} total documents")
    return outputs


@task
async def simple_transform(text: str) -> str:
    """Clean Prefect task without tracing."""
    return text.upper()


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


@pipeline_flow(
    config=Stage1Config,
    name="stage1_analysis",
    trace_level="always",
    trace_cost=0.01,  # Track total cost for this flow (e.g., $0.01 per run)
    trace_trim_documents=False,  # Keep full documents in flow traces
    retries=2,
    timeout_seconds=600,
)
async def stage1_flow(
    project_name: str, documents: DocumentList, flow_options: ShowcaseFlowOptions
) -> DocumentList:
    """First stage: analyze input documents and demonstrate batch processing."""
    config = Stage1Config()
    input_docs = config.get_input_documents(documents)
    research_batch = []
    large_task_content = (
        "This is temporary research data that will be trimmed in traces. " * 15
        + "It contains detailed analysis, experiments, and intermediate results. " * 10
        + "The data includes metrics, observations, and hypotheses. " * 10
    )

    large_flow_content = (
        "This is permanent research that must be preserved in full. " * 15
        + "It represents validated conclusions and final results. " * 10
        + "These findings are critical for long-term reference. " * 10
        + "The document contains verified data and approved outcomes. " * 8
    )

    for i in range(3):
        task_doc = ResearchTaskDocument.create(
            name=f"research_task_{i}.txt",
            content=large_task_content + f"\n\nBatch ID: {i}\nProject: {project_name}",
        )

        flow_doc = ResearchFlowDocument.create(
            name=f"research_flow_{i}.txt",
            content=large_flow_content + f"\n\nResult ID: {i}\nValidated: True",
        )

        research_batch.append(
            InputForTask(
                task_doc=task_doc,
                flow_doc=flow_doc,
                processing_params={
                    "temperature": flow_options.temperature,
                    "batch_id": i,
                    "reasoning_effort": flow_options.reasoning_effort,
                },
            )
        )

    logger.info("Processing research batch with large documents")
    batch_results = await process_research_batch(
        research_batch,
        model=flow_options.core_model,
    )
    logger.info(f"Batch processing produced {len(batch_results)} documents")

    prompts = PromptManager(__file__)

    outputs = DocumentList(validate_same_type=True)

    for doc in input_docs:
        # Demonstrate canonical_name_key utility
        canonical = canonical_name_key(doc.__class__)
        logger.info(f"Processing {canonical}: {doc.name}")

        # Using current_date variable from PromptManager
        prompt = prompts.get("showcase.jinja2", text=doc.text, temperature=flow_options.temperature)

        if flow_options.enable_structured:
            analysis = await analyze_with_advanced_tracing(doc, flow_options.core_model)

            doc_size_kb = doc.size / 1024
            dynamic_cost = doc_size_kb * 0.0001
            set_trace_cost(dynamic_cost)

            output = AnalysisDocument.create(
                name=f"analysis_{doc.id}.json",
                content=analysis,
                sources=[doc.sha256],
            )
        else:
            messages = AIMessages([doc, prompt])
            response: ModelResponse = await llm.generate(
                flow_options.small_model, messages=messages
            )

            logger.info(f"Model used: {response.model}")
            logger.info(f"Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")

            output = AnalysisDocument.create(
                name=f"analysis_{doc.id}.txt",
                content=response.content,
                sources=[doc.sha256],
            )

        outputs.append(output)

    for doc in batch_results:
        if isinstance(doc, (ResearchTaskDocument, ResearchFlowDocument)):
            analysis_doc = AnalysisDocument.create(
                name=f"batch_{doc.name}",
                content=doc.content,
                sources=doc.sources,
            )
            outputs.append(analysis_doc)

    return Stage1Config.create_and_validate_output(outputs)


@pipeline_flow(config=Stage2Config, name="stage2_enhancement")
async def stage2_flow(
    project_name: str, documents: DocumentList, flow_options: ShowcaseFlowOptions
) -> DocumentList:
    """Second stage: enhance analysis documents."""
    config = Stage2Config()
    input_docs = config.get_input_documents(documents)

    outputs = DocumentList()

    for doc in input_docs:
        temp_task_doc = TempProcessingDocument.create(
            name="temp.txt",
            content="This document exists only during task execution",
        )
        logger.debug(f"Created temporary task document: {temp_task_doc.id}")

        if doc.is_text:
            content = doc.text
        elif doc.mime_type.startswith("application/json"):
            content = str(doc.as_json())
        else:
            content = f"Binary document: {doc.size} bytes"

        enhanced = await simple_transform(content)

        safe_name = sanitize_url(f"https://example.com/doc/{doc.id}")

        if doc.name.endswith(".json"):
            try:
                data = doc.as_json()
                data["enhanced"] = True
                data["safe_name"] = safe_name

                output = EnhancedDocument.create(
                    name=doc.name.replace(".", "_enhanced."),
                    content=data,
                    sources=[
                        doc.sha256,
                        f"stage2_flow:{project_name}",
                    ],
                )
            except Exception:
                output = EnhancedDocument.create(
                    name=doc.name.replace(".", "_enhanced."),
                    content=enhanced,
                    sources=[doc.sha256],
                )
        else:
            output = EnhancedDocument.create(
                name=doc.name.replace(".", "_enhanced."),
                content=enhanced,
                sources=[
                    doc.sha256,
                    "enhancement_process",
                ],
            )

        logger.info(f"Document {output.name} has {len(output.sources)} sources")
        if output.sources:
            doc_sources = output.get_source_documents()
            ref_sources = output.get_source_references()
            if doc_sources:
                logger.debug(f"  Document sources: {len(doc_sources)} documents")
            if ref_sources:
                logger.debug(f"  Reference sources: {ref_sources}")

        outputs.append(output)

    return Stage2Config.create_and_validate_output(outputs)


@flow(name="cleanup_flow")
async def cleanup_flow(project_name: str) -> None:
    """Demonstrate clean Prefect flow without tracing."""
    logger.info(f"Cleaning up project: {project_name}")
    # Could perform cleanup tasks here


def initialize_showcase(options: FlowOptions) -> tuple[str, DocumentList]:
    """Initialize with sample documents for CLI mode."""
    logger.info("Initializing showcase with sample data")

    external_source = "https://example.com/data-source"

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

    docs = DocumentList([
        InputDocument.create(
            name="input.txt",
            content="""AI Pipeline Core is a powerful async library for building
            production-grade AI pipelines with strong typing, observability,
            and Prefect integration.""",
            sources=[external_source],
        ),
        InputDocument.create(
            name="data.json",
            content={
                "project": "ai-pipeline-core",
                "version": "0.2.1",
                "features": ["async", "typed", "observable"],
                "models": [m.model_dump() for m in sample_models],
            },
        ),
        InputDocument.create(
            name="config.yaml",
            content={"model": "gpt-5-mini", "temperature": 0.7},
        ),
    ])

    return "showcase-project", docs


def main():
    """Main entry point - CLI mode with run_cli."""
    logging_config = LoggingConfig()
    logging_config.apply()

    run_cli(
        flows=[stage1_flow, stage2_flow],
        options_cls=ShowcaseFlowOptions,
        initializer=initialize_showcase,
        trace_name="showcase",
    )


if __name__ == "__main__":
    main()
