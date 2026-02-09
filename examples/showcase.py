#!/usr/bin/env python3
"""Showcase of ai_pipeline_core features (v0.4.1)

Full-featured 3-stage pipeline with real LLM interactions demonstrating
every framework capability:
  - Documents: immutable, content-addressed, typed, with attachments and provenance
  - LLM: generate(), generate_structured(), AIMessages, context caching, multi-turn
  - Pipeline: @pipeline_flow, @pipeline_task with auto-save, retries, user_summary
  - Deployment: PipelineDeployment with CLI, resume/skip, progress tracking
  - Observability: @trace, set_trace_cost, Laminar integration
  - PromptManager: Jinja2 templates with smart path resolution
  - Logging: setup_logging, get_pipeline_logger
  - Settings: FlowOptions with env-based configuration
  - Images: process_image, process_image_to_documents, ImagePreset
  - Document Store: auto-configured (Local for CLI, ClickHouse if configured)

Prerequisites:
  OPENAI_BASE_URL and OPENAI_API_KEY must be set (LiteLLM proxy).
  Optional: LMNR_PROJECT_API_KEY (tracing), CLICKHOUSE_HOST (production store).

Usage:
  python examples/showcase.py ./output
  python examples/showcase.py ./output --reasoning-effort high --core-model gpt-5.2
  python examples/showcase.py ./output --start 2
"""

import time
from enum import StrEnum
from io import BytesIO
from typing import ClassVar, Literal

from PIL import Image
from pydantic import BaseModel, Field

from ai_pipeline_core import (
    AIMessages,
    Attachment,
    DeploymentResult,
    Document,
    FlowOptions,
    ImagePreset,
    ModelOptions,
    ModelResponse,
    PipelineDeployment,
    PromptManager,
    get_pipeline_logger,
    is_document_sha256,
    llm,
    pipeline_flow,
    pipeline_task,
    process_image,
    process_image_to_documents,
    sanitize_url,
    set_trace_cost,
    setup_logging,
    trace,
)

logger = get_pipeline_logger(__name__)


# ---------------------------------------------------------------------------
# @trace on standalone function
# ---------------------------------------------------------------------------


@trace(name="validate_provenance")
def validate_provenance(doc: Document) -> None:
    """Validate and log document source references."""
    for src in doc.sources:
        if is_document_sha256(src):
            logger.debug(f"Document source: {src[:12]}...")
        else:
            logger.debug(f"Reference source: {sanitize_url(src)}")


# ---------------------------------------------------------------------------
# Document types
# ---------------------------------------------------------------------------


class AllowedInputFiles(StrEnum):
    """Restrict InputDocument filenames via FILES enum."""

    CONFIG = "config.yaml"
    DATA = "data.json"
    NOTES = "notes.txt"


class InputDocument(Document):
    """Pipeline input with FILES enum validation."""

    FILES: ClassVar[type[AllowedInputFiles]] = AllowedInputFiles


class AnalysisDocument(Document):
    """Stage 1 output: LLM-generated multi-turn analysis (markdown)."""


class InsightDocument(Document):
    """Stage 2 output: structured extraction (JSON via BaseModel)."""


class ReportDocument(Document):
    """Stage 3 output: compiled report with attachments."""


# ---------------------------------------------------------------------------
# Structured output model (decomposition fields before decision fields)
# ---------------------------------------------------------------------------


class DocumentInsight(BaseModel):
    """Structured extraction from analysis."""

    topics: list[str] = Field(description="Main topics identified in the document")
    technical_concepts: list[str] = Field(description="Technical terms and concepts found")
    key_findings: list[str] = Field(description="3-5 key findings from the analysis")
    complexity: Literal["low", "medium", "high"] = Field(description="Content complexity level")
    actionable_items: list[str] = Field(description="Concrete next steps or recommendations")


# ---------------------------------------------------------------------------
# Flow options (env-configurable via BaseSettings)
# ---------------------------------------------------------------------------


class ShowcaseFlowOptions(FlowOptions):
    """Pipeline configuration — all fields overridable via CLI flags or env vars."""

    core_model: str = Field(default="gemini-3-pro", description="Model for multi-turn analysis")
    fast_model: str = Field(default="gemini-3-flash", description="Model for structured extraction")
    reasoning_effort: Literal["low", "medium", "high"] = Field(default="medium", description="Reasoning intensity")
    max_analysis_turns: int = Field(default=2, ge=1, le=5, description="Multi-turn analysis depth")


# ---------------------------------------------------------------------------
# Pipeline tasks
# ---------------------------------------------------------------------------


@pipeline_task(user_summary=True, estimated_minutes=2)
async def analyze_document(
    document: InputDocument,
    *,
    core_model: str,
    max_turns: int,
) -> AnalysisDocument:
    """Multi-turn LLM analysis of a single document."""
    prompts = PromptManager(__file__, prompts_dir="templates")

    t0 = time.perf_counter()
    logger.info(f"Starting analysis: document={document.name}, size={document.size}")

    # Context: static, cacheable prefix — the document itself
    context = AIMessages([document])
    context.freeze()
    logger.info(f"Context: ~{context.approximate_tokens_count} tokens, cache key: {context.get_prompt_cache_key()[:16]}...")

    # Messages: dynamic conversation history
    messages = AIMessages()

    # Turn 1: initial analysis via Jinja2 template (current_date auto-available)
    prompt = prompts.get("analyze", document_name=document.name)
    messages.append(prompt)

    response = await llm.generate(
        core_model,
        context=context,
        messages=messages,
        purpose="document_analysis",
    )
    response.validate_output()
    messages.append(response)

    # Log model reasoning and usage if available
    if response.reasoning_content:
        logger.debug(f"Model reasoning: {response.reasoning_content[:100]}...")
    if response.usage:
        logger.info(f"Tokens: {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out")

    # Follow-up turns for deeper analysis
    for turn in range(1, max_turns):
        messages.append(f"What are the most important implications and connections you see? (Turn {turn + 1}/{max_turns})")
        response = await llm.generate(
            core_model,
            context=context,
            messages=messages,
            purpose="analysis_followup",
        )
        messages.append(response)

    # Deep copy messages before extracting content
    messages_snapshot = messages.copy()

    # Combine all assistant responses
    combined = "\n\n---\n\n".join(msg.content for msg in messages_snapshot if isinstance(msg, ModelResponse))

    set_trace_cost(0.01)
    validate_provenance(document)

    duration_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Completed analysis: document={document.name}, duration_ms={duration_ms:.1f}")

    return AnalysisDocument.create(
        name=f"analysis_{document.sha256[:12]}.md",
        content=combined,
        sources=(document.sha256, *document.source_references),
    )


@pipeline_task(user_summary="Insight extraction", estimated_minutes=1)
async def extract_insights(
    analysis: AnalysisDocument,
    *,
    fast_model: str,
    reasoning_effort: Literal["low", "medium", "high"],
    original_input_sha256: str,
) -> InsightDocument:
    """Structured extraction from analysis text."""
    prompts = PromptManager(__file__, prompts_dir="templates")
    prompt = prompts.get("extract", document_name=analysis.name)
    options = ModelOptions(reasoning_effort=reasoning_effort)

    structured = await llm.generate_structured(
        fast_model,
        DocumentInsight,
        context=AIMessages([analysis]),
        messages=prompt,
        options=options,
        purpose="insight_extraction",
    )

    insight: DocumentInsight = structured.parsed
    logger.info(f"Extracted {len(insight.key_findings)} findings, complexity: {insight.complexity}")

    return InsightDocument.create(
        name=f"insight_{analysis.sha256[:12]}.json",
        content=insight,
        sources=(analysis.sha256,),
        origins=(original_input_sha256,) if original_input_sha256 else (),
    )


@pipeline_task(estimated_minutes=1, trace_trim_documents=False)
async def compile_report(
    project_name: str,
    insights: list[InsightDocument],
    analyses: list[AnalysisDocument],
) -> ReportDocument:
    """Compile final report from structured insights with raw analysis attachments."""
    lines = [
        f"# {project_name} — Analysis Report",
        "",
        f"Documents analyzed: {len(insights)}",
        f"Insight type: `{InsightDocument.__name__}`",
        "",
        "## Insights",
        "",
    ]

    all_sources: list[str] = []

    for i, insight_doc in enumerate(insights, start=1):
        insight = insight_doc.parse(DocumentInsight)
        safe_name = sanitize_url(f"https://example.com/insight/{insight_doc.id}")

        lines.append(f"### {i}. {safe_name}")
        lines.append(f"**Topics:** {', '.join(insight.topics[:5])}")
        lines.append(f"**Complexity:** {insight.complexity}")
        lines.append("")
        lines.extend(f"- {finding}" for finding in insight.key_findings)
        lines.append("")

        if insight.actionable_items:
            lines.append("**Action items:**")
            lines.extend(f"- [ ] {item}" for item in insight.actionable_items)
            lines.append("")

        all_sources.append(insight_doc.sha256)

        # Source filtering via is_document_sha256
        for src in insight_doc.sources:
            if is_document_sha256(src):
                logger.debug(f"Tracked document source: {src[:12]}...")

        # has_source check
        if analyses and insight_doc.has_source(analyses[min(i - 1, len(analyses) - 1)]):
            logger.debug(f"Insight {i} correctly references its analysis")

        # Origins access
        if insight_doc.origins:
            lines.append(f"*Origin: `{insight_doc.origins[0][:12]}...`*")
            lines.append("")

    # Raw analyses as Attachment objects
    attachments = [Attachment(name=f"raw_{a.name}", content=a.content, description=f"Full analysis text for {a.name}") for a in analyses[:2]]

    # Log attachment properties
    for att in attachments:
        logger.debug(f"Attachment: {att.name}, {att.mime_type}, {att.size} bytes, is_text={att.is_text}")

    lines.extend([
        "## Provenance",
        "",
        f"Total document sources: {len(all_sources)}",
    ])
    lines.extend(f"- `{src[:16]}...`" for src in all_sources)

    return ReportDocument.create(
        name="report.md",
        content="\n".join(lines),
        sources=tuple(all_sources),
        attachments=tuple(attachments),
    )


# ---------------------------------------------------------------------------
# Pipeline flows — annotation-driven input/output types
# ---------------------------------------------------------------------------


@pipeline_flow(estimated_minutes=5, retries=1, timeout_seconds=600)
async def analysis_flow(
    project_name: str,
    documents: list[InputDocument],
    flow_options: ShowcaseFlowOptions,
) -> list[AnalysisDocument]:
    """Stage 1: multi-turn LLM analysis of each input document."""
    results: list[AnalysisDocument] = []
    for doc in documents:
        logger.info(f"Analyzing {doc.__class__.__name__}: {doc.name} ({doc.mime_type}, is_text={doc.is_text})")
        analysis = await analyze_document(
            doc,
            core_model=flow_options.core_model,
            max_turns=flow_options.max_analysis_turns,
        )
        results.append(analysis)
    logger.info(f"Stage 1 complete: {len(results)} analyses")
    return results


@pipeline_flow(estimated_minutes=3)
async def extraction_flow(
    project_name: str,
    documents: list[AnalysisDocument],
    flow_options: ShowcaseFlowOptions,
) -> list[InsightDocument]:
    """Stage 2: structured extraction from each analysis."""
    results: list[InsightDocument] = []
    for analysis in documents:
        original_sha = analysis.source_documents[0] if analysis.source_documents else ""
        insight = await extract_insights(
            analysis,
            fast_model=flow_options.fast_model,
            reasoning_effort=flow_options.reasoning_effort,
            original_input_sha256=original_sha,
        )
        results.append(insight)
    logger.info(f"Stage 2 complete: {len(results)} insights")
    return results


@pipeline_flow(estimated_minutes=1)
async def report_flow(
    project_name: str,
    documents: list[InsightDocument | AnalysisDocument],
    flow_options: ShowcaseFlowOptions,
) -> list[ReportDocument]:
    """Stage 3: compile report from insights + analysis attachments.

    Accepts both InsightDocument and AnalysisDocument — the deployment loads
    all matching types from the store. Demonstrates multi-type flow input.
    """
    insights = [d for d in documents if isinstance(d, InsightDocument)]
    analyses = [d for d in documents if isinstance(d, AnalysisDocument)]
    report = await compile_report(project_name, insights, analyses)
    logger.info(f"Report: {len(report.source_documents)} sources, {len(report.attachments)} attachments, {report.size} bytes")
    return [report]


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


class ShowcaseResult(DeploymentResult):
    """Typed result from the showcase pipeline."""

    analysis_count: int = 0
    insight_count: int = 0
    report_files: list[str] = Field(default_factory=list)


class ShowcasePipeline(PipelineDeployment[ShowcaseFlowOptions, ShowcaseResult]):
    """3-stage pipeline: analyze → extract → report.

    Flow chain validated at class definition time:
    InputDocument → AnalysisDocument → InsightDocument → ReportDocument
    """

    flows: ClassVar = [analysis_flow, extraction_flow, report_flow]

    @staticmethod
    def build_result(
        project_name: str,
        documents: list[Document],
        options: ShowcaseFlowOptions,
    ) -> ShowcaseResult:
        analyses = [d for d in documents if isinstance(d, AnalysisDocument)]
        insights = [d for d in documents if isinstance(d, InsightDocument)]
        reports = [d for d in documents if isinstance(d, ReportDocument)]
        return ShowcaseResult(
            success=True,
            analysis_count=len(analyses),
            insight_count=len(insights),
            report_files=[d.name for d in reports],
        )


showcase_pipeline = ShowcasePipeline()


# ---------------------------------------------------------------------------
# CLI initializer
# ---------------------------------------------------------------------------


def initialize_showcase(options: ShowcaseFlowOptions) -> tuple[str, list[Document]]:
    """Create sample input documents and demonstrate standalone features."""
    logger.info("Initializing showcase with sample data")

    docs: list[Document] = [
        InputDocument.create(
            name="notes.txt",
            content=(
                "AI Pipeline Core is a production-ready framework for building "
                "AI pipelines with strong typing, observability, and Prefect integration. "
                "Documents are immutable, content-addressed, and auto-saved by pipeline tasks. "
                "The framework provides LLM routing via LiteLLM, structured output with Pydantic, "
                "multi-turn conversations, context caching, and distributed tracing via Laminar.\n\n"
                "Pipeline deployments support resume/skip logic based on content fingerprinting, "
                "progress webhooks for real-time monitoring, and CLI execution with step control. "
                "The document store supports MemoryDocumentStore for testing, LocalDocumentStore "
                "for CLI/debug workflows, and ClickHouseDocumentStore for production with "
                "content-addressed deduplication and automatic LLM-generated document summaries."
            ),
            sources=("https://github.com/example/ai-pipeline-core",),
        ),
        InputDocument.create(
            name="data.json",
            content={
                "project": "ai-pipeline-core",
                "version": "0.4.1",
                "modules": ["documents", "llm", "pipeline", "deployment", "observability"],
                "backends": {"store": ["memory", "local", "clickhouse"], "tracing": ["laminar", "otel"]},
            },
        ),
        InputDocument.create(
            name="config.yaml",
            content={
                "pipeline": {"stages": 3, "retry_policy": "exponential"},
                "models": {"analysis": options.core_model, "extraction": options.fast_model},
                "features": {"caching": True, "summaries": True, "attachments": True},
            },
        ),
    ]

    # Document.serialize_model() / from_dict() roundtrip
    serialized = docs[0].serialize_model()
    roundtripped = InputDocument.from_dict(serialized)
    assert roundtripped.sha256 == docs[0].sha256
    logger.info(f"Roundtrip verified: {roundtripped.name} (sha256 match)")

    # Image processing demonstration (standalone)
    _demo_image_processing()

    return "showcase-full", docs


def _demo_image_processing() -> None:
    """Demonstrate process_image() and process_image_to_documents() with ImagePreset."""
    img = Image.new("RGB", (400, 800), color=(70, 130, 180))
    buf = BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    # Default GEMINI preset
    result_gemini = process_image(image_bytes)
    logger.info(
        f"process_image (GEMINI): {result_gemini.original_width}x{result_gemini.original_height} "
        f"→ {len(result_gemini.parts)} part(s), "
        f"compression: {result_gemini.compression_ratio:.2f}, "
        f"trimmed: {result_gemini.was_trimmed}"
    )
    for part in result_gemini:
        logger.debug(f"  {part.label}: {part.width}x{part.height}, {len(part.data)} bytes")

    # Explicit CLAUDE preset (different constraints)
    result_claude = process_image(image_bytes, preset=ImagePreset.CLAUDE)
    logger.info(f"process_image (CLAUDE): → {len(result_claude.parts)} part(s), compression: {result_claude.compression_ratio:.2f}")

    # Convert to ImageDocument list
    docs = process_image_to_documents(image_bytes, name_prefix="demo_image")
    logger.info(f"process_image_to_documents: {len(docs)} doc(s), first: {docs[0].name} ({docs[0].mime_type})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point — setup logging first, then delegate to run_cli()."""
    setup_logging(level="INFO")
    showcase_pipeline.run_cli(
        initializer=initialize_showcase,
        trace_name="showcase",
    )


if __name__ == "__main__":
    main()
