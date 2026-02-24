#!/usr/bin/env python3
"""Showcase of ai_pipeline_core features (v0.9.3)

Full-featured 3-stage pipeline with real LLM interactions demonstrating
every framework capability:
  - Documents: immutable, content-addressed, typed, with attachments and provenance
  - LLM: Conversation with send(), send_structured(), context caching, multi-turn
  - Pipeline: @pipeline_flow, @pipeline_task with auto-save, retries
  - Deployment: PipelineDeployment with CLI, resume/skip, progress tracking
  - Observability: @trace, set_trace_cost, Laminar integration
  - Logging: setup_logging, get_pipeline_logger
  - Settings: FlowOptions with env-based configuration
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
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from ai_pipeline_core import (
    Attachment,
    Conversation,
    DeploymentResult,
    Document,
    FlowOptions,
    ModelOptions,
    PipelineDeployment,
    get_pipeline_logger,
    is_document_sha256,
    pipeline_flow,
    pipeline_task,
    sanitize_url,
    set_trace_cost,
    setup_logging,
    trace,
)

logger = get_pipeline_logger(__name__)

EXAMPLE_SOURCE_URL = "https://github.com/example/ai-pipeline-core"


# ---------------------------------------------------------------------------
# @trace on standalone function
# ---------------------------------------------------------------------------


@trace(name="validate_provenance")
def validate_provenance(doc: Document) -> None:
    """Validate and log document source references."""
    for src in doc.derived_from:
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


@pipeline_task(estimated_minutes=2)
async def analyze_document(
    document: InputDocument,
    *,
    core_model: str,
    max_turns: int,
) -> AnalysisDocument:
    """Multi-turn LLM analysis of a single document."""
    t0 = time.perf_counter()
    logger.info(f"Starting analysis: document={document.name}, size={document.size}")

    conv = Conversation(model=core_model)
    conv = conv.with_context(document)
    logger.info(f"Context: ~{conv.approximate_tokens_count} tokens")

    prompt = f"Analyze the document '{document.name}' thoroughly. Identify key themes, technical concepts, and actionable insights."
    conv = await conv.send(prompt, purpose="document_analysis")

    if conv.reasoning_content:
        logger.debug(f"Model reasoning: {conv.reasoning_content[:100]}...")
    logger.info(f"Tokens: {conv.usage.prompt_tokens} in / {conv.usage.completion_tokens} out")

    for turn in range(1, max_turns):
        conv = await conv.send(
            f"What are the most important implications and connections you see? (Turn {turn + 1}/{max_turns})",
            purpose="analysis_followup",
        )

    combined = conv.content

    set_trace_cost(0.01)
    validate_provenance(document)

    duration_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Completed analysis: document={document.name}, duration_ms={duration_ms:.1f}")

    return AnalysisDocument.create(
        name=f"analysis_{document.sha256[:12]}.md",
        content=combined,
        derived_from=(document.sha256, *document.content_references),
    )


@pipeline_task(estimated_minutes=1)
async def extract_insights(
    analysis: AnalysisDocument,
    *,
    fast_model: str,
    reasoning_effort: str,
    original_input_sha256: str,
) -> InsightDocument:
    """Structured extraction from analysis text."""
    options = ModelOptions(reasoning_effort=reasoning_effort)
    conv = Conversation(model=fast_model, model_options=options)
    conv = conv.with_context(analysis)

    prompt = f"Extract structured insights from the analysis of '{analysis.name}'."
    conv = await conv.send_structured(prompt, response_format=DocumentInsight, purpose="insight_extraction")

    insight: DocumentInsight = conv.parsed  # type: ignore[assignment]
    logger.info(f"Extracted {len(insight.key_findings)} findings, complexity: {insight.complexity}")

    return InsightDocument.create(
        name=f"insight_{analysis.sha256[:12]}.json",
        content=insight,
        derived_from=(analysis.sha256,),
        triggered_by=(original_input_sha256,) if original_input_sha256 else (),
    )


@pipeline_task(estimated_minutes=1, trace_trim_documents=False)
async def compile_report(
    run_id: str,
    insights: list[InsightDocument],
    analyses: list[AnalysisDocument],
) -> ReportDocument:
    """Compile final report from structured insights with raw analysis attachments."""
    lines = [
        f"# {run_id} — Analysis Report",
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

        for src in insight_doc.derived_from:
            if is_document_sha256(src):
                logger.debug(f"Tracked document source: {src[:12]}...")

        if analyses and insight_doc.has_derived_from(analyses[min(i - 1, len(analyses) - 1)]):
            logger.debug(f"Insight {i} correctly references its analysis")

        if insight_doc.triggered_by:
            lines.append(f"*Trigger: `{insight_doc.triggered_by[0][:12]}...`*")
            lines.append("")

    attachments = [Attachment(name=f"raw_{a.name}", content=a.content, description=f"Full analysis text for {a.name}") for a in analyses[:2]]

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
        derived_from=tuple(all_sources),
        attachments=tuple(attachments),
    )


# ---------------------------------------------------------------------------
# Pipeline flows — annotation-driven input/output types
# ---------------------------------------------------------------------------


@pipeline_flow(estimated_minutes=5, retries=1, timeout_seconds=600)
async def analysis_flow(
    run_id: str,
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
    run_id: str,
    documents: list[AnalysisDocument],
    flow_options: ShowcaseFlowOptions,
) -> list[InsightDocument]:
    """Stage 2: structured extraction from each analysis."""
    results: list[InsightDocument] = []
    for analysis in documents:
        original_sha = analysis.content_documents[0] if analysis.content_documents else ""
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
    run_id: str,
    documents: list[InsightDocument | AnalysisDocument],
    flow_options: ShowcaseFlowOptions,
) -> list[ReportDocument]:
    """Stage 3: compile report from insights + analysis attachments.

    Accepts both InsightDocument and AnalysisDocument — the deployment loads
    all matching types from the store. Demonstrates multi-type flow input.
    """
    insights = [d for d in documents if isinstance(d, InsightDocument)]
    analyses = [d for d in documents if isinstance(d, AnalysisDocument)]
    report = await compile_report(run_id, insights, analyses)
    logger.info(f"Report: {len(report.content_documents)} sources, {len(report.attachments)} attachments, {report.size} bytes")
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
        run_id: str,
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
                "progress tracking via Prefect labels, and CLI execution with step control. "
                "The document store supports MemoryDocumentStore for testing, LocalDocumentStore "
                "for CLI/debug workflows, and ClickHouseDocumentStore for production with "
                "content-addressed deduplication and automatic LLM-generated document summaries."
            ),
            derived_from=(EXAMPLE_SOURCE_URL,),
        ),
        InputDocument.create(
            name="data.json",
            content={
                "project": "ai-pipeline-core",
                "version": "0.9.3",
                "modules": ["documents", "llm", "pipeline", "deployment", "observability"],
                "backends": {"store": ["memory", "local", "clickhouse"], "tracing": ["laminar", "otel"]},
            },
            derived_from=(EXAMPLE_SOURCE_URL,),
        ),
        InputDocument.create(
            name="config.yaml",
            content={
                "pipeline": {"stages": 3, "retry_policy": "exponential"},
                "models": {"analysis": options.core_model, "extraction": options.fast_model},
                "features": {"caching": True, "summaries": True, "attachments": True},
            },
            derived_from=(EXAMPLE_SOURCE_URL,),
        ),
    ]

    serialized = docs[0].serialize_model()
    roundtripped = InputDocument.from_dict(serialized)
    assert roundtripped.sha256 == docs[0].sha256
    logger.info(f"Roundtrip verified: {roundtripped.name} (sha256 match)")

    return "showcase-full", docs


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
