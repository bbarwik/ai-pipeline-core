#!/usr/bin/env python3
"""Class-based pipeline showcase.

Runs locally with an in-memory database so ``python examples/showcase.py`` works
without extra setup beyond LLM credentials.

If you want persistent execution data instead, run this deployment through
``showcase_pipeline.run_cli(...)`` and inspect the stored execution with:
  ai-trace list
  ai-trace show <deployment-id>
  ai-trace download <deployment-id>
  ai-replay run --from-db <node-id>
"""

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from ai_pipeline_core import (
    Conversation,
    DeploymentResult,
    Document,
    FlowOptions,
    ModelOptions,
    PipelineDeployment,
    PipelineFlow,
    PipelineTask,
    find_document,
    get_run_id,
    setup_logging,
)
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)


class InputDocument(Document):
    """Input text to analyze."""


class AnalysisDocument(Document):
    """Unstructured LLM analysis."""


class ReportDocument(Document):
    """Final markdown report."""


class ShowcaseConfig(BaseModel, frozen=True):
    core_model: str
    fast_model: str
    reasoning_effort: Literal["low", "medium", "high"]


class ShowcaseConfigDocument(Document[ShowcaseConfig]):
    """Root configuration for this run."""


class InsightModel(BaseModel, frozen=True):
    topics: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    complexity: Literal["low", "medium", "high"] = "low"


class InsightDocument(Document[InsightModel]):
    """Structured extraction output."""


class ShowcaseFlowOptions(FlowOptions):
    core_model: str = "gemini-3-pro"
    fast_model: str = "gemini-3-flash"
    reasoning_effort: Literal["low", "medium", "high"] = "medium"


class AnalyzeDocumentTask(PipelineTask):
    name = "analyze_document"

    @classmethod
    async def run(cls, documents: tuple[InputDocument | ShowcaseConfigDocument, ...]) -> tuple[AnalysisDocument, ...]:
        logger.info("Running %s", cls.name)
        cfg = find_document(documents, ShowcaseConfigDocument).parsed
        source = find_document(documents, InputDocument)
        conv = Conversation(model=cfg.core_model).with_context(source)
        conv = await conv.send(
            f"Analyze '{source.name}' and summarize key themes.",
            purpose=f"analyze {source.name}",
        )
        return (
            AnalysisDocument.derive(
                from_documents=(source,),
                name=f"analysis_{source.id}.md",
                content=conv.content,
            ),
        )


class ExtractInsightsTask(PipelineTask):
    name = "extract_insights"

    @classmethod
    async def run(cls, documents: tuple[AnalysisDocument | ShowcaseConfigDocument, ...]) -> tuple[InsightDocument, ...]:
        logger.info("Running %s", cls.name)
        cfg = find_document(documents, ShowcaseConfigDocument).parsed
        analysis = find_document(documents, AnalysisDocument)
        options = ModelOptions(reasoning_effort=cfg.reasoning_effort)
        conv = Conversation(model=cfg.fast_model, model_options=options).with_context(analysis)
        conv = await conv.send_structured(
            "Extract structured insights.",
            response_format=InsightModel,
            purpose=f"extract insights from {analysis.name}",
        )
        parsed = conv.parsed
        if parsed is None:
            raise RuntimeError(f"Structured output parsing failed for '{analysis.name}'")
        return (
            InsightDocument.derive(
                from_documents=(analysis,),
                name=f"insight_{analysis.id}.json",
                content=parsed,
            ),
        )


class CompileReportTask(PipelineTask):
    name = "compile_report"

    @classmethod
    async def run(cls, documents: tuple[InsightDocument, ...]) -> tuple[ReportDocument, ...]:
        logger.info("Running %s", cls.name)
        insights: list[InsightModel] = [document.parsed for document in documents]
        lines = ["# Showcase Report", "", f"Insights: {len(insights)}", ""]
        for idx, insight in enumerate(insights, start=1):
            lines.append(f"## Insight {idx}")
            lines.append(f"Complexity: {insight.complexity}")
            lines.extend(f"- {finding}" for finding in insight.findings)
            lines.append("")
        return (
            ReportDocument.derive(
                from_documents=tuple(documents),
                name="report.md",
                content="\n".join(lines),
            ),
        )


class AnalysisFlow(PipelineFlow):
    estimated_minutes = 5

    async def run(
        self,
        documents: tuple[InputDocument | ShowcaseConfigDocument, ...],
        options: ShowcaseFlowOptions,
    ) -> tuple[AnalysisDocument, ...]:
        logger.info("Running %s [%s]", type(self).name, get_run_id())
        cfg = find_document(documents, ShowcaseConfigDocument)
        outputs: list[AnalysisDocument] = []
        for source in [doc for doc in documents if isinstance(doc, InputDocument)]:
            outputs.extend(await AnalyzeDocumentTask.run((source, cfg)))
        return tuple(outputs)


class ExtractionFlow(PipelineFlow):
    estimated_minutes = 3

    async def run(
        self,
        documents: tuple[AnalysisDocument | ShowcaseConfigDocument, ...],
        options: ShowcaseFlowOptions,
    ) -> tuple[InsightDocument, ...]:
        logger.info("Running %s [%s]", type(self).name, get_run_id())
        cfg = find_document(documents, ShowcaseConfigDocument)
        outputs: list[InsightDocument] = []
        for analysis in [doc for doc in documents if isinstance(doc, AnalysisDocument)]:
            outputs.extend(await ExtractInsightsTask.run((analysis, cfg)))
        return tuple(outputs)


class ReportFlow(PipelineFlow):
    estimated_minutes = 1

    async def run(
        self,
        documents: tuple[InsightDocument | AnalysisDocument | ShowcaseConfigDocument, ...],
        options: ShowcaseFlowOptions,
    ) -> tuple[ReportDocument, ...]:
        logger.info("Running %s [%s]", type(self).name, get_run_id())
        insights = [doc for doc in documents if isinstance(doc, InsightDocument)]
        return await CompileReportTask.run(tuple(insights))


class ShowcaseResult(DeploymentResult):
    analysis_count: int = 0
    insight_count: int = 0
    report_count: int = 0


class ShowcasePipeline(PipelineDeployment[ShowcaseFlowOptions, ShowcaseResult]):
    pubsub_service_type: ClassVar[str] = ""

    def build_flows(self, options: ShowcaseFlowOptions) -> list[PipelineFlow]:
        logger.info("Building flows for %s", type(self).__name__)
        return [AnalysisFlow(), ExtractionFlow(), ReportFlow()]

    @staticmethod
    def build_result(
        run_id: str,
        documents: tuple[Document, ...],
        options: ShowcaseFlowOptions,
    ) -> ShowcaseResult:
        _ = (run_id, options)
        return ShowcaseResult(
            success=True,
            analysis_count=len([d for d in documents if isinstance(d, AnalysisDocument)]),
            insight_count=len([d for d in documents if isinstance(d, InsightDocument)]),
            report_count=len([d for d in documents if isinstance(d, ReportDocument)]),
        )


showcase_pipeline = ShowcasePipeline()


def initialize_showcase(options: ShowcaseFlowOptions) -> tuple[str, tuple[Document, ...]]:
    cfg = ShowcaseConfigDocument.create_root(
        name="showcase_config.json",
        content=ShowcaseConfig(
            core_model=options.core_model,
            fast_model=options.fast_model,
            reasoning_effort=options.reasoning_effort,
        ),
        reason="showcase configuration document",
    )
    docs: tuple[Document, ...] = (
        cfg,
        InputDocument.create_root(
            name="notes_a.txt",
            content="AI pipelines benefit from immutable typed documents and explicit provenance.",
            reason="showcase sample input A",
        ),
        InputDocument.create_root(
            name="notes_b.txt",
            content="Class-based tasks simplify import-time validation and observability.",
            reason="showcase sample input B",
        ),
    )
    return "showcase-v2", docs


def main() -> None:
    setup_logging(level="INFO")
    options = ShowcaseFlowOptions()
    run_id, documents = initialize_showcase(options)
    result = showcase_pipeline.run_local(
        run_id=run_id,
        documents=documents,
        options=options,
    )
    print(result.model_dump_json(indent=2))
    print("\nUse showcase_pipeline.run_cli(initializer=initialize_showcase) when you want persistent execution data for ai-trace and ai-replay.")


if __name__ == "__main__":
    main()
