#!/usr/bin/env python3
"""Document-store showcase using class-based PipelineTask/PipelineFlow."""

from ai_pipeline_core import (
    DeploymentResult,
    Document,
    FlowOptions,
    PipelineDeployment,
    PipelineFlow,
    PipelineTask,
)
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)


class RawDataDocument(Document):
    """Root input."""


class CleanedDataDocument(Document):
    """Normalized output from cleaning."""


class SummaryReportDocument(Document):
    """Final report."""


class CleanDataTask(PipelineTask):
    @classmethod
    async def run(cls, documents: list[RawDataDocument]) -> list[CleanedDataDocument]:
        logger.debug("Running %s", cls.name)
        return [
            CleanedDataDocument.derive(
                from_documents=(raw,),
                name=f"cleaned_{raw.name}",
                content=" ".join(raw.text.split()).upper(),
            )
            for raw in documents
        ]


class BuildSummaryTask(PipelineTask):
    @classmethod
    async def run(cls, documents: list[CleanedDataDocument]) -> list[SummaryReportDocument]:
        logger.debug("Running %s", cls.name)
        lines = ["# Summary", "", f"Total documents: {len(documents)}", ""]
        for idx, doc in enumerate(documents, start=1):
            lines.append(f"- Doc {idx} ({doc.name}): {doc.text[:60]}")
        return [
            SummaryReportDocument.create(
                name="summary.md",
                content="\n".join(lines),
                derived_from=tuple(doc.sha256 for doc in documents),
            )
        ]


class CleaningFlow(PipelineFlow):
    async def run(
        self,
        run_id: str,
        documents: list[RawDataDocument],
        options: FlowOptions,
    ) -> list[CleanedDataDocument]:
        logger.debug("Running %s [%s]", type(self).name, run_id)
        return await CleanDataTask.run(documents)


class SummaryFlow(PipelineFlow):
    async def run(
        self,
        run_id: str,
        documents: list[CleanedDataDocument],
        options: FlowOptions,
    ) -> list[SummaryReportDocument]:
        logger.debug("Running %s [%s]", type(self).name, run_id)
        return await BuildSummaryTask.run(documents)


class StoreShowcaseResult(DeploymentResult):
    summary_preview: str = ""
    document_count: int = 0


class StoreShowcasePipeline(PipelineDeployment[FlowOptions, StoreShowcaseResult]):
    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        logger.debug("Building flows for %s", type(self).__name__)
        return [CleaningFlow(), SummaryFlow()]

    @staticmethod
    def build_result(
        run_id: str,
        documents: list[Document],
        options: FlowOptions,
    ) -> StoreShowcaseResult:
        _ = (run_id, options)
        summaries = [d for d in documents if isinstance(d, SummaryReportDocument)]
        if not summaries:
            return StoreShowcaseResult(success=False, error="No summary produced")
        return StoreShowcaseResult(
            success=True,
            summary_preview=summaries[0].text[:200],
            document_count=len(documents),
        )


def main() -> None:
    pipeline = StoreShowcasePipeline()
    input_docs: list[Document] = [
        RawDataDocument.create_root(
            name="file_a.txt",
            content="First raw document with important data",
            reason="showcase sample input data for document store demo",
        ),
        RawDataDocument.create_root(
            name="file_b.txt",
            content="Second raw document with more data",
            reason="showcase sample input data for document store demo",
        ),
    ]
    result = pipeline.run_local(
        run_id="store-showcase",
        documents=input_docs,
        options=FlowOptions(),
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
