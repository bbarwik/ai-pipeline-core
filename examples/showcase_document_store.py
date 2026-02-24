#!/usr/bin/env python3
"""Document store showcase — runs standalone without external services.

Demonstrates:
  - Defining Document subclasses
  - @pipeline_task auto-save behavior
  - @pipeline_flow annotation-driven type extraction
  - A simple 2-flow PipelineDeployment with run_local()
  - Reading documents back from the store after pipeline execution

Usage:
  python examples/showcase_document_store.py
"""

from typing import ClassVar

from ai_pipeline_core import (
    DeploymentResult,
    Document,
    FlowOptions,
    PipelineDeployment,
    pipeline_flow,
    pipeline_task,
)

# ---------------------------------------------------------------------------
# Document types
# ---------------------------------------------------------------------------


class RawDataDocument(Document):
    """Raw input data."""


class CleanedDataDocument(Document):
    """Cleaned/processed data."""


class SummaryReportDocument(Document):
    """Final summary report."""


# ---------------------------------------------------------------------------
# Pipeline tasks with auto-save
# ---------------------------------------------------------------------------


@pipeline_task
async def clean_data(raw: RawDataDocument) -> CleanedDataDocument:
    """Clean raw data. Returned document is auto-saved by @pipeline_task."""
    content = " ".join(raw.text.split()).upper()
    return CleanedDataDocument.create(
        name=f"cleaned_{raw.name}",
        content=content,
        derived_from=(raw.sha256,),
    )


@pipeline_task
async def build_summary(
    run_id: str,
    cleaned_docs: list[CleanedDataDocument],
) -> SummaryReportDocument:
    """Summarize cleaned documents into a report."""
    lines = [
        f"# {run_id} Summary",
        "",
        f"Total documents: {len(cleaned_docs)}",
        "",
    ]
    for i, doc in enumerate(cleaned_docs, start=1):
        preview = doc.text[:60]
        lines.append(f"- Doc {i} ({doc.name}): {preview}...")

    return SummaryReportDocument.create(
        name="summary.md",
        content="\n".join(lines),
        derived_from=tuple(d.sha256 for d in cleaned_docs),
    )


# ---------------------------------------------------------------------------
# Pipeline flows with annotation-driven types
# ---------------------------------------------------------------------------


@pipeline_flow(estimated_minutes=1)
async def cleaning_flow(
    run_id: str,
    documents: list[RawDataDocument],
    flow_options: FlowOptions,
) -> list[CleanedDataDocument]:
    """Flow 1: clean all raw data documents."""
    results: list[CleanedDataDocument] = []
    for doc in documents:
        cleaned = await clean_data(doc)
        results.append(cleaned)
    return results


@pipeline_flow(estimated_minutes=1)
async def summary_flow(
    run_id: str,
    documents: list[CleanedDataDocument],
    flow_options: FlowOptions,
) -> list[SummaryReportDocument]:
    """Flow 2: generate summary from cleaned documents."""
    report = await build_summary(run_id, documents)
    return [report]


# ---------------------------------------------------------------------------
# PipelineDeployment with run_local()
# ---------------------------------------------------------------------------


class StoreShowcaseResult(DeploymentResult):
    """Result from the document store showcase pipeline."""

    summary_preview: str = ""
    document_count: int = 0


class StoreShowcasePipeline(PipelineDeployment[FlowOptions, StoreShowcaseResult]):
    """2-flow pipeline: clean raw data, then summarize."""

    flows: ClassVar = [cleaning_flow, summary_flow]

    @staticmethod
    def build_result(
        run_id: str,
        documents: list[Document],
        options: FlowOptions,
    ) -> StoreShowcaseResult:
        summaries = [d for d in documents if isinstance(d, SummaryReportDocument)]
        if summaries:
            return StoreShowcaseResult(
                success=True,
                summary_preview=summaries[0].text[:200],
                document_count=len(documents),
            )
        return StoreShowcaseResult(success=False, error="No summary produced")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run full pipeline via PipelineDeployment.run_local()."""
    print("\n=== PipelineDeployment.run_local() Demo ===\n")

    pipeline = StoreShowcasePipeline()
    input_docs: list[Document] = [
        RawDataDocument.create(name="file_a.txt", content="First raw document with important data"),
        RawDataDocument.create(name="file_b.txt", content="Second raw document with more data"),
        RawDataDocument.create(name="file_c.txt", content="Third raw document with final data"),
    ]

    result = pipeline.run_local(
        run_id="store-showcase",
        documents=input_docs,
        options=FlowOptions(),
    )

    print(f"Pipeline success: {result.success}")
    print(f"Total documents in store: {result.document_count}")
    print(f"Summary preview:\n{result.summary_preview}")
    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
