#!/usr/bin/env python3
"""MemoryDatabase showcase for DatabaseReader and DatabaseWriter.

Demonstrates practical usage patterns:
  - pipeline runs persisting nodes and documents into MemoryDatabase
  - direct node insertion with DatabaseWriter
  - DatabaseReader lookups for documents, searches, ancestry, and run scopes
"""

import asyncio
from datetime import UTC, datetime
from typing import override
from uuid import UUID, uuid4

from ai_pipeline_core import (
    DeploymentResult,
    Document,
    FlowOptions,
    PipelineDeployment,
    PipelineFlow,
    PipelineTask,
    RunScope,
)
from ai_pipeline_core.database import (
    DatabaseReader,
    DatabaseWriter,
    ExecutionNode,
    MemoryDatabase,
    NodeKind,
    NodeStatus,
)
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

RUN_SCOPE_LIMIT = 5
SEARCH_LIMIT = 10
MANUAL_NOTE_SEQUENCE = 3


class RawDataDocument(Document):
    """Root input stored at the deployment boundary."""


class CleanedDataDocument(Document):
    """Normalized output from cleaning."""


class SummaryReportDocument(Document):
    """Final summary produced from cleaned documents."""


class CleanDataTask(PipelineTask):
    """Normalize root inputs before summarization."""

    name = "clean-data"

    @classmethod
    async def run(cls, documents: tuple[RawDataDocument, ...]) -> tuple[CleanedDataDocument, ...]:
        logger.info("Running %s", cls.name)
        return tuple(
            CleanedDataDocument.derive(
                from_documents=(raw,),
                name=f"cleaned_{raw.name}",
                content=" ".join(raw.text.split()).upper(),
            )
            for raw in documents
        )


class BuildSummaryTask(PipelineTask):
    """Compile cleaned outputs into a single markdown summary."""

    name = "build-summary"

    @classmethod
    async def run(cls, documents: tuple[CleanedDataDocument, ...]) -> tuple[SummaryReportDocument, ...]:
        logger.info("Running %s", cls.name)
        lines = ["# Summary", "", f"Total documents: {len(documents)}", ""]
        for index, document in enumerate(documents, start=1):
            lines.append(f"- Doc {index} ({document.name}): {document.text[:60]}")
        return (
            SummaryReportDocument.derive(
                from_documents=tuple(documents),
                name="summary.md",
                content="\n".join(lines),
            ),
        )


class CleaningFlow(PipelineFlow):
    """First flow: normalize raw inputs."""

    @override
    async def run(
        self,
        documents: tuple[RawDataDocument, ...],
        options: FlowOptions,
    ) -> tuple[CleanedDataDocument, ...]:
        _ = options
        return await CleanDataTask.run(documents)


class SummaryFlow(PipelineFlow):
    """Second flow: build a final summary."""

    @override
    async def run(
        self,
        documents: tuple[CleanedDataDocument, ...],
        options: FlowOptions,
    ) -> tuple[SummaryReportDocument, ...]:
        _ = options
        return await BuildSummaryTask.run(documents)


class DatabaseShowcaseResult(DeploymentResult):
    """Small result model for the example deployment."""

    summary_preview: str = ""
    document_count: int = 0


class DatabaseShowcasePipeline(PipelineDeployment[FlowOptions, DatabaseShowcaseResult]):
    """Minimal deployment used to populate MemoryDatabase."""

    @override
    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        _ = options
        return [CleaningFlow(), SummaryFlow()]

    @staticmethod
    def build_result(
        run_id: str,
        documents: tuple[Document, ...],
        options: FlowOptions,
    ) -> DatabaseShowcaseResult:
        _ = (run_id, options)
        summaries = [document for document in documents if isinstance(document, SummaryReportDocument)]
        if not summaries:
            return DatabaseShowcaseResult(success=False, error="No summary produced")
        return DatabaseShowcaseResult(
            success=True,
            summary_preview=summaries[0].text[:200],
            document_count=len(documents),
        )


def _completed_node(
    *,
    node_id: UUID,
    deployment_id: UUID,
    run_id: str,
    run_scope: RunScope,
    parent_node_id: UUID,
    sequence_no: int,
) -> ExecutionNode:
    timestamp = datetime.now(UTC)
    return ExecutionNode(
        node_id=node_id,
        node_kind=NodeKind.TASK,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        run_id=run_id,
        run_scope=run_scope,
        deployment_name="database-showcase",
        name="manual-note",
        sequence_no=sequence_no,
        parent_node_id=parent_node_id,
        task_class="ManualDatabaseNote",
        status=NodeStatus.COMPLETED,
        started_at=timestamp,
        ended_at=timestamp,
        updated_at=timestamp,
    )


async def _run_showcase_pipeline(
    pipeline: DatabaseShowcasePipeline,
    database: MemoryDatabase,
    *,
    run_id: str,
    documents: tuple[RawDataDocument, ...],
) -> None:
    await pipeline.run(run_id, documents, FlowOptions(), database=database)


async def main() -> None:
    database = MemoryDatabase()
    writer: DatabaseWriter = database
    reader: DatabaseReader = database
    pipeline = DatabaseShowcasePipeline()

    await _run_showcase_pipeline(
        pipeline,
        database,
        run_id="examples-database-main",
        documents=(
            RawDataDocument.create_root(
                name="main_file_a.txt",
                content="First raw document with duplicate onboarding steps.",
                reason="seed the main database showcase run",
            ),
            RawDataDocument.create_root(
                name="main_file_b.txt",
                content="Second raw document with inconsistent button labels.",
                reason="seed the main database showcase run",
            ),
        ),
    )
    await _run_showcase_pipeline(
        pipeline,
        database,
        run_id="examples-database-archive",
        documents=(
            RawDataDocument.create_root(
                name="archive_file_a.txt",
                content="Archived batch used only to demonstrate multiple run scopes.",
                reason="seed the archive database showcase run",
            ),
        ),
    )

    deployment = await reader.get_deployment_by_run_id("examples-database-main")
    if deployment is None:
        raise RuntimeError("Expected the main deployment node to exist in MemoryDatabase.")

    await writer.insert_node(
        _completed_node(
            node_id=uuid4(),
            deployment_id=deployment.deployment_id,
            run_id=deployment.run_id,
            run_scope=deployment.run_scope,
            parent_node_id=deployment.node_id,
            sequence_no=MANUAL_NOTE_SEQUENCE,
        )
    )

    deployment_tree = await reader.get_deployment_tree(deployment.deployment_id)
    clean_task = next((node for node in deployment_tree if node.name == CleanDataTask.name), None)
    if clean_task is None:
        raise RuntimeError("Expected a clean-data task node in the deployment tree.")
    produced_by_clean = await reader.get_documents_by_node(clean_task.node_id)

    exact_lookup = await reader.find_document_by_name("main_file_a.txt")
    if exact_lookup is None:
        raise RuntimeError("Expected 'main_file_a.txt' to exist in MemoryDatabase.")
    scoped_summary_results = await reader.search_documents(
        name="summary.md",
        document_type=SummaryReportDocument.__name__,
        run_scope=str(deployment.run_scope),
        limit=1,
        offset=0,
    )
    if not scoped_summary_results:
        raise RuntimeError("Expected a scoped summary document for the main run scope.")
    scoped_summary = scoped_summary_results[0]
    cleaned_search = await reader.search_documents(
        name="cleaned_",
        document_type=CleanedDataDocument.__name__,
        run_scope=str(deployment.run_scope),
        limit=SEARCH_LIMIT,
        offset=0,
    )
    ancestry = await reader.get_document_ancestry(scoped_summary.document_sha256)
    run_scopes = await reader.list_run_scopes(limit=RUN_SCOPE_LIMIT)

    print("Execution nodes:")
    for node in deployment_tree:
        print(f"  - {node.node_kind.value}: {node.name}")

    print("\nDocuments produced by clean-data:")
    for record in produced_by_clean:
        print(f"  - {record.name} [{record.document_type}]")

    print("\nExact lookup:")
    print(f"  - {exact_lookup.name} [{exact_lookup.document_type}] in run scope {exact_lookup.run_scope}")

    print("\nSearch results:")
    for record in cleaned_search:
        print(f"  - {record.name}")

    print("\nSummary ancestry:")
    for record in ancestry.values():
        print(f"  - {record.name} [{record.document_type}]")

    print("\nRun scopes:")
    for scope_info in run_scopes:
        print(f"  - {scope_info.run_scope}: {scope_info.document_count} document(s)")


if __name__ == "__main__":
    asyncio.run(main())
