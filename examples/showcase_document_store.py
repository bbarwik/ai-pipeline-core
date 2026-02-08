#!/usr/bin/env python3
"""Document store showcase — runs standalone without external services.

Demonstrates:
  - Defining Document subclasses
  - MemoryDocumentStore and LocalDocumentStore
  - Saving and loading documents with run scoping
  - RunContext for scope management
  - @pipeline_task auto-save behavior
  - @pipeline_flow annotation-driven type extraction
  - A simple 2-flow PipelineDeployment with run_local()

Usage:
  python examples/showcase_document_store.py
"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

from ai_pipeline_core import (
    DeploymentResult,
    Document,
    FlowOptions,
    PipelineDeployment,
    RunContext,
    pipeline_flow,
    pipeline_task,
    reset_run_context,
    set_run_context,
)
from ai_pipeline_core.document_store.local import LocalDocumentStore
from ai_pipeline_core.document_store.memory import MemoryDocumentStore

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
# 1. MemoryDocumentStore: basic operations
# ---------------------------------------------------------------------------


async def demo_memory_store() -> None:
    """Demonstrate MemoryDocumentStore: save, load, deduplication, scoping."""
    print("\n=== MemoryDocumentStore Demo ===\n")

    store = MemoryDocumentStore()
    run_scope = "demo-project"

    # Create documents
    doc1 = RawDataDocument.create(name="data.txt", content="Hello from document store!")
    doc2 = RawDataDocument.create(name="config.json", content={"key": "value", "items": [1, 2, 3]})

    print(f"doc1: name={doc1.name}, sha256={doc1.sha256[:12]}..., size={doc1.size}")
    print(f"doc2: name={doc2.name}, sha256={doc2.sha256[:12]}..., mime={doc2.mime_type}")

    # Save
    await store.save(doc1, run_scope)
    await store.save(doc2, run_scope)
    print(f"\nSaved 2 documents to scope '{run_scope}'")

    # Idempotent save — same document again is a no-op
    await store.save(doc1, run_scope)
    print("Idempotent save: no duplicate created")

    # Load by type
    loaded = await store.load(run_scope, [RawDataDocument])
    print(f"Loaded {len(loaded)} RawDataDocument(s)")

    # has_documents check
    has_raw = await store.has_documents(run_scope, RawDataDocument)
    has_cleaned = await store.has_documents(run_scope, CleanedDataDocument)
    print(f"Has RawDataDocument: {has_raw}")
    print(f"Has CleanedDataDocument: {has_cleaned}")

    # check_existing by SHA256
    existing = await store.check_existing([doc1.sha256, "nonexistent_hash"])
    print(f"Existing SHA256s: {len(existing)} of 2 queried")

    # check_existing by SHA256
    found = doc1.sha256 in await store.check_existing([doc1.sha256])
    print(f"Lookup by SHA256: found={found}")


# ---------------------------------------------------------------------------
# 2. LocalDocumentStore: filesystem persistence
# ---------------------------------------------------------------------------


async def demo_local_store(base_path: Path) -> None:
    """Demonstrate LocalDocumentStore: filesystem layout and persistence."""
    print("\n=== LocalDocumentStore Demo ===\n")

    store = LocalDocumentStore(base_path=base_path)
    run_scope = "local-demo"

    # Create documents with source tracking
    raw = RawDataDocument.create(name="input.txt", content="Raw input data for processing")
    cleaned = CleanedDataDocument.create(
        name="output.txt",
        content=f"Processed: {raw.text.upper()}",
        sources=(raw.sha256,),
    )

    # save_batch saves both in dependency order
    await store.save_batch([raw, cleaned], run_scope)

    scope_path = base_path / run_scope
    print(f"Saved 2 documents to {scope_path}")
    print(f"  {raw.__class__.__name__}/input.txt")
    print(f"  {cleaned.__class__.__name__}/output.txt")

    # Load specific types
    loaded_raw = await store.load(run_scope, [RawDataDocument])
    loaded_cleaned = await store.load(run_scope, [CleanedDataDocument])
    print(f"\nLoaded: {len(loaded_raw)} raw, {len(loaded_cleaned)} cleaned")

    # Verify source tracking round-trips
    if loaded_cleaned:
        lc = loaded_cleaned[0]
        print("\nSource tracking preserved:")
        print(f"  sources: {lc.sources}")
        print(f"  doc sources: {lc.source_documents}")
        print(f"  has raw as source: {lc.has_source(raw)}")


# ---------------------------------------------------------------------------
# 3. RunContext: scope isolation
# ---------------------------------------------------------------------------


async def demo_run_context() -> None:
    """Demonstrate RunContext: scope management via context variables."""
    print("\n=== RunContext Scoping Demo ===\n")

    store = MemoryDocumentStore()

    # Save to "project-alpha" scope
    token_alpha = set_run_context(RunContext(run_scope="project-alpha"))
    doc_alpha = RawDataDocument.create(name="alpha.txt", content="Alpha project data")
    await store.save(doc_alpha, "project-alpha")
    reset_run_context(token_alpha)

    # Save to "project-beta" scope
    token_beta = set_run_context(RunContext(run_scope="project-beta"))
    doc_beta = RawDataDocument.create(name="beta.txt", content="Beta project data")
    await store.save(doc_beta, "project-beta")
    reset_run_context(token_beta)

    # Scopes are fully isolated
    alpha_docs = await store.load("project-alpha", [RawDataDocument])
    beta_docs = await store.load("project-beta", [RawDataDocument])
    print(f"project-alpha: {len(alpha_docs)} doc(s) — {[d.name for d in alpha_docs]}")
    print(f"project-beta:  {len(beta_docs)} doc(s) — {[d.name for d in beta_docs]}")


# ---------------------------------------------------------------------------
# 4. Pipeline tasks with auto-save
# ---------------------------------------------------------------------------


@pipeline_task
async def clean_data(raw: RawDataDocument) -> CleanedDataDocument:
    """Clean raw data. Returned document is auto-saved by @pipeline_task."""
    content = " ".join(raw.text.split()).upper()
    return CleanedDataDocument.create(
        name=f"cleaned_{raw.name}",
        content=content,
        sources=(raw.sha256,),
    )


@pipeline_task
async def build_summary(
    project_name: str,
    cleaned_docs: list[CleanedDataDocument],
) -> SummaryReportDocument:
    """Summarize cleaned documents into a report."""
    lines = [
        f"# {project_name} Summary",
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
        sources=tuple(d.sha256 for d in cleaned_docs),
    )


# ---------------------------------------------------------------------------
# 5. Pipeline flows with annotation-driven types
# ---------------------------------------------------------------------------


@pipeline_flow(estimated_minutes=1)
async def cleaning_flow(
    project_name: str,
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
    project_name: str,
    documents: list[CleanedDataDocument],
    flow_options: FlowOptions,
) -> list[SummaryReportDocument]:
    """Flow 2: generate summary from cleaned documents."""
    report = await build_summary(project_name, documents)
    return [report]


# ---------------------------------------------------------------------------
# 6. PipelineDeployment with run_local()
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
        project_name: str,
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


async def run_store_demos() -> None:
    """Run standalone store demos (pure async, no Prefect needed)."""
    await demo_memory_store()

    with TemporaryDirectory() as tmpdir:
        await demo_local_store(Path(tmpdir))

    await demo_run_context()


def run_pipeline_demo() -> None:
    """Run full pipeline via PipelineDeployment.run_local()."""
    print("\n=== PipelineDeployment.run_local() Demo ===\n")

    pipeline = StoreShowcasePipeline()
    input_docs: list[Document] = [
        RawDataDocument.create(name="file_a.txt", content="First raw document with important data"),
        RawDataDocument.create(name="file_b.txt", content="Second raw document with more data"),
        RawDataDocument.create(name="file_c.txt", content="Third raw document with final data"),
    ]

    result = pipeline.run_local(
        project_name="store-showcase",
        documents=input_docs,
        options=FlowOptions(),
    )

    print(f"Pipeline success: {result.success}")
    print(f"Total documents in store: {result.document_count}")
    print(f"Summary preview:\n{result.summary_preview}")


def main() -> None:
    # Part 1: standalone store operations (pure async, no Prefect)
    asyncio.run(run_store_demos())

    # Part 2: full pipeline via run_local() (uses Prefect test harness + MemoryDocumentStore)
    run_pipeline_demo()

    print("\nAll demos completed successfully.")


if __name__ == "__main__":
    main()
