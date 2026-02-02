"""Tests for pipeline_flow with DocumentStore integration."""

import pytest

from ai_pipeline_core import pipeline_flow
from ai_pipeline_core.document_store import set_document_store
from ai_pipeline_core.document_store.memory import MemoryDocumentStore
from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents.context import RunContext, reset_run_context, set_run_context
from ai_pipeline_core.pipeline.options import FlowOptions


class StorageInputDoc(Document):
    """Storage test input document."""


class StorageOutputDoc(Document):
    """Storage test output document."""


@pytest.fixture
def memory_store():
    """Set up and tear down a MemoryDocumentStore for testing."""
    store = MemoryDocumentStore()
    set_document_store(store)
    yield store
    set_document_store(None)


@pytest.fixture
def run_context():
    """Set up and tear down a RunContext for testing."""
    token = set_run_context(RunContext(run_scope="test-project"))
    yield
    reset_run_context(token)


@pytest.mark.asyncio
async def test_pipeline_flow_returns_documents_with_store_configured(prefect_test_fixture, memory_store, run_context):
    """Test that pipeline_flow returns documents correctly when a store is configured."""

    @pipeline_flow()
    async def test_flow(project_name: str, documents: list[StorageInputDoc], flow_options: FlowOptions) -> list[StorageOutputDoc]:
        return [StorageOutputDoc(name="output.txt", content=b"test output")]

    input_docs = [StorageInputDoc(name="input.txt", content=b"test input")]

    result = await test_flow("test-project", input_docs, FlowOptions())

    assert len(result) == 1
    assert isinstance(result[0], StorageOutputDoc)


@pytest.mark.asyncio
async def test_pipeline_flow_works_without_store(prefect_test_fixture):
    """Test that pipeline_flow works when no store is configured."""
    set_document_store(None)

    @pipeline_flow()
    async def test_flow(project_name: str, documents: list[StorageInputDoc], flow_options: FlowOptions) -> list[StorageOutputDoc]:
        return [StorageOutputDoc(name="output.txt", content=b"test output")]

    result = await test_flow("test-project", [], FlowOptions())

    assert len(result) == 1
    assert isinstance(result[0], StorageOutputDoc)


@pytest.mark.asyncio
async def test_pipeline_flow_sets_run_context_when_missing(prefect_test_fixture, memory_store):
    """Test that pipeline_flow sets RunContext if none exists."""
    from ai_pipeline_core.documents.context import get_run_context

    captured_ctx = None

    @pipeline_flow()
    async def test_flow(project_name: str, documents: list[StorageInputDoc], flow_options: FlowOptions) -> list[StorageOutputDoc]:
        nonlocal captured_ctx
        captured_ctx = get_run_context()
        return []

    await test_flow("my-project", [], FlowOptions())

    assert captured_ctx is not None
    assert captured_ctx.run_scope == "my-project/test_flow"


@pytest.mark.asyncio
async def test_pipeline_flow_preserves_existing_run_context(prefect_test_fixture, memory_store, run_context):
    """Test that pipeline_flow does not override RunContext set by deployment."""
    from ai_pipeline_core.documents.context import get_run_context

    captured_ctx = None

    @pipeline_flow()
    async def test_flow(project_name: str, documents: list[StorageInputDoc], flow_options: FlowOptions) -> list[StorageOutputDoc]:
        nonlocal captured_ctx
        captured_ctx = get_run_context()
        return []

    await test_flow("my-project", [], FlowOptions())

    assert captured_ctx is not None
    # Should use the deployment-level context, not create a new one
    assert captured_ctx.run_scope == "test-project"


@pytest.mark.asyncio
async def test_pipeline_flow_saves_returned_documents(prefect_test_fixture, memory_store: MemoryDocumentStore, run_context):
    """Test that @pipeline_flow saves returned documents to the store."""

    @pipeline_flow()
    async def test_flow(project_name: str, documents: list[StorageInputDoc], flow_options: FlowOptions) -> list[StorageOutputDoc]:
        return [StorageOutputDoc(name="output.txt", content=b"test output")]

    input_docs = [StorageInputDoc(name="input.txt", content=b"test input")]
    await test_flow("test-project", input_docs, FlowOptions())

    loaded = await memory_store.load("test-project", [StorageOutputDoc])
    assert len(loaded) == 1
    assert loaded[0].name == "output.txt"


@pytest.mark.asyncio
async def test_pipeline_flow_deduplicates_returned_documents(prefect_test_fixture, memory_store: MemoryDocumentStore, run_context):
    """Test that @pipeline_flow deduplicates returned documents by SHA256."""
    doc = StorageOutputDoc(name="output.txt", content=b"test output")

    @pipeline_flow()
    async def test_flow(project_name: str, documents: list[StorageInputDoc], flow_options: FlowOptions) -> list[StorageOutputDoc]:
        return [doc, doc]  # Same document twice

    await test_flow("test-project", [], FlowOptions())

    loaded = await memory_store.load("test-project", [StorageOutputDoc])
    assert len(loaded) == 1
