"""Common test fixtures for pipeline projects."""

import pytest

from ai_pipeline_core.document_store import set_document_store
from ai_pipeline_core.document_store.memory import MemoryDocumentStore
from ai_pipeline_core.documents.context import RunContext, reset_run_context, set_run_context


@pytest.fixture
def memory_store():
    """Provide a MemoryDocumentStore and set it as the process-global singleton.

    Automatically cleans up the global singleton after the test.
    """
    store = MemoryDocumentStore()
    set_document_store(store)
    yield store
    set_document_store(None)


@pytest.fixture
def run_context():
    """Provide a RunContext with a deterministic run_scope and set it via ContextVar.

    Automatically resets the ContextVar after the test.
    """
    ctx = RunContext(run_scope="test-run-scope")
    token = set_run_context(ctx)
    yield ctx
    reset_run_context(token)


@pytest.fixture
def pipeline_context(memory_store, run_context):
    """Provide both a MemoryDocumentStore singleton and RunContext.

    Convenience fixture combining memory_store and run_context for
    integration-style tests that need the full document lifecycle.
    """
    return memory_store, run_context
