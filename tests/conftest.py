"""Common test fixtures for pipeline projects."""

import logging
import warnings

import pytest

from ai_pipeline_core import disable_run_logger, prefect_test_harness
from ai_pipeline_core.document_store import set_document_store
from ai_pipeline_core.document_store.memory import MemoryDocumentStore
from ai_pipeline_core.documents.context import RunContext, reset_run_context, set_run_context


# Suppress SQLAlchemy connection cleanup errors during test teardown
class SQLAlchemyConnectionFilter(logging.Filter):
    """Filter to suppress SQLAlchemy connection termination errors during shutdown.

    These errors occur when Prefect's test harness shuts down and cancels async
    database connections. They are harmless but noisy, so we suppress them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Suppress AsyncAdaptedQueuePool errors related to connection termination
        if record.name == "sqlalchemy.pool.impl.AsyncAdaptedQueuePool":
            if "Exception terminating connection" in record.getMessage():
                return False
        # Also suppress related aiosqlite errors
        if "aiosqlite" in record.name and "CancelledError" in record.getMessage():
            return False
        return True


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    """Session-scoped fixture that runs all tests against a temporary Prefect database.

    This isolates tests and prevents them from affecting the main Prefect database.
    Also suppresses harmless SQLAlchemy connection cleanup errors that occur during
    test teardown when async connections are cancelled.
    """
    # Add filter to suppress SQLAlchemy connection cleanup errors
    sqlalchemy_logger = logging.getLogger("sqlalchemy.pool.impl.AsyncAdaptedQueuePool")
    aiosqlite_logger = logging.getLogger("aiosqlite")

    filter_instance = SQLAlchemyConnectionFilter()
    sqlalchemy_logger.addFilter(filter_instance)
    aiosqlite_logger.addFilter(filter_instance)

    # Also suppress asyncio.exceptions.CancelledError warnings during cleanup
    warnings.filterwarnings("ignore", message=".*CancelledError.*", category=Warning)

    # Suppress unclosed database warnings
    warnings.filterwarnings("ignore", message=".*unclosed.*", category=ResourceWarning)

    try:
        with prefect_test_harness():
            yield
    finally:
        # Clean up the filters after tests
        sqlalchemy_logger.removeFilter(filter_instance)
        aiosqlite_logger.removeFilter(filter_instance)


@pytest.fixture(autouse=True)
def disable_prefect_logging():
    """Function-scoped fixture that disables Prefect run logger for each test.
    This prevents RuntimeError from missing flow context when testing tasks/flows directly.
    """
    with disable_run_logger():
        yield


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
