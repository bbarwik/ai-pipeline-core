"""Common test fixtures for pipeline projects."""

import logging
import shutil
import subprocess
import warnings

import pytest

from ai_pipeline_core.document_store._protocol import set_document_store
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.documents import RunContext
from ai_pipeline_core.documents._context import _reset_run_context, _set_run_context
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness


class _SQLAlchemyConnectionFilter(logging.Filter):
    """Suppress SQLAlchemy connection termination errors during Prefect shutdown."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "sqlalchemy.pool.impl.AsyncAdaptedQueuePool":
            if "Exception terminating connection" in record.getMessage():
                return False
        if "aiosqlite" in record.name and "CancelledError" in record.getMessage():
            return False
        return True


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    """Session-scoped Prefect ephemeral database shared across all test directories."""
    sqlalchemy_logger = logging.getLogger("sqlalchemy.pool.impl.AsyncAdaptedQueuePool")
    aiosqlite_logger = logging.getLogger("aiosqlite")
    filter_instance = _SQLAlchemyConnectionFilter()
    sqlalchemy_logger.addFilter(filter_instance)
    aiosqlite_logger.addFilter(filter_instance)
    warnings.filterwarnings("ignore", message=".*CancelledError.*", category=Warning)
    warnings.filterwarnings("ignore", message=".*unclosed.*", category=ResourceWarning)
    try:
        with prefect_test_harness():
            yield
    finally:
        sqlalchemy_logger.removeFilter(filter_instance)
        aiosqlite_logger.removeFilter(filter_instance)


@pytest.fixture(autouse=True)
def disable_prefect_logging():
    """Disable Prefect run logger to prevent RuntimeError from missing flow context."""
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
    token = _set_run_context(ctx)
    yield ctx
    _reset_run_context(token)


# ---------------------------------------------------------------------------
# Infrastructure availability detection
# ---------------------------------------------------------------------------

DOCKER_INFO_TIMEOUT_SECONDS = 10


@pytest.fixture(scope="session")
def docker_available() -> bool:
    """Check once per session whether Docker daemon is reachable."""
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=DOCKER_INFO_TIMEOUT_SECONDS,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="session")
def require_docker(docker_available: bool) -> None:
    """Skip the entire test session/module if Docker is not available."""
    if not docker_available:
        pytest.skip("Docker not available — install Docker or start the daemon")
