"""Common test fixtures for pipeline projects."""

import logging
import warnings

import pytest

from ai_pipeline_core import disable_run_logger, prefect_test_harness


# Suppress SQLAlchemy connection cleanup errors during test teardown
class SQLAlchemyConnectionFilter(logging.Filter):
    """Filter to suppress SQLAlchemy connection termination errors during shutdown.

    These errors occur when Prefect's test harness shuts down and cancels async
    database connections. They are harmless but noisy, so we suppress them.
    """

    def filter(self, record):
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
