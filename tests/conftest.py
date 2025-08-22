"""Common test fixtures for pipeline projects."""

import pytest

from ai_pipeline_core import disable_run_logger, prefect_test_harness


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    """
    Session-scoped fixture that runs all tests against a temporary Prefect database.
    This isolates tests and prevents them from affecting the main Prefect database.
    """
    with prefect_test_harness():
        yield


@pytest.fixture(autouse=True)
def disable_prefect_logging():
    """
    Function-scoped fixture that disables Prefect run logger for each test.
    This prevents RuntimeError from missing flow context when testing tasks/flows directly.
    """
    with disable_run_logger():
        yield
