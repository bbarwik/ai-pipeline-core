"""Document store tests don't need a Prefect ephemeral server.

Override the root conftest's session-scoped prefect_test_fixture with a no-op
so ClickHouse testcontainer tests don't waste time (or timeout) spinning up
Prefect servers — especially under xdist parallelization.
"""

import pytest


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    """No-op override — document store tests are Prefect-independent."""
    return


@pytest.fixture(autouse=True)
def disable_prefect_logging():
    """No-op override — no Prefect logging context needed."""
    return
