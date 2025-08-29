"""Prefect core features for pipeline orchestration.

This module provides clean re-exports of essential Prefect functionality.

IMPORTANT: You should NEVER use the `task` and `flow` decorators directly
unless it is 100% impossible to use `pipeline_task` and `pipeline_flow`.
The standard Prefect decorators are exported here only for extremely
limited edge cases where the pipeline decorators cannot be used.

Always prefer:
    >>> from ai_pipeline_core import pipeline_task, pipeline_flow
    >>>
    >>> @pipeline_task
    >>> async def my_task(...): ...
    >>>
    >>> @pipeline_flow
    >>> async def my_flow(...): ...

The `task` and `flow` decorators should only be used when:
- You absolutely cannot convert to async (pipeline decorators require async)
- You have a very specific Prefect integration that conflicts with tracing
- You are writing test utilities or infrastructure code

Exported components:
    task: Prefect task decorator (AVOID - use pipeline_task instead).
    flow: Prefect flow decorator (AVOID - use pipeline_flow instead).
    disable_run_logger: Context manager to suppress Prefect logging.
    prefect_test_harness: Test harness for unit testing flows/tasks.

Testing utilities (use as fixtures):
    The disable_run_logger and prefect_test_harness should be used as
    pytest fixtures as shown in tests/conftest.py:

    >>> @pytest.fixture(autouse=True, scope="session")
    >>> def prefect_test_fixture():
    ...     with prefect_test_harness():
    ...         yield
    >>>
    >>> @pytest.fixture(autouse=True)
    >>> def disable_prefect_logging():
    ...     with disable_run_logger():
    ...         yield

Note:
    The pipeline_task and pipeline_flow decorators from
    ai_pipeline_core.pipeline provide async-only execution with
    integrated LMNR tracing and are the standard for this library.
"""

from prefect import flow, task
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness

__all__ = ["task", "flow", "disable_run_logger", "prefect_test_harness"]
