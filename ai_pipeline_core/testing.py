"""Test utilities for applications built on ai-pipeline-core.

Re-exports Prefect testing helpers so that application test suites don't need
to depend on or import Prefect directly. Prefect is an internal implementation
detail of the framework — apps should only interact with it through ai-pipeline-core.
"""

from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness

__all__ = ["disable_run_logger", "prefect_test_harness"]
