"""Simple pipeline execution for local development.

Utilities for running AI pipelines locally without full Prefect orchestration.
"""

from .cli import run_cli
from .simple_runner import FlowSequence, run_pipeline, run_pipelines

__all__ = [
    "run_cli",
    "run_pipeline",
    "run_pipelines",
    "FlowSequence",
]
