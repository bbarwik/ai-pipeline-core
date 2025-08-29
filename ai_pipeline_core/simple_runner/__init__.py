"""Simple pipeline execution for local development.

Utilities for running AI pipelines locally without full Prefect orchestration.
"""

from .cli import run_cli
from .simple_runner import (
    ConfigSequence,
    FlowSequence,
    load_documents_from_directory,
    run_pipeline,
    run_pipelines,
    save_documents_to_directory,
)

__all__ = [
    "run_cli",
    "run_pipeline",
    "run_pipelines",
    "load_documents_from_directory",
    "save_documents_to_directory",
    "FlowSequence",
    "ConfigSequence",
]
