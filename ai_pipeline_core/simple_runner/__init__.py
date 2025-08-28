"""Simple pipeline execution framework for local development.

@public

The simple_runner module provides utilities for running AI pipelines
locally without full Prefect orchestration. It's designed for rapid
prototyping, testing, and simple workflows that don't need distributed
execution or advanced scheduling.

Key features:
    - Sequential flow execution without Prefect server
    - Directory-based document loading and saving
    - CLI interface for command-line execution
    - Flow chaining with automatic document passing
    - Test environment detection and configuration

Main components:
    run_pipeline: Execute a single flow with documents
    run_pipelines: Execute multiple flows in sequence
    run_cli: Command-line interface for flow execution
    load_documents_from_directory: Load documents from filesystem
    save_documents_to_directory: Save results to filesystem
    FlowSequence: Type for flow + config pairs
    ConfigSequence: Type for config class + options pairs

Example:
    >>> from ai_pipeline_core.simple_runner import run_pipeline
    >>> from my_flows import AnalysisFlow, AnalysisConfig
    >>>
    >>> # Run single flow
    >>> results = await run_pipeline(
    ...     project_name="test_project",
    ...     flow=AnalysisFlow,
    ...     config_class=AnalysisConfig,
    ...     input_dir="./data",
    ...     output_dir="./results"
    ... )
    >>>
    >>> # Or use CLI
    >>> # python -m my_module --input-dir ./data --output-dir ./results

Note:
    Simple runner is for local development. For production,
    use Prefect deployment with proper orchestration.
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
