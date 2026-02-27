"""Local debug tracing system for AI pipelines.

This module provides filesystem-based debug tracing that saves all spans
with their inputs/outputs for LLM-assisted debugging. Includes static
summary generation.

Enabled automatically in CLI mode (``run_cli``), writing to ``<working_dir>/.trace``.
Disable with ``--no-trace``.
"""

from ._backend import FilesystemBackend
from ._config import TraceDebugConfig
from ._content import ContentWriter
from ._materializer import TraceMaterializer

__all__ = [
    "ContentWriter",
    "FilesystemBackend",
    "TraceDebugConfig",
    "TraceMaterializer",
]
