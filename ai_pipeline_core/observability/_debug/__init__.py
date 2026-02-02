"""Local trace debugging system for AI pipelines.

This module provides filesystem-based trace debugging that saves all spans
with their inputs/outputs for LLM-assisted debugging. Includes static
summary generation and LLM-powered auto-summary capabilities.

Enabled automatically in CLI mode (``run_cli``), writing to ``<working_dir>/.trace``.
Disable with ``--no-trace``.
"""

from ._config import TraceDebugConfig
from ._content import ArtifactStore, ContentRef, ContentWriter, reconstruct_span_content
from ._processor import LocalDebugSpanProcessor
from ._summary import generate_summary
from ._types import SpanInfo, TraceState, WriteJob
from ._writer import LocalTraceWriter

__all__ = [
    "ArtifactStore",
    "ContentRef",
    "ContentWriter",
    "LocalDebugSpanProcessor",
    "LocalTraceWriter",
    "SpanInfo",
    "TraceDebugConfig",
    "TraceState",
    "WriteJob",
    "generate_summary",
    "reconstruct_span_content",
]
