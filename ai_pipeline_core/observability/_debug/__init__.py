"""Local trace debugging system for AI pipelines.

This module provides filesystem-based trace debugging that saves all spans
with their inputs/outputs for LLM-assisted debugging. Includes static
summary generation.

Enabled automatically in CLI mode (``run_cli``), writing to ``<working_dir>/.trace``.
Disable with ``--no-trace``.
"""

from ._config import SpanInfo, TraceDebugConfig, TraceState, WriteJob
from ._content import ContentWriter
from ._processor import LocalDebugSpanProcessor
from ._summary import generate_costs, generate_summary
from ._writer import LocalTraceWriter

__all__ = [
    "ContentWriter",
    "LocalDebugSpanProcessor",
    "LocalTraceWriter",
    "SpanInfo",
    "TraceDebugConfig",
    "TraceState",
    "WriteJob",
    "generate_costs",
    "generate_summary",
]
