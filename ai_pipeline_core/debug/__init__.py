"""Local trace debugging system for AI pipelines.

This module provides filesystem-based trace debugging that saves all spans
with their inputs/outputs for LLM-assisted debugging.

Enable by setting TRACE_DEBUG_PATH environment variable.
"""

from .config import TraceDebugConfig
from .content import ArtifactStore, ContentRef, ContentWriter, reconstruct_span_content
from .processor import LocalDebugSpanProcessor
from .summary import generate_summary
from .writer import LocalTraceWriter, TraceState, WriteJob

__all__ = [
    "TraceDebugConfig",
    "ContentRef",
    "ContentWriter",
    "ArtifactStore",
    "reconstruct_span_content",
    "LocalDebugSpanProcessor",
    "LocalTraceWriter",
    "TraceState",
    "WriteJob",
    "generate_summary",
]
