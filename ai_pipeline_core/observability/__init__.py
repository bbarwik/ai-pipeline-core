"""Observability system for AI pipelines.

Contains local debug tracing, ClickHouse-based tracking, initialization utilities,
and the ``ai-trace`` CLI for trace listing, inspection, and download.
"""

from ai_pipeline_core.observability._debug._config import TraceDebugConfig
from ai_pipeline_core.observability.tracing import TraceInfo, TraceLevel, set_trace_cost, trace

__all__ = [
    "TraceDebugConfig",
    "TraceInfo",
    "TraceLevel",
    "set_trace_cost",
    "trace",
]
