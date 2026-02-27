"""Protocol definitions for the tracking subsystem."""

from typing import Protocol, runtime_checkable

from ai_pipeline_core.observability._span_data import SpanData


@runtime_checkable
class SpanBackend(Protocol):
    """Protocol for span data sinks (ClickHouseBackend, FilesystemBackend)."""

    def on_span_start(self, span_data: SpanData) -> None:
        """Handle span start event."""
        ...

    def on_span_end(self, span_data: SpanData) -> None:
        """Handle span end event."""
        ...

    def shutdown(self) -> None:
        """Flush pending data and release resources."""
        ...
