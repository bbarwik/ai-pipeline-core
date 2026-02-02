"""Logging bridge — captures Python log records as OTel span events.

Attaches a singleton ``SpanEventLoggingHandler`` to every logger created
via ``get_pipeline_logger()``.  The handler is safe to attach eagerly
because ``emit()`` is a no-op when no OTel span is recording.

This is the only module that legitimately needs ``import logging`` directly
to subclass ``logging.Handler``. The ruff ban on ``import logging``
(pyproject.toml) is suppressed with ``# noqa: TID251``.
"""

import contextlib
import logging  # noqa: TID251

from opentelemetry import trace as otel_trace

_MIN_LEVEL = logging.INFO


class SpanEventLoggingHandler(logging.Handler):
    """Logging handler that writes log records as OTel span events.

    Attached to each logger returned by ``get_pipeline_logger()``.
    Only captures records at INFO level and above. Each record becomes
    a span event with ``log.level`` and ``log.message`` attributes.
    """

    def __init__(self) -> None:
        super().__init__(level=_MIN_LEVEL)

    def emit(self, record: logging.LogRecord) -> None:
        """Write a log record as an OTel span event."""
        with contextlib.suppress(Exception):
            # Prevent duplicate events when handler is on both parent and child logger
            if getattr(record, "_span_event_logged", False):
                return
            span = otel_trace.get_current_span()
            if not span.is_recording():
                return
            span.add_event(
                name="log",
                attributes={
                    "log.level": record.levelname,
                    "log.message": self.format(record),
                    "log.logger": record.name,
                },
            )
            record._span_event_logged = True


# Module-level singleton — safe because emit() checks is_recording().
_bridge_handler = SpanEventLoggingHandler()


def get_bridge_handler() -> SpanEventLoggingHandler:
    """Return the singleton bridge handler for attaching to pipeline loggers."""
    return _bridge_handler
