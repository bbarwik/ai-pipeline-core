"""Integration tests for the logging bridge — verifies app logs become OTel span events."""

import logging
from collections.abc import Generator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._logging_bridge import SpanEventLoggingHandler, get_bridge_handler

type OtelEnv = tuple[InMemorySpanExporter, TracerProvider]


@pytest.fixture
def otel_env() -> Generator[OtelEnv]:
    """Isolated OTel TracerProvider with in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter, provider
    provider.shutdown()


class TestBridgeAttachment:
    """Verify get_pipeline_logger attaches the bridge handler."""

    def test_pipeline_logger_has_bridge_handler(self) -> None:
        logger = get_pipeline_logger("test_bridge_attach")
        assert get_bridge_handler() in logger.handlers

    def test_app_named_logger_gets_handler(self) -> None:
        """Non-framework logger names (e.g. __main__) also get the bridge."""
        logger = get_pipeline_logger("__main__")
        assert get_bridge_handler() in logger.handlers

    def test_handler_attached_once(self) -> None:
        """Calling get_pipeline_logger twice with the same name doesn't duplicate."""
        logger1 = get_pipeline_logger("test_dedup_attach")
        logger2 = get_pipeline_logger("test_dedup_attach")
        assert logger1 is logger2
        count = sum(1 for h in logger1.handlers if isinstance(h, SpanEventLoggingHandler))
        assert count == 1

    def test_third_party_logger_has_no_bridge(self) -> None:
        """Loggers not created via get_pipeline_logger must not have the bridge."""
        httpx_logger = logging.getLogger("httpx")
        assert get_bridge_handler() not in httpx_logger.handlers

        root_logger = logging.getLogger()
        assert get_bridge_handler() not in root_logger.handlers


class TestSpanEventCapture:
    """Verify log records become OTel span events end-to-end."""

    def test_log_captured_as_span_event(self, otel_env: OtelEnv) -> None:
        """Log via get_pipeline_logger() during an active span produces span events."""
        exporter, provider = otel_env
        tracer = provider.get_tracer("test")
        logger = get_pipeline_logger("test_e2e_capture")

        with tracer.start_as_current_span("test-span"):
            logger.info("hello from app")
            logger.warning("warning from app")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        log_events = [e for e in spans[0].events if e.name == "log"]
        assert len(log_events) == 2

        messages = [str(e.attributes.get("log.message", "")) for e in log_events if e.attributes]
        assert any("hello from app" in m for m in messages)
        assert any("warning from app" in m for m in messages)

        levels = [str(e.attributes.get("log.level", "")) for e in log_events if e.attributes]
        assert "INFO" in levels
        assert "WARNING" in levels

    def test_no_span_event_without_active_span(self) -> None:
        """Logging outside a span produces no errors (handler is a no-op)."""
        logger = get_pipeline_logger("test_no_span")
        # Should not raise — handler checks is_recording() and returns early
        logger.info("this should be silently ignored")

    def test_debug_records_filtered_by_handler_level(self, otel_env: OtelEnv) -> None:
        """Handler level is INFO — DEBUG records should not become span events."""
        exporter, provider = otel_env
        tracer = provider.get_tracer("test")
        logger = get_pipeline_logger("test_debug_filter")
        logger.setLevel(logging.DEBUG)

        with tracer.start_as_current_span("test-span"):
            logger.debug("debug msg should be filtered")
            logger.info("info msg should pass")

        spans = exporter.get_finished_spans()
        log_events = [e for e in spans[0].events if e.name == "log"]
        assert len(log_events) == 1
        assert log_events[0].attributes
        assert "info msg should pass" in str(log_events[0].attributes.get("log.message", ""))

    def test_dedup_prevents_double_events_on_propagation(self, otel_env: OtelEnv) -> None:
        """Same record reaching the handler twice (via propagation) produces only one event."""
        exporter, provider = otel_env
        tracer = provider.get_tracer("test")

        get_pipeline_logger("test_dedup_parent")  # register parent so handler is attached
        child_logger = get_pipeline_logger("test_dedup_parent.child")

        with tracer.start_as_current_span("test-span"):
            child_logger.info("child message")

        spans = exporter.get_finished_spans()
        log_events = [e for e in spans[0].events if e.name == "log"]
        # Should be exactly 1, not 2 (dedup flag prevents propagation duplicate)
        assert len(log_events) == 1
        assert log_events[0].attributes
        assert "child message" in str(log_events[0].attributes.get("log.message", ""))
