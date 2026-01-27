"""Tests for logging mixin classes."""

from ai_pipeline_core.logging.logging_mixin import LoggerMixin, StructuredLoggerMixin


class SampleClass(LoggerMixin):
    """Test class using LoggerMixin."""


class StructuredSample(StructuredLoggerMixin):
    """Test class using StructuredLoggerMixin."""


class TestLoggerMixin:
    """Test LoggerMixin convenience methods."""

    def test_logger_creation(self):
        """Test logger property returns a logger."""
        obj = SampleClass()
        assert obj.logger is not None

    def test_log_debug(self):
        """Test log_debug method."""
        obj = SampleClass()
        obj.log_debug("debug message")

    def test_log_info(self):
        """Test log_info method."""
        obj = SampleClass()
        obj.log_info("info message")

    def test_log_warning(self):
        """Test log_warning method."""
        obj = SampleClass()
        obj.log_warning("warning message")

    def test_log_error(self):
        """Test log_error method."""
        obj = SampleClass()
        obj.log_error("error message")

    def test_log_critical(self):
        """Test log_critical method."""
        obj = SampleClass()
        obj.log_critical("critical message")

    def test_log_with_context(self):
        """Test log_with_context method."""
        obj = SampleClass()
        obj.log_with_context("INFO", "test message", {"key": "value"})

    def test_log_error_with_exc_info(self):
        """Test log_error with exc_info=True."""
        obj = SampleClass()
        obj.log_error("error with traceback", exc_info=True)


class TestStructuredLoggerMixin:
    """Test StructuredLoggerMixin convenience methods."""

    def test_log_event(self):
        """Test log_event method."""
        obj = StructuredSample()
        obj.log_event("document_processed", status="success")

    def test_log_metric(self):
        """Test log_metric method."""
        obj = StructuredSample()
        obj.log_metric("processing_time", 1.23, "seconds")

    def test_log_span(self):
        """Test log_span method."""
        obj = StructuredSample()
        obj.log_span("llm_call", 500.0, model="gpt-4")
