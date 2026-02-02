"""Tests for logging mixin classes."""

import pytest

from ai_pipeline_core.logging.logging_mixin import (
    LoggerMixin,
    PrefectLoggerMixin,
    StructuredLoggerMixin,
)


class SampleClass(LoggerMixin):
    """Test class using LoggerMixin."""


class StructuredSample(StructuredLoggerMixin):
    """Test class using StructuredLoggerMixin."""


class PrefectSample(PrefectLoggerMixin):
    """Test class using PrefectLoggerMixin."""


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
        obj.log_span("llm_call", 500.0, model="gpt-5.1")

    def test_log_operation_success(self):
        """Test log_operation context manager on success."""
        obj = StructuredSample()
        with obj.log_operation("test_op", doc_id="123"):
            pass  # Simulates successful operation

    def test_log_operation_failure(self):
        """Test log_operation context manager on exception."""
        obj = StructuredSample()
        with pytest.raises(ValueError, match="test error"), obj.log_operation("failing_op", doc_id="456"):
            raise ValueError("test error")


class TestPrefectLoggerMixin:
    """Test PrefectLoggerMixin convenience methods."""

    def test_log_flow_start(self):
        """Test log_flow_start method."""
        obj = PrefectSample()
        obj.log_flow_start("my_flow", {"param1": "value1"})

    def test_log_flow_end(self):
        """Test log_flow_end method."""
        obj = PrefectSample()
        obj.log_flow_end("my_flow", "success", 1234.5)

    def test_log_task_start(self):
        """Test log_task_start method."""
        obj = PrefectSample()
        obj.log_task_start("my_task", {"input1": "data"})

    def test_log_task_end(self):
        """Test log_task_end method."""
        obj = PrefectSample()
        obj.log_task_end("my_task", "success", 567.8)

    def test_log_retry(self):
        """Test log_retry method."""
        obj = PrefectSample()
        obj.log_retry("llm_call", attempt=2, max_attempts=3, error="timeout")

    def test_log_checkpoint(self):
        """Test log_checkpoint method."""
        obj = PrefectSample()
        obj.log_checkpoint("step_1_complete", items_processed=50)
