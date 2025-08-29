"""Logging mixin for consistent logging across components using Prefect logging.

@public
"""

import contextlib
import time
from contextlib import contextmanager
from functools import cached_property
from typing import Any, Dict, Generator, Optional

from prefect import get_run_logger
from prefect.context import FlowRunContext, TaskRunContext
from prefect.logging import get_logger


class LoggerMixin:
    """Mixin class that provides consistent logging functionality using Prefect's logging system.

    @public

    Note for users: In your code, always obtain loggers via get_pipeline_logger(__name__).
    The mixin's internal behavior routes to the appropriate backend; you should not call
    logging.getLogger directly.

    Automatically uses appropriate logger based on context:
    - prefect.get_run_logger() when in flow/task context
    - Internal routing when outside flow/task context
    """

    _logger_name: Optional[str] = None

    @cached_property
    def logger(self):
        """Get appropriate logger based on context."""
        if logger := self._get_run_logger():
            return logger
        return get_logger(self._logger_name or self.__class__.__module__)

    def _get_run_logger(self):
        """Attempt to get Prefect run logger.

        Returns:
            The Prefect run logger if in a flow/task context, None otherwise.
        """
        # Intentionally broad: Must handle any exception when checking context
        with contextlib.suppress(Exception):
            if FlowRunContext.get() or TaskRunContext.get():
                return get_run_logger()
        return None

    def log_debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with optional context."""
        self.logger.debug(message, extra=kwargs)

    def log_info(self, message: str, **kwargs: Any) -> None:
        """Log info message with optional context."""
        self.logger.info(message, extra=kwargs)

    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with optional context."""
        self.logger.warning(message, extra=kwargs)

    def log_error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log error message with optional exception info."""
        self.logger.error(message, exc_info=exc_info, extra=kwargs)

    def log_critical(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log critical message with optional exception info."""
        self.logger.critical(message, exc_info=exc_info, extra=kwargs)

    def log_with_context(self, level: str, message: str, context: Dict[str, Any]) -> None:
        """Log message with structured context.

        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message
            context: Additional context as dictionary

        Example:
            self.log_with_context("info", "Processing document", {
                "document_id": doc.id,
                "document_size": doc.size,
                "document_type": doc.type
            })
        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)

        # Format context for logging
        context_str = " | ".join(f"{k}={v}" for k, v in context.items())
        full_message = f"{message} | {context_str}" if context else message

        log_method(full_message, extra={"context": context})


class StructuredLoggerMixin(LoggerMixin):
    """Extended mixin for structured logging with Prefect.

    @public
    """

    def log_event(self, event: str, **kwargs: Any) -> None:
        """Log a structured event.

        Args:
            event: Event name
            **kwargs: Event attributes

        Example:
            self.log_event("document_processed",
                          document_id=doc.id,
                          duration_ms=processing_time,
                          status="success")
        """
        self.logger.info(event, extra={"event": event, "structured": True, **kwargs})

    def log_metric(self, metric_name: str, value: float, unit: str = "", **tags: Any) -> None:
        """Log a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            **tags: Additional tags

        Example:
            self.log_metric("processing_time", 1.23, "seconds",
                          document_type="pdf", model="gpt-4")
        """
        self.logger.info(
            f"Metric: {metric_name}",
            extra={
                "metric": metric_name,
                "value": value,
                "unit": unit,
                "tags": tags,
                "structured": True,
            },
        )

    def log_span(self, operation: str, duration_ms: float, **attributes: Any) -> None:
        """Log a span (operation with duration).

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            **attributes: Additional attributes

        Example:
            self.log_span("llm_generation", 1234.5,
                         model="gpt-4", tokens=500)
        """
        self.logger.info(
            f"Span: {operation}",
            extra={
                "span": operation,
                "duration_ms": duration_ms,
                "attributes": attributes,
                "structured": True,
            },
        )

    @contextmanager
    def log_operation(self, operation: str, **context: Any) -> Generator[None, None, None]:
        """Context manager for logging operations with timing.

        Args:
            operation: Operation name
            **context: Additional context

        Example:
            with self.log_operation("document_processing", doc_id=doc.id):
                process_document(doc)
        """
        start_time = time.perf_counter()

        self.log_debug(f"Starting {operation}", **context)

        try:
            yield
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_info(
                f"Completed {operation}", duration_ms=duration_ms, status="success", **context
            )
        except Exception as e:
            # Intentionally broad: Context manager must catch all exceptions to log them
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_error(
                f"Failed {operation}: {str(e)}",
                exc_info=True,
                duration_ms=duration_ms,
                status="failure",
                **context,
            )
            raise


class PrefectLoggerMixin(StructuredLoggerMixin):
    """Enhanced mixin specifically for Prefect flows and tasks."""

    def log_flow_start(self, flow_name: str, parameters: Dict[str, Any]) -> None:
        """Log flow start with parameters."""
        self.log_event("flow_started", flow_name=flow_name, parameters=parameters)

    def log_flow_end(self, flow_name: str, status: str, duration_ms: float) -> None:
        """Log flow completion."""
        self.log_event(
            "flow_completed", flow_name=flow_name, status=status, duration_ms=duration_ms
        )

    def log_task_start(self, task_name: str, inputs: Dict[str, Any]) -> None:
        """Log task start with inputs."""
        self.log_event("task_started", task_name=task_name, inputs=inputs)

    def log_task_end(self, task_name: str, status: str, duration_ms: float) -> None:
        """Log task completion."""
        self.log_event(
            "task_completed", task_name=task_name, status=status, duration_ms=duration_ms
        )

    def log_retry(self, operation: str, attempt: int, max_attempts: int, error: str) -> None:
        """Log retry attempt."""
        self.log_warning(
            f"Retrying {operation}", attempt=attempt, max_attempts=max_attempts, error=error
        )

    def log_checkpoint(self, checkpoint_name: str, **data: Any) -> None:
        """Log a checkpoint in processing."""
        self.log_info(f"Checkpoint: {checkpoint_name}", checkpoint=checkpoint_name, **data)
