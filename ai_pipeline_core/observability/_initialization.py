"""Observability system initialization.

Provides ``initialize_observability()`` as the single entry point for
setting up Laminar and ClickHouse tracking.
"""

import importlib
from typing import Any, Protocol
from uuid import UUID

from opentelemetry import trace as otel_trace
from pydantic import BaseModel, ConfigDict

from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._tracking._client import ClickHouseClient
from ai_pipeline_core.observability._tracking._models import DocumentEventType, RunStatus
from ai_pipeline_core.observability._tracking._processor import TrackingSpanProcessor
from ai_pipeline_core.observability._tracking._service import TrackingService
from ai_pipeline_core.settings import settings

logger = get_pipeline_logger(__name__)


class TrackingServiceProtocol(Protocol):
    """Protocol for the tracking service methods used by deployment, decorators, and document tracking."""

    # Run lifecycle
    def set_run_context(self, *, run_id: UUID, project_name: str, flow_name: str, run_scope: str = "") -> None:
        """Store run metadata in context vars for downstream span attribution."""
        ...

    def track_run_start(self, *, run_id: UUID, project_name: str, flow_name: str, run_scope: str = "") -> None:
        """Record a pipeline run start event to ClickHouse."""
        ...

    def track_run_end(
        self,
        *,
        run_id: UUID,
        status: RunStatus,
        total_cost: float = ...,
        total_tokens: int = ...,
        metadata: dict[str, object] | None = ...,
    ) -> None:
        """Record a pipeline run completion event with final metrics."""
        ...

    def clear_run_context(self) -> None:
        """Reset run-scoped context vars after a run finishes."""
        ...

    # Document tracking
    def track_document_event(
        self,
        *,
        document_sha256: str,
        span_id: str,
        event_type: DocumentEventType,
        metadata: dict[str, str] | None = ...,
    ) -> None:
        """Record a document lifecycle event (created, read, transformed)."""
        ...

    # Summaries
    def schedule_summary(self, span_id: str, label: str, output_hint: str) -> None:
        """Queue an LLM-generated summary for a span's output."""
        ...

    # Lifecycle
    def flush(self, timeout: float = 30.0) -> None:
        """Flush all pending tracking events to ClickHouse."""
        ...

    def shutdown(self, timeout: float = 30.0) -> None:
        """Flush pending events and release tracking resources."""
        ...


_tracking_service: TrackingServiceProtocol | None = None


def get_tracking_service() -> TrackingServiceProtocol | None:
    """Return the global TrackingService instance, or None if not initialized."""
    return _tracking_service


class ObservabilityConfig(BaseModel):
    """Configuration for the observability system."""

    model_config = ConfigDict(frozen=True)

    # Laminar
    lmnr_project_api_key: str = ""
    lmnr_debug: str = ""

    # ClickHouse tracking
    clickhouse_host: str = ""
    clickhouse_port: int = 8443
    clickhouse_database: str = "default"
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_secure: bool = True

    # Tracking behavior
    tracking_enabled: bool = True
    tracking_summary_model: str = "gemini-3-flash"

    @property
    def has_clickhouse(self) -> bool:
        """Whether ClickHouse is configured."""
        return bool(self.clickhouse_host)

    @property
    def has_lmnr(self) -> bool:
        """Whether Laminar is configured."""
        return bool(self.lmnr_project_api_key)


def _build_config_from_settings() -> ObservabilityConfig:
    """Build ObservabilityConfig from framework Settings."""
    return ObservabilityConfig(
        lmnr_project_api_key=getattr(settings, "lmnr_project_api_key", ""),
        lmnr_debug=getattr(settings, "lmnr_debug", ""),
        clickhouse_host=getattr(settings, "clickhouse_host", ""),
        clickhouse_port=getattr(settings, "clickhouse_port", 8443),
        clickhouse_database=getattr(settings, "clickhouse_database", "default"),
        clickhouse_user=getattr(settings, "clickhouse_user", "default"),
        clickhouse_password=getattr(settings, "clickhouse_password", ""),
        clickhouse_secure=getattr(settings, "clickhouse_secure", True),
        tracking_enabled=getattr(settings, "tracking_enabled", True),
        tracking_summary_model=getattr(settings, "tracking_summary_model", "gemini-3-flash"),
    )


def _setup_tracking(config: ObservabilityConfig) -> TrackingServiceProtocol | None:
    """Set up ClickHouse tracking if configured. Returns TrackingService or None."""
    if not config.has_clickhouse or not config.tracking_enabled:
        return None

    client = ClickHouseClient(
        host=config.clickhouse_host,
        port=config.clickhouse_port,
        database=config.clickhouse_database,
        username=config.clickhouse_user,
        password=config.clickhouse_password,
        secure=config.clickhouse_secure,
    )
    summary_mod = importlib.import_module("ai_pipeline_core.observability._summary")
    service = TrackingService(
        client,
        summary_model=config.tracking_summary_model,
        span_summary_fn=summary_mod.generate_span_summary,
    )

    # Register span processor with OTel
    try:
        provider: Any = otel_trace.get_tracer_provider()
        if hasattr(provider, "add_span_processor"):
            processor = TrackingSpanProcessor(service)
            provider.add_span_processor(processor)
            logger.info("ClickHouse tracking initialized")
    except Exception as e:
        logger.warning(f"Failed to register TrackingSpanProcessor: {e}")

    return service


def initialize_observability(config: ObservabilityConfig | None = None) -> None:
    """Initialize the full observability stack.

    Call once at pipeline startup. Safe to call multiple times (idempotent
    for Laminar). Reads from Settings if no config provided.
    """
    global _tracking_service  # noqa: PLW0603

    if _tracking_service is not None:
        return  # Already initialized

    if config is None:
        config = _build_config_from_settings()

    # 1. Laminar - use canonical initializer from tracing module
    if config.has_lmnr:
        try:
            from ai_pipeline_core.observability import tracing  # noqa: PLC0415

            tracing._initialise_laminar()
            logger.info("Laminar initialized")
        except Exception as e:
            logger.warning(f"Laminar initialization failed: {e}")

    # 2. ClickHouse tracking
    _tracking_service = _setup_tracking(config)

    # 3. Logging bridge — attached per-logger in get_pipeline_logger(), nothing to do here.
