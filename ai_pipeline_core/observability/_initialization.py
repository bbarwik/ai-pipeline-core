"""Observability system initialization.

Provides ``initialize_observability()`` as the single entry point for
setting up Laminar and ClickHouse tracking.
"""

from typing import Any

from opentelemetry import trace as otel_trace
from pydantic import BaseModel, ConfigDict

from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._tracking._client import ClickHouseClient
from ai_pipeline_core.observability._tracking._processor import TrackingSpanProcessor
from ai_pipeline_core.observability._tracking._service import TrackingService
from ai_pipeline_core.settings import settings

logger = get_pipeline_logger(__name__)


_tracking_service: TrackingService | None = None


def get_tracking_service() -> TrackingService | None:
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
    return ObservabilityConfig(**{field: getattr(settings, field) for field in ObservabilityConfig.model_fields})


def _setup_tracking(config: ObservabilityConfig) -> TrackingService | None:
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
    service = TrackingService(client)

    # Register span processor with OTel
    try:
        provider: Any = otel_trace.get_tracer_provider()
        if hasattr(provider, "add_span_processor"):
            processor = TrackingSpanProcessor(service)
            provider.add_span_processor(processor)
            logger.info("ClickHouse tracking initialized")
    except Exception as e:
        logger.warning("Failed to register TrackingSpanProcessor: %s", e)

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
            logger.warning("Laminar initialization failed: %s", e)

    # 2. ClickHouse tracking
    _tracking_service = _setup_tracking(config)

    # 3. Logging bridge — attached per-logger in get_pipeline_logger(), nothing to do here.
