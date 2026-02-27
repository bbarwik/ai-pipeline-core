"""Observability system initialization.

Provides ``initialize_observability()`` as the single entry point for
setting up Laminar and ClickHouse tracking.
"""

from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace as otel_trace
from pydantic import BaseModel, ConfigDict

from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._tracking._backend import ClickHouseBackend
from ai_pipeline_core.observability._tracking._processor import PipelineSpanProcessor
from ai_pipeline_core.observability._tracking._writer import ClickHouseWriter
from ai_pipeline_core.settings import settings

logger = get_pipeline_logger(__name__)


@dataclass
class _ObservabilityState:
    """Mutable module-level state for observability singletons."""

    clickhouse_backend: ClickHouseBackend | None = None
    processors: list[PipelineSpanProcessor] = field(default_factory=list)


_state = _ObservabilityState()


def get_clickhouse_backend() -> ClickHouseBackend | None:
    """Return the global ClickHouseBackend instance, or None if not initialized."""
    return _state.clickhouse_backend


def get_pipeline_processors() -> list[PipelineSpanProcessor]:
    """Return all registered PipelineSpanProcessor instances."""
    return _state.processors


def register_pipeline_processor(processor: PipelineSpanProcessor) -> None:
    """Register a PipelineSpanProcessor for run context propagation."""
    _state.processors.append(processor)


class ObservabilityConfig(BaseModel):
    """Configuration for the observability system."""

    model_config = ConfigDict(frozen=True)

    # Laminar
    lmnr_project_api_key: str = ""

    # ClickHouse tracking
    clickhouse_host: str = ""
    clickhouse_port: int = 8443
    clickhouse_database: str = "default"
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_secure: bool = True
    clickhouse_connect_timeout: int = 10
    clickhouse_send_receive_timeout: int = 30

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


def _setup_clickhouse(config: ObservabilityConfig) -> ClickHouseBackend | None:
    """Set up ClickHouse backend if configured.

    Creates the backend and writer but does NOT register a SpanProcessor.
    The processor is registered by the caller (CLI or Prefect worker) to
    avoid duplicate registration when CLI adds its own processor with
    both FilesystemBackend and ClickHouseBackend.
    """
    if not config.has_clickhouse or not config.tracking_enabled:
        return None

    writer = ClickHouseWriter(
        host=config.clickhouse_host,
        port=config.clickhouse_port,
        database=config.clickhouse_database,
        username=config.clickhouse_user,
        password=config.clickhouse_password,
        secure=config.clickhouse_secure,
        connect_timeout=config.clickhouse_connect_timeout,
        send_receive_timeout=config.clickhouse_send_receive_timeout,
    )
    writer.start()
    return ClickHouseBackend(writer)


def initialize_observability(config: ObservabilityConfig | None = None) -> None:
    """Initialize the full observability stack.

    Call once at pipeline startup. Safe to call multiple times (idempotent
    for Laminar). Reads from Settings if no config provided.

    For non-CLI paths (Prefect workers), registers a PipelineSpanProcessor
    with the ClickHouseBackend. CLI mode skips this and registers its own
    processor via _init_debug_tracing() to avoid duplicate registration.
    """
    if _state.clickhouse_backend is not None:
        return  # Already initialized

    if config is None:
        config = _build_config_from_settings()

    # 1. Laminar
    if config.has_lmnr:
        try:
            from ai_pipeline_core.observability import tracing  # noqa: PLC0415

            tracing._initialise_laminar()
            logger.info("Laminar initialized")
        except Exception as e:
            logger.warning("Laminar initialization failed: %s", e)

    # 2. ClickHouse tracking (backend only, no processor yet)
    _state.clickhouse_backend = _setup_clickhouse(config)
    if _state.clickhouse_backend is not None:
        logger.info("ClickHouse tracking backend initialized")

    # 3. Logging bridge — attached per-logger in get_pipeline_logger(), nothing to do here.


def ensure_tracking_processor() -> None:
    """Register a ClickHouse-only PipelineSpanProcessor if none exists yet.

    Called by non-CLI paths (Prefect workers) after initialize_observability().
    CLI mode registers its own processor with both backends via _init_debug_tracing(),
    so this is a no-op when a processor is already registered.
    """
    if _state.processors or _state.clickhouse_backend is None:
        return

    try:
        provider: Any = otel_trace.get_tracer_provider()
        if hasattr(provider, "add_span_processor"):
            processor = PipelineSpanProcessor(backends=(_state.clickhouse_backend,))
            provider.add_span_processor(processor)
            register_pipeline_processor(processor)
            logger.info("ClickHouse tracking processor registered")
    except Exception as e:
        logger.warning("Failed to register ClickHouse tracking processor: %s", e)
