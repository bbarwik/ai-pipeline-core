"""Unit tests for observability initialization — _setup_clickhouse and initialize_observability."""

from unittest.mock import MagicMock, patch

import pytest

clickhouse_connect = pytest.importorskip("clickhouse_connect")

from ai_pipeline_core.observability._initialization import (
    ObservabilityConfig,
    _build_config_from_settings,
    _setup_clickhouse,
    ensure_tracking_processor,
    get_pipeline_processors,
    initialize_observability,
    register_pipeline_processor,
)


class TestBuildConfigFromSettings:
    def test_builds_from_settings(self):
        with patch("ai_pipeline_core.observability._initialization.settings") as mock_settings:
            for field in ObservabilityConfig.model_fields:
                if field == "tracking_enabled":
                    setattr(mock_settings, field, True)
                elif field in ("clickhouse_port", "clickhouse_connect_timeout", "clickhouse_send_receive_timeout"):
                    setattr(mock_settings, field, 8443)
                elif field == "clickhouse_secure":
                    setattr(mock_settings, field, True)
                else:
                    setattr(mock_settings, field, "test_val")
            config = _build_config_from_settings()
            assert isinstance(config, ObservabilityConfig)


class TestSetupClickhouse:
    def test_returns_none_when_no_clickhouse(self):
        config = ObservabilityConfig(clickhouse_host="")
        result = _setup_clickhouse(config)
        assert result is None

    def test_returns_none_when_tracking_disabled(self):
        config = ObservabilityConfig(clickhouse_host="localhost", tracking_enabled=False)
        result = _setup_clickhouse(config)
        assert result is None

    def test_returns_backend_when_configured(self):
        config = ObservabilityConfig(clickhouse_host="localhost")
        with patch("ai_pipeline_core.observability._initialization.ClickHouseWriter") as mock_writer_cls:
            mock_writer = MagicMock()
            mock_writer_cls.return_value = mock_writer
            result = _setup_clickhouse(config)
        assert result is not None
        mock_writer.start.assert_called_once()


class TestInitializeObservability:
    def test_idempotent(self):
        import ai_pipeline_core.observability._initialization as mod

        original = mod._state.clickhouse_backend
        try:
            mod._state.clickhouse_backend = MagicMock()
            initialize_observability(ObservabilityConfig())
            # Should return early
            assert mod._state.clickhouse_backend is not None
        finally:
            mod._state.clickhouse_backend = original

    def test_with_lmnr(self):
        import ai_pipeline_core.observability._initialization as mod

        original = mod._state.clickhouse_backend
        try:
            mod._state.clickhouse_backend = None
            config = ObservabilityConfig(lmnr_project_api_key="test_key")
            with (
                patch("ai_pipeline_core.observability._initialization._setup_clickhouse", return_value=None),
                patch("ai_pipeline_core.observability.tracing._initialise_laminar") as mock_lmnr,
            ):
                initialize_observability(config)
            mock_lmnr.assert_called_once()
        finally:
            mod._state.clickhouse_backend = original

    def test_lmnr_failure_swallowed(self):
        import ai_pipeline_core.observability._initialization as mod

        original = mod._state.clickhouse_backend
        try:
            mod._state.clickhouse_backend = None
            config = ObservabilityConfig(lmnr_project_api_key="test_key")
            with (
                patch("ai_pipeline_core.observability._initialization._setup_clickhouse", return_value=None),
                patch("ai_pipeline_core.observability.tracing._initialise_laminar", side_effect=RuntimeError("fail")),
            ):
                initialize_observability(config)
        finally:
            mod._state.clickhouse_backend = original

    def test_builds_config_from_settings_when_none(self):
        import ai_pipeline_core.observability._initialization as mod

        original = mod._state.clickhouse_backend
        try:
            mod._state.clickhouse_backend = None
            with (
                patch("ai_pipeline_core.observability._initialization._build_config_from_settings") as mock_build,
                patch("ai_pipeline_core.observability._initialization._setup_clickhouse", return_value=None),
            ):
                mock_build.return_value = ObservabilityConfig()
                initialize_observability(None)
            mock_build.assert_called_once()
        finally:
            mod._state.clickhouse_backend = original


# ---------------------------------------------------------------------------
# ensure_tracking_processor
# ---------------------------------------------------------------------------


class TestEnsureTrackingProcessor:
    def test_registers_when_empty(self):
        import ai_pipeline_core.observability._initialization as mod

        original_backend = mod._state.clickhouse_backend
        original_processors = mod._state.processors[:]
        try:
            mock_backend = MagicMock()
            mod._state.clickhouse_backend = mock_backend
            mod._state.processors.clear()
            mock_provider = MagicMock()
            with patch("ai_pipeline_core.observability._initialization.otel_trace") as mock_otel:
                mock_otel.get_tracer_provider.return_value = mock_provider
                ensure_tracking_processor()
            mock_provider.add_span_processor.assert_called_once()
            assert len(mod._state.processors) == 1
        finally:
            mod._state.clickhouse_backend = original_backend
            mod._state.processors.clear()
            mod._state.processors.extend(original_processors)

    def test_noop_when_processor_exists(self):
        import ai_pipeline_core.observability._initialization as mod

        original_backend = mod._state.clickhouse_backend
        original_processors = mod._state.processors[:]
        try:
            mod._state.clickhouse_backend = MagicMock()
            existing = MagicMock()
            mod._state.processors.clear()
            mod._state.processors.append(existing)
            mock_provider = MagicMock()
            with patch("ai_pipeline_core.observability._initialization.otel_trace") as mock_otel:
                mock_otel.get_tracer_provider.return_value = mock_provider
                ensure_tracking_processor()
            mock_provider.add_span_processor.assert_not_called()
            assert len(mod._state.processors) == 1
        finally:
            mod._state.clickhouse_backend = original_backend
            mod._state.processors.clear()
            mod._state.processors.extend(original_processors)

    def test_noop_when_no_backend(self):
        import ai_pipeline_core.observability._initialization as mod

        original_backend = mod._state.clickhouse_backend
        original_processors = mod._state.processors[:]
        try:
            mod._state.clickhouse_backend = None
            mod._state.processors.clear()
            ensure_tracking_processor()
            assert len(mod._state.processors) == 0
        finally:
            mod._state.clickhouse_backend = original_backend
            mod._state.processors.clear()
            mod._state.processors.extend(original_processors)


# ---------------------------------------------------------------------------
# register_pipeline_processor
# ---------------------------------------------------------------------------


class TestRegisterPipelineProcessor:
    def test_appends_multiple(self):
        import ai_pipeline_core.observability._initialization as mod

        original_processors = mod._state.processors[:]
        try:
            mod._state.processors.clear()
            p1 = MagicMock()
            p2 = MagicMock()
            register_pipeline_processor(p1)
            register_pipeline_processor(p2)
            assert len(get_pipeline_processors()) == 2
            assert get_pipeline_processors()[0] is p1
            assert get_pipeline_processors()[1] is p2
        finally:
            mod._state.processors.clear()
            mod._state.processors.extend(original_processors)


# ---------------------------------------------------------------------------
# _setup_clickhouse does NOT register processor
# ---------------------------------------------------------------------------


class TestSetupClickhouseNoProcessor:
    def test_does_not_register_processor(self):
        import ai_pipeline_core.observability._initialization as mod

        original_processors = mod._state.processors[:]
        try:
            mod._state.processors.clear()
            config = ObservabilityConfig(clickhouse_host="localhost")
            with patch("ai_pipeline_core.observability._initialization.ClickHouseWriter") as mock_writer_cls:
                mock_writer_cls.return_value = MagicMock()
                _setup_clickhouse(config)
            assert len(mod._state.processors) == 0
        finally:
            mod._state.processors.clear()
            mod._state.processors.extend(original_processors)
