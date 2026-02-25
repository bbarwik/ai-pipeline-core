"""Unit tests for observability initialization — _setup_tracking and initialize_observability."""

from unittest.mock import MagicMock, patch

import pytest

clickhouse_connect = pytest.importorskip("clickhouse_connect")

from ai_pipeline_core.observability._initialization import (
    ObservabilityConfig,
    _build_config_from_settings,
    _setup_tracking,
    initialize_observability,
)


class TestBuildConfigFromSettings:
    def test_builds_from_settings(self):
        with patch("ai_pipeline_core.observability._initialization.settings") as mock_settings:
            for field in ObservabilityConfig.model_fields:
                if field == "tracking_enabled":
                    setattr(mock_settings, field, True)
                elif field in ("clickhouse_port",):
                    setattr(mock_settings, field, 8443)
                elif field == "clickhouse_secure":
                    setattr(mock_settings, field, True)
                else:
                    setattr(mock_settings, field, "test_val")
            config = _build_config_from_settings()
            assert isinstance(config, ObservabilityConfig)


class TestSetupTracking:
    def test_returns_none_when_no_clickhouse(self):
        config = ObservabilityConfig(clickhouse_host="")
        result = _setup_tracking(config)
        assert result is None

    def test_returns_none_when_tracking_disabled(self):
        config = ObservabilityConfig(clickhouse_host="localhost", tracking_enabled=False)
        result = _setup_tracking(config)
        assert result is None

    def test_returns_service_when_configured(self):
        config = ObservabilityConfig(clickhouse_host="localhost")
        with (
            patch("ai_pipeline_core.observability._initialization.ClickHouseClient"),
            patch("ai_pipeline_core.observability._initialization.TrackingService") as mock_ts_cls,
            patch("ai_pipeline_core.observability._initialization.otel_trace") as mock_otel,
        ):
            mock_provider = MagicMock()
            mock_otel.get_tracer_provider.return_value = mock_provider
            mock_service = MagicMock()
            mock_ts_cls.return_value = mock_service
            result = _setup_tracking(config)
        assert result is mock_service
        mock_provider.add_span_processor.assert_called_once()

    def test_warning_on_processor_failure(self):
        config = ObservabilityConfig(clickhouse_host="localhost")
        with (
            patch("ai_pipeline_core.observability._initialization.ClickHouseClient"),
            patch("ai_pipeline_core.observability._initialization.TrackingService") as mock_ts_cls,
            patch("ai_pipeline_core.observability._initialization.otel_trace") as mock_otel,
        ):
            mock_provider = MagicMock()
            mock_provider.add_span_processor.side_effect = RuntimeError("bad")
            mock_otel.get_tracer_provider.return_value = mock_provider
            mock_service = MagicMock()
            mock_ts_cls.return_value = mock_service
            result = _setup_tracking(config)
        assert result is mock_service


class TestInitializeObservability:
    def test_idempotent(self):
        import ai_pipeline_core.observability._initialization as mod

        original = mod._tracking_service
        try:
            mod._tracking_service = MagicMock()
            initialize_observability(ObservabilityConfig())
            # Should return early — _tracking_service unchanged
            assert mod._tracking_service is not None
        finally:
            mod._tracking_service = original

    def test_with_lmnr(self):
        import ai_pipeline_core.observability._initialization as mod

        original = mod._tracking_service
        try:
            mod._tracking_service = None
            config = ObservabilityConfig(lmnr_project_api_key="test_key")
            with (
                patch("ai_pipeline_core.observability._initialization._setup_tracking", return_value=None),
                patch("ai_pipeline_core.observability.tracing._initialise_laminar") as mock_lmnr,
            ):
                initialize_observability(config)
            mock_lmnr.assert_called_once()
        finally:
            mod._tracking_service = original

    def test_lmnr_failure_swallowed(self):
        import ai_pipeline_core.observability._initialization as mod

        original = mod._tracking_service
        try:
            mod._tracking_service = None
            config = ObservabilityConfig(lmnr_project_api_key="test_key")
            with (
                patch("ai_pipeline_core.observability._initialization._setup_tracking", return_value=None),
                patch("ai_pipeline_core.observability.tracing._initialise_laminar", side_effect=RuntimeError("fail")),
            ):
                initialize_observability(config)
        finally:
            mod._tracking_service = original

    def test_builds_config_from_settings_when_none(self):
        import ai_pipeline_core.observability._initialization as mod

        original = mod._tracking_service
        try:
            mod._tracking_service = None
            with (
                patch("ai_pipeline_core.observability._initialization._build_config_from_settings") as mock_build,
                patch("ai_pipeline_core.observability._initialization._setup_tracking", return_value=None),
            ):
                mock_build.return_value = ObservabilityConfig()
                initialize_observability(None)
            mock_build.assert_called_once()
        finally:
            mod._tracking_service = original
