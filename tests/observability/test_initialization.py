"""Tests for observability initialization."""

import pytest
from pydantic import ValidationError

from ai_pipeline_core.observability._initialization import ObservabilityConfig


class TestObservabilityConfig:
    """Test ObservabilityConfig properties and defaults."""

    def test_defaults(self):
        config = ObservabilityConfig()
        assert config.has_clickhouse is False
        assert config.has_lmnr is False
        assert config.tracking_enabled is True

    def test_has_clickhouse(self):
        config = ObservabilityConfig(clickhouse_host="localhost")
        assert config.has_clickhouse is True

    def test_has_lmnr(self):
        config = ObservabilityConfig(lmnr_project_api_key="key123")
        assert config.has_lmnr is True

    def test_config_is_frozen(self):
        config = ObservabilityConfig()
        with pytest.raises(ValidationError):
            config.clickhouse_host = "changed"  # type: ignore[misc]
