"""Tests for TraceDebugConfig."""

from pathlib import Path

import pytest

from ai_pipeline_core.observability._debug import TraceDebugConfig


class TestTraceDebugConfig:
    """Tests for TraceDebugConfig."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default configuration values."""
        config = TraceDebugConfig(path=tmp_path)

        assert config.max_file_bytes == 500_000
        assert config.verbose is False
        assert config.merge_wrapper_spans is True
        assert config.generate_summary is True

    def test_redact_patterns_default(self, tmp_path: Path) -> None:
        """Test default redaction patterns are set."""
        config = TraceDebugConfig(path=tmp_path)

        # Should have patterns for common secrets
        assert len(config.redact_patterns) > 0
        # Check for OpenAI pattern
        assert any("sk-" in p for p in config.redact_patterns)
        # Check for AWS pattern
        assert any("AKIA" in p for p in config.redact_patterns)

    def test_custom_values(self, tmp_path: Path) -> None:
        """Test custom configuration values."""
        config = TraceDebugConfig(
            path=tmp_path,
            verbose=True,
            redact_patterns=("custom_pattern",),
        )

        assert config.verbose is True
        assert config.redact_patterns == ("custom_pattern",)

    def test_frozen_config(self, tmp_path: Path) -> None:
        """Test that config is immutable."""
        config = TraceDebugConfig(path=tmp_path)

        with pytest.raises(Exception):  # Pydantic raises ValidationError for frozen models
            config.verbose = True  # type: ignore[misc]
