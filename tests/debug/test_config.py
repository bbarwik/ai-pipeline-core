"""Tests for TraceDebugConfig."""

from pathlib import Path

import pytest

from ai_pipeline_core.debug import TraceDebugConfig


class TestTraceDebugConfig:
    """Tests for TraceDebugConfig."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default configuration values."""
        config = TraceDebugConfig(path=tmp_path)

        assert config.enabled is True
        assert config.max_file_bytes == 50_000
        assert config.max_element_bytes == 10_000
        assert config.element_excerpt_bytes == 2_000
        assert config.max_content_bytes == 10_000_000
        assert config.extract_base64_images is True
        assert config.merge_wrapper_spans is True
        assert config.events_file_mode == "errors_only"
        assert config.include_llm_index is True
        assert config.include_error_index is True
        assert config.max_traces is None  # Unlimited by default
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
            max_element_bytes=5000,
            max_traces=10,
            extract_base64_images=False,
            redact_patterns=("custom_pattern",),
        )

        assert config.max_element_bytes == 5000
        assert config.max_traces == 10
        assert config.extract_base64_images is False
        assert config.redact_patterns == ("custom_pattern",)

    def test_frozen_config(self, tmp_path: Path) -> None:
        """Test that config is immutable."""
        config = TraceDebugConfig(path=tmp_path)

        with pytest.raises(Exception):  # Pydantic raises ValidationError for frozen models
            config.max_element_bytes = 999
