"""Tests for logging configuration."""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from ai_pipeline_core.logging import get_pipeline_logger, setup_logging
from ai_pipeline_core.logging.logging_config import LoggingConfig


class TestLoggingConfig:
    """Test LoggingConfig class."""

    def test_default_config_path_from_env(self):
        """Test getting config path from environment."""
        with patch.dict(os.environ, {"AI_PIPELINE_LOGGING_CONFIG": "/path/to/config.yml"}):
            config = LoggingConfig()
            assert config.config_path == Path("/path/to/config.yml")

    def test_default_config_path_from_prefect_env(self):
        """Test getting config path from Prefect environment."""
        with patch.dict(os.environ, {"PREFECT_LOGGING_SETTINGS_PATH": "/prefect/config.yml"}):
            config = LoggingConfig()
            assert config.config_path == Path("/prefect/config.yml")

    def test_no_config_path_returns_none(self):
        """Test that no env vars results in None config path."""
        with patch.dict(os.environ, clear=True):
            config = LoggingConfig()
            assert config.config_path is None

    def test_load_config_from_file(self, tmp_path: Path) -> None:
        """Test loading config from YAML file."""
        config_file = tmp_path / "logging.yml"
        config_file.write_text("""
version: 1
disable_existing_loggers: false
handlers:
  console:
    class: logging.StreamHandler
""")

        config = LoggingConfig(config_path=config_file)
        loaded = config.load_config()

        assert loaded["version"] == 1
        assert loaded["disable_existing_loggers"] is False
        assert "console" in loaded["handlers"]

    def test_load_default_config_when_no_file(self):
        """Test loading default config when no file exists."""
        config = LoggingConfig()
        loaded = config.load_config()

        assert loaded["version"] == 1
        assert "formatters" in loaded
        assert "handlers" in loaded
        assert "loggers" in loaded

    @patch("logging.config.dictConfig")
    def test_apply_config(self, mock_dict_config: Mock) -> None:
        """Test applying logging configuration."""
        config = LoggingConfig()
        config.apply()

        mock_dict_config.assert_called_once()
        call_args = mock_dict_config.call_args[0][0]
        assert call_args["version"] == 1

    @patch("logging.config.dictConfig")
    def test_apply_with_prefect_settings(self, mock_dict_config: Mock) -> None:
        """Test applying config with Prefect settings."""
        with patch.dict(os.environ, clear=True):
            # Use patch to inject the config during initialization
            custom_config = {"version": 1, "loggers": {"prefect": {"level": "DEBUG"}}}
            with patch.object(LoggingConfig, "load_config", return_value=custom_config):
                config = LoggingConfig()
                config.apply()

                # Should set Prefect env var
                assert os.environ.get("PREFECT_LOGGING_LEVEL") == "DEBUG"


class TestSetupLogging:
    """Test setup_logging function."""

    @patch("ai_pipeline_core.logging.logging_config.LoggingConfig.apply")
    def test_setup_logging_basic(self, mock_apply: Mock) -> None:
        """Test basic setup_logging call."""
        setup_logging()
        mock_apply.assert_called_once()

    @patch("ai_pipeline_core.logging.logging_config.get_logger")
    @patch("ai_pipeline_core.logging.logging_config.LoggingConfig.apply")
    def test_setup_logging_with_level(self, mock_apply: Mock, mock_get_logger: Mock) -> None:
        """Test setup_logging with custom level."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        setup_logging(level="DEBUG")

        # Should set level on loggers
        assert mock_get_logger.call_count > 0
        mock_logger.setLevel.assert_called_with("DEBUG")

        # Should set Prefect env
        assert os.environ["PREFECT_LOGGING_LEVEL"] == "DEBUG"

    @patch("ai_pipeline_core.logging.logging_config.LoggingConfig")
    def test_setup_logging_with_config_path(self, mock_config_class: Mock, tmp_path: Path) -> None:
        """Test setup_logging with custom config path."""
        config_file = tmp_path / "custom.yml"
        mock_instance = MagicMock()
        mock_config_class.return_value = mock_instance

        setup_logging(config_path=config_file)

        mock_config_class.assert_called_once_with(config_file)
        mock_instance.apply.assert_called_once()


class TestGetPipelineLogger:
    """Test get_pipeline_logger function."""

    @patch("ai_pipeline_core.logging.logging_config.setup_logging")
    @patch("ai_pipeline_core.logging.logging_config.get_logger")
    def test_get_pipeline_logger_ensures_setup(self, mock_get_logger: Mock, mock_setup: Mock) -> None:
        """Test that get_pipeline_logger ensures logging is setup."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Reset global state
        import ai_pipeline_core.logging.logging_config

        ai_pipeline_core.logging.logging_config._logging_config = None  # type: ignore[attr-defined]

        logger = get_pipeline_logger("test.module")

        mock_setup.assert_called_once()
        mock_get_logger.assert_called_with("test.module")
        assert logger == mock_logger

    @patch("ai_pipeline_core.logging.logging_config.get_logger")
    def test_get_pipeline_logger_reuses_config(self, mock_get_logger: Mock) -> None:
        """Test that subsequent calls don't re-setup logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Simulate already configured
        import ai_pipeline_core.logging.logging_config

        ai_pipeline_core.logging.logging_config._logging_config = MagicMock()  # type: ignore[attr-defined]

        with patch("ai_pipeline_core.logging.logging_config.setup_logging") as mock_setup:
            get_pipeline_logger("module1")
            get_pipeline_logger("module2")

            # Should not call setup_logging
            mock_setup.assert_not_called()

            assert mock_get_logger.call_count == 2
