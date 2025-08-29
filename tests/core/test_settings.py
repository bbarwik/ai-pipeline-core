"""Tests for Settings."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from ai_pipeline_core.settings import Settings, settings


class TestSettings:
    """Test Settings configuration."""

    def test_default_values(self):
        """Test default Settings values when env vars/files are not set."""
        # This test can only work if there's no .env file or env vars
        # In practice, we can test that Settings loads correctly
        s = Settings()

        # If values are loaded from .env, skip the test
        if s.openai_base_url or s.openai_api_key:
            pytest.skip("Settings loaded from .env file or environment")

        assert s.openai_base_url == ""
        assert s.openai_api_key == ""
        assert s.prefect_api_url == ""
        assert s.prefect_api_key == ""
        assert s.lmnr_project_api_key == ""

    @patch.dict(
        os.environ,
        {
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "OPENAI_API_KEY": "sk-test123",
            "PREFECT_API_URL": "https://api.prefect.io",
            "PREFECT_API_KEY": "pf-key456",
            "LMNR_PROJECT_API_KEY": "lmnr-key789",
        },
    )
    def test_env_variable_loading(self):
        """Test loading settings from environment variables."""
        s = Settings()
        assert s.openai_base_url == "https://api.openai.com/v1"
        assert s.openai_api_key == "sk-test123"
        assert s.prefect_api_url == "https://api.prefect.io"
        assert s.prefect_api_key == "pf-key456"
        assert s.lmnr_project_api_key == "lmnr-key789"

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-key",
            "UNKNOWN_SETTING": "should-be-ignored",
            "RANDOM_VAR": "also-ignored",
        },
    )
    def test_extra_env_ignored(self):
        """Test that unknown environment variables are ignored."""
        # Should not raise even with unknown env vars (extra="ignore")
        s = Settings()
        assert s.openai_api_key == "test-key"
        # Unknown vars are not added as attributes
        assert not hasattr(s, "unknown_setting")
        assert not hasattr(s, "random_var")

    def test_settings_singleton(self):
        """Test that the module provides a settings singleton."""
        # The module exports a pre-created instance
        assert isinstance(settings, Settings)

        # It should be the same instance
        from ai_pipeline_core.settings import settings as settings2

        assert settings is settings2

    def test_env_file_loading(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading from .env file."""
        # Create a temporary .env file
        env_file = tmp_path / ".env"
        env_file.write_text("""
OPENAI_API_KEY=from-env-file
PREFECT_API_URL=http://localhost:4200
LMNR_PROJECT_API_KEY=lmnr-from-file
""")

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create new Settings instance (will look for .env in current dir)
        s = Settings()

        assert s.openai_api_key == "from-env-file"
        assert s.prefect_api_url == "http://localhost:4200"
        assert s.lmnr_project_api_key == "lmnr-from-file"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "from-env-var"})
    def test_env_var_overrides_env_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that environment variables override .env file."""
        # Create .env file
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=from-env-file")

        monkeypatch.chdir(tmp_path)

        s = Settings()

        # Environment variable should win
        assert s.openai_api_key == "from-env-var"

    def test_partial_configuration(self):
        """Test that partial configuration works."""
        # Only some settings provided
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            # Note: Settings might still load from .env file
            s = Settings()

            # Check if .env file is providing base_url
            if s.openai_base_url and s.openai_base_url != "":
                pytest.skip("Settings loaded OPENAI_BASE_URL from .env file")

            assert s.openai_api_key == "test-key"
            assert s.openai_base_url == ""  # Default
            assert s.prefect_api_url == ""  # Default

    def test_settings_immutable_config(self):
        """Test that Settings uses proper Pydantic configuration."""
        s = Settings()

        # Settings should be immutable (frozen=True)
        with pytest.raises(ValidationError) as exc_info:
            s.openai_api_key = "new-key"
        assert "frozen" in str(exc_info.value).lower()

    def test_model_config_attributes(self):
        """Test that model_config is properly set."""
        assert Settings.model_config.get("env_file") == ".env"
        assert Settings.model_config.get("env_file_encoding") == "utf-8"
        assert Settings.model_config.get("extra") == "ignore"
        assert Settings.model_config.get("frozen") is True
