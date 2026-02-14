# MODULE: settings
# CLASSES: Settings
# DEPENDS: BaseSettings
# SIZE: ~7KB

# === IMPORTS ===
from ai_pipeline_core import Settings
# === PUBLIC API ===

class Settings(BaseSettings):
    """Base configuration class for AI Pipeline applications.

Settings is designed to be inherited by your application's configuration
class. It provides core AI Pipeline settings and type-safe configuration
management with automatic loading from environment variables and .env files.
All settings are immutable after initialization.

Inherit from Settings to add your application-specific configuration:

    >>> from ai_pipeline_core import Settings
    >>>
    >>> class ProjectSettings(Settings):
    ...     # Your custom settings
    ...     app_name: str = "my-app"
    ...     max_retries: int = 3
    ...     enable_cache: bool = True
    >>>
    >>> # Create singleton instance for your app
    >>> settings = ProjectSettings()

Core Attributes:
    openai_base_url: LiteLLM proxy URL for OpenAI-compatible API.
                    Required for all LLM operations. Usually
                    http://localhost:4000 for local development.

    openai_api_key: Authentication key for LiteLLM proxy. Required
                   for LLM operations. Format depends on proxy config.

    prefect_api_url: Prefect server API endpoint. Required for flow
                    deployment and remote execution. Leave empty for
                    local-only execution.

    prefect_api_key: Prefect API authentication key. Required only
                    when connecting to Prefect Cloud or secured server.

    lmnr_project_api_key: Laminar (LMNR) project API key for observability.
                          Optional but recommended for production monitoring.

    lmnr_debug: Debug mode flag for Laminar. Set to "true" to
               enable debug-level logging. Empty string by default.

    gcs_service_account_file: Path to GCS service account JSON file.
                              Used for Prefect deployment bundles to GCS.
                              Optional - if not set, default credentials will be used.

Configuration sources:
    - Environment variables (highest priority)
    - .env file in current directory
    - Default values in class definition

Empty strings are used as defaults to allow optional services.
Check for empty values before using service-specific settings."""
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore', frozen=True)
    openai_base_url: str = ''
    openai_api_key: str = ''
    prefect_api_url: str = ''
    prefect_api_key: str = ''
    prefect_api_auth_string: str = ''
    prefect_work_pool_name: str = 'default'
    prefect_work_queue_name: str = 'default'
    prefect_gcs_bucket: str = ''
    lmnr_project_api_key: str = ''
    lmnr_debug: str = ''
    gcs_service_account_file: str = ''
    clickhouse_host: str = ''
    clickhouse_port: int = 8443
    clickhouse_database: str = 'default'
    clickhouse_user: str = 'default'
    clickhouse_password: str = ''
    clickhouse_secure: bool = True
    tracking_enabled: bool = True
    tracking_summary_model: str = 'gemini-3-flash'
    doc_summary_enabled: bool = True
    doc_summary_model: str = 'gemini-3-flash'


# === EXAMPLES (from tests/) ===

# Example: Settings singleton
# Source: tests/test_settings.py:52
def test_settings_singleton(self):
    """Test that the module provides a settings singleton."""
    # The module exports a pre-created instance
    assert isinstance(settings, Settings)

    # It should be the same instance
    from ai_pipeline_core.settings import settings as settings2

    assert settings is settings2

# Example: Model config attributes
# Source: tests/test_settings.py:114
def test_model_config_attributes(self):
    """Test that model_config is properly set."""
    assert Settings.model_config.get("env_file") == ".env"
    assert Settings.model_config.get("env_file_encoding") == "utf-8"
    assert Settings.model_config.get("extra") == "ignore"
    assert Settings.model_config.get("frozen") is True

# Example: Partial configuration
# Source: tests/test_settings.py:96
def test_partial_configuration(self):
    """Test that partial configuration works."""
    # Only some settings provided
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": ""}, clear=True):
        s = Settings()

        assert s.openai_api_key == "test-key"
        assert s.openai_base_url == ""  # Default

# Example: Env file loading
# Source: tests/test_settings.py:62
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

# Example: Env variable loading
# Source: tests/test_settings.py:26
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

# Example: Extra env ignored
# Source: tests/test_settings.py:43
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

# Example: Env var overrides env file
# Source: tests/test_settings.py:83
@patch.dict(os.environ, {"OPENAI_API_KEY": "from-env-var"})
def test_env_var_overrides_env_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that environment variables override .env file."""
    # Create .env file
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-env-file")

    monkeypatch.chdir(tmp_path)

    s = Settings()

    # Environment variable should win
    assert s.openai_api_key == "from-env-var"

# === ERROR EXAMPLES (What NOT to Do) ===

# Error: Settings immutable config
# Source: tests/test_settings.py:105
def test_settings_immutable_config(self):
    """Test that Settings uses proper Pydantic configuration."""
    s = Settings()

    # Settings should be immutable (frozen=True)
    with pytest.raises(ValidationError) as exc_info:
        s.openai_api_key = "new-key"
    assert "frozen" in str(exc_info.value).lower()
