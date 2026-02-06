"""Core configuration settings for pipeline operations.

This module provides the Settings base class for configuration management.
Applications should inherit from Settings to create their own ProjectSettings
class with additional configuration fields.

Environment variables:
    OPENAI_BASE_URL: LiteLLM proxy endpoint (e.g., http://localhost:4000)
    OPENAI_API_KEY: API key for LiteLLM proxy authentication
    PREFECT_API_URL: Prefect server endpoint for flow orchestration
    PREFECT_API_KEY: Prefect API authentication key
    LMNR_PROJECT_API_KEY: Laminar project key for observability
    GCS_SERVICE_ACCOUNT_FILE: Path to GCS service account JSON file (for Prefect deployment bundles)

Configuration precedence:
    1. Environment variables (highest priority)
    2. .env file in current directory
    3. Default values (empty strings)

.env file format:
    OPENAI_BASE_URL=http://localhost:4000
    OPENAI_API_KEY=sk-1234567890
    PREFECT_API_URL=http://localhost:4200/api
    PREFECT_API_KEY=pnu_abc123
    LMNR_PROJECT_API_KEY=lmnr_proj_xyz
    GCS_SERVICE_ACCOUNT_FILE=/path/to/service-account.json  # For Prefect deployment
    APP_NAME=production-app
    DEBUG_MODE=false

Settings are loaded once at initialization and frozen. There is no
built-in reload mechanism - the process must be restarted to pick up
changes to environment variables or .env file. This is by design to
ensure consistency during execution.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    Check for empty values before using service-specific settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,  # Settings are immutable after initialization
    )

    # LLM API Configuration
    openai_base_url: str = ""
    openai_api_key: str = ""

    # Prefect Configuration
    prefect_api_url: str = ""
    prefect_api_key: str = ""
    prefect_api_auth_string: str = ""
    prefect_work_pool_name: str = "default"
    prefect_work_queue_name: str = "default"
    prefect_gcs_bucket: str = ""

    # Observability
    lmnr_project_api_key: str = ""
    lmnr_debug: str = ""

    # GCS (for Prefect deployment bundles)
    gcs_service_account_file: str = ""  # Path to GCS service account JSON file

    # ClickHouse tracking
    clickhouse_host: str = ""
    clickhouse_port: int = 8443
    clickhouse_database: str = "default"
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_secure: bool = True

    # Tracking behavior
    tracking_enabled: bool = True
    tracking_summary_model: str = "gemini-3-flash"

    # Document summary generation (store-level)
    doc_summary_enabled: bool = True
    doc_summary_model: str = "gemini-3-flash"

    # URL/address substitution for LLM calls (deterministic labels only)
    substitutor_enabled: bool = False


settings = Settings()
