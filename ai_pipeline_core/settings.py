"""Core configuration settings for pipeline operations.

@public

This module provides centralized configuration management for AI Pipeline Core,
handling all external service credentials and endpoints. Settings are loaded
from environment variables with .env file support via pydantic-settings.

Environment variables:
    OPENAI_BASE_URL: LiteLLM proxy endpoint (e.g., http://localhost:4000)
    OPENAI_API_KEY: API key for LiteLLM proxy authentication
    PREFECT_API_URL: Prefect server endpoint for flow orchestration
    PREFECT_API_KEY: Prefect API authentication key
    LMNR_PROJECT_API_KEY: Laminar project key for observability

Configuration precedence:
    1. Environment variables (highest priority)
    2. .env file in current directory
    3. Default values (empty strings)

Example:
    >>> from ai_pipeline_core.settings import settings
    >>>
    >>> # Access configuration
    >>> print(settings.openai_base_url)
    >>> print(settings.prefect_api_url)
    >>>
    >>> # Settings are frozen after initialization
    >>> settings.openai_api_key = "new_key"  # Raises error

.env file format:
    OPENAI_BASE_URL=http://localhost:4000
    OPENAI_API_KEY=sk-1234567890
    PREFECT_API_URL=http://localhost:4200/api
    PREFECT_API_KEY=pnu_abc123
    LMNR_PROJECT_API_KEY=lmnr_proj_xyz

Note:
    Settings are loaded once at module import and frozen. There is no
    built-in reload mechanism - the process must be restarted to pick up
    changes to environment variables or .env file. This is by design to
    ensure consistency during execution.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core configuration for AI Pipeline external services.

    @public

    Settings provides type-safe configuration management with automatic
    loading from environment variables and .env files. All settings are
    immutable after initialization.

    Attributes:
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

        lmnr_project_api_key: Laminar (LMNR) project API key for tracing
                              and observability. Optional but recommended
                              for production monitoring.

    Configuration sources:
        - Environment variables (OPENAI_BASE_URL, etc.)
        - .env file in current directory
        - Default empty strings if not configured

    Example:
        >>> # Typically accessed via module-level instance
        >>> from ai_pipeline_core.settings import settings
        >>>
        >>> if not settings.openai_base_url:
        ...     raise ValueError("OPENAI_BASE_URL must be configured")
        >>>
        >>> # Settings are frozen (immutable)
        >>> print(settings.model_dump())  # View all settings

    Note:
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

    # Observability
    lmnr_project_api_key: str = ""


# Create a single, importable instance of the settings
settings = Settings()
"""Global settings instance for the entire application.

@public

This singleton instance is created at module import and provides
configuration to all pipeline components. Access this instance
rather than creating new Settings objects.

Example:
    >>> from ai_pipeline_core.settings import settings
    >>> print(f"Using LLM proxy at {settings.openai_base_url}")
"""
