"""Core configuration settings for pipeline operations."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core settings for pipeline operations."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

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
