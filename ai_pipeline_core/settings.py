"""Core configuration settings for pipeline operations."""

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "Settings",
    "settings",
]


class Settings(BaseSettings):
    """Base configuration for AI Pipeline applications.

    Fields map to environment variables via Pydantic BaseSettings
    (e.g. ``clickhouse_host`` → ``CLICKHOUSE_HOST``). Uses ``.env`` file when present.

    Inherit to add application-specific fields::

        class ProjectSettings(Settings):
            app_name: str = "my-app"

        settings = ProjectSettings()
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

    # GCS (for Prefect deployment bundles)
    gcs_service_account_file: str = ""  # Path to GCS service account JSON file

    # ClickHouse tracking
    clickhouse_host: str = ""
    clickhouse_port: int = 8443
    clickhouse_database: str = "default"
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_secure: bool = True
    clickhouse_connect_timeout: int = 10
    clickhouse_send_receive_timeout: int = 30

    # Tracking behavior
    tracking_enabled: bool = True  # When False, disables ClickHouse tracking even if ClickHouse is configured

    # Document summary generation (store-level)
    doc_summary_enabled: bool = True
    doc_summary_model: str = "gemini-3.1-flash-lite"

    # Pub/Sub event delivery
    pubsub_project_id: str = ""
    pubsub_topic_id: str = ""

    @model_validator(mode="after")
    def _disable_summary_without_llm(self) -> "Settings":
        """Auto-disable doc summary generation when LLM credentials are not configured."""
        if self.doc_summary_enabled and (not self.openai_api_key or not self.openai_base_url):
            object.__setattr__(self, "doc_summary_enabled", False)  # noqa: PLC2801 — frozen model rejects normal setattr
        return self


settings = Settings()
