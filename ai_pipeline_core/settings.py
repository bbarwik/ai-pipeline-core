"""Core configuration settings for pipeline operations."""

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "Settings",
    "settings",
]


class Settings(BaseSettings):
    """Base configuration for AI Pipeline applications.

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
    clickhouse_connect_timeout: int = 10
    clickhouse_send_receive_timeout: int = 30

    # Tracking behavior
    tracking_enabled: bool = True  # When False, disables ClickHouse tracking even if ClickHouse is configured

    # Document summary generation (store-level)
    doc_summary_enabled: bool = True
    doc_summary_model: str = "gemini-3-flash"

    # Pub/Sub event delivery
    pubsub_project_id: str = ""
    pubsub_topic_id: str = ""


settings = Settings()
