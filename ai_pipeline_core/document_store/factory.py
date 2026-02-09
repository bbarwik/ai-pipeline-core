"""Factory function for creating document store instances based on settings."""

from ai_pipeline_core.document_store._summary import SummaryGenerator
from ai_pipeline_core.document_store.protocol import DocumentStore
from ai_pipeline_core.settings import Settings


def create_document_store(
    settings: Settings,
    *,
    summary_generator: SummaryGenerator | None = None,
) -> DocumentStore:
    """Create a DocumentStore based on settings.

    Selects ClickHouseDocumentStore when clickhouse_host is configured,
    otherwise falls back to LocalDocumentStore.

    Backends are imported lazily to avoid circular imports.
    """
    if settings.clickhouse_host:
        from ai_pipeline_core.document_store.clickhouse import ClickHouseDocumentStore

        return ClickHouseDocumentStore(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            database=settings.clickhouse_database,
            username=settings.clickhouse_user,
            password=settings.clickhouse_password,
            secure=settings.clickhouse_secure,
            summary_generator=summary_generator,
        )

    from ai_pipeline_core.document_store.local import LocalDocumentStore

    return LocalDocumentStore(summary_generator=summary_generator)
