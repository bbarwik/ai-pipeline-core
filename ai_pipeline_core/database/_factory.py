"""Database backend factory functions and type alias."""

from pathlib import Path

from ai_pipeline_core.database._memory import _MemoryDatabase
from ai_pipeline_core.database.clickhouse._backend import ClickHouseDatabase
from ai_pipeline_core.database.filesystem._backend import FilesystemDatabase
from ai_pipeline_core.settings import Settings

# Type alias combining both protocols — all backends implement both
Database = _MemoryDatabase | FilesystemDatabase | ClickHouseDatabase

__all__ = [
    "Database",
    "create_database",
    "create_database_from_settings",
]


def create_database(
    *,
    backend: str = "memory",
    base_path: Path | None = None,
    settings: Settings | None = None,
) -> Database:
    """Factory for creating database backends.

    Args:
        backend: Backend type — 'memory', 'filesystem', or 'clickhouse'.
        base_path: Root directory for filesystem backend.
        settings: Application settings (required for 'clickhouse' backend).

    Returns:
        A database backend implementing both DatabaseWriter and DatabaseReader.
    """
    if backend == "memory":
        return _MemoryDatabase()

    if backend == "filesystem":
        if base_path is None:
            msg = "FilesystemDatabase requires base_path parameter"
            raise ValueError(msg)
        return FilesystemDatabase(base_path)

    if backend == "clickhouse":
        return ClickHouseDatabase(settings=settings)

    supported = "'memory', 'filesystem', 'clickhouse'"
    msg = f"Unknown database backend: {backend!r}. Supported: {supported}"
    raise ValueError(msg)


def create_database_from_settings(
    settings: Settings,
    base_path: Path | None = None,
) -> Database:
    """Create the right database backend based on application settings.

    Args:
        settings: Application settings with ClickHouse configuration.
        base_path: Root directory for filesystem backend (used when ClickHouse is not configured).

    Returns:
        ClickHouse backend if clickhouse_host is set, filesystem if base_path is provided,
        otherwise in-memory backend.
    """
    if settings.clickhouse_host:
        return create_database(backend="clickhouse", settings=settings)
    if base_path is not None:
        return create_database(backend="filesystem", base_path=base_path)
    return create_database(backend="memory")
