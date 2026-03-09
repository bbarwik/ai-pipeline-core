"""Unified database module for execution DAG and document storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_pipeline_core.database._download import download_deployment
from ai_pipeline_core.database._factory import Database as _Database
from ai_pipeline_core.database._factory import create_database, create_database_from_settings
from ai_pipeline_core.database._memory import MemoryDatabase as _MemoryDatabase
from ai_pipeline_core.database._protocol import DatabaseReader as _DatabaseReader
from ai_pipeline_core.database._protocol import DatabaseWriter as _DatabaseWriter
from ai_pipeline_core.database._types import NULL_PARENT as _NULL_PARENT
from ai_pipeline_core.database._types import BlobRecord as _BlobRecord
from ai_pipeline_core.database._types import DocumentRecord as _DocumentRecord
from ai_pipeline_core.database._types import ExecutionLog as _ExecutionLog
from ai_pipeline_core.database._types import ExecutionNode as _ExecutionNode
from ai_pipeline_core.database._types import NodeKind as _NodeKind
from ai_pipeline_core.database._types import NodeStatus as _NodeStatus
from ai_pipeline_core.database._types import RunScopeInfo as _RunScopeInfo

if TYPE_CHECKING:
    Database = _Database
    MemoryDatabase = _MemoryDatabase
    DatabaseReader = _DatabaseReader
    DatabaseWriter = _DatabaseWriter
    NULL_PARENT = _NULL_PARENT
    BlobRecord = _BlobRecord
    DocumentRecord = _DocumentRecord
    ExecutionNode = _ExecutionNode
    ExecutionLog = _ExecutionLog
    NodeKind = _NodeKind
    NodeStatus = _NodeStatus
    RunScopeInfo = _RunScopeInfo

__all__ = [
    "RunScopeInfo",
    "create_database",
    "create_database_from_settings",
    "download_deployment",
]


def __getattr__(name: str) -> Any:
    """Resolve stable package-level re-exports lazily.

    Keeping these re-exports dynamic preserves the import path
    ``ai_pipeline_core.database`` for internal consumers without inflating the
    generated public docs with every implementation detail.
    """
    if name in {"Database", "create_database", "create_database_from_settings", "download_deployment"}:
        mapping = {
            "Database": _Database,
            "create_database": create_database,
            "create_database_from_settings": create_database_from_settings,
            "download_deployment": download_deployment,
        }
        return mapping[name]

    if name == "MemoryDatabase":
        return _MemoryDatabase

    if name in {"DatabaseReader", "DatabaseWriter"}:
        mapping = {
            "DatabaseReader": _DatabaseReader,
            "DatabaseWriter": _DatabaseWriter,
        }
        return mapping[name]

    if name in {"BlobRecord", "DocumentRecord", "ExecutionLog", "ExecutionNode", "NodeKind", "NodeStatus", "NULL_PARENT", "RunScopeInfo"}:
        mapping = {
            "BlobRecord": _BlobRecord,
            "DocumentRecord": _DocumentRecord,
            "ExecutionLog": _ExecutionLog,
            "ExecutionNode": _ExecutionNode,
            "NodeKind": _NodeKind,
            "NodeStatus": _NodeStatus,
            "NULL_PARENT": _NULL_PARENT,
            "RunScopeInfo": _RunScopeInfo,
        }
        return mapping[name]

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
