# MODULE: database
# CLASSES: RunScopeInfo
# PURPOSE: Unified database module for execution DAG and document storage.
# VERSION: 0.14.0
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core.database import RunScopeInfo, create_database, create_database_from_settings, download_deployment
```

## Public API

```python
@dataclass(frozen=True, slots=True)
class RunScopeInfo:
    """Aggregated metadata for a non-empty document run scope."""
    run_scope: RunScope
    document_count: int
    latest_created_at: datetime


```

## Functions

```python
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

async def download_deployment(
    source: DatabaseReader,
    deployment_id: UUID,
    output_path: Path,
) -> None:
    """Export a deployment as a FilesystemDatabase snapshot."""
    tree = await source.get_deployment_tree(deployment_id)
    target = await asyncio.to_thread(FilesystemDatabase, output_path)

    for node in _sort_nodes_parent_first(tree):
        await target.insert_node(node)

    document_shas: set[DocumentSha256] = set()
    for node in tree:
        document_shas.update(DocumentSha256(sha) for sha in node.context_document_shas)
        document_shas.update(DocumentSha256(sha) for sha in node.input_document_shas)
        document_shas.update(DocumentSha256(sha) for sha in node.output_document_shas)
        _collect_replay_payload_document_shas(node.payload.get("replay_payload"), document_shas)

    for document in await source.get_documents_by_deployment(deployment_id):
        document_shas.add(document.document_sha256)

    documents = await source.get_documents_batch(list(document_shas))
    if documents:
        await target.save_document_batch(list(documents.values()))

    blob_shas: set[str] = set()
    for document in documents.values():
        blob_shas.add(document.content_sha256)
        blob_shas.update(document.attachment_sha256s)

    blobs = await source.get_blobs_batch(list(blob_shas))
    if blobs:
        await target.save_blob_batch(list(blobs.values()))

    logs = await source.get_deployment_logs(deployment_id)
    if logs:
        await target.save_logs_batch(logs)
    else:
        await asyncio.to_thread((output_path / "logs.jsonl").write_text, "", encoding="utf-8")

    summary = await generate_summary(target, deployment_id)
    costs = await generate_costs(target, deployment_id)
    await asyncio.to_thread((output_path / "summary.md").write_text, summary, encoding="utf-8")
    await asyncio.to_thread((output_path / "costs.md").write_text, costs, encoding="utf-8")

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
        return MemoryDatabase()

    if backend == "filesystem":
        from ai_pipeline_core.database._filesystem import FilesystemDatabase

        if base_path is None:
            msg = "FilesystemDatabase requires base_path parameter"
            raise ValueError(msg)
        return FilesystemDatabase(base_path)

    if backend == "clickhouse":
        from ai_pipeline_core.database._clickhouse import ClickHouseDatabase

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

```

## Examples

**Creation** (`tests/database/test_types.py:162`)

```python
def test_creation(self) -> None:
    blob = BlobRecord(content_sha256="abc123", content=b"hello world", size_bytes=11)
    assert blob.content == b"hello world"
    assert blob.size_bytes == 11
```

**All values** (`tests/database/test_types.py:21`)

```python
def test_all_values(self) -> None:
    assert len(NodeKind) == 5
```

**All values** (`tests/database/test_types.py:33`)

```python
def test_all_values(self) -> None:
    assert len(NodeStatus) == 5
```

**Alphanumeric passes through** (`tests/database/test_filesystem.py:87`)

```python
def test_alphanumeric_passes_through(self) -> None:
    assert _sanitize_dir_name("MyFlow") == "MyFlow"
```


## Error Examples

**Frozen immutability** (`tests/database/test_types.py:167`)

```python
def test_frozen_immutability(self) -> None:
    blob = BlobRecord(content_sha256="abc123", content=b"data", size_bytes=4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        blob.content = b"changed"  # type: ignore[misc]
```

**Update nonexistent raises** (`tests/database/test_filesystem.py:204`)

```python
async def test_update_nonexistent_raises(self, tmp_path: Path) -> None:
    db = FilesystemDatabase(tmp_path)
    with pytest.raises(KeyError):
        await db.update_node(uuid4(), status=NodeStatus.COMPLETED)
```

**Update nonexistent raises** (`tests/database/test_memory.py:147`)

```python
@pytest.mark.asyncio
async def test_update_nonexistent_raises(self) -> None:
    db = MemoryDatabase()
    with pytest.raises(KeyError):
        await db.update_node(uuid4(), status=NodeStatus.COMPLETED)
```

**Frozen immutability** (`tests/database/test_types.py:205`)

```python
def test_frozen_immutability(self) -> None:
    deployment_id = uuid4()
    log = ExecutionLog(
        node_id=uuid4(),
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        flow_id=None,
        task_id=None,
        timestamp=datetime.now(UTC),
        sequence_no=0,
        level="INFO",
        category="framework",
        logger_name="ai_pipeline_core.tests",
        message="hello",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        log.message = "changed"  # type: ignore[misc]
```
