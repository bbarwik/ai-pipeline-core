# MODULE: document_store
# CLASSES: DocumentReader, FlowCompletion, DocumentNode
# DEPENDS: Protocol
# PURPOSE: Document store protocol and backends for AI pipeline flows.
# VERSION: 0.10.1
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import DocumentReader, get_document_store
from ai_pipeline_core.document_store import DocumentNode, FlowCompletion
```

## Public API

```python
# Protocol — implement in concrete class
@runtime_checkable
class DocumentReader(Protocol):
    """Read-only protocol for application code that consumes documents.

Users should depend on this protocol when they only need to read documents."""
    async def check_existing(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Return the subset of sha256s that already exist in the store."""
        ...

    async def find_by_source(
        self,
        source_values: list[str],
        document_type: type[Document],
        *,
        max_age: timedelta | None = None,
    ) -> dict[str, Document]:
        """Find the most recent document per source value."""
        ...

    async def get_flow_completion(
        self,
        run_scope: RunScope,
        flow_name: str,
        *,
        max_age: timedelta | None = None,
    ) -> FlowCompletion | None:
        """Get the completion record for a flow, or None if not found / expired."""
        ...

    async def has_documents(self, run_scope: RunScope, document_type: type[Document], *, max_age: timedelta | None = None) -> bool:
        """Check if any documents of this type exist in the run scope."""
        ...

    async def load(self, run_scope: RunScope, document_types: list[type[Document]]) -> list[Document]:
        """Load all documents of the given types from a run scope."""
        ...

    async def load_by_sha256s(self, sha256s: list[DocumentSha256], document_type: type[_D], run_scope: RunScope | None = None) -> dict[DocumentSha256, _D]:
        """Batch-load full documents by SHA256."""
        ...

    async def load_nodes_by_sha256s(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentNode]:
        """Batch-load lightweight metadata for documents by SHA256."""
        ...

    async def load_scope_metadata(self, run_scope: RunScope) -> list[DocumentNode]:
        """Load lightweight metadata for ALL documents in a run scope."""
        ...

    async def load_summaries(self, document_sha256s: list[DocumentSha256]) -> dict[DocumentSha256, str]:
        """Load summaries by SHA256."""
        ...


@dataclass(frozen=True, slots=True)
class FlowCompletion:
    """Record of a successful flow execution for resume cache.

Written after a flow returns successfully. Resume checks this record
instead of inferring completion from document presence (which gives
false positives when a flow crashes after partial output)."""
    flow_name: str
    input_sha256s: tuple[str, ...]
    output_sha256s: tuple[str, ...]
    stored_at: datetime


@dataclass(frozen=True, slots=True)
class DocumentNode:
    """Lightweight document metadata without content or attachments."""
    sha256: DocumentSha256
    class_name: str
    name: str
    description: str = ''
    derived_from: tuple[str, ...] = ()
    triggered_by: tuple[str, ...] = ()
    summary: str = ''


```

## Functions

```python
def get_document_store() -> DocumentReader | None:
    """Get the process-global document store for read-only access."""
    return get_store()

```

## Examples

**Get document store returns reader** (`tests/document_store/test_reader_api.py:46`)

```python
@pytest.mark.asyncio
async def test_get_document_store_returns_reader(populated_store: MemoryDocumentStore):
    """get_document_store() returns a DocumentReader for read-only access."""
    reader = get_document_store()
    assert reader is not None
    assert isinstance(reader, DocumentReader)
```

**Load documents by type** (`tests/document_store/test_reader_api.py:55`)

```python
@pytest.mark.asyncio
async def test_load_documents_by_type(populated_store: MemoryDocumentStore):
    """Load all documents of specific types from a run scope."""
    report, src_a, src_b = await _seed(populated_store)
    reader = get_document_store()

    reports = await reader.load(RunScope("project/run-1"), [ReportDoc])
    assert len(reports) == 1
    assert reports[0].name == "report.md"

    sources = await reader.load(RunScope("project/run-1"), [SourceDoc])
    assert len(sources) == 2

    all_docs = await reader.load(RunScope("project/run-1"), [ReportDoc, SourceDoc])
    assert len(all_docs) == 3
```

**Has documents check** (`tests/document_store/test_reader_api.py:73`)

```python
@pytest.mark.asyncio
async def test_has_documents_check(populated_store: MemoryDocumentStore):
    """Check whether documents of a given type exist in a scope."""
    await _seed(populated_store)
    reader = get_document_store()

    assert await reader.has_documents(RunScope("project/run-1"), ReportDoc) is True
    assert await reader.has_documents(RunScope("project/run-1"), SourceDoc) is True
    assert await reader.has_documents(RunScope("nonexistent"), ReportDoc) is False
```

**Check existing sha256s** (`tests/document_store/test_reader_api.py:85`)

```python
@pytest.mark.asyncio
async def test_check_existing_sha256s(populated_store: MemoryDocumentStore):
    """Check which SHA256 hashes exist in the store."""
    report, *_ = await _seed(populated_store)
    reader = get_document_store()

    existing = await reader.check_existing([report.sha256, DocumentSha256("NONEXISTENT" * 4 + "AAAA")])
    assert report.sha256 in existing
    assert len(existing) == 1
```

**Load by sha256s** (`tests/document_store/test_reader_api.py:97`)

```python
@pytest.mark.asyncio
async def test_load_by_sha256s(populated_store: MemoryDocumentStore):
    """Batch-load full documents by their SHA256 hashes."""
    report, src_a, src_b = await _seed(populated_store)
    reader = get_document_store()

    result = await reader.load_by_sha256s([src_a.sha256, src_b.sha256], SourceDoc, RunScope("project/run-1"))
    assert len(result) == 2
    assert result[src_a.sha256].name == "source_a.md"
    assert result[src_b.sha256].name == "source_b.md"
```

**Load scope metadata** (`tests/document_store/test_reader_api.py:110`)

```python
@pytest.mark.asyncio
async def test_load_scope_metadata(populated_store: MemoryDocumentStore):
    """Load lightweight metadata for all documents in a scope."""
    report, src_a, src_b = await _seed(populated_store)
    reader = get_document_store()

    nodes = await reader.load_scope_metadata(RunScope("project/run-1"))
    assert len(nodes) == 3
    sha_set = {n.sha256 for n in nodes}
    assert report.sha256 in sha_set

    report_node = next(n for n in nodes if n.sha256 == report.sha256)
    assert report_node.class_name == "ReportDoc"
    assert report_node.name == "report.md"
    assert src_a.sha256 in report_node.derived_from
```

**Load nodes by sha256s** (`tests/document_store/test_reader_api.py:128`)

```python
@pytest.mark.asyncio
async def test_load_nodes_by_sha256s(populated_store: MemoryDocumentStore):
    """Batch-load lightweight metadata nodes by SHA256 — works across scopes."""
    report, src_a, _ = await _seed(populated_store)
    reader = get_document_store()

    nodes = await reader.load_nodes_by_sha256s([report.sha256, src_a.sha256])
    assert len(nodes) == 2
    assert nodes[report.sha256].name == "report.md"
    assert nodes[src_a.sha256].class_name == "SourceDoc"
```

**Get document store returns none by default** (`tests/document_store/test_protocol.py:94`)

```python
def test_get_document_store_returns_none_by_default():
    """Before any set call, the store is None."""
    assert get_document_store() is None
```

**Set and get document store** (`tests/document_store/test_protocol.py:99`)

```python
def test_set_and_get_document_store():
    """Setting a store makes it retrievable."""
    store = _DummyStore()
    set_document_store(store)
    assert get_document_store() is store
```


## Error Examples

**Create document store rejects non settings** (`tests/document_store/test_local.py:419`)

```python
def test_create_document_store_rejects_non_settings(self):
    with pytest.raises(AttributeError):
        create_document_store("not_settings")  # type: ignore[arg-type]
```
