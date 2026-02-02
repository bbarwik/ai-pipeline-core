# MODULE: document_store
# CLASSES: ClickHouseDocumentStore, LocalDocumentStore, MemoryDocumentStore, DocumentStore
# DEPENDS: Protocol
# SIZE: ~16KB

# === DEPENDENCIES (Resolved) ===

class Protocol:
    """External base class (not fully documented)."""
    ...

# === PUBLIC API ===

class ClickHouseDocumentStore:
    """ClickHouse-backed document store with circuit breaker.

All sync operations run on a single-thread executor (max_workers=1),
so circuit breaker state needs no locking. Async methods dispatch to
this executor via loop.run_in_executor()."""
    def __init__(
        self,
        *,
        host: str,
        port: int = 8443,
        database: str = "default",
        username: str = "default",
        password: str = "",
        secure: bool = True,
        summary_generator: SummaryGenerator | None = None,
    ) -> None:
        self._params = {
            "host": host,
            "port": port,
            "database": database,
            "username": username,
            "password": password,
            "secure": secure,
        }
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ch-docstore")
        self._client: Any = None
        self._tables_initialized = False

        # Circuit breaker state (accessed only from the single executor thread)
        self._consecutive_failures = 0
        self._circuit_open = False
        self._last_reconnect_attempt = 0.0
        self._buffer: deque[_BufferedWrite] = deque(maxlen=_MAX_BUFFER_SIZE)

        # Summary worker
        self._summary_worker: SummaryWorker | None = None
        if summary_generator:
            self._summary_worker = SummaryWorker(
                generator=summary_generator,
                update_fn=self.update_summary,
            )
            self._summary_worker.start()

    async def check_existing(self, sha256s: list[str]) -> set[str]:
        """Return the subset of sha256s that exist in the document index."""
        return await self._run(self._check_existing_sync, sha256s)

    def flush(self) -> None:
        """Block until all pending document summaries are processed."""
        if self._summary_worker:
            self._summary_worker.flush()

    async def has_documents(self, run_scope: str, document_type: type[Document]) -> bool:
        """Check if any documents of this type exist in the run scope."""
        return await self._run(self._has_documents_sync, run_scope, document_type)

    async def load(self, run_scope: str, document_types: list[type[Document]]) -> list[Document]:
        """Load documents via index JOIN content, then batch-fetch attachments."""
        return await self._run(self._load_sync, run_scope, document_types)

    async def load_summaries(self, run_scope: str, document_sha256s: list[str]) -> dict[str, str]:
        """Load summaries by SHA256 from the document index."""
        return await self._run(self._load_summaries_sync, run_scope, document_sha256s)

    async def save(self, document: Document, run_scope: str) -> None:
        """Save a document. Buffers writes when circuit breaker is open."""
        await self._run(self._save_sync, document, run_scope)
        if self._summary_worker and not self._circuit_open:
            self._summary_worker.schedule(run_scope, document)

    async def save_batch(self, documents: list[Document], run_scope: str) -> None:
        """Save multiple documents. Remaining docs are buffered on failure."""
        await self._run(self._save_batch_sync, documents, run_scope)
        if self._summary_worker and not self._circuit_open:
            for doc in documents:
                self._summary_worker.schedule(run_scope, doc)

    def shutdown(self) -> None:
        """Flush pending summaries, stop the summary worker, and release the executor."""
        if self._summary_worker:
            self._summary_worker.shutdown()
        self._executor.shutdown(wait=True)

    async def update_summary(self, run_scope: str, document_sha256: str, summary: str) -> None:
        """Update summary column for a stored document via ALTER TABLE UPDATE."""
        await self._run(self._update_summary_sync, run_scope, document_sha256, summary)


class LocalDocumentStore:
    """Filesystem-backed document store for local development and debugging.

Documents are stored as browsable files organized by canonical type name.
Write order (content before meta) ensures crash safety — load() ignores
content files without a valid .meta.json."""
    def __init__(
        self,
        base_path: Path | None = None,
        *,
        summary_generator: SummaryGenerator | None = None,
    ) -> None:
        self._base_path = base_path or Path.cwd()
        self._meta_path_cache: dict[str, Path] = {}  # "{run_scope}:{sha256}" -> meta file path
        self._summary_worker: SummaryWorker | None = None
        if summary_generator:
            self._summary_worker = SummaryWorker(
                generator=summary_generator,
                update_fn=self.update_summary,
            )
            self._summary_worker.start()

    @property
    def base_path(self) -> Path:
        """Root directory for all stored documents."""
        return self._base_path

    async def check_existing(self, sha256s: list[str]) -> set[str]:
        """Scan all meta files to find matching document_sha256 values."""
        return await asyncio.to_thread(self._check_existing_sync, sha256s)

    def flush(self) -> None:
        """Block until all pending document summaries are processed."""
        if self._summary_worker:
            self._summary_worker.flush()

    async def has_documents(self, run_scope: str, document_type: type[Document]) -> bool:
        """Check for meta files in the type's directory without loading content."""
        return await asyncio.to_thread(self._has_documents_sync, run_scope, document_type)

    async def load(self, run_scope: str, document_types: list[type[Document]]) -> list[Document]:
        """Load documents by type from the run scope directory."""
        return await asyncio.to_thread(self._load_sync, run_scope, document_types)

    async def load_summaries(self, run_scope: str, document_sha256s: list[str]) -> dict[str, str]:
        """Load summaries from .meta.json files."""
        return await asyncio.to_thread(self._load_summaries_sync, run_scope, document_sha256s)

    async def save(self, document: Document, run_scope: str) -> None:
        """Save a document to disk. Idempotent — same SHA256 is a no-op."""
        written = await asyncio.to_thread(self._save_sync, document, run_scope)
        if written and self._summary_worker:
            self._summary_worker.schedule(run_scope, document)

    async def save_batch(self, documents: list[Document], run_scope: str) -> None:
        """Save multiple documents sequentially."""
        for doc in documents:
            await self.save(doc, run_scope)

    def shutdown(self) -> None:
        """Flush pending summaries and stop the summary worker."""
        if self._summary_worker:
            self._summary_worker.shutdown()

    async def update_summary(self, run_scope: str, document_sha256: str, summary: str) -> None:
        """Update summary in the document's .meta.json file."""
        await asyncio.to_thread(self._update_summary_sync, run_scope, document_sha256, summary)


class MemoryDocumentStore:
    """Dict-based document store for unit tests.

Storage layout: dict[run_scope, dict[document_sha256, Document]]."""
    def __init__(
        self,
        *,
        summary_generator: SummaryGenerator | None = None,
    ) -> None:
        self._data: dict[str, dict[str, Document]] = {}
        self._summaries: dict[str, dict[str, str]] = {}  # run_scope -> sha256 -> summary
        self._summary_worker: SummaryWorker | None = None
        if summary_generator:
            self._summary_worker = SummaryWorker(
                generator=summary_generator,
                update_fn=self.update_summary,
            )
            self._summary_worker.start()

    async def check_existing(self, sha256s: list[str]) -> set[str]:
        """Return the subset of sha256s that exist across all scopes."""
        all_hashes: set[str] = set()
        for scope in self._data.values():
            all_hashes.update(scope.keys())
        return all_hashes & set(sha256s)

    def flush(self) -> None:
        """Block until all pending document summaries are processed."""
        if self._summary_worker:
            self._summary_worker.flush()

    async def has_documents(self, run_scope: str, document_type: type[Document]) -> bool:
        """Check if any documents of this type exist in the run scope."""
        scope = self._data.get(run_scope, {})
        return any(isinstance(doc, document_type) for doc in scope.values())

    async def load(self, run_scope: str, document_types: list[type[Document]]) -> list[Document]:
        """Return all documents matching the given types from a run scope."""
        scope = self._data.get(run_scope, {})
        type_tuple = tuple(document_types)
        return [doc for doc in scope.values() if isinstance(doc, type_tuple)]

    async def load_summaries(self, run_scope: str, document_sha256s: list[str]) -> dict[str, str]:
        """Load summaries by SHA256."""
        scope_summaries = self._summaries.get(run_scope, {})
        return {sha: scope_summaries[sha] for sha in document_sha256s if sha in scope_summaries}

    async def save(self, document: Document, run_scope: str) -> None:
        """Store document in memory, keyed by SHA256."""
        scope = self._data.setdefault(run_scope, {})
        if document.sha256 in scope:
            return  # Idempotent — same document already saved
        scope[document.sha256] = document
        if self._summary_worker:
            self._summary_worker.schedule(run_scope, document)

    async def save_batch(self, documents: list[Document], run_scope: str) -> None:
        """Save multiple documents sequentially."""
        for doc in documents:
            await self.save(doc, run_scope)

    def shutdown(self) -> None:
        """Flush pending summaries and stop the summary worker."""
        if self._summary_worker:
            self._summary_worker.shutdown()

    async def update_summary(self, run_scope: str, document_sha256: str, summary: str) -> None:
        """Update summary for a stored document. No-op if document doesn't exist."""
        scope = self._data.get(run_scope, {})
        if document_sha256 not in scope:
            return
        self._summaries.setdefault(run_scope, {})[document_sha256] = summary


@runtime_checkable
class DocumentStore(Protocol):
    """Protocol for document storage backends.

Implementations: ClickHouseDocumentStore (production), LocalDocumentStore (CLI/debug),
MemoryDocumentStore (testing)."""
    async def check_existing(self, sha256s: list[str]) -> set[str]:
        """Return the subset of sha256s that already exist in the store."""
        ...

    def flush(self) -> None:
        """Block until all pending background work (summaries) is processed."""
        ...

    async def has_documents(self, run_scope: str, document_type: type[Document]) -> bool:
        """Check if any documents of this type exist in the run scope."""
        ...

    async def load(self, run_scope: str, document_types: list[type[Document]]) -> list[Document]:
        """Load all documents of the given types from a run scope."""
        ...

    async def load_summaries(self, run_scope: str, document_sha256s: list[str]) -> dict[str, str]:
        """Load summaries by SHA256. Returns {sha256: summary} for docs that have summaries."""
        ...

    async def save(self, document: Document, run_scope: str) -> None:
        """Save a single document to the store. Idempotent — same SHA256 is a no-op."""
        ...

    async def save_batch(self, documents: list[Document], run_scope: str) -> None:
        """Save multiple documents. Dependencies must be sorted (caller's responsibility)."""
        ...

    def shutdown(self) -> None:
        """Flush pending work and stop background workers."""
        ...

    async def update_summary(self, run_scope: str, document_sha256: str, summary: str) -> None:
        """Update summary for a stored document. No-op if document doesn't exist."""
        ...


# === FUNCTIONS ===

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
    if not isinstance(settings, Settings):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"Expected Settings instance, got {type(settings).__name__}")  # pyright: ignore[reportUnreachable]

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

def get_document_store() -> DocumentStore | None:
    """Get the process-global document store singleton."""
    return _document_store

def set_document_store(store: DocumentStore | None) -> None:
    """Set the process-global document store singleton."""
    global _document_store
    _document_store = store

# === EXAMPLES (from tests/) ===

# Example: Create document store returns local when no clickhouse
# Source: tests/document_store/test_local.py:294
def test_create_document_store_returns_local_when_no_clickhouse(self):
    from ai_pipeline_core.settings import Settings

    settings = Settings(clickhouse_host="")
    store = create_document_store(settings)
    assert isinstance(store, LocalDocumentStore)

# Example: Get document store returns none by default
# Source: tests/document_store/test_protocol.py:70
def test_get_document_store_returns_none_by_default():
    """Before any set call, the store is None."""
    assert get_document_store() is None

# Example: Set document store to none
# Source: tests/document_store/test_protocol.py:82
def test_set_document_store_to_none():
    """Store can be reset to None."""
    store = _DummyStore()
    set_document_store(store)
    assert get_document_store() is store
    set_document_store(None)
    assert get_document_store() is None

# Example: Set and get document store
# Source: tests/document_store/test_protocol.py:75
def test_set_and_get_document_store():
    """Setting a store makes it retrievable."""
    store = _DummyStore()
    set_document_store(store)
    assert get_document_store() is store

# Example: Basic round trip
# Source: tests/document_store/test_local.py:45
@pytest.mark.asyncio
async def test_basic_round_trip(self, store: LocalDocumentStore):
    doc = _make(ReportDoc, "report.md", "# Hello\nWorld")
    await store.save(doc, "run1")
    loaded = await store.load("run1", [ReportDoc])
    assert len(loaded) == 1
    assert loaded[0].name == "report.md"
    assert loaded[0].content == b"# Hello\nWorld"
    assert isinstance(loaded[0], ReportDoc)

# Example: Satisfies document store protocol
# Source: tests/document_store/test_local.py:38
def test_satisfies_document_store_protocol(self, tmp_path: Path):
    store = LocalDocumentStore(base_path=tmp_path)
    assert isinstance(store, DocumentStore)

# Example: Satisfies document store protocol
# Source: tests/document_store/test_memory.py:28
def test_satisfies_document_store_protocol(self):
    store = MemoryDocumentStore()
    assert isinstance(store, DocumentStore)

# === ERROR EXAMPLES (What NOT to Do) ===

# Error: Create document store rejects non settings
# Source: tests/document_store/test_local.py:301
def test_create_document_store_rejects_non_settings(self):
    with pytest.raises(TypeError, match="Expected Settings"):
        create_document_store("not_settings")  # type: ignore[arg-type]
