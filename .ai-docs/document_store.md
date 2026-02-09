# MODULE: document_store
# CLASSES: ClickHouseDocumentStore, LocalDocumentStore, MemoryDocumentStore, DocumentStore
# DEPENDS: Protocol
# SIZE: ~22KB

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

    async def check_existing(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Return the subset of sha256s that exist in the document index."""
        return await self._run(self._check_existing_sync, sha256s)

    def flush(self) -> None:
        """Block until all pending document summaries are processed."""
        if self._summary_worker:
            self._summary_worker.flush()

    async def has_documents(self, run_scope: RunScope, document_type: type[Document]) -> bool:
        """Check if any documents of this type exist in the run scope."""
        return await self._run(self._has_documents_sync, run_scope, document_type)

    async def load(self, run_scope: RunScope, document_types: list[type[Document]]) -> list[Document]:
        """Load documents via run_documents + index JOIN content, then batch-fetch attachments."""
        return await self._run(self._load_sync, run_scope, document_types)

    async def load_by_sha256s(self, sha256s: list[DocumentSha256], document_type: type[_D], run_scope: RunScope | None = None) -> dict[DocumentSha256, _D]:
        """Batch-load documents by SHA256 and type. Verifies run membership when run_scope provided."""
        return await self._run(self._load_by_sha256s_sync, sha256s, document_type, run_scope)

    async def load_nodes_by_sha256s(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentNode]:
        """Batch-load lightweight metadata by SHA256. No JOIN needed."""
        return await self._run(self._load_nodes_by_sha256s_sync, sha256s)

    async def load_scope_metadata(self, run_scope: RunScope) -> list[DocumentNode]:
        """Load lightweight metadata for all documents in a run scope."""
        return await self._run(self._load_scope_metadata_sync, run_scope)

    async def load_summaries(self, document_sha256s: list[DocumentSha256]) -> dict[DocumentSha256, str]:
        """Load summaries by SHA256 from the document index."""
        return await self._run(self._load_summaries_sync, document_sha256s)

    async def save(self, document: Document, run_scope: RunScope) -> None:
        """Save a document. Buffers writes when circuit breaker is open."""
        await self._run(self._save_sync, document, run_scope)
        if self._summary_worker and not self._circuit_open:
            self._summary_worker.schedule(document)

    async def save_batch(self, documents: list[Document], run_scope: RunScope) -> None:
        """Save multiple documents. Remaining docs are buffered on failure."""
        await self._run(self._save_batch_sync, documents, run_scope)
        if self._summary_worker and not self._circuit_open:
            for doc in documents:
                self._summary_worker.schedule(doc)

    def shutdown(self) -> None:
        """Flush pending summaries, stop the summary worker, and release the executor."""
        if self._summary_worker:
            self._summary_worker.shutdown()
        self._executor.shutdown(wait=True)

    async def update_summary(self, document_sha256: DocumentSha256, summary: str) -> None:
        """Update summary via INSERT with incremented version (ReplacingMergeTree dedup)."""
        await self._run(self._update_summary_sync, document_sha256, summary)


class LocalDocumentStore:
    """Filesystem-backed document store for local development and debugging.

Documents are stored as browsable files organized by class name.
Write order (content before meta) ensures crash safety — load() ignores
content files without a valid .meta.json."""
    def __init__(
        self,
        base_path: Path | None = None,
        *,
        summary_generator: SummaryGenerator | None = None,
    ) -> None:
        self._base_path = base_path or Path.cwd()
        self._meta_path_cache: dict[str, list[Path]] = {}  # sha256 -> list of meta file paths
        self._cache_lock = threading.Lock()
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

    async def check_existing(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Scan all meta files to find matching document_sha256 values."""
        return await asyncio.to_thread(self._check_existing_sync, sha256s)

    def flush(self) -> None:
        """Block until all pending document summaries are processed."""
        if self._summary_worker:
            self._summary_worker.flush()

    async def has_documents(self, run_scope: RunScope, document_type: type[Document]) -> bool:
        """Check for meta files in the type's directory without loading content."""
        return await asyncio.to_thread(self._has_documents_sync, run_scope, document_type)

    async def load(self, run_scope: RunScope, document_types: list[type[Document]]) -> list[Document]:
        """Load documents by type from the run scope directory."""
        return await asyncio.to_thread(self._load_sync, run_scope, document_types)

    async def load_by_sha256s(self, sha256s: list[DocumentSha256], document_type: type[_D], run_scope: RunScope | None = None) -> dict[DocumentSha256, _D]:
        """Batch-load documents by SHA256. Searches all directories — class_name is not enforced."""
        return await asyncio.to_thread(self._load_by_sha256s_sync, sha256s, document_type, run_scope)  # pyright: ignore[reportReturnType]

    async def load_nodes_by_sha256s(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentNode]:
        """Batch-load lightweight metadata by SHA256, searching all scopes."""
        return await asyncio.to_thread(self._load_nodes_by_sha256s_sync, sha256s)

    async def load_scope_metadata(self, run_scope: RunScope) -> list[DocumentNode]:
        """Load lightweight metadata for all documents in a run scope."""
        return await asyncio.to_thread(self._load_scope_metadata_sync, run_scope)

    async def load_summaries(self, document_sha256s: list[DocumentSha256]) -> dict[DocumentSha256, str]:
        """Load summaries from .meta.json files across all scopes."""
        return await asyncio.to_thread(self._load_summaries_sync, document_sha256s)

    async def save(self, document: Document, run_scope: RunScope) -> None:
        """Save a document to disk. Idempotent — same SHA256 is a no-op."""
        written = await asyncio.to_thread(self._save_sync, document, run_scope)
        if written and self._summary_worker:
            self._summary_worker.schedule(document)

    async def save_batch(self, documents: list[Document], run_scope: RunScope) -> None:
        """Save multiple documents sequentially."""
        for doc in documents:
            await self.save(doc, run_scope)

    def shutdown(self) -> None:
        """Flush pending summaries and stop the summary worker."""
        if self._summary_worker:
            self._summary_worker.shutdown()

    async def update_summary(self, document_sha256: DocumentSha256, summary: str) -> None:
        """Update summary in all .meta.json files for this document across all scopes."""
        await asyncio.to_thread(self._update_summary_sync, document_sha256, summary)


class MemoryDocumentStore:
    """Dict-based document store for unit tests.

Storage layout: global documents dict + per-run membership sets + global summaries."""
    def __init__(
        self,
        *,
        summary_generator: SummaryGenerator | None = None,
    ) -> None:
        self._documents: dict[str, Document] = {}  # sha256 -> Document (global)
        self._run_docs: dict[str, set[str]] = {}  # run_scope -> set of sha256s
        self._summaries: dict[str, str] = {}  # sha256 -> summary (global)
        self._summary_worker: SummaryWorker | None = None
        if summary_generator:
            self._summary_worker = SummaryWorker(
                generator=summary_generator,
                update_fn=self.update_summary,
            )
            self._summary_worker.start()

    async def check_existing(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Return the subset of sha256s that exist in global documents."""
        return {sha for sha in sha256s if sha in self._documents}

    def flush(self) -> None:
        """Block until all pending document summaries are processed."""
        if self._summary_worker:
            self._summary_worker.flush()

    async def has_documents(self, run_scope: RunScope, document_type: type[Document]) -> bool:
        """Check if any documents of this type exist in the run scope."""
        sha256s = self._run_docs.get(run_scope, set())
        return any(sha in self._documents and isinstance(self._documents[sha], document_type) for sha in sha256s)

    async def load(self, run_scope: RunScope, document_types: list[type[Document]]) -> list[Document]:
        """Return all documents matching the given types from a run scope."""
        sha256s = self._run_docs.get(run_scope, set())
        type_tuple = tuple(document_types)
        return [self._documents[sha] for sha in sha256s if sha in self._documents and isinstance(self._documents[sha], type_tuple)]

    async def load_by_sha256s(self, sha256s: list[DocumentSha256], document_type: type[_D], run_scope: RunScope | None = None) -> dict[DocumentSha256, _D]:
        """Batch-load documents by SHA256. document_type is for construction hint only, not enforced."""
        if not sha256s:
            return {}
        scope_members = self._run_docs.get(run_scope, set()) if run_scope is not None else None
        result: dict[DocumentSha256, _D] = {}
        for sha256 in sha256s:
            if scope_members is not None and sha256 not in scope_members:
                continue
            doc = self._documents.get(sha256)
            if doc is not None:
                result[sha256] = doc  # type: ignore[assignment]
        return result

    async def load_nodes_by_sha256s(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentNode]:
        """Batch-load lightweight metadata by SHA256 from global documents."""
        if not sha256s:
            return {}
        result: dict[DocumentSha256, DocumentNode] = {}
        for sha256 in sha256s:
            doc = self._documents.get(sha256)
            if doc is not None:
                result[sha256] = DocumentNode(
                    sha256=sha256,
                    class_name=doc.__class__.__name__,
                    name=doc.name,
                    description=doc.description or "",
                    sources=doc.sources,
                    origins=doc.origins,
                    summary=self._summaries.get(sha256, ""),
                )
        return result

    async def load_scope_metadata(self, run_scope: RunScope) -> list[DocumentNode]:
        """Load lightweight metadata for all documents in a run scope."""
        sha256s = self._run_docs.get(run_scope, set())
        return [
            DocumentNode(
                sha256=DocumentSha256(doc.sha256),
                class_name=doc.__class__.__name__,
                name=doc.name,
                description=doc.description or "",
                sources=doc.sources,
                origins=doc.origins,
                summary=self._summaries.get(doc.sha256, ""),
            )
            for sha in sha256s
            if (doc := self._documents.get(sha)) is not None
        ]

    async def load_summaries(self, document_sha256s: list[DocumentSha256]) -> dict[DocumentSha256, str]:
        """Load summaries by SHA256."""
        return {sha: self._summaries[sha] for sha in document_sha256s if sha in self._summaries}

    async def save(self, document: Document, run_scope: RunScope) -> None:
        """Store document in memory, keyed by SHA256."""
        is_new = document.sha256 not in self._documents
        if is_new:
            self._documents[document.sha256] = document
        self._run_docs.setdefault(run_scope, set()).add(document.sha256)
        if is_new and self._summary_worker:
            self._summary_worker.schedule(document)

    async def save_batch(self, documents: list[Document], run_scope: RunScope) -> None:
        """Save multiple documents sequentially."""
        for doc in documents:
            await self.save(doc, run_scope)

    def shutdown(self) -> None:
        """Flush pending summaries and stop the summary worker."""
        if self._summary_worker:
            self._summary_worker.shutdown()

    async def update_summary(self, document_sha256: DocumentSha256, summary: str) -> None:
        """Update summary for a stored document. No-op if document doesn't exist."""
        if document_sha256 not in self._documents:
            return
        self._summaries[document_sha256] = summary


@runtime_checkable
class DocumentStore(Protocol):
    """Protocol for document storage backends.

Implementations: ClickHouseDocumentStore (production), LocalDocumentStore (CLI/debug),
MemoryDocumentStore (testing)."""
    async def check_existing(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Return the subset of sha256s that already exist in the store."""
        ...

    def flush(self) -> None:
        """Block until all pending background work (summaries) is processed."""
        ...

    async def has_documents(self, run_scope: RunScope, document_type: type[Document]) -> bool:
        """Check if any documents of this type exist in the run scope."""
        ...

    async def load(self, run_scope: RunScope, document_types: list[type[Document]]) -> list[Document]:
        """Load all documents of the given types from a run scope."""
        ...

    async def load_by_sha256s(self, sha256s: list[DocumentSha256], document_type: type[_D], run_scope: RunScope | None = None) -> dict[DocumentSha256, _D]:
        """Batch-load full documents by SHA256.

        document_type is used for construction only — class_name is not enforced as a filter.
        When run_scope is provided, only returns documents belonging to that scope.
        When run_scope is None, searches across all scopes (cross-pipeline lookups).
        Returns {sha256: document} for found documents. Missing SHA256s are omitted.
        """
        ...

    async def load_nodes_by_sha256s(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentNode]:
        """Batch-load lightweight metadata for documents by SHA256, searching all scopes.

        Returns {sha256: DocumentNode} for found documents. Missing SHA256s are omitted.
        No content or attachments loaded. No document type required.
        """
        ...

    async def load_scope_metadata(self, run_scope: RunScope) -> list[DocumentNode]:
        """Load lightweight metadata for ALL documents in a run scope.

        No content or attachments loaded.
        """
        ...

    async def load_summaries(self, document_sha256s: list[DocumentSha256]) -> dict[DocumentSha256, str]:
        """Load summaries by SHA256. Returns {sha256: summary} for docs that have summaries."""
        ...

    async def save(self, document: Document, run_scope: RunScope) -> None:
        """Save a single document to the store. Idempotent — same SHA256 is a no-op."""
        ...

    async def save_batch(self, documents: list[Document], run_scope: RunScope) -> None:
        """Save multiple documents. Dependencies must be sorted (caller's responsibility)."""
        ...

    def shutdown(self) -> None:
        """Flush pending work and stop background workers."""
        ...

    async def update_summary(self, document_sha256: DocumentSha256, summary: str) -> None:
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
# Source: tests/document_store/test_local.py:380
def test_create_document_store_returns_local_when_no_clickhouse(self):
    from ai_pipeline_core.settings import Settings

    settings = Settings(clickhouse_host="")
    store = create_document_store(settings)
    assert isinstance(store, LocalDocumentStore)

# Example: Get document store returns none by default
# Source: tests/document_store/test_protocol.py:83
def test_get_document_store_returns_none_by_default():
    """Before any set call, the store is None."""
    assert get_document_store() is None

# Example: Set document store to none
# Source: tests/document_store/test_protocol.py:95
def test_set_document_store_to_none():
    """Store can be reset to None."""
    store = _DummyStore()
    set_document_store(store)
    assert get_document_store() is store
    set_document_store(None)
    assert get_document_store() is None

# Example: Set and get document store
# Source: tests/document_store/test_protocol.py:88
def test_set_and_get_document_store():
    """Setting a store makes it retrievable."""
    store = _DummyStore()
    set_document_store(store)
    assert get_document_store() is store

# === ERROR EXAMPLES (What NOT to Do) ===

# Error: Create document store rejects non settings
# Source: tests/document_store/test_local.py:387
def test_create_document_store_rejects_non_settings(self):
    with pytest.raises(AttributeError):
        create_document_store("not_settings")  # type: ignore[arg-type]

# Error: Primary flush failure still attempts secondary
# Source: tests/document_store/test_dual_store.py:181
def test_primary_flush_failure_still_attempts_secondary(self):
    flushed = []

    class FailingFlush(MemoryDocumentStore):
        def flush(self) -> None:
            raise RuntimeError("primary flush failed")

    class TrackingFlush(MemoryDocumentStore):
        def flush(self) -> None:
            flushed.append(True)

    dual = DualDocumentStore(primary=FailingFlush(), secondary=TrackingFlush())
    with pytest.raises(RuntimeError, match="primary flush failed"):
        dual.flush()
    assert flushed == [True]

# Error: Primary save failure propagates
# Source: tests/document_store/test_dual_store.py:152
@pytest.mark.asyncio
async def test_primary_save_failure_propagates(self):
    class FailingSave(MemoryDocumentStore):
        async def save(self, document: Document, run_scope: str) -> None:
            raise RuntimeError("primary down")

    secondary = MemoryDocumentStore()
    dual = DualDocumentStore(primary=FailingSave(), secondary=secondary)
    doc = _make("a.md", "content")
    with pytest.raises(RuntimeError, match="primary down"):
        await dual.save(doc, RunScope("run1"))
    # Secondary should NOT have been called since primary failed first
    assert not await secondary.has_documents(RunScope("run1"), DualReportDoc)

# Error: Primary update summary failure still attempts secondary
# Source: tests/document_store/test_dual_store.py:166
@pytest.mark.asyncio
async def test_primary_update_summary_failure_still_attempts_secondary(self):
    class FailingSummary(MemoryDocumentStore):
        async def update_summary(self, document_sha256: str, summary: str) -> None:
            raise RuntimeError("primary down")

    secondary = MemoryDocumentStore()
    dual = DualDocumentStore(primary=FailingSummary(), secondary=secondary)
    doc = _make("a.md", "content")
    sha = compute_document_sha256(doc)
    await secondary.save(doc, RunScope("run1"))
    with pytest.raises(RuntimeError, match="primary down"):
        await dual.update_summary(sha, "summary")
    # Secondary should still have been updated
    assert (await secondary.load_summaries([sha]))[sha] == "summary"
