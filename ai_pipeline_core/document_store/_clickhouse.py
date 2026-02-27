"""ClickHouse-backed document store for production use.

Four-table schema: document_content (deduplicated blobs), document_index
(global document metadata), run_documents (run membership mapping), and
flow_completions (per-flow resume records).
Uses ReplacingMergeTree for idempotent writes.
Circuit breaker buffers writes when ClickHouse is unavailable.
"""

import asyncio
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar, cast

import clickhouse_connect

from ai_pipeline_core.document_store._models import DocumentNode, FlowCompletion
from ai_pipeline_core.document_store._summary_worker import SummaryGenerator, SummaryWorker
from ai_pipeline_core.documents._context import DocumentSha256, RunScope, _suppress_document_registration
from ai_pipeline_core.documents._hashing import compute_content_sha256, compute_document_sha256
from ai_pipeline_core.documents.attachment import Attachment
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

__all__ = [
    "ClickHouseDocumentStore",
]

_D = TypeVar("_D", bound=Document)

_MAX_BUFFER_SIZE = 10_000
_BUFFER_WARNING_THRESHOLD = int(_MAX_BUFFER_SIZE * 0.9)
_BUFFER_RETRY_INTERVAL = 10  # seconds between buffer flush attempts when not recovering from circuit open
_RECONNECT_INTERVAL_SEC = 60
_FAILURE_THRESHOLD = 3
_MAX_BUFFER_RETRIES = 3
_SHUTDOWN_FLUSH_TIMEOUT = 30

# --- DDL: table names and CREATE TABLE statements ---

TABLE_DOCUMENT_CONTENT = "document_content"
TABLE_DOCUMENT_INDEX = "document_index"
TABLE_RUN_DOCUMENTS = "run_documents"
TABLE_FLOW_COMPLETIONS = "flow_completions"

DDL_CONTENT = f"""
CREATE TABLE IF NOT EXISTS {TABLE_DOCUMENT_CONTENT}
(
    content_sha256     String,
    content            String CODEC(ZSTD(3)),
    stored_at          DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree()
ORDER BY (content_sha256)
SETTINGS index_granularity = 8192
"""

DDL_INDEX = f"""
CREATE TABLE IF NOT EXISTS {TABLE_DOCUMENT_INDEX}
(
    document_sha256        String,
    content_sha256         String,
    class_name             LowCardinality(String),
    name                   String,
    description            String DEFAULT '',
    mime_type              LowCardinality(String),
    derived_from           Array(String),
    triggered_by           Array(String),
    attachment_names        Array(String),
    attachment_descriptions Array(String),
    attachment_sha256s      Array(String),
    summary                String DEFAULT '' CODEC(ZSTD(3)),
    stored_at              DateTime64(3, 'UTC'),
    version                UInt64 DEFAULT 1,

    INDEX idx_derived_from derived_from TYPE bloom_filter GRANULARITY 1
)
ENGINE = ReplacingMergeTree(version)
ORDER BY (document_sha256)
SETTINGS index_granularity = 8192
"""

DDL_RUN_DOCUMENTS = f"""
CREATE TABLE IF NOT EXISTS {TABLE_RUN_DOCUMENTS}
(
    run_scope          LowCardinality(String),
    document_sha256    String,
    class_name         LowCardinality(String),
    stored_at          DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree()
ORDER BY (run_scope, document_sha256, class_name)
SETTINGS index_granularity = 8192
"""

DDL_FLOW_COMPLETIONS = f"""
CREATE TABLE IF NOT EXISTS {TABLE_FLOW_COMPLETIONS}
(
    run_scope          LowCardinality(String),
    flow_name          String,
    input_sha256s      Array(String),
    output_sha256s     Array(String),
    stored_at          DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree()
ORDER BY (run_scope, flow_name)
SETTINGS index_granularity = 8192
"""


@dataclass(slots=True)
class _BufferedWrite:
    """A pending write operation buffered during circuit breaker open state."""

    document: Document
    run_scope: RunScope
    retry_count: int = 0


def _reconstruct_attachments(
    att_names: list[str],
    att_descs: list[str],
    att_sha256s: list[str],
    att_content_by_sha: dict[str, bytes],
    doc_name: str,
) -> tuple[Attachment, ...]:
    """Reconstruct Attachment objects from parallel arrays and a content lookup dict."""
    if not att_sha256s:
        return ()
    att_list: list[Attachment] = []
    for a_name, a_desc, a_sha in zip(att_names, att_descs, att_sha256s, strict=True):
        a_content = att_content_by_sha.get(a_sha)
        if a_content is None:
            logger.warning("Attachment content %s... not found for document '%s'", a_sha[:12], doc_name)
            continue
        att_list.append(Attachment(name=a_name, content=a_content, description=a_desc or None))
    return tuple(att_list)


@dataclass(frozen=True, slots=True)
class _ParsedDocumentRow:
    """Decoded fields shared by all ClickHouse document load paths."""

    name: str
    description: str | None
    derived_from: tuple[str, ...]
    triggered_by: tuple[str, ...]
    att_names: list[str]
    att_descs: list[str]
    att_sha256s: list[str]
    content: bytes


def _parse_document_row(fields: tuple[Any, ...]) -> _ParsedDocumentRow:
    """Decode raw ClickHouse row fields (name through content_length) into typed Python values.

    Expects a 9-element tuple: (name, description, derived_from, triggered_by,
    att_names, att_descs, att_sha256s, content, content_length).
    """
    name_raw, description_raw, derived_from_raw, triggered_by_raw, att_names_raw, att_descs_raw, att_sha256s_raw, content_raw, content_length = fields
    return _ParsedDocumentRow(
        name=_decode(name_raw),
        description=_decode(description_raw) or None,
        derived_from=tuple(_decode(s) for s in derived_from_raw),
        triggered_by=tuple(_decode(o) for o in triggered_by_raw),
        att_names=[_decode(n) for n in att_names_raw],
        att_descs=[_decode(d) for d in att_descs_raw],
        att_sha256s=[_decode(s) for s in att_sha256s_raw],
        content=_decode_content(content_raw, content_length),
    )


def _build_document(
    doc_type: type[Document],
    row: _ParsedDocumentRow,
    att_content_by_sha: dict[str, bytes],
) -> Document:
    """Reconstruct a Document from parsed row fields and attachment content."""
    attachments = _reconstruct_attachments(row.att_names, row.att_descs, row.att_sha256s, att_content_by_sha, row.name)
    return doc_type(
        name=row.name,
        content=row.content,
        description=row.description,
        derived_from=row.derived_from,
        triggered_by=cast(tuple[DocumentSha256, ...], row.triggered_by) if row.triggered_by else (),
        attachments=attachments if attachments else None,
    )


def _parse_node_row(row: tuple[Any, ...]) -> DocumentNode:
    """Parse a ClickHouse row (document_sha256..summary) into a DocumentNode."""
    doc_sha256_raw, class_name_raw, name_raw, description_raw, derived_from_raw, triggered_by_raw, summary_raw = row
    doc_sha = DocumentSha256(_decode(doc_sha256_raw))
    return DocumentNode(
        sha256=doc_sha,
        class_name=_decode(class_name_raw),
        name=_decode(name_raw),
        description=_decode(description_raw) or "",
        derived_from=tuple(_decode(s) for s in derived_from_raw),
        triggered_by=tuple(_decode(o) for o in triggered_by_raw),
        summary=_decode(summary_raw) if summary_raw else "",
    )


class ClickHouseDocumentStore:
    """ClickHouse-backed document store with circuit breaker.

    All sync operations run on a single-thread executor (max_workers=1),
    so circuit breaker state needs no locking. Async methods dispatch to
    this executor via loop.run_in_executor().
    """

    def __init__(
        self,
        *,
        host: str,
        port: int = 8443,
        database: str = "default",
        username: str = "default",
        password: str = "",
        secure: bool = True,
        connect_timeout: int = 10,
        send_receive_timeout: int = 30,
        summary_generator: SummaryGenerator | None = None,
    ) -> None:
        self._params = {
            "host": host,
            "port": port,
            "database": database,
            "username": username,
            "password": password,
            "secure": secure,
            "connect_timeout": connect_timeout,
            "send_receive_timeout": send_receive_timeout,
        }
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ch-docstore")
        self._client: Any = None
        self._tables_initialized = False

        # Circuit breaker state (accessed only from the single executor thread)
        self._consecutive_failures = 0
        self._circuit_open = False
        self._last_reconnect_attempt = 0.0
        self._last_flush_attempt = 0.0
        self._buffer: deque[_BufferedWrite] = deque(maxlen=_MAX_BUFFER_SIZE)

        # Summary worker
        self._summary_worker: SummaryWorker | None = None
        if summary_generator:
            self._summary_worker = SummaryWorker(
                generator=summary_generator,
                update_fn=self.update_summary,
            )
            self._summary_worker.start()

    async def _run(self, fn: Any, *args: Any) -> Any:
        """Run a sync function on the dedicated executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    # --- Connection management (sync, executor thread only) ---

    def _connect(self) -> None:
        self._client = clickhouse_connect.get_client(  # pyright: ignore[reportUnknownMemberType]
            **self._params,  # pyright: ignore[reportArgumentType]
        )
        logger.info("Document store connected to ClickHouse at %s:%s", self._params["host"], self._params["port"])

    def _ensure_tables(self) -> None:
        if self._tables_initialized:
            return
        if self._client is None:
            self._connect()
        self._client.command(DDL_CONTENT)
        self._client.command(DDL_INDEX)
        self._client.command(DDL_RUN_DOCUMENTS)
        self._client.command(DDL_FLOW_COMPLETIONS)
        self._tables_initialized = True
        logger.info("Document store tables verified/created")

    def _ensure_connected(self) -> bool:
        try:
            if self._client is None:
                self._connect()
            self._ensure_tables()
            return True
        except Exception as e:
            logger.warning("ClickHouse connection failed: %s", e)
            self._client = None
            self._tables_initialized = False
            return False

    def _try_reconnect(self) -> bool:
        now = time.monotonic()
        if now - self._last_reconnect_attempt < _RECONNECT_INTERVAL_SEC:
            return False
        self._last_reconnect_attempt = now
        if self._ensure_connected():
            self._circuit_open = False
            self._consecutive_failures = 0
            logger.info("ClickHouse reconnected, flushing buffer")
            return True
        return False

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        if self._circuit_open:
            self._circuit_open = False
            logger.info("Circuit breaker closed")

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= _FAILURE_THRESHOLD and not self._circuit_open:
            self._circuit_open = True
            self._client = None
            self._tables_initialized = False
            logger.warning("Circuit breaker opened after %d failures", self._consecutive_failures)

    # --- Async public API ---

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

    async def load(self, run_scope: RunScope, document_types: list[type[Document]]) -> list[Document]:
        """Load documents via run_documents + index JOIN content, then batch-fetch attachments."""
        return await self._run(self._load_sync, run_scope, document_types)

    async def has_documents(self, run_scope: RunScope, document_type: type[Document], *, max_age: timedelta | None = None) -> bool:
        """Check if any documents of this type exist in the run scope."""
        return await self._run(self._has_documents_sync, run_scope, document_type, max_age)

    async def check_existing(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Return the subset of sha256s that exist in the document index."""
        return await self._run(self._check_existing_sync, sha256s)

    async def update_summary(self, document_sha256: DocumentSha256, summary: str) -> None:
        """Update summary via INSERT with incremented version (ReplacingMergeTree dedup)."""
        await self._run(self._update_summary_sync, document_sha256, summary)

    async def load_summaries(self, document_sha256s: list[DocumentSha256]) -> dict[DocumentSha256, str]:
        """Load summaries by SHA256 from the document index."""
        return await self._run(self._load_summaries_sync, document_sha256s)

    async def load_by_sha256s(self, sha256s: list[DocumentSha256], document_type: type[_D], run_scope: RunScope | None = None) -> dict[DocumentSha256, _D]:
        """Batch-load documents by SHA256 and type. Verifies run membership when run_scope provided."""
        return await self._run(self._load_by_sha256s_sync, sha256s, document_type, run_scope)

    async def load_nodes_by_sha256s(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentNode]:
        """Batch-load lightweight metadata by SHA256. No JOIN needed."""
        return await self._run(self._load_nodes_by_sha256s_sync, sha256s)

    async def load_scope_metadata(self, run_scope: RunScope) -> list[DocumentNode]:
        """Load lightweight metadata for all documents in a run scope."""
        return await self._run(self._load_scope_metadata_sync, run_scope)

    async def find_by_source(
        self,
        source_values: list[str],
        document_type: type[Document],
        *,
        max_age: timedelta | None = None,
    ) -> dict[str, Document]:
        """Find most recent document per source value via derived_from array lookup."""
        return await self._run(self._find_by_source_sync, source_values, document_type, max_age)

    async def save_flow_completion(
        self,
        run_scope: RunScope,
        flow_name: str,
        input_sha256s: tuple[str, ...],
        output_sha256s: tuple[str, ...],
    ) -> None:
        """Save flow completion record to ClickHouse."""
        await self._run(self._save_flow_completion_sync, run_scope, flow_name, input_sha256s, output_sha256s)

    async def get_flow_completion(
        self,
        run_scope: RunScope,
        flow_name: str,
        *,
        max_age: timedelta | None = None,
    ) -> FlowCompletion | None:
        """Load flow completion record from ClickHouse."""
        return await self._run(self._get_flow_completion_sync, run_scope, flow_name, max_age)

    def flush(self) -> None:
        """Block until all pending document summaries are processed."""
        if self._summary_worker:
            self._summary_worker.flush()

    def shutdown(self) -> None:
        """Flush pending buffer and summaries, stop workers, and release the executor."""
        if self._summary_worker:
            self._summary_worker.shutdown()
        if self._buffer:
            try:
                future = self._executor.submit(self._flush_buffer)
                future.result(timeout=_SHUTDOWN_FLUSH_TIMEOUT)
            except RuntimeError:
                logger.warning("Could not flush document buffer on shutdown — executor already shut down")
            except TimeoutError:
                logger.warning("Document buffer flush timed out on shutdown, %d documents undelivered", len(self._buffer))
            except Exception as e:
                logger.warning("Document buffer flush failed on shutdown (%d documents undelivered): %s", len(self._buffer), e)
        self._executor.shutdown(wait=False)

    # --- Sync implementations (executor thread only) ---

    def _buffer_append(self, document: Document, run_scope: RunScope) -> None:
        """Append a document to the retry buffer with overflow warning."""
        buf_len = len(self._buffer)
        if buf_len >= _MAX_BUFFER_SIZE:
            logger.warning(
                "Document buffer full (%d items), dropping oldest document to make room for '%s'",
                buf_len,
                document.name,
            )
        elif buf_len >= _BUFFER_WARNING_THRESHOLD:
            logger.warning("Document buffer at %d/%d capacity", buf_len, _MAX_BUFFER_SIZE)
        self._buffer.append(_BufferedWrite(document=document, run_scope=run_scope))

    def _buffer_flush_due(self) -> bool:
        """Check if enough time has passed since the last buffer flush attempt."""
        now = time.monotonic()
        if now - self._last_flush_attempt < _BUFFER_RETRY_INTERVAL:
            return False
        self._last_flush_attempt = now
        return True

    def _save_sync(self, document: Document, run_scope: RunScope) -> None:
        if self._circuit_open:
            if not self._try_reconnect():
                self._buffer_append(document, run_scope)
                return
            self._flush_buffer()
        elif self._buffer and self._buffer_flush_due():
            self._flush_buffer()

        try:
            self._ensure_tables()
            self._insert_document(document, run_scope)
            self._record_success()
        except Exception as e:
            logger.warning("Failed to save document '%s': %s", document.name, e)
            self._record_failure()
            self._buffer_append(document, run_scope)

    def _save_batch_sync(self, documents: list[Document], run_scope: RunScope) -> None:
        if self._circuit_open:
            if not self._try_reconnect():
                for doc in documents:
                    self._buffer_append(doc, run_scope)
                return
            self._flush_buffer()
        elif self._buffer and self._buffer_flush_due():
            self._flush_buffer()

        try:
            self._ensure_tables()
            self._insert_documents_batch(documents, run_scope)
            self._record_success()
        except Exception as e:
            logger.warning("Failed to save batch of %d documents: %s", len(documents), e)
            self._record_failure()
            for doc in documents:
                self._buffer_append(doc, run_scope)

    def _rebuffer_failed_items(self, items: list[_BufferedWrite], error: Exception) -> None:
        """Increment retry_count per item, re-buffer or drop individually."""
        for it in items:
            it.retry_count += 1
            if it.retry_count >= _MAX_BUFFER_RETRIES:
                logger.warning("Dropping document '%s' after %d failed flush attempts: %s", it.document.name, it.retry_count, error)
            else:
                logger.warning("Failed to flush buffered document '%s' (attempt %d): %s", it.document.name, it.retry_count, error)
                self._buffer.append(it)

    def _flush_buffer(self) -> None:
        """Drain buffer in chunks, grouped by run_scope. Per-item retry tracking."""
        FLUSH_CHUNK_SIZE = 100
        while self._buffer:
            chunk: list[_BufferedWrite] = []
            while self._buffer and len(chunk) < FLUSH_CHUNK_SIZE:
                chunk.append(self._buffer.popleft())

            groups: dict[RunScope, list[_BufferedWrite]] = {}
            for item in chunk:
                groups.setdefault(item.run_scope, []).append(item)

            failed = False
            for run_scope, items in groups.items():
                if failed:
                    # Re-buffer unprocessed groups from this chunk (preserving retry_count)
                    for it in items:
                        self._buffer.append(it)
                    continue
                try:
                    self._insert_documents_batch([it.document for it in items], run_scope)
                    if self._summary_worker:
                        for it in items:
                            self._summary_worker.schedule(it.document)
                except Exception as e:
                    failed = True
                    self._rebuffer_failed_items(items, e)

            if failed:
                break  # Stop flushing — retry on next throttled drain or shutdown

    def _insert_document(self, document: Document, run_scope: RunScope) -> None:
        """Insert a single document. Thin wrapper around batch insert."""
        self._insert_documents_batch([document], run_scope)

    def _insert_documents_batch(self, documents: list[Document], run_scope: RunScope) -> None:
        """Insert multiple documents in exactly 3 client.insert() calls (one per table).

        Content rows are deduplicated by content_sha256 within the batch.
        """
        now = datetime.now(UTC)
        content_rows: list[list[Any]] = []
        index_rows: list[list[Any]] = []
        run_doc_rows: list[list[Any]] = []
        seen_content: set[str] = set()

        for document in documents:
            doc_sha256 = compute_document_sha256(document)
            content_sha256 = compute_content_sha256(document.content)

            # Content blob (deduplicated)
            if content_sha256 not in seen_content:
                content_rows.append([content_sha256, document.content, now])
                seen_content.add(content_sha256)

            # Attachment content
            att_names: list[str] = []
            att_descriptions: list[str] = []
            att_sha256s: list[str] = []
            for att in sorted(document.attachments, key=lambda a: a.name):
                att_sha = compute_content_sha256(att.content)
                att_names.append(att.name)
                att_descriptions.append(att.description or "")
                att_sha256s.append(att_sha)
                if att_sha not in seen_content:
                    content_rows.append([att_sha, att.content, now])
                    seen_content.add(att_sha)

            # document_index row
            index_rows.append([
                doc_sha256,
                content_sha256,
                document.__class__.__name__,
                document.name,
                document.description or "",
                document.mime_type,
                list(document.derived_from),
                list(document.triggered_by),
                att_names,
                att_descriptions,
                att_sha256s,
                "",
                now,
                1,
            ])

            # run_documents row
            run_doc_rows.append([run_scope, doc_sha256, document.__class__.__name__, now])

        # 3 inserts regardless of batch size
        if content_rows:
            self._client.insert(
                TABLE_DOCUMENT_CONTENT,
                content_rows,
                column_names=["content_sha256", "content", "stored_at"],
            )
        if index_rows:
            self._client.insert(
                TABLE_DOCUMENT_INDEX,
                index_rows,
                column_names=[
                    "document_sha256",
                    "content_sha256",
                    "class_name",
                    "name",
                    "description",
                    "mime_type",
                    "derived_from",
                    "triggered_by",
                    "attachment_names",
                    "attachment_descriptions",
                    "attachment_sha256s",
                    "summary",
                    "stored_at",
                    "version",
                ],
            )
        if run_doc_rows:
            self._client.insert(
                TABLE_RUN_DOCUMENTS,
                run_doc_rows,
                column_names=["run_scope", "document_sha256", "class_name", "stored_at"],
            )

    def _load_sync(self, run_scope: RunScope, document_types: list[type[Document]]) -> list[Document]:
        """Three-table load: run_documents → document_index → document_content."""
        self._ensure_tables()

        type_by_name: dict[str, type[Document]] = {t.__name__: t for t in document_types}
        class_names = list(type_by_name.keys())

        # Query: join run_documents → document_index → document_content
        rows = self._client.query(
            f"SELECT rd.class_name, di.name, di.description, di.derived_from, di.triggered_by, "
            f"di.attachment_names, di.attachment_descriptions, di.attachment_sha256s, "
            f"dc.content, length(dc.content) "
            f"FROM {TABLE_RUN_DOCUMENTS} AS rd FINAL "
            f"JOIN {TABLE_DOCUMENT_INDEX} AS di FINAL ON rd.document_sha256 = di.document_sha256 "
            f"JOIN {TABLE_DOCUMENT_CONTENT} AS dc ON di.content_sha256 = dc.content_sha256 "
            f"WHERE rd.run_scope = {{run_scope:String}} "
            f"AND rd.class_name IN {{class_names:Array(String)}}",
            parameters={"run_scope": run_scope, "class_names": class_names},
        )

        # Parse and decode all rows, collecting attachment SHA256s
        parsed_rows: list[tuple[type[Document], _ParsedDocumentRow]] = []
        all_att_sha256s: set[str] = set()

        for row in rows.result_rows:
            class_name_raw, *rest = row
            class_name = _decode(class_name_raw)
            doc_type = type_by_name.get(class_name)
            if doc_type is None:
                logger.warning("Unknown document class_name '%s' in run_scope '%s', skipping row", class_name, run_scope)
                continue
            parsed = _parse_document_row(tuple(rest))
            all_att_sha256s.update(parsed.att_sha256s)
            parsed_rows.append((doc_type, parsed))

        att_content_by_sha = self._fetch_attachment_content(all_att_sha256s)

        # Reconstruct documents (suppress registration — these are deserialized, not new)
        documents: list[Document] = []
        with _suppress_document_registration():
            for doc_type, parsed in parsed_rows:
                documents.append(_build_document(doc_type, parsed, att_content_by_sha))

        return documents

    def _has_documents_sync(self, run_scope: RunScope, document_type: type[Document], max_age: timedelta | None = None) -> bool:
        self._ensure_tables()
        params: dict[str, Any] = {"run_scope": run_scope, "class_name": document_type.__name__}
        age_filter = ""
        if max_age is not None:
            params["cutoff"] = datetime.now(UTC) - max_age
            age_filter = " AND rd.stored_at >= {cutoff:DateTime64(3, 'UTC')}"

        expected_files = document_type.get_expected_files()
        if expected_files is None:
            # No FILES enum — any document of this type is sufficient
            result = self._client.query(
                f"SELECT 1 FROM {TABLE_RUN_DOCUMENTS} AS rd FINAL "
                f"WHERE rd.run_scope = {{run_scope:String}} AND rd.class_name = {{class_name:String}}{age_filter} LIMIT 1",
                parameters=params,
            )
            return len(result.result_rows) > 0

        # FILES enum defined — all expected filenames must be present
        params["expected_names"] = expected_files
        result = self._client.query(
            f"SELECT count(DISTINCT di.name) "
            f"FROM {TABLE_RUN_DOCUMENTS} AS rd FINAL "
            f"JOIN {TABLE_DOCUMENT_INDEX} AS di FINAL ON rd.document_sha256 = di.document_sha256 "
            f"WHERE rd.run_scope = {{run_scope:String}} AND rd.class_name = {{class_name:String}} "
            f"AND di.name IN {{expected_names:Array(String)}}{age_filter}",
            parameters=params,
        )
        found_count = result.result_rows[0][0] if result.result_rows else 0
        return found_count >= len(expected_files)

    def _check_existing_sync(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        if not sha256s:
            return set()
        self._ensure_tables()
        result = self._client.query(
            f"SELECT document_sha256 FROM {TABLE_DOCUMENT_INDEX} FINAL WHERE document_sha256 IN {{sha256s:Array(String)}}",
            parameters={"sha256s": sha256s},
        )
        return {DocumentSha256(_decode(row[0])) for row in result.result_rows}

    def _update_summary_sync(self, document_sha256: DocumentSha256, summary: str) -> None:
        """Update summary via INSERT with timestamp-based version (ReplacingMergeTree dedup).

        Single-query INSERT...SELECT copies all fields from existing row, sets new summary
        and uses millisecond timestamp as version to avoid read+increment race conditions.
        """
        try:
            self._ensure_tables()
            version = int(datetime.now(UTC).timestamp() * 1000)
            self._client.command(
                f"INSERT INTO {TABLE_DOCUMENT_INDEX} "
                "(document_sha256, content_sha256, class_name, name, description, mime_type, "
                "derived_from, triggered_by, attachment_names, attachment_descriptions, attachment_sha256s, "
                "summary, stored_at, version) "
                f"SELECT document_sha256, content_sha256, class_name, name, description, mime_type, "
                f"derived_from, triggered_by, attachment_names, attachment_descriptions, attachment_sha256s, "
                f"{{summary:String}}, stored_at, {{version:UInt64}} "
                f"FROM {TABLE_DOCUMENT_INDEX} FINAL "
                f"WHERE document_sha256 = {{sha256:String}}",
                parameters={
                    "summary": summary,
                    "sha256": document_sha256,
                    "version": version,
                },
            )
        except Exception as e:
            logger.warning("Failed to update summary for %s...: %s", document_sha256[:12], e)

    def _load_summaries_sync(self, document_sha256s: list[DocumentSha256]) -> dict[DocumentSha256, str]:
        """Query summaries from the document index (global, no run_scope needed)."""
        if not document_sha256s:
            return {}
        try:
            self._ensure_tables()
            result = self._client.query(
                f"SELECT document_sha256, summary FROM {TABLE_DOCUMENT_INDEX} FINAL WHERE document_sha256 IN {{sha256s:Array(String)}} AND summary != ''",
                parameters={"sha256s": document_sha256s},
            )
            return {DocumentSha256(_decode(row[0])): _decode(row[1]) for row in result.result_rows}
        except Exception as e:
            logger.warning("Failed to load summaries: %s", e)
            return {}

    def _load_by_sha256s_sync(self, sha256s: list[DocumentSha256], document_type: type[Document], run_scope: RunScope | None) -> dict[DocumentSha256, Document]:
        """Batch lookup by SHA256. class_name is not enforced — document_type is for construction only."""
        if not sha256s:
            return {}
        self._ensure_tables()

        if run_scope is not None:
            # 3-table JOIN: run_documents → document_index → document_content
            rows = self._client.query(
                f"SELECT di.document_sha256, di.class_name, di.name, di.description, di.derived_from, di.triggered_by, "
                f"di.attachment_names, di.attachment_descriptions, di.attachment_sha256s, "
                f"dc.content, length(dc.content) "
                f"FROM {TABLE_RUN_DOCUMENTS} AS rd FINAL "
                f"JOIN {TABLE_DOCUMENT_INDEX} AS di FINAL ON rd.document_sha256 = di.document_sha256 "
                f"JOIN {TABLE_DOCUMENT_CONTENT} AS dc ON di.content_sha256 = dc.content_sha256 "
                f"WHERE rd.run_scope = {{run_scope:String}} AND rd.document_sha256 IN {{sha256s:Array(String)}}",
                parameters={"run_scope": run_scope, "sha256s": sha256s},
            )
        else:
            # Cross-scope: document_index → document_content (no run membership check)
            rows = self._client.query(
                f"SELECT di.document_sha256, di.class_name, di.name, di.description, di.derived_from, di.triggered_by, "
                f"di.attachment_names, di.attachment_descriptions, di.attachment_sha256s, "
                f"dc.content, length(dc.content) "
                f"FROM {TABLE_DOCUMENT_INDEX} AS di FINAL "
                f"JOIN {TABLE_DOCUMENT_CONTENT} AS dc ON di.content_sha256 = dc.content_sha256 "
                f"WHERE di.document_sha256 IN {{sha256s:Array(String)}}",
                parameters={"sha256s": sha256s},
            )

        if not rows.result_rows:
            return {}

        # Parse and decode all rows, collecting attachment SHA256s
        parsed_rows: list[tuple[DocumentSha256, _ParsedDocumentRow]] = []
        all_att_sha256s: set[str] = set()

        for row in rows.result_rows:
            doc_sha256_raw, _class_name, *rest = row
            parsed = _parse_document_row(tuple(rest))
            all_att_sha256s.update(parsed.att_sha256s)
            parsed_rows.append((DocumentSha256(_decode(doc_sha256_raw)), parsed))

        att_content_by_sha = self._fetch_attachment_content(all_att_sha256s)

        # Reconstruct documents (suppress registration — deserialized)
        result: dict[DocumentSha256, Document] = {}
        with _suppress_document_registration():
            for doc_sha256, parsed in parsed_rows:
                result[doc_sha256] = _build_document(document_type, parsed, att_content_by_sha)

        return result

    def _load_nodes_by_sha256s_sync(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentNode]:
        """Batch metadata lookup by SHA256 from global document_index. No JOIN."""
        if not sha256s:
            return {}
        self._ensure_tables()
        result = self._client.query(
            f"SELECT document_sha256, class_name, name, description, derived_from, triggered_by, summary "
            f"FROM {TABLE_DOCUMENT_INDEX} FINAL "
            f"WHERE document_sha256 IN {{sha256s:Array(String)}}",
            parameters={"sha256s": sha256s},
        )
        nodes: dict[DocumentSha256, DocumentNode] = {}
        for row in result.result_rows:
            node = _parse_node_row(row)
            nodes[node.sha256] = node
        return nodes

    def _load_scope_metadata_sync(self, run_scope: RunScope) -> list[DocumentNode]:
        """Join run_documents → document_index for metadata in a run scope."""
        self._ensure_tables()
        result = self._client.query(
            f"SELECT rd.document_sha256, rd.class_name, di.name, di.description, "
            f"di.derived_from, di.triggered_by, di.summary "
            f"FROM {TABLE_RUN_DOCUMENTS} AS rd FINAL "
            f"JOIN {TABLE_DOCUMENT_INDEX} AS di FINAL ON rd.document_sha256 = di.document_sha256 "
            f"WHERE rd.run_scope = {{run_scope:String}}",
            parameters={"run_scope": run_scope},
        )
        return [_parse_node_row(row) for row in result.result_rows]

    def _find_by_source_sync(self, source_values: list[str], document_type: type[Document], max_age: timedelta | None) -> dict[str, Document]:
        """Find most recent document per source value. Two-query pattern: metadata+content, then attachments."""
        if not source_values:
            return {}
        self._ensure_tables()

        parsed_rows, all_att_sha256s = self._query_by_source(source_values, document_type.__name__, max_age)
        if not parsed_rows:
            return {}

        att_content_by_sha = self._fetch_attachment_content(all_att_sha256s)
        return self._reconstruct_source_documents(parsed_rows, att_content_by_sha, document_type)

    def _query_by_source(
        self,
        source_values: list[str],
        class_name: str,
        max_age: timedelta | None,
    ) -> tuple[list[tuple[str, str, str | None, tuple[str, ...], tuple[str, ...], list[str], list[str], list[str], bytes]], set[str]]:
        """Query document_index for newest document per source value, joined with main content."""
        params: dict[str, Any] = {"class_name": class_name, "source_values": source_values}

        age_filter = ""
        if max_age is not None:
            params["cutoff"] = datetime.now(UTC) - max_age
            age_filter = "AND di.stored_at >= {cutoff:DateTime64(3, 'UTC')}"

        rows = self._client.query(
            f"WITH matched AS ("
            f"  SELECT"
            f"    arrayJoin(di.derived_from) AS matched_source,"
            f"    di.content_sha256,"
            f"    di.name, di.description, di.derived_from, di.triggered_by,"
            f"    di.attachment_names, di.attachment_descriptions, di.attachment_sha256s,"
            f"    ROW_NUMBER() OVER (PARTITION BY matched_source ORDER BY di.stored_at DESC, di.document_sha256 DESC) AS rn"
            f"  FROM {TABLE_DOCUMENT_INDEX} AS di FINAL"
            f"  WHERE di.class_name = {{class_name:String}}"
            f"  AND hasAny(di.derived_from, {{source_values:Array(String)}})"
            f"  {age_filter}"
            f")"
            f" SELECT m.matched_source, m.name, m.description, m.derived_from, m.triggered_by,"
            f"   m.attachment_names, m.attachment_descriptions, m.attachment_sha256s,"
            f"   dc.content, length(dc.content)"
            f" FROM matched AS m"
            f" JOIN {TABLE_DOCUMENT_CONTENT} AS dc ON m.content_sha256 = dc.content_sha256"
            f" WHERE m.rn = 1 AND m.matched_source IN {{source_values:Array(String)}}",
            parameters=params,
        )

        source_set = set(source_values)
        parsed: list[tuple[str, str, str | None, tuple[str, ...], tuple[str, ...], list[str], list[str], list[str], bytes]] = []
        all_att_sha256s: set[str] = set()

        for row in rows.result_rows:
            (
                matched_source_raw,
                name_raw,
                desc_raw,
                derived_from_raw,
                triggered_by_raw,
                att_names_raw,
                att_descs_raw,
                att_sha256s_raw,
                content_raw,
                content_len,
            ) = row
            matched_source = _decode(matched_source_raw)
            if matched_source not in source_set:
                continue
            att_sha256s = [_decode(s) for s in att_sha256s_raw]
            all_att_sha256s.update(att_sha256s)
            parsed.append((
                matched_source,
                _decode(name_raw),
                _decode(desc_raw) or None,
                tuple(_decode(s) for s in derived_from_raw),
                tuple(_decode(o) for o in triggered_by_raw),
                [_decode(n) for n in att_names_raw],
                [_decode(d) for d in att_descs_raw],
                att_sha256s,
                _decode_content(content_raw, content_len),
            ))

        return parsed, all_att_sha256s

    def _fetch_attachment_content(self, att_sha256s: set[str]) -> dict[str, bytes]:
        """Batch-fetch attachment content blobs by SHA256."""
        if not att_sha256s:
            return {}
        att_rows = self._client.query(
            f"SELECT content_sha256, content, length(content) FROM {TABLE_DOCUMENT_CONTENT} WHERE content_sha256 IN {{sha256s:Array(String)}}",
            parameters={"sha256s": list(att_sha256s)},
        )
        return {_decode(row[0]): _decode_content(row[1], row[2]) for row in att_rows.result_rows}

    def _save_flow_completion_sync(
        self,
        run_scope: RunScope,
        flow_name: str,
        input_sha256s: tuple[str, ...],
        output_sha256s: tuple[str, ...],
    ) -> None:
        """Insert flow completion record into ClickHouse (ReplacingMergeTree overwrites by key)."""
        self._ensure_tables()
        now = datetime.now(UTC)
        self._client.insert(
            TABLE_FLOW_COMPLETIONS,
            [[run_scope, flow_name, list(input_sha256s), list(output_sha256s), now]],
            column_names=["run_scope", "flow_name", "input_sha256s", "output_sha256s", "stored_at"],
        )

    def _get_flow_completion_sync(self, run_scope: RunScope, flow_name: str, max_age: timedelta | None) -> FlowCompletion | None:
        """Query ClickHouse for a flow completion record."""
        self._ensure_tables()
        params: dict[str, Any] = {"run_scope": run_scope, "flow_name": flow_name}
        age_filter = ""
        if max_age is not None:
            params["cutoff"] = datetime.now(UTC) - max_age
            age_filter = "AND stored_at >= {cutoff:DateTime64(3, 'UTC')}"
        result = self._client.query(
            f"SELECT input_sha256s, output_sha256s, stored_at"
            f" FROM {TABLE_FLOW_COMPLETIONS} FINAL"
            f" WHERE run_scope = {{run_scope:String}} AND flow_name = {{flow_name:String}}"
            f" {age_filter}"
            f" LIMIT 1",
            parameters=params,
        )
        if not result.result_rows:
            return None
        row = result.result_rows[0]
        input_shas = tuple(_decode(s) for s in row[0])
        output_shas = tuple(_decode(s) for s in row[1])
        stored_at = row[2] if isinstance(row[2], datetime) else datetime.fromisoformat(str(row[2]))
        return FlowCompletion(
            flow_name=flow_name,
            input_sha256s=input_shas,
            output_sha256s=output_shas,
            stored_at=stored_at,
        )

    @staticmethod
    def _reconstruct_source_documents(
        parsed_rows: list[tuple[str, str, str | None, tuple[str, ...], tuple[str, ...], list[str], list[str], list[str], bytes]],
        att_content_by_sha: dict[str, bytes],
        document_type: type[Document],
    ) -> dict[str, Document]:
        """Reconstruct Document objects from parsed query rows and attachment content."""
        result: dict[str, Document] = {}
        with _suppress_document_registration():
            for matched_source, name, description, derived_from, triggered_by, att_names, att_descs, att_sha256s, content in parsed_rows:
                if matched_source in result:
                    continue

                attachments = _reconstruct_attachments(att_names, att_descs, att_sha256s, att_content_by_sha, name)
                result[matched_source] = document_type(
                    name=name,
                    content=content,
                    description=description,
                    derived_from=derived_from,
                    triggered_by=cast(tuple[DocumentSha256, ...], triggered_by) if triggered_by else (),
                    attachments=attachments if attachments else None,
                )

        return result


def _decode(value: bytes | str) -> str:
    """Decode bytes to str if needed (strings_as_bytes=True mode)."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


_HEX_CHARS = frozenset("0123456789abcdefABCDEF")


def _decode_content(raw: bytes | str, expected_length: int) -> bytes:
    """Decode content from ClickHouse using length comparison to detect hex encoding.

    ClickHouse stores binary content in String columns. The clickhouse_connect driver
    returns binary content as hex-encoded strings (e.g., "89504e47" for PNG header).
    Hex encoding always produces exactly 2x the original byte count.

    Args:
        raw: Content from ClickHouse (bytes or string)
        expected_length: Actual byte length from length(content) in query

    Returns:
        Decoded binary content as bytes
    """
    if isinstance(raw, bytes):
        return raw

    # Hex-encoded binary: string length is exactly 2x the stored byte length
    if len(raw) == 2 * expected_length and expected_length > 0 and all(c in _HEX_CHARS for c in raw):
        return bytes.fromhex(raw)

    return raw.encode("utf-8")
