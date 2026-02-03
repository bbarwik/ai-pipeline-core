"""ClickHouse-backed document store for production use.

Two-table schema: document_content (deduplicated blobs) and document_index
(per-run document metadata). Uses ReplacingMergeTree for idempotent writes.
Circuit breaker buffers writes when ClickHouse is unavailable.
"""

import asyncio
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import clickhouse_connect

from ai_pipeline_core.document_store._summary import SummaryGenerator
from ai_pipeline_core.document_store._summary_worker import SummaryWorker
from ai_pipeline_core.documents._context_vars import suppress_registration
from ai_pipeline_core.documents._hashing import compute_content_sha256, compute_document_sha256
from ai_pipeline_core.documents.attachment import Attachment
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

TABLE_DOCUMENT_CONTENT = "document_content"
TABLE_DOCUMENT_INDEX = "document_index"

_DDL_CONTENT = f"""
CREATE TABLE IF NOT EXISTS {TABLE_DOCUMENT_CONTENT}
(
    content_sha256     String,
    content            String CODEC(ZSTD(3)),
    created_at         DateTime64(3, 'UTC'),
    INDEX content_sha256_idx content_sha256 TYPE bloom_filter GRANULARITY 1
)
ENGINE = ReplacingMergeTree()
ORDER BY (content_sha256)
SETTINGS index_granularity = 8192
"""

_DDL_INDEX = f"""
CREATE TABLE IF NOT EXISTS {TABLE_DOCUMENT_INDEX}
(
    document_sha256        String,
    run_scope              String,
    content_sha256         String,
    class_name             LowCardinality(String),
    name                   String,
    description            String DEFAULT '',
    mime_type              LowCardinality(String),
    sources                Array(String),
    origins                Array(String),
    attachment_names        Array(String),
    attachment_descriptions Array(String),
    attachment_sha256s      Array(String),
    summary                String DEFAULT '',
    stored_at              DateTime64(3, 'UTC'),
    version                UInt64 DEFAULT 1,
    INDEX doc_sha256_idx document_sha256 TYPE bloom_filter GRANULARITY 1
)
ENGINE = ReplacingMergeTree(version)
ORDER BY (run_scope, class_name, document_sha256)
SETTINGS index_granularity = 8192
"""

_MAX_BUFFER_SIZE = 10_000
_RECONNECT_INTERVAL_SEC = 60
_FAILURE_THRESHOLD = 3


@dataclass
class _BufferedWrite:
    """A pending write operation buffered during circuit breaker open state."""

    document: Document
    run_scope: str


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

    async def _run(self, fn: Any, *args: Any) -> Any:
        """Run a sync function on the dedicated executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    # --- Connection management (sync, executor thread only) ---

    def _connect(self) -> None:
        self._client = clickhouse_connect.get_client(  # pyright: ignore[reportUnknownMemberType]
            **self._params,  # pyright: ignore[reportArgumentType]
        )
        logger.info(f"Document store connected to ClickHouse at {self._params['host']}:{self._params['port']}")

    def _ensure_tables(self) -> None:
        if self._tables_initialized:
            return
        if self._client is None:
            self._connect()
        self._client.command(_DDL_CONTENT)
        self._client.command(_DDL_INDEX)
        self._tables_initialized = True
        logger.info("Document store tables verified/created")

    def _ensure_connected(self) -> bool:
        try:
            if self._client is None:
                self._connect()
            self._ensure_tables()
            return True
        except Exception as e:
            logger.warning(f"ClickHouse connection failed: {e}")
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
            logger.warning(f"Circuit breaker opened after {self._consecutive_failures} failures")

    # --- Async public API ---

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

    async def load(self, run_scope: str, document_types: list[type[Document]]) -> list[Document]:
        """Load documents via index JOIN content, then batch-fetch attachments."""
        return await self._run(self._load_sync, run_scope, document_types)

    async def has_documents(self, run_scope: str, document_type: type[Document]) -> bool:
        """Check if any documents of this type exist in the run scope."""
        return await self._run(self._has_documents_sync, run_scope, document_type)

    async def check_existing(self, sha256s: list[str]) -> set[str]:
        """Return the subset of sha256s that exist in the document index."""
        return await self._run(self._check_existing_sync, sha256s)

    async def update_summary(self, run_scope: str, document_sha256: str, summary: str) -> None:
        """Update summary column for a stored document via ALTER TABLE UPDATE."""
        await self._run(self._update_summary_sync, run_scope, document_sha256, summary)

    async def load_summaries(self, run_scope: str, document_sha256s: list[str]) -> dict[str, str]:
        """Load summaries by SHA256 from the document index."""
        return await self._run(self._load_summaries_sync, run_scope, document_sha256s)

    def flush(self) -> None:
        """Block until all pending document summaries are processed."""
        if self._summary_worker:
            self._summary_worker.flush()

    def shutdown(self) -> None:
        """Flush pending summaries, stop the summary worker, and release the executor."""
        if self._summary_worker:
            self._summary_worker.shutdown()
        self._executor.shutdown(wait=True)

    # --- Sync implementations (executor thread only) ---

    def _save_sync(self, document: Document, run_scope: str) -> None:
        if self._circuit_open:
            if not self._try_reconnect():
                self._buffer.append(_BufferedWrite(document=document, run_scope=run_scope))
                return
            self._flush_buffer()

        try:
            self._ensure_tables()
            self._insert_document(document, run_scope)
            self._record_success()
        except Exception as e:
            logger.warning(f"Failed to save document '{document.name}': {e}")
            self._record_failure()
            self._buffer.append(_BufferedWrite(document=document, run_scope=run_scope))

    def _save_batch_sync(self, documents: list[Document], run_scope: str) -> None:
        if self._circuit_open:
            if not self._try_reconnect():
                for doc in documents:
                    self._buffer.append(_BufferedWrite(document=doc, run_scope=run_scope))
                return
            self._flush_buffer()

        try:
            self._ensure_tables()
        except Exception as e:
            logger.warning(f"Failed to ensure tables for batch: {e}")
            self._record_failure()
            for doc in documents:
                self._buffer.append(_BufferedWrite(document=doc, run_scope=run_scope))
            return

        for i, doc in enumerate(documents):
            try:
                self._insert_document(doc, run_scope)
            except Exception as e:
                logger.warning(f"Failed to save document '{doc.name}' in batch: {e}")
                self._record_failure()
                for remaining in documents[i:]:
                    self._buffer.append(_BufferedWrite(document=remaining, run_scope=run_scope))
                return
        self._record_success()

    def _flush_buffer(self) -> None:
        while self._buffer:
            item = self._buffer.popleft()
            try:
                self._insert_document(item.document, item.run_scope)
                if self._summary_worker:
                    self._summary_worker.schedule(item.run_scope, item.document)
            except Exception as e:
                logger.warning(f"Failed to flush buffered document: {e}")
                self._buffer.appendleft(item)
                break

    def _insert_document(self, document: Document, run_scope: str) -> None:
        doc_sha256 = compute_document_sha256(document)
        content_sha256 = compute_content_sha256(document.content)

        # Insert content using insert() for binary-safe handling (idempotent via ReplacingMergeTree)
        now = datetime.now(UTC)
        content_rows: list[list[Any]] = [
            [content_sha256, document.content, now],
        ]

        # Collect attachment content
        att_names: list[str] = []
        att_descriptions: list[str] = []
        att_sha256s: list[str] = []
        for att in sorted(document.attachments, key=lambda a: a.name):
            att_sha = compute_content_sha256(att.content)
            att_names.append(att.name)
            att_descriptions.append(att.description or "")
            att_sha256s.append(att_sha)
            content_rows.append([att_sha, att.content, now])

        self._client.insert(
            TABLE_DOCUMENT_CONTENT,
            content_rows,
            column_names=["content_sha256", "content", "created_at"],
        )

        # Insert index entry
        self._client.command(
            f"INSERT INTO {TABLE_DOCUMENT_INDEX} "
            "(document_sha256, run_scope, content_sha256, class_name, name, description, "
            "mime_type, sources, origins, "
            "attachment_names, attachment_descriptions, attachment_sha256s, stored_at, version) "
            "VALUES ("
            "{doc_sha256:String}, {run_scope:String}, {content_sha256:String}, "
            "{class_name:String}, {name:String}, {description:String}, "
            "{mime_type:String}, "
            "{sources:Array(String)}, {origins:Array(String)}, "
            "{att_names:Array(String)}, {att_descs:Array(String)}, {att_sha256s:Array(String)}, "
            "now64(3), 1)",
            parameters={
                "doc_sha256": doc_sha256,
                "run_scope": run_scope,
                "content_sha256": content_sha256,
                "class_name": document.__class__.__name__,
                "name": document.name,
                "description": document.description or "",
                "mime_type": document.mime_type,
                "sources": list(document.sources),
                "origins": list(document.origins),
                "att_names": att_names,
                "att_descs": att_descriptions,
                "att_sha256s": att_sha256s,
            },
        )

    def _load_sync(self, run_scope: str, document_types: list[type[Document]]) -> list[Document]:
        """Two-query load: index JOIN content, then batch attachment fetch."""
        self._ensure_tables()

        type_by_name: dict[str, type[Document]] = {t.__name__: t for t in document_types}
        class_names = list(type_by_name.keys())

        # Query 1: index JOIN content for document bodies
        rows = self._client.query(
            f"SELECT di.class_name, di.name, di.description, di.sources, di.origins, "
            f"di.attachment_names, di.attachment_descriptions, di.attachment_sha256s, "
            f"dc.content, length(dc.content) "
            f"FROM {TABLE_DOCUMENT_INDEX} AS di FINAL "
            f"JOIN {TABLE_DOCUMENT_CONTENT} AS dc FINAL ON di.content_sha256 = dc.content_sha256 "
            f"WHERE di.run_scope = {{run_scope:String}} "
            f"AND di.class_name IN {{class_names:Array(String)}}",
            parameters={"run_scope": run_scope, "class_names": class_names},
        )

        # Collect all attachment SHA256s needed across all rows
        parsed_rows: list[tuple[type[Document], str, str | None, tuple[str, ...], tuple[str, ...], list[str], list[str], list[str], bytes]] = []
        all_att_sha256s: set[str] = set()

        for row in rows.result_rows:
            class_name = _decode(row[0])
            doc_type = type_by_name.get(class_name)
            if doc_type is None:
                continue

            att_sha256s = [_decode(s) for s in row[7]]
            all_att_sha256s.update(att_sha256s)
            content = _decode_content(row[8], row[9])

            parsed_rows.append((
                doc_type,
                _decode(row[1]),  # name
                _decode(row[2]) or None,  # description
                tuple(_decode(s) for s in row[3]),  # sources
                tuple(_decode(o) for o in row[4]),  # origins
                [_decode(n) for n in row[5]],  # att_names
                [_decode(d) for d in row[6]],  # att_descs
                att_sha256s,
                content,
            ))

        # Query 2: batch fetch ALL attachment content in one query
        att_content_by_sha: dict[str, bytes] = {}
        if all_att_sha256s:
            att_rows = self._client.query(
                f"SELECT content_sha256, content, length(content) FROM {TABLE_DOCUMENT_CONTENT} FINAL WHERE content_sha256 IN {{sha256s:Array(String)}}",
                parameters={"sha256s": list(all_att_sha256s)},
            )
            for att_row in att_rows.result_rows:
                sha = _decode(att_row[0])
                att_content_by_sha[sha] = _decode_content(att_row[1], att_row[2])

        # Reconstruct documents (suppress registration to avoid polluting TaskDocumentContext)
        documents: list[Document] = []
        with suppress_registration():
            for doc_type, name, description, sources, origins, att_names, att_descs, att_sha256s, content in parsed_rows:
                attachments: tuple[Attachment, ...] = ()
                if att_sha256s:
                    att_list: list[Attachment] = []
                    for a_name, a_desc, a_sha in zip(att_names, att_descs, att_sha256s, strict=False):
                        a_content = att_content_by_sha.get(a_sha)
                        if a_content is None:
                            logger.warning(f"Attachment content {a_sha[:12]}... not found for document '{name}'")
                            continue
                        att_list.append(
                            Attachment(
                                name=a_name,
                                content=a_content,
                                description=a_desc or None,
                            )
                        )
                    attachments = tuple(att_list)

                doc = doc_type(
                    name=name,
                    content=content,
                    description=description,
                    sources=sources,
                    origins=origins if origins else (),
                    attachments=attachments if attachments else None,
                )
                documents.append(doc)

        return documents

    def _has_documents_sync(self, run_scope: str, document_type: type[Document]) -> bool:
        self._ensure_tables()
        result = self._client.query(
            f"SELECT 1 FROM {TABLE_DOCUMENT_INDEX} FINAL WHERE run_scope = {{run_scope:String}} AND class_name = {{class_name:String}} LIMIT 1",
            parameters={"run_scope": run_scope, "class_name": document_type.__name__},
        )
        return len(result.result_rows) > 0

    def _check_existing_sync(self, sha256s: list[str]) -> set[str]:
        if not sha256s:
            return set()
        self._ensure_tables()
        result = self._client.query(
            f"SELECT document_sha256 FROM {TABLE_DOCUMENT_INDEX} FINAL WHERE document_sha256 IN {{sha256s:Array(String)}}",
            parameters={"sha256s": sha256s},
        )
        return {_decode(row[0]) for row in result.result_rows}

    def _update_summary_sync(self, run_scope: str, document_sha256: str, summary: str) -> None:
        """Update summary column via ALTER TABLE UPDATE mutation."""
        try:
            self._ensure_tables()
            self._client.command(
                f"ALTER TABLE {TABLE_DOCUMENT_INDEX} UPDATE summary = {{summary:String}} "
                f"WHERE document_sha256 = {{sha256:String}} AND run_scope = {{run_scope:String}}",
                parameters={"summary": summary, "sha256": document_sha256, "run_scope": run_scope},
            )
        except Exception as e:
            logger.warning(f"Failed to update summary for {document_sha256[:12]}...: {e}")

    def _load_summaries_sync(self, run_scope: str, document_sha256s: list[str]) -> dict[str, str]:
        """Query summaries from the document index."""
        if not document_sha256s:
            return {}
        try:
            self._ensure_tables()
            result = self._client.query(
                f"SELECT document_sha256, summary FROM {TABLE_DOCUMENT_INDEX} FINAL "
                f"WHERE run_scope = {{run_scope:String}} "
                f"AND document_sha256 IN {{sha256s:Array(String)}} "
                f"AND summary != ''",
                parameters={"run_scope": run_scope, "sha256s": document_sha256s},
            )
            return {_decode(row[0]): _decode(row[1]) for row in result.result_rows}
        except Exception as e:
            logger.warning(f"Failed to load summaries: {e}")
            return {}


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
