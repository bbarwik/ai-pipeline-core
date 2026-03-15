"""ClickHouse backend for the redesigned span/document/blob/log schema."""

import time
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from clickhouse_connect.driver.asyncclient import AsyncClient
from clickhouse_connect.driver.exceptions import DatabaseError as ClickHouseDatabaseError

from ai_pipeline_core.database._json_helpers import parse_json_object
from ai_pipeline_core.database._serialization import (
    LOG_COLUMNS,
    SPAN_COLUMNS,
    decode_text,
    row_to_log,
    row_to_span,
    string_tuple,
)
from ai_pipeline_core.database._types import (
    TOKENS_CACHE_READ_KEY,
    TOKENS_INPUT_KEY,
    TOKENS_OUTPUT_KEY,
    TOKENS_REASONING_KEY,
    CostTotals,
    DocumentRecord,
    HydratedDocument,
    LogRecord,
    SpanKind,
    SpanRecord,
    SpanStatus,
    _BlobRecord,
    get_token_count,
)
from ai_pipeline_core.database.clickhouse._connection import get_async_clickhouse_client
from ai_pipeline_core.database.clickhouse._ddl import BLOBS_TABLE, DOCUMENTS_TABLE, LOGS_TABLE, SPANS_TABLE
from ai_pipeline_core.database.clickhouse._rows import (
    BLOB_COLUMNS,
    DOCUMENT_COLUMNS,
    blob_to_row,
    document_to_row,
    log_to_row,
    row_to_blob,
    row_to_document,
    span_to_row,
)
from ai_pipeline_core.logger import get_pipeline_logger
from ai_pipeline_core.settings import Settings

__all__ = [
    "ClickHouseDatabase",
]

logger = get_pipeline_logger(__name__)

_FAILURE_THRESHOLD = 3
_RECONNECT_INTERVAL_SEC = 60


class ClickHouseDatabase:
    """ClickHouse backend implementing the redesigned span protocols."""

    supports_remote = True

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client: AsyncClient | None = None
        self._consecutive_failures = 0
        self._circuit_open = False
        self._last_reconnect_attempt = 0.0

    async def _ensure_client(self) -> AsyncClient:
        now = time.monotonic()
        if self._client is not None:
            return self._client
        if self._circuit_open and now - self._last_reconnect_attempt < _RECONNECT_INTERVAL_SEC:
            raise ConnectionError("ClickHouse circuit breaker is open. Wait before retrying or restore ClickHouse connectivity.")
        self._last_reconnect_attempt = now
        try:
            self._client = await get_async_clickhouse_client(self._settings)
        except ClickHouseDatabaseError, ConnectionError, OSError:
            await self._record_failure()
            raise
        self._record_success()
        return self._client

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._circuit_open = False

    async def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._client is not None:
            await self._client.close()
            self._client = None
        if self._consecutive_failures >= _FAILURE_THRESHOLD:
            self._circuit_open = True
            logger.warning("ClickHouse circuit breaker opened after %d consecutive failures", self._consecutive_failures)

    async def _query(self, sql: str, *, parameters: dict[str, Any] | None = None) -> Any:
        client = await self._ensure_client()
        try:
            result = await client.query(sql, parameters=parameters)
        except ClickHouseDatabaseError, ConnectionError, OSError:
            await self._record_failure()
            raise
        self._record_success()
        return result

    async def _insert(self, table: str, rows: list[list[Any]], *, column_names: tuple[str, ...]) -> None:
        if not rows:
            return
        client = await self._ensure_client()
        try:
            await client.insert(table, rows, column_names=list(column_names))
        except ClickHouseDatabaseError, ConnectionError, OSError:
            await self._record_failure()
            raise
        self._record_success()

    async def _command(self, sql: str, *, parameters: dict[str, Any] | None = None) -> Any:
        client = await self._ensure_client()
        try:
            result = await client.command(sql, parameters=parameters)
        except ClickHouseDatabaseError, ConnectionError, OSError:
            await self._record_failure()
            raise
        self._record_success()
        return result

    async def insert_span(self, span: SpanRecord) -> None:
        await self._insert(SPANS_TABLE, [span_to_row(span)], column_names=SPAN_COLUMNS)

    async def save_document(self, record: DocumentRecord) -> None:
        await self._insert(DOCUMENTS_TABLE, [document_to_row(record)], column_names=DOCUMENT_COLUMNS)

    async def save_document_batch(self, records: list[DocumentRecord]) -> None:
        await self._insert(
            DOCUMENTS_TABLE,
            [document_to_row(record) for record in records],
            column_names=DOCUMENT_COLUMNS,
        )

    async def save_blob(self, blob: _BlobRecord) -> None:
        await self._insert(BLOBS_TABLE, [blob_to_row(blob)], column_names=BLOB_COLUMNS)

    async def save_blob_batch(self, blobs: list[_BlobRecord]) -> None:
        await self._insert(
            BLOBS_TABLE,
            [blob_to_row(blob) for blob in blobs],
            column_names=BLOB_COLUMNS,
        )

    async def save_logs_batch(self, logs: list[LogRecord]) -> None:
        await self._insert(
            LOGS_TABLE,
            [log_to_row(log) for log in logs],
            column_names=LOG_COLUMNS,
        )

    async def update_document_summary(self, document_sha256: str, summary: str) -> None:
        await self._command(
            f"INSERT INTO {DOCUMENTS_TABLE} ({', '.join(DOCUMENT_COLUMNS)}) "
            "SELECT document_sha256, content_sha256, document_type, name, "
            "description, mime_type, size_bytes, {summary:String}, "
            "derived_from, triggered_by, "
            "`attachments.name`, `attachments.description`, "
            "`attachments.content_sha256`, `attachments.mime_type`, "
            "`attachments.size_bytes`, "
            "publicly_visible "
            f"FROM {DOCUMENTS_TABLE} FINAL WHERE document_sha256 = {{document_sha256:String}}",
            parameters={"summary": summary, "document_sha256": document_sha256},
        )

    async def flush(self) -> None:
        return None

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def get_span(self, span_id: UUID) -> SpanRecord | None:
        result = await self._query(
            f"SELECT {', '.join(SPAN_COLUMNS)} FROM {SPANS_TABLE} FINAL WHERE span_id = {{span_id:UUID}}",
            parameters={"span_id": span_id},
        )
        if not result.result_rows:
            return None
        return row_to_span(tuple(result.result_rows[0]))

    async def get_child_spans(self, parent_span_id: UUID) -> list[SpanRecord]:
        result = await self._query(
            f"SELECT {', '.join(SPAN_COLUMNS)} FROM {SPANS_TABLE} FINAL "
            "WHERE parent_span_id = {parent_span_id:UUID} "
            "ORDER BY sequence_no, started_at, span_id",
            parameters={"parent_span_id": parent_span_id},
        )
        return [row_to_span(tuple(row)) for row in result.result_rows]

    async def get_deployment_tree(self, root_deployment_id: UUID) -> list[SpanRecord]:
        result = await self._query(
            f"SELECT {', '.join(SPAN_COLUMNS)} FROM {SPANS_TABLE} FINAL "
            "WHERE root_deployment_id = {root_deployment_id:UUID} "
            "ORDER BY started_at, deployment_id, sequence_no, version, span_id",
            parameters={"root_deployment_id": root_deployment_id},
        )
        return [row_to_span(tuple(row)) for row in result.result_rows]

    async def get_deployment_by_run_id(self, run_id: str) -> SpanRecord | None:
        result = await self._query(
            f"SELECT {', '.join(SPAN_COLUMNS)} FROM {SPANS_TABLE} FINAL "
            "WHERE run_id = {run_id:String} AND kind = {kind:String} "
            "ORDER BY started_at DESC, span_id DESC LIMIT 1",
            parameters={"run_id": run_id, "kind": SpanKind.DEPLOYMENT},
        )
        if not result.result_rows:
            return None
        return row_to_span(tuple(result.result_rows[0]))

    async def list_deployments(
        self,
        limit: int,
        *,
        status: str | None = None,
    ) -> list[SpanRecord]:
        if limit <= 0:
            return []
        filters = ["kind = {kind:String}"]
        parameters: dict[str, Any] = {"kind": SpanKind.DEPLOYMENT, "limit": limit}
        if status is not None:
            filters.append("status = {status:String}")
            parameters["status"] = status
        result = await self._query(
            f"SELECT {', '.join(SPAN_COLUMNS)} FROM {SPANS_TABLE} FINAL "
            f"WHERE {' AND '.join(filters)} ORDER BY started_at DESC, span_id DESC LIMIT {{limit:UInt64}}",
            parameters=parameters,
        )
        return [row_to_span(tuple(row)) for row in result.result_rows]

    async def get_cached_completion(
        self,
        cache_key: str,
        *,
        max_age: timedelta | None = None,
    ) -> SpanRecord | None:
        filters = [
            "cache_key = {cache_key:String}",
            "status = {status:String}",
        ]
        parameters: dict[str, Any] = {
            "cache_key": cache_key,
            "status": SpanStatus.COMPLETED,
        }
        if max_age is not None:
            filters.append("ended_at IS NOT NULL")
            filters.append("ended_at >= {min_ended_at:DateTime64(3)}")
            parameters["min_ended_at"] = datetime.now(UTC) - max_age
        result = await self._query(
            f"SELECT {', '.join(SPAN_COLUMNS)} FROM {SPANS_TABLE} FINAL "
            f"WHERE {' AND '.join(filters)} ORDER BY ended_at DESC, version DESC, span_id DESC LIMIT 1",
            parameters=parameters,
        )
        if not result.result_rows:
            return None
        return row_to_span(tuple(result.result_rows[0]))

    async def get_deployment_cost_totals(self, root_deployment_id: UUID) -> CostTotals:
        result = await self._query(
            f"SELECT kind, cost_usd, metrics_json FROM {SPANS_TABLE} FINAL WHERE root_deployment_id = {{root_deployment_id:UUID}}",
            parameters={"root_deployment_id": root_deployment_id},
        )
        cost_usd = 0.0
        ti, to, tc, tr = 0, 0, 0, 0
        for kind, span_cost_usd, metrics_json in result.result_rows:
            cost_usd += float(span_cost_usd)
            if decode_text(kind) == SpanKind.LLM_ROUND:
                metrics = parse_json_object(
                    decode_text(metrics_json),
                    context=f"Span tree {root_deployment_id}",
                    field_name="metrics_json",
                )
                ti += get_token_count(metrics, TOKENS_INPUT_KEY)
                to += get_token_count(metrics, TOKENS_OUTPUT_KEY)
                tc += get_token_count(metrics, TOKENS_CACHE_READ_KEY)
                tr += get_token_count(metrics, TOKENS_REASONING_KEY)
        return CostTotals(cost_usd=cost_usd, tokens_input=ti, tokens_output=to, tokens_cache_read=tc, tokens_reasoning=tr)

    async def get_deployment_span_count(
        self,
        root_deployment_id: UUID,
        *,
        kinds: list[str] | None = None,
    ) -> int:
        if kinds == []:
            return 0
        filters = ["root_deployment_id = {root_deployment_id:UUID}"]
        parameters: dict[str, Any] = {"root_deployment_id": root_deployment_id}
        if kinds is not None:
            filters.append("kind IN {kinds:Array(String)}")
            parameters["kinds"] = kinds
        result = await self._query(
            f"SELECT count() FROM {SPANS_TABLE} FINAL WHERE {' AND '.join(filters)}",
            parameters=parameters,
        )
        return int(result.result_rows[0][0]) if result.result_rows else 0

    async def get_spans_referencing_document(
        self,
        document_sha256: str,
        *,
        kinds: list[str] | None = None,
    ) -> list[SpanRecord]:
        if kinds == []:
            return []
        filters = [
            "("
            "has(input_document_shas, {document_sha256:String}) "
            "OR has(output_document_shas, {document_sha256:String}) "
            "OR has(input_blob_shas, {document_sha256:String}) "
            "OR has(output_blob_shas, {document_sha256:String})"
            ")"
        ]
        parameters: dict[str, Any] = {"document_sha256": document_sha256}
        if kinds is not None:
            filters.append("kind IN {kinds:Array(String)}")
            parameters["kinds"] = kinds
        result = await self._query(
            f"SELECT {', '.join(SPAN_COLUMNS)} FROM {SPANS_TABLE} FINAL "
            f"WHERE {' AND '.join(filters)} ORDER BY started_at, deployment_id, sequence_no, version, span_id",
            parameters=parameters,
        )
        return [row_to_span(tuple(row)) for row in result.result_rows]

    async def get_document(self, document_sha256: str) -> DocumentRecord | None:
        result = await self._query(
            f"SELECT {', '.join(DOCUMENT_COLUMNS)} FROM {DOCUMENTS_TABLE} FINAL WHERE document_sha256 = {{document_sha256:String}}",
            parameters={"document_sha256": document_sha256},
        )
        if not result.result_rows:
            return None
        return row_to_document(tuple(result.result_rows[0]))

    async def get_documents_batch(self, sha256s: list[str]) -> dict[str, DocumentRecord]:
        unique_shas = list(dict.fromkeys(sha256s))
        if not unique_shas:
            return {}
        result = await self._query(
            f"SELECT {', '.join(DOCUMENT_COLUMNS)} FROM {DOCUMENTS_TABLE} FINAL WHERE document_sha256 IN {{document_sha256s:Array(String)}}",
            parameters={"document_sha256s": unique_shas},
        )
        return {document.document_sha256: document for document in (row_to_document(tuple(row)) for row in result.result_rows)}

    async def get_document_with_content(self, document_sha256: str) -> HydratedDocument | None:
        record = await self.get_document(document_sha256)
        if record is None:
            return None

        needed_shas = [record.content_sha256, *record.attachment_content_sha256s]
        blobs = await self.get_blobs_batch(needed_shas)

        primary_blob = blobs.get(record.content_sha256)
        if primary_blob is None:
            return None

        attachment_contents: dict[str, bytes] = {}
        missing_attachment_shas: list[str] = []
        for att_sha in record.attachment_content_sha256s:
            att_blob = blobs.get(att_sha)
            if att_blob is None:
                missing_attachment_shas.append(att_sha)
            else:
                attachment_contents[att_sha] = att_blob.content

        if missing_attachment_shas:
            missing_list = ", ".join(sorted(missing_attachment_shas))
            raise ValueError(
                f"Document {record.document_sha256} references attachment blobs that are missing from storage: {missing_list}. "
                "Persist every attachment blob before reading the document."
            )

        return HydratedDocument(
            record=record,
            content=primary_blob.content,
            attachment_contents=attachment_contents,
        )

    async def get_all_document_shas_for_tree(self, root_deployment_id: UUID) -> set[str]:
        result = await self._query(
            f"SELECT input_document_shas, output_document_shas FROM {SPANS_TABLE} FINAL WHERE root_deployment_id = {{root_deployment_id:UUID}}",
            parameters={"root_deployment_id": root_deployment_id},
        )
        shas: set[str] = set()
        for input_shas, output_shas in result.result_rows:
            shas.update(string_tuple(input_shas))
            shas.update(string_tuple(output_shas))
        return shas

    async def get_blob(self, content_sha256: str) -> _BlobRecord | None:
        result = await self._query(
            f"SELECT {', '.join(BLOB_COLUMNS)} FROM {BLOBS_TABLE} FINAL WHERE content_sha256 = {{content_sha256:String}} LIMIT 1",
            parameters={"content_sha256": content_sha256},
        )
        if not result.result_rows:
            return None
        return row_to_blob(tuple(result.result_rows[0]))

    async def get_blobs_batch(self, content_sha256s: list[str]) -> dict[str, _BlobRecord]:
        unique_shas = list(dict.fromkeys(content_sha256s))
        if not unique_shas:
            return {}
        result = await self._query(
            f"SELECT {', '.join(BLOB_COLUMNS)} FROM {BLOBS_TABLE} FINAL WHERE content_sha256 IN {{content_sha256s:Array(String)}}",
            parameters={"content_sha256s": unique_shas},
        )
        return {blob.content_sha256: blob for blob in (row_to_blob(tuple(row)) for row in result.result_rows)}

    async def get_span_logs(
        self,
        span_id: UUID,
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[LogRecord]:
        filters = ["span_id = {span_id:UUID}"]
        parameters: dict[str, Any] = {"span_id": span_id}
        if level is not None:
            filters.append("level = {level:String}")
            parameters["level"] = level
        if category is not None:
            filters.append("category = {category:String}")
            parameters["category"] = category
        result = await self._query(
            f"SELECT {', '.join(LOG_COLUMNS)} FROM {LOGS_TABLE} WHERE {' AND '.join(filters)} ORDER BY sequence_no, timestamp, span_id",
            parameters=parameters,
        )
        return [row_to_log(tuple(row)) for row in result.result_rows]

    async def get_deployment_logs(
        self,
        deployment_id: UUID,
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[LogRecord]:
        filters = ["deployment_id = {deployment_id:UUID}"]
        parameters: dict[str, Any] = {"deployment_id": deployment_id}
        if level is not None:
            filters.append("level = {level:String}")
            parameters["level"] = level
        if category is not None:
            filters.append("category = {category:String}")
            parameters["category"] = category
        result = await self._query(
            f"SELECT {', '.join(LOG_COLUMNS)} FROM {LOGS_TABLE} WHERE {' AND '.join(filters)} ORDER BY timestamp, sequence_no, span_id",
            parameters=parameters,
        )
        return [row_to_log(tuple(row)) for row in result.result_rows]

    async def get_deployment_logs_batch(
        self,
        deployment_ids: list[UUID],
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[LogRecord]:
        if not deployment_ids:
            return []
        filters = ["toString(deployment_id) IN {deployment_ids:Array(String)}"]
        parameters: dict[str, Any] = {"deployment_ids": [str(uid) for uid in deployment_ids]}
        if level is not None:
            filters.append("level = {level:String}")
            parameters["level"] = level
        if category is not None:
            filters.append("category = {category:String}")
            parameters["category"] = category
        result = await self._query(
            f"SELECT {', '.join(LOG_COLUMNS)} FROM {LOGS_TABLE} WHERE {' AND '.join(filters)} ORDER BY timestamp, sequence_no, span_id",
            parameters=parameters,
        )
        return [row_to_log(tuple(row)) for row in result.result_rows]

    async def find_documents_by_name(
        self,
        names: list[str],
        *,
        document_type: str | None = None,
    ) -> dict[str, DocumentRecord]:
        if not names:
            return {}
        unique_names = list(dict.fromkeys(names))
        filters = ["name IN {names:Array(String)}"]
        parameters: dict[str, Any] = {"names": unique_names}
        if document_type is not None:
            filters.append("document_type = {document_type:String}")
            parameters["document_type"] = document_type
        result = await self._query(
            f"SELECT {', '.join(DOCUMENT_COLUMNS)} FROM {DOCUMENTS_TABLE} FINAL WHERE {' AND '.join(filters)}",
            parameters=parameters,
        )
        found: dict[str, DocumentRecord] = {}
        for row in result.result_rows:
            record = row_to_document(tuple(row))
            existing = found.get(record.name)
            if existing is None or record.document_sha256 > existing.document_sha256:
                found[record.name] = record
        return found
