"""In-memory backend for the span-based database schema."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

from ai_pipeline_core.database._json_helpers import parse_json_object
from ai_pipeline_core.database._sorting import (
    child_span_sort_key,
    deployment_sort_key,
    log_sort_key,
    span_sort_key,
)
from ai_pipeline_core.database._types import (
    TOKENS_CACHE_READ_KEY,
    TOKENS_INPUT_KEY,
    TOKENS_OUTPUT_KEY,
    TOKENS_REASONING_KEY,
    BlobRecord,
    CostTotals,
    DocumentRecord,
    HydratedDocument,
    LogRecord,
    SpanKind,
    SpanRecord,
    SpanStatus,
    get_token_count,
)

__all__ = ["MemoryDatabase"]


class MemoryDatabase:
    """Dict-based backend for tests covering the span schema."""

    supports_remote = False

    def __init__(self) -> None:
        self._spans: dict[UUID, SpanRecord] = {}
        self._documents: dict[str, DocumentRecord] = {}
        self._blobs: dict[str, BlobRecord] = {}
        self._logs: list[LogRecord] = []

    async def insert_span(self, span: SpanRecord) -> None:
        existing = self._spans.get(span.span_id)
        if existing is None or span.version > existing.version:
            self._spans[span.span_id] = span

    async def save_document(self, record: DocumentRecord) -> None:
        if record.document_sha256 in self._documents:
            return
        self._documents[record.document_sha256] = record

    async def save_document_batch(self, records: list[DocumentRecord]) -> None:
        for record in records:
            await self.save_document(record)

    async def save_blob(self, blob: BlobRecord) -> None:
        if blob.content_sha256 in self._blobs:
            return
        self._blobs[blob.content_sha256] = blob

    async def save_blob_batch(self, blobs: list[BlobRecord]) -> None:
        for blob in blobs:
            await self.save_blob(blob)

    async def save_logs_batch(self, logs: list[LogRecord]) -> None:
        self._logs.extend(logs)

    async def update_document_summary(self, document_sha256: str, summary: str) -> None:
        existing = self._documents.get(document_sha256)
        if existing is None:
            return
        self._documents[document_sha256] = replace(existing, summary=summary)

    async def flush(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def get_span(self, span_id: UUID) -> SpanRecord | None:
        return self._spans.get(span_id)

    async def get_child_spans(self, parent_span_id: UUID) -> list[SpanRecord]:
        matches = [span for span in self._spans.values() if span.parent_span_id == parent_span_id]
        return sorted(matches, key=child_span_sort_key)

    async def get_deployment_tree(self, root_deployment_id: UUID) -> list[SpanRecord]:
        matches = [span for span in self._spans.values() if span.root_deployment_id == root_deployment_id]
        return sorted(matches, key=span_sort_key)

    async def get_deployment_by_run_id(self, run_id: str) -> SpanRecord | None:
        matches = [span for span in self._spans.values() if span.kind == SpanKind.DEPLOYMENT and span.run_id == run_id]
        if not matches:
            return None
        return max(matches, key=deployment_sort_key)

    async def list_deployments(
        self,
        limit: int,
        *,
        status: str | None = None,
    ) -> list[SpanRecord]:
        matches = [span for span in self._spans.values() if span.kind == SpanKind.DEPLOYMENT]
        if status is not None:
            matches = [span for span in matches if span.status == status]
        return sorted(matches, key=deployment_sort_key, reverse=True)[:limit]

    async def get_cached_completion(
        self,
        cache_key: str,
        *,
        max_age: timedelta | None = None,
    ) -> SpanRecord | None:
        now = datetime.now(UTC)
        matches: list[SpanRecord] = []
        for span in self._spans.values():
            if span.cache_key != cache_key or span.status != SpanStatus.COMPLETED:
                continue
            if max_age is not None and (span.ended_at is None or now - span.ended_at > max_age):
                continue
            matches.append(span)
        if not matches:
            return None
        return max(matches, key=lambda span: (span.ended_at or span.started_at, span.version, str(span.span_id)))

    async def get_deployment_cost_totals(self, root_deployment_id: UUID) -> CostTotals:
        totals = CostTotals()
        for span in self._spans.values():
            if span.root_deployment_id != root_deployment_id or span.kind != SpanKind.LLM_ROUND:
                continue
            metrics = parse_json_object(span.metrics_json, context=f"Span {span.span_id}", field_name="metrics_json")
            totals = CostTotals(
                cost_usd=totals.cost_usd + span.cost_usd,
                tokens_input=totals.tokens_input + get_token_count(metrics, TOKENS_INPUT_KEY),
                tokens_output=totals.tokens_output + get_token_count(metrics, TOKENS_OUTPUT_KEY),
                tokens_cache_read=totals.tokens_cache_read + get_token_count(metrics, TOKENS_CACHE_READ_KEY),
                tokens_reasoning=totals.tokens_reasoning + get_token_count(metrics, TOKENS_REASONING_KEY),
            )
        return totals

    async def get_deployment_span_count(
        self,
        root_deployment_id: UUID,
        *,
        kinds: list[str] | None = None,
    ) -> int:
        allowed_kinds = set(kinds) if kinds is not None else None
        return sum(
            1 for span in self._spans.values() if span.root_deployment_id == root_deployment_id and (allowed_kinds is None or span.kind in allowed_kinds)
        )

    async def get_spans_referencing_document(
        self,
        document_sha256: str,
        *,
        kinds: list[str] | None = None,
    ) -> list[SpanRecord]:
        allowed_kinds = set(kinds) if kinds is not None else None
        matches: list[SpanRecord] = []
        for span in self._spans.values():
            if allowed_kinds is not None and span.kind not in allowed_kinds:
                continue
            if document_sha256 in span.input_document_shas or document_sha256 in span.output_document_shas:
                matches.append(span)
                continue
            if document_sha256 in span.input_blob_shas or document_sha256 in span.output_blob_shas:
                matches.append(span)
        return sorted(matches, key=span_sort_key)

    async def get_document(self, document_sha256: str) -> DocumentRecord | None:
        return self._documents.get(document_sha256)

    async def get_documents_batch(self, sha256s: list[str]) -> dict[str, DocumentRecord]:
        return {sha256: self._documents[sha256] for sha256 in sha256s if sha256 in self._documents}

    async def get_document_with_content(self, document_sha256: str) -> HydratedDocument | None:
        record = self._documents.get(document_sha256)
        if record is None:
            return None
        blob = self._blobs.get(record.content_sha256)
        if blob is None:
            return None

        attachment_contents: dict[str, bytes] = {}
        missing_attachment_shas: list[str] = []
        for attachment_sha in record.attachment_content_sha256s:
            attachment_blob = self._blobs.get(attachment_sha)
            if attachment_blob is None:
                missing_attachment_shas.append(attachment_sha)
                continue
            attachment_contents[attachment_sha] = attachment_blob.content

        if missing_attachment_shas:
            missing_list = ", ".join(sorted(missing_attachment_shas))
            raise ValueError(
                f"Document {record.document_sha256} references attachment blobs that are missing from storage: {missing_list}. "
                "Persist every attachment blob before reading the document."
            )

        return HydratedDocument(record=record, content=blob.content, attachment_contents=attachment_contents)

    async def get_all_document_shas_for_tree(self, root_deployment_id: UUID) -> set[str]:
        shas: set[str] = set()
        for span in self._spans.values():
            if span.root_deployment_id != root_deployment_id:
                continue
            shas.update(span.input_document_shas)
            shas.update(span.output_document_shas)
        return shas

    async def get_blob(self, content_sha256: str) -> BlobRecord | None:
        return self._blobs.get(content_sha256)

    async def get_blobs_batch(self, content_sha256s: list[str]) -> dict[str, BlobRecord]:
        return {sha256: self._blobs[sha256] for sha256 in content_sha256s if sha256 in self._blobs}

    async def get_span_logs(
        self,
        span_id: UUID,
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[LogRecord]:
        return sorted(
            (log for log in self._logs if log.span_id == span_id and (level is None or log.level == level) and (category is None or log.category == category)),
            key=lambda log: (log.sequence_no, log.timestamp, str(log.span_id)),
        )

    async def get_deployment_logs(
        self,
        deployment_id: UUID,
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[LogRecord]:
        return sorted(
            (
                log
                for log in self._logs
                if log.deployment_id == deployment_id and (level is None or log.level == level) and (category is None or log.category == category)
            ),
            key=log_sort_key,
        )

    async def get_deployment_logs_batch(
        self,
        deployment_ids: list[UUID],
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[LogRecord]:
        allowed_ids = set(deployment_ids)
        return sorted(
            (
                log
                for log in self._logs
                if log.deployment_id in allowed_ids and (level is None or log.level == level) and (category is None or log.category == category)
            ),
            key=log_sort_key,
        )
