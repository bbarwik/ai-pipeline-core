"""Trace reconstruction from ClickHouse.

Downloads span data and content from ClickHouse and rebuilds a local .trace/
directory structure for debugging. Uses the same directory layout as LocalTraceWriter.
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml

from ai_pipeline_core.logging import get_pipeline_logger

from ._config import SpanInfo, TraceDebugConfig, TraceState
from ._content import ContentWriter
from ._summary import generate_costs, generate_summary
from ._writer import LocalTraceWriter

logger = get_pipeline_logger(__name__)

__all__ = ["TraceDownloader", "download_trace"]

_REPLAY_FILENAMES: dict[str, str] = {
    "conversation": "conversation.yaml",
    "pipeline_task": "task.yaml",
    "pipeline_flow": "flow.yaml",
}

_SPAN_QUERY = """
SELECT
    ts.span_id,
    ts.trace_id,
    ts.parent_span_id,
    ts.name,
    ts.span_type,
    ts.status,
    ts.start_time,
    ts.end_time,
    ts.duration_ms,
    ts.cost,
    ts.tokens_input,
    ts.tokens_output,
    ts.llm_model,
    coalesce(tc.span_order, 0) AS span_order,
    coalesce(tc.input_json, '') AS input_json,
    coalesce(tc.output_json, '') AS output_json,
    coalesce(tc.replay_payload, '') AS replay_payload,
    coalesce(tc.attributes_json, '{}') AS attributes_json,
    coalesce(tc.events_json, '[]') AS events_json
FROM tracked_spans FINAL AS ts
LEFT JOIN trace_span_content AS tc
    ON ts.span_id = tc.span_id AND ts.execution_id = tc.execution_id
WHERE ts.execution_id = {execution_id:UUID}
ORDER BY tc.span_order ASC
"""

_RUN_METADATA_QUERY = """
SELECT run_id, flow_name, run_scope, status, start_time, end_time, total_cost, total_tokens, metadata
FROM pipeline_runs FINAL
WHERE execution_id = {execution_id:UUID}
"""

_CHILD_RUNS_QUERY = """
SELECT execution_id, run_id
FROM pipeline_runs FINAL
WHERE run_id LIKE {prefix:String}
AND run_id != {parent:String}
ORDER BY start_time
"""

_DOCUMENT_INDEX_QUERY = """
SELECT di.document_sha256, di.content_sha256, di.class_name,
       di.name, di.description, di.mime_type,
       di.derived_from, di.triggered_by,
       di.attachment_names, di.attachment_descriptions, di.attachment_sha256s
FROM document_index AS di FINAL
WHERE di.document_sha256 IN {sha256s:Array(String)}
"""

_DOCUMENT_CONTENT_QUERY = """
SELECT content_sha256, content, length(content)
FROM document_content FINAL
WHERE content_sha256 IN {sha256s:Array(String)}
"""

DOC_ID_LENGTH = 6


@dataclass(frozen=True, slots=True)
class _SpanData:
    """Parsed span data from ClickHouse query."""

    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    span_type: str
    status: str
    start_time: datetime
    end_time: datetime
    duration_ms: int
    cost: float
    tokens_input: int
    tokens_output: int
    llm_model: str | None
    span_order: int
    input_json: str
    output_json: str
    replay_payload: str
    attributes_json: str
    events_json: str


def _parse_json_safe(raw: str) -> Any:
    """Parse JSON string, returning None on empty or raw string on failure."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _parse_span_row(row: tuple) -> _SpanData:
    """Parse a query result row into _SpanData."""
    return _SpanData(
        span_id=row[0],
        trace_id=row[1],
        parent_span_id=row[2] if row[2] else None,
        name=row[3],
        span_type=row[4],
        status=row[5],
        start_time=row[6],
        end_time=row[7],
        duration_ms=row[8],
        cost=row[9],
        tokens_input=row[10],
        tokens_output=row[11],
        llm_model=row[12] if row[12] else None,
        span_order=row[13],
        input_json=row[14],
        output_json=row[15],
        replay_payload=row[16],
        attributes_json=row[17],
        events_json=row[18],
    )


def _walk_doc_refs(obj: Any, sha256s: set[str]) -> None:
    """Recursively find $doc_ref values in a nested structure."""
    if isinstance(obj, dict):
        if "$doc_ref" in obj:
            sha256s.add(obj["$doc_ref"])
        for v in obj.values():
            _walk_doc_refs(v, sha256s)
    elif isinstance(obj, list):
        for item in obj:
            _walk_doc_refs(item, sha256s)


def _decode_content_blob(raw: Any, expected_length: int) -> bytes:
    """Decode content from ClickHouse, handling hex encoding quirk."""
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, str) and len(raw) == 2 * expected_length:
        return bytes.fromhex(raw)
    if isinstance(raw, str):
        return raw.encode("utf-8")
    return bytes(raw)


class TraceDownloader:
    """Downloads and reconstructs .trace/ directories from ClickHouse."""

    def __init__(self, *, client: Any) -> None:
        self._client = client

    def download_trace(
        self,
        execution_id: UUID,
        output_path: Path,
        *,
        include_documents: bool = False,
        follow_children: bool = False,
    ) -> Path:
        """Download trace data and rebuild .trace/ directory structure."""
        output_path.mkdir(parents=True, exist_ok=True)

        span_result = self._client.query(_SPAN_QUERY, parameters={"execution_id": str(execution_id)})
        span_rows = [_parse_span_row(row) for row in span_result.result_rows]

        run_result = self._client.query(_RUN_METADATA_QUERY, parameters={"execution_id": str(execution_id)})
        run_rows = run_result.result_rows

        trace_name = "trace"
        if run_rows:
            trace_name = run_rows[0][1]
        elif span_rows:
            trace_name = span_rows[0].name

        trace_state = self._build_trace(span_rows, output_path, trace_name)

        config = TraceDebugConfig(path=output_path)
        content_writer = ContentWriter(config)

        for span_data in span_rows:
            span_info = trace_state.spans.get(span_data.span_id)
            if not span_info:
                continue
            self._write_span_files(span_data, span_info, content_writer)

        self._finalize_trace(trace_state)

        if include_documents:
            doc_sha256s = self._extract_document_refs(span_rows)
            if doc_sha256s:
                self._download_documents(doc_sha256s, output_path)

        if follow_children and run_rows:
            parent_run_id = run_rows[0][0]
            child_ids = self._find_child_executions(parent_run_id)
            for child_exec_id, child_run_id in child_ids:
                child_path = output_path / f"child_{child_run_id}"
                self.download_trace(
                    UUID(child_exec_id),
                    child_path,
                    include_documents=include_documents,
                    follow_children=True,
                )

        return output_path

    def _build_trace(self, span_rows: list[_SpanData], output_path: Path, trace_name: str) -> TraceState:
        """Build TraceState and create directory hierarchy."""
        trace = TraceState(
            trace_id=span_rows[0].trace_id if span_rows else "unknown",
            name=trace_name,
            path=output_path,
            start_time=span_rows[0].start_time if span_rows else datetime.now(UTC),
        )

        for idx, span_data in enumerate(span_rows, start=1):
            effective_order = span_data.span_order if span_data.span_order > 0 else idx

            parent_path = output_path
            depth = 0
            if span_data.parent_span_id and span_data.parent_span_id in trace.spans:
                parent_info = trace.spans[span_data.parent_span_id]
                parent_path = parent_info.path
                depth = parent_info.depth + 1

            span_dir = self._make_span_dir(parent_path, effective_order, span_data.name)

            span_info = SpanInfo(
                span_id=span_data.span_id,
                parent_id=span_data.parent_span_id,
                name=span_data.name,
                span_type=span_data.span_type,
                status=span_data.status,
                start_time=span_data.start_time,
                path=span_dir,
                depth=depth,
                order=effective_order,
                end_time=span_data.end_time,
                duration_ms=span_data.duration_ms,
            )

            attrs = _parse_json_safe(span_data.attributes_json)
            if isinstance(attrs, dict):
                llm_info = self._extract_llm_info(attrs, span_data)
                if llm_info:
                    span_info.llm_info = llm_info
                    trace.llm_call_count += 1
                    trace.total_tokens += llm_info.get("total_tokens", 0)
                    trace.total_cost += llm_info.get("cost", 0.0)

            trace.spans[span_data.span_id] = span_info
            trace.span_counter = max(trace.span_counter, effective_order)

            if span_data.parent_span_id is None:
                trace.root_span_id = span_data.span_id

            if span_data.parent_span_id and span_data.parent_span_id in trace.spans:
                trace.spans[span_data.parent_span_id].children.append(span_data.span_id)

        return trace

    @staticmethod
    def _make_span_dir(parent_path: Path, order: int, name: str) -> Path:
        """Create a span directory with NNN_name prefix."""
        safe_name = LocalTraceWriter._sanitize_name(name)
        width = 3 if order < 1000 else len(str(order))
        dir_name = f"{order:0{width}d}_{safe_name}"
        span_dir = parent_path / dir_name
        span_dir.mkdir(parents=True, exist_ok=True)
        return span_dir

    @staticmethod
    def _extract_llm_info(attrs: dict[str, Any], span_data: _SpanData) -> dict[str, Any] | None:
        """Extract LLM info from attributes and span data."""
        input_tokens = attrs.get("gen_ai.usage.input_tokens") or attrs.get("gen_ai.usage.prompt_tokens")
        output_tokens = attrs.get("gen_ai.usage.output_tokens") or attrs.get("gen_ai.usage.completion_tokens")

        if input_tokens is None and output_tokens is None and not span_data.llm_model:
            return None

        return {
            "model": span_data.llm_model or attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model"),
            "provider": attrs.get("gen_ai.system"),
            "input_tokens": int(input_tokens or span_data.tokens_input or 0),
            "output_tokens": int(output_tokens or span_data.tokens_output or 0),
            "cached_tokens": int(attrs.get("gen_ai.usage.cache_read_input_tokens", 0)),
            "total_tokens": int(input_tokens or span_data.tokens_input or 0) + int(output_tokens or span_data.tokens_output or 0),
            "cost": float(attrs.get("gen_ai.usage.cost", span_data.cost or 0.0)),
            "purpose": attrs.get("purpose"),
        }

    def _write_span_files(
        self,
        span_data: _SpanData,
        span_info: SpanInfo,
        content_writer: ContentWriter,
    ) -> None:
        """Write all files for a single span."""
        span_dir = span_info.path

        input_content = _parse_json_safe(span_data.input_json)
        output_content = _parse_json_safe(span_data.output_json)

        input_ref = content_writer.write(input_content, span_dir, "input")
        output_ref = content_writer.write(output_content, span_dir, "output")

        attrs = _parse_json_safe(span_data.attributes_json) or {}
        span_meta = self._build_span_meta(span_data, span_info, input_ref, output_ref, attrs)
        (span_dir / "span.yaml").write_text(
            yaml.dump(span_meta, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        if span_data.replay_payload:
            self._write_replay_file(span_data.replay_payload, span_dir)

        events = _parse_json_safe(span_data.events_json)
        if isinstance(events, list) and events:
            (span_dir / "events.yaml").write_text(
                yaml.dump(events, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )

    @staticmethod
    def _build_span_meta(
        span_data: _SpanData,
        span_info: SpanInfo,
        input_ref: dict[str, Any],
        output_ref: dict[str, Any],
        attrs: dict[str, Any],
    ) -> dict[str, Any]:
        """Build span metadata dictionary matching LocalTraceWriter format."""
        meta: dict[str, Any] = {
            "span_id": span_data.span_id,
            "trace_id": span_data.trace_id,
            "parent_id": span_data.parent_span_id,
            "name": span_data.name,
            "type": span_data.span_type,
            "timing": {
                "start": span_data.start_time.isoformat(),
                "end": span_data.end_time.isoformat() if span_data.end_time else None,
                "duration_ms": span_data.duration_ms,
            },
            "status": span_data.status,
        }

        if span_info.llm_info:
            meta["llm"] = span_info.llm_info

        if span_info.description:
            meta["description"] = span_info.description

        meta["input"] = input_ref
        meta["output"] = output_ref

        if span_data.status == "failed":
            error_msg = attrs.get("status_description", "")
            if error_msg:
                meta["error"] = {"message": error_msg}

        if isinstance(attrs, dict) and attrs:
            meta["attributes"] = attrs

        return meta

    @staticmethod
    def _write_replay_file(replay_payload: str, span_dir: Path) -> None:
        """Write replay YAML file from payload string."""
        try:
            replay_data = json.loads(replay_payload)
            payload_type = replay_data.get("payload_type", "conversation")
            filename = _REPLAY_FILENAMES.get(payload_type, "replay.yaml")
            (span_dir / filename).write_text(
                yaml.dump(replay_data, default_flow_style=False, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to write replay file in %s: %s", span_dir, e)

    @staticmethod
    def _finalize_trace(trace: TraceState) -> None:
        """Generate summary and index files."""
        sorted_spans = sorted(trace.spans.values(), key=lambda s: s.order)
        LocalTraceWriter._write_llm_index(trace, sorted_spans)
        LocalTraceWriter._write_errors_index(trace, sorted_spans)

        summary = generate_summary(trace)
        (trace.path / "summary.md").write_text(summary, encoding="utf-8")
        costs = generate_costs(trace)
        if costs:
            (trace.path / "costs.md").write_text(costs, encoding="utf-8")

    @staticmethod
    def _extract_document_refs(spans: list[_SpanData]) -> set[str]:
        """Extract all document SHA256 references from replay payloads."""
        sha256s: set[str] = set()
        for span in spans:
            if not span.replay_payload:
                continue
            payload = _parse_json_safe(span.replay_payload)
            if isinstance(payload, dict):
                _walk_doc_refs(payload, sha256s)
        return sha256s

    def _download_documents(self, sha256s: set[str], output_path: Path) -> None:
        """Download documents and write in LocalDocumentStore layout for replay."""
        result = self._client.query(_DOCUMENT_INDEX_QUERY, parameters={"sha256s": list(sha256s)})
        if not result.result_rows:
            logger.warning("No documents found in ClickHouse for %d SHA256s", len(sha256s))
            return

        content_sha256s: set[str] = set()
        for row in result.result_rows:
            content_sha256s.add(row[1])
            for att_sha in row[10] or []:
                content_sha256s.add(att_sha)

        content_by_sha = self._fetch_content_blobs(content_sha256s)

        for row in result.result_rows:
            self._write_document_files(row, content_by_sha, output_path)

    def _fetch_content_blobs(self, sha256s: set[str]) -> dict[str, bytes]:
        """Batch-fetch content blobs from document_content table."""
        result = self._client.query(_DOCUMENT_CONTENT_QUERY, parameters={"sha256s": list(sha256s)})
        blobs: dict[str, bytes] = {}
        for row in result.result_rows:
            sha = row[0] if isinstance(row[0], str) else row[0].decode("utf-8")
            blobs[sha] = _decode_content_blob(row[1], row[2])
        return blobs

    @staticmethod
    def _write_document_files(row: tuple, content_by_sha: dict[str, bytes], output_path: Path) -> None:
        """Write a single document in LocalDocumentStore layout."""
        doc_sha256, content_sha256, class_name, name = row[0], row[1], row[2], row[3]
        doc_content = content_by_sha.get(content_sha256)
        if doc_content is None:
            return

        class_dir = output_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(name).stem
        suffix = Path(name).suffix or ".bin"
        base_name = f"{stem}_{doc_sha256[:DOC_ID_LENGTH]}{suffix}"

        (class_dir / base_name).write_bytes(doc_content)

        meta = {
            "document_sha256": doc_sha256,
            "content_sha256": content_sha256,
            "class_name": class_name,
            "name": name,
            "description": row[4] or "",
            "mime_type": row[5] or "",
            "derived_from": list(row[6] or []),
            "triggered_by": list(row[7] or []),
        }
        (class_dir / f"{base_name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        att_names = row[8] or []
        att_descriptions = row[9] or []
        att_sha256s = row[10] or []
        if att_names and att_sha256s:
            meta["attachments"] = [{"name": n, "description": d} for n, d in zip(att_names, att_descriptions, strict=False)]
            att_dir = class_dir / f"{base_name}.att"
            att_dir.mkdir(exist_ok=True)
            for att_name, att_sha in zip(att_names, att_sha256s, strict=False):
                att_content = content_by_sha.get(att_sha)
                if att_content:
                    (att_dir / att_name).write_bytes(att_content)

    def _find_child_executions(self, parent_run_id: str) -> list[tuple[str, str]]:
        """Find execution_ids of child pipelines triggered by this run."""
        result = self._client.query(
            _CHILD_RUNS_QUERY,
            parameters={"prefix": f"{parent_run_id}-%", "parent": parent_run_id},
        )
        return [(str(row[0]), row[1]) for row in result.result_rows]


async def download_trace(
    execution_id: UUID,
    output_path: Path,
    *,
    host: str,
    port: int = 8443,
    database: str = "default",
    username: str = "default",
    password: str = "",
    secure: bool = True,
    include_documents: bool = False,
    follow_children: bool = False,
) -> Path:
    """Async convenience wrapper for downloading a trace from ClickHouse."""
    import clickhouse_connect  # noqa: PLC0415

    loop = asyncio.get_running_loop()

    def _sync_download() -> Path:
        client = clickhouse_connect.get_client(  # pyright: ignore[reportUnknownMemberType]
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            secure=secure,
        )
        downloader = TraceDownloader(client=client)
        return downloader.download_trace(
            execution_id,
            output_path,
            include_documents=include_documents,
            follow_children=follow_children,
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        return await loop.run_in_executor(pool, _sync_download)
