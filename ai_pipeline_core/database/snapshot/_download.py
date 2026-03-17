"""Export deployment data as a portable FilesystemDatabase snapshot."""

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from ai_pipeline_core.database._json_helpers import parse_json_object
from ai_pipeline_core.database._protocol import DatabaseReader
from ai_pipeline_core.database._sorting import span_sort_key
from ai_pipeline_core.database._types import DocumentRecord, SpanKind, SpanRecord, SpanStatus
from ai_pipeline_core.database.filesystem._backend import FilesystemDatabase
from ai_pipeline_core.database.filesystem._validation import validate_bundle
from ai_pipeline_core.database.snapshot._spans import generate_costs_from_tree, generate_summary_from_tree
from ai_pipeline_core.database.snapshot._summary import generate_costs, generate_summary

__all__ = ["download_deployment", "generate_run_artifacts"]

VALIDATION_FILENAME = "validation.json"


def _collect_attachment_blob_shas(documents: list[DocumentRecord]) -> set[str]:
    blob_shas: set[str] = set()
    for document in documents:
        blob_shas.add(document.content_sha256)
        blob_shas.update(document.attachment_content_sha256s)
    return blob_shas


def _collect_span_blob_shas(tree: list[SpanRecord]) -> set[str]:
    blob_shas: set[str] = set()
    for span in tree:
        blob_shas.update(span.input_blob_shas)
        blob_shas.update(span.output_blob_shas)
    return blob_shas


def _raise_if_missing_records(
    *,
    record_kind: str,
    expected_shas: set[str],
    actual_shas: set[str],
    deployment_id: UUID,
) -> None:
    missing_shas = sorted(expected_shas - actual_shas)
    if not missing_shas:
        return
    missing_list = ", ".join(missing_shas)
    singular = record_kind[:-1] if record_kind.endswith("s") else record_kind
    msg = (
        f"download_deployment({deployment_id}) could not produce a complete snapshot because {record_kind} "
        f"are missing from the source database: {missing_list}. Persist every referenced {singular} before downloading the deployment tree."
    )
    raise ValueError(msg)


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def _create_staging_path(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output_path.name}-staging-", dir=output_path.parent))


def _replace_output_path(staged_path: Path, output_path: Path) -> None:
    backup_path = output_path.with_name(f".{output_path.name}-backup")
    _remove_path(backup_path)
    if not output_path.exists():
        staged_path.replace(output_path)
        return
    output_path.replace(backup_path)
    try:
        staged_path.replace(output_path)
    except Exception:
        backup_path.replace(output_path)
        raise
    _remove_path(backup_path)


async def _get_tree_logs(
    source: DatabaseReader,
    deployment_ids: list[UUID],
) -> list[Any]:
    return await source.get_deployment_logs_batch(deployment_ids)


def _build_llm_call_lines(tree: list[SpanRecord]) -> list[str]:
    spans_by_id: dict[UUID, SpanRecord] = {span.span_id: span for span in tree}
    lines: list[str] = []
    for span in sorted(tree, key=span_sort_key):
        if span.kind != SpanKind.LLM_ROUND:
            continue
        meta = parse_json_object(span.meta_json, context=f"Span {span.span_id}", field_name="meta_json")
        metrics = parse_json_object(span.metrics_json, context=f"Span {span.span_id}", field_name="metrics_json")
        purpose = ""
        if span.parent_span_id is not None:
            parent = spans_by_id.get(span.parent_span_id)
            if parent is not None:
                parent_meta = parse_json_object(parent.meta_json, context=f"Span {parent.span_id}", field_name="meta_json")
                purpose = parent_meta.get("purpose", "")
        time_taken_ms: float | None = None
        if span.started_at is not None and span.ended_at is not None:
            time_taken_ms = (span.ended_at - span.started_at).total_seconds() * 1000
        raw_request_messages = meta.get("request_messages", [])
        request_messages = raw_request_messages if isinstance(raw_request_messages, list) else []
        raw_response_tool_calls = meta.get("response_tool_calls", [])
        response_tool_calls = raw_response_tool_calls if isinstance(raw_response_tool_calls, list) else []
        lines.append(
            json.dumps(
                {
                    "span_id": str(span.span_id),
                    "parent_span_id": str(span.parent_span_id) if span.parent_span_id is not None else None,
                    "deployment_id": str(span.deployment_id),
                    "run_id": span.run_id,
                    "started_at": span.started_at.isoformat(),
                    "ended_at": span.ended_at.isoformat() if span.ended_at is not None else None,
                    "model": meta.get("model", ""),
                    "round_index": meta.get("round_index", 0),
                    "prompt_tokens": metrics.get("tokens_input", 0),
                    "completion_tokens": metrics.get("tokens_output", 0),
                    "cached_tokens": metrics.get("tokens_cache_read", 0),
                    "reasoning_tokens": metrics.get("tokens_reasoning", 0),
                    "cost_usd": span.cost_usd,
                    "finish_reason": meta.get("finish_reason", ""),
                    "response_id": meta.get("response_id", ""),
                    "tool_call_count": meta.get("tool_call_count", len(response_tool_calls)),
                    "request_message_count": len(request_messages),
                    "response_format_path": meta.get("response_format_path"),
                    "purpose": purpose,
                    "time_taken_ms": time_taken_ms,
                    "first_token_ms": metrics.get("first_token_ms"),
                },
                sort_keys=True,
            )
        )
    return lines


def _build_error_lines(tree: list[SpanRecord]) -> list[str]:
    failed_spans = [span for span in sorted(tree, key=span_sort_key) if span.status == SpanStatus.FAILED]
    if not failed_spans:
        return []
    error_lines = ["# Failures", ""]
    for span in failed_spans:
        error_text = ": ".join(part for part in (span.error_type, span.error_message) if part) or "(no error recorded)"
        error_lines.append(f"- `{span.kind}: {span.name}`")
        error_lines.append(f"  `{error_text}`")
    error_lines.append("")
    return error_lines


def _build_document_lines(tree: list[SpanRecord], documents: dict[str, DocumentRecord]) -> list[str]:
    if not documents:
        return []
    producer_map: dict[str, list[SpanRecord]] = {}
    for span in tree:
        input_shas = set(span.input_document_shas)
        for document_sha in span.output_document_shas:
            if document_sha not in input_shas:
                producer_map.setdefault(document_sha, []).append(span)
    document_lines = ["# Documents", ""]
    for document in sorted(documents.values(), key=lambda item: item.name):
        producers = sorted(producer_map.get(document.document_sha256, []), key=span_sort_key)
        producer_label = ", ".join(f"{span.kind}: {span.name}" for span in producers) if producers else "referenced input"
        document_lines.append(f"- `{document.name}` [{document.document_type}]")
        document_lines.append(
            f"  sha={document.document_sha256}"
            f" content_sha={document.content_sha256}"
            f" mime={document.mime_type}"
            f" size={document.size_bytes}"
            f" producer={producer_label}"
        )
        if document.derived_from:
            document_lines.append(f"  derived_from: {', '.join(document.derived_from)}")
        if document.triggered_by:
            document_lines.append(f"  triggered_by: {', '.join(document.triggered_by)}")
    document_lines.append("")
    return document_lines


async def generate_run_artifacts(
    database: FilesystemDatabase,
    deployment_id: UUID,
    output_path: Path,
    *,
    tree: list[SpanRecord] | None = None,
    documents: dict[str, DocumentRecord] | None = None,
) -> None:
    """Generate summary.md, costs.md, llm_calls.jsonl, errors.md, and documents.md for a filesystem snapshot."""
    if tree is None:
        summary = await generate_summary(database, deployment_id)
        costs = await generate_costs(database, deployment_id)
        tree = await database.get_deployment_tree(deployment_id)
    else:
        summary = generate_summary_from_tree(tree, deployment_id)
        costs = generate_costs_from_tree(tree, deployment_id)
    if documents is None:
        document_shas = await database.get_all_document_shas_for_tree(deployment_id)
        documents = await database.get_documents_batch(sorted(document_shas))
    await asyncio.to_thread((output_path / "summary.md").write_text, summary, encoding="utf-8")
    await asyncio.to_thread((output_path / "costs.md").write_text, costs, encoding="utf-8")

    llm_lines = _build_llm_call_lines(tree)
    await asyncio.to_thread((output_path / "llm_calls.jsonl").write_text, "\n".join(llm_lines) + ("\n" if llm_lines else ""), encoding="utf-8")

    error_lines = _build_error_lines(tree)
    if error_lines:
        await asyncio.to_thread((output_path / "errors.md").write_text, "\n".join(error_lines), encoding="utf-8")

    document_lines = _build_document_lines(tree, documents)
    if document_lines:
        await asyncio.to_thread((output_path / "documents.md").write_text, "\n".join(document_lines), encoding="utf-8")


async def download_deployment(
    source: DatabaseReader,
    deployment_id: UUID,
    output_path: Path,
) -> None:
    """Export a deployment tree as a FilesystemDatabase snapshot."""
    tree = await source.get_deployment_tree(deployment_id)
    staged_path = await asyncio.to_thread(_create_staging_path, output_path)
    target = await asyncio.to_thread(FilesystemDatabase, staged_path)
    committed = False
    try:
        for span in tree:
            await target.insert_span(span)

        document_shas = await source.get_all_document_shas_for_tree(deployment_id)
        documents = await source.get_documents_batch(sorted(document_shas))
        _raise_if_missing_records(
            record_kind="documents",
            expected_shas=document_shas,
            actual_shas=set(documents),
            deployment_id=deployment_id,
        )
        if documents:
            await target.save_document_batch(list(documents.values()))

        blob_shas = _collect_attachment_blob_shas(list(documents.values()))
        blob_shas.update(_collect_span_blob_shas(tree))
        blobs = await source.get_blobs_batch(sorted(blob_shas))
        _raise_if_missing_records(
            record_kind="blobs",
            expected_shas=blob_shas,
            actual_shas=set(blobs),
            deployment_id=deployment_id,
        )
        if blobs:
            await target.save_blob_batch(list(blobs.values()))

        deployment_ids = sorted({span.deployment_id for span in tree}, key=str)
        logs = await _get_tree_logs(source, deployment_ids)
        if logs:
            await target.save_logs_batch(logs)
        else:
            await asyncio.to_thread((staged_path / "logs.jsonl").write_text, "", encoding="utf-8")

        await generate_run_artifacts(
            target,
            deployment_id,
            staged_path,
            tree=tree,
            documents=documents,
        )
        schema_meta = {"schema_version": 3, "source": "download_deployment"}
        await asyncio.to_thread(
            (staged_path / "schema_meta.json").write_text,
            json.dumps(schema_meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        await target.shutdown()
        validation = await asyncio.to_thread(validate_bundle, staged_path)
        await asyncio.to_thread(
            (staged_path / VALIDATION_FILENAME).write_text,
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        await asyncio.to_thread(_replace_output_path, staged_path, output_path)
        committed = True
    finally:
        if not committed:
            await target.shutdown()
            await asyncio.to_thread(_remove_path, staged_path)
