"""Download documents referenced by replay files from ClickHouse.

Scans materialized .trace/ directories for replay YAMLs (conversation.yaml,
task.yaml, flow.yaml), collects $doc_ref SHA256 references, batch-fetches
documents from ClickHouse, and writes them in LocalDocumentStore format
so ai-replay can resolve them.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from ai_pipeline_core.document_store._clickhouse import (
    TABLE_DOCUMENT_CONTENT,
    TABLE_DOCUMENT_INDEX,
    _decode,
    _decode_content,
)
from ai_pipeline_core.document_store._local import DOC_ID_LENGTH
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._debug._config import REPLAY_FILENAME_TO_LABEL

logger = get_pipeline_logger(__name__)

_REPLAY_FILENAMES = frozenset(REPLAY_FILENAME_TO_LABEL.keys())


def fetch_trace_documents(client: Any, output_path: Path) -> tuple[int, int]:
    """Download documents referenced by replay files in the trace directory.

    Scans .trace/ for replay YAMLs, extracts $doc_ref SHA256s, fetches from
    ClickHouse, and writes in LocalDocumentStore format under output_path.

    Returns (found, total) — count of documents written vs total referenced.
    """
    trace_path = output_path / ".trace"
    if not trace_path.is_dir():
        return 0, 0

    doc_refs = _collect_doc_refs(trace_path)
    if not doc_refs:
        return 0, 0

    total = len(doc_refs)
    found, written_shas = _fetch_and_write_documents(client, doc_refs, output_path)

    for sha in sorted(set(doc_refs.keys()) - written_shas):
        class_name, name = doc_refs[sha]
        logger.warning(
            "Document %s (%s, SHA %s...) referenced in replay but not found in ClickHouse.",
            name,
            class_name,
            sha[:12],
        )
    return found, total


def _collect_doc_refs(trace_path: Path) -> dict[str, tuple[str, str]]:
    """Scan replay YAMLs under trace_path, return {sha256: (class_name, name)}."""
    refs: dict[str, tuple[str, str]] = {}
    for yaml_path in trace_path.rglob("*.yaml"):
        if yaml_path.name not in _REPLAY_FILENAMES:
            continue
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if data:
                _extract_doc_refs(data, refs)
        except (yaml.YAMLError, OSError) as e:
            logger.debug("Failed to parse replay file %s: %s", yaml_path, e)
    return refs


def _extract_doc_refs(value: Any, refs: dict[str, tuple[str, str]]) -> None:
    """Walk parsed YAML, collecting {sha256: (class_name, name)} from $doc_ref dicts."""
    if isinstance(value, dict):
        d = cast(dict[str, Any], value)
        if "$doc_ref" in d:
            sha: str = d["$doc_ref"]
            refs[sha] = (str(d.get("class_name", "Document")), str(d.get("name", "unknown")))
        else:
            for v in d.values():
                _extract_doc_refs(v, refs)
    elif isinstance(value, list):
        for item in cast(list[Any], value):
            _extract_doc_refs(item, refs)


def _fetch_and_write_documents(client: Any, doc_refs: dict[str, tuple[str, str]], output_path: Path) -> tuple[int, set[str]]:
    """Fetch documents from ClickHouse and write in LocalDocumentStore format.

    Returns (count_written, set_of_written_sha256s).
    """
    sha256s = list(doc_refs.keys())

    # Query 1: document metadata + main content
    result = client.query(
        f"SELECT di.document_sha256, di.content_sha256, di.class_name, di.name, di.description, "
        f"di.mime_type, di.derived_from, di.triggered_by, "
        f"di.attachment_names, di.attachment_descriptions, di.attachment_sha256s, "
        f"dc.content, length(dc.content) "
        f"FROM {TABLE_DOCUMENT_INDEX} AS di FINAL "
        f"JOIN {TABLE_DOCUMENT_CONTENT} AS dc ON di.content_sha256 = dc.content_sha256 "
        f"WHERE di.document_sha256 IN {{sha256s:Array(String)}}",
        parameters={"sha256s": sha256s},
    )

    all_att_sha256s: set[str] = set()
    docs: list[dict[str, Any]] = []

    for row in result.result_rows:
        doc_sha256 = _decode(row[0])
        att_sha256s = [_decode(s) for s in row[10]]
        all_att_sha256s.update(att_sha256s)
        docs.append({
            "document_sha256": doc_sha256,
            "content_sha256": _decode(row[1]),
            "class_name": _decode(row[2]),
            "name": _decode(row[3]),
            "description": _decode(row[4]) or None,
            "mime_type": _decode(row[5]),
            "derived_from": [_decode(s) for s in row[6]],
            "triggered_by": [_decode(s) for s in row[7]],
            "attachment_names": [_decode(n) for n in row[8]],
            "attachment_descriptions": [_decode(d) for d in row[9]],
            "attachment_sha256s": att_sha256s,
            "content": _decode_content(row[11], row[12]),
        })

    # Query 2: attachment content blobs
    att_content: dict[str, bytes] = {}
    if all_att_sha256s:
        att_result = client.query(
            f"SELECT content_sha256, content, length(content) FROM {TABLE_DOCUMENT_CONTENT} WHERE content_sha256 IN {{sha256s:Array(String)}}",
            parameters={"sha256s": list(all_att_sha256s)},
        )
        att_content = {_decode(r[0]): _decode_content(r[1], r[2]) for r in att_result.result_rows}

    # Write each document
    written_shas: set[str] = set()
    for doc in docs:
        _write_document(doc, att_content, output_path)
        written_shas.add(doc["document_sha256"])

    return len(docs), written_shas


def _write_document(doc: dict[str, Any], att_content: dict[str, bytes], output_path: Path) -> None:
    """Write a single document in LocalDocumentStore format."""
    class_name = doc["class_name"]
    doc_sha256 = doc["document_sha256"]
    name = doc["name"]

    doc_dir = output_path / class_name
    doc_dir.mkdir(parents=True, exist_ok=True)

    p = Path(name)
    safe_name = f"{p.stem}_{doc_sha256[:DOC_ID_LENGTH]}{p.suffix}"
    content_path = doc_dir / safe_name
    meta_path = doc_dir / f"{safe_name}.meta.json"

    content_path.write_bytes(doc["content"])

    # Write attachments
    att_meta_list: list[dict[str, str]] = []
    for i, att_name in enumerate(doc["attachment_names"]):
        att_sha = doc["attachment_sha256s"][i]
        att_bytes = att_content.get(att_sha)
        if att_bytes is None:
            logger.warning("Attachment %s... for document '%s' not found, skipping.", att_sha[:12], name)
            continue
        att_dir = doc_dir / f"{safe_name}.att"
        att_dir.mkdir(exist_ok=True)
        (att_dir / att_name).write_bytes(att_bytes)
        att_meta_list.append({
            "name": att_name,
            "description": doc["attachment_descriptions"][i] if i < len(doc["attachment_descriptions"]) else "",
            "sha256": att_sha,
        })

    meta = {
        "name": name,
        "document_sha256": doc_sha256,
        "content_sha256": doc["content_sha256"],
        "class_name": class_name,
        "description": doc["description"],
        "derived_from": doc["derived_from"],
        "triggered_by": doc["triggered_by"],
        "mime_type": doc["mime_type"],
        "attachments": att_meta_list,
        "stored_at": datetime.now(UTC).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
