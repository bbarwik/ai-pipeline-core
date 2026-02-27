"""Local filesystem document store for CLI/debug mode.

Layout:
    {base_path}/{ClassName}/{filename}           <- raw content
    {base_path}/{ClassName}/{filename}.meta.json  <- metadata
    {base_path}/{ClassName}/{filename}.att/        <- attachments directory
"""

import asyncio
import contextlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from ai_pipeline_core.document_store._models import DocumentNode, FlowCompletion
from ai_pipeline_core.document_store._summary_worker import SummaryGenerator, SummaryWorker
from ai_pipeline_core.documents._context import DocumentSha256, RunScope, _suppress_document_registration
from ai_pipeline_core.documents._hashing import compute_content_sha256, compute_document_sha256
from ai_pipeline_core.documents.attachment import Attachment
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

__all__ = [
    "LocalDocumentStore",
]

_D = TypeVar("_D", bound=Document)
_T = TypeVar("_T")

DOC_ID_LENGTH = 6


def _safe_filename(name: str, doc_sha256: str) -> str:
    """Build collision-safe filename: {stem}_{sha256[:DOC_ID_LENGTH]}{suffix}."""
    p = Path(name)
    return f"{p.stem}_{doc_sha256[:DOC_ID_LENGTH]}{p.suffix}"


def _atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    """Write text atomically via tempfile + os.replace (single rename syscall on POSIX)."""
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically via tempfile + os.replace (single rename syscall on POSIX)."""
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def _node_from_meta(doc_sha: DocumentSha256, meta: dict[str, Any]) -> DocumentNode:
    """Build a DocumentNode from a parsed meta.json dict."""
    return DocumentNode(
        sha256=doc_sha,
        class_name=meta.get("class_name", ""),
        name=meta.get("name", ""),
        description=meta.get("description") or "",
        derived_from=tuple(meta.get("derived_from", ())),
        triggered_by=tuple(meta.get("triggered_by", ())),
        summary=meta.get("summary") or "",
    )


class LocalDocumentStore:
    """Filesystem-backed document store for local development and debugging.

    Documents are stored as browsable files organized by class name directly
    under base_path. The run_scope parameter is accepted for protocol compliance
    but ignored for path computation — all documents share the same directory tree.

    Write order (content before meta) ensures crash safety — load() ignores
    content files without a valid .meta.json.
    """

    def __init__(
        self,
        base_path: Path | None = None,
        *,
        summary_generator: SummaryGenerator | None = None,
    ) -> None:
        self._base_path = base_path or Path.cwd()
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

    async def save(self, document: Document, run_scope: RunScope) -> None:
        """Save a document to disk. Idempotent — same SHA256 is a no-op."""
        written = await asyncio.to_thread(self._save_sync, document, run_scope)
        if written and self._summary_worker:
            self._summary_worker.schedule(document)

    async def save_batch(self, documents: list[Document], run_scope: RunScope) -> None:
        """Save multiple documents sequentially."""
        for doc in documents:
            await self.save(doc, run_scope)

    async def load(self, run_scope: RunScope, document_types: list[type[Document]]) -> list[Document]:
        """Load documents by type from the store."""
        return await asyncio.to_thread(self._load_sync, run_scope, document_types)

    async def has_documents(self, run_scope: RunScope, document_type: type[Document], *, max_age: timedelta | None = None) -> bool:
        """Check for meta files in the type's directory without loading content."""
        return await asyncio.to_thread(self._has_documents_sync, run_scope, document_type, max_age)

    async def check_existing(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Scan all meta files to find matching document_sha256 values."""
        return await asyncio.to_thread(self._check_existing_sync, sha256s)

    async def update_summary(self, document_sha256: DocumentSha256, summary: str) -> None:
        """Update summary in all .meta.json files matching this document SHA256."""
        await asyncio.to_thread(self._update_summary_sync, document_sha256, summary)

    async def load_summaries(self, document_sha256s: list[DocumentSha256]) -> dict[DocumentSha256, str]:
        """Load summaries from .meta.json files for the given SHA256s."""
        return await asyncio.to_thread(self._load_summaries_sync, document_sha256s)

    async def load_by_sha256s(self, sha256s: list[DocumentSha256], document_type: type[_D], run_scope: RunScope | None = None) -> dict[DocumentSha256, _D]:
        """Batch-load documents by SHA256. Searches all directories — class_name is not enforced."""
        return await asyncio.to_thread(self._load_by_sha256s_sync, sha256s, document_type, run_scope)  # pyright: ignore[reportReturnType]

    async def load_nodes_by_sha256s(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentNode]:
        """Batch-load lightweight metadata by SHA256."""
        return await asyncio.to_thread(self._load_nodes_by_sha256s_sync, sha256s)

    async def load_scope_metadata(self, run_scope: RunScope) -> list[DocumentNode]:
        """Load lightweight metadata for all documents in the store."""
        return await asyncio.to_thread(self._load_scope_metadata_sync, run_scope)

    async def find_by_source(
        self,
        source_values: list[str],
        document_type: type[Document],
        *,
        max_age: timedelta | None = None,
    ) -> dict[str, Document]:
        """Find most recent document per source value by scanning meta files."""
        return await asyncio.to_thread(self._find_by_source_sync, source_values, document_type, max_age)

    async def save_flow_completion(
        self,
        run_scope: RunScope,
        flow_name: str,
        input_sha256s: tuple[str, ...],
        output_sha256s: tuple[str, ...],
    ) -> None:
        """Save flow completion record as JSON file."""
        await asyncio.to_thread(self._save_flow_completion_sync, run_scope, flow_name, input_sha256s, output_sha256s)

    async def get_flow_completion(
        self,
        run_scope: RunScope,
        flow_name: str,
        *,
        max_age: timedelta | None = None,
    ) -> FlowCompletion | None:
        """Load flow completion record from JSON file."""
        return await asyncio.to_thread(self._get_flow_completion_sync, run_scope, flow_name, max_age)

    def flush(self) -> None:
        """Block until all pending document summaries are processed."""
        if self._summary_worker:
            self._summary_worker.flush()

    def shutdown(self) -> None:
        """Flush pending summaries and stop the summary worker."""
        if self._summary_worker:
            self._summary_worker.shutdown()

    # --- Sync implementation (called via asyncio.to_thread) ---

    def _save_sync(self, document: Document, run_scope: RunScope) -> bool:
        canonical = document.__class__.__name__
        doc_dir = self._base_path / canonical
        doc_dir.mkdir(parents=True, exist_ok=True)

        # Path traversal check on original name before any filesystem operations
        raw_check = doc_dir / document.name
        if not raw_check.resolve().is_relative_to(doc_dir.resolve()):
            raise ValueError(f"Path traversal detected: document name '{document.name}' escapes store directory")

        doc_sha256 = compute_document_sha256(document)
        content_sha256 = compute_content_sha256(document.content)
        safe_name = _safe_filename(document.name, doc_sha256)

        content_path = doc_dir / safe_name
        meta_path = doc_dir / f"{safe_name}.meta.json"

        # Check for concurrent access: if meta exists with different SHA256, log warning
        if meta_path.exists():
            existing_meta = self._read_meta(meta_path)
            if existing_meta and existing_meta.get("document_sha256") == doc_sha256:
                return False  # Idempotent — same document already saved
            if existing_meta:
                logger.warning(
                    "Overwriting document '%s' in '%s': existing SHA256 %s... differs from new %s...",
                    document.name,
                    canonical,
                    existing_meta.get("document_sha256", "?")[:12],
                    doc_sha256[:12],
                )

        # Write content before meta (crash safety)
        _atomic_write_bytes(content_path, document.content)

        # Write attachments
        att_meta_list: list[dict[str, Any]] = []
        if document.attachments:
            att_dir = doc_dir / f"{safe_name}.att"
            att_dir.mkdir(exist_ok=True)
            for att in document.attachments:
                att_path = att_dir / att.name
                if not att_path.resolve().is_relative_to(att_dir.resolve()):
                    raise ValueError(f"Path traversal detected: attachment name '{att.name}' escapes store directory")
                _atomic_write_bytes(att_path, att.content)
                att_meta_list.append({
                    "name": att.name,
                    "description": att.description,
                    "sha256": compute_content_sha256(att.content),
                })

        # Write meta last (crash safety — content is already on disk)
        meta = {
            "name": document.name,
            "document_sha256": doc_sha256,
            "content_sha256": content_sha256,
            "class_name": document.__class__.__name__,
            "description": document.description,
            "derived_from": list(document.derived_from),
            "triggered_by": list(document.triggered_by),
            "mime_type": document.mime_type,
            "attachments": att_meta_list,
            "stored_at": datetime.now(UTC).isoformat(),
        }
        _atomic_write_text(meta_path, json.dumps(meta, indent=2))
        return True

    def _load_sync(self, run_scope: RunScope, document_types: list[type[Document]]) -> list[Document]:
        if not self._base_path.exists():
            return []

        # Build reverse map: class_name -> document type
        type_by_name: dict[str, type[Document]] = {}
        for doc_type in document_types:
            type_by_name[doc_type.__name__] = doc_type

        documents: list[Document] = []

        for class_name, doc_type in type_by_name.items():
            type_dir = self._base_path / class_name
            if not type_dir.is_dir():
                continue
            self._load_type_dir(type_dir, doc_type, documents)

        return documents

    @staticmethod
    def _load_attachments(type_dir: Path, fs_name: str, meta: dict[str, Any]) -> tuple[Attachment, ...] | None:
        """Load attachments from disk for a document. Returns None if no attachments."""
        att_meta_list = meta.get("attachments", [])
        if not att_meta_list:
            return None
        att_dir = type_dir / f"{fs_name}.att"
        att_list: list[Attachment] = []
        for att_meta in att_meta_list:
            att_path = att_dir / att_meta["name"]
            if not att_path.exists():
                logger.warning("Attachment file %s missing, skipping", att_path)
                continue
            att_list.append(
                Attachment(
                    name=att_meta["name"],
                    content=att_path.read_bytes(),
                    description=att_meta.get("description"),
                )
            )
        return tuple(att_list) if att_list else None

    def _load_type_dir(self, type_dir: Path, doc_type: type[Document], out: list[Document]) -> None:
        """Load all documents of a single type from its directory."""
        for meta_path in type_dir.glob("*.meta.json"):
            meta = self._read_meta(meta_path)
            if meta is None:
                continue
            doc = self._construct_document_from_meta(type_dir, meta_path, meta, doc_type)
            if doc is None:
                logger.warning("Meta file %s has no corresponding content file, skipping", meta_path)
                continue
            out.append(doc)

    def _has_documents_sync(self, run_scope: RunScope, document_type: type[Document], max_age: timedelta | None = None) -> bool:
        """Check for meta files in the type's directory without loading content.

        When the document type has a FILES enum, verifies all expected filenames
        are present — not just any document of the type.
        """
        canonical = document_type.__name__
        type_dir = self._base_path / canonical
        if not type_dir.is_dir():
            return False

        expected_files = document_type.get_expected_files()
        if expected_files is None:
            # No FILES enum — any document of this type is sufficient
            if max_age is None:
                return any(type_dir.glob("*.meta.json"))
            cutoff = datetime.now(UTC) - max_age
            for meta_path in type_dir.glob("*.meta.json"):
                meta = self._read_meta(meta_path)
                if meta is not None and self._check_stored_at(meta, cutoff) is not None:
                    return True
            return False

        # FILES enum — all expected filenames must be present
        cutoff = datetime.now(UTC) - max_age if max_age is not None else None
        found_names: set[str] = set()
        for meta_path in type_dir.glob("*.meta.json"):
            meta = self._read_meta(meta_path)
            if meta is None:
                continue
            if cutoff is not None and self._check_stored_at(meta, cutoff) is None:
                continue
            name = meta.get("name")
            if name is not None:
                found_names.add(name)
        return all(f in found_names for f in expected_files)

    def _scan_meta_by_sha256(self, target_sha256s: set[str], builder_fn: Callable[[DocumentSha256, dict[str, Any]], _T]) -> dict[DocumentSha256, _T]:
        """Scan all meta files, match document_sha256 against target set, build results via builder_fn.

        Short-circuits once all targets are found. Skips duplicates.
        """
        if not target_sha256s or not self._base_path.exists():
            return {}
        result: dict[DocumentSha256, _T] = {}
        for _path, meta in self._iter_all_meta(self._base_path):
            sha256_raw = meta.get("document_sha256")
            if not isinstance(sha256_raw, str) or sha256_raw not in target_sha256s or sha256_raw in result:
                continue
            doc_sha = DocumentSha256(sha256_raw)
            result[doc_sha] = builder_fn(doc_sha, meta)
            if len(result) == len(target_sha256s):
                break
        return result

    def _check_existing_sync(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Scan all meta files to find matching document_sha256 values."""
        return set(self._scan_meta_by_sha256(set(sha256s), lambda sha, _meta: sha))

    def _update_summary_sync(self, document_sha256: DocumentSha256, summary: str) -> None:
        """Update summary in all .meta.json files matching this document SHA256."""
        all_paths = self._find_all_meta_by_sha256(document_sha256)
        if not all_paths:
            return

        for meta_path in all_paths:
            meta = self._read_meta(meta_path)
            if meta is None:
                continue
            meta["summary"] = summary
            _atomic_write_text(meta_path, json.dumps(meta, indent=2))

    def _load_summaries_sync(self, document_sha256s: list[DocumentSha256]) -> dict[DocumentSha256, str]:
        """Scan meta files for matching sha256s and return their summaries."""
        if not document_sha256s or not self._base_path.exists():
            return {}
        target = set(document_sha256s)
        result: dict[DocumentSha256, str] = {}

        for _path, meta in self._iter_all_meta(self._base_path):
            sha = meta.get("document_sha256")
            summary = meta.get("summary")
            if isinstance(sha, str) and sha in target and summary:
                result[DocumentSha256(sha)] = summary
                if len(result) == len(target):
                    break
        return result

    def _load_by_sha256s_sync(self, sha256s: list[DocumentSha256], document_type: type[Document], run_scope: RunScope | None) -> dict[DocumentSha256, Document]:
        """Batch-load documents by SHA256. Searches all type directories (class_name not enforced)."""
        if not sha256s:
            return {}
        target = set(sha256s)
        result: dict[DocumentSha256, Document] = {}

        # run_scope is accepted for protocol compliance but ignored for path computation
        if not self._base_path.exists():
            return {}
        self._find_by_sha256s_in_path(target, self._base_path, document_type, result)
        return result

    def _find_by_sha256s_in_path(
        self,
        target: set[DocumentSha256],
        search_path: Path,
        document_type: type[Document],
        result: dict[DocumentSha256, Document],
    ) -> None:
        """Search for documents by SHA256 under a path by scanning all meta files."""
        remaining = set(target)
        for meta_path, meta in self._iter_all_meta(search_path):
            if not remaining:
                break
            sha = meta.get("document_sha256")
            if not isinstance(sha, str) or sha not in remaining or sha in result:
                continue
            doc_sha = DocumentSha256(sha)
            doc = self._construct_document_from_meta(meta_path.parent, meta_path, meta, document_type)
            if doc is not None:
                result[doc_sha] = doc
                remaining.discard(doc_sha)

    def _construct_document_from_meta(
        self,
        type_dir: Path,
        meta_path: Path,
        meta: dict[str, Any],
        doc_type: type[Document],
    ) -> Document | None:
        """Construct a single Document from a meta file on disk."""
        fs_name = meta_path.name.removesuffix(".meta.json")
        content_path = type_dir / fs_name
        if not content_path.exists():
            return None

        content = content_path.read_bytes()
        attachments = self._load_attachments(type_dir, fs_name, meta)
        original_name = meta.get("name", fs_name)

        with _suppress_document_registration():
            return doc_type(
                name=original_name,
                content=content,
                description=meta.get("description"),
                derived_from=tuple(meta.get("derived_from", ())),
                triggered_by=tuple(meta.get("triggered_by", ())),
                attachments=attachments,
            )

    def _load_nodes_by_sha256s_sync(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentNode]:
        """Scan all meta files for matching document_sha256 values."""
        return self._scan_meta_by_sha256(set(sha256s), _node_from_meta)

    def _load_scope_metadata_sync(self, run_scope: RunScope) -> list[DocumentNode]:
        """Scan all meta files and return lightweight metadata."""
        if not self._base_path.exists():
            return []

        nodes: list[DocumentNode] = []
        for _path, meta in self._iter_all_meta(self._base_path):
            doc_sha256_raw = meta.get("document_sha256")
            if not doc_sha256_raw:
                continue
            nodes.append(_node_from_meta(DocumentSha256(doc_sha256_raw), meta))
        return nodes

    @staticmethod
    def _check_stored_at(meta: dict[str, Any], cutoff: datetime | None) -> str | None:
        """Check if meta's stored_at is recent enough. Returns stored_at string or None to skip."""
        stored_at_str = meta.get("stored_at")
        if cutoff is None:
            return stored_at_str or ""
        if not stored_at_str:
            return None
        try:
            return stored_at_str if datetime.fromisoformat(stored_at_str) >= cutoff else None
        except (ValueError, TypeError):
            return None

    def _find_by_source_sync(self, source_values: list[str], document_type: type[Document], max_age: timedelta | None) -> dict[str, Document]:
        """Scan all meta files to find documents with matching derived_from. Returns newest per source value."""
        if not source_values or not self._base_path.exists():
            return {}

        source_set = set(source_values)
        class_name = document_type.__name__
        cutoff = (datetime.now(UTC) - max_age) if max_age is not None else None

        # Collect candidates: {source_value: (stored_at_iso, meta_path, meta)}
        best: dict[str, tuple[str, Path, dict[str, Any]]] = {}

        for meta_path, meta in self._iter_all_meta(self._base_path):
            if meta.get("class_name") != class_name:
                continue

            sort_key = self._check_stored_at(meta, cutoff)
            if sort_key is None:
                continue

            for src in meta.get("derived_from", ()):
                if src in source_set:
                    prev = best.get(src)
                    if prev is None or sort_key > prev[0]:
                        best[src] = (sort_key, meta_path, meta)

        # Construct documents for each match
        result: dict[str, Document] = {}
        for source_val, (_ts, meta_path, meta) in best.items():
            doc = self._construct_document_from_meta(meta_path.parent, meta_path, meta, document_type)
            if doc is not None:
                result[source_val] = doc

        return result

    def _save_flow_completion_sync(
        self,
        run_scope: RunScope,
        flow_name: str,
        input_sha256s: tuple[str, ...],
        output_sha256s: tuple[str, ...],
    ) -> None:
        """Write flow completion JSON to {base_path}/.flow_completions/{scope_dir}/{flow_name}.json."""
        scope_dir = run_scope.replace("/", "__")
        completions_dir = self._base_path / ".flow_completions" / scope_dir
        completions_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "flow_name": flow_name,
            "input_sha256s": list(input_sha256s),
            "output_sha256s": list(output_sha256s),
            "stored_at": datetime.now(UTC).isoformat(),
        }
        _atomic_write_text(completions_dir / f"{flow_name}.json", json.dumps(record, indent=2))

    def _get_flow_completion_sync(self, run_scope: RunScope, flow_name: str, max_age: timedelta | None) -> FlowCompletion | None:
        """Read flow completion JSON from disk. Returns None if missing or expired."""
        scope_dir = run_scope.replace("/", "__")
        path = self._base_path / ".flow_completions" / scope_dir / f"{flow_name}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read flow completion %s: %s", path, e)
            return None
        stored_at_str = data.get("stored_at")
        if not stored_at_str:
            return None
        try:
            stored_at = datetime.fromisoformat(stored_at_str)
        except (ValueError, TypeError):
            return None
        if max_age is not None and stored_at < datetime.now(UTC) - max_age:
            return None
        return FlowCompletion(
            flow_name=data.get("flow_name", flow_name),
            input_sha256s=tuple(data.get("input_sha256s", ())),
            output_sha256s=tuple(data.get("output_sha256s", ())),
            stored_at=stored_at,
        )

    def _find_all_meta_by_sha256(self, document_sha256: DocumentSha256) -> list[Path]:
        """Scan all meta files to find ones matching the given sha256."""
        if not self._base_path.exists():
            return []
        return [path for path, meta in self._iter_all_meta(self._base_path) if meta.get("document_sha256") == document_sha256]

    def _iter_all_meta(self, root: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
        """Yield (meta_path, parsed_meta) for all valid .meta.json files under root."""
        for meta_path in root.rglob("*.meta.json"):
            meta = self._read_meta(meta_path)
            if meta is not None:
                yield meta_path, meta

    @staticmethod
    def _read_meta(meta_path: Path) -> dict[str, Any] | None:
        """Read and parse a meta.json file, returning None on any error."""
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read meta file %s: %s", meta_path, e)
            return None
