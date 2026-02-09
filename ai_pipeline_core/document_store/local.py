"""Local filesystem document store for CLI/debug mode.

Layout:
    {base_path}/{ClassName}/{filename}           <- raw content
    {base_path}/{ClassName}/{filename}.meta.json  <- metadata
    {base_path}/{ClassName}/{filename}.att/        <- attachments directory
"""

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, TypeVar

from ai_pipeline_core.document_store._models import DocumentNode
from ai_pipeline_core.document_store._summary import SummaryGenerator
from ai_pipeline_core.document_store._summary_worker import SummaryWorker
from ai_pipeline_core.documents._context_vars import suppress_registration
from ai_pipeline_core.documents._hashing import compute_content_sha256, compute_document_sha256
from ai_pipeline_core.documents._types import DocumentSha256, RunScope
from ai_pipeline_core.documents.attachment import Attachment
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

_D = TypeVar("_D", bound=Document)

DOC_ID_LENGTH = 6


def _safe_filename(name: str, doc_sha256: str) -> str:
    """Build collision-safe filename: {stem}_{sha256[:DOC_ID_LENGTH]}{suffix}."""
    p = Path(name)
    return f"{p.stem}_{doc_sha256[:DOC_ID_LENGTH]}{p.suffix}"


class LocalDocumentStore:
    """Filesystem-backed document store for local development and debugging.

    Documents are stored as browsable files organized by class name.
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
        """Load documents by type from the run scope directory."""
        return await asyncio.to_thread(self._load_sync, run_scope, document_types)

    async def has_documents(self, run_scope: RunScope, document_type: type[Document]) -> bool:
        """Check for meta files in the type's directory without loading content."""
        return await asyncio.to_thread(self._has_documents_sync, run_scope, document_type)

    async def check_existing(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Scan all meta files to find matching document_sha256 values."""
        return await asyncio.to_thread(self._check_existing_sync, sha256s)

    async def update_summary(self, document_sha256: DocumentSha256, summary: str) -> None:
        """Update summary in all .meta.json files for this document across all scopes."""
        await asyncio.to_thread(self._update_summary_sync, document_sha256, summary)

    async def load_summaries(self, document_sha256s: list[DocumentSha256]) -> dict[DocumentSha256, str]:
        """Load summaries from .meta.json files across all scopes."""
        return await asyncio.to_thread(self._load_summaries_sync, document_sha256s)

    async def load_by_sha256s(self, sha256s: list[DocumentSha256], document_type: type[_D], run_scope: RunScope | None = None) -> dict[DocumentSha256, _D]:
        """Batch-load documents by SHA256. Searches all directories — class_name is not enforced."""
        return await asyncio.to_thread(self._load_by_sha256s_sync, sha256s, document_type, run_scope)  # pyright: ignore[reportReturnType]

    async def load_nodes_by_sha256s(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentNode]:
        """Batch-load lightweight metadata by SHA256, searching all scopes."""
        return await asyncio.to_thread(self._load_nodes_by_sha256s_sync, sha256s)

    async def load_scope_metadata(self, run_scope: RunScope) -> list[DocumentNode]:
        """Load lightweight metadata for all documents in a run scope."""
        return await asyncio.to_thread(self._load_scope_metadata_sync, run_scope)

    def flush(self) -> None:
        """Block until all pending document summaries are processed."""
        if self._summary_worker:
            self._summary_worker.flush()

    def shutdown(self) -> None:
        """Flush pending summaries and stop the summary worker."""
        if self._summary_worker:
            self._summary_worker.shutdown()

    # --- Sync implementation (called via asyncio.to_thread) ---

    def _scope_path(self, run_scope: RunScope) -> Path:
        if ".." in run_scope:
            raise ValueError(f"run_scope contains path traversal '..': {run_scope!r}")
        if "\\" in run_scope:
            raise ValueError(f"run_scope contains backslash: {run_scope!r}")
        resolved = (self._base_path / run_scope).resolve()
        if not resolved.is_relative_to(self._base_path.resolve()):
            raise ValueError(f"run_scope escapes base path: {run_scope!r}")
        return self._base_path / run_scope

    def _cache_meta_path(self, doc_sha256: str, meta_path: Path) -> None:
        """Add a meta path to the cache for a given document SHA256."""
        with self._cache_lock:
            paths = self._meta_path_cache.setdefault(doc_sha256, [])
            if meta_path not in paths:
                paths.append(meta_path)

    def _save_sync(self, document: Document, run_scope: RunScope) -> bool:
        canonical = document.__class__.__name__
        doc_dir = self._scope_path(run_scope) / canonical
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
                # Populate cache even for idempotent saves
                self._cache_meta_path(doc_sha256, meta_path)
                return False  # Idempotent — same document already saved
            if existing_meta:
                logger.warning(
                    f"Overwriting document '{document.name}' in '{canonical}': "
                    f"existing SHA256 {existing_meta.get('document_sha256', '?')[:12]}... "
                    f"differs from new {doc_sha256[:12]}..."
                )

        # Write content before meta (crash safety)
        content_path.write_bytes(document.content)

        # Write attachments
        att_meta_list: list[dict[str, Any]] = []
        if document.attachments:
            att_dir = doc_dir / f"{safe_name}.att"
            att_dir.mkdir(exist_ok=True)
            for att in document.attachments:
                att_path = att_dir / att.name
                if not att_path.resolve().is_relative_to(att_dir.resolve()):
                    raise ValueError(f"Path traversal detected: attachment name '{att.name}' escapes store directory")
                att_path.write_bytes(att.content)
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
            "sources": list(document.sources),
            "origins": list(document.origins),
            "mime_type": document.mime_type,
            "attachments": att_meta_list,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Cache meta path for summary updates
        self._cache_meta_path(doc_sha256, meta_path)
        return True

    def _load_sync(self, run_scope: RunScope, document_types: list[type[Document]]) -> list[Document]:
        scope_path = self._scope_path(run_scope)
        if not scope_path.exists():
            return []

        # Build reverse map: class_name -> document type
        type_by_name: dict[str, type[Document]] = {}
        for doc_type in document_types:
            type_by_name[doc_type.__name__] = doc_type

        documents: list[Document] = []

        with suppress_registration():
            for class_name, doc_type in type_by_name.items():
                type_dir = scope_path / class_name
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
                logger.warning(f"Attachment file {att_path} missing, skipping")
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

            fs_name = meta_path.name.removesuffix(".meta.json")
            content_path = type_dir / fs_name

            if not content_path.exists():
                logger.warning(f"Meta file {meta_path} has no corresponding content file, skipping")
                continue

            content = content_path.read_bytes()
            attachments = self._load_attachments(type_dir, fs_name, meta)
            original_name = meta.get("name", fs_name)

            doc = doc_type(
                name=original_name,
                content=content,
                description=meta.get("description"),
                sources=tuple(meta.get("sources", ())),
                origins=tuple(meta.get("origins", ())),
                attachments=attachments,
            )
            out.append(doc)

    def _has_documents_sync(self, run_scope: RunScope, document_type: type[Document]) -> bool:
        """Check for meta files in the type's directory without loading content."""
        scope_path = self._scope_path(run_scope)
        canonical = document_type.__name__
        type_dir = scope_path / canonical
        if not type_dir.is_dir():
            return False
        return any(type_dir.glob("*.meta.json"))

    def _check_existing_sync(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Scan all meta files to find matching document_sha256 values."""
        target = set(sha256s)
        found: set[DocumentSha256] = set()
        if not self._base_path.exists():
            return found

        for meta_path in self._base_path.rglob("*.meta.json"):
            meta = self._read_meta(meta_path)
            doc_sha = meta.get("document_sha256") if meta else None
            if isinstance(doc_sha, str) and doc_sha in target:
                found.add(DocumentSha256(doc_sha))
                if found == target:
                    break
        return found

    def _update_summary_sync(self, document_sha256: DocumentSha256, summary: str) -> None:
        """Update summary in ALL .meta.json files for this document across all scopes."""
        # Always scan to ensure we find all copies (cache may be incomplete)
        all_paths = self._find_all_meta_by_sha256(document_sha256)
        if not all_paths:
            return

        for meta_path in all_paths:
            meta = self._read_meta(meta_path)
            if meta is None:
                continue
            meta["summary"] = summary
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self._cache_meta_path(document_sha256, meta_path)

    def _load_summaries_sync(self, document_sha256s: list[DocumentSha256]) -> dict[DocumentSha256, str]:
        """Scan meta files across all scopes for matching sha256s and return their summaries."""
        if not document_sha256s or not self._base_path.exists():
            return {}
        target = set(document_sha256s)
        result: dict[DocumentSha256, str] = {}

        for meta_path in self._base_path.rglob("*.meta.json"):
            meta = self._read_meta(meta_path)
            if meta is None:
                continue
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

        if run_scope is not None:
            scope_path = self._scope_path(run_scope)
            if not scope_path.exists():
                return {}
            self._find_by_sha256s_in_path(target, scope_path, document_type, result)
            return result

        # Cross-scope lookup — search all meta files under base_path
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
        """Search for documents by SHA256 under a path. Checks cache first, then scans all meta files."""
        remaining = set(target)

        # Cache hit path
        for sha256 in list(remaining):
            with self._cache_lock:
                cached_paths = list(self._meta_path_cache.get(sha256, []))
            for cached_meta_path in cached_paths:
                if cached_meta_path.is_relative_to(search_path) and cached_meta_path.exists():
                    meta = self._read_meta(cached_meta_path)
                    if meta is not None and meta.get("document_sha256") == sha256:
                        doc = self._construct_document_from_meta(cached_meta_path.parent, cached_meta_path, meta, document_type)
                        if doc is not None:
                            result[sha256] = doc
                            remaining.discard(sha256)
                        break

        if not remaining:
            return

        # Scan all meta files under search_path for remaining
        for meta_path in search_path.rglob("*.meta.json"):
            if not remaining:
                break
            meta = self._read_meta(meta_path)
            if meta is None:
                continue
            sha = meta.get("document_sha256")
            if not isinstance(sha, str) or sha not in remaining or sha in result:
                continue
            doc_sha = DocumentSha256(sha)
            self._cache_meta_path(doc_sha, meta_path)
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

        with suppress_registration():
            return doc_type(
                name=original_name,
                content=content,
                description=meta.get("description"),
                sources=tuple(meta.get("sources", ())),
                origins=tuple(meta.get("origins", ())),
                attachments=attachments,
            )

    def _load_nodes_by_sha256s_sync(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentNode]:
        """Scan all meta files across all scopes for matching document_sha256 values."""
        if not sha256s or not self._base_path.exists():
            return {}
        target = set(sha256s)
        result: dict[DocumentSha256, DocumentNode] = {}
        for meta_path in self._base_path.rglob("*.meta.json"):
            if result.keys() >= target:
                break
            meta = self._read_meta(meta_path)
            if meta is None:
                continue
            sha256_raw = meta.get("document_sha256")
            if not isinstance(sha256_raw, str) or sha256_raw not in target or sha256_raw in result:
                continue
            doc_sha = DocumentSha256(sha256_raw)
            result[doc_sha] = DocumentNode(
                sha256=doc_sha,
                class_name=meta.get("class_name", ""),
                name=meta.get("name", ""),
                description=meta.get("description") or "",
                sources=tuple(meta.get("sources", ())),
                origins=tuple(meta.get("origins", ())),
                summary=meta.get("summary") or "",
            )
        return result

    def _load_scope_metadata_sync(self, run_scope: RunScope) -> list[DocumentNode]:
        """Scan all meta files in a run scope and return lightweight metadata."""
        scope_path = self._scope_path(run_scope)
        if not scope_path.exists():
            return []

        nodes: list[DocumentNode] = []
        for meta_path in scope_path.rglob("*.meta.json"):
            meta = self._read_meta(meta_path)
            if meta is None:
                continue
            doc_sha256_raw = meta.get("document_sha256")
            if not doc_sha256_raw:
                continue
            nodes.append(
                DocumentNode(
                    sha256=DocumentSha256(doc_sha256_raw),
                    class_name=meta.get("class_name", ""),
                    name=meta.get("name", ""),
                    description=meta.get("description") or "",
                    sources=tuple(meta.get("sources", ())),
                    origins=tuple(meta.get("origins", ())),
                    summary=meta.get("summary") or "",
                )
            )
        return nodes

    def _find_all_meta_by_sha256(self, document_sha256: DocumentSha256) -> list[Path]:
        """Scan all meta files to find ones matching the given sha256."""
        results: list[Path] = []
        if not self._base_path.exists():
            return results
        for meta_path in self._base_path.rglob("*.meta.json"):
            meta = self._read_meta(meta_path)
            if meta and meta.get("document_sha256") == document_sha256:
                results.append(meta_path)
        return results

    @staticmethod
    def _read_meta(meta_path: Path) -> dict[str, Any] | None:
        """Read and parse a meta.json file, returning None on any error."""
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read meta file {meta_path}: {e}")
            return None
