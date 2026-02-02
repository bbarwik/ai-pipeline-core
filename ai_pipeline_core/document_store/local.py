"""Local filesystem document store for CLI/debug mode.

Layout:
    {base_path}/{canonical_name}/{filename}           <- raw content
    {base_path}/{canonical_name}/{filename}.meta.json  <- metadata
    {base_path}/{canonical_name}/{filename}.att/        <- attachments directory
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from ai_pipeline_core.document_store._summary import SummaryGenerator
from ai_pipeline_core.document_store._summary_worker import SummaryWorker
from ai_pipeline_core.documents._context_vars import suppress_registration
from ai_pipeline_core.documents._hashing import compute_content_sha256, compute_document_sha256
from ai_pipeline_core.documents.attachment import Attachment
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)


class LocalDocumentStore:
    """Filesystem-backed document store for local development and debugging.

    Documents are stored as browsable files organized by canonical type name.
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
        self._meta_path_cache: dict[str, Path] = {}  # "{run_scope}:{sha256}" -> meta file path
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

    async def save(self, document: Document, run_scope: str) -> None:
        """Save a document to disk. Idempotent — same SHA256 is a no-op."""
        written = await asyncio.to_thread(self._save_sync, document, run_scope)
        if written and self._summary_worker:
            self._summary_worker.schedule(run_scope, document)

    async def save_batch(self, documents: list[Document], run_scope: str) -> None:
        """Save multiple documents sequentially."""
        for doc in documents:
            await self.save(doc, run_scope)

    async def load(self, run_scope: str, document_types: list[type[Document]]) -> list[Document]:
        """Load documents by type from the run scope directory."""
        return await asyncio.to_thread(self._load_sync, run_scope, document_types)

    async def has_documents(self, run_scope: str, document_type: type[Document]) -> bool:
        """Check for meta files in the type's directory without loading content."""
        return await asyncio.to_thread(self._has_documents_sync, run_scope, document_type)

    async def check_existing(self, sha256s: list[str]) -> set[str]:
        """Scan all meta files to find matching document_sha256 values."""
        return await asyncio.to_thread(self._check_existing_sync, sha256s)

    async def update_summary(self, run_scope: str, document_sha256: str, summary: str) -> None:
        """Update summary in the document's .meta.json file."""
        await asyncio.to_thread(self._update_summary_sync, run_scope, document_sha256, summary)

    async def load_summaries(self, run_scope: str, document_sha256s: list[str]) -> dict[str, str]:
        """Load summaries from .meta.json files."""
        return await asyncio.to_thread(self._load_summaries_sync, run_scope, document_sha256s)

    def flush(self) -> None:
        """Block until all pending document summaries are processed."""
        if self._summary_worker:
            self._summary_worker.flush()

    def shutdown(self) -> None:
        """Flush pending summaries and stop the summary worker."""
        if self._summary_worker:
            self._summary_worker.shutdown()

    # --- Sync implementation (called via asyncio.to_thread) ---

    def _scope_path(self, run_scope: str) -> Path:
        return self._base_path / run_scope

    def _save_sync(self, document: Document, run_scope: str) -> bool:
        canonical = document.canonical_name()
        doc_dir = self._scope_path(run_scope) / canonical
        doc_dir.mkdir(parents=True, exist_ok=True)

        content_path = doc_dir / document.name
        if not content_path.resolve().is_relative_to(doc_dir.resolve()):
            raise ValueError(f"Path traversal detected: document name '{document.name}' escapes store directory")
        meta_path = doc_dir / f"{document.name}.meta.json"

        doc_sha256 = compute_document_sha256(document)
        content_sha256 = compute_content_sha256(document.content)

        # Check for concurrent access: if meta exists with different SHA256, log warning
        if meta_path.exists():
            existing_meta = self._read_meta(meta_path)
            if existing_meta and existing_meta.get("document_sha256") == doc_sha256:
                # Populate cache even for idempotent saves
                self._meta_path_cache[f"{run_scope}:{doc_sha256}"] = meta_path
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
            att_dir = doc_dir / f"{document.name}.att"
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
        self._meta_path_cache[f"{run_scope}:{doc_sha256}"] = meta_path
        return True

    def _load_sync(self, run_scope: str, document_types: list[type[Document]]) -> list[Document]:
        scope_path = self._scope_path(run_scope)
        if not scope_path.exists():
            return []

        # Build reverse map: canonical_name -> document type
        type_by_canonical: dict[str, type[Document]] = {}
        for doc_type in document_types:
            cn = doc_type.canonical_name()
            type_by_canonical[cn] = doc_type

        documents: list[Document] = []

        with suppress_registration():
            for canonical, doc_type in type_by_canonical.items():
                type_dir = scope_path / canonical
                if not type_dir.is_dir():
                    continue
                self._load_type_dir(type_dir, doc_type, documents)

        return documents

    def _load_type_dir(self, type_dir: Path, doc_type: type[Document], out: list[Document]) -> None:
        """Load all documents of a single type from its directory."""
        for meta_path in type_dir.glob("*.meta.json"):
            meta = self._read_meta(meta_path)
            if meta is None:
                continue

            content_name = meta_path.name.removesuffix(".meta.json")
            content_path = type_dir / content_name

            if not content_path.exists():
                logger.warning(f"Meta file {meta_path} has no corresponding content file, skipping")
                continue

            content = content_path.read_bytes()

            # Load attachments
            attachments: tuple[Attachment, ...] = ()
            att_meta_list = meta.get("attachments", [])
            if att_meta_list:
                att_dir = type_dir / f"{content_name}.att"
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
                attachments = tuple(att_list)

            doc = doc_type(
                name=content_name,
                content=content,
                description=meta.get("description"),
                sources=tuple(meta.get("sources", ())),
                origins=tuple(meta.get("origins", ())),
                attachments=attachments or None,
            )
            out.append(doc)

    def _has_documents_sync(self, run_scope: str, document_type: type[Document]) -> bool:
        """Check for meta files in the type's directory without loading content."""
        scope_path = self._scope_path(run_scope)
        canonical = document_type.canonical_name()
        type_dir = scope_path / canonical
        if not type_dir.is_dir():
            return False
        return any(type_dir.glob("*.meta.json"))

    def _check_existing_sync(self, sha256s: list[str]) -> set[str]:
        """Scan all meta files to find matching document_sha256 values."""
        target = set(sha256s)
        found: set[str] = set()
        if not self._base_path.exists():
            return found

        for meta_path in self._base_path.rglob("*.meta.json"):
            meta = self._read_meta(meta_path)
            if meta and meta.get("document_sha256") in target:
                found.add(meta["document_sha256"])
                if found == target:
                    break
        return found

    def _update_summary_sync(self, run_scope: str, document_sha256: str, summary: str) -> None:
        """Update summary in the document's .meta.json file."""
        cache_key = f"{run_scope}:{document_sha256}"
        meta_path = self._meta_path_cache.get(cache_key)

        # Fallback: scan for the meta file if not cached
        if meta_path is None or not meta_path.exists():
            meta_path = self._find_meta_by_sha256(run_scope, document_sha256)
            if meta_path is None:
                return

        meta = self._read_meta(meta_path)
        if meta is None:
            return

        meta["summary"] = summary
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._meta_path_cache[cache_key] = meta_path

    def _load_summaries_sync(self, run_scope: str, document_sha256s: list[str]) -> dict[str, str]:
        """Scan meta files for matching sha256s and return their summaries."""
        target = set(document_sha256s)
        result: dict[str, str] = {}
        scope_path = self._scope_path(run_scope)
        if not scope_path.exists():
            return result

        for meta_path in scope_path.rglob("*.meta.json"):
            meta = self._read_meta(meta_path)
            if meta is None:
                continue
            sha = meta.get("document_sha256")
            summary = meta.get("summary")
            if sha in target and summary:
                result[sha] = summary
                if len(result) == len(target):
                    break
        return result

    def _find_meta_by_sha256(self, run_scope: str, document_sha256: str) -> Path | None:
        """Scan meta files in run_scope to find one matching the given sha256."""
        scope_path = self._scope_path(run_scope)
        if not scope_path.exists():
            return None
        for meta_path in scope_path.rglob("*.meta.json"):
            meta = self._read_meta(meta_path)
            if meta and meta.get("document_sha256") == document_sha256:
                return meta_path
        return None

    @staticmethod
    def _read_meta(meta_path: Path) -> dict[str, Any] | None:
        """Read and parse a meta.json file, returning None on any error."""
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read meta file {meta_path}: {e}")
            return None
