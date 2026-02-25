"""Document reference resolution from LocalDocumentStore.

Resolves $doc_ref SHA256 references to full Document instances by scanning
the store's filesystem layout: {store_base}/{class_name}/*_{sha256[:6]}*.meta.json
"""

import json
from pathlib import Path

from ai_pipeline_core.document_store._local import DOC_ID_LENGTH, LocalDocumentStore
from ai_pipeline_core.documents._context_vars import _suppress_document_registration
from ai_pipeline_core.documents.document import Document, _class_name_registry

from .types import DocumentRef

__all__ = ["resolve_document_ref"]


def _find_document_class(class_name: str) -> type[Document]:
    """Find a Document subclass by name, checking registry first then all subclasses."""
    if class_name in _class_name_registry:
        return _class_name_registry[class_name]

    # Search all loaded Document subclasses (covers test-defined classes not in registry)
    queue: list[type[Document]] = list(Document.__subclasses__())
    while queue:
        cls = queue.pop()
        if cls.__name__ == class_name:
            return cls
        queue.extend(cls.__subclasses__())

    raise ValueError(f"No Document subclass found for class_name '{class_name}'. Import the module that defines this Document subclass before replaying.")


def resolve_document_ref(ref: DocumentRef, store_base: Path) -> Document:
    """Resolve a DocumentRef to a full Document by loading from the store filesystem.

    Searches {store_base}/{class_name}/ for files matching the SHA256 prefix,
    verifies the full hash, and constructs the Document instance.
    """
    type_dir = store_base / ref.class_name
    prefix = ref.doc_ref[:DOC_ID_LENGTH]

    if not type_dir.is_dir():
        raise FileNotFoundError(
            f"Document directory not found: {type_dir}. "
            f"Looking for {ref.class_name} with SHA256 {ref.doc_ref[:12]}... "
            f"Run the pipeline first to populate the store."
        )

    for meta_path in type_dir.glob(f"*_{prefix}*.meta.json"):
        meta_text = meta_path.read_text(encoding="utf-8")
        meta = json.loads(meta_text)
        if meta.get("document_sha256") != ref.doc_ref:
            continue

        # Found matching document -- load content and construct
        fs_name = meta_path.name.removesuffix(".meta.json")
        content_path = meta_path.parent / fs_name
        if not content_path.exists():
            raise FileNotFoundError(f"Content file missing for {ref.class_name}/{fs_name}. Meta file exists at {meta_path} but content is gone.")

        doc_cls = _find_document_class(ref.class_name)
        content_bytes = content_path.read_bytes()
        attachments = LocalDocumentStore._load_attachments(type_dir, fs_name, meta)

        with _suppress_document_registration():
            return doc_cls(
                name=meta.get("name", ref.name),
                content=content_bytes,
                description=meta.get("description"),
                derived_from=tuple(meta.get("derived_from", ())),
                triggered_by=tuple(meta.get("triggered_by", ())),
                attachments=attachments,
            )

    raise FileNotFoundError(
        f"Document not found: {ref.class_name} with SHA256 {ref.doc_ref[:12]}... Searched in {type_dir}. Run the pipeline first to populate the store."
    )
