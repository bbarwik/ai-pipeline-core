"""Filesystem database overlay/fork helpers for replay and debug execution."""

import json
from datetime import UTC, datetime
from pathlib import Path

from ai_pipeline_core.database.filesystem._backend import (
    BLOBS_DIRNAME,
    DOCUMENTS_DIRNAME,
    RUNS_DIRNAME,
    SPANS_DIRNAME,
    FilesystemDatabase,
)

__all__ = ["create_debug_sink"]

_DB_ARTIFACT_DIRS = frozenset({SPANS_DIRNAME, DOCUMENTS_DIRNAME, BLOBS_DIRNAME, RUNS_DIRNAME})


def create_debug_sink(
    output_dir: Path,
    *,
    parent: FilesystemDatabase | None = None,
) -> FilesystemDatabase:
    """Create a writable FilesystemDatabase at output_dir for replay/debug output.

    When parent is provided, writes overlay_meta.json recording the parent
    snapshot path so the relationship can be reopened later by tooling.

    Raises FileExistsError if the directory already contains database artifact directories.
    """
    if output_dir.exists():
        existing = {d.name for d in output_dir.iterdir() if d.is_dir()} & _DB_ARTIFACT_DIRS
        if existing:
            raise FileExistsError(
                f"Output directory {output_dir} already contains database artifacts ({', '.join(sorted(existing))}). "
                "Use a fresh directory for replay/debug output, or remove the existing artifacts first."
            )

    db = FilesystemDatabase(output_dir)

    if parent is not None:
        meta = {
            "parent_path": str(parent.base_path.resolve()),
            "created_at": datetime.now(UTC).isoformat(),
            "type": "overlay",
        }
        meta_path = output_dir / "overlay_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return db
