"""Test helpers for observability tests.

Contains functions moved from production code that are only used in tests.
"""

from pathlib import Path
from typing import Any, cast

import yaml


def reconstruct_span_content(trace_root: Path, span_dir: Path, content_type: str) -> dict[str, Any]:
    """Reconstruct full content from input.yaml/output.yaml + artifacts.

    Args:
        trace_root: Trace root directory
        span_dir: Span directory containing input.yaml or output.yaml
        content_type: "input" or "output"

    Returns:
        Complete reconstructed content with all artifact refs resolved
    """
    content_path = span_dir / f"{content_type}.yaml"
    if not content_path.exists():
        return {}

    content = yaml.safe_load(content_path.read_text(encoding="utf-8"))
    return _rehydrate(content, trace_root)


def _rehydrate(obj: Any, trace_root: Path) -> Any:
    """Recursively replace content_ref entries with actual content."""
    if isinstance(obj, dict):
        obj_dict = cast(dict[str, Any], obj)
        if "content_ref" in obj_dict:
            # This is an artifact reference - load the full content
            ref: dict[str, Any] = obj_dict["content_ref"]
            artifact_path: Path = trace_root / ref["path"]

            full_content: str | bytes
            if ref.get("encoding") == "utf-8":
                full_content = artifact_path.read_text(encoding="utf-8")
            else:
                full_content = artifact_path.read_bytes()

            # Replace ref with full content
            obj_dict = obj_dict.copy()
            obj_dict["content"] = full_content
            del obj_dict["content_ref"]
            if "excerpt" in obj_dict:
                del obj_dict["excerpt"]

        return {k: _rehydrate(v, trace_root) for k, v in obj_dict.items()}

    if isinstance(obj, list):
        obj_list = cast(list[Any], obj)
        return [_rehydrate(v, trace_root) for v in obj_list]

    return obj
