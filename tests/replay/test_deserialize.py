"""Tests for replay deserialization: resolve_doc_refs and resolve_task_kwargs."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from ai_pipeline_core.document_store._local import LocalDocumentStore
from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents import RunScope
from ai_pipeline_core.replay._deserialize import resolve_doc_refs, resolve_task_kwargs

from .conftest import ReplayTextDocument, doc_ref_dict


# ---------------------------------------------------------------------------
# Test types for deserialization
# ---------------------------------------------------------------------------


class DeserializeDocument(Document):
    """Document used in deserialization tests."""


class DeserializeOptions(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_items: int
    tags: tuple[str, ...]


async def deserialization_target(
    source: DeserializeDocument,
    options: DeserializeOptions,
    retries: int,
    enabled: bool,
) -> DeserializeDocument: ...


# ---------------------------------------------------------------------------
# resolve_doc_refs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_doc_refs_replaces_doc_ref_dict(store_base: Path) -> None:
    """A $doc_ref dict pointing at a saved document resolves to the Document."""
    doc = ReplayTextDocument(name="source.txt", content=b"hello replay")
    store = LocalDocumentStore(base_path=store_base)
    await store.save(doc, RunScope("replay/test"))

    ref = doc_ref_dict(doc)
    result = resolve_doc_refs(ref, store_base)

    assert isinstance(result, Document)
    assert result.content == b"hello replay"
    assert result.name == "source.txt"


@pytest.mark.asyncio
async def test_resolve_doc_refs_inline_content_escape_hatch() -> None:
    """Inline content dict (no $doc_ref) creates an ephemeral Document without store access."""
    data = {
        "class_name": "DeserializeDocument",
        "name": "inline.txt",
        "content": "ephemeral content",
    }
    result = resolve_doc_refs(data, Path("/nonexistent"))

    assert isinstance(result, Document)
    assert result.name == "inline.txt"
    assert result.content == b"ephemeral content"


@pytest.mark.asyncio
async def test_resolve_doc_refs_nested_in_list_and_dict(store_base: Path) -> None:
    """$doc_ref dicts inside lists and nested dicts are all resolved."""
    doc = ReplayTextDocument(name="nested.txt", content=b"nested content")
    store = LocalDocumentStore(base_path=store_base)
    await store.save(doc, RunScope("replay/test"))

    ref = doc_ref_dict(doc)
    data = {
        "items": [ref, "plain_string"],
        "nested": {"inner": ref},
    }
    result = resolve_doc_refs(data, store_base)

    assert isinstance(result["items"][0], Document)
    assert result["items"][0].name == "nested.txt"
    assert result["items"][1] == "plain_string"
    assert isinstance(result["nested"]["inner"], Document)
    assert result["nested"]["inner"].content == b"nested content"


def test_resolve_doc_refs_primitives_unchanged() -> None:
    """Plain dict with str/int/bool/None values passes through unchanged."""
    data = {"name": "test", "count": 42, "active": True, "extra": None}
    result = resolve_doc_refs(data, Path("/nonexistent"))

    assert result == {"name": "test", "count": 42, "active": True, "extra": None}


# ---------------------------------------------------------------------------
# resolve_task_kwargs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_task_kwargs_validates_basemodel() -> None:
    """A dict arg whose type hint is a BaseModel subclass gets model_validate'd."""
    raw = {
        "source": {
            "class_name": "DeserializeDocument",
            "name": "input.txt",
            "content": "task input",
        },
        "options": {"max_items": 5, "tags": ("alpha", "beta")},
        "retries": 3,
        "enabled": True,
    }
    function_path = f"{__name__}:deserialization_target"
    result = resolve_task_kwargs(function_path, raw, Path("/nonexistent"))

    assert isinstance(result["options"], DeserializeOptions)
    assert result["options"].max_items == 5
    assert result["options"].tags == ("alpha", "beta")


@pytest.mark.asyncio
async def test_resolve_task_kwargs_validates_document(store_base: Path) -> None:
    """A $doc_ref arg whose type hint is a Document subclass resolves correctly."""
    doc = DeserializeDocument(name="task-source.txt", content=b"doc for kwargs")
    store = LocalDocumentStore(base_path=store_base)
    await store.save(doc, RunScope("replay/test"))

    raw = {
        "source": doc_ref_dict(doc),
        "options": {"max_items": 1, "tags": ()},
        "retries": 0,
        "enabled": False,
    }
    function_path = f"{__name__}:deserialization_target"
    result = resolve_task_kwargs(function_path, raw, store_base)

    assert isinstance(result["source"], Document)
    assert result["source"].name == "task-source.txt"
    assert result["source"].content == b"doc for kwargs"


@pytest.mark.asyncio
async def test_resolve_task_kwargs_mixed_args(store_base: Path) -> None:
    """Document, BaseModel, and primitive args all resolve in one call."""
    doc = DeserializeDocument(name="mixed.txt", content=b"mixed content")
    store = LocalDocumentStore(base_path=store_base)
    await store.save(doc, RunScope("replay/test"))

    raw = {
        "source": doc_ref_dict(doc),
        "options": {"max_items": 10, "tags": ("x",)},
        "retries": 2,
        "enabled": True,
    }
    function_path = f"{__name__}:deserialization_target"
    result = resolve_task_kwargs(function_path, raw, store_base)

    assert isinstance(result["source"], Document)
    assert result["source"].content == b"mixed content"
    assert isinstance(result["options"], DeserializeOptions)
    assert result["options"].max_items == 10
    assert result["retries"] == 2
    assert result["enabled"] is True
