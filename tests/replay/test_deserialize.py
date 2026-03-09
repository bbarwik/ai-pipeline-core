"""Tests for replay deserialization: resolve_doc_refs and resolve_task_kwargs."""

import pytest
from pydantic import BaseModel, ConfigDict

from ai_pipeline_core.documents import Document
from ai_pipeline_core.llm.conversation import Conversation
from ai_pipeline_core.replay._deserialize import resolve_doc_refs, resolve_task_kwargs

from .conftest import ReplayTextDocument, doc_ref_dict, store_document_in_database


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


@pytest.mark.asyncio
async def test_resolve_doc_refs_replaces_doc_ref_dict(memory_database) -> None:
    doc = ReplayTextDocument(name="source.txt", content=b"hello replay")
    await store_document_in_database(memory_database, doc)

    ref = doc_ref_dict(doc)
    result = await resolve_doc_refs(ref, memory_database)

    assert isinstance(result, Document)
    assert result.content == b"hello replay"
    assert result.name == "source.txt"


@pytest.mark.asyncio
async def test_resolve_doc_refs_inline_content_escape_hatch(memory_database) -> None:
    data = {
        "class_name": "DeserializeDocument",
        "name": "inline.txt",
        "content": "ephemeral content",
    }
    result = await resolve_doc_refs(data, memory_database)

    assert isinstance(result, Document)
    assert result.name == "inline.txt"
    assert result.content == b"ephemeral content"


@pytest.mark.asyncio
async def test_resolve_doc_refs_nested_in_list_and_dict(memory_database) -> None:
    doc = ReplayTextDocument(name="nested.txt", content=b"nested content")
    await store_document_in_database(memory_database, doc)

    ref = doc_ref_dict(doc)
    data = {
        "items": [ref, "plain_string"],
        "nested": {"inner": ref},
    }
    result = await resolve_doc_refs(data, memory_database)

    assert isinstance(result["items"][0], Document)
    assert result["items"][0].name == "nested.txt"
    assert result["items"][1] == "plain_string"
    assert isinstance(result["nested"]["inner"], Document)
    assert result["nested"]["inner"].content == b"nested content"


@pytest.mark.asyncio
async def test_resolve_doc_refs_primitives_unchanged(memory_database) -> None:
    data = {"name": "test", "count": 42, "active": True, "extra": None}
    result = await resolve_doc_refs(data, memory_database)
    assert result == {"name": "test", "count": 42, "active": True, "extra": None}


@pytest.mark.asyncio
async def test_resolve_task_kwargs_validates_basemodel(memory_database) -> None:
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
    result = await resolve_task_kwargs(function_path, raw, memory_database)

    assert isinstance(result["options"], DeserializeOptions)
    assert result["options"].max_items == 5
    assert result["options"].tags == ("alpha", "beta")


@pytest.mark.asyncio
async def test_resolve_task_kwargs_validates_document(memory_database) -> None:
    doc = DeserializeDocument(name="task-source.txt", content=b"doc for kwargs")
    await store_document_in_database(memory_database, doc)

    raw = {
        "source": doc_ref_dict(doc),
        "options": {"max_items": 1, "tags": ()},
        "retries": 0,
        "enabled": False,
    }
    function_path = f"{__name__}:deserialization_target"
    result = await resolve_task_kwargs(function_path, raw, memory_database)

    assert isinstance(result["source"], Document)
    assert result["source"].name == "task-source.txt"
    assert result["source"].content == b"doc for kwargs"


@pytest.mark.asyncio
async def test_resolve_task_kwargs_mixed_args(memory_database) -> None:
    doc = DeserializeDocument(name="mixed.txt", content=b"mixed content")
    await store_document_in_database(memory_database, doc)

    raw = {
        "source": doc_ref_dict(doc),
        "options": {"max_items": 10, "tags": ("x",)},
        "retries": 2,
        "enabled": True,
    }
    function_path = f"{__name__}:deserialization_target"
    result = await resolve_task_kwargs(function_path, raw, memory_database)

    assert isinstance(result["source"], Document)
    assert result["source"].content == b"mixed content"
    assert isinstance(result["options"], DeserializeOptions)
    assert result["options"].max_items == 10
    assert result["retries"] == 2
    assert result["enabled"] is True


@pytest.mark.asyncio
async def test_resolve_doc_refs_conversation_sentinel(memory_database) -> None:
    doc = ReplayTextDocument(name="conversation.txt", content=b"conversation context")
    await store_document_in_database(memory_database, doc)

    raw = {
        "$conversation": {
            "model": "test-model",
            "model_options": {},
            "context": [doc_ref_dict(doc)],
            "history": [{"type": "assistant_text", "text": "hello"}],
            "enable_substitutor": False,
            "extract_result_tags": False,
            "include_date": False,
            "current_date": None,
        }
    }

    result = await resolve_doc_refs(raw, memory_database)

    assert isinstance(result, Conversation)
    assert result.model == "test-model"
    assert len(result.context) == 1
    assert result.context[0].name == "conversation.txt"
    assert len(result.messages) == 1


async def _fixed_tuple_target(pair: tuple[int, int]) -> None:
    """Target function with fixed-length tuple annotation."""


@pytest.mark.asyncio
async def test_fixed_tuple_replay_rejects_length_mismatch(memory_database) -> None:
    function_path = f"{__name__}:_fixed_tuple_target"
    with pytest.raises(ValueError, match="expects 2 items but replay data has 3"):
        await resolve_task_kwargs(
            function_path,
            {"pair": [1, 2, 3]},
            memory_database,
        )
