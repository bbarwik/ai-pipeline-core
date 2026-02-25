"""Tests for replay capture: serialize_kwargs and serialize_prior_messages."""

from ai_pipeline_core.llm.conversation import _AssistantMessage, _UserMessage
from ai_pipeline_core.replay._capture import serialize_kwargs, serialize_prior_messages
from tests.support.helpers import create_test_model_response

from .conftest import ReplayArgsModel, ReplayMode, ReplayTextDocument, doc_ref_dict


# ---------------------------------------------------------------------------
# serialize_kwargs
# ---------------------------------------------------------------------------


def test_serialize_kwargs_document_becomes_doc_ref() -> None:
    """Document argument is serialized to a $doc_ref dict."""
    doc = ReplayTextDocument(name="input.txt", content=b"serialize me")
    result = serialize_kwargs({"source": doc})

    expected = doc_ref_dict(doc)
    assert result["source"] == expected


def test_serialize_kwargs_basemodel_becomes_dict() -> None:
    """Frozen BaseModel argument is serialized via model_dump(mode='json')."""
    model = ReplayArgsModel(max_items=5, label="test")
    result = serialize_kwargs({"options": model})

    assert result["options"] == {"max_items": 5, "label": "test"}


def test_serialize_kwargs_primitive_passthrough() -> None:
    """str, int, float, and bool values pass through unchanged."""
    kwargs = {"name": "hello", "count": 7, "ratio": 0.5, "active": True}
    result = serialize_kwargs(kwargs)

    assert result == {"name": "hello", "count": 7, "ratio": 0.5, "active": True}


def test_serialize_kwargs_enum_becomes_value() -> None:
    """StrEnum argument is serialized to its string value."""
    result = serialize_kwargs({"mode": ReplayMode.DEEP})

    assert result["mode"] == "deep"


def test_serialize_kwargs_list_of_documents() -> None:
    """List of Documents should serialize each element to a $doc_ref dict."""
    doc1 = ReplayTextDocument(name="a.txt", content=b"alpha")
    doc2 = ReplayTextDocument(name="b.txt", content=b"bravo")
    result = serialize_kwargs({"docs": [doc1, doc2]})

    assert result["docs"] == [doc_ref_dict(doc1), doc_ref_dict(doc2)]


def test_serialize_kwargs_tuple_of_documents() -> None:
    """Tuple of Documents should serialize each element to a $doc_ref dict."""
    doc1 = ReplayTextDocument(name="x.txt", content=b"xray")
    doc2 = ReplayTextDocument(name="y.txt", content=b"yankee")
    result = serialize_kwargs({"docs": (doc1, doc2)})

    assert result["docs"] == [doc_ref_dict(doc1), doc_ref_dict(doc2)]


def test_serialize_kwargs_nested_dict_with_document() -> None:
    """Nested dict values containing Documents should be recursively serialized."""
    doc = ReplayTextDocument(name="src.md", content=b"source")
    result = serialize_kwargs({"config": {"source": doc}})

    assert result["config"] == {"source": doc_ref_dict(doc)}


def test_serialize_kwargs_mixed_list() -> None:
    """List with mixed types should serialize Documents but leave primitives unchanged."""
    doc = ReplayTextDocument(name="m.txt", content=b"mixed")
    result = serialize_kwargs({"items": [doc, "string", 42]})

    assert result["items"] == [doc_ref_dict(doc), "string", 42]


# ---------------------------------------------------------------------------
# serialize_prior_messages
# ---------------------------------------------------------------------------


def test_serialize_prior_messages_user_text() -> None:
    """_UserMessage is serialized with type 'user_text'."""
    messages = (_UserMessage("hello world"),)
    result = serialize_prior_messages(messages)

    assert len(result) == 1
    assert result[0] == {"type": "user_text", "text": "hello world"}


def test_serialize_prior_messages_assistant_text() -> None:
    """_AssistantMessage is serialized with type 'assistant_text'."""
    messages = (_AssistantMessage("note"),)
    result = serialize_prior_messages(messages)

    assert len(result) == 1
    assert result[0] == {"type": "assistant_text", "text": "note"}


def test_serialize_prior_messages_response() -> None:
    """ModelResponse is serialized inline with type 'response' and content string."""
    response = create_test_model_response(content="LLM said this")
    messages = (response,)
    result = serialize_prior_messages(messages)

    assert len(result) == 1
    assert result[0]["type"] == "response"
    assert result[0]["content"] == "LLM said this"


def test_serialize_prior_messages_document() -> None:
    """Document in messages is serialized as type 'document' with $doc_ref."""
    doc = ReplayTextDocument(name="ctx.md", content=b"context data")
    messages = (doc,)
    result = serialize_prior_messages(messages)

    assert len(result) == 1
    assert result[0]["type"] == "document"
    assert result[0]["$doc_ref"] == doc.sha256
    assert result[0]["class_name"] == "ReplayTextDocument"
    assert result[0]["name"] == "ctx.md"


def test_serialize_prior_messages_multi_turn() -> None:
    """A 3-turn conversation (6 messages) serializes to 6 history entries in order."""
    doc = ReplayTextDocument(name="ref.txt", content=b"reference")
    response_1 = create_test_model_response(content="first reply")
    response_2 = create_test_model_response(content="second reply")

    messages = (
        _UserMessage("turn one"),
        response_1,
        doc,
        _AssistantMessage("injected"),
        _UserMessage("turn three"),
        response_2,
    )
    result = serialize_prior_messages(messages)

    assert len(result) == 6
    assert result[0] == {"type": "user_text", "text": "turn one"}
    assert result[1]["type"] == "response"
    assert result[1]["content"] == "first reply"
    assert result[2]["type"] == "document"
    assert result[2]["$doc_ref"] == doc.sha256
    assert result[3] == {"type": "assistant_text", "text": "injected"}
    assert result[4] == {"type": "user_text", "text": "turn three"}
    assert result[5]["type"] == "response"
    assert result[5]["content"] == "second reply"
