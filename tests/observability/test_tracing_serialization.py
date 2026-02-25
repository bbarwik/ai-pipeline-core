"""Unit tests for tracing serialization helpers in ai_pipeline_core.observability.tracing."""

# pyright: reportPrivateUsage=false

import json

from pydantic import BaseModel

from ai_pipeline_core.documents import Document
from ai_pipeline_core.observability.tracing import (
    _serialize_and_trim,
    _serialize_for_tracing,
    _trim_formatted,
)


class TracingDoc(Document):
    pass


class TestSerializeForTracing:
    def test_single_document(self):
        doc = TracingDoc(name="test.txt", content=b"hello")
        result = _serialize_for_tracing(doc)
        assert isinstance(result, dict)
        assert result["name"] == "test.txt"

    def test_document_list(self):
        docs = [TracingDoc(name="a.txt", content=b"a"), TracingDoc(name="b.txt", content=b"b")]
        result = _serialize_for_tracing(docs)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_pydantic_model(self):
        class Item(BaseModel):
            val: int = 1

        result = _serialize_for_tracing(Item())
        assert isinstance(result, dict)
        assert result["val"] == 1

    def test_pydantic_with_document_field(self):
        class Container(BaseModel):
            doc: TracingDoc

            model_config = {"arbitrary_types_allowed": True}

        doc = TracingDoc(name="test.txt", content=b"data")
        container = Container(doc=doc)
        result = _serialize_for_tracing(container)
        assert isinstance(result, dict)
        assert isinstance(result["doc"], dict)

    def test_fallback_to_str(self):
        result = _serialize_for_tracing(42)
        assert result == "42"

    def test_fallback_str_fails(self):
        class BadStr:
            def __str__(self):
                raise RuntimeError("no str")

        result = _serialize_for_tracing(BadStr())
        assert "<BadStr>" in result


class TestTrimFormatted:
    def test_string_valid_json(self):
        data = json.dumps({"key": "value"})
        result = _trim_formatted(data)
        assert "key" in result

    def test_string_non_json(self):
        result = _trim_formatted("plain text")
        assert result == "plain text"

    def test_dict_data(self):
        result = _trim_formatted({"key": "value"})
        parsed = json.loads(result)
        assert parsed["key"] == "value"


class TestSerializeAndTrim:
    def test_basic_dict(self):
        result = _serialize_and_trim({"message": "hello"})
        parsed = json.loads(result)
        assert parsed["message"] == "hello"
