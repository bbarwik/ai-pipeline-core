"""Tests for _build_output_hint and _resolve_label helpers."""

from ai_pipeline_core.documents import Document
from ai_pipeline_core.pipeline.decorators import _build_output_hint, _resolve_label


class SampleDoc(Document):
    """Sample document for testing."""


class TestBuildOutputHint:
    """Test _build_output_hint with various result types."""

    def test_none(self):
        assert _build_output_hint(None) == "None"

    def test_empty_list(self):
        assert _build_output_hint(list([])) == "list with 0 items"

    def test_list_with_documents(self):
        doc = SampleDoc.create(name="test.txt", content="hello world")
        result = _build_output_hint(list([doc]))
        assert "1 documents" in result
        assert "SampleDoc" in result

    def test_single_document(self):
        doc = SampleDoc.create(name="output.md", content="# Report")
        result = _build_output_hint(doc)
        assert "SampleDoc" in result
        assert "output.md" in result

    def test_string(self):
        result = _build_output_hint("hello world")
        assert "str" in result
        assert "11 chars" in result

    def test_list(self):
        result = _build_output_hint([1, 2, 3])
        assert "list with 3 items" in result

    def test_other_type(self):
        result = _build_output_hint(42)
        assert result == "int"


class TestResolveLabel:
    """Test _resolve_label with various inputs."""

    def test_bool_true_uses_function_name(self):
        def my_cool_task():
            pass

        label = _resolve_label(True, my_cool_task, {})
        assert label == "My Cool Task"

    def test_string_template(self):
        label = _resolve_label("Processing {source_id}", lambda: None, {"source_id": "abc123"})
        assert label == "Processing abc123"

    def test_string_template_missing_key(self):
        label = _resolve_label("Processing {missing_key}", lambda: None, {})
        assert label == "Processing {missing_key}"
