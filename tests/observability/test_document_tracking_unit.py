"""Unit tests for document tracking helpers."""

from unittest.mock import MagicMock, patch

from ai_pipeline_core.documents import Document
from ai_pipeline_core.observability._span_data import ATTR_INPUT_DOC_SHA256S, ATTR_OUTPUT_DOC_SHA256S


class TrackDoc(Document):
    pass


def _make_doc(name="test.txt", content="hello"):
    return TrackDoc.create_root(name=name, content=content, reason="test")


class TestCollectSha256s:
    def test_single_document(self):
        from ai_pipeline_core.observability._document_tracking import _collect_sha256s

        doc = _make_doc()
        sha256s = []
        _collect_sha256s(doc, sha256s)
        assert len(sha256s) == 1
        assert sha256s[0] == doc.sha256

    def test_list_of_documents(self):
        from ai_pipeline_core.observability._document_tracking import _collect_sha256s

        doc1 = _make_doc("a.txt", "aaa")
        doc2 = _make_doc("b.txt", "bbb")
        sha256s = []
        _collect_sha256s([doc1, doc2], sha256s)
        assert len(sha256s) == 2

    def test_nested_containers(self):
        from ai_pipeline_core.observability._document_tracking import _collect_sha256s

        doc1 = _make_doc("a.txt", "aaa")
        doc2 = _make_doc("b.txt", "bbb")
        sha256s = []
        _collect_sha256s(([doc1], [doc2]), sha256s)
        assert len(sha256s) == 2

    def test_non_document_ignored(self):
        from ai_pipeline_core.observability._document_tracking import _collect_sha256s

        sha256s = []
        _collect_sha256s("not a doc", sha256s)
        assert sha256s == []


class TestTrackIo:
    def test_noop_when_no_active_span(self):
        from ai_pipeline_core.observability._document_tracking import _track_io

        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.span_id = 0
        mock_span.get_span_context.return_value = mock_ctx
        with patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel:
            mock_otel.get_current_span.return_value = mock_span
            _track_io([], None)
        mock_span.set_attribute.assert_not_called()

    def test_sets_span_attributes(self):
        from ai_pipeline_core.observability._document_tracking import _track_io

        doc = _make_doc()
        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.span_id = 1
        mock_span.get_span_context.return_value = mock_ctx

        with patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel:
            mock_otel.get_current_span.return_value = mock_span
            _track_io([doc], doc)

        mock_span.set_attribute.assert_any_call(ATTR_INPUT_DOC_SHA256S, [doc.sha256])
        mock_span.set_attribute.assert_any_call(ATTR_OUTPUT_DOC_SHA256S, [doc.sha256])


class TestTrackTaskIo:
    def test_noop_no_active_span(self):
        from ai_pipeline_core.observability._document_tracking import track_task_io

        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.span_id = 0
        mock_span.get_span_context.return_value = mock_ctx
        with patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel:
            mock_otel.get_current_span.return_value = mock_span
            track_task_io(("not_doc",), {}, "result")


class TestTrackFlowIo:
    def test_noop_no_active_span(self):
        from ai_pipeline_core.observability._document_tracking import track_flow_io

        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.span_id = 0
        mock_span.get_span_context.return_value = mock_ctx
        with patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel:
            mock_otel.get_current_span.return_value = mock_span
            track_flow_io([], [])
