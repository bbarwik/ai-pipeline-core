"""Unit tests for document tracking helpers."""

from unittest.mock import MagicMock, patch


from ai_pipeline_core.documents import Document
from ai_pipeline_core.observability._tracking._models import (
    ATTR_INPUT_DOCUMENT_SHA256S,
    ATTR_OUTPUT_DOCUMENT_SHA256S,
    DocumentEventType,
)


class TrackDoc(Document):
    pass


def _make_doc(name="test.txt", content="hello"):
    return TrackDoc.create_root(name=name, content=content, reason="test")


class TestGetCurrentSpanId:
    def test_returns_hex_when_span_active(self):
        from ai_pipeline_core.observability._document_tracking import get_current_span_id

        mock_ctx = MagicMock()
        mock_ctx.span_id = 255
        mock_span = MagicMock()
        mock_span.get_span_context.return_value = mock_ctx
        with patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel:
            mock_otel.get_current_span.return_value = mock_span
            result = get_current_span_id()
        assert result == "00000000000000ff"

    def test_returns_empty_when_no_span(self):
        from ai_pipeline_core.observability._document_tracking import get_current_span_id

        mock_ctx = MagicMock()
        mock_ctx.span_id = 0
        mock_span = MagicMock()
        mock_span.get_span_context.return_value = mock_ctx
        with patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel:
            mock_otel.get_current_span.return_value = mock_span
            result = get_current_span_id()
        assert result == ""


class TestCollectAndTrack:
    def test_single_document(self):
        from ai_pipeline_core.observability._document_tracking import _collect_and_track

        doc = _make_doc()
        sha256s = []
        service = MagicMock()
        _collect_and_track(doc, sha256s, service, "span1", DocumentEventType.TASK_INPUT)
        assert len(sha256s) == 1
        assert sha256s[0] == doc.sha256
        service.track_document_event.assert_called_once()

    def test_list_of_documents(self):
        from ai_pipeline_core.observability._document_tracking import _collect_and_track

        doc1 = _make_doc("a.txt", "aaa")
        doc2 = _make_doc("b.txt", "bbb")
        sha256s = []
        service = MagicMock()
        _collect_and_track([doc1, doc2], sha256s, service, "span1", DocumentEventType.TASK_OUTPUT)
        assert len(sha256s) == 2
        assert service.track_document_event.call_count == 2

    def test_nested_containers(self):
        from ai_pipeline_core.observability._document_tracking import _collect_and_track

        doc1 = _make_doc("a.txt", "aaa")
        doc2 = _make_doc("b.txt", "bbb")
        sha256s = []
        service = MagicMock()
        _collect_and_track(([doc1], [doc2]), sha256s, service, "span1", DocumentEventType.TASK_OUTPUT)
        assert len(sha256s) == 2

    def test_non_document_ignored(self):
        from ai_pipeline_core.observability._document_tracking import _collect_and_track

        sha256s = []
        service = MagicMock()
        _collect_and_track("not a doc", sha256s, service, "span1", DocumentEventType.TASK_INPUT)
        assert sha256s == []
        service.track_document_event.assert_not_called()


class TestTrackIo:
    def test_noop_when_no_service(self):
        from ai_pipeline_core.observability._document_tracking import _track_io

        with patch("ai_pipeline_core.observability._document_tracking.get_tracking_service", return_value=None):
            _track_io([], None, DocumentEventType.TASK_INPUT, DocumentEventType.TASK_OUTPUT)

    def test_sets_span_attributes(self):
        from ai_pipeline_core.observability._document_tracking import _track_io

        doc = _make_doc()
        mock_service = MagicMock()
        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.span_id = 1
        mock_span.get_span_context.return_value = mock_ctx

        with (
            patch("ai_pipeline_core.observability._document_tracking.get_tracking_service", return_value=mock_service),
            patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel,
        ):
            mock_otel.get_current_span.return_value = mock_span
            _track_io([doc], doc, DocumentEventType.TASK_INPUT, DocumentEventType.TASK_OUTPUT)

        mock_span.set_attribute.assert_any_call(ATTR_INPUT_DOCUMENT_SHA256S, [doc.sha256])
        mock_span.set_attribute.assert_any_call(ATTR_OUTPUT_DOCUMENT_SHA256S, [doc.sha256])


class TestTrackTaskIo:
    def test_noop_no_service(self):
        from ai_pipeline_core.observability._document_tracking import track_task_io

        with patch("ai_pipeline_core.observability._document_tracking.get_tracking_service", return_value=None):
            track_task_io(("not_doc",), {}, "result")


class TestTrackFlowIo:
    def test_noop_no_service(self):
        from ai_pipeline_core.observability._document_tracking import track_flow_io

        with patch("ai_pipeline_core.observability._document_tracking.get_tracking_service", return_value=None):
            track_flow_io([], [])


class TestTrackLlmDocuments:
    def test_noop_no_service(self):
        from ai_pipeline_core.observability._document_tracking import track_llm_documents

        with patch("ai_pipeline_core.observability._document_tracking.get_tracking_service", return_value=None):
            track_llm_documents(None, None)

    def test_tracks_context_documents(self):
        from ai_pipeline_core.observability._document_tracking import track_llm_documents

        doc = _make_doc()
        mock_service = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.span_id = 1
        mock_span = MagicMock()
        mock_span.get_span_context.return_value = mock_ctx

        with (
            patch("ai_pipeline_core.observability._document_tracking.get_tracking_service", return_value=mock_service),
            patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel,
        ):
            mock_otel.get_current_span.return_value = mock_span
            track_llm_documents([doc], [doc])

        assert mock_service.track_document_event.call_count == 2

    def test_non_list_ignored(self):
        from ai_pipeline_core.observability._document_tracking import track_llm_documents

        mock_service = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.span_id = 1
        mock_span = MagicMock()
        mock_span.get_span_context.return_value = mock_ctx

        with (
            patch("ai_pipeline_core.observability._document_tracking.get_tracking_service", return_value=mock_service),
            patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel,
        ):
            mock_otel.get_current_span.return_value = mock_span
            track_llm_documents("not a list", "also not a list")

        mock_service.track_document_event.assert_not_called()
