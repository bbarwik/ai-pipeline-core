"""Tests for output tracking in observability.

Verifies that track_task_io correctly sets OTel span attributes for
document SHA256 arrays on various output types (single, list, tuple, None).
"""

from typing import Any
from unittest.mock import MagicMock, patch

from pydantic import BaseModel, ConfigDict

from ai_pipeline_core.documents import DocumentSha256
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.observability._document_tracking import _collect_sha256s, track_task_io
from ai_pipeline_core.observability._span_data import ATTR_INPUT_DOC_SHA256S, ATTR_OUTPUT_DOC_SHA256S


class _DocA(Document):
    """Test document type A."""


class _DocB(Document):
    """Test document type B."""


def _make_active_span() -> MagicMock:
    """Create a mock active span with a valid context."""
    mock_span = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.span_id = 1
    mock_span.get_span_context.return_value = mock_ctx
    return mock_span


class TestTrackTaskIoOutputTypes:
    """track_task_io must set span attributes for output documents."""

    def test_single_document_output_tracked(self) -> None:
        doc = _DocA.create_root(name="a.txt", content=b"aaa", reason="test input")
        mock_span = _make_active_span()

        with patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel:
            mock_otel.get_current_span.return_value = mock_span
            track_task_io((), {}, doc)

        mock_span.set_attribute.assert_any_call(ATTR_OUTPUT_DOC_SHA256S, [doc.sha256])

    def test_list_document_output_tracked(self) -> None:
        docs = [
            _DocA.create_root(name="a.txt", content=b"aaa", reason="test input"),
            _DocB.create_root(name="b.txt", content=b"bbb", reason="test input"),
        ]
        mock_span = _make_active_span()

        with patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel:
            mock_otel.get_current_span.return_value = mock_span
            track_task_io((), {}, docs)

        output_call = [c for c in mock_span.set_attribute.call_args_list if c[0][0] == ATTR_OUTPUT_DOC_SHA256S]
        assert len(output_call) == 1
        sha256s = output_call[0][0][1]
        assert set(sha256s) == {docs[0].sha256, docs[1].sha256}

    def test_tuple_of_documents_output_tracked(self) -> None:
        doc_a = _DocA.create_root(name="a.txt", content=b"aaa", reason="test input")
        doc_b = _DocB.create_root(name="b.txt", content=b"bbb", reason="test input")
        result = (doc_a, doc_b)

        mock_span = _make_active_span()

        with patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel:
            mock_otel.get_current_span.return_value = mock_span
            track_task_io((), {}, result)

        output_call = [c for c in mock_span.set_attribute.call_args_list if c[0][0] == ATTR_OUTPUT_DOC_SHA256S]
        assert len(output_call) == 1
        sha256s = output_call[0][0][1]
        assert set(sha256s) == {doc_a.sha256, doc_b.sha256}

    def test_tuple_of_lists_output_tracked(self) -> None:
        docs_a = [_DocA.create_root(name="a1.txt", content=b"a1", reason="test input"), _DocA.create_root(name="a2.txt", content=b"a2", reason="test input")]
        docs_b = [_DocB.create_root(name="b1.txt", content=b"b1", reason="test input")]
        result = (docs_a, docs_b)

        mock_span = _make_active_span()

        with patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel:
            mock_otel.get_current_span.return_value = mock_span
            track_task_io((), {}, result)

        output_call = [c for c in mock_span.set_attribute.call_args_list if c[0][0] == ATTR_OUTPUT_DOC_SHA256S]
        assert len(output_call) == 1
        sha256s = output_call[0][0][1]
        assert len(sha256s) == 3

    def test_none_output_no_events(self) -> None:
        mock_span = _make_active_span()

        with patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel:
            mock_otel.get_current_span.return_value = mock_span
            track_task_io((), {}, None)

        output_calls = [c for c in mock_span.set_attribute.call_args_list if c[0][0] == ATTR_OUTPUT_DOC_SHA256S]
        assert len(output_calls) == 0

    def test_input_documents_tracked_from_args(self) -> None:
        input_doc = _DocA.create_root(name="input.txt", content=b"input", reason="test input")
        mock_span = _make_active_span()

        with patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel:
            mock_otel.get_current_span.return_value = mock_span
            track_task_io((input_doc,), {}, None)

        mock_span.set_attribute.assert_any_call(ATTR_INPUT_DOC_SHA256S, [input_doc.sha256])


class TestDeploymentSpanOutput:
    """Deployment spans must set both input AND output on the Laminar span."""

    def test_as_prefect_flow_sets_span_output(self) -> None:
        from ai_pipeline_core.deployment.base import DeploymentResult, PipelineDeployment
        from ai_pipeline_core.pipeline._flow import PipelineFlow
        from ai_pipeline_core.pipeline.options import FlowOptions

        class _TestResult(DeploymentResult):
            summary: str = ""

        class _TestOptions(FlowOptions):
            pass

        class _SpanTestInputDoc(Document):
            """Input document for span output test."""

        class _SpanTestOutputDoc(Document):
            """Output document for span output test."""

        class _SpanTestFlow(PipelineFlow):
            """Test flow for span output test."""

            async def run(self, run_id: str, documents: list[_SpanTestInputDoc], options: FlowOptions) -> list[_SpanTestOutputDoc]:
                return []

        class _TestDeployment(PipelineDeployment[_TestOptions, _TestResult]):
            def build_flows(self, options: _TestOptions) -> list[PipelineFlow]:
                return [_SpanTestFlow()]

            @staticmethod
            def build_result(run_id: str, documents: list[Document], options: Any) -> _TestResult:
                return _TestResult(success=True, summary="done")

        deployment = _TestDeployment()
        prefect_flow = deployment.as_prefect_flow()
        inner_fn = getattr(prefect_flow, "fn", prefect_flow)
        import inspect

        source = inspect.getsource(inner_fn)
        assert "set_span_output" in source

    def test_cli_sets_span_output(self) -> None:
        import inspect

        from ai_pipeline_core.deployment._cli import run_cli_for_deployment

        source = inspect.getsource(run_cli_for_deployment)
        assert "set_span_output" in source


class _TrackDoc(Document):
    """Doc for tracking test."""


def test_collect_sha256s_recurses_into_dict_and_basemodel() -> None:
    """_collect_sha256s must find documents in dict values and BaseModel fields."""

    class Carrier(BaseModel):
        model_config = ConfigDict(frozen=True)
        payload: dict[str, _TrackDoc]

    doc = _TrackDoc(name="t.txt", content=b"x")
    carrier = Carrier(payload={"a": doc})

    sha256s: list[DocumentSha256] = []
    _collect_sha256s(carrier, sha256s)
    assert doc.sha256 in sha256s
