"""Tests for output tracking in observability.

Proves two bugs:
1. track_task_io doesn't handle tuple results — output documents are lost.
2. Deployment spans (as_prefect_flow, run_cli) set input but never set output.
"""

from typing import Any
from unittest.mock import MagicMock, patch

from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.observability._document_tracking import track_task_io
from ai_pipeline_core.observability._tracking._models import ATTR_OUTPUT_DOCUMENT_SHA256S, DocumentEventType


class _DocA(Document):
    """Test document type A."""


class _DocB(Document):
    """Test document type B."""


def _make_tracking_env() -> tuple[MagicMock, MagicMock]:
    """Create mock tracking service and OTel span, patched into document tracking module."""
    mock_service = MagicMock()
    mock_span = MagicMock()
    return mock_service, mock_span


class TestTrackTaskIoOutputTypes:
    """track_task_io must track output documents regardless of container type.

    @pipeline_task allows these return types:
        Document, list[Document], tuple[Document, ...], tuple[list[Document], ...], None

    Input tracking works for all types because task parameters are typically list[Document].
    Output tracking is broken for tuple results.
    """

    def test_single_document_output_tracked(self) -> None:
        """Baseline: single Document output is tracked correctly."""
        doc = _DocA.create(name="a.txt", content=b"aaa")
        mock_service, mock_span = _make_tracking_env()

        with (
            patch("ai_pipeline_core.observability._document_tracking.get_tracking_service", return_value=mock_service),
            patch("ai_pipeline_core.observability._document_tracking.get_current_span_id", return_value="span1"),
            patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel,
        ):
            mock_otel.get_current_span.return_value = mock_span
            track_task_io((), {}, doc)

        # Output document event should be emitted
        output_calls = [c for c in mock_service.track_document_event.call_args_list if c[1]["event_type"] == DocumentEventType.TASK_OUTPUT]
        assert len(output_calls) == 1
        assert output_calls[0][1]["document_sha256"] == doc.sha256

        # Span attribute should be set
        mock_span.set_attribute.assert_any_call(ATTR_OUTPUT_DOCUMENT_SHA256S, [doc.sha256])

    def test_list_document_output_tracked(self) -> None:
        """Baseline: list[Document] output is tracked correctly."""
        docs = [
            _DocA.create(name="a.txt", content=b"aaa"),
            _DocB.create(name="b.txt", content=b"bbb"),
        ]
        mock_service, mock_span = _make_tracking_env()

        with (
            patch("ai_pipeline_core.observability._document_tracking.get_tracking_service", return_value=mock_service),
            patch("ai_pipeline_core.observability._document_tracking.get_current_span_id", return_value="span1"),
            patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel,
        ):
            mock_otel.get_current_span.return_value = mock_span
            track_task_io((), {}, docs)

        output_calls = [c for c in mock_service.track_document_event.call_args_list if c[1]["event_type"] == DocumentEventType.TASK_OUTPUT]
        assert len(output_calls) == 2

        sha256s = {c[1]["document_sha256"] for c in output_calls}
        assert sha256s == {docs[0].sha256, docs[1].sha256}

    def test_tuple_of_documents_output_tracked(self) -> None:
        """Bug: tuple[DocA, DocB] output — documents must be tracked."""
        doc_a = _DocA.create(name="a.txt", content=b"aaa")
        doc_b = _DocB.create(name="b.txt", content=b"bbb")
        result = (doc_a, doc_b)

        mock_service, mock_span = _make_tracking_env()

        with (
            patch("ai_pipeline_core.observability._document_tracking.get_tracking_service", return_value=mock_service),
            patch("ai_pipeline_core.observability._document_tracking.get_current_span_id", return_value="span1"),
            patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel,
        ):
            mock_otel.get_current_span.return_value = mock_span
            track_task_io((), {}, result)

        output_calls = [c for c in mock_service.track_document_event.call_args_list if c[1]["event_type"] == DocumentEventType.TASK_OUTPUT]
        assert len(output_calls) == 2, (
            f"Expected 2 TASK_OUTPUT events for tuple(DocA, DocB), got {len(output_calls)}. track_task_io does not handle tuple results."
        )

        sha256s = {c[1]["document_sha256"] for c in output_calls}
        assert sha256s == {doc_a.sha256, doc_b.sha256}

    def test_tuple_of_lists_output_tracked(self) -> None:
        """Bug: tuple[list[DocA], list[DocB]] output — documents must be tracked."""
        docs_a = [_DocA.create(name="a1.txt", content=b"a1"), _DocA.create(name="a2.txt", content=b"a2")]
        docs_b = [_DocB.create(name="b1.txt", content=b"b1")]
        result = (docs_a, docs_b)

        mock_service, mock_span = _make_tracking_env()

        with (
            patch("ai_pipeline_core.observability._document_tracking.get_tracking_service", return_value=mock_service),
            patch("ai_pipeline_core.observability._document_tracking.get_current_span_id", return_value="span1"),
            patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel,
        ):
            mock_otel.get_current_span.return_value = mock_span
            track_task_io((), {}, result)

        output_calls = [c for c in mock_service.track_document_event.call_args_list if c[1]["event_type"] == DocumentEventType.TASK_OUTPUT]
        assert len(output_calls) == 3, (
            f"Expected 3 TASK_OUTPUT events for tuple(list[DocA], list[DocB]), got {len(output_calls)}. track_task_io does not handle tuple-of-lists results."
        )

        sha256s = {c[1]["document_sha256"] for c in output_calls}
        assert sha256s == {docs_a[0].sha256, docs_a[1].sha256, docs_b[0].sha256}

    def test_none_output_no_events(self) -> None:
        """None result should not produce any output events."""
        mock_service, mock_span = _make_tracking_env()

        with (
            patch("ai_pipeline_core.observability._document_tracking.get_tracking_service", return_value=mock_service),
            patch("ai_pipeline_core.observability._document_tracking.get_current_span_id", return_value="span1"),
            patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel,
        ):
            mock_otel.get_current_span.return_value = mock_span
            track_task_io((), {}, None)

        output_calls = [c for c in mock_service.track_document_event.call_args_list if c[1]["event_type"] == DocumentEventType.TASK_OUTPUT]
        assert len(output_calls) == 0

    def test_input_documents_tracked_from_args(self) -> None:
        """Baseline: input documents from args are tracked."""
        input_doc = _DocA.create(name="input.txt", content=b"input")
        mock_service, mock_span = _make_tracking_env()

        with (
            patch("ai_pipeline_core.observability._document_tracking.get_tracking_service", return_value=mock_service),
            patch("ai_pipeline_core.observability._document_tracking.get_current_span_id", return_value="span1"),
            patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel,
        ):
            mock_otel.get_current_span.return_value = mock_span
            track_task_io((input_doc,), {}, None)

        input_calls = [c for c in mock_service.track_document_event.call_args_list if c[1]["event_type"] == DocumentEventType.TASK_INPUT]
        assert len(input_calls) == 1
        assert input_calls[0][1]["document_sha256"] == input_doc.sha256

    def test_input_list_documents_tracked_from_args(self) -> None:
        """Baseline: input list[Document] from args are tracked."""
        input_docs = [
            _DocA.create(name="i1.txt", content=b"i1"),
            _DocA.create(name="i2.txt", content=b"i2"),
        ]
        mock_service, mock_span = _make_tracking_env()

        with (
            patch("ai_pipeline_core.observability._document_tracking.get_tracking_service", return_value=mock_service),
            patch("ai_pipeline_core.observability._document_tracking.get_current_span_id", return_value="span1"),
            patch("ai_pipeline_core.observability._document_tracking.otel_trace") as mock_otel,
        ):
            mock_otel.get_current_span.return_value = mock_span
            track_task_io((input_docs,), {}, None)

        input_calls = [c for c in mock_service.track_document_event.call_args_list if c[1]["event_type"] == DocumentEventType.TASK_INPUT]
        assert len(input_calls) == 2


class TestDeploymentSpanOutput:
    """Deployment spans must set both input AND output on the Laminar span.

    The LLM client correctly calls Laminar.set_span_output() (client.py:444).
    Deployment code only sets input= on start_as_current_span, never sets output.
    """

    def test_as_prefect_flow_sets_span_output(self) -> None:
        """as_prefect_flow must call Laminar.set_span_output with the deployment result."""
        from ai_pipeline_core.deployment.base import DeploymentResult, PipelineDeployment
        from ai_pipeline_core.pipeline.options import FlowOptions

        class _TestResult(DeploymentResult):
            summary: str = ""

        class _TestOptions(FlowOptions):
            pass

        # Create a minimal flow
        async def _test_flow(
            project_name: str,
            documents: list[Document],
            flow_options: _TestOptions,
        ) -> list[Document]:
            return []

        # Manually set flow attributes that @pipeline_flow would set
        _test_flow.input_document_types = []  # type: ignore[attr-defined]
        _test_flow.output_document_types = []  # type: ignore[attr-defined]
        _test_flow.estimated_minutes = 1  # type: ignore[attr-defined]
        _test_flow.name = "test_flow"  # type: ignore[attr-defined]

        class _TestDeployment(PipelineDeployment[_TestOptions, _TestResult]):
            flows = [_test_flow]

            @staticmethod
            def build_result(project_name: str, documents: list[Document], options: Any) -> _TestResult:
                return _TestResult(success=True, summary="done")

        deployment = _TestDeployment()
        prefect_flow = deployment.as_prefect_flow()

        # Extract the inner async function (unwrap Prefect decoration)
        inner_fn = getattr(prefect_flow, "fn", prefect_flow)

        # Check that the source contains Laminar.set_span_output
        import inspect

        source = inspect.getsource(inner_fn)
        assert "set_span_output" in source, (
            "as_prefect_flow's inner function does not call Laminar.set_span_output(). Input is set via input= parameter but output is never recorded."
        )

    def test_cli_sets_span_output(self) -> None:
        """run_cli_for_deployment must call Laminar.set_span_output with the result."""
        import inspect

        from ai_pipeline_core.deployment._cli import run_cli_for_deployment

        source = inspect.getsource(run_cli_for_deployment)
        assert "set_span_output" in source, (
            "run_cli_for_deployment does not call Laminar.set_span_output(). Input is set via input= parameter but output is never recorded."
        )
