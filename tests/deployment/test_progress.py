"""Tests for progress tracking module."""

# pyright: reportPrivateUsage=false, reportOptionalMemberAccess=false

from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest

from ai_pipeline_core.deployment.contract import ProgressRun
from ai_pipeline_core.deployment.progress import ProgressContext, flow_context, update
from ai_pipeline_core.documents import Document
from ai_pipeline_core.pipeline import pipeline_flow
from ai_pipeline_core.pipeline.options import FlowOptions
from prefect.testing.utilities import prefect_test_harness


class _ProgressTestDoc(Document):
    """Minimal document for progress integration tests."""


class TestUpdate:
    """Test progress update function."""

    async def test_noop_without_context(self):
        """Test update is a no-op when no context is set."""
        await update(0.5, "test")  # Should not raise

    async def test_no_webhook_without_webhook_url(self):
        """Test webhook is not sent when webhook_url is empty (labels still update)."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with flow_context(
            webhook_url="",
            project_name="test",
            flow_run_id=str(UUID(int=1)),
            flow_name="flow1",
            step=1,
            total_steps=3,
            flow_minutes=(1.0, 1.0, 1.0),
            completed_minutes=0.0,
        ):
            with (
                patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client),
                patch("ai_pipeline_core.deployment.progress.send_webhook", new_callable=AsyncMock) as mock_send,
            ):
                await update(0.5, "halfway")

        mock_send.assert_not_called()
        mock_client.update_flow_run_labels.assert_called_once()

    async def test_sends_progress_payload(self):
        """Test update sends ProgressRun via send_webhook."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with flow_context(
            webhook_url="http://example.com/progress",
            project_name="my-project",
            flow_run_id=str(UUID(int=42)),
            flow_name="analysis",
            step=2,
            total_steps=4,
            flow_minutes=(1.0, 2.0, 1.0, 1.0),
            completed_minutes=1.0,
        ):
            with (
                patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client),
                patch("ai_pipeline_core.deployment.progress.send_webhook", new_callable=AsyncMock) as mock_send,
            ):
                await update(0.5, "halfway through analysis")

        mock_send.assert_called_once()
        payload = mock_send.call_args[0][1]
        assert isinstance(payload, ProgressRun)
        assert payload.project_name == "my-project"
        assert payload.flow_name == "analysis"
        assert payload.step == 2
        assert payload.total_steps == 4
        assert payload.status == "progress"
        assert payload.state == "RUNNING"
        assert payload.step_progress == 0.5
        assert payload.progress == 0.4  # (1.0 + 2.0 * 0.5) / 5.0
        assert payload.message == "halfway through analysis"

    async def test_clamps_fraction(self):
        """Test fraction is clamped to [0.0, 1.0]."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        payloads: list[ProgressRun] = []

        async def capture_send(url: str, payload: ProgressRun, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            payloads.append(payload)

        with flow_context(
            webhook_url="http://example.com/progress",
            project_name="test",
            flow_run_id=str(UUID(int=1)),
            flow_name="flow",
            step=1,
            total_steps=1,
            flow_minutes=(1.0,),
            completed_minutes=0.0,
        ):
            with (
                patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client),
                patch("ai_pipeline_core.deployment.progress.send_webhook", side_effect=capture_send),
            ):
                await update(-0.5, "negative")
                await update(1.5, "over")

        assert len(payloads) == 2
        assert payloads[0].step_progress == 0.0
        assert payloads[1].step_progress == 1.0


class TestUpdateLabels:
    """Test that update() refreshes Prefect flow run labels."""

    async def test_updates_prefect_labels(self):
        """update() must call update_flow_run_labels with correct progress data."""
        flow_run_id = str(UUID(int=42))
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with flow_context(
            webhook_url="http://example.com/progress",
            project_name="test-project",
            flow_run_id=flow_run_id,
            flow_name="analysis",
            step=2,
            total_steps=4,
            flow_minutes=(1.0, 2.0, 1.0, 1.0),
            completed_minutes=1.0,
        ):
            with (
                patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client),
                patch("ai_pipeline_core.deployment.progress.send_webhook", new_callable=AsyncMock),
            ):
                await update(0.5, "halfway")

        mock_client.update_flow_run_labels.assert_called_once()
        labels = mock_client.update_flow_run_labels.call_args.kwargs["labels"]
        assert labels["progress.step"] == 2
        assert labels["progress.total_steps"] == 4
        assert labels["progress.flow_name"] == "analysis"
        assert labels["progress.status"] == "progress"
        assert labels["progress.step_progress"] == 0.5
        assert labels["progress.progress"] == 0.4  # (1.0 + 2.0 * 0.5) / 5.0
        assert labels["progress.message"] == "halfway"

    async def test_no_labels_without_flow_run_id(self):
        """No label update when flow_run_id is empty (CLI mode)."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with flow_context(
            webhook_url="http://example.com/progress",
            project_name="test",
            flow_run_id="",
            flow_name="flow",
            step=1,
            total_steps=1,
            flow_minutes=(1.0,),
            completed_minutes=0.0,
        ):
            with (
                patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client),
                patch("ai_pipeline_core.deployment.progress.send_webhook", new_callable=AsyncMock) as mock_send,
            ):
                await update(0.5, "msg")

        # Webhook still sent (webhook_url is set)
        mock_send.assert_called_once()
        # But no label update (no flow_run_id)
        mock_client.update_flow_run_labels.assert_not_called()

    async def test_label_failure_does_not_raise(self):
        """Failed label update is logged but does not crash the flow."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.update_flow_run_labels.side_effect = Exception("Prefect unavailable")

        with flow_context(
            webhook_url="http://example.com/progress",
            project_name="test",
            flow_run_id=str(UUID(int=1)),
            flow_name="flow",
            step=1,
            total_steps=1,
            flow_minutes=(1.0,),
            completed_minutes=0.0,
        ):
            with (
                patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client),
                patch("ai_pipeline_core.deployment.progress.send_webhook", new_callable=AsyncMock) as mock_send,
            ):
                await update(0.5, "msg")  # Must not raise

        # Webhook still sent despite label failure
        mock_send.assert_called_once()


class TestFlowContext:
    """Test flow_context context manager."""

    def test_sets_and_resets_context(self):
        """Test context is set during the block and reset after."""
        from ai_pipeline_core.deployment.progress import _context

        assert _context.get() is None

        with flow_context(
            webhook_url="http://example.com",
            project_name="test",
            flow_run_id=str(UUID(int=1)),
            flow_name="flow",
            step=1,
            total_steps=2,
            flow_minutes=(1.0, 2.0),
            completed_minutes=0.0,
        ):
            ctx = _context.get()
            assert ctx is not None
            assert ctx.project_name == "test"
            assert ctx.flow_name == "flow"
            assert ctx.current_flow_minutes == 1.0

        assert _context.get() is None

    def test_calculates_current_flow_minutes(self):
        """Test weight extraction from weights tuple."""
        from ai_pipeline_core.deployment.progress import _context

        with flow_context(
            webhook_url="http://example.com",
            project_name="test",
            flow_run_id=str(UUID(int=1)),
            flow_name="flow2",
            step=2,
            total_steps=3,
            flow_minutes=(10.0, 20.0, 30.0),
            completed_minutes=10.0,
        ):
            ctx = _context.get()
            assert ctx is not None
            assert ctx.current_flow_minutes == 20.0


class TestProgressContext:
    """Test ProgressContext dataclass."""

    def test_creation(self):
        """Test ProgressContext creation."""
        ctx = ProgressContext(
            webhook_url="http://example.com",
            project_name="test",
            flow_run_id=str(UUID(int=1)),
            flow_name="flow",
            step=1,
            total_steps=3,
            total_minutes=6.0,
            completed_minutes=0.0,
            current_flow_minutes=1.0,
        )
        assert ctx.step == 1
        assert ctx.total_steps == 3

    def test_frozen(self):
        """Test ProgressContext is immutable."""
        ctx = ProgressContext(
            webhook_url="http://example.com",
            project_name="test",
            flow_run_id=str(UUID(int=1)),
            flow_name="flow",
            step=1,
            total_steps=1,
            total_minutes=1.0,
            completed_minutes=0.0,
            current_flow_minutes=1.0,
        )
        with pytest.raises(AttributeError):
            ctx.step = 2  # type: ignore[misc]


# --- Integration tests: ContextVar propagation through @pipeline_flow + Prefect ---


@pipeline_flow(estimated_minutes=1)
async def _progress_test_flow(
    project_name: str,
    documents: list[Document],
    flow_options: FlowOptions,
) -> list[_ProgressTestDoc]:
    """Test flow that calls progress_update."""
    await update(0.25, "quarter")
    await update(0.75, "three-quarters")
    return [_ProgressTestDoc.create(name="test.md", content="done")]


class TestProgressIntegration:
    """Integration tests: progress_update() called from inside @pipeline_flow."""

    @pytest.mark.integration
    async def test_progress_propagates_through_pipeline_flow(self):
        """ContextVar set via flow_context must be visible inside @pipeline_flow."""
        payloads: list[ProgressRun] = []

        async def capture_send(url: str, payload: ProgressRun, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            payloads.append(payload)

        with flow_context(
            webhook_url="http://test.example.com/progress",
            project_name="test-project",
            flow_run_id=str(UUID(int=1)),
            flow_name="progress_test_flow",
            step=1,
            total_steps=1,
            flow_minutes=(1.0,),
            completed_minutes=0.0,
        ):
            with (
                patch("ai_pipeline_core.deployment.progress.send_webhook", side_effect=capture_send),
                patch("ai_pipeline_core.deployment.progress.get_client") as mock_gc,
            ):
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_gc.return_value = mock_client
                with prefect_test_harness():
                    await _progress_test_flow("test-project", [], FlowOptions())

        assert len(payloads) == 2
        assert payloads[0].step_progress == 0.25
        assert payloads[0].message == "quarter"
        assert payloads[0].status == "progress"
        assert payloads[1].step_progress == 0.75
        assert payloads[1].message == "three-quarters"

    @pytest.mark.integration
    async def test_progress_overall_calculation(self):
        """Verify overall progress is computed correctly from step weights."""
        payloads: list[ProgressRun] = []

        async def capture_send(url: str, payload: ProgressRun, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            payloads.append(payload)

        # Simulate step 2 of 3 flows, with minutes (10, 20, 30), 10 completed
        with flow_context(
            webhook_url="http://test.example.com/progress",
            project_name="test-project",
            flow_run_id=str(UUID(int=1)),
            flow_name="progress_test_flow",
            step=2,
            total_steps=3,
            flow_minutes=(10.0, 20.0, 30.0),
            completed_minutes=10.0,
        ):
            with (
                patch("ai_pipeline_core.deployment.progress.send_webhook", side_effect=capture_send),
                patch("ai_pipeline_core.deployment.progress.get_client") as mock_gc,
            ):
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_gc.return_value = mock_client
                with prefect_test_harness():
                    await _progress_test_flow("test-project", [], FlowOptions())

        assert len(payloads) == 2
        # First update: fraction=0.25, overall = (10 + 20*0.25) / 60 = 15/60 = 0.25
        assert payloads[0].progress == 0.25
        # Second update: fraction=0.75, overall = (10 + 20*0.75) / 60 = 25/60 ~ 0.4167
        assert payloads[1].progress == pytest.approx(0.4167, abs=0.001)


class TestInvalidFlowRunId:
    """Test handling of non-UUID flow_run_id values (e.g. str(None) → "None")."""

    async def test_non_uuid_flow_run_id_does_not_raise(self):
        """update() must not raise when flow_run_id is a non-UUID string.

        Regression: runtime.flow_run.get_id() can return None, and
        str(None) produces "None" which passes truthiness but fails UUID().
        """
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with flow_context(
            webhook_url="",
            project_name="test",
            flow_run_id="None",  # str(None) — the actual bug scenario
            flow_name="flow",
            step=1,
            total_steps=1,
            flow_minutes=(1.0,),
            completed_minutes=0.0,
        ):
            with patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client):
                await update(0.5, "halfway")  # Must not raise ValueError

        # Label update should be skipped for invalid UUID
        mock_client.update_flow_run_labels.assert_not_called()

    async def test_non_uuid_flow_run_id_skips_webhook_uuid(self):
        """Webhook payload uses UUID(int=0) fallback for invalid flow_run_id."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with flow_context(
            webhook_url="http://example.com/progress",
            project_name="test",
            flow_run_id="not-a-uuid",
            flow_name="flow",
            step=1,
            total_steps=1,
            flow_minutes=(1.0,),
            completed_minutes=0.0,
        ):
            with (
                patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client),
                patch("ai_pipeline_core.deployment.progress.send_webhook", new_callable=AsyncMock) as mock_send,
            ):
                await update(0.5, "halfway")  # Must not raise

        # Webhook should still be sent with fallback UUID
        mock_send.assert_called_once()
        payload = mock_send.call_args[0][1]
        assert payload.flow_run_id == UUID(int=0)

        # Label update should be skipped for invalid UUID
        mock_client.update_flow_run_labels.assert_not_called()


class TestProgressWebhookFailure:
    """Test that webhook failure in update() is logged and doesn't propagate."""

    async def test_webhook_send_failure_does_not_raise(self):
        """If send_webhook raises, update() must log a warning and not propagate."""
        mock_send = AsyncMock(side_effect=ConnectionError("webhook down"))
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.update_flow_run_labels = AsyncMock()

        with (
            patch("ai_pipeline_core.deployment.progress.send_webhook", mock_send),
            patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client),
            flow_context(
                webhook_url="http://will-fail.example",
                project_name="test",
                flow_run_id="00000000-0000-0000-0000-000000000001",
                flow_name="test_flow",
                step=1,
                total_steps=1,
                flow_minutes=(1.0,),
                completed_minutes=0.0,
            ),
        ):
            # Should NOT raise despite webhook failure
            await update(0.5, "half done")

        mock_send.assert_called_once()
