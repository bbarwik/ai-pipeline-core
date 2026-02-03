"""Tests for progress tracking module."""

# pyright: reportPrivateUsage=false, reportOptionalMemberAccess=false

import asyncio
from uuid import UUID

import pytest

from ai_pipeline_core.deployment.contract import ProgressRun
from ai_pipeline_core.deployment.progress import ProgressContext, flow_context, update
from ai_pipeline_core.documents import Document
from ai_pipeline_core.pipeline import pipeline_flow
from ai_pipeline_core.pipeline.options import FlowOptions
from ai_pipeline_core.testing import prefect_test_harness


class _ProgressTestDoc(Document):
    """Minimal document for progress integration tests."""


class TestUpdate:
    """Test progress update function."""

    async def test_noop_without_context(self):
        """Test update is a no-op when no context is set."""
        await update(0.5, "test")  # Should not raise

    async def test_no_webhook_without_webhook_url(self):
        """Test webhook is not enqueued when webhook_url is empty (labels still update)."""
        from unittest.mock import AsyncMock, patch

        queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with flow_context(
            webhook_url="",
            project_name="test",
            run_id="run-1",
            flow_run_id=str(UUID(int=1)),
            flow_name="flow1",
            step=1,
            total_steps=3,
            flow_minutes=(1.0, 1.0, 1.0),
            completed_minutes=0.0,
            queue=queue,
        ):
            with patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client):
                await update(0.5, "halfway")

        assert queue.empty()
        mock_client.update_flow_run_labels.assert_called_once()

    async def test_sends_progress_payload(self):
        """Test update creates and enqueues ProgressRun."""
        queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()

        with flow_context(
            webhook_url="http://example.com/progress",
            project_name="my-project",
            run_id="run-1",
            flow_run_id=str(UUID(int=42)),
            flow_name="analysis",
            step=2,
            total_steps=4,
            flow_minutes=(1.0, 2.0, 1.0, 1.0),
            completed_minutes=1.0,
            queue=queue,
        ):
            await update(0.5, "halfway through analysis")

        assert not queue.empty()
        payload = queue.get_nowait()
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
        queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()

        with flow_context(
            webhook_url="http://example.com/progress",
            project_name="test",
            run_id="run-1",
            flow_run_id=str(UUID(int=1)),
            flow_name="flow",
            step=1,
            total_steps=1,
            flow_minutes=(1.0,),
            completed_minutes=0.0,
            queue=queue,
        ):
            await update(-0.5, "negative")
            await update(1.5, "over")

        p1 = queue.get_nowait()
        p2 = queue.get_nowait()
        assert p1.step_progress == 0.0
        assert p2.step_progress == 1.0


class TestUpdateLabels:
    """Test that update() refreshes Prefect flow run labels."""

    async def test_updates_prefect_labels(self):
        """update() must call update_flow_run_labels with correct progress data."""
        from unittest.mock import AsyncMock, patch

        queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()
        flow_run_id = str(UUID(int=42))
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with flow_context(
            webhook_url="http://example.com/progress",
            project_name="test-project",
            run_id="run-1",
            flow_run_id=flow_run_id,
            flow_name="analysis",
            step=2,
            total_steps=4,
            flow_minutes=(1.0, 2.0, 1.0, 1.0),
            completed_minutes=1.0,
            queue=queue,
        ):
            with patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client):
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
        from unittest.mock import AsyncMock, patch

        queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with flow_context(
            webhook_url="http://example.com/progress",
            project_name="test",
            run_id="run-1",
            flow_run_id="",
            flow_name="flow",
            step=1,
            total_steps=1,
            flow_minutes=(1.0,),
            completed_minutes=0.0,
            queue=queue,
        ):
            with patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client):
                await update(0.5, "msg")

        # Webhook still enqueued
        assert not queue.empty()
        # But no label update
        mock_client.update_flow_run_labels.assert_not_called()

    async def test_label_failure_does_not_raise(self):
        """Failed label update is logged but does not crash the flow."""
        from unittest.mock import AsyncMock, patch

        queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.update_flow_run_labels.side_effect = Exception("Prefect unavailable")

        with flow_context(
            webhook_url="http://example.com/progress",
            project_name="test",
            run_id="run-1",
            flow_run_id=str(UUID(int=1)),
            flow_name="flow",
            step=1,
            total_steps=1,
            flow_minutes=(1.0,),
            completed_minutes=0.0,
            queue=queue,
        ):
            with patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client):
                await update(0.5, "msg")  # Must not raise

        # Webhook still enqueued despite label failure
        assert not queue.empty()


class TestFlowContext:
    """Test flow_context context manager."""

    def test_sets_and_resets_context(self):
        """Test context is set during the block and reset after."""
        from ai_pipeline_core.deployment.progress import _context

        queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()

        assert _context.get() is None

        with flow_context(
            webhook_url="http://example.com",
            project_name="test",
            run_id="run-1",
            flow_run_id=str(UUID(int=1)),
            flow_name="flow",
            step=1,
            total_steps=2,
            flow_minutes=(1.0, 2.0),
            completed_minutes=0.0,
            queue=queue,
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

        queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()

        with flow_context(
            webhook_url="http://example.com",
            project_name="test",
            run_id="run-1",
            flow_run_id=str(UUID(int=1)),
            flow_name="flow2",
            step=2,
            total_steps=3,
            flow_minutes=(10.0, 20.0, 30.0),
            completed_minutes=10.0,
            queue=queue,
        ):
            ctx = _context.get()
            assert ctx is not None
            assert ctx.current_flow_minutes == 20.0


class TestProgressContext:
    """Test ProgressContext dataclass."""

    def test_creation(self):
        """Test ProgressContext creation."""
        queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()
        ctx = ProgressContext(
            webhook_url="http://example.com",
            project_name="test",
            run_id="r1",
            flow_run_id=str(UUID(int=1)),
            flow_name="flow",
            step=1,
            total_steps=3,
            total_minutes=6.0,
            completed_minutes=0.0,
            current_flow_minutes=1.0,
            queue=queue,
        )
        assert ctx.step == 1
        assert ctx.total_steps == 3

    def test_frozen(self):
        """Test ProgressContext is immutable."""
        import pytest

        queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()
        ctx = ProgressContext(
            webhook_url="http://example.com",
            project_name="test",
            run_id="r1",
            flow_run_id=str(UUID(int=1)),
            flow_name="flow",
            step=1,
            total_steps=1,
            total_minutes=1.0,
            completed_minutes=0.0,
            current_flow_minutes=1.0,
            queue=queue,
        )
        with pytest.raises(AttributeError):
            ctx.step = 2  # type: ignore[misc]


class TestWebhookWorker:
    """Test webhook_worker async function."""

    async def test_processes_payloads(self):
        """Test worker processes payloads and stops on None."""
        from unittest.mock import AsyncMock, patch

        from ai_pipeline_core.deployment.progress import webhook_worker

        queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()

        payload = ProgressRun(
            flow_run_id=UUID(int=1),
            project_name="test",
            state="RUNNING",
            timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            step=1,
            total_steps=1,
            flow_name="flow",
            status="progress",
            progress=0.5,
            step_progress=0.5,
            message="halfway",
        )

        queue.put_nowait(payload)
        queue.put_nowait(None)  # Signal to stop

        with patch("ai_pipeline_core.deployment.progress.send_webhook", new_callable=AsyncMock) as mock_send:
            await webhook_worker(queue, "http://example.com/hook")
            mock_send.assert_called_once()

    async def test_continues_on_exception(self):
        """Test worker continues processing after send failure."""
        from unittest.mock import AsyncMock, patch

        from ai_pipeline_core.deployment.progress import webhook_worker

        queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()

        payload = ProgressRun(
            flow_run_id=UUID(int=1),
            project_name="test",
            state="RUNNING",
            timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            step=1,
            total_steps=1,
            flow_name="flow",
            status="progress",
            progress=0.5,
            step_progress=0.5,
            message="msg",
        )

        queue.put_nowait(payload)
        queue.put_nowait(None)

        with patch(
            "ai_pipeline_core.deployment.progress.send_webhook",
            new_callable=AsyncMock,
            side_effect=Exception("Network error"),
        ) as mock_send:
            await webhook_worker(queue, "http://example.com/hook")
            # Should complete without raising, but send_webhook must have been called
            mock_send.assert_called_once()


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
        queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()

        with flow_context(
            webhook_url="http://test.example.com/progress",
            project_name="test-project",
            run_id="test-run",
            flow_run_id=str(UUID(int=1)),
            flow_name="progress_test_flow",
            step=1,
            total_steps=1,
            flow_minutes=(1.0,),
            completed_minutes=0.0,
            queue=queue,
        ):
            with prefect_test_harness():
                await _progress_test_flow("test-project", [], FlowOptions())

        payloads: list[ProgressRun] = []
        while not queue.empty():
            item = queue.get_nowait()
            if item is not None:
                payloads.append(item)

        assert len(payloads) == 2
        assert payloads[0].step_progress == 0.25
        assert payloads[0].message == "quarter"
        assert payloads[0].status == "progress"
        assert payloads[1].step_progress == 0.75
        assert payloads[1].message == "three-quarters"

    @pytest.mark.integration
    async def test_progress_overall_calculation(self):
        """Verify overall progress is computed correctly from step weights."""
        queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()

        # Simulate step 2 of 3 flows, with minutes (10, 20, 30), 10 completed
        with flow_context(
            webhook_url="http://test.example.com/progress",
            project_name="test-project",
            run_id="test-run",
            flow_run_id=str(UUID(int=1)),
            flow_name="progress_test_flow",
            step=2,
            total_steps=3,
            flow_minutes=(10.0, 20.0, 30.0),
            completed_minutes=10.0,
            queue=queue,
        ):
            with prefect_test_harness():
                await _progress_test_flow("test-project", [], FlowOptions())

        payloads: list[ProgressRun] = []
        while not queue.empty():
            item = queue.get_nowait()
            if item is not None:
                payloads.append(item)

        assert len(payloads) == 2
        # First update: fraction=0.25, overall = (10 + 20*0.25) / 60 = 15/60 = 0.25
        assert payloads[0].progress == 0.25
        # Second update: fraction=0.75, overall = (10 + 20*0.75) / 60 = 25/60 ≈ 0.4167
        assert payloads[1].progress == pytest.approx(0.4167, abs=0.001)
