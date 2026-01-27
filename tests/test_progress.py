"""Tests for progress tracking module."""

# pyright: reportPrivateUsage=false, reportOptionalMemberAccess=false

import asyncio
from uuid import UUID

from ai_pipeline_core.deployment.contract import ProgressRun
from ai_pipeline_core.progress import ProgressContext, flow_context, update


class TestUpdate:
    """Test progress update function."""

    async def test_noop_without_context(self):
        """Test update is a no-op when no context is set."""
        await update(0.5, "test")  # Should not raise

    async def test_noop_without_webhook_url(self):
        """Test update is a no-op when webhook_url is empty."""
        queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()

        with flow_context(
            webhook_url="",
            project_name="test",
            run_id="run-1",
            flow_run_id=str(UUID(int=1)),
            flow_name="flow1",
            step=1,
            total_steps=3,
            weights=(1.0, 1.0, 1.0),
            completed_weight=0.0,
            queue=queue,
        ):
            await update(0.5, "halfway")

        assert queue.empty()

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
            weights=(1.0, 2.0, 1.0, 1.0),
            completed_weight=1.0,
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
            weights=(1.0,),
            completed_weight=0.0,
            queue=queue,
        ):
            await update(-0.5, "negative")
            await update(1.5, "over")

        p1 = queue.get_nowait()
        p2 = queue.get_nowait()
        assert p1.step_progress == 0.0
        assert p2.step_progress == 1.0


class TestFlowContext:
    """Test flow_context context manager."""

    def test_sets_and_resets_context(self):
        """Test context is set during the block and reset after."""
        from ai_pipeline_core.progress import _context

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
            weights=(1.0, 2.0),
            completed_weight=0.0,
            queue=queue,
        ):
            ctx = _context.get()
            assert ctx is not None
            assert ctx.project_name == "test"
            assert ctx.flow_name == "flow"
            assert ctx.current_flow_weight == 1.0

        assert _context.get() is None

    def test_calculates_current_flow_weight(self):
        """Test weight extraction from weights tuple."""
        from ai_pipeline_core.progress import _context

        queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()

        with flow_context(
            webhook_url="http://example.com",
            project_name="test",
            run_id="run-1",
            flow_run_id=str(UUID(int=1)),
            flow_name="flow2",
            step=2,
            total_steps=3,
            weights=(10.0, 20.0, 30.0),
            completed_weight=10.0,
            queue=queue,
        ):
            ctx = _context.get()
            assert ctx is not None
            assert ctx.current_flow_weight == 20.0


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
            weights=(1.0, 2.0, 3.0),
            completed_weight=0.0,
            current_flow_weight=1.0,
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
            weights=(1.0,),
            completed_weight=0.0,
            current_flow_weight=1.0,
            queue=queue,
        )
        with pytest.raises(AttributeError):
            ctx.step = 2  # type: ignore[misc]


class TestWebhookWorker:
    """Test webhook_worker async function."""

    async def test_processes_payloads(self):
        """Test worker processes payloads and stops on None."""
        from unittest.mock import AsyncMock, patch

        from ai_pipeline_core.progress import webhook_worker

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

        with patch(
            "ai_pipeline_core.deployment.helpers.send_webhook", new_callable=AsyncMock
        ) as mock_send:
            await webhook_worker(queue, "http://example.com/hook")
            mock_send.assert_called_once()

    async def test_continues_on_exception(self):
        """Test worker continues processing after send failure."""
        from unittest.mock import AsyncMock, patch

        from ai_pipeline_core.progress import webhook_worker

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
            "ai_pipeline_core.deployment.helpers.send_webhook",
            new_callable=AsyncMock,
            side_effect=Exception("Network error"),
        ):
            await webhook_worker(queue, "http://example.com/hook")
            # Should complete without raising
