"""Tests for progress tracking module."""

# pyright: reportPrivateUsage=false, reportOptionalMemberAccess=false

from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest

from ai_pipeline_core.deployment._types import MemoryPublisher
from ai_pipeline_core.deployment._types import ProgressEvent
from ai_pipeline_core.deployment.contract import FlowStatus
from ai_pipeline_core.deployment.progress import _ProgressContext, _flow_context, progress_update
from ai_pipeline_core.documents import Document
from ai_pipeline_core.pipeline import pipeline_flow
from ai_pipeline_core.pipeline.options import FlowOptions


class _ProgressInputDoc(Document):
    """Minimal input document for progress integration tests."""


class _ProgressTestDoc(Document):
    """Minimal output document for progress integration tests."""


class TestUpdate:
    """Test progress update function."""

    async def test_noop_without_context(self):
        """Test update is a no-op when no context is set."""
        await progress_update(0.5, "test")  # Should not raise

    async def test_updates_labels(self):
        """Test labels are updated when context is set."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with _flow_context(
            "test",
            str(UUID(int=1)),
            "flow1",
            1,
            total_steps=3,
            flow_minutes=(1.0, 1.0, 1.0),
            completed_minutes=0.0,
        ):
            with patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client):
                await progress_update(0.5, "halfway")

        mock_client.update_flow_run_labels.assert_called_once()

    async def test_clamps_fraction(self):
        """Test fraction is clamped to [0.0, 1.0]."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with _flow_context(
            "test",
            str(UUID(int=1)),
            "flow",
            1,
            total_steps=1,
            flow_minutes=(1.0,),
            completed_minutes=0.0,
        ):
            with patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client):
                await progress_update(-0.5, "negative")
                await progress_update(1.5, "over")

        calls = mock_client.update_flow_run_labels.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["labels"]["progress.step_progress"] == 0.0
        assert calls[1].kwargs["labels"]["progress.step_progress"] == 1.0


class TestUpdateLabels:
    """Test that update() refreshes Prefect flow run labels."""

    async def test_updates_prefect_labels(self):
        """update() must call update_flow_run_labels with correct progress data."""
        flow_run_id = str(UUID(int=42))
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with _flow_context(
            "test-project",
            flow_run_id,
            "analysis",
            2,
            total_steps=4,
            flow_minutes=(1.0, 2.0, 1.0, 1.0),
            completed_minutes=1.0,
        ):
            with patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client):
                await progress_update(0.5, "halfway")

        mock_client.update_flow_run_labels.assert_called_once()
        labels = mock_client.update_flow_run_labels.call_args.kwargs["labels"]
        assert labels["progress.step"] == 2
        assert labels["progress.total_steps"] == 4
        assert labels["progress.flow_name"] == "analysis"
        assert labels["progress.status"] == FlowStatus.PROGRESS
        assert labels["progress.step_progress"] == 0.5
        assert labels["progress.progress"] == 0.4  # (1.0 + 2.0 * 0.5) / 5.0
        assert labels["progress.message"] == "halfway"

    async def test_no_labels_without_flow_run_id(self):
        """No label update when flow_run_id is empty (CLI mode)."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with _flow_context(
            "test",
            "",
            "flow",
            1,
            total_steps=1,
            flow_minutes=(1.0,),
            completed_minutes=0.0,
        ):
            with patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client):
                await progress_update(0.5, "msg")

        mock_client.update_flow_run_labels.assert_not_called()

    async def test_label_failure_does_not_raise(self):
        """Failed label update is logged but does not crash the flow."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.update_flow_run_labels.side_effect = Exception("Prefect unavailable")

        with _flow_context(
            "test",
            str(UUID(int=1)),
            "flow",
            1,
            total_steps=1,
            flow_minutes=(1.0,),
            completed_minutes=0.0,
        ):
            with patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client):
                await progress_update(0.5, "msg")  # Must not raise


class TestFlowContext:
    """Test _flow_context context manager."""

    def test_sets_and_resets_context(self):
        """Test context is set during the block and reset after."""
        from ai_pipeline_core.deployment.progress import _context

        assert _context.get() is None

        with _flow_context(
            "test",
            str(UUID(int=1)),
            "flow",
            1,
            total_steps=2,
            flow_minutes=(1.0, 2.0),
            completed_minutes=0.0,
        ):
            ctx = _context.get()
            assert ctx is not None
            assert ctx.run_id == "test"
            assert ctx.flow_name == "flow"
            assert ctx.current_flow_minutes == 1.0

        assert _context.get() is None

    def test_calculates_current_flow_minutes(self):
        """Test weight extraction from weights tuple."""
        from ai_pipeline_core.deployment.progress import _context

        with _flow_context(
            "test",
            str(UUID(int=1)),
            "flow2",
            2,
            total_steps=3,
            flow_minutes=(10.0, 20.0, 30.0),
            completed_minutes=10.0,
        ):
            ctx = _context.get()
            assert ctx is not None
            assert ctx.current_flow_minutes == 20.0


class TestProgressContext:
    """Test _ProgressContext dataclass."""

    def test_creation(self):
        """Test _ProgressContext creation."""
        ctx = _ProgressContext(
            run_id="test",
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
        """Test _ProgressContext is immutable."""
        ctx = _ProgressContext(
            run_id="test",
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
    run_id: str,
    documents: list[_ProgressInputDoc],
    flow_options: FlowOptions,
) -> list[_ProgressTestDoc]:
    """Test flow that calls progress_update."""
    await progress_update(0.25, "quarter")
    await progress_update(0.75, "three-quarters")
    return [_ProgressTestDoc.create_root(name="test.md", content="done", reason="test input")]


class TestProgressIntegration:
    """Integration tests: progress_update() called from inside @pipeline_flow."""

    @pytest.mark.integration
    async def test_progress_propagates_through_pipeline_flow(self):
        """ContextVar set via _flow_context must be visible inside @pipeline_flow."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with _flow_context(
            "test-project",
            str(UUID(int=1)),
            "progress_test_flow",
            1,
            total_steps=1,
            flow_minutes=(1.0,),
            completed_minutes=0.0,
        ):
            with patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client):
                await _progress_test_flow("test-project", [], FlowOptions())

        calls = mock_client.update_flow_run_labels.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["labels"]["progress.step_progress"] == 0.25
        assert calls[0].kwargs["labels"]["progress.message"] == "quarter"
        assert calls[1].kwargs["labels"]["progress.step_progress"] == 0.75
        assert calls[1].kwargs["labels"]["progress.message"] == "three-quarters"

    @pytest.mark.integration
    async def test_progress_overall_calculation(self):
        """Verify overall progress is computed correctly from step weights."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # Simulate step 2 of 3 flows, with minutes (10, 20, 30), 10 completed
        with _flow_context(
            "test-project",
            str(UUID(int=1)),
            "progress_test_flow",
            2,
            total_steps=3,
            flow_minutes=(10.0, 20.0, 30.0),
            completed_minutes=10.0,
        ):
            with patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client):
                await _progress_test_flow("test-project", [], FlowOptions())

        calls = mock_client.update_flow_run_labels.call_args_list
        assert len(calls) == 2
        # First update: fraction=0.25, overall = (10 + 20*0.25) / 60 = 15/60 = 0.25
        assert calls[0].kwargs["labels"]["progress.progress"] == 0.25
        # Second update: fraction=0.75, overall = (10 + 20*0.75) / 60 = 25/60 ~ 0.4167
        assert calls[1].kwargs["labels"]["progress.progress"] == pytest.approx(0.4167, abs=0.001)


class TestInvalidFlowRunId:
    """Test handling of non-UUID flow_run_id values (e.g. str(None) -> "None")."""

    async def test_non_uuid_flow_run_id_does_not_raise(self):
        """update() must not raise when flow_run_id is a non-UUID string.

        Regression: runtime.flow_run.get_id() can return None, and
        str(None) produces "None" which passes truthiness but fails UUID().
        """
        with _flow_context(
            "test",
            "None",  # str(None) — the actual bug scenario
            "flow",
            1,
            total_steps=1,
            flow_minutes=(1.0,),
            completed_minutes=0.0,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            with patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client):
                await progress_update(0.5, "halfway")  # Must not raise ValueError

        # Label update should be skipped for invalid UUID
        mock_client.update_flow_run_labels.assert_not_called()


class TestPublisherIntegration:
    """Test that update() publishes ProgressEvent via the publisher."""

    async def test_publishes_progress_event(self):
        """update() must fire-and-forget a ProgressEvent via publisher."""
        import asyncio

        pub = MemoryPublisher()
        with _flow_context(
            "test-run",
            str(UUID(int=1)),
            "analysis",
            2,
            total_steps=4,
            flow_minutes=(1.0, 2.0, 1.0, 1.0),
            completed_minutes=1.0,
            publisher=pub,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            with patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client):
                await progress_update(0.5, "halfway")
                # Allow fire-and-forget task to complete
                await asyncio.sleep(0.01)

        assert len(pub.events) == 1
        event = pub.events[0]
        assert isinstance(event, ProgressEvent)
        assert event.run_id == "test-run"
        assert event.flow_name == "analysis"
        assert event.step == 2
        assert event.step_progress == 0.5
        assert event.message == "halfway"

    async def test_publisher_error_does_not_crash_flow(self):
        """Publisher failure must not crash the flow or prevent label update."""
        import asyncio

        failing_pub = AsyncMock(spec=MemoryPublisher)
        failing_pub.publish_progress.side_effect = RuntimeError("pub error")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with _flow_context(
            "test",
            str(UUID(int=1)),
            "flow",
            1,
            total_steps=1,
            flow_minutes=(1.0,),
            completed_minutes=0.0,
            publisher=failing_pub,
        ):
            with patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client):
                await progress_update(0.5, "test")  # Must not raise
                await asyncio.sleep(0.01)

        # Labels should still be updated even if publisher fails
        mock_client.update_flow_run_labels.assert_called_once()

    async def test_no_publish_without_publisher(self):
        """update() with publisher=None does not attempt to publish."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with _flow_context(
            "test",
            str(UUID(int=1)),
            "flow",
            1,
            total_steps=1,
            flow_minutes=(1.0,),
            completed_minutes=0.0,
            publisher=None,
        ):
            with patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client):
                await progress_update(0.5, "msg")  # Should work fine without publisher

        # Labels should still be updated
        mock_client.update_flow_run_labels.assert_called_once()

    async def test_publish_event_has_correct_overall_progress(self):
        """ProgressEvent overall progress is computed from step weights."""
        import asyncio

        pub = MemoryPublisher()
        # Step 2 of 3, minutes (10, 20, 30), 10 completed, fraction 0.5
        # Expected overall: (10 + 20*0.5) / 60 = 20/60 = 0.3333
        with _flow_context(
            "test",
            str(UUID(int=1)),
            "step2",
            2,
            total_steps=3,
            flow_minutes=(10.0, 20.0, 30.0),
            completed_minutes=10.0,
            publisher=pub,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            with patch("ai_pipeline_core.deployment.progress.get_client", return_value=mock_client):
                await progress_update(0.5, "halfway")
                await asyncio.sleep(0.01)

        assert len(pub.events) == 1
        assert pub.events[0].progress == pytest.approx(0.3333, abs=0.001)
