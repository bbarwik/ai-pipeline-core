"""Publisher implementation tests — NoopPublisher, MemoryPublisher, all event types."""

from ai_pipeline_core.deployment import _MemoryPublisher, _NoopPublisher
from ai_pipeline_core.deployment._types import (
    ErrorCode,
    FlowCompletedEvent,
    FlowFailedEvent,
    FlowSkippedEvent,
    FlowStartedEvent,
    ProgressEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)


# ---------------------------------------------------------------------------
# _NoopPublisher
# ---------------------------------------------------------------------------


class TestNoopPublisher:
    async def test_accepts_run_started(self):
        pub = _NoopPublisher()
        await pub.publish_run_started(RunStartedEvent(run_id="r", flow_run_id="f", run_scope="s", flow_plan=[]))

    async def test_accepts_run_completed(self):
        pub = _NoopPublisher()
        await pub.publish_run_completed(RunCompletedEvent(run_id="r", flow_run_id="f", result={}, chain_context={}, actual_cost=0.0))

    async def test_accepts_run_failed(self):
        pub = _NoopPublisher()
        await pub.publish_run_failed(RunFailedEvent(run_id="r", flow_run_id="f", error_code=ErrorCode.UNKNOWN, error_message="x"))

    async def test_accepts_progress(self):
        pub = _NoopPublisher()
        await pub.publish_progress(
            ProgressEvent(
                run_id="r",
                flow_run_id="f",
                flow_name="flow",
                step=1,
                total_steps=1,
                progress=1.0,
                step_progress=1.0,
                status="completed",
                message="done",
            )
        )

    async def test_accepts_heartbeat(self):
        pub = _NoopPublisher()
        await pub.publish_heartbeat("run-1")

    async def test_accepts_flow_started(self):
        pub = _NoopPublisher()
        await pub.publish_flow_started(FlowStartedEvent(run_id="r", flow_name="flow", flow_class="MyFlow", step=1, total_steps=2))

    async def test_accepts_flow_completed(self):
        pub = _NoopPublisher()
        await pub.publish_flow_completed(FlowCompletedEvent(run_id="r", flow_name="flow", flow_class="MyFlow", step=1, total_steps=2, duration_ms=100))

    async def test_accepts_flow_failed(self):
        pub = _NoopPublisher()
        await pub.publish_flow_failed(FlowFailedEvent(run_id="r", flow_name="flow", flow_class="MyFlow", step=1, total_steps=2, error_message="boom"))

    async def test_accepts_flow_skipped(self):
        pub = _NoopPublisher()
        await pub.publish_flow_skipped(FlowSkippedEvent(run_id="r", flow_name="flow", step=1, total_steps=2, reason="cached"))

    async def test_accepts_task_started(self):
        pub = _NoopPublisher()
        await pub.publish_task_started(
            TaskStartedEvent(
                run_id="r",
                flow_name="flow",
                step=1,
                task_name="t",
                task_class="T",
                task_invocation_id="inv-1",
                parent_task=None,
                task_depth=0,
            )
        )

    async def test_accepts_task_completed(self):
        pub = _NoopPublisher()
        await pub.publish_task_completed(
            TaskCompletedEvent(
                run_id="r",
                flow_name="flow",
                step=1,
                task_name="t",
                task_class="T",
                task_invocation_id="inv-1",
                parent_task=None,
                task_depth=0,
                duration_ms=50,
            )
        )

    async def test_accepts_task_failed(self):
        pub = _NoopPublisher()
        await pub.publish_task_failed(
            TaskFailedEvent(
                run_id="r",
                flow_name="flow",
                step=1,
                task_name="t",
                task_class="T",
                task_invocation_id="inv-1",
                parent_task=None,
                task_depth=0,
                error_message="err",
            )
        )

    async def test_close(self):
        pub = _NoopPublisher()
        await pub.close()


# ---------------------------------------------------------------------------
# _MemoryPublisher
# ---------------------------------------------------------------------------


class TestMemoryPublisher:
    async def test_records_run_started(self):
        pub = _MemoryPublisher()
        event = RunStartedEvent(run_id="r", flow_run_id="f", run_scope="s", flow_plan=[])
        await pub.publish_run_started(event)
        assert pub.events == [event]

    async def test_records_run_completed(self):
        pub = _MemoryPublisher()
        event = RunCompletedEvent(run_id="r", flow_run_id="f", result={}, chain_context={}, actual_cost=0.0)
        await pub.publish_run_completed(event)
        assert pub.events == [event]

    async def test_records_run_failed(self):
        pub = _MemoryPublisher()
        event = RunFailedEvent(run_id="r", flow_run_id="f", error_code=ErrorCode.PIPELINE_ERROR, error_message="fail")
        await pub.publish_run_failed(event)
        assert pub.events == [event]

    async def test_records_progress(self):
        pub = _MemoryPublisher()
        event = ProgressEvent(
            run_id="r", flow_run_id="f", flow_name="f", step=1, total_steps=1, progress=0.5, step_progress=0.5, status="progress", message="half"
        )
        await pub.publish_progress(event)
        assert pub.events == [event]

    async def test_records_heartbeats(self):
        pub = _MemoryPublisher()
        await pub.publish_heartbeat("run-1")
        await pub.publish_heartbeat("run-1")
        assert pub.heartbeats == ["run-1", "run-1"]

    async def test_records_flow_events(self):
        pub = _MemoryPublisher()
        started = FlowStartedEvent(run_id="r", flow_name="f", flow_class="F", step=1, total_steps=2)
        completed = FlowCompletedEvent(run_id="r", flow_name="f", flow_class="F", step=1, total_steps=2, duration_ms=100)
        skipped = FlowSkippedEvent(run_id="r", flow_name="f2", step=2, total_steps=2, reason="cached")
        await pub.publish_flow_started(started)
        await pub.publish_flow_completed(completed)
        await pub.publish_flow_skipped(skipped)
        assert pub.events == [started, completed, skipped]

    async def test_records_task_events(self):
        pub = _MemoryPublisher()
        started = TaskStartedEvent(run_id="r", flow_name="f", step=1, task_name="t", task_class="T", task_invocation_id="inv-1", parent_task=None, task_depth=0)
        completed = TaskCompletedEvent(
            run_id="r", flow_name="f", step=1, task_name="t", task_class="T", task_invocation_id="inv-1", parent_task=None, task_depth=0, duration_ms=50
        )
        failed = TaskFailedEvent(
            run_id="r", flow_name="f", step=1, task_name="t2", task_class="T2", task_invocation_id="inv-2", parent_task="t", task_depth=1, error_message="err"
        )
        await pub.publish_task_started(started)
        await pub.publish_task_completed(completed)
        await pub.publish_task_failed(failed)
        assert pub.events == [started, completed, failed]

    async def test_event_ordering_preserved(self):
        """Events must be appended in call order."""
        pub = _MemoryPublisher()
        e1 = RunStartedEvent(run_id="r", flow_run_id="f", run_scope="s", flow_plan=[])
        e2 = FlowStartedEvent(run_id="r", flow_name="f1", flow_class="F", step=1, total_steps=1)
        e3 = TaskStartedEvent(run_id="r", flow_name="f1", step=1, task_name="t1", task_class="T", task_invocation_id="inv-1", parent_task=None, task_depth=0)
        e4 = TaskCompletedEvent(
            run_id="r", flow_name="f1", step=1, task_name="t1", task_class="T", task_invocation_id="inv-1", parent_task=None, task_depth=0, duration_ms=10
        )
        e5 = FlowCompletedEvent(run_id="r", flow_name="f1", flow_class="F", step=1, total_steps=1, duration_ms=100)
        e6 = RunCompletedEvent(run_id="r", flow_run_id="f", result={}, chain_context={}, actual_cost=0.0)

        await pub.publish_run_started(e1)
        await pub.publish_flow_started(e2)
        await pub.publish_task_started(e3)
        await pub.publish_task_completed(e4)
        await pub.publish_flow_completed(e5)
        await pub.publish_run_completed(e6)

        assert pub.events == [e1, e2, e3, e4, e5, e6]

    async def test_close(self):
        pub = _MemoryPublisher()
        await pub.close()
        assert pub.events == []
