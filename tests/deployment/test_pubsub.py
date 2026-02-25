"""Tests for PubSubPublisher, TimestampSequencer, _classify_error, and CloudEvents envelope."""

# pyright: reportPrivateUsage=false

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_pipeline_core.deployment._pubsub import (
    CLOUDEVENTS_SPEC_VERSION,
    CRITICAL_MAX_RETRIES,
    MAX_PUBSUB_MESSAGE_BYTES,
    PubSubPublisher,
    ResultTooLargeError,
    TimestampSequencer,
)
from ai_pipeline_core.deployment._types import (
    CompletedEvent,
    ErrorCode,
    FailedEvent,
    ProgressEvent,
    StartedEvent,
)
from ai_pipeline_core.deployment.base import _classify_error
from ai_pipeline_core.deployment.contract import FlowStatus
from ai_pipeline_core.exceptions import LLMError, PipelineCoreError


class TestTimestampSequencer:
    """Test monotonic microsecond sequencer."""

    def test_monotonically_increasing(self):
        """Successive calls produce strictly increasing values."""
        seq = TimestampSequencer()
        values = [seq.next() for _ in range(100)]
        assert values == sorted(values)
        assert len(set(values)) == 100

    def test_microsecond_resolution(self):
        """Values are in microsecond range (>10^15)."""
        seq = TimestampSequencer()
        value = seq.next()
        assert value > 10**15

    def test_restart_safe(self):
        """A new sequencer produces values >= prior ones (wall clock provides ordering)."""
        seq1 = TimestampSequencer()
        val1 = seq1.next()
        seq2 = TimestampSequencer()
        val2 = seq2.next()
        assert val2 >= val1

    def test_rapid_calls_dont_collide(self):
        """Rapid successive calls never produce duplicates."""
        seq = TimestampSequencer()
        values = set()
        for _ in range(1000):
            values.add(seq.next())
        assert len(values) == 1000


class TestClassifyError:
    """Test _classify_error maps exception types to ErrorCode."""

    def test_llm_error(self):
        """LLMError maps to PROVIDER_ERROR."""
        assert _classify_error(LLMError("fail")) == ErrorCode.PROVIDER_ERROR

    def test_cancelled_error(self):
        """CancelledError maps to CANCELLED."""
        assert _classify_error(asyncio.CancelledError()) == ErrorCode.CANCELLED

    def test_timeout_error(self):
        """TimeoutError maps to DURATION_EXCEEDED."""
        assert _classify_error(TimeoutError("timed out")) == ErrorCode.DURATION_EXCEEDED

    def test_value_error(self):
        """ValueError maps to INVALID_INPUT."""
        assert _classify_error(ValueError("bad value")) == ErrorCode.INVALID_INPUT

    def test_type_error(self):
        """TypeError maps to INVALID_INPUT."""
        assert _classify_error(TypeError("bad type")) == ErrorCode.INVALID_INPUT

    def test_pipeline_core_error(self):
        """PipelineCoreError maps to PIPELINE_ERROR."""
        assert _classify_error(PipelineCoreError("pipeline fail")) == ErrorCode.PIPELINE_ERROR

    def test_unknown_error(self):
        """Unrecognized exceptions map to UNKNOWN."""
        assert _classify_error(RuntimeError("unknown")) == ErrorCode.UNKNOWN
        assert _classify_error(OSError("os error")) == ErrorCode.UNKNOWN

    def test_accepts_base_exception(self):
        """_classify_error accepts BaseException (not just Exception)."""
        assert _classify_error(KeyboardInterrupt()) == ErrorCode.UNKNOWN

    def test_llm_subclass_maps_to_provider(self):
        """Subclasses of LLMError also map to PROVIDER_ERROR."""

        class CustomLLMError(LLMError):
            """Custom LLM error."""

        assert _classify_error(CustomLLMError("fail")) == ErrorCode.PROVIDER_ERROR


def _make_pubsub_publisher() -> tuple[PubSubPublisher, MagicMock, AsyncMock]:
    """Create a PubSubPublisher with mocked pubsub_v1 and result_store."""
    mock_result_store = AsyncMock()

    mock_client = MagicMock()
    mock_client.topic_path.return_value = "projects/test/topics/events"

    pub = PubSubPublisher.__new__(PubSubPublisher)
    pub._client = mock_client
    pub._topic_path = "projects/test/topics/events"
    pub._service_type = "research"
    pub._result_store = mock_result_store
    pub._sequencer = TimestampSequencer()

    return pub, mock_client, mock_result_store


class TestPubSubPublisher:
    """Test PubSubPublisher CloudEvents envelope and publish behavior."""

    def test_build_envelope_structure(self):
        """_build_envelope produces a valid CloudEvents 1.0 envelope."""
        pub, _, _ = _make_pubsub_publisher()
        data_bytes = pub._build_envelope("task.started", "run-1", {"flow_run_id": "fr-1"})
        envelope = json.loads(data_bytes)

        assert envelope["specversion"] == CLOUDEVENTS_SPEC_VERSION
        assert envelope["type"] == "task.started"
        assert envelope["source"] == "ai-research-worker"
        assert envelope["subject"] == "run-1"
        assert envelope["datacontenttype"] == "application/json"
        assert "id" in envelope
        assert "time" in envelope

        data = envelope["data"]
        assert data["run_id"] == "run-1"
        assert "seq" in data
        assert data["flow_run_id"] == "fr-1"

    def test_build_envelope_seq_is_monotonic(self):
        """Sequential envelopes have strictly increasing seq values."""
        pub, _, _ = _make_pubsub_publisher()
        env1 = json.loads(pub._build_envelope("task.started", "run-1", {}))
        env2 = json.loads(pub._build_envelope("task.progress", "run-1", {}))
        assert env2["data"]["seq"] > env1["data"]["seq"]

    async def test_publish_started_is_critical(self):
        """publish_started uses critical publish path."""
        pub, mock_client, _ = _make_pubsub_publisher()
        event = StartedEvent(run_id="run-1", flow_run_id="fr-1", run_scope="run-1:abc")

        mock_future = asyncio.Future()
        mock_future.set_result("msg-id")
        mock_client.publish.return_value = mock_future

        await pub.publish_started(event)
        mock_client.publish.assert_called_once()
        call_kwargs = mock_client.publish.call_args
        assert call_kwargs[1]["event_type"] == "task.started"

    async def test_publish_progress_is_noncritical(self):
        """publish_progress uses non-critical publish path."""
        pub, mock_client, _ = _make_pubsub_publisher()
        event = ProgressEvent(
            run_id="run-1",
            flow_run_id="fr-1",
            flow_name="extract",
            step=1,
            total_steps=3,
            progress=0.33,
            step_progress=0.5,
            status=FlowStatus.STARTED,
            message="test",
        )

        mock_future = asyncio.Future()
        mock_future.set_result("msg-id")
        mock_client.publish.return_value = mock_future

        await pub.publish_progress(event)
        mock_client.publish.assert_called_once()

    async def test_publish_heartbeat(self):
        """publish_heartbeat publishes task.heartbeat event."""
        pub, mock_client, _ = _make_pubsub_publisher()

        mock_future = asyncio.Future()
        mock_future.set_result("msg-id")
        mock_client.publish.return_value = mock_future

        await pub.publish_heartbeat("run-1")
        mock_client.publish.assert_called_once()
        call_kwargs = mock_client.publish.call_args
        assert call_kwargs[1]["event_type"] == "task.heartbeat"

    async def test_heartbeat_contains_timestamp(self):
        """publish_heartbeat includes a timestamp field in the data payload."""
        pub, mock_client, _ = _make_pubsub_publisher()

        mock_future = asyncio.Future()
        mock_future.set_result("msg-id")
        mock_client.publish.return_value = mock_future

        await pub.publish_heartbeat("run-1")
        published_data = mock_client.publish.call_args[0][1]
        envelope = json.loads(published_data)
        assert "timestamp" in envelope["data"]
        # Verify it's an ISO format string
        assert "T" in envelope["data"]["timestamp"]

    async def test_publish_completed_writes_before_publish(self):
        """publish_completed writes to result_store BEFORE publishing to Pub/Sub."""
        pub, mock_client, mock_result_store = _make_pubsub_publisher()

        call_order = []
        mock_result_store.write_result.side_effect = lambda *a, **kw: call_order.append("write")

        mock_future = asyncio.Future()
        mock_future.set_result("msg-id")
        mock_client.publish.side_effect = lambda *a, **kw: (call_order.append("publish"), mock_future)[1]

        event = CompletedEvent(
            run_id="run-1",
            flow_run_id="fr-1",
            result={"success": True},
            chain_context={"version": 1},
            actual_cost=0.0,
        )
        await pub.publish_completed(event)

        assert call_order == ["write", "publish"]
        mock_result_store.write_result.assert_called_once()

    async def test_publish_completed_continues_when_write_result_fails(self):
        """publish_completed still publishes to Pub/Sub even when write_result raises."""
        pub, mock_client, mock_result_store = _make_pubsub_publisher()

        mock_result_store.write_result.side_effect = RuntimeError("ClickHouse down")

        mock_future = asyncio.Future()
        mock_future.set_result("msg-id")
        mock_client.publish.return_value = mock_future

        event = CompletedEvent(
            run_id="run-1",
            flow_run_id="fr-1",
            result={"success": True},
            chain_context={"version": 1},
            actual_cost=0.0,
        )
        await pub.publish_completed(event)

        # write_result was called and failed
        mock_result_store.write_result.assert_called_once()
        # But publish still happened
        mock_client.publish.assert_called_once()
        call_kwargs = mock_client.publish.call_args
        assert call_kwargs[1]["event_type"] == "task.completed"

    async def test_publish_completed_size_guard(self):
        """publish_completed raises ResultTooLargeError for oversized messages."""
        pub, mock_client, mock_result_store = _make_pubsub_publisher()

        huge_result = {"data": "x" * (MAX_PUBSUB_MESSAGE_BYTES + 1)}
        event = CompletedEvent(
            run_id="run-1",
            flow_run_id="fr-1",
            result=huge_result,
            chain_context={"version": 1},
            actual_cost=0.0,
        )

        with pytest.raises(ResultTooLargeError):
            await pub.publish_completed(event)

        # Result store should NOT be written for oversized messages
        mock_result_store.write_result.assert_not_called()

    async def test_publish_failed_is_critical(self):
        """publish_failed uses critical publish path."""
        pub, mock_client, _ = _make_pubsub_publisher()
        event = FailedEvent(
            run_id="run-1",
            flow_run_id="fr-1",
            error_code=ErrorCode.PIPELINE_ERROR,
            error_message="something broke",
        )

        mock_future = asyncio.Future()
        mock_future.set_result("msg-id")
        mock_client.publish.return_value = mock_future

        await pub.publish_failed(event)
        mock_client.publish.assert_called_once()
        call_kwargs = mock_client.publish.call_args
        assert call_kwargs[1]["event_type"] == "task.failed"

    def test_make_attributes(self):
        """_make_attributes returns correct Pub/Sub message attributes."""
        pub, _, _ = _make_pubsub_publisher()
        attrs = pub._make_attributes("task.started", "run-1")
        assert attrs == {
            "service_type": "research",
            "event_type": "task.started",
            "run_id": "run-1",
        }

    async def test_noncritical_failure_logs_warning(self):
        """Non-critical publish failure is logged but does not raise."""
        pub, mock_client, _ = _make_pubsub_publisher()

        mock_future = asyncio.Future()
        mock_future.set_exception(RuntimeError("network error"))
        mock_client.publish.return_value = mock_future

        # Should not raise
        await pub.publish_heartbeat("run-1")

    async def test_critical_retries_on_failure(self):
        """Critical publish retries with exponential backoff."""
        pub, mock_client, _ = _make_pubsub_publisher()

        # First call fails, second succeeds
        fail_future = asyncio.Future()
        fail_future.set_exception(RuntimeError("temporary failure"))
        success_future = asyncio.Future()
        success_future.set_result("msg-id")
        mock_client.publish.side_effect = [fail_future, success_future]

        event = StartedEvent(run_id="run-1", flow_run_id="fr-1", run_scope="run-1:abc")

        with patch("ai_pipeline_core.deployment._pubsub.asyncio.sleep", new_callable=AsyncMock):
            await pub.publish_started(event)

        assert mock_client.publish.call_count == 2

    async def test_critical_exhausts_retries(self):
        """Critical publish raises after exhausting all retries."""
        pub, mock_client, _ = _make_pubsub_publisher()

        # All calls fail
        def make_fail_future(*args, **kwargs):
            f = asyncio.Future()
            f.set_exception(RuntimeError("persistent failure"))
            return f

        mock_client.publish.side_effect = make_fail_future

        event = StartedEvent(run_id="run-1", flow_run_id="fr-1", run_scope="run-1:abc")

        with patch("ai_pipeline_core.deployment._pubsub.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match=f"failed after {CRITICAL_MAX_RETRIES} attempts"):
                await pub.publish_started(event)

        assert mock_client.publish.call_count == CRITICAL_MAX_RETRIES
