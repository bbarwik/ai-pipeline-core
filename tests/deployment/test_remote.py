"""Tests for remote deployment utilities."""

from functools import wraps
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from ai_pipeline_core import (
    DeploymentResult,
    Document,
    FlowOptions,
    PipelineDeployment,
    pipeline_flow,
    trace,
)

# --- Module-level test infrastructure ---


class SampleInputDoc(Document):
    """Input document for testing."""


class SampleOutputDoc(Document):
    """Output document for testing."""


@pipeline_flow()
async def sample_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
    """Sample flow for testing."""
    return []


class SampleResult(DeploymentResult):
    """Result type for testing."""

    report: str = ""


class SamplePipeline(PipelineDeployment[FlowOptions, SampleResult]):
    """Pipeline for testing."""

    flows = [sample_flow]  # type: ignore[reportAssignmentType]

    @staticmethod
    def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> SampleResult:
        """Build result from pipeline output."""
        return SampleResult(success=True, report="done")


# --- run_remote_deployment tests ---


class TestRunRemoteDeployment:
    """Test run_remote_deployment function error handling."""

    async def test_not_found_no_api_url(self):
        """Test error when deployment not found and PREFECT_API_URL is not set."""
        from prefect.exceptions import ObjectNotFound

        from ai_pipeline_core.deployment.remote import run_remote_deployment

        with patch("ai_pipeline_core.deployment.remote.get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.read_deployment_by_name = AsyncMock(side_effect=ObjectNotFound(http_exc=Exception("Not found")))
            mock_get_client.return_value = mock_client

            with patch("ai_pipeline_core.deployment.remote.settings") as mock_settings:
                mock_settings.prefect_api_url = None

                with pytest.raises(ValueError, match="not set"):
                    await run_remote_deployment("test-deployment", {"param": "value"})

    async def test_not_found_anywhere(self):
        """Test error when deployment not found on local or remote Prefect."""
        from prefect.exceptions import ObjectNotFound

        from ai_pipeline_core.deployment.remote import run_remote_deployment

        with patch("ai_pipeline_core.deployment.remote.get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.read_deployment_by_name = AsyncMock(side_effect=ObjectNotFound(http_exc=Exception("Not found")))
            mock_get_client.return_value = mock_client

            with patch("ai_pipeline_core.deployment.remote.PrefectClient") as mock_pc:
                mock_remote = AsyncMock()
                mock_remote.__aenter__.return_value = mock_remote
                mock_remote.__aexit__.return_value = None
                mock_remote.read_deployment_by_name = AsyncMock(side_effect=ObjectNotFound(http_exc=Exception("Not found remotely")))
                mock_pc.return_value = mock_remote

                with patch("ai_pipeline_core.deployment.remote.settings") as mock_settings:
                    mock_settings.prefect_api_url = "http://api.example.com"
                    mock_settings.prefect_api_key = "key"
                    mock_settings.prefect_api_auth_string = "auth"

                    with pytest.raises(ValueError, match="not found"):
                        await run_remote_deployment("test-deployment", {"param": "value"})


# --- Utility function tests ---


class TestIsAlreadyTraced:
    """Test _is_already_traced utility function."""

    def test_false_for_untraced(self):
        """Test returns False for untraced function."""
        from ai_pipeline_core.deployment.remote import (
            _is_already_traced,  # type: ignore[reportPrivateUsage]
        )

        async def my_func() -> None:
            pass

        assert _is_already_traced(my_func) is False

    def test_true_for_traced(self):
        """Test returns True for traced function."""
        from ai_pipeline_core.deployment.remote import (
            _is_already_traced,  # type: ignore[reportPrivateUsage]
        )

        @trace(level="always")
        async def my_func() -> None:
            pass

        assert _is_already_traced(my_func) is True

    def test_detects_nested_trace(self):
        """Test detects trace with double decoration."""
        from ai_pipeline_core.deployment.remote import (
            _is_already_traced,  # type: ignore[reportPrivateUsage]
        )

        @trace(level="always")
        @trace(level="always")
        async def my_func() -> None:
            pass

        assert _is_already_traced(my_func) is True

    def test_deep_wrapped_chain(self):
        """Test detects trace through __wrapped__ chain."""
        from ai_pipeline_core.deployment.remote import (
            _is_already_traced,  # type: ignore[reportPrivateUsage]
        )

        @trace(level="always")
        async def base_func() -> None:
            pass

        @wraps(base_func)
        async def wrapper() -> None:
            pass

        wrapper.__wrapped__ = base_func  # type: ignore[attr-defined]

        assert _is_already_traced(wrapper) is True


# --- Polling tests ---


def _make_flow_run(
    labels: dict[str, Any] | None = None,
    *,
    is_final: bool = False,
    is_completed: bool = False,
    result: Any = None,
    error: Exception | None = None,
) -> MagicMock:
    """Create a mock FlowRun with controlled state and labels."""
    fr = MagicMock()
    fr.id = UUID(int=1)
    fr.labels = labels or {}
    fr.state = MagicMock()
    fr.state.is_final.return_value = is_final
    fr.state.is_completed.return_value = is_completed
    fr.state.result = AsyncMock(side_effect=error) if error else AsyncMock(return_value=result)
    return fr


def _progress_labels(progress: float, flow_name: str = "", message: str = "") -> dict[str, Any]:
    """Build progress.* labels matching what PipelineDeployment writes."""
    return {
        "progress.progress": progress,
        "progress.flow_name": flow_name,
        "progress.message": message,
    }


class TestPollRemoteFlowRun:
    """Test _poll_remote_flow_run with and without callback."""

    async def test_callback_receives_progress(self):
        """Callback is called with fractions from remote labels."""
        from ai_pipeline_core.deployment.remote import _poll_remote_flow_run

        client = AsyncMock()
        client.read_flow_run = AsyncMock(
            side_effect=[
                _make_flow_run(labels=_progress_labels(0.3, "plan", "Planning")),
                _make_flow_run(labels=_progress_labels(0.7, "execute", "Executing")),
                _make_flow_run(is_final=True, is_completed=True, result={"success": True}),
            ]
        )

        callback = AsyncMock()
        result = await _poll_remote_flow_run(client, UUID(int=1), "test-deploy", poll_interval=0, on_progress=callback)

        assert result == {"success": True}
        assert callback.call_count == 3
        fractions = [call.args[0] for call in callback.call_args_list]
        assert fractions == [0.3, 0.7, 1.0]

    async def test_no_completion_on_failure(self):
        """Callback does NOT receive 1.0 when run fails."""
        from ai_pipeline_core.deployment.remote import _poll_remote_flow_run

        client = AsyncMock()
        client.read_flow_run = AsyncMock(
            side_effect=[
                _make_flow_run(labels=_progress_labels(0.5, "step1", "Working")),
                _make_flow_run(is_final=True, is_completed=False, error=RuntimeError("Crashed")),
            ]
        )

        callback = AsyncMock()
        with pytest.raises(RuntimeError, match="Crashed"):
            await _poll_remote_flow_run(client, UUID(int=1), "test-deploy", poll_interval=0, on_progress=callback)

        fractions = [call.args[0] for call in callback.call_args_list]
        assert 0.5 in fractions
        assert 1.0 not in fractions

    async def test_progress_never_regresses(self):
        """Max guard prevents fraction from going backwards."""
        from ai_pipeline_core.deployment.remote import _poll_remote_flow_run

        client = AsyncMock()
        client.read_flow_run = AsyncMock(
            side_effect=[
                _make_flow_run(labels=_progress_labels(0.8, "s1", "high")),
                _make_flow_run(labels=_progress_labels(0.3, "s2", "lower")),
                _make_flow_run(is_final=True, is_completed=True, result="ok"),
            ]
        )

        callback = AsyncMock()
        await _poll_remote_flow_run(client, UUID(int=1), "test-deploy", poll_interval=0, on_progress=callback)

        fractions = [call.args[0] for call in callback.call_args_list]
        assert fractions == [0.8, 0.8, 1.0]

    async def test_no_callback_result_still_returned(self):
        """Without callback, result is returned and no progress is reported."""
        from ai_pipeline_core.deployment.remote import _poll_remote_flow_run

        client = AsyncMock()
        client.read_flow_run = AsyncMock(
            side_effect=[
                _make_flow_run(labels=_progress_labels(0.5, "step1", "Working")),
                _make_flow_run(is_final=True, is_completed=True, result=42),
            ]
        )

        result = await _poll_remote_flow_run(client, UUID(int=1), "test-deploy", poll_interval=0)

        assert result == 42
        assert client.read_flow_run.call_count == 2

    async def test_no_labels_sends_waiting(self):
        """No progress labels → callback receives 'Waiting to start' message."""
        from ai_pipeline_core.deployment.remote import _poll_remote_flow_run

        client = AsyncMock()
        client.read_flow_run = AsyncMock(
            side_effect=[
                _make_flow_run(labels={}),
                _make_flow_run(is_final=True, is_completed=True, result="done"),
            ]
        )

        callback = AsyncMock()
        await _poll_remote_flow_run(client, UUID(int=1), "test-deploy", poll_interval=0, on_progress=callback)

        # First call: waiting (0.0), second call: completion (1.0)
        assert callback.call_count == 2
        assert callback.call_args_list[0].args[0] == 0.0
        assert "Waiting to start" in callback.call_args_list[0].args[1]
        assert callback.call_args_list[1].args[0] == 1.0

    async def test_api_error_retries(self):
        """Prefect API error on poll → logged, continues polling, returns result."""
        from ai_pipeline_core.deployment.remote import _poll_remote_flow_run

        client = AsyncMock()
        client.read_flow_run = AsyncMock(
            side_effect=[
                _make_flow_run(labels=_progress_labels(0.3, "s1", "Starting")),
                ConnectionError("API unavailable"),
                _make_flow_run(is_final=True, is_completed=True, result="recovered"),
            ]
        )

        callback = AsyncMock()
        result = await _poll_remote_flow_run(client, UUID(int=1), "test-deploy", poll_interval=0, on_progress=callback)

        assert result == "recovered"
        assert client.read_flow_run.call_count == 3
        fractions = [call.args[0] for call in callback.call_args_list]
        assert fractions == [0.3, 1.0]

    async def test_display_includes_flow_name(self):
        """Progress message includes flow_name from labels when available."""
        from ai_pipeline_core.deployment.remote import _poll_remote_flow_run

        client = AsyncMock()
        client.read_flow_run = AsyncMock(
            side_effect=[
                _make_flow_run(labels=_progress_labels(0.5, "research", "Analyzing sources")),
                _make_flow_run(is_final=True, is_completed=True, result="ok"),
            ]
        )

        callback = AsyncMock()
        await _poll_remote_flow_run(client, UUID(int=1), "test-deploy", poll_interval=0, on_progress=callback)

        display = callback.call_args_list[0].args[1]
        assert "[test-deploy]" in display
        assert "research" in display
        assert "Analyzing sources" in display
