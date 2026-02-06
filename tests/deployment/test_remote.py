"""Tests for remote_deployment decorator and utilities."""

from functools import wraps
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from ai_pipeline_core import (
    DeploymentContext,
    DeploymentResult,
    Document,
    FlowOptions,
    PipelineDeployment,
    pipeline_flow,
    trace,
)
from ai_pipeline_core.deployment.remote import remote_deployment

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
    """Pipeline for testing remote_deployment decorator."""

    flows = [sample_flow]  # type: ignore[reportAssignmentType]

    @staticmethod
    def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> SampleResult:
        """Build result from pipeline output."""
        return SampleResult(success=True, report="done")


# --- Decorator tests ---


class TestRemoteDeploymentDecorator:
    """Test the remote_deployment decorator functionality."""

    async def test_basic_remote_deployment(self):
        """Test basic decorator usage returns correct result type."""

        @remote_deployment(SamplePipeline)
        async def my_flow(
            project_name: str,
            documents: list[Document],
            options: FlowOptions,
            context: DeploymentContext | None = None,
        ) -> SampleResult: ...

        with patch("ai_pipeline_core.deployment.remote.run_remote_deployment") as mock_run:
            mock_run.return_value = SampleResult(success=True, report="test")
            result = await my_flow("test-project", [], FlowOptions())

            assert isinstance(result, SampleResult)
            assert result.success is True
            assert result.report == "test"
            mock_run.assert_called_once()

    async def test_deployment_name_default(self):
        """Test default deployment name converts hyphens to underscores (matching Deployer)."""

        @remote_deployment(SamplePipeline)
        async def my_flow(
            project_name: str,
            documents: list[Document],
            options: FlowOptions,
            context: DeploymentContext | None = None,
        ) -> SampleResult: ...

        with patch("ai_pipeline_core.deployment.remote.run_remote_deployment") as mock_run:
            mock_run.return_value = SampleResult(success=True)
            await my_flow("project", [], FlowOptions())

            full_name = mock_run.call_args[0][0]
            # Deployer registers as {flow_name}/{package_name} where package_name uses underscores.
            # flow_name is kebab-case (from class_name_to_deployment_name), deployment name is underscore.
            assert full_name == "sample-pipeline/sample_pipeline"

    async def test_custom_deployment_name(self):
        """Test custom deployment_name overrides default."""

        @remote_deployment(SamplePipeline, deployment_name="custom-name")
        async def my_flow(
            project_name: str,
            documents: list[Document],
            options: FlowOptions,
            context: DeploymentContext | None = None,
        ) -> SampleResult: ...

        with patch("ai_pipeline_core.deployment.remote.run_remote_deployment") as mock_run:
            mock_run.return_value = SampleResult(success=True)
            await my_flow("project", [], FlowOptions())

            full_name = mock_run.call_args[0][0]
            assert full_name == "sample-pipeline/custom-name"

    async def test_parameters_passed_correctly(self):
        """Test all four deployment parameters are forwarded to run_remote_deployment."""

        @remote_deployment(SamplePipeline)
        async def my_flow(
            project_name: str,
            documents: list[Document],
            options: FlowOptions,
            context: DeploymentContext | None = None,
        ) -> SampleResult: ...

        with patch("ai_pipeline_core.deployment.remote.run_remote_deployment") as mock_run:
            mock_run.return_value = SampleResult(success=True)
            test_options = FlowOptions()
            await my_flow("project", [], test_options)

            params = mock_run.call_args[0][1]
            assert params["project_name"] == "project"
            assert params["documents"] == []
            assert params["options"] is test_options
            assert isinstance(params["context"], DeploymentContext)

    async def test_context_none_replaced_with_default(self):
        """Test that context=None is replaced with DeploymentContext()."""

        @remote_deployment(SamplePipeline)
        async def my_flow(
            project_name: str,
            documents: list[Document],
            options: FlowOptions,
            context: DeploymentContext | None = None,
        ) -> SampleResult: ...

        with patch("ai_pipeline_core.deployment.remote.run_remote_deployment") as mock_run:
            mock_run.return_value = SampleResult(success=True)
            await my_flow("project", [], FlowOptions())

            params = mock_run.call_args[0][1]
            assert isinstance(params["context"], DeploymentContext)

    async def test_documents_forwarded_correctly(self):
        """Test documents parameter is forwarded to run_remote_deployment."""

        @remote_deployment(SamplePipeline)
        async def my_flow(
            project_name: str,
            documents: list[Document],
            options: FlowOptions,
            context: DeploymentContext | None = None,
        ) -> SampleResult: ...

        with patch("ai_pipeline_core.deployment.remote.run_remote_deployment") as mock_run:
            mock_run.return_value = SampleResult(success=True)
            test_docs = [SampleInputDoc.create(name="test.txt", content="test data")]
            await my_flow("project", test_docs, FlowOptions())

            params = mock_run.call_args[0][1]
            assert params["documents"] == [doc.serialize_model() for doc in test_docs]
            assert len(params["documents"]) == 1

    async def test_explicit_context_preserved(self):
        """Test explicit DeploymentContext is passed through, not replaced."""

        @remote_deployment(SamplePipeline)
        async def my_flow(
            project_name: str,
            documents: list[Document],
            options: FlowOptions,
            context: DeploymentContext | None = None,
        ) -> SampleResult: ...

        custom_context = DeploymentContext(progress_webhook_url="http://example.com")
        with patch("ai_pipeline_core.deployment.remote.run_remote_deployment") as mock_run:
            mock_run.return_value = SampleResult(success=True)
            await my_flow("project", [], FlowOptions(), context=custom_context)

            params = mock_run.call_args[0][1]
            assert params["context"] is custom_context
            assert params["context"].progress_webhook_url == "http://example.com"

    async def test_trace_cost_called(self):
        """Test trace_cost > 0 calls set_trace_cost."""
        with patch("ai_pipeline_core.deployment.remote.set_trace_cost") as mock_cost:

            @remote_deployment(SamplePipeline, trace_cost=0.05)
            async def my_flow(
                project_name: str,
                documents: list[Document],
                options: FlowOptions,
                context: DeploymentContext | None = None,
            ) -> SampleResult: ...

            with patch("ai_pipeline_core.deployment.remote.run_remote_deployment") as mock_run:
                mock_run.return_value = SampleResult(success=True)
                await my_flow("project", [], FlowOptions())
                mock_cost.assert_called_once_with(0.05)

    async def test_trace_cost_zero_not_called(self):
        """Test trace_cost=0 does not call set_trace_cost."""
        with patch("ai_pipeline_core.deployment.remote.set_trace_cost") as mock_cost:

            @remote_deployment(SamplePipeline, trace_cost=0)
            async def my_flow(
                project_name: str,
                documents: list[Document],
                options: FlowOptions,
                context: DeploymentContext | None = None,
            ) -> SampleResult: ...

            with patch("ai_pipeline_core.deployment.remote.run_remote_deployment") as mock_run:
                mock_run.return_value = SampleResult(success=True)
                await my_flow("project", [], FlowOptions())
                mock_cost.assert_not_called()

    async def test_dict_result_deserialized_to_result_type(self):
        """Test dict return value is deserialized via deployment result_type."""

        @remote_deployment(SamplePipeline)
        async def my_flow(
            project_name: str,
            documents: list[Document],
            options: FlowOptions,
            context: DeploymentContext | None = None,
        ) -> SampleResult: ...

        with patch("ai_pipeline_core.deployment.remote.run_remote_deployment") as mock_run:
            mock_run.return_value = {"success": True, "report": "from dict"}
            result = await my_flow("project", [], FlowOptions())

            assert isinstance(result, SampleResult)
            assert result.report == "from dict"

    async def test_invalid_result_raises_type_error(self):
        """Test non-dict, non-DeploymentResult return raises TypeError."""

        @remote_deployment(SamplePipeline)
        async def my_flow(
            project_name: str,
            documents: list[Document],
            options: FlowOptions,
            context: DeploymentContext | None = None,
        ) -> SampleResult: ...

        with patch("ai_pipeline_core.deployment.remote.run_remote_deployment") as mock_run:
            mock_run.return_value = "invalid"

            with pytest.raises(TypeError, match="Expected DeploymentResult"):
                await my_flow("project", [], FlowOptions())

    async def test_already_traced_raises_error(self):
        """Test that applying @trace before @remote_deployment raises TypeError."""
        with pytest.raises(TypeError, match="already has @trace"):

            @remote_deployment(SamplePipeline)
            @trace(level="always")
            async def my_flow(
                project_name: str,
                documents: list[Document],
                options: FlowOptions,
                context: DeploymentContext | None = None,
            ) -> SampleResult: ...

    async def test_default_name_matches_deployer_convention(self):
        """Test that default deployment path matches what Deployer registers in Prefect.

        Deployer registers: flow_name=kebab-case, deployment_name=underscore (Python package name).
        remote_deployment must produce the same '{flow_name}/{package_name}' path.
        """

        @remote_deployment(SamplePipeline)
        async def my_flow(
            project_name: str,
            documents: list[Document],
            options: FlowOptions,
            context: DeploymentContext | None = None,
        ) -> SampleResult: ...

        with patch("ai_pipeline_core.deployment.remote.run_remote_deployment") as mock_run:
            mock_run.return_value = SampleResult(success=True)
            await my_flow("project", [], FlowOptions())

            full_name = mock_run.call_args[0][0]
            flow_name, deployment_name = full_name.split("/")
            # flow_name is kebab-case (class_name_to_deployment_name)
            assert flow_name == SamplePipeline.name
            assert "-" not in deployment_name, "deployment name must use underscores, not hyphens"
            assert deployment_name == SamplePipeline.name.replace("-", "_")

    async def test_deployment_not_found_propagates(self):
        """Test ValueError from run_remote_deployment propagates."""

        @remote_deployment(SamplePipeline)
        async def my_flow(
            project_name: str,
            documents: list[Document],
            options: FlowOptions,
            context: DeploymentContext | None = None,
        ) -> SampleResult: ...

        with patch("ai_pipeline_core.deployment.remote.run_remote_deployment") as mock_run:
            mock_run.side_effect = ValueError("deployment not found")

            with pytest.raises(ValueError, match="deployment not found"):
                await my_flow("project", [], FlowOptions())

    async def test_on_progress_forwarded(self):
        """Test on_progress callback is forwarded to run_remote_deployment."""

        @remote_deployment(SamplePipeline)
        async def my_flow(
            project_name: str,
            documents: list[Document],
            options: FlowOptions,
            context: DeploymentContext | None = None,
        ) -> SampleResult: ...

        callback = AsyncMock()
        with patch("ai_pipeline_core.deployment.remote.run_remote_deployment") as mock_run:
            mock_run.return_value = SampleResult(success=True)
            await my_flow("project", [], FlowOptions(), on_progress=callback)

            assert mock_run.call_args.kwargs["on_progress"] is callback

    async def test_on_progress_none_by_default(self):
        """Test on_progress defaults to None when not provided."""

        @remote_deployment(SamplePipeline)
        async def my_flow(
            project_name: str,
            documents: list[Document],
            options: FlowOptions,
            context: DeploymentContext | None = None,
        ) -> SampleResult: ...

        with patch("ai_pipeline_core.deployment.remote.run_remote_deployment") as mock_run:
            mock_run.return_value = SampleResult(success=True)
            await my_flow("project", [], FlowOptions())

            assert mock_run.call_args.kwargs["on_progress"] is None


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
