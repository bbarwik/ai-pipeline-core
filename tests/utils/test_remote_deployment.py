"""Tests for remote_deployment decorator and utilities."""

# remote_deployment's type signature uses Callable[P, TResult] which doesn't
# match async functions (Callable[P, Coroutine[..., TResult]]). The runtime
# behavior is correct; the type narrowing is a known limitation.
# pyright: reportArgumentType=false, reportGeneralTypeIssues=false

from functools import wraps
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from ai_pipeline_core import (
    DeploymentResult,
    DocumentList,
    FlowConfig,
    FlowDocument,
    FlowOptions,
    PipelineDeployment,
    pipeline_flow,
    trace,
)
from ai_pipeline_core.utils.remote_deployment import remote_deployment

# --- Module-level test infrastructure ---


class SampleInputDoc(FlowDocument):
    """Input document for testing."""


class SampleOutputDoc(FlowDocument):
    """Output document for testing."""


class SampleFlowConfig(FlowConfig):
    """Flow config for testing."""

    INPUT_DOCUMENT_TYPES = [SampleInputDoc]
    OUTPUT_DOCUMENT_TYPE = SampleOutputDoc


@pipeline_flow(config=SampleFlowConfig)
async def sample_flow(
    project_name: str, documents: DocumentList, flow_options: FlowOptions
) -> DocumentList:
    """Sample flow for testing."""
    return SampleFlowConfig.create_and_validate_output([])


class SampleResult(DeploymentResult):
    """Result type for testing."""

    report: str = ""


class SamplePipeline(PipelineDeployment[FlowOptions, SampleResult]):
    """Pipeline for testing remote_deployment decorator."""

    flows = [sample_flow]  # type: ignore[reportAssignmentType]

    @staticmethod
    def build_result(
        project_name: str, documents: DocumentList, options: FlowOptions
    ) -> SampleResult:
        """Build result from pipeline output."""
        return SampleResult(success=True, report="done")


# --- Decorator tests ---


class TestRemoteDeploymentDecorator:
    """Test the remote_deployment decorator functionality."""

    async def test_basic_remote_deployment(self):
        """Test basic decorator usage returns correct result type."""

        @remote_deployment(SamplePipeline)
        async def my_flow(project_name: str, options: FlowOptions) -> SampleResult:  # pyright: ignore[reportReturnType]
            ...

        with patch("ai_pipeline_core.utils.remote_deployment.run_remote_deployment") as mock_run:
            mock_run.return_value = SampleResult(success=True, report="test")
            result = await my_flow("test-project", FlowOptions())

            assert isinstance(result, SampleResult)
            assert result.success is True
            assert result.report == "test"
            mock_run.assert_called_once()

    async def test_deployment_name_default(self):
        """Test default deployment name uses deployment_class.name."""

        @remote_deployment(SamplePipeline)
        async def my_flow(project_name: str) -> SampleResult:  # pyright: ignore[reportReturnType]
            ...

        with patch("ai_pipeline_core.utils.remote_deployment.run_remote_deployment") as mock_run:
            mock_run.return_value = SampleResult(success=True)
            await my_flow("project")

            full_name = mock_run.call_args[0][0]
            assert full_name == "sample-pipeline/sample-pipeline"

    async def test_custom_deployment_name(self):
        """Test custom deployment_name overrides default."""

        @remote_deployment(SamplePipeline, deployment_name="custom-name")
        async def my_flow(project_name: str) -> SampleResult:  # pyright: ignore[reportReturnType]
            ...

        with patch("ai_pipeline_core.utils.remote_deployment.run_remote_deployment") as mock_run:
            mock_run.return_value = SampleResult(success=True)
            await my_flow("project")

            full_name = mock_run.call_args[0][0]
            assert full_name == "sample-pipeline/custom-name"

    async def test_parameters_passed_correctly(self):
        """Test all function parameters are forwarded to run_remote_deployment."""

        @remote_deployment(SamplePipeline)
        async def my_flow(
            project_name: str, options: FlowOptions, extra: str = "default"
        ) -> SampleResult:  # pyright: ignore[reportReturnType]
            ...

        with patch("ai_pipeline_core.utils.remote_deployment.run_remote_deployment") as mock_run:
            mock_run.return_value = SampleResult(success=True)
            await my_flow("project", FlowOptions(), extra="custom")

            params = mock_run.call_args[0][1]
            assert params["project_name"] == "project"
            assert params["extra"] == "custom"

    async def test_context_none_replaced_with_default(self):
        """Test that context=None is replaced with DeploymentContext()."""
        from ai_pipeline_core import DeploymentContext

        @remote_deployment(SamplePipeline)
        async def my_flow(project_name: str, context: Any = None) -> SampleResult:  # pyright: ignore[reportReturnType]
            ...

        with patch("ai_pipeline_core.utils.remote_deployment.run_remote_deployment") as mock_run:
            mock_run.return_value = SampleResult(success=True)
            await my_flow("project")

            params = mock_run.call_args[0][1]
            assert isinstance(params["context"], DeploymentContext)

    async def test_trace_cost_called(self):
        """Test trace_cost > 0 calls set_trace_cost."""
        with patch("ai_pipeline_core.utils.remote_deployment.set_trace_cost") as mock_cost:

            @remote_deployment(SamplePipeline, trace_cost=0.05)
            async def my_flow(
                project_name: str,
            ) -> SampleResult:  # pyright: ignore[reportReturnType]
                ...

            with patch(
                "ai_pipeline_core.utils.remote_deployment.run_remote_deployment"
            ) as mock_run:
                mock_run.return_value = SampleResult(success=True)
                await my_flow("project")
                mock_cost.assert_called_once_with(0.05)

    async def test_trace_cost_zero_not_called(self):
        """Test trace_cost=0 does not call set_trace_cost."""
        with patch("ai_pipeline_core.utils.remote_deployment.set_trace_cost") as mock_cost:

            @remote_deployment(SamplePipeline, trace_cost=0)
            async def my_flow(
                project_name: str,
            ) -> SampleResult:  # pyright: ignore[reportReturnType]
                ...

            with patch(
                "ai_pipeline_core.utils.remote_deployment.run_remote_deployment"
            ) as mock_run:
                mock_run.return_value = SampleResult(success=True)
                await my_flow("project")
                mock_cost.assert_not_called()

    async def test_dict_result_deserialized_to_result_type(self):
        """Test dict return value is deserialized via deployment result_type."""

        @remote_deployment(SamplePipeline)
        async def my_flow(project_name: str) -> SampleResult:  # pyright: ignore[reportReturnType]
            ...

        with patch("ai_pipeline_core.utils.remote_deployment.run_remote_deployment") as mock_run:
            mock_run.return_value = {"success": True, "report": "from dict"}
            result = await my_flow("project")

            assert isinstance(result, SampleResult)
            assert result.report == "from dict"

    async def test_invalid_result_raises_type_error(self):
        """Test non-dict, non-DeploymentResult return raises TypeError."""

        @remote_deployment(SamplePipeline)
        async def my_flow(project_name: str) -> SampleResult:  # pyright: ignore[reportReturnType]
            ...

        with patch("ai_pipeline_core.utils.remote_deployment.run_remote_deployment") as mock_run:
            mock_run.return_value = "invalid"

            with pytest.raises(TypeError, match="Expected DeploymentResult"):
                await my_flow("project")

    async def test_already_traced_raises_error(self):
        """Test that applying @trace before @remote_deployment raises TypeError."""
        with pytest.raises(TypeError, match="already has @trace"):

            @remote_deployment(SamplePipeline)
            @trace(level="always")
            async def my_flow(  # pyright: ignore[reportUnusedFunction]
                project_name: str,
            ) -> SampleResult:  # pyright: ignore[reportReturnType]
                ...

    async def test_deployment_not_found_propagates(self):
        """Test ValueError from run_remote_deployment propagates."""

        @remote_deployment(SamplePipeline)
        async def my_flow(project_name: str) -> SampleResult:  # pyright: ignore[reportReturnType]
            ...

        with patch("ai_pipeline_core.utils.remote_deployment.run_remote_deployment") as mock_run:
            mock_run.side_effect = ValueError("deployment not found")

            with pytest.raises(ValueError, match="deployment not found"):
                await my_flow("project")


# --- run_remote_deployment tests ---


class TestRunRemoteDeployment:
    """Test run_remote_deployment function error handling."""

    async def test_not_found_no_api_url(self):
        """Test error when deployment not found and PREFECT_API_URL is not set."""
        from prefect.exceptions import ObjectNotFound

        from ai_pipeline_core.utils.remote_deployment import run_remote_deployment

        with patch("ai_pipeline_core.utils.remote_deployment.get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.read_deployment_by_name = AsyncMock(
                side_effect=ObjectNotFound(http_exc=Exception("Not found"))
            )
            mock_get_client.return_value = mock_client

            with patch("ai_pipeline_core.utils.remote_deployment.settings") as mock_settings:
                mock_settings.prefect_api_url = None

                with pytest.raises(ValueError, match="not set"):
                    await run_remote_deployment("test-deployment", {"param": "value"})

    async def test_not_found_anywhere(self):
        """Test error when deployment not found on local or remote Prefect."""
        from prefect.exceptions import ObjectNotFound

        from ai_pipeline_core.utils.remote_deployment import run_remote_deployment

        with patch("ai_pipeline_core.utils.remote_deployment.get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.read_deployment_by_name = AsyncMock(
                side_effect=ObjectNotFound(http_exc=Exception("Not found"))
            )
            mock_get_client.return_value = mock_client

            with patch("ai_pipeline_core.utils.remote_deployment.PrefectClient") as mock_pc:
                mock_remote = AsyncMock()
                mock_remote.__aenter__.return_value = mock_remote
                mock_remote.__aexit__.return_value = None
                mock_remote.read_deployment_by_name = AsyncMock(
                    side_effect=ObjectNotFound(http_exc=Exception("Not found remotely"))
                )
                mock_pc.return_value = mock_remote

                with patch("ai_pipeline_core.utils.remote_deployment.settings") as mock_settings:
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
        from ai_pipeline_core.utils.remote_deployment import (
            _is_already_traced,  # type: ignore[reportPrivateUsage]
        )

        async def my_func() -> None:
            pass

        assert _is_already_traced(my_func) is False

    def test_true_for_traced(self):
        """Test returns True for traced function."""
        from ai_pipeline_core.utils.remote_deployment import (
            _is_already_traced,  # type: ignore[reportPrivateUsage]
        )

        @trace(level="always")
        async def my_func() -> None:
            pass

        assert _is_already_traced(my_func) is True

    def test_detects_nested_trace(self):
        """Test detects trace with double decoration."""
        from ai_pipeline_core.utils.remote_deployment import (
            _is_already_traced,  # type: ignore[reportPrivateUsage]
        )

        @trace(level="always")
        @trace(level="always")
        async def my_func() -> None:
            pass

        assert _is_already_traced(my_func) is True

    def test_deep_wrapped_chain(self):
        """Test detects trace through __wrapped__ chain."""
        from ai_pipeline_core.utils.remote_deployment import (
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
