"""Tests for deployment base module classes and validation."""

# pyright: reportArgumentType=false, reportGeneralTypeIssues=false, reportPrivateUsage=false, reportUnusedClass=false, reportFunctionMemberAccess=false

from datetime import UTC
from typing import Any

import pytest
from pydantic import Field

from ai_pipeline_core import (
    DeploymentResult,
    Document,
    FlowOptions,
    PipelineDeployment,
    pipeline_flow,
)
from ai_pipeline_core.deployment import DeploymentContext
from ai_pipeline_core.deployment.contract import (
    CompletedRun,
    DeploymentResultData,
    FailedRun,
    PendingRun,
    ProgressRun,
)

# --- Module-level test infrastructure ---


class InputDoc(Document):
    """Input for testing."""


class OutputDoc(Document):
    """Output for testing."""


class MiddleDoc(Document):
    """Intermediate document for testing."""


@pipeline_flow()
async def valid_flow(project_name: str, documents: list[InputDoc], flow_options: FlowOptions) -> list[OutputDoc]:
    """Valid flow."""
    return [OutputDoc(name="output.txt", content=b"result")]


@pipeline_flow()
async def flow_a(project_name: str, documents: list[InputDoc], flow_options: FlowOptions) -> list[MiddleDoc]:
    """First flow in multi-step pipeline."""
    return [MiddleDoc(name="middle.txt", content=b"middle")]


@pipeline_flow()
async def flow_b(project_name: str, documents: list[MiddleDoc], flow_options: FlowOptions) -> list[OutputDoc]:
    """Second flow in multi-step pipeline."""
    return [OutputDoc(name="output.txt", content=b"final")]


class ValidResult(DeploymentResult):
    """Result for testing."""

    count: int = 0


# --- DeploymentContext tests ---


class TestDeploymentContext:
    """Test DeploymentContext model."""

    def test_default_creation(self):
        """Test default context has empty values."""
        ctx = DeploymentContext()
        assert ctx.progress_webhook_url == ""
        assert ctx.status_webhook_url == ""
        assert ctx.completion_webhook_url == ""
        assert ctx.input_documents_urls == ()
        assert ctx.output_documents_urls == {}

    def test_creation_with_urls(self):
        """Test context with webhook URLs."""
        ctx = DeploymentContext(
            progress_webhook_url="http://progress",
            status_webhook_url="http://status",
            completion_webhook_url="http://complete",
        )
        assert ctx.progress_webhook_url == "http://progress"
        assert ctx.status_webhook_url == "http://status"
        assert ctx.completion_webhook_url == "http://complete"

    def test_frozen(self):
        """Test context is immutable."""
        from pydantic import ValidationError

        ctx = DeploymentContext()
        with pytest.raises(ValidationError):
            ctx.progress_webhook_url = "http://new"  # type: ignore[misc]


# --- DeploymentResult tests ---


class TestDeploymentResult:
    """Test DeploymentResult model."""

    def test_success(self):
        """Test successful result."""
        result = DeploymentResult(success=True)
        assert result.success is True
        assert result.error is None

    def test_failure(self):
        """Test failed result with error."""
        result = DeploymentResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_subclass(self):
        """Test custom result subclass."""
        result = ValidResult(success=True, count=42)
        assert result.count == 42


# --- Contract models tests ---


class TestContractModels:
    """Test webhook contract Pydantic models."""

    def test_pending_run(self):
        """Test PendingRun creation."""
        from datetime import datetime
        from uuid import UUID

        run = PendingRun(
            flow_run_id=UUID(int=1),
            project_name="test",
            state="PENDING",
            timestamp=datetime.now(UTC),
        )
        assert run.type == "pending"

    def test_progress_run(self):
        """Test ProgressRun creation with all fields."""
        from datetime import datetime
        from uuid import UUID

        run = ProgressRun(
            flow_run_id=UUID(int=1),
            project_name="test",
            state="RUNNING",
            timestamp=datetime.now(UTC),
            step=2,
            total_steps=5,
            flow_name="analysis",
            status="started",
            progress=0.4,
            step_progress=0.0,
            message="Starting analysis",
        )
        assert run.type == "progress"
        assert run.progress == 0.4

    def test_completed_run(self):
        """Test CompletedRun creation."""
        from datetime import datetime
        from uuid import UUID

        run = CompletedRun(
            flow_run_id=UUID(int=1),
            project_name="test",
            state="COMPLETED",
            timestamp=datetime.now(UTC),
            result=DeploymentResultData(success=True),
        )
        assert run.type == "completed"
        assert run.result.success is True

    def test_failed_run(self):
        """Test FailedRun creation."""
        from datetime import datetime
        from uuid import UUID

        run = FailedRun(
            flow_run_id=UUID(int=1),
            project_name="test",
            state="FAILED",
            timestamp=datetime.now(UTC),
            error="Pipeline crashed",
        )
        assert run.type == "failed"
        assert run.error == "Pipeline crashed"

    def test_deployment_result_data(self):
        """Test DeploymentResultData."""
        data = DeploymentResultData(success=True, error=None)
        assert data.success is True
        dumped = data.model_dump()
        assert "success" in dumped

    def test_no_storage_uri_field(self):
        """Test that storage_uri has been removed from contract models."""
        from datetime import datetime
        from uuid import UUID

        _run = PendingRun(
            flow_run_id=UUID(int=1),
            project_name="test",
            state="PENDING",
            timestamp=datetime.now(UTC),
        )
        assert "storage_uri" not in PendingRun.model_fields


# --- PipelineDeployment validation tests ---


class TestPipelineDeploymentValidation:
    """Test PipelineDeployment.__init_subclass__ validation."""

    def test_valid_subclass(self):
        """Test valid deployment creation."""

        class MyDeployment(PipelineDeployment[FlowOptions, ValidResult]):
            """Valid deployment."""

            flows = [valid_flow]  # type: ignore[reportAssignmentType]

            @staticmethod
            def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> ValidResult:
                """Build."""
                return ValidResult(success=True)

        assert MyDeployment.name == "my-deployment"
        assert MyDeployment.options_type is FlowOptions
        assert MyDeployment.result_type is ValidResult

    def test_name_starts_with_test_raises(self):
        """Test that 'Test' prefix raises TypeError."""
        with pytest.raises(TypeError, match="cannot start with 'Test'"):

            class TestDeployment(PipelineDeployment[FlowOptions, ValidResult]):
                """Invalid name."""

                flows = [valid_flow]  # type: ignore[reportAssignmentType]

                @staticmethod
                def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> ValidResult:
                    """Build."""
                    return ValidResult(success=True)

    def test_empty_flows_raises(self):
        """Test that empty flows raises TypeError."""
        with pytest.raises(TypeError, match="cannot be empty"):

            class EmptyDeployment(PipelineDeployment[FlowOptions, ValidResult]):
                """Empty flows."""

                flows = []  # type: ignore[reportAssignmentType]

                @staticmethod
                def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> ValidResult:
                    """Build."""
                    return ValidResult(success=True)

    def test_missing_generic_params_raises(self):
        """Test that missing generic params raises TypeError."""
        with pytest.raises(TypeError, match="must specify Generic parameters"):

            class RawDeployment(PipelineDeployment):  # type: ignore[type-arg]
                """No generics."""

                flows = [valid_flow]  # type: ignore[reportAssignmentType]

                @staticmethod
                def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> ValidResult:
                    """Build."""
                    return ValidResult(success=True)


# --- Status webhook hook tests ---


class TestStatusWebhookHook:
    """Test _StatusWebhookHook dataclass and __call__."""

    async def test_sends_status_payload(self):
        """Test hook sends status webhook on state transition."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from ai_pipeline_core.deployment.base import _StatusWebhookHook

        hook = _StatusWebhookHook(
            webhook_url="http://example.com/status",
            flow_run_id="00000000-0000-0000-0000-000000000001",
            project_name="my-project",
            step=2,
            total_steps=5,
            flow_name="analysis",
        )

        mock_flow = MagicMock()
        mock_flow_run = MagicMock()
        mock_flow_run.id = "00000000-0000-0000-0000-000000000001"
        mock_state = MagicMock()
        mock_state.type.value = "RUNNING"
        mock_state.name = "Running"

        mock_response = AsyncMock()
        mock_response.raise_for_status = lambda: None

        with patch("ai_pipeline_core.deployment.base.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            await hook(mock_flow, mock_flow_run, mock_state)
            mock_client.post.assert_called_once()

    async def test_handles_webhook_failure(self):
        """Test hook handles httpx errors gracefully."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from ai_pipeline_core.deployment.base import _StatusWebhookHook

        hook = _StatusWebhookHook(
            webhook_url="http://example.com/status",
            flow_run_id="00000000-0000-0000-0000-000000000001",
            project_name="project",
            step=1,
            total_steps=1,
            flow_name="flow",
        )

        mock_flow = MagicMock()
        mock_flow_run = MagicMock()
        mock_flow_run.id = "id"
        mock_state = MagicMock()
        mock_state.type.value = "FAILED"
        mock_state.name = "Failed"

        with patch("ai_pipeline_core.deployment.base.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.side_effect = Exception("Network error")
            mock_client_cls.return_value = mock_client

            # Should not raise
            await hook(mock_flow, mock_flow_run, mock_state)


class TestBuildStatusHooks:
    """Test _build_status_hooks method."""

    def test_returns_hooks_dict(self):
        """Test that _build_status_hooks returns expected hook structure."""
        deployment = ValidDeployment()
        ctx = DeploymentContext(status_webhook_url="http://example.com/status")
        hooks = deployment._build_status_hooks(ctx, "00000000-0000-0000-0000-000000000001", "my-project", 1, 3, "analysis")
        assert "on_running" in hooks
        assert "on_completion" in hooks
        assert "on_failure" in hooks
        assert "on_crashed" in hooks
        assert "on_cancellation" in hooks
        assert len(hooks["on_running"]) == 1


class TestAbstractSubclass:
    """Test PipelineDeployment partial subclassing (no flows)."""

    def test_subclass_without_flows_skipped(self):
        """Test that subclass without flows attribute is silently skipped."""

        class PartialDeployment(PipelineDeployment[FlowOptions, ValidResult]):
            """Intermediate abstract class without flows."""

        # Should not raise - flows not defined, so validation is skipped
        assert not hasattr(PartialDeployment, "name")


# --- Webhook method tests ---


class ValidDeployment(PipelineDeployment[FlowOptions, ValidResult]):
    """Deployment for webhook method testing."""

    flows = [valid_flow]  # type: ignore[reportAssignmentType]

    @staticmethod
    def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> ValidResult:
        """Build result."""
        return ValidResult(success=True, count=len(documents))


class TestSendCompletion:
    """Test PipelineDeployment._send_completion method."""

    async def test_skips_when_no_webhook_url(self):
        """Test _send_completion is a no-op without completion_webhook_url."""
        from unittest.mock import AsyncMock, patch

        deployment = ValidDeployment()
        ctx = DeploymentContext()  # No webhook URLs

        with patch("ai_pipeline_core.deployment.base.send_webhook", new_callable=AsyncMock) as mock:
            await deployment._send_completion(ctx, "", "project", result=None, error="err")
            mock.assert_not_called()

    async def test_sends_completed_run_on_success(self):
        """Test sends CompletedRun when result is provided."""
        from unittest.mock import AsyncMock, patch

        deployment = ValidDeployment()
        ctx = DeploymentContext(completion_webhook_url="http://example.com/done")
        result = ValidResult(success=True, count=5)

        with patch("ai_pipeline_core.deployment.base.send_webhook", new_callable=AsyncMock) as mock:
            await deployment._send_completion(
                ctx,
                "00000000-0000-0000-0000-000000000001",
                "my-project",
                result=result,
                error=None,
            )
            mock.assert_called_once()
            payload = mock.call_args[0][1]
            assert payload.type == "completed"
            assert payload.state == "COMPLETED"

    async def test_sends_failed_run_on_error(self):
        """Test sends FailedRun when result is None."""
        from unittest.mock import AsyncMock, patch

        deployment = ValidDeployment()
        ctx = DeploymentContext(completion_webhook_url="http://example.com/done")

        with patch("ai_pipeline_core.deployment.base.send_webhook", new_callable=AsyncMock) as mock:
            await deployment._send_completion(ctx, "", "my-project", result=None, error="Pipeline crashed")
            mock.assert_called_once()
            payload = mock.call_args[0][1]
            assert payload.type == "failed"
            assert payload.error == "Pipeline crashed"

    async def test_handles_webhook_failure(self):
        """Test _send_completion handles send_webhook exceptions gracefully."""
        from unittest.mock import AsyncMock, patch

        deployment = ValidDeployment()
        ctx = DeploymentContext(completion_webhook_url="http://example.com/done")

        with patch(
            "ai_pipeline_core.deployment.base.send_webhook",
            new_callable=AsyncMock,
            side_effect=Exception("Network error"),
        ):
            # Should not raise
            await deployment._send_completion(ctx, "", "project", result=None, error="err")


class TestSendProgress:
    """Test PipelineDeployment._send_progress method."""

    async def test_sends_progress_webhook(self):
        """Test _send_progress sends webhook when URL provided."""
        from unittest.mock import AsyncMock, patch

        deployment = ValidDeployment()
        ctx = DeploymentContext(progress_webhook_url="http://example.com/progress")

        with (
            patch("ai_pipeline_core.deployment.base.send_webhook", new_callable=AsyncMock) as mock_wh,
            patch("ai_pipeline_core.deployment.base.get_client") as mock_gc,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_gc.return_value = mock_client

            await deployment._send_progress(
                ctx,
                "00000000-0000-0000-0000-000000000002",
                "my-project",
                step=2,
                total_steps=5,
                flow_name="analysis",
                status="started",
                step_progress=0.0,
                message="Starting",
            )
            mock_wh.assert_called_once()
            payload = mock_wh.call_args[0][1]
            assert payload.step == 2
            assert payload.flow_name == "analysis"

    async def test_skips_webhook_when_no_url(self):
        """Test _send_progress skips webhook without URL."""
        from unittest.mock import AsyncMock, patch

        deployment = ValidDeployment()
        ctx = DeploymentContext()  # No progress_webhook_url

        with (
            patch("ai_pipeline_core.deployment.base.send_webhook", new_callable=AsyncMock) as mock_wh,
            patch("ai_pipeline_core.deployment.base.get_client") as mock_gc,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_gc.return_value = mock_client

            await deployment._send_progress(
                ctx,
                "",
                "project",
                step=1,
                total_steps=1,
                flow_name="flow",
                status="started",
                step_progress=0.0,
                message="msg",
            )
            mock_wh.assert_not_called()


# --- DocumentStore integration tests ---


class TestAllDocumentTypes:
    """Test _all_document_types helper."""

    def test_collects_types_from_flows(self):
        """Test that all input/output types are collected and deduplicated."""

        class MultiFlowDeployment(PipelineDeployment[FlowOptions, ValidResult]):
            """Multi-flow deployment."""

            flows = [flow_a, flow_b]  # type: ignore[reportAssignmentType]

            @staticmethod
            def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> ValidResult:
                return ValidResult(success=True)

        deployment = MultiFlowDeployment()
        types = deployment._all_document_types()
        type_names = {t.__name__ for t in types}
        assert "InputDoc" in type_names
        assert "MiddleDoc" in type_names
        assert "OutputDoc" in type_names


class TestComputeRunScope:
    """Test _compute_run_scope function."""

    def test_different_options_produce_different_scope_with_documents(self):
        """Test that different options produce different run_scope when documents are provided."""
        from ai_pipeline_core.deployment.base import _compute_run_scope

        doc = InputDoc(name="input.txt", content=b"test")

        class CustomOptions(FlowOptions):
            flag: bool = False

        scope1 = _compute_run_scope("project", [doc], CustomOptions(flag=True))
        scope2 = _compute_run_scope("project", [doc], CustomOptions(flag=False))

        assert scope1 != scope2
        assert scope1.startswith("project:")
        assert scope2.startswith("project:")

    def test_different_options_produce_different_scope_without_documents(self):
        """Test that different options produce different run_scope even with empty documents."""
        from ai_pipeline_core.deployment.base import _compute_run_scope

        class CustomOptions(FlowOptions):
            flag: bool = False

        scope1 = _compute_run_scope("project", [], CustomOptions(flag=True))
        scope2 = _compute_run_scope("project", [], CustomOptions(flag=False))

        assert scope1 != scope2
        assert scope1.startswith("project:")
        assert scope2.startswith("project:")

    def test_same_inputs_produce_same_scope(self):
        """Test that identical inputs produce identical run_scope."""
        from ai_pipeline_core.deployment.base import _compute_run_scope

        doc = InputDoc(name="input.txt", content=b"test")

        scope1 = _compute_run_scope("project", [doc], FlowOptions())
        scope2 = _compute_run_scope("project", [doc], FlowOptions())

        assert scope1 == scope2

    def test_cli_fields_are_excluded(self):
        """Test that CLI-only fields do not affect run_scope."""
        from ai_pipeline_core.deployment.base import _CLI_FIELDS, _compute_run_scope

        class CliOptions(FlowOptions):
            working_directory: str = ""
            project_name: str | None = None
            start: int = 1
            end: int | None = None
            no_trace: bool = False
            actual_option: str = "value"

        # Verify the CLI fields are in _CLI_FIELDS
        assert "working_directory" in _CLI_FIELDS
        assert "start" in _CLI_FIELDS

        # Different CLI field values should produce the same scope
        opts1 = CliOptions(working_directory="/path1", start=1, actual_option="same")
        opts2 = CliOptions(working_directory="/path2", start=5, actual_option="same")

        scope1 = _compute_run_scope("project", [], opts1)
        scope2 = _compute_run_scope("project", [], opts2)

        assert scope1 == scope2

        # But different actual_option values should produce different scope
        opts3 = CliOptions(actual_option="different")
        scope3 = _compute_run_scope("project", [], opts3)

        assert scope1 != scope3


class TestReattachFlowMetadata:
    """Test _reattach_flow_metadata helper."""

    def test_reattaches_missing_attributes(self):
        """Test that missing attributes are copied from original."""
        from ai_pipeline_core.deployment.base import _reattach_flow_metadata

        class Original:
            input_document_types = [InputDoc]
            output_document_types = [OutputDoc]
            estimated_minutes = 5

        class Target:
            pass

        original = Original()
        target = Target()
        _reattach_flow_metadata(original, target)  # type: ignore[arg-type]
        assert target.input_document_types == [InputDoc]  # type: ignore[attr-defined]
        assert target.output_document_types == [OutputDoc]  # type: ignore[attr-defined]
        assert target.estimated_minutes == 5  # type: ignore[attr-defined]

    def test_does_not_overwrite_existing(self):
        """Test that existing attributes are not overwritten."""
        from ai_pipeline_core.deployment.base import _reattach_flow_metadata

        class Original:
            input_document_types = [InputDoc]
            output_document_types = [OutputDoc]
            estimated_minutes = 5

        class Target:
            input_document_types = [MiddleDoc]
            output_document_types = [MiddleDoc]
            estimated_minutes = 10

        original = Original()
        target = Target()
        _reattach_flow_metadata(original, target)  # type: ignore[arg-type]
        assert target.input_document_types == [MiddleDoc]
        assert target.estimated_minutes == 10


# --- as_prefect_flow parameter schema tests ---


class _SchemaTestOptions(FlowOptions):
    """Options with concrete fields for schema testing."""

    task_name: str = Field(default="", description="Name of the task")
    max_retries: int = Field(default=3, ge=1, le=10, description="Max retry count")
    threshold: float = Field(default=0.5, description="Score threshold")


class _SchemaTestResult(DeploymentResult):
    """Result for schema testing."""

    output: str = ""


@pipeline_flow()
async def _schema_test_flow(project_name: str, documents: list[InputDoc], flow_options: _SchemaTestOptions) -> list[OutputDoc]:
    return [OutputDoc(name="out.txt", content=b"ok")]


class _SchemaTestDeployment(PipelineDeployment[_SchemaTestOptions, _SchemaTestResult]):
    flows = [_schema_test_flow]  # type: ignore[reportAssignmentType]

    @staticmethod
    def build_result(project_name: str, documents: list[Document], options: _SchemaTestOptions) -> _SchemaTestResult:
        return _SchemaTestResult(success=True, output="done")


def _resolve_options_properties(schema: Any) -> dict[str, Any]:
    """Resolve the 'options' parameter properties from a ParameterSchema, following $ref if needed."""
    options_schema = schema.properties.get("options", {})
    if "$ref" in options_schema:
        ref_name = options_schema["$ref"].split("/")[-1]
        return schema.definitions.get(ref_name, {}).get("properties", {})
    if "allOf" in options_schema:
        for item in options_schema["allOf"]:
            if "$ref" in item:
                ref_name = item["$ref"].split("/")[-1]
                return schema.definitions.get(ref_name, {}).get("properties", {})
    return options_schema.get("properties", {})


class TestAsPrefectFlowParameterSchema:
    """Test that as_prefect_flow() exposes concrete options schema to Prefect."""

    def test_parameter_schema_contains_concrete_options_fields(self):
        """Prefect flow parameter schema must include fields from the concrete options type."""
        prefect_flow = _SchemaTestDeployment().as_prefect_flow()
        schema = prefect_flow.parameters

        options_props = _resolve_options_properties(schema)

        assert "task_name" in options_props, f"task_name missing from schema: {options_props}"
        assert "max_retries" in options_props, f"max_retries missing from schema: {options_props}"
        assert "threshold" in options_props, f"threshold missing from schema: {options_props}"

    def test_parameter_schema_field_types(self):
        """Schema field types must match the Pydantic model field types."""
        prefect_flow = _SchemaTestDeployment().as_prefect_flow()
        schema = prefect_flow.parameters
        options_props = _resolve_options_properties(schema)

        assert options_props["task_name"]["type"] == "string"
        assert options_props["max_retries"]["type"] == "integer"
        assert options_props["threshold"]["type"] == "number"

    def test_parameter_schema_field_defaults(self):
        """Schema field defaults must match the Pydantic model defaults."""
        prefect_flow = _SchemaTestDeployment().as_prefect_flow()
        schema = prefect_flow.parameters
        options_props = _resolve_options_properties(schema)

        assert options_props["task_name"].get("default") == ""
        assert options_props["max_retries"].get("default") == 3
        assert options_props["threshold"].get("default") == 0.5

    def test_base_flow_options_produces_no_custom_fields(self):
        """Base FlowOptions (no fields) should still appear in schema but with no custom properties."""
        prefect_flow = ValidDeployment().as_prefect_flow()
        schema = prefect_flow.parameters

        # options parameter must exist in the schema
        assert "options" in schema.properties
