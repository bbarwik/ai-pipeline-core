"""Tests for deploy.py to verify it uses settings correctly."""

# pyright: reportPrivateUsage=false, reportUnusedClass=false

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pydantic import Field
from prefect.deployments.runner import RunnerDeployment

from ai_pipeline_core import DeploymentResult, Document, FlowOptions, PipelineDeployment, pipeline_flow
from ai_pipeline_core.deployment.deploy import Deployer


class TestDeployer:
    """Test the Deployer class uses settings instead of environment variables."""

    @patch("ai_pipeline_core.deployment.deploy.settings")
    def test_init_validates_prefect_api_url(self, mock_settings):
        """Test that Deployer validates PREFECT_API_URL from settings."""
        # Test with empty API URL - should fail
        mock_settings.prefect_api_url = ""
        mock_settings.prefect_gcs_bucket = "test-bucket"
        mock_settings.prefect_work_pool_name = "default"
        mock_settings.prefect_work_queue_name = "default"

        with patch("ai_pipeline_core.deployment.deploy.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("builtins.open", create=True), patch("ai_pipeline_core.deployment.deploy.tomllib.load") as mock_toml:
                mock_toml.return_value = {"project": {"name": "test-project", "version": "1.0.0"}}

                with pytest.raises(SystemExit) as exc_info:
                    Deployer()
                assert exc_info.value.code == 1

    @patch("ai_pipeline_core.deployment.deploy.settings")
    def test_init_validates_prefect_gcs_bucket(self, mock_settings):
        """Test that Deployer validates PREFECT_GCS_BUCKET from settings."""
        # Test with empty bucket - should fail
        mock_settings.prefect_api_url = "http://test.api"
        mock_settings.prefect_gcs_bucket = ""
        mock_settings.prefect_work_pool_name = "default"
        mock_settings.prefect_work_queue_name = "default"

        with patch("ai_pipeline_core.deployment.deploy.Path") as mock_path:
            mock_path.return_value.exists.return_value = True

            with pytest.raises(SystemExit) as exc_info:
                Deployer()
            assert exc_info.value.code == 1

    @patch("ai_pipeline_core.deployment.deploy.settings")
    def test_init_loads_config_from_settings(self, mock_settings):
        """Test that Deployer loads configuration from settings."""
        # Set up valid settings
        mock_settings.prefect_api_url = "http://test.api"
        mock_settings.prefect_gcs_bucket = "test-bucket"
        mock_settings.prefect_work_pool_name = "test-pool"
        mock_settings.prefect_work_queue_name = "test-queue"

        with patch("ai_pipeline_core.deployment.deploy.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("builtins.open", create=True), patch("ai_pipeline_core.deployment.deploy.tomllib.load") as mock_toml:
                mock_toml.return_value = {"project": {"name": "test-project", "version": "1.0.0"}}

                deployer = Deployer()

                # Verify config was loaded from settings
                assert deployer.config["bucket"] == "test-bucket"
                assert deployer.config["work_pool"] == "test-pool"
                assert deployer.config["work_queue"] == "test-queue"
                assert deployer.api_url == "http://test.api"

    @patch("ai_pipeline_core.deployment.deploy.settings")
    def test_no_os_environ_usage(self, mock_settings):
        """Test that Deployer does not use os.environ directly."""
        # Set up valid settings
        mock_settings.prefect_api_url = "http://test.api"
        mock_settings.prefect_gcs_bucket = "test-bucket"
        mock_settings.prefect_work_pool_name = "default"
        mock_settings.prefect_work_queue_name = "default"

        with patch("ai_pipeline_core.deployment.deploy.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("builtins.open", create=True), patch("ai_pipeline_core.deployment.deploy.tomllib.load") as mock_toml:
                mock_toml.return_value = {"project": {"name": "test-project", "version": "1.0.0"}}

                # Patch os.environ to track any access
                with patch.dict("os.environ", {}, clear=True) as mock_environ:
                    # Create a Mock that will fail if accessed
                    mock_environ.__setitem__ = Mock(side_effect=AssertionError("os.environ should not be modified"))

                    # This should succeed without modifying os.environ
                    deployer = Deployer()
                    assert deployer.api_url == "http://test.api"

    @patch("ai_pipeline_core.deployment.deploy.settings")
    def test_deploy_uses_settings_for_client(self, mock_settings):
        """Test that deployment uses settings for Prefect client configuration."""
        # Set up valid settings
        mock_settings.prefect_api_url = "http://test.api"
        mock_settings.prefect_gcs_bucket = "test-bucket"
        mock_settings.prefect_work_pool_name = "test-pool"
        mock_settings.prefect_work_queue_name = "test-queue"
        mock_settings.prefect_api_key = "test-key"

        with patch("ai_pipeline_core.deployment.deploy.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("builtins.open", create=True), patch("ai_pipeline_core.deployment.deploy.tomllib.load") as mock_toml:
                mock_toml.return_value = {"project": {"name": "test-project", "version": "1.0.0"}}

                deployer = Deployer()

                # Verify that Prefect client will receive settings
                # The actual Prefect client reads from environment variables
                # which are set by pydantic_settings when settings loads .env
                assert deployer.api_url == "http://test.api"
                assert deployer.config["bucket"] == "test-bucket"

    @patch("ai_pipeline_core.deployment.deploy.settings")
    def test_config_normalization(self, mock_settings):
        """Test that project names are normalized correctly."""
        mock_settings.prefect_api_url = "http://test.api"
        mock_settings.prefect_gcs_bucket = "test-bucket"
        mock_settings.prefect_work_pool_name = "default"
        mock_settings.prefect_work_queue_name = "default"

        with patch("ai_pipeline_core.deployment.deploy.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("builtins.open", create=True), patch("ai_pipeline_core.deployment.deploy.tomllib.load") as mock_toml:
                # Test with hyphenated name
                mock_toml.return_value = {"project": {"name": "my-test-project", "version": "1.0.0"}}

                deployer = Deployer()

                # Verify normalization
                assert deployer.config["name"] == "my-test-project"
                assert deployer.config["package"] == "my_test_project"  # Hyphens to underscores
                assert deployer.config["folder"] == "flows/my-test-project"


# --- Test types for deploy schema tests ---


class _DeployInputDoc(Document):
    """Input document for deploy schema tests."""


class _DeployOutputDoc(Document):
    """Output document for deploy schema tests."""


class _DeployTestOptions(FlowOptions):
    """Options with concrete fields for deploy schema testing."""

    search_query: str = Field(default="", description="Search query")
    max_results: int = Field(default=10, description="Max results")


class _DeployTestResult(DeploymentResult):
    """Result for deploy schema testing."""

    report: str = ""


@pipeline_flow()
async def _deploy_test_flow(project_name: str, documents: list[_DeployInputDoc], flow_options: _DeployTestOptions) -> list[_DeployOutputDoc]:
    return [_DeployOutputDoc(name="out.txt", content=b"ok")]


class _DeploySchemaTestDeployment(PipelineDeployment[_DeployTestOptions, _DeployTestResult]):
    flows = [_deploy_test_flow]  # type: ignore[reportAssignmentType]

    @staticmethod
    def build_result(project_name: str, documents: list[Document], options: _DeployTestOptions) -> _DeployTestResult:
        return _DeployTestResult(success=True, report="done")


def _resolve_options_props(schema: Any) -> dict[str, Any]:
    """Resolve options properties from a ParameterSchema object."""
    options_entry = schema.properties.get("options") or schema.properties.get("flow_options") or {}
    if "$ref" in options_entry:
        ref_name = options_entry["$ref"].split("/")[-1]
        return schema.definitions.get(ref_name, {}).get("properties", {})
    if "allOf" in options_entry:
        for item in options_entry["allOf"]:
            if "$ref" in item:
                ref_name = item["$ref"].split("/")[-1]
                return schema.definitions.get(ref_name, {}).get("properties", {})
    return options_entry.get("properties", {})


async def _capture_deployed_runner() -> RunnerDeployment:
    """Run _deploy_via_api with mocked dependencies and capture the RunnerDeployment before apply()."""
    prefect_flow = _DeploySchemaTestDeployment().as_prefect_flow()
    captured: dict[str, Any] = {}

    async def capture_apply(deployment_self: Any) -> str:
        captured["deployment"] = deployment_self
        return "test-deployment-id"

    deployer = Deployer.__new__(Deployer)
    deployer.config = {
        "package": "test_pkg",
        "name": "test-pkg",
        "version": "1.0.0",
        "bucket": "test-bucket",
        "folder": "flows/test",
        "tarball": "test_pkg-1.0.0.tar.gz",
        "work_pool": "test-pool",
        "work_queue": "default",
    }
    deployer.api_url = "http://test.api"
    deployer._info = deployer._success = lambda *a, **k: None  # type: ignore[assignment]
    deployer._die = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))  # type: ignore[assignment]

    mock_client = AsyncMock()
    mock_client.read_work_pool.return_value = MagicMock(type="process")

    with (
        patch("ai_pipeline_core.deployment.deploy.load_flow_from_entrypoint", return_value=prefect_flow),
        patch("ai_pipeline_core.deployment.deploy.get_client") as mock_gc,
        patch.object(RunnerDeployment, "apply", capture_apply),
    ):
        mock_gc.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_gc.return_value.__aexit__ = AsyncMock(return_value=None)
        await deployer._deploy_via_api()

    return captured["deployment"]


class TestDeployParameterSchemaPopulation:
    """Test that _deploy_via_api populates RunnerDeployment's parameter_openapi_schema.

    Before the fix, RunnerDeployment was constructed without calling
    _set_defaults_from_flow(flow), leaving parameter_openapi_schema empty.
    """

    async def test_parameter_schema_is_populated(self):
        """_parameter_openapi_schema must have properties after deploy (was empty before fix)."""
        deployment = await _capture_deployed_runner()
        schema = deployment._parameter_openapi_schema
        assert schema.properties, "parameter schema properties must be populated"
        assert "options" in schema.properties

    async def test_options_schema_has_concrete_fields(self):
        """Options schema must include fields from the concrete FlowOptions subclass."""
        deployment = await _capture_deployed_runner()
        schema = deployment._parameter_openapi_schema
        options_props = _resolve_options_props(schema)
        assert "search_query" in options_props, f"search_query missing: {options_props}"
        assert "max_results" in options_props, f"max_results missing: {options_props}"

    async def test_result_schema_injected(self):
        """_ResultSchema must be present in definitions with concrete result fields."""
        deployment = await _capture_deployed_runner()
        schema = deployment._parameter_openapi_schema
        assert "_ResultSchema" in schema.definitions, f"_ResultSchema missing: {list(schema.definitions.keys())}"
        result_props = schema.definitions["_ResultSchema"].get("properties", {})
        assert "success" in result_props
        assert "error" in result_props
        assert "report" in result_props
