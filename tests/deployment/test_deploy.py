"""Tests for deploy.py — bundled wheels deployment."""

# pyright: reportPrivateUsage=false, reportUnusedClass=false

import tarfile
from pathlib import Path
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
        "bundle": "test_pkg-1.0.0-bundle.tar.gz",
        "work_pool": "test-pool",
        "work_queue": "default",
    }
    deployer.api_url = "http://test.api"
    deployer._project_wheel_name = ""
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


def _make_deployer(**config_overrides: Any) -> Deployer:
    """Create a Deployer with mocked config (no filesystem/settings dependency)."""
    deployer = Deployer.__new__(Deployer)
    deployer.config = {
        "name": "test-project",
        "package": "test_project",
        "version": "1.0.0",
        "bucket": "test-bucket",
        "folder": "flows/test-project",
        "bundle": "test_project-1.0.0-bundle.tar.gz",
        "work_pool": "default",
        "work_queue": "default",
        **config_overrides,
    }
    deployer.api_url = "http://test.api"
    deployer._project_wheel_name = ""
    return deployer


class TestBundleNaming:
    """Bundle tarball uses '-bundle' suffix to distinguish from legacy sdist."""

    @patch("ai_pipeline_core.deployment.deploy.settings")
    def test_bundle_name_format(self, mock_settings: Any) -> None:
        mock_settings.prefect_api_url = "http://test.api"
        mock_settings.prefect_gcs_bucket = "bucket"
        mock_settings.prefect_work_pool_name = "default"
        mock_settings.prefect_work_queue_name = "default"

        with (
            patch("ai_pipeline_core.deployment.deploy.Path") as mock_path,
            patch("builtins.open", create=True),
            patch("ai_pipeline_core.deployment.deploy.tomllib.load") as mock_toml,
        ):
            mock_path.return_value.exists.return_value = True
            mock_toml.return_value = {"project": {"name": "my-app", "version": "2.1.0"}}
            deployer = Deployer()

        assert deployer.config["bundle"] == "my_app-2.1.0-bundle.tar.gz"
        assert "-bundle" in deployer.config["bundle"]


class TestInstallScript:
    """Install script must use offline mode (--no-index --find-links)."""

    def test_install_script_offline_mode(self) -> None:
        deployer = _make_deployer()
        script = deployer._build_install_script()

        assert "--no-index" in script, "Install must not contact PyPI"
        assert "--find-links wheels/" in script, "Install must use bundled wheels"
        assert "--system" in script, "Install must target system Python"
        assert "tar xzf" in script, "Install must extract the bundle first"
        # Must use newline, not &&, because Prefect's run_shell_script splits by lines
        assert "&&" not in script, "Must use newline separator (Prefect runs each line via shlex.split, not shell)"

    def test_install_script_references_bundle_name(self) -> None:
        deployer = _make_deployer(bundle="custom_pkg-3.0.0-bundle.tar.gz")
        script = deployer._build_install_script()

        assert "custom_pkg-3.0.0-bundle.tar.gz" in script

    def test_install_script_uses_explicit_wheel_name(self) -> None:
        deployer = _make_deployer()
        deployer._project_wheel_name = "test_project-1.0.0-py3-none-any.whl"
        script = deployer._build_install_script()

        # Explicit wheel name, not glob
        assert "test_project-1.0.0-py3-none-any.whl" in script

    def test_install_script_falls_back_to_glob_without_wheel_name(self) -> None:
        deployer = _make_deployer()
        deployer._project_wheel_name = ""
        script = deployer._build_install_script()

        assert "./*.whl" in script


class TestUvValidation:
    """Deployer must fail early if uv is not available."""

    def test_build_bundle_fails_without_uv(self) -> None:
        deployer = _make_deployer()

        with patch("ai_pipeline_core.deployment.deploy.shutil.which", return_value=None):
            with pytest.raises(SystemExit) as exc_info:
                deployer._build_bundle()
            assert exc_info.value.code == 1


class TestBuildBundle:
    """Test _build_bundle executes correct commands and creates correct tarball structure."""

    def test_build_bundle_commands(self) -> None:
        """Verify the correct sequence of shell commands is executed."""
        deployer = _make_deployer()
        commands_run: list[str] = []

        def mock_run(cmd: str, *, check: bool = True) -> str:
            commands_run.append(cmd)
            # Simulate side effects of each command:
            # After wheel build, create a .whl file; after download, create dep wheels
            if "build --wheel" in cmd:
                outdir = cmd.rsplit("--outdir ", maxsplit=1)[-1].strip().strip("'\"")
                (Path(outdir) / "test_project-1.0.0-py3-none-any.whl").write_bytes(b"wheel")
            elif "pip download" in cmd:
                dest = cmd.rsplit("-d ", maxsplit=1)[-1].split()[0].strip("'\"")
                (Path(dest) / "dep-1.0.0-py3-none-any.whl").write_bytes(b"dep")
            return ""

        deployer._run = mock_run  # type: ignore[assignment]
        deployer._info = deployer._success = lambda *a, **k: None  # type: ignore[assignment]

        with patch("ai_pipeline_core.deployment.deploy.shutil.which", return_value="/usr/bin/uv"):
            deployer._build_bundle()

        assert len(commands_run) == 3, f"Expected 3 commands, got {len(commands_run)}: {commands_run}"

        # Command 1: build wheel
        assert "python -m build --wheel" in commands_run[0]

        # Command 2: compile lock (must target worker platform)
        assert "uv pip compile" in commands_run[1]
        assert "pyproject.toml" in commands_run[1]
        assert "--python-platform" in commands_run[1], "Compile must target worker platform"
        assert "--python-version" in commands_run[1], "Compile must target worker Python version"

        # Command 3: download wheels via pip (not uv — uv lacks download subcommand)
        assert "pip download" in commands_run[2]
        assert "--platform" in commands_run[2]
        assert "--python-version" in commands_run[2]
        assert "--only-binary" in commands_run[2]
        assert "--no-deps" in commands_run[2]

    def test_build_bundle_tarball_structure(self) -> None:
        """Verify the bundle tarball has project wheel at root and deps in wheels/."""
        deployer = _make_deployer()

        def mock_run(cmd: str, *, check: bool = True) -> str:
            if "build --wheel" in cmd:
                outdir = cmd.rsplit("--outdir ", maxsplit=1)[-1].strip().strip("'\"")
                (Path(outdir) / "test_project-1.0.0-py3-none-any.whl").write_bytes(b"project wheel")
            elif "pip download" in cmd:
                dest = cmd.rsplit("-d ", maxsplit=1)[-1].split()[0].strip("'\"")
                (Path(dest) / "httpx-0.28.0-py3-none-any.whl").write_bytes(b"httpx")
                (Path(dest) / "pydantic-2.11.0-py3-none-any.whl").write_bytes(b"pydantic")
            return ""

        deployer._run = mock_run  # type: ignore[assignment]
        deployer._info = deployer._success = lambda *a, **k: None  # type: ignore[assignment]

        with patch("ai_pipeline_core.deployment.deploy.shutil.which", return_value="/usr/bin/uv"):
            bundle_path = deployer._build_bundle()

        # Verify tarball structure
        assert bundle_path.exists()
        with tarfile.open(bundle_path, "r:gz") as tar:
            names = tar.getnames()

        # Project wheel at root (no directory prefix)
        assert "test_project-1.0.0-py3-none-any.whl" in names

        # Dependency wheels in wheels/ subdirectory
        dep_wheels = [n for n in names if n.startswith("wheels/")]
        assert len(dep_wheels) == 2
        assert "wheels/httpx-0.28.0-py3-none-any.whl" in names
        assert "wheels/pydantic-2.11.0-py3-none-any.whl" in names

        # Clean up
        bundle_path.unlink(missing_ok=True)


class TestPullStepConfiguration:
    """Verify pull steps use the correct install script."""

    async def test_pull_step_uses_offline_install(self) -> None:
        """The pull step install script must use bundled offline install."""
        deployment = await _capture_deployed_runner()
        storage = deployment.storage

        # Extract the install script from pull steps
        pull_steps = storage.pull_steps  # type: ignore[union-attr]
        shell_step = pull_steps[1]
        script = shell_step["prefect.deployments.steps.run_shell_script"]["script"]

        assert "--no-index" in script
        assert "--find-links wheels/" in script
        assert "tar xzf" in script


class TestShellQuoting:
    """Install script must properly quote filenames for shell safety."""

    def test_bundle_name_is_quoted(self) -> None:
        deployer = _make_deployer(bundle="my app-1.0-bundle.tar.gz")
        script = deployer._build_install_script()

        # Bundle name with spaces must be quoted
        assert "my app-1.0-bundle.tar.gz" in script
        # shlex.quote wraps in single quotes
        assert "'" in script

    def test_wheel_name_is_quoted(self) -> None:
        deployer = _make_deployer()
        deployer._project_wheel_name = "my_pkg-1.0.0-py3-none-any.whl"
        script = deployer._build_install_script()

        assert "my_pkg-1.0.0-py3-none-any.whl" in script


class TestPlatformTargeting:
    """Dependency resolution must target the worker platform, not the deployer's."""

    def test_compile_targets_worker_platform(self) -> None:
        """uv pip compile must use --python-platform to target linux worker."""
        from ai_pipeline_core.deployment.deploy import PIP_TARGET_PLATFORMS, TARGET_PYTHON_VERSION, UV_TARGET_PLATFORM

        deployer = _make_deployer()
        commands_run: list[str] = []

        def mock_run(cmd: str, *, check: bool = True) -> str:
            commands_run.append(cmd)
            if "build --wheel" in cmd:
                outdir = cmd.rsplit("--outdir ", maxsplit=1)[-1].strip().strip("'\"")
                (Path(outdir) / "test_project-1.0.0-py3-none-any.whl").write_bytes(b"wheel")
            elif "pip download" in cmd:
                dest = cmd.rsplit("-d ", maxsplit=1)[-1].split()[0].strip("'\"")
                (Path(dest) / "dep-1.0.0-py3-none-any.whl").write_bytes(b"dep")
            return ""

        deployer._run = mock_run  # type: ignore[assignment]
        deployer._info = deployer._success = lambda *a, **k: None  # type: ignore[assignment]

        with patch("ai_pipeline_core.deployment.deploy.shutil.which", return_value="/usr/bin/uv"):
            deployer._build_bundle()

        compile_cmd = commands_run[1]
        assert UV_TARGET_PLATFORM in compile_cmd, f"Compile must target {UV_TARGET_PLATFORM}"
        assert TARGET_PYTHON_VERSION in compile_cmd

        download_cmd = commands_run[2]
        for plat in PIP_TARGET_PLATFORMS:
            assert plat in download_cmd, f"Download must include platform {plat}"
        assert TARGET_PYTHON_VERSION in download_cmd

    def test_build_bundle_stores_project_wheel_name(self) -> None:
        """_build_bundle must store the project wheel name for use in install script."""
        deployer = _make_deployer()

        def mock_run(cmd: str, *, check: bool = True) -> str:
            if "build --wheel" in cmd:
                outdir = cmd.rsplit("--outdir ", maxsplit=1)[-1].strip().strip("'\"")
                (Path(outdir) / "test_project-1.0.0-py3-none-any.whl").write_bytes(b"wheel")
            elif "pip download" in cmd:
                dest = cmd.rsplit("-d ", maxsplit=1)[-1].split()[0].strip("'\"")
                (Path(dest) / "dep-1.0.0-py3-none-any.whl").write_bytes(b"dep")
            return ""

        deployer._run = mock_run  # type: ignore[assignment]
        deployer._info = deployer._success = lambda *a, **k: None  # type: ignore[assignment]

        with patch("ai_pipeline_core.deployment.deploy.shutil.which", return_value="/usr/bin/uv"):
            deployer._build_bundle()

        assert deployer._project_wheel_name == "test_project-1.0.0-py3-none-any.whl"

        # Install script should use the explicit name
        script = deployer._build_install_script()
        assert "test_project-1.0.0-py3-none-any.whl" in script
        assert "./*.whl" not in script


class TestIsExcludedPackage:
    """Test lock file line filtering for vendor packages."""

    def test_pinned_package_match(self) -> None:
        from ai_pipeline_core.deployment.deploy import _is_excluded_package

        assert _is_excluded_package("lib-common==0.2.0", {"lib-common"})

    def test_pinned_package_no_match(self) -> None:
        from ai_pipeline_core.deployment.deploy import _is_excluded_package

        assert not _is_excluded_package("httpx==0.28.0", {"lib-common"})

    def test_comment_line_skipped(self) -> None:
        from ai_pipeline_core.deployment.deploy import _is_excluded_package

        assert not _is_excluded_package("# via lib-common", {"lib-common"})

    def test_blank_line_skipped(self) -> None:
        from ai_pipeline_core.deployment.deploy import _is_excluded_package

        assert not _is_excluded_package("", {"lib-common"})

    def test_flag_line_skipped(self) -> None:
        from ai_pipeline_core.deployment.deploy import _is_excluded_package

        assert not _is_excluded_package("-e .", {"lib-common"})

    def test_case_insensitive(self) -> None:
        from ai_pipeline_core.deployment.deploy import _is_excluded_package

        assert _is_excluded_package("Lib-Common==0.2.0", {"lib-common"})

    def test_underscore_normalized(self) -> None:
        from ai_pipeline_core.deployment.deploy import _is_excluded_package

        assert _is_excluded_package("lib_common==0.2.0", {"lib-common"})

    def test_extras_stripped(self) -> None:
        """Packages with extras like lib-common[grpc]==1.0 must still match."""
        from ai_pipeline_core.deployment.deploy import _is_excluded_package

        assert _is_excluded_package("lib-common[grpc]==0.2.0", {"lib-common"})

    def test_url_reference(self) -> None:
        """Direct URL references like pkg @ file:///... must match."""
        from ai_pipeline_core.deployment.deploy import _is_excluded_package

        assert _is_excluded_package("lib-common @ file:///tmp/lib_common-0.2.0.whl", {"lib-common"})


class TestFilterLockFile:
    """Test end-to-end lock file filtering."""

    def test_removes_vendor_keeps_public(self, tmp_path: Path) -> None:
        from ai_pipeline_core.deployment.deploy import _filter_lock_file

        lock = tmp_path / "requirements.lock"
        lock.write_text("# generated by uv\nlib-common==0.2.0\nlib-drive==0.2.0\nhttpx==0.28.0\npydantic==2.11.0\n")

        _filter_lock_file(lock, {"lib-common", "lib-drive"})

        lines = [line for line in lock.read_text().splitlines() if line.strip()]
        assert "lib-common==0.2.0" not in lines
        assert "lib-drive==0.2.0" not in lines
        assert "httpx==0.28.0" in lines
        assert "pydantic==2.11.0" in lines
        assert "# generated by uv" in lock.read_text()


class TestVendorPackages:
    """Test vendor package wheel building and bundle integration."""

    def test_build_bundle_with_vendor_includes_find_links(self, tmp_path: Path) -> None:
        """When vendor_packages is set, uv pip compile must receive --find-links."""
        vendor_dir = tmp_path / "libs" / "common"
        vendor_dir.mkdir(parents=True)
        deployer = _make_deployer()
        deployer.config["vendor_packages"] = [str(vendor_dir)]
        commands_run: list[str] = []

        def mock_run(cmd: str, *, check: bool = True) -> str:
            commands_run.append(cmd)
            if "build --wheel" in cmd:
                outdir = cmd.rsplit("--outdir ", maxsplit=1)[-1].split()[0].strip("'\"")
                Path(outdir).mkdir(parents=True, exist_ok=True)
                if "common" in cmd:
                    (Path(outdir) / "lib_common-0.2.0-py3-none-any.whl").write_bytes(b"vendor")
                else:
                    (Path(outdir) / "test_project-1.0.0-py3-none-any.whl").write_bytes(b"wheel")
            elif "pip download" in cmd:
                dest = cmd.rsplit("-d ", maxsplit=1)[-1].split()[0].strip("'\"")
                (Path(dest) / "httpx-0.28.0-py3-none-any.whl").write_bytes(b"dep")
            elif "uv pip compile" in cmd:
                out_flag = cmd.rsplit("-o ", maxsplit=1)[-1].split()[0].strip("'\"")
                Path(out_flag).write_text("lib-common==0.2.0\nhttpx==0.28.0\n")
            return ""

        deployer._run = mock_run  # type: ignore[assignment]
        deployer._info = deployer._success = lambda *a, **k: None  # type: ignore[assignment]

        with patch("ai_pipeline_core.deployment.deploy.shutil.which", return_value="/usr/bin/uv"):
            deployer._build_bundle()

        compile_cmds = [c for c in commands_run if "uv pip compile" in c]
        assert compile_cmds, "uv pip compile was not called"
        assert "--find-links" in compile_cmds[0]

    def test_build_bundle_without_vendor_no_find_links(self) -> None:
        """When no vendor_packages, uv pip compile must NOT have --find-links."""
        deployer = _make_deployer()
        commands_run: list[str] = []

        def mock_run(cmd: str, *, check: bool = True) -> str:
            commands_run.append(cmd)
            if "build --wheel" in cmd:
                outdir = cmd.rsplit("--outdir ", maxsplit=1)[-1].strip().strip("'\"")
                (Path(outdir) / "test_project-1.0.0-py3-none-any.whl").write_bytes(b"wheel")
            elif "pip download" in cmd:
                dest = cmd.rsplit("-d ", maxsplit=1)[-1].split()[0].strip("'\"")
                (Path(dest) / "dep-1.0.0-py3-none-any.whl").write_bytes(b"dep")
            return ""

        deployer._run = mock_run  # type: ignore[assignment]
        deployer._info = deployer._success = lambda *a, **k: None  # type: ignore[assignment]

        with patch("ai_pipeline_core.deployment.deploy.shutil.which", return_value="/usr/bin/uv"):
            deployer._build_bundle()

        compile_cmds = [c for c in commands_run if "uv pip compile" in c]
        assert "--find-links" not in compile_cmds[0]

    def test_vendor_wheel_names_normalized(self, tmp_path: Path) -> None:
        """Wheel filenames with underscores must normalize to hyphenated names for lock file filtering."""
        vendor_dir = tmp_path / "libs" / "common"
        vendor_dir.mkdir(parents=True)
        deployer = _make_deployer()
        deployer.config["vendor_packages"] = [str(vendor_dir)]

        def mock_run(cmd: str, *, check: bool = True) -> str:
            if "build --wheel" in cmd:
                outdir = cmd.rsplit("--outdir ", maxsplit=1)[-1].split()[0].strip("'\"")
                Path(outdir).mkdir(parents=True, exist_ok=True)
                (Path(outdir) / "lib_common-0.2.0-py3-none-any.whl").write_bytes(b"vendor")
            return ""

        deployer._run = mock_run  # type: ignore[assignment]
        deployer._info = deployer._success = lambda *a, **k: None  # type: ignore[assignment]

        wheels_dir = tmp_path / "wheels"
        wheels_dir.mkdir()
        vendor_names = deployer._build_vendor_wheels(wheels_dir)

        assert "lib-common" in vendor_names
