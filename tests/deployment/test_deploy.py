"""Tests for deploy.py to verify it uses settings correctly."""

# pyright: reportPrivateUsage=false

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

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


class TestDeployerAgentConfig:
    """Tests for agent configuration loading from [tool.deploy.agents]."""

    def _make_deployer(self, pyproject_data: dict[str, Any]) -> Deployer:
        """Create a Deployer with mocked settings and pyproject data."""
        with patch("ai_pipeline_core.deployment.deploy.settings") as mock_settings:
            mock_settings.prefect_api_url = "http://test.api"
            mock_settings.prefect_gcs_bucket = "test-bucket"
            mock_settings.prefect_work_pool_name = "default"
            mock_settings.prefect_work_queue_name = "default"

            with patch("ai_pipeline_core.deployment.deploy.Path") as mock_path:
                mock_path.return_value.exists.return_value = True
                with patch("builtins.open", create=True), patch("ai_pipeline_core.deployment.deploy.tomllib.load", return_value=pyproject_data):
                    return Deployer()

    def test_loads_agent_config(self):
        """Should return agent config from [tool.deploy.agents]."""
        deployer = self._make_deployer({
            "project": {"name": "test", "version": "1.0.0"},
            "tool": {
                "deploy": {
                    "cli_agents_source": "vendor/cli-agents",
                    "agents": {
                        "initial_research": {
                            "path": "agents/initial_research",
                            "extra_vendor": ["crypto_mcp"],
                        }
                    },
                }
            },
        })

        config = deployer._load_agent_config()
        assert "initial_research" in config
        assert config["initial_research"]["path"] == "agents/initial_research"
        assert config["initial_research"]["extra_vendor"] == ["crypto_mcp"]

    def test_empty_when_no_agents_section(self):
        """Should return empty dict when no [tool.deploy.agents] section."""
        deployer = self._make_deployer({
            "project": {"name": "test", "version": "1.0.0"},
        })

        assert deployer._load_agent_config() == {}

    def test_cli_agents_source(self):
        """Should return cli_agents_source from [tool.deploy]."""
        deployer = self._make_deployer({
            "project": {"name": "test", "version": "1.0.0"},
            "tool": {"deploy": {"cli_agents_source": "vendor/cli-agents"}},
        })

        assert deployer._get_cli_agents_source() == "vendor/cli-agents"


class TestBuildAgentsEdgeCases:
    """Test _build_agents error paths and edge cases."""

    def test_returns_empty_when_no_agents(self):
        """Should return empty dict when no agents configured."""
        deployer = Deployer.__new__(Deployer)
        deployer._pyproject_data = {"project": {"name": "test"}}
        assert deployer._build_agents() == {}

    def test_dies_when_no_cli_agents_source(self):
        """Should die when agents configured but cli_agents_source missing."""
        deployer = Deployer.__new__(Deployer)
        deployer._pyproject_data = {
            "tool": {"deploy": {"agents": {"my_agent": {"path": "/tmp"}}}},
        }
        deployer._die = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))

        with pytest.raises(RuntimeError, match="cli_agents_source is not set"):
            deployer._build_agents()

    def test_dies_when_cli_agents_dir_missing(self, tmp_path: Path):
        """Should die when cli_agents_source points to non-existent dir."""
        deployer = Deployer.__new__(Deployer)
        deployer._pyproject_data = {
            "tool": {
                "deploy": {
                    "cli_agents_source": str(tmp_path / "nonexistent"),
                    "agents": {"my_agent": {"path": "/tmp"}},
                }
            },
        }
        deployer._die = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
        deployer._info = lambda *a, **k: None

        with pytest.raises(RuntimeError, match="cli-agents source not found"):
            deployer._build_agents()

    def test_dies_when_agent_path_missing(self, tmp_path: Path):
        """Should die when agent path has no pyproject.toml."""
        cli_dir = tmp_path / "cli-agents"
        cli_dir.mkdir()
        (cli_dir / "pyproject.toml").write_text('[project]\nname = "cli"\n')

        deployer = Deployer.__new__(Deployer)
        deployer._pyproject_data = {
            "tool": {
                "deploy": {
                    "cli_agents_source": str(cli_dir),
                    "agents": {"bad_agent": {"path": str(tmp_path / "nonexistent_agent")}},
                }
            },
        }
        deployer._die = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
        deployer._info = lambda *a, **k: None
        deployer._success = lambda *a, **k: None

        def mock_build(source_dir: Path) -> Path:
            whl = tmp_path / "dist" / f"{source_dir.name}-0.1.0-py3-none-any.whl"
            whl.parent.mkdir(exist_ok=True)
            whl.write_bytes(b"wheel")
            return whl

        deployer._build_wheel_from_source = mock_build  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="path not found"):
            deployer._build_agents()

    def test_dies_when_agent_missing_module(self, tmp_path: Path):
        """Should die when agent pyproject.toml lacks [tool.agent].module."""
        cli_dir = tmp_path / "cli-agents"
        cli_dir.mkdir()
        (cli_dir / "pyproject.toml").write_text('[project]\nname = "cli"\n')
        agent_dir = tmp_path / "my_agent"
        agent_dir.mkdir()
        (agent_dir / "pyproject.toml").write_text('[project]\nname = "agent"\nversion = "1.0"\n')

        deployer = Deployer.__new__(Deployer)
        deployer._pyproject_data = {
            "tool": {
                "deploy": {
                    "cli_agents_source": str(cli_dir),
                    "agents": {"my_agent": {"path": str(agent_dir)}},
                }
            },
        }
        deployer._die = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
        deployer._info = lambda *a, **k: None
        deployer._success = lambda *a, **k: None

        def mock_build(source_dir: Path) -> Path:
            whl = tmp_path / "dist" / f"{source_dir.name}-0.1.0-py3-none-any.whl"
            whl.parent.mkdir(exist_ok=True)
            whl.write_bytes(b"wheel")
            return whl

        deployer._build_wheel_from_source = mock_build  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match=r"missing.*module"):
            deployer._build_agents()


class TestDeployerBuildAgentsDedup:
    """Test that _build_agents deduplicates extra_vendor vs vendor/ packages."""

    def test_extra_vendor_skips_matching_vendor_tarball(self, tmp_path: Path):
        """When extra_vendor builds crypto_mcp wheel, vendor/crypto_mcp-*.tar.gz is skipped.

        Without dedup, the bundle would contain both crypto_mcp-0.2.0.whl AND
        crypto_mcp-0.2.0.tar.gz, wasting bandwidth and creating confusion.
        """
        # Set up agent directory with existing vendor tarball
        agent_dir = tmp_path / "agents" / "initial_research"
        agent_dir.mkdir(parents=True)
        (agent_dir / "pyproject.toml").write_text(
            '[project]\nname = "agent-initial-research"\nversion = "0.6.0"\n[tool.agent]\nmodule = "agent_initial_research"\n'
        )
        vendor_dir = agent_dir / "vendor"
        vendor_dir.mkdir()
        (vendor_dir / "crypto_mcp-0.2.0.tar.gz").write_bytes(b"old tarball")
        (vendor_dir / "other_pkg-1.0.0-py3-none-any.whl").write_bytes(b"other wheel")

        # Set up extra_vendor source and cli-agents source
        (tmp_path / "crypto_mcp").mkdir()
        (tmp_path / "crypto_mcp" / "pyproject.toml").write_text('[project]\nname = "crypto-mcp"\nversion = "0.2.0"\n')
        cli_dir = tmp_path / "vendor" / "cli-agents"
        cli_dir.mkdir(parents=True)
        (cli_dir / "pyproject.toml").write_text('[project]\nname = "cli-agents"\nversion = "0.7.0"\n')

        # Create Deployer with bypassed __init__
        deployer = Deployer.__new__(Deployer)
        deployer._pyproject_data = {
            "tool": {
                "deploy": {
                    "cli_agents_source": str(cli_dir),
                    "agents": {
                        "initial_research": {
                            "path": str(agent_dir),
                            "extra_vendor": [str(tmp_path / "crypto_mcp")],
                        }
                    },
                }
            },
        }
        deployer.config = {"bucket": "b", "folder": "flows/test"}
        deployer._info = lambda *a, **k: None
        deployer._success = lambda *a, **k: None
        deployer._die = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))

        # Mock _build_wheel_from_source to return fake wheels
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()

        def mock_build(source_dir: Path) -> Path:
            name = source_dir.name.replace("-", "_")
            whl = dist_dir / f"{name}-0.1.0-py3-none-any.whl"
            whl.write_bytes(b"wheel bytes")
            return whl

        deployer._build_wheel_from_source = mock_build  # type: ignore[assignment]

        builds = deployer._build_agents()

        assert "initial_research" in builds
        filenames = set(builds["initial_research"]["files"].keys())

        # crypto_mcp WHEEL should be present (from extra_vendor build)
        assert any("crypto_mcp" in f and f.endswith(".whl") for f in filenames)

        # crypto_mcp TARBALL should NOT be present (skipped by dedup)
        assert "crypto_mcp-0.2.0.tar.gz" not in filenames

        # other_pkg should still be present (not in extra_vendor)
        assert "other_pkg-1.0.0-py3-none-any.whl" in filenames

        # Manifest should not list the tarball
        manifest = json.loads(builds["initial_research"]["manifest_json"])
        assert "crypto_mcp-0.2.0.tar.gz" not in manifest["vendor_packages"]


class TestBuildWheelFromSource:
    """Test _build_wheel_from_source method."""

    def test_successful_build(self, tmp_path: Path):
        """Should build wheel and copy to dist/ under source dir."""
        source_dir = tmp_path / "my_pkg"
        source_dir.mkdir()
        (source_dir / "pyproject.toml").write_text('[project]\nname = "my-pkg"\nversion = "1.0"\n')

        deployer = Deployer.__new__(Deployer)
        deployer._die = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))

        fake_wheel = "my_pkg-1.0-py3-none-any.whl"

        def mock_run(*args: Any, **kwargs: Any) -> Mock:
            # Create wheel in the outdir that subprocess would use
            outdir = Path(args[0][args[0].index("--outdir") + 1])
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / fake_wheel).write_bytes(b"fake wheel content")
            result = Mock()
            result.returncode = 0
            return result

        with patch("ai_pipeline_core.deployment.deploy.subprocess.run", side_effect=mock_run):
            wheel_path = deployer._build_wheel_from_source(source_dir)

        assert wheel_path.name == fake_wheel
        assert wheel_path.parent == source_dir / "dist"
        assert wheel_path.read_bytes() == b"fake wheel content"

    def test_missing_pyproject_raises(self, tmp_path: Path):
        """Should die if source dir has no pyproject.toml."""
        deployer = Deployer.__new__(Deployer)
        deployer._die = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))

        with pytest.raises(RuntimeError, match=r"No pyproject.toml"):
            deployer._build_wheel_from_source(tmp_path)

    def test_no_wheel_produced_raises(self, tmp_path: Path):
        """Should die if build succeeds but produces no .whl file."""
        source_dir = tmp_path / "empty_pkg"
        source_dir.mkdir()
        (source_dir / "pyproject.toml").write_text('[project]\nname = "empty"\n')

        deployer = Deployer.__new__(Deployer)
        deployer._die = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))

        mock_result = Mock()
        mock_result.returncode = 0

        with patch("ai_pipeline_core.deployment.deploy.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="No wheel produced"):
                deployer._build_wheel_from_source(source_dir)

    def test_build_failure_raises(self, tmp_path: Path):
        """Should die if subprocess returns non-zero."""
        source_dir = tmp_path / "bad_pkg"
        source_dir.mkdir()
        (source_dir / "pyproject.toml").write_text('[project]\nname = "bad"\n')

        deployer = Deployer.__new__(Deployer)
        deployer._die = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))

        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "build error"

        with patch("ai_pipeline_core.deployment.deploy.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="Wheel build failed"):
                deployer._build_wheel_from_source(source_dir)
