#!/usr/bin/env python3
"""Universal Prefect deployment script using Python API.

This script:
1. Builds a Python package from pyproject.toml
2. Uploads it to Google Cloud Storage
3. Creates/updates a Prefect deployment using the RunnerDeployment pattern

Requirements:
- Settings configured with PREFECT_API_URL and optionally PREFECT_API_KEY
- Settings configured with PREFECT_GCS_BUCKET
- pyproject.toml with project name and version
- Local package installed for flow metadata extraction

Usage:
    python -m ai_pipeline_core.deployment.deploy
"""

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import tomllib
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prefect.cli.deploy._storage import _PullStepStorage  # type: ignore
from prefect.client.orchestration import get_client
from prefect.deployments.runner import RunnerDeployment
from prefect.flows import load_flow_from_entrypoint
from prefect_gcp.cloud_storage import GcpCredentials, GcsBucket  # pyright: ignore[reportMissingTypeStubs]

from ai_pipeline_core.settings import settings

# ============================================================================
# Deployer Class
# ============================================================================


class Deployer:
    """Deploy Prefect flows using the RunnerDeployment pattern.

    This is the official Prefect approach that handles flow registration,
    deployment creation/updates, and all edge cases automatically.
    """

    def __init__(self):
        """Initialize deployer."""
        self.config = self._load_config()
        self._validate_prefect_settings()

    def _load_config(self) -> dict[str, Any]:
        """Load and normalize project configuration from pyproject.toml.

        Returns:
            Configuration dictionary with project metadata and deployment settings.
        """
        if not settings.prefect_gcs_bucket:
            self._die("PREFECT_GCS_BUCKET not configured in settings.\nConfigure via environment variable or .env file:\n  PREFECT_GCS_BUCKET=your-bucket-name")

        pyproject_path = Path("pyproject.toml")
        if not pyproject_path.exists():
            self._die("pyproject.toml not found. Run from project root.")

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        self._pyproject_data = data

        project = data.get("project", {})
        name = project.get("name")
        version = project.get("version")

        if not name:
            self._die("Project name not found in pyproject.toml")
        if not version:
            self._die("Project version not found in pyproject.toml")

        # Normalize naming conventions
        # Hyphens in package names become underscores in Python imports
        package_name = name.replace("-", "_")
        flow_folder = name.replace("_", "-")

        return {
            "name": name,
            "package": package_name,
            "version": version,
            "bucket": settings.prefect_gcs_bucket,
            "folder": f"flows/{flow_folder}",
            "tarball": f"{package_name}-{version}.tar.gz",
            "work_pool": settings.prefect_work_pool_name,
            "work_queue": settings.prefect_work_queue_name,
        }

    def _validate_prefect_settings(self):
        """Validate that required Prefect settings are configured."""
        self.api_url = settings.prefect_api_url
        if not self.api_url:
            self._die(
                "PREFECT_API_URL not configured in settings.\n"
                "Configure via environment variable or .env file:\n"
                "  PREFECT_API_URL=https://api.prefect.cloud/api/accounts/.../workspaces/..."
            )

    def _run(self, cmd: str, *, check: bool = True) -> str | None:
        """Execute shell command and return output.

        Args:
            cmd: Shell command to execute
            check: Whether to raise on non-zero exit code

        Returns:
            Command stdout if successful, None if failed and check=False
        """
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)

        if check and result.returncode != 0:
            self._die(f"Command failed: {cmd}\n{result.stderr}")

        return result.stdout.strip() if result.returncode == 0 else None

    @staticmethod
    def _info(msg: str):
        """Print info message."""
        print(f"→ {msg}")

    @staticmethod
    def _success(msg: str):
        """Print success message."""
        print(f"✓ {msg}")

    @staticmethod
    def _die(msg: str):
        """Print error and exit."""
        print(f"✗ {msg}", file=sys.stderr)
        sys.exit(1)

    def _build_package(self) -> Path:
        """Build Python package using `python -m build`.

        Returns:
            Path to the built tarball
        """
        self._info(f"Building {self.config['name']} v{self.config['version']}")

        # Build sdist (source distribution)
        build_cmd = "python -m build --sdist"

        self._run(build_cmd)

        # Verify tarball was created
        tarball_path = Path("dist") / self.config["tarball"]
        if not tarball_path.exists():
            self._die(f"Build artifact not found: {tarball_path}\nExpected tarball name: {self.config['tarball']}\nCheck that pyproject.toml version matches.")

        self._success(f"Built {tarball_path.name} ({tarball_path.stat().st_size // 1024} KB)")
        return tarball_path

    # -- Agent build/upload support --

    def _load_agent_config(self) -> dict[str, dict[str, Any]]:
        """Load [tool.deploy.agents] from pyproject.toml.

        Returns:
            Dict mapping agent name to config (path, extra_vendor).
            Empty dict if no agents configured.
        """
        return self._pyproject_data.get("tool", {}).get("deploy", {}).get("agents", {})

    def _get_cli_agents_source(self) -> str | None:
        """Get cli_agents_source path from [tool.deploy]."""
        return self._pyproject_data.get("tool", {}).get("deploy", {}).get("cli_agents_source")

    def _build_wheel_from_source(self, source_dir: Path) -> Path:
        """Build a wheel from a source directory.

        Args:
            source_dir: Directory containing pyproject.toml

        Returns:
            Path to built .whl file in a temp dist directory
        """
        if not (source_dir / "pyproject.toml").exists():
            self._die(f"No pyproject.toml in {source_dir}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dist = Path(tmpdir) / "dist"
            result = subprocess.run(
                [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_dist)],
                cwd=source_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                self._die(f"Wheel build failed for {source_dir.name}:\n{result.stderr}")

            wheels = list(tmp_dist.glob("*.whl"))
            if not wheels:
                self._die(f"No wheel produced for {source_dir.name}")

            # Copy to persistent dist/ under source_dir
            dist_dir = source_dir / "dist"
            dist_dir.mkdir(exist_ok=True)
            output = dist_dir / wheels[0].name
            output.write_bytes(wheels[0].read_bytes())
            return output

    def _build_agents(self) -> dict[str, dict[str, Any]]:  # noqa: PLR0914
        """Build agent wheels and manifests for all configured agents.

        Returns:
            Dict mapping agent name to build info:
                {name: {"manifest_json": str, "files": {filename: Path}}}
            Empty dict if no agents configured.
        """
        agent_config = self._load_agent_config()
        if not agent_config:
            return {}

        cli_agents_source = self._get_cli_agents_source()
        if not cli_agents_source:
            self._die(
                "Agents configured in [tool.deploy.agents] but "
                "[tool.deploy].cli_agents_source is not set.\n"
                "Add to pyproject.toml:\n"
                '  [tool.deploy]\n  cli_agents_source = "vendor/cli-agents"'
            )

        self._info(f"Building {len(agent_config)} agent(s): {', '.join(agent_config)}")

        # Build cli-agents wheel once (shared across all agents)
        cli_agents_dir = Path(cli_agents_source).resolve()  # pyright: ignore[reportArgumentType]
        if not (cli_agents_dir / "pyproject.toml").exists():
            self._die(f"cli-agents source not found at {cli_agents_dir}")

        cli_agents_wheel = self._build_wheel_from_source(cli_agents_dir)
        self._success(f"Built cli-agents wheel: {cli_agents_wheel.name}")

        builds: dict[str, dict[str, Any]] = {}

        for agent_name, config in agent_config.items():
            agent_path = Path(config["path"]).resolve()
            if not (agent_path / "pyproject.toml").exists():
                self._die(f"Agent '{agent_name}' path not found: {agent_path}\nCheck [tool.deploy.agents.{agent_name}].path in pyproject.toml")

            # Read module_name from agent's pyproject.toml
            with open(agent_path / "pyproject.toml", "rb") as f:
                agent_pyproject = tomllib.load(f)

            module_name = agent_pyproject.get("tool", {}).get("agent", {}).get("module")
            if not module_name:
                self._die(
                    f"Agent '{agent_name}' missing [tool.agent].module in "
                    f"{agent_path / 'pyproject.toml'}\n"
                    f'Add:\n  [tool.agent]\n  module = "agent_{agent_name}"'
                )

            # Build agent wheel
            agent_wheel = self._build_wheel_from_source(agent_path)
            self._success(f"Built agent wheel: {agent_wheel.name}")

            # Collect all files for this agent bundle
            files: dict[str, Path] = {
                agent_wheel.name: agent_wheel,
                cli_agents_wheel.name: cli_agents_wheel,
            }

            # Build extra_vendor packages from repo root
            vendor_packages: list[str] = []
            extra_built: set[str] = set()
            for vendor_name in config.get("extra_vendor", []):
                extra_source_dir = Path(vendor_name).resolve()
                if not (extra_source_dir / "pyproject.toml").exists():
                    self._die(
                        f"Extra vendor '{vendor_name}' for agent '{agent_name}' "
                        f"not found at {extra_source_dir}\n"
                        f"Ensure the directory exists at repo root with pyproject.toml"
                    )
                vendor_wheel = self._build_wheel_from_source(extra_source_dir)
                files[vendor_wheel.name] = vendor_wheel
                vendor_packages.append(vendor_wheel.name)
                extra_built.add(extra_source_dir.name.replace("-", "_"))
                self._success(f"Built vendor wheel: {vendor_wheel.name}")

            # Collect existing vendor/*.whl and vendor/*.tar.gz from agent directory,
            # skipping packages already built from extra_vendor
            agent_vendor_dir = agent_path / "vendor"
            if agent_vendor_dir.exists():
                for pkg in list(agent_vendor_dir.glob("*.whl")) + list(agent_vendor_dir.glob("*.tar.gz")):
                    pkg_base = pkg.name.split("-")[0].replace("-", "_")
                    if pkg.name not in files and pkg_base not in extra_built:
                        files[pkg.name] = pkg
                        vendor_packages.append(pkg.name)

            # Write manifest (plain JSON dict, compatible with AgentManifest schema)
            manifest = {
                "module_name": module_name,
                "agent_wheel": agent_wheel.name,
                "cli_agents_wheel": cli_agents_wheel.name,
                "vendor_packages": vendor_packages,
                "built_at": datetime.now(UTC).isoformat(),
            }
            manifest_json = json.dumps(manifest, indent=2)

            builds[agent_name] = {"manifest_json": manifest_json, "files": files}
            self._success(f"Agent '{agent_name}' bundle ready ({module_name}, {len(files)} files)")

        return builds

    def _build_vendor_packages(self) -> list[Path]:
        """Build vendor wheels from [tool.deploy].vendor_packages paths.

        Returns:
            List of built wheel paths, deduplicated by filename.
        """
        vendor_paths: list[str] = self._pyproject_data.get("tool", {}).get("deploy", {}).get("vendor_packages", [])
        if not vendor_paths:
            return []

        self._info(f"Building {len(vendor_paths)} vendor package(s)")
        wheels: list[Path] = []
        seen: set[str] = set()
        for vendor_path_str in vendor_paths:
            vendor_dir = Path(vendor_path_str).resolve()
            wheel = self._build_wheel_from_source(vendor_dir)
            if wheel.name not in seen:
                wheels.append(wheel)
                seen.add(wheel.name)
                self._success(f"Built vendor wheel: {wheel.name}")
        return wheels

    def _create_gcs_bucket(self, bucket_folder: str) -> Any:
        """Create a GcsBucket instance for uploading files.

        Args:
            bucket_folder: Folder path within the bucket.
        """
        creds = GcpCredentials()
        if hasattr(settings, "gcs_service_account_file") and settings.gcs_service_account_file:
            creds = GcpCredentials(service_account_file=Path(settings.gcs_service_account_file))
        return GcsBucket(bucket=self.config["bucket"], bucket_folder=bucket_folder, gcp_credentials=creds)

    async def _upload_agents(self, agent_builds: dict[str, dict[str, Any]]):
        """Upload agent bundles to GCS.

        Args:
            agent_builds: Output from _build_agents()
        """
        if not agent_builds:
            return

        flow_folder = self.config["folder"]

        for agent_name, build_info in agent_builds.items():
            agent_folder = f"{flow_folder}/agents/{agent_name}"
            bucket = self._create_gcs_bucket(agent_folder)
            self._info(f"Uploading agent '{agent_name}' bundle to gs://{self.config['bucket']}/{agent_folder}")

            # Upload manifest
            await bucket.write_path("manifest.json", build_info["manifest_json"].encode())

            # Upload wheels
            for filename, filepath in build_info["files"].items():
                await bucket.write_path(filename, filepath.read_bytes())

            self._success(f"Agent '{agent_name}' uploaded ({len(build_info['files'])} files)")

    async def _upload_package(self, tarball: Path, vendor_wheels: list[Path] | None = None):
        """Upload package tarball and vendor wheels to Google Cloud Storage.

        Args:
            tarball: Path to the tarball to upload
            vendor_wheels: Optional private dependency wheels to upload alongside
        """
        flow_folder = self.config["folder"]
        bucket = self._create_gcs_bucket(flow_folder)

        dest_uri = f"gs://{self.config['bucket']}/{flow_folder}/{tarball.name}"
        self._info(f"Uploading to {dest_uri}")

        tarball_bytes = tarball.read_bytes()
        await bucket.write_path(tarball.name, tarball_bytes)

        self._success(f"Package uploaded to {flow_folder}/{tarball.name}")

        for wheel in vendor_wheels or []:
            await bucket.write_path(wheel.name, wheel.read_bytes())
            self._success(f"Vendor wheel uploaded: {wheel.name}")

    async def _deploy_via_api(self, agent_builds: dict[str, dict[str, Any]] | None = None):
        """Create or update Prefect deployment using RunnerDeployment pattern.

        This is the official Prefect approach that:
        1. Automatically creates/updates the flow registration
        2. Handles deployment create vs update logic
        3. Properly formats all parameters for the API

        Args:
            agent_builds: Output from _build_agents(). If non-empty, sets
                AGENT_BUNDLES_URI env var on the deployment.
        """
        # Define entrypoint (assumes flow function has same name as package)
        entrypoint = f"{self.config['package']}:{self.config['package']}"

        # Load flow to get metadata
        # This requires the package to be installed locally (typical dev workflow)
        self._info(f"Loading flow from entrypoint: {entrypoint}")
        try:
            flow = load_flow_from_entrypoint(entrypoint)
            self._success(f"Loaded flow: {flow.name}")
        except ImportError as e:
            self._die(
                f"Failed to import flow: {e}\n\n"
                f"The package must be installed locally to extract flow metadata.\n"
                f"Install it with: pip install -e .\n\n"
                f"Expected entrypoint: {entrypoint}\n"
                f"This means: Python package '{self.config['package']}' "
                f"with flow function '{self.config['package']}'"
            )
        except AttributeError as e:
            self._die(
                f"Flow function not found: {e}\n\n"
                f"Expected flow function named '{self.config['package']}' "
                f"in package '{self.config['package']}'.\n"
                f"Check that your flow is decorated with @flow and named correctly."
            )

        # Define pull steps for workers
        # These steps tell workers how to get and install the flow code
        pull_steps = [
            {
                "prefect_gcp.deployments.steps.pull_from_gcs": {
                    "id": "pull_code",
                    "requires": "prefect-gcp>=0.6",
                    "bucket": self.config["bucket"],
                    "folder": self.config["folder"],
                }
            },
            {
                "prefect.deployments.steps.run_shell_script": {
                    "id": "install_project",
                    "stream_output": True,
                    "directory": "{{ pull_code.directory }}",
                    # Use uv for fast installation (worker has it installed)
                    # --find-links . resolves private dependencies from co-uploaded wheels
                    "script": f"uv pip install --system --find-links . ./{self.config['tarball']}",
                }
            },
        ]

        # Create RunnerDeployment
        # This is the official Prefect pattern that handles all the complexity
        self._info(f"Creating deployment for flow '{flow.name}'")  # pyright: ignore[reportPossiblyUnboundVariable]

        # Set AGENT_BUNDLES_URI env var if agents were built
        job_variables: dict[str, Any] = {}
        if agent_builds:
            bundles_uri = f"gs://{self.config['bucket']}/{self.config['folder']}/agents"
            job_variables["env"] = {"AGENT_BUNDLES_URI": bundles_uri}
            self._info(f"Setting AGENT_BUNDLES_URI={bundles_uri}")

        deployment = RunnerDeployment(
            name=self.config["package"],
            flow_name=flow.name,  # pyright: ignore[reportPossiblyUnboundVariable]
            entrypoint=entrypoint,
            work_pool_name=self.config["work_pool"],
            work_queue_name=self.config["work_queue"],
            tags=[self.config["name"]],
            version=self.config["version"],
            description=flow.description or f"Deployment for {self.config['package']} v{self.config['version']}",  # pyright: ignore[reportPossiblyUnboundVariable]
            storage=_PullStepStorage(pull_steps),
            parameters={},
            job_variables=job_variables,
            paused=False,
        )

        # Populate parameter schema from flow function signature
        deployment._set_defaults_from_flow(flow)  # pyright: ignore[reportPossiblyUnboundVariable]

        # Inject result type schema so consumers can discover the response shape
        return_type = getattr(flow.fn, "__annotations__", {}).get("return")  # pyright: ignore[reportPossiblyUnboundVariable]
        if return_type is not None and hasattr(return_type, "model_json_schema"):
            deployment._parameter_openapi_schema.definitions["_ResultSchema"] = return_type.model_json_schema()

        # Verify work pool exists before deploying
        async with get_client() as client:
            try:
                work_pool = await client.read_work_pool(self.config["work_pool"])
                self._success(f"Work pool '{self.config['work_pool']}' verified (type: {work_pool.type})")
            except Exception as e:
                self._die(f"Work pool '{self.config['work_pool']}' not accessible: {e}\nCreate it in the Prefect UI or with: prefect work-pool create")

        # Apply deployment
        # This automatically handles create vs update based on whether deployment exists
        self._info("Applying deployment (create or update)...")
        try:
            deployment_id = await deployment.apply()  # type: ignore
            self._success(f"Deployment ID: {deployment_id}")

            # Print helpful URLs
            if self.api_url:
                ui_url = self.api_url.replace("/api/", "/")
                print(f"\n🌐 View deployment: {ui_url}/deployments/deployment/{deployment_id}")
                print(f"🚀 Run now: prefect deployment run '{flow.name}/{self.config['package']}'")  # pyright: ignore[reportPossiblyUnboundVariable]
        except Exception as e:
            self._die(f"Failed to apply deployment: {e}")

    async def run(self):
        """Execute the complete deployment pipeline."""
        print("=" * 70)
        print(f"Prefect Deployment: {self.config['name']} v{self.config['version']}")
        print(f"Target: gs://{self.config['bucket']}/{self.config['folder']}")
        print("=" * 70)
        print()

        # Phase 1: Build flow package
        tarball = self._build_package()

        # Phase 2: Build agent bundles (if configured)
        agent_builds = self._build_agents()

        # Phase 3: Build vendor packages from [tool.deploy].vendor_packages
        vendor_wheels = self._build_vendor_packages()

        # Build cli-agents wheel if source is configured — it's a private package
        # not on PyPI, so the worker needs the wheel even when no agents are deployed
        cli_agents_source = self._get_cli_agents_source()
        if cli_agents_source:
            cli_dir = Path(cli_agents_source).resolve()
            if (cli_dir / "pyproject.toml").exists():
                cli_wheel = self._build_wheel_from_source(cli_dir)
                if cli_wheel.name not in {w.name for w in vendor_wheels}:
                    vendor_wheels.append(cli_wheel)
                    self._success(f"Built cli-agents vendor wheel: {cli_wheel.name}")

        # Phase 4: Upload flow package + vendor wheels
        await self._upload_package(tarball, vendor_wheels)

        # Phase 5: Upload agent bundles
        await self._upload_agents(agent_builds)

        # Phase 6: Create/update Prefect deployment
        await self._deploy_via_api(agent_builds)

        print()
        print("=" * 70)
        self._success("Deployment complete!")
        print("=" * 70)


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    """Command-line interface for deployment script."""
    parser = argparse.ArgumentParser(
        description="Deploy Prefect flows to GCP using the official RunnerDeployment pattern",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prerequisites:
  - Settings configured with PREFECT_API_URL (and optionally PREFECT_API_KEY)
  - Settings configured with PREFECT_GCS_BUCKET
  - pyproject.toml with project name and version
  - Package installed locally: pip install -e .
  - GCP authentication configured (via service account or default credentials)
  - Work pool created in Prefect UI or CLI

Settings can be configured via:
  - Environment variables (e.g., export PREFECT_API_URL=...)
  - .env file in the current directory
        """,
    )

    parser.parse_args()

    try:
        deployer = Deployer()
        asyncio.run(deployer.run())
    except KeyboardInterrupt:
        print("\n✗ Deployment cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
