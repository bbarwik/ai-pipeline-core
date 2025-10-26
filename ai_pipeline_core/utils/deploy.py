#!/usr/bin/env python3
"""Universal Prefect deployment script using Python API.

This script:
1. Builds a Python package from pyproject.toml
2. Uploads it to Google Cloud Storage
3. Creates/updates a Prefect deployment using the RunnerDeployment pattern

Requirements:
- .env file with PREFECT_API_URL and optionally PREFECT_API_KEY
- .env file with PREFECT_GCS_BUCKET
- pyproject.toml with project name and version
- Local package installed for flow metadata extraction

Usage:
    python -m ai_pipeline_core.utils.deploy
"""

import argparse
import asyncio
import os
import subprocess
import sys
import tomllib
import traceback
from pathlib import Path
from typing import Any, Optional

from prefect.cli.deploy._storage import _PullStepStorage  # type: ignore
from prefect.client.orchestration import get_client
from prefect.deployments.runner import RunnerDeployment
from prefect.flows import load_flow_from_entrypoint

from ai_pipeline_core.settings import settings
from ai_pipeline_core.storage import Storage

# ============================================================================
# Configuration
# ============================================================================

WORK_POOL_NAME = settings.prefect_work_pool_name
DEFAULT_WORK_QUEUE = settings.prefect_work_queue_name
PREDEFINED_BUCKET = settings.prefect_gcs_bucket

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
        self._setup_prefect_env()

    def _load_config(self) -> dict[str, Any]:
        """Load and normalize project configuration from pyproject.toml.

        Returns:
            Configuration dictionary with project metadata and deployment settings.
        """
        if not PREDEFINED_BUCKET:
            self._die(
                "PREFECT_GCS_BUCKET not found in .env file.\n"
                "Create a .env file with:\n"
                "  PREFECT_GCS_BUCKET=your-bucket-name"
            )

        pyproject_path = Path("pyproject.toml")
        if not pyproject_path.exists():
            self._die("pyproject.toml not found. Run from project root.")

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

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
            "bucket": PREDEFINED_BUCKET,
            "folder": f"flows/{flow_folder}",
            "tarball": f"{package_name}-{version}.tar.gz",
            "work_pool": WORK_POOL_NAME,
            "work_queue": DEFAULT_WORK_QUEUE,
        }

    def _setup_prefect_env(self):
        """Configure Prefect environment variables from .env file."""
        self.api_url = os.getenv("PREFECT_API_URL")
        if not self.api_url:
            self._die(
                "PREFECT_API_URL not found in .env file.\n"
                "Create a .env file with:\n"
                "  PREFECT_API_URL=https://api.prefect.cloud/api/accounts/.../workspaces/..."
            )

        os.environ["PREFECT_API_URL"] = self.api_url

        # Optional: API key for authentication
        if api_key := os.getenv("PREFECT_API_KEY"):
            os.environ["PREFECT_API_KEY"] = api_key

        # Optional: Alternative auth method
        if api_auth := os.getenv("PREFECT_API_AUTH_STRING"):
            os.environ["PREFECT_API_AUTH_STRING"] = api_auth

    def _run(self, cmd: str, check: bool = True) -> Optional[str]:
        """Execute shell command and return output.

        Args:
            cmd: Shell command to execute
            check: Whether to raise on non-zero exit code

        Returns:
            Command stdout if successful, None if failed and check=False
        """
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if check and result.returncode != 0:
            self._die(f"Command failed: {cmd}\n{result.stderr}")

        return result.stdout.strip() if result.returncode == 0 else None

    def _info(self, msg: str):
        """Print info message."""
        print(f"→ {msg}")

    def _success(self, msg: str):
        """Print success message."""
        print(f"✓ {msg}")

    def _die(self, msg: str):
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
            self._die(
                f"Build artifact not found: {tarball_path}\n"
                f"Expected tarball name: {self.config['tarball']}\n"
                f"Check that pyproject.toml version matches."
            )

        self._success(f"Built {tarball_path.name} ({tarball_path.stat().st_size // 1024} KB)")
        return tarball_path

    async def _upload_package(self, tarball: Path):
        """Upload package tarball to Google Cloud Storage using Storage abstraction.

        Args:
            tarball: Path to the tarball to upload
        """
        # Extract flow_folder from the config folder path
        # e.g., "flows/ai-document-writer" -> "ai-document-writer"
        flow_folder = self.config["folder"].split("/", 1)[1] if "/" in self.config["folder"] else ""

        # Initialize storage with gs://bucket-name/flows and set subfolder to flow_folder
        base_uri = f"gs://{self.config['bucket']}/flows"
        storage = await Storage.from_uri(base_uri)
        storage = storage.with_base(flow_folder)

        dest_uri = storage.url_for(tarball.name)
        self._info(f"Uploading to {dest_uri}")

        # Read and upload the tarball
        tarball_bytes = tarball.read_bytes()
        await storage.write_bytes(tarball.name, tarball_bytes)

        self._success(f"Package uploaded to {self.config['folder']}/{tarball.name}")

    async def _deploy_via_api(self):
        """Create or update Prefect deployment using RunnerDeployment pattern.

        This is the official Prefect approach that:
        1. Automatically creates/updates the flow registration
        2. Handles deployment create vs update logic
        3. Properly formats all parameters for the API
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
                    "script": f"uv pip install --system ./{self.config['tarball']}",
                }
            },
        ]

        # Create RunnerDeployment
        # This is the official Prefect pattern that handles all the complexity
        self._info(f"Creating deployment for flow '{flow.name}'")

        deployment = RunnerDeployment(
            name=self.config["package"],
            flow_name=flow.name,
            entrypoint=entrypoint,
            work_pool_name=self.config["work_pool"],
            work_queue_name=self.config["work_queue"],
            tags=[self.config["name"]],
            version=self.config["version"],
            description=flow.description
            or f"Deployment for {self.config['package']} v{self.config['version']}",
            storage=_PullStepStorage(pull_steps),
            parameters={},
            job_variables={},
            paused=False,
        )

        # Verify work pool exists before deploying
        async with get_client() as client:
            try:
                work_pool = await client.read_work_pool(self.config["work_pool"])
                self._success(
                    f"Work pool '{self.config['work_pool']}' verified (type: {work_pool.type})"
                )
            except Exception as e:
                self._die(
                    f"Work pool '{self.config['work_pool']}' not accessible: {e}\n"
                    "Create it in the Prefect UI or with: prefect work-pool create"
                )

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
                print(f"🚀 Run now: prefect deployment run '{flow.name}/{self.config['package']}'")
        except Exception as e:
            self._die(f"Failed to apply deployment: {e}")

    async def run(self):
        """Execute the complete deployment pipeline."""
        print("=" * 70)
        print(f"Prefect Deployment: {self.config['name']} v{self.config['version']}")
        print(f"Target: gs://{self.config['bucket']}/{self.config['folder']}")
        print("=" * 70)
        print()

        # Phase 1: Build
        tarball = self._build_package()

        # Phase 2: Upload
        await self._upload_package(tarball)

        # Phase 3: Deploy
        await self._deploy_via_api()

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
Example:
  python -m ai_pipeline_core.utils.deploy

Prerequisites:
  - .env file with PREFECT_API_URL (and optionally PREFECT_API_KEY)
  - .env file with PREFECT_GCS_BUCKET
  - pyproject.toml with project name and version
  - Package installed locally: pip install -e .
  - GCP authentication configured (via service account or default credentials)
  - Work pool created in Prefect UI or CLI
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
