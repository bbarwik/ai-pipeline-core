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
import subprocess
import sys
import tomllib
import traceback
from pathlib import Path
from typing import Any

from prefect.cli.deploy._storage import _PullStepStorage  # type: ignore
from prefect.client.orchestration import get_client
from prefect.deployments.runner import RunnerDeployment
from prefect.flows import load_flow_from_entrypoint
from prefect_gcp.cloud_storage import GcpCredentials, GcsBucket  # pyright: ignore[reportMissingTypeStubs]

from ai_pipeline_core.settings import settings


class Deployer:
    """Deploy Prefect flows using the RunnerDeployment pattern.

    Handles flow registration, deployment creation/updates, and all edge cases
    using the official Prefect approach.
    """

    def __init__(self):
        self.config = self._load_config()
        self._validate_prefect_settings()

    def _load_config(self) -> dict[str, Any]:
        """Load and normalize project configuration from pyproject.toml."""
        if not settings.prefect_gcs_bucket:
            self._die("PREFECT_GCS_BUCKET not configured in settings.\nConfigure via environment variable or .env file:\n  PREFECT_GCS_BUCKET=your-bucket-name")

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

    def _validate_prefect_settings(self) -> None:
        """Validate that required Prefect settings are configured."""
        self.api_url = settings.prefect_api_url
        if not self.api_url:
            self._die(
                "PREFECT_API_URL not configured in settings.\n"
                "Configure via environment variable or .env file:\n"
                "  PREFECT_API_URL=https://api.prefect.cloud/api/accounts/.../workspaces/..."
            )

    def _run(self, cmd: str, *, check: bool = True) -> str | None:
        """Execute shell command and return output."""
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
        if check and result.returncode != 0:
            self._die(f"Command failed: {cmd}\n{result.stderr}")
        return result.stdout.strip() if result.returncode == 0 else None

    @staticmethod
    def _info(msg: str) -> None:
        print(f"\u2192 {msg}")

    @staticmethod
    def _success(msg: str) -> None:
        print(f"\u2713 {msg}")

    @staticmethod
    def _die(msg: str) -> None:
        print(f"\u2717 {msg}", file=sys.stderr)
        sys.exit(1)

    def _build_package(self) -> Path:
        """Build Python package using `python -m build`."""
        self._info(f"Building {self.config['name']} v{self.config['version']}")
        self._run("python -m build --sdist")

        tarball_path = Path("dist") / self.config["tarball"]
        if not tarball_path.exists():
            self._die(f"Build artifact not found: {tarball_path}\nExpected tarball name: {self.config['tarball']}\nCheck that pyproject.toml version matches.")

        self._success(f"Built {tarball_path.name} ({tarball_path.stat().st_size // 1024} KB)")
        return tarball_path

    def _create_gcs_bucket(self, bucket_folder: str) -> Any:
        """Create a GcsBucket instance for uploading files."""
        creds = GcpCredentials()
        if hasattr(settings, "gcs_service_account_file") and settings.gcs_service_account_file:
            creds = GcpCredentials(service_account_file=Path(settings.gcs_service_account_file))
        return GcsBucket(bucket=self.config["bucket"], bucket_folder=bucket_folder, gcp_credentials=creds)

    async def _upload_package(self, tarball: Path) -> None:
        """Upload package tarball to Google Cloud Storage."""
        flow_folder = self.config["folder"]
        bucket = self._create_gcs_bucket(flow_folder)

        dest_uri = f"gs://{self.config['bucket']}/{flow_folder}/{tarball.name}"
        self._info(f"Uploading to {dest_uri}")

        await bucket.write_path(tarball.name, tarball.read_bytes())
        self._success(f"Package uploaded to {flow_folder}/{tarball.name}")

    async def _deploy_via_api(self) -> None:
        """Create or update Prefect deployment using RunnerDeployment pattern."""
        entrypoint = f"{self.config['package']}:{self.config['package']}"

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
                    "script": f"uv pip install --system --find-links . ./{self.config['tarball']}",
                }
            },
        ]

        self._info(f"Creating deployment for flow '{flow.name}'")  # pyright: ignore[reportPossiblyUnboundVariable]

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
            job_variables={},
            paused=False,
        )

        deployment._set_defaults_from_flow(flow)  # pyright: ignore[reportPossiblyUnboundVariable]

        return_type = getattr(flow.fn, "__annotations__", {}).get("return")  # pyright: ignore[reportPossiblyUnboundVariable]
        if return_type is not None and hasattr(return_type, "model_json_schema"):
            deployment._parameter_openapi_schema.definitions["_ResultSchema"] = return_type.model_json_schema()

        async with get_client() as client:
            try:
                work_pool = await client.read_work_pool(self.config["work_pool"])
                self._success(f"Work pool '{self.config['work_pool']}' verified (type: {work_pool.type})")
            except Exception as e:
                self._die(f"Work pool '{self.config['work_pool']}' not accessible: {e}\nCreate it in the Prefect UI or with: prefect work-pool create")

        self._info("Applying deployment (create or update)...")
        try:
            deployment_id = await deployment.apply()  # type: ignore
            self._success(f"Deployment ID: {deployment_id}")

            if self.api_url:
                ui_url = self.api_url.replace("/api/", "/")
                print(f"\nView deployment: {ui_url}/deployments/deployment/{deployment_id}")
                print(f"Run now: prefect deployment run '{flow.name}/{self.config['package']}'")  # pyright: ignore[reportPossiblyUnboundVariable]
        except Exception as e:
            self._die(f"Failed to apply deployment: {e}")

    async def run(self) -> None:
        """Execute the complete deployment pipeline: build, upload, deploy."""
        print("=" * 70)
        print(f"Prefect Deployment: {self.config['name']} v{self.config['version']}")
        print(f"Target: gs://{self.config['bucket']}/{self.config['folder']}")
        print("=" * 70)
        print()

        tarball = self._build_package()
        await self._upload_package(tarball)
        await self._deploy_via_api()

        print()
        print("=" * 70)
        self._success("Deployment complete!")
        print("=" * 70)


def main() -> None:
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
        """,
    )

    parser.parse_args()

    try:
        deployer = Deployer()
        asyncio.run(deployer.run())
    except KeyboardInterrupt:
        print("\n\u2717 Deployment cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n\u2717 Unexpected error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
