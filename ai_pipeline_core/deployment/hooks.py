"""Deployment hook system for extensibility.

Hooks allow external packages to extend the deployment process.
For example, an agent provider can add a hook that:
- Builds agent bundles during deployment
- Uploads them to cloud storage
- Injects environment variables into Prefect workers

Unlike auto-registration patterns, hooks are explicitly configured
in pyproject.toml for predictable, debuggable behavior.

Configuration:
    [tool.deploy]
    hooks = ["my_provider.deployment_hook"]

The deploy script loads and executes each hook module.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DeploymentHook",
    "DeploymentHookResult",
    "load_deployment_hooks",
]


@dataclass
class DeploymentHookResult:
    """Result from a deployment hook.

    Attributes:
        artifacts: List of (relative_path, bytes) tuples to upload
        job_variables: Dict to deep-merge into Prefect job_variables,
            e.g. {"env": {"MY_VAR": "value"}}
    """

    artifacts: list[tuple[str, bytes]] = field(default_factory=list)
    job_variables: dict[str, Any] = field(default_factory=dict)


class DeploymentHook(ABC):
    """Abstract base class for deployment hooks.

    Implementations extend the deployment process with custom logic.
    Each hook is called once during deployment and can:
    - Build additional artifacts (agent bundles, config files, etc.)
    - Add environment variables to Prefect worker configuration

    Hooks are loaded explicitly from pyproject.toml configuration,
    not auto-registered, for predictable behavior.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Hook name for logging and debugging."""

    @abstractmethod
    async def process(
        self,
        project_root: Path,
        pyproject: dict[str, Any],
        build_dir: Path,
        upload_uri: str,
    ) -> DeploymentHookResult | None:
        """Process the deployment.

        Called during deployment after the main package is built.
        Return None to skip (e.g., if this hook doesn't apply),
        or DeploymentHookResult with artifacts and job_variables.

        Args:
            project_root: Root directory of the project being deployed
            pyproject: Parsed pyproject.toml contents
            build_dir: Temporary directory for build artifacts
            upload_uri: Base URI where artifacts will be uploaded
                (e.g., "gs://bucket/flows/my-project")

        Returns:
            DeploymentHookResult with artifacts and job_variables,
            or None to skip this hook
        """


def load_deployment_hooks(pyproject: dict[str, Any]) -> list[DeploymentHook]:
    """Load deployment hooks from pyproject.toml configuration.

    Hooks are specified as module paths in [tool.deploy.hooks].
    Each module must have a get_hook() function returning a DeploymentHook.

    Args:
        pyproject: Parsed pyproject.toml contents

    Returns:
        List of DeploymentHook instances

    Example pyproject.toml:
        [tool.deploy]
        hooks = ["cli_agents.pipeline_integration.deployment_hook"]
    """
    import importlib

    hook_modules = pyproject.get("tool", {}).get("deploy", {}).get("hooks", [])
    hooks: list[DeploymentHook] = []

    for module_path in hook_modules:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, "get_hook"):
                hook = module.get_hook()
                if isinstance(hook, DeploymentHook):
                    hooks.append(hook)
                else:
                    raise TypeError(f"get_hook() must return DeploymentHook, got {type(hook).__name__}")
            else:
                raise AttributeError(f"Hook module {module_path} must have get_hook() function")
        except Exception as e:
            raise RuntimeError(f"Failed to load deployment hook '{module_path}': {e}") from e

    return hooks
