"""Abstract agent execution interfaces.

This module defines the contract for agent providers. ai-pipeline-core
does not implement any concrete providers - install a provider package
and register it to use agents.

Usage: See tests/agents/ for usage patterns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

__all__ = ["AgentProvider", "AgentResult"]


@dataclass(frozen=True)
class AgentResult:
    """Result from agent execution.

    This is an immutable container for all data returned by an agent run.
    The frozen=True ensures results cannot be accidentally modified.

    Attributes:
        success: Whether the agent completed successfully
        output: Structured output data from the agent (dict)
        artifacts: Binary files produced by the agent {name: bytes}
        error: Error message if success=False
        traceback: Full traceback if success=False
        exit_code: Process exit code if applicable
        duration_seconds: Total execution time
        stdout: Captured stdout if available
        stderr: Captured stderr if available
        agent_name: Name of the agent that ran
        agent_version: Version of the agent
    """

    success: bool
    output: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, bytes] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None

    # Debugging metadata
    exit_code: int | None = None
    duration_seconds: float | None = None
    stdout: str | None = None
    stderr: str | None = None
    agent_name: str | None = None
    agent_version: str | None = None

    def get_artifact(self, name: str, encoding: str = "utf-8") -> str | None:
        """Get artifact content decoded as string.

        Args:
            name: Artifact filename (e.g., "output.md")
            encoding: Text encoding (default UTF-8)

        Returns:
            Decoded string content, or None if artifact doesn't exist
        """
        data = self.artifacts.get(name)
        if data is not None:
            return data.decode(encoding)
        return None

    def get_artifact_bytes(self, name: str) -> bytes | None:
        """Get artifact content as raw bytes.

        Args:
            name: Artifact filename

        Returns:
            Raw bytes, or None if artifact doesn't exist
        """
        return self.artifacts.get(name)


class AgentProvider(ABC):
    """Abstract base class for agent execution providers.

    Implementations handle all details of agent execution:
    - Resolving agent name to executable (local path, downloaded bundle, etc.)
    - Executing the agent (subprocess, remote worker, container, etc.)
    - Collecting results and artifacts

    ai-pipeline-core only interacts through this interface.
    """

    @abstractmethod
    async def run(
        self,
        agent_name: str,
        inputs: dict[str, Any],
        *,
        files: dict[str, str | bytes] | None = None,
        target_worker: str | None = None,
        backend: str | None = None,
        timeout_seconds: int = 3600,
        env_vars: dict[str, str] | None = None,
    ) -> AgentResult:
        """Execute an agent and return results.

        This is the main entry point for running agents. Implementations
        should handle resolution, execution, and result collection.

        Args:
            agent_name: Identifier for the agent (e.g., "initial_research")
            inputs: Input parameters passed to the agent
            files: Additional files to make available to the agent
            target_worker: Hint for which worker to run on (provider-specific)
            backend: Which backend to use (provider-specific, e.g., "codex")
            timeout_seconds: Maximum execution time before cancellation
            env_vars: Environment variables to set for the agent

        Returns:
            AgentResult containing success status, outputs, and artifacts

        Raises:
            ValueError: If agent_name is invalid
            RuntimeError: If provider is misconfigured
        """

    async def list_agents(self) -> list[str]:
        """List available agent names.

        Override this method to support agent discovery.
        Default implementation returns empty list.

        Returns:
            List of agent names that can be passed to run()
        """
        return []

    async def validate(self) -> None:
        """Validate provider configuration.

        Override this method to add configuration checks.
        Called lazily on first run() to fail fast with clear errors.

        Raises:
            RuntimeError: If provider is misconfigured with helpful message
        """
