"""Agent execution entry point.

This module provides run_agent(), the primary way to execute agents.
It's a thin wrapper that delegates to the registered provider.
"""

import asyncio
from typing import Any

from ai_pipeline_core.logging import get_pipeline_logger

from .base import AgentResult
from .registry import get_agent_provider

__all__ = ["run_agent"]

logger = get_pipeline_logger(__name__)

# Lock for provider validation (prevents duplicate validation calls)
_validation_lock = asyncio.Lock()

# Track validated providers by id() - works with all providers including __slots__
# Using set of ids since providers are long-lived (registered singleton)
# nosemgrep: no-mutable-module-globals - intentional mutable state protected by lock
_validated_provider_ids: set[int] = set()


async def run_agent(
    agent_name: str,
    inputs: dict[str, Any],
    *,
    files: dict[str, str | bytes] | None = None,
    target_worker: str | None = None,
    backend: str | None = None,
    timeout_seconds: int = 3600,
    env_vars: dict[str, str] | None = None,
) -> AgentResult:
    """Execute an agent via the registered provider.

    This function is the main entry point for running agents. It:
    1. Gets the registered provider
    2. Validates the provider on first use (lazy validation)
    3. Delegates to provider.run()
    4. Logs success/failure

    The actual execution logic is entirely in the provider implementation.
    This keeps ai-pipeline-core decoupled from any specific agent system.

    Args:
        agent_name: Name of the agent to run (e.g., "initial_research")
        inputs: Input parameters passed to the agent as a dict
        files: Additional files to include in agent workspace {name: content}
        target_worker: Which worker to run on (provider-specific)
        backend: Which backend to use (provider-specific, e.g., "codex")
        timeout_seconds: Maximum execution time (default 1 hour)
        env_vars: Environment variables to set for the agent

    Returns:
        AgentResult with success status, output, and artifacts

    Raises:
        RuntimeError: If no provider is registered
        ValueError: If agent_name is invalid
    """
    provider = get_agent_provider()
    provider_name = type(provider).__name__
    provider_id = id(provider)

    logger.info(f"Running agent '{agent_name}' via {provider_name}")

    # Lazy validation on first use (with lock to prevent duplicate validation)
    # Using id() to track validated providers - works with all providers including __slots__
    if provider_id not in _validated_provider_ids:
        async with _validation_lock:
            # Double-check after acquiring lock
            if provider_id not in _validated_provider_ids:
                await provider.validate()
                _validated_provider_ids.add(provider_id)

    result = await provider.run(
        agent_name=agent_name,
        inputs=inputs,
        files=files,
        target_worker=target_worker,
        backend=backend,
        timeout_seconds=timeout_seconds,
        env_vars=env_vars,
    )

    if result.success:
        duration = f"{result.duration_seconds:.1f}s" if result.duration_seconds else "unknown"
        logger.info(f"Agent '{agent_name}' completed successfully (duration: {duration})")
    else:
        logger.warning(f"Agent '{agent_name}' failed: {result.error}")

    return result
