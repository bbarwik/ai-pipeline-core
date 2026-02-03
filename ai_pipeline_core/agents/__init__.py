"""Agent execution framework.

This module provides a provider-agnostic interface for running agents.
ai-pipeline-core defines the interfaces; concrete providers (installed
separately) implement the actual execution logic.

Quick Start:
    1. Install and register a provider (once, at startup):

        from your_provider import register_provider
        register_provider()

    2. Run agents in your tasks:

        from ai_pipeline_core.agents import run_agent, AgentOutputDocument

        @pipeline_task
        async def my_task(input_doc: MyDocument) -> AgentOutputDocument:
            result = await run_agent("my_agent", {"param": "value"})
            return AgentOutputDocument.from_result(
                result,
                origins=(input_doc.sha256,),
            )

Providers:
    Providers implement AgentProvider ABC and handle:
    - Agent resolution (finding agent code/bundles)
    - Agent execution (running on workers)
    - Result collection

    See AgentProvider documentation for implementation details.
"""

from .base import AgentProvider, AgentResult
from .documents import AgentOutputDocument
from .registry import (
    get_agent_provider,
    register_agent_provider,
    reset_agent_provider,
    temporary_provider,
)
from .runner import run_agent

__all__ = [
    # Documents
    "AgentOutputDocument",
    "AgentProvider",
    # Core types
    "AgentResult",
    "get_agent_provider",
    # Registry
    "register_agent_provider",
    "reset_agent_provider",
    # Execution
    "run_agent",
    "temporary_provider",
]
