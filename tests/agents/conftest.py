"""Shared fixtures for agents module tests."""

import pytest

from typing import Any

from ai_pipeline_core.agents import AgentProvider, AgentResult, reset_agent_provider
from ai_pipeline_core.agents.runner import _validated_provider_ids


class MockAgentProvider(AgentProvider):
    """Mock provider for testing with call tracking."""

    def __init__(self, default_result: AgentResult | None = None):
        self.run_calls: list[dict[str, Any]] = []
        self.validate_calls = 0
        self.default_result = default_result or AgentResult(success=True)
        self.next_result: AgentResult | None = None

    async def run(
        self,
        agent_name: str,
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> AgentResult:
        self.run_calls.append({"agent_name": agent_name, "inputs": inputs, **kwargs})
        return self.next_result or self.default_result

    async def validate(self) -> None:
        self.validate_calls += 1

    async def list_agents(self) -> list[str]:
        return ["test_agent_a", "test_agent_b"]


@pytest.fixture
def mock_provider() -> MockAgentProvider:
    """Create a mock provider instance."""
    return MockAgentProvider()


@pytest.fixture
def sample_result() -> AgentResult:
    """Successful agent result with artifacts."""
    return AgentResult(
        success=True,
        output={"key": "value", "count": 42},
        artifacts={
            "output.md": b"# Report\n\nThis is the main output.",
            "data.json": b'{"result": 42}',
            "nested/file.txt": b"nested content",
        },
        duration_seconds=2.5,
        agent_name="test_agent",
        agent_version="1.0.0",
    )


@pytest.fixture
def failed_result() -> AgentResult:
    """Failed agent result with error info."""
    return AgentResult(
        success=False,
        output={},
        artifacts={},
        error="Connection timeout",
        traceback="Traceback (most recent call last):\n  File ...\nTimeoutError: Connection timeout",
        stderr="ERROR: Failed to connect to worker",
        exit_code=1,
        agent_name="test_agent",
    )


@pytest.fixture(autouse=True)
def reset_registry_and_validation():
    """Reset registry and validation tracking before and after each test."""
    reset_agent_provider()
    _validated_provider_ids.clear()
    yield
    reset_agent_provider()
    _validated_provider_ids.clear()
