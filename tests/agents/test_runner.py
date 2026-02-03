"""Tests for ai_pipeline_core.agents.runner module."""

import asyncio

import pytest

from ai_pipeline_core.agents import (
    AgentResult,
    register_agent_provider,
    run_agent,
)
from ai_pipeline_core.agents.runner import _validated_provider_ids

from .conftest import MockAgentProvider


class TestRunAgentDelegation:
    """Tests for run_agent() argument delegation."""

    async def test_delegates_to_provider(self, mock_provider: MockAgentProvider):
        """run_agent() should delegate all arguments to provider.run()."""
        register_agent_provider(mock_provider)

        await run_agent(
            "my_agent",
            {"input_key": "input_value"},
            files={"file.txt": "content"},
            target_worker="worker1",
            backend="codex",
            timeout_seconds=300,
            env_vars={"VAR": "value"},
        )

        assert len(mock_provider.run_calls) == 1
        call = mock_provider.run_calls[0]
        assert call["agent_name"] == "my_agent"
        assert call["inputs"] == {"input_key": "input_value"}
        assert call["files"] == {"file.txt": "content"}
        assert call["target_worker"] == "worker1"
        assert call["backend"] == "codex"
        assert call["timeout_seconds"] == 300
        assert call["env_vars"] == {"VAR": "value"}

    async def test_returns_provider_result(self, mock_provider: MockAgentProvider):
        """run_agent() should return the result from provider.run()."""
        register_agent_provider(mock_provider)

        expected = AgentResult(
            success=True,
            output={"result": 42},
            agent_name="test",
        )
        mock_provider.next_result = expected

        result = await run_agent("test", {})

        assert result is expected

    async def test_minimal_arguments(self, mock_provider: MockAgentProvider):
        """run_agent() should work with minimal arguments."""
        register_agent_provider(mock_provider)

        result = await run_agent("agent_name", {"key": "value"})

        assert result.success
        assert len(mock_provider.run_calls) == 1


class TestValidation:
    """Tests for lazy provider validation."""

    async def test_validates_on_first_call(self, mock_provider: MockAgentProvider):
        """validate() should be called on first run_agent() call."""
        register_agent_provider(mock_provider)

        await run_agent("test", {})

        assert mock_provider.validate_calls == 1

    async def test_validates_only_once(self, mock_provider: MockAgentProvider):
        """validate() should only be called once for repeated calls."""
        register_agent_provider(mock_provider)

        await run_agent("test1", {})
        await run_agent("test2", {})
        await run_agent("test3", {})

        assert mock_provider.validate_calls == 1

    async def test_concurrent_calls_validate_once(self, mock_provider: MockAgentProvider):
        """Concurrent run_agent() calls should only trigger one validation."""
        register_agent_provider(mock_provider)

        # Add delay to validation to increase chance of race
        original_validate = mock_provider.validate

        async def slow_validate():
            await asyncio.sleep(0.01)
            await original_validate()

        mock_provider.validate = slow_validate

        # Run 10 concurrent calls
        await asyncio.gather(*[run_agent(f"test{i}", {}) for i in range(10)])

        assert mock_provider.validate_calls == 1

    async def test_new_provider_is_validated(self, mock_provider: MockAgentProvider):
        """A newly registered provider should be validated."""
        register_agent_provider(mock_provider)
        await run_agent("test", {})
        assert mock_provider.validate_calls == 1

        # Clear tracked IDs to simulate fresh state
        _validated_provider_ids.clear()

        # Same provider, but tracking cleared - should validate again
        await run_agent("test", {})
        assert mock_provider.validate_calls == 2

    async def test_id_based_tracking(self, mock_provider: MockAgentProvider):
        """Validation tracking should use provider id."""
        register_agent_provider(mock_provider)
        provider_id = id(mock_provider)

        await run_agent("test", {})

        assert provider_id in _validated_provider_ids


class TestErrorHandling:
    """Tests for error handling in run_agent()."""

    async def test_without_provider_raises(self):
        """run_agent() should raise when no provider registered."""
        with pytest.raises(RuntimeError, match="No agent provider"):
            await run_agent("test", {})

    async def test_provider_exception_propagates(self, mock_provider: MockAgentProvider):
        """Exceptions from provider.run() should propagate."""
        register_agent_provider(mock_provider)

        async def failing_run(*args, **kwargs):
            raise ValueError("Provider error")

        mock_provider.run = failing_run  # type: ignore

        with pytest.raises(ValueError, match="Provider error"):
            await run_agent("test", {})

    async def test_validation_exception_propagates(self, mock_provider: MockAgentProvider):
        """Exceptions from provider.validate() should propagate."""
        register_agent_provider(mock_provider)

        async def failing_validate():
            raise RuntimeError("Validation failed")

        mock_provider.validate = failing_validate  # type: ignore

        with pytest.raises(RuntimeError, match="Validation failed"):
            await run_agent("test", {})


class TestLogging:
    """Tests for logging in run_agent()."""

    async def test_logs_success_with_duration(self, mock_provider: MockAgentProvider, caplog):
        """Successful run should log completion with duration."""
        register_agent_provider(mock_provider)
        mock_provider.next_result = AgentResult(
            success=True,
            duration_seconds=2.5,
            agent_name="test",
        )

        with caplog.at_level("INFO"):
            await run_agent("test", {})

        assert "completed successfully" in caplog.text
        assert "2.5s" in caplog.text

    async def test_logs_failure_with_error(self, mock_provider: MockAgentProvider, caplog):
        """Failed run should log warning with error."""
        register_agent_provider(mock_provider)
        mock_provider.next_result = AgentResult(
            success=False,
            error="Connection failed",
            agent_name="test",
        )

        with caplog.at_level("WARNING"):
            await run_agent("test", {})

        assert "failed" in caplog.text
        assert "Connection failed" in caplog.text

    async def test_logs_provider_name(self, mock_provider: MockAgentProvider, caplog):
        """Should log which provider is being used."""
        register_agent_provider(mock_provider)

        with caplog.at_level("INFO"):
            await run_agent("my_agent", {})

        assert "MockAgentProvider" in caplog.text
        assert "my_agent" in caplog.text
