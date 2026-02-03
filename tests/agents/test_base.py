"""Tests for ai_pipeline_core.agents.base module."""

import pytest
from dataclasses import FrozenInstanceError

from ai_pipeline_core.agents import AgentProvider, AgentResult


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_is_frozen(self, sample_result: AgentResult):
        """AgentResult should be immutable."""
        with pytest.raises(FrozenInstanceError):
            sample_result.success = False  # type: ignore

    def test_default_values(self):
        """Optional fields should have sensible defaults."""
        result = AgentResult(success=True)
        assert result.output == {}
        assert result.artifacts == {}
        assert result.error is None
        assert result.traceback is None
        assert result.exit_code is None
        assert result.duration_seconds is None
        assert result.stdout is None
        assert result.stderr is None
        assert result.agent_name is None
        assert result.agent_version is None

    def test_get_artifact_returns_decoded_string(self, sample_result: AgentResult):
        """get_artifact() should decode bytes to UTF-8 string."""
        content = sample_result.get_artifact("output.md")
        assert content == "# Report\n\nThis is the main output."
        assert isinstance(content, str)

    def test_get_artifact_with_custom_encoding(self):
        """get_artifact() should use specified encoding."""
        result = AgentResult(
            success=True,
            artifacts={"file.txt": "héllo wörld".encode("latin-1")},
        )
        content = result.get_artifact("file.txt", encoding="latin-1")
        assert content == "héllo wörld"

    def test_get_artifact_missing_returns_none(self, sample_result: AgentResult):
        """get_artifact() should return None for missing artifact."""
        assert sample_result.get_artifact("nonexistent.md") is None

    def test_get_artifact_bytes_returns_raw(self, sample_result: AgentResult):
        """get_artifact_bytes() should return raw bytes."""
        data = sample_result.get_artifact_bytes("output.md")
        assert data == b"# Report\n\nThis is the main output."
        assert isinstance(data, bytes)

    def test_get_artifact_bytes_missing_returns_none(self, sample_result: AgentResult):
        """get_artifact_bytes() should return None for missing artifact."""
        assert sample_result.get_artifact_bytes("nonexistent.md") is None

    def test_all_fields_preserved(self):
        """All fields should be accessible after creation."""
        result = AgentResult(
            success=False,
            output={"a": 1},
            artifacts={"file.txt": b"content"},
            error="Error msg",
            traceback="Traceback...",
            exit_code=2,
            duration_seconds=3.14,
            stdout="stdout output",
            stderr="stderr output",
            agent_name="my_agent",
            agent_version="2.0.0",
        )
        assert result.success is False
        assert result.output == {"a": 1}
        assert result.artifacts == {"file.txt": b"content"}
        assert result.error == "Error msg"
        assert result.traceback == "Traceback..."
        assert result.exit_code == 2
        assert result.duration_seconds == 3.14
        assert result.stdout == "stdout output"
        assert result.stderr == "stderr output"
        assert result.agent_name == "my_agent"
        assert result.agent_version == "2.0.0"


class TestAgentProvider:
    """Tests for AgentProvider ABC."""

    def test_is_abstract(self):
        """AgentProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            AgentProvider()  # type: ignore

    def test_run_is_abstract(self):
        """Subclass must implement run() method."""

        class IncompleteProvider(AgentProvider):
            pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteProvider()  # type: ignore

    async def test_validate_default_is_noop(self):
        """Default validate() implementation does nothing."""

        class MinimalProvider(AgentProvider):
            async def run(self, agent_name, inputs, **kwargs):
                return AgentResult(success=True)

        provider = MinimalProvider()
        # Should not raise
        await provider.validate()

    async def test_list_agents_default_returns_empty(self):
        """Default list_agents() returns empty list."""

        class MinimalProvider(AgentProvider):
            async def run(self, agent_name, inputs, **kwargs):
                return AgentResult(success=True)

        provider = MinimalProvider()
        agents = await provider.list_agents()
        assert agents == []

    async def test_concrete_provider_works(self):
        """A properly implemented provider should work."""

        class ConcreteProvider(AgentProvider):
            async def run(self, agent_name, inputs, **kwargs):
                return AgentResult(
                    success=True,
                    output={"agent": agent_name, "received": inputs},
                    agent_name=agent_name,
                )

            async def list_agents(self):
                return ["agent1", "agent2"]

        provider = ConcreteProvider()
        result = await provider.run("test", {"key": "value"})
        assert result.success
        assert result.output["agent"] == "test"
        assert await provider.list_agents() == ["agent1", "agent2"]
