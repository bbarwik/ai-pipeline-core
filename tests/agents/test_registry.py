"""Tests for ai_pipeline_core.agents.registry module."""

import concurrent.futures
import threading

import pytest

from ai_pipeline_core.agents import (
    AgentProvider,
    AgentResult,
    get_agent_provider,
    register_agent_provider,
    reset_agent_provider,
    temporary_provider,
)

from .conftest import MockAgentProvider


class TestRegisterAndGetProvider:
    """Tests for register_agent_provider() and get_agent_provider()."""

    def test_register_and_get_provider(self, mock_provider: MockAgentProvider):
        """Registered provider should be retrievable."""
        register_agent_provider(mock_provider)
        retrieved = get_agent_provider()
        assert retrieved is mock_provider

    def test_get_without_registration_raises(self):
        """get_agent_provider() should raise with helpful message when nothing registered."""
        with pytest.raises(RuntimeError, match="No agent provider registered"):
            get_agent_provider()

    def test_error_message_is_helpful(self):
        """Error message should guide user without mentioning specific packages."""
        try:
            get_agent_provider()
            pytest.fail("Should have raised RuntimeError")
        except RuntimeError as e:
            msg = str(e)
            assert "register" in msg.lower()
            assert "example" in msg.lower()
            # Should NOT mention specific private packages
            assert "cli_agents" not in msg
            assert "cli-agents" not in msg

    def test_register_twice_raises(self, mock_provider: MockAgentProvider):
        """Registering a second provider without reset should raise."""
        register_agent_provider(mock_provider)

        another_provider = MockAgentProvider()
        with pytest.raises(RuntimeError, match="already registered"):
            register_agent_provider(another_provider)

    def test_error_shows_existing_provider_type(self, mock_provider: MockAgentProvider):
        """Error should mention the type of already-registered provider."""
        register_agent_provider(mock_provider)

        try:
            register_agent_provider(MockAgentProvider())
            pytest.fail("Should have raised RuntimeError")
        except RuntimeError as e:
            assert "MockAgentProvider" in str(e)


class TestResetProvider:
    """Tests for reset_agent_provider()."""

    def test_reset_clears_provider(self, mock_provider: MockAgentProvider):
        """reset_agent_provider() should clear the registration."""
        register_agent_provider(mock_provider)
        reset_agent_provider()

        with pytest.raises(RuntimeError):
            get_agent_provider()

    def test_reset_allows_new_registration(self, mock_provider: MockAgentProvider):
        """After reset, a new provider can be registered."""
        register_agent_provider(mock_provider)
        reset_agent_provider()

        another = MockAgentProvider()
        register_agent_provider(another)
        assert get_agent_provider() is another

    def test_reset_when_empty_is_noop(self):
        """reset_agent_provider() on empty registry should not raise."""
        reset_agent_provider()  # Should not raise
        reset_agent_provider()  # Still should not raise


class TestTemporaryProvider:
    """Tests for temporary_provider() context manager."""

    def test_registers_and_yields_provider(self, mock_provider: MockAgentProvider):
        """temporary_provider() should register and yield the provider."""
        with temporary_provider(mock_provider) as p:
            assert p is mock_provider
            assert get_agent_provider() is mock_provider

    def test_restores_to_none_when_nothing_registered(self, mock_provider: MockAgentProvider):
        """After context exits, should restore to None if nothing was registered."""
        with temporary_provider(mock_provider):
            pass

        with pytest.raises(RuntimeError):
            get_agent_provider()

    def test_restores_previous_provider(self, mock_provider: MockAgentProvider):
        """After context exits, should restore the previous provider."""
        original = MockAgentProvider()
        register_agent_provider(original)

        with temporary_provider(mock_provider):
            assert get_agent_provider() is mock_provider

        # Should restore original
        assert get_agent_provider() is original

    def test_restores_on_exception(self, mock_provider: MockAgentProvider):
        """Should restore previous provider even if exception raised."""
        original = MockAgentProvider()
        register_agent_provider(original)

        with pytest.raises(ValueError):
            with temporary_provider(mock_provider):
                assert get_agent_provider() is mock_provider
                raise ValueError("test error")

        # Should still restore original
        assert get_agent_provider() is original

    def test_nested_temporary_providers(self):
        """Nested temporary_provider() should work correctly."""
        provider_a = MockAgentProvider()
        provider_b = MockAgentProvider()
        provider_c = MockAgentProvider()

        register_agent_provider(provider_a)

        with temporary_provider(provider_b):
            assert get_agent_provider() is provider_b

            with temporary_provider(provider_c):
                assert get_agent_provider() is provider_c

            assert get_agent_provider() is provider_b

        assert get_agent_provider() is provider_a


class TestThreadSafety:
    """Tests for thread safety of the registry."""

    def test_concurrent_registrations_only_one_succeeds(self):
        """When multiple threads try to register, exactly one should succeed."""
        results = {"success": 0, "error": 0}
        barrier = threading.Barrier(10)

        def try_register(provider_id: int):
            class NumberedProvider(AgentProvider):
                id = provider_id

                async def run(self, agent_name, inputs, **kwargs):
                    return AgentResult(success=True)

            barrier.wait()  # Ensure all threads start at the same time
            try:
                register_agent_provider(NumberedProvider())
                results["success"] += 1
            except RuntimeError:
                results["error"] += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(try_register, i) for i in range(10)]
            concurrent.futures.wait(futures)

        assert results["success"] == 1
        assert results["error"] == 9

    def test_concurrent_get_is_safe(self, mock_provider: MockAgentProvider):
        """Concurrent calls to get_agent_provider() should be safe."""
        register_agent_provider(mock_provider)
        results = []

        def get_provider():
            for _ in range(100):
                p = get_agent_provider()
                results.append(p is mock_provider)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_provider) for _ in range(5)]
            concurrent.futures.wait(futures)

        assert all(results)
        assert len(results) == 500
