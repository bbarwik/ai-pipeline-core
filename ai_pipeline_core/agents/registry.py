"""Thread-safe agent provider registry.

The registry follows a singleton pattern - only one provider can be
registered at a time. This simplifies the mental model and avoids
questions about which provider handles which agent.

For testing, use reset_agent_provider() or the temporary_provider()
context manager to safely swap providers.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from .base import AgentProvider

__all__ = [
    "get_agent_provider",
    "register_agent_provider",
    "reset_agent_provider",
    "temporary_provider",
]

# Global state protected by lock
_provider: AgentProvider | None = None
_lock = threading.Lock()


def register_agent_provider(provider: AgentProvider) -> None:
    """Register an agent provider.

    Call once at application startup, typically in __init__.py.
    Raises if a provider is already registered to prevent accidental
    overwrites. Use reset_agent_provider() first if you need to replace.

    Args:
        provider: The AgentProvider implementation to register

    Raises:
        RuntimeError: If a provider is already registered
    """
    global _provider
    with _lock:
        if _provider is not None:
            raise RuntimeError(f"Agent provider already registered: {type(_provider).__name__}. Call reset_agent_provider() first to replace it.")
        _provider = provider


def get_agent_provider() -> AgentProvider:
    """Get the registered agent provider.

    Returns:
        The registered AgentProvider instance

    Raises:
        RuntimeError: If no provider is registered, with instructions
            on how to register one (generic, no specific package names)
    """
    with _lock:
        if _provider is None:
            raise RuntimeError(
                "No agent provider registered.\n\n"
                "To use agents, install an agent provider package and register it.\n"
                "See tests/agents/conftest.py for example of provider registration."
            )
        return _provider


def reset_agent_provider() -> None:
    """Reset the provider registration.

    Primarily for testing - allows registering a new provider.
    In production code, providers should be registered once at startup.
    """
    global _provider
    with _lock:
        _provider = None


@contextmanager
def temporary_provider(provider: AgentProvider) -> Iterator[AgentProvider]:
    """Context manager for temporarily registering a provider.

    Useful for testing - registers the provider, yields it, then
    restores the previous registration on exit (even if an exception occurs).

    Args:
        provider: The provider to temporarily register

    Yields:
        The registered provider
    """
    global _provider
    with _lock:
        previous_provider = _provider
        _provider = provider
    try:
        yield provider
    finally:
        with _lock:
            _provider = previous_provider
