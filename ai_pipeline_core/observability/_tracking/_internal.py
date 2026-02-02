"""Thread-local flag to prevent tracking recursion.

When summary generation calls ``llm.generate()``, the resulting span must NOT
be tracked again (infinite loop).  The flag is checked by
``TrackingSpanProcessor.on_end()``.
"""

import threading
from collections.abc import Generator
from contextlib import contextmanager

_internal = threading.local()


def is_internal_tracking() -> bool:
    """Return True if the current thread is inside a tracking-internal LLM call."""
    return getattr(_internal, "active", False)


@contextmanager
def internal_tracking_context() -> Generator[None, None, None]:
    """Mark the current thread as performing internal tracking work."""
    prev = getattr(_internal, "active", False)
    _internal.active = True
    try:
        yield
    finally:
        _internal.active = prev
