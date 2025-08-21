"""Tracing utilities that integrate Laminar (``lmnr``) with our code-base.

This module centralises:
• ``TraceInfo`` - a small helper object for propagating contextual metadata.
• ``trace`` decorator - augments a callable with Laminar tracing, automatic
``observe`` instrumentation, and optional support for test runs.
"""

from __future__ import annotations

import inspect
import os
from functools import wraps
from typing import Any, Callable, Literal, ParamSpec, TypeVar, cast, overload

from lmnr import Instruments, Laminar, observe
from pydantic import BaseModel

from ai_pipeline_core.settings import settings

# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------
P = ParamSpec("P")
R = TypeVar("R")

TraceLevel = Literal["always", "debug", "off"]


# ---------------------------------------------------------------------------
# ``TraceInfo`` – metadata container
# ---------------------------------------------------------------------------
class TraceInfo(BaseModel):
    """A container that holds contextual metadata for the current trace."""

    session_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, str] = {}
    tags: list[str] = []

    def get_observe_kwargs(self) -> dict[str, Any]:
        """Return kwargs suitable for passing to the observe decorator."""
        kwargs: dict[str, Any] = {}

        # Use environment variable fallback for session_id
        session_id = self.session_id or os.getenv("LMNR_SESSION_ID")
        if session_id:
            kwargs["session_id"] = session_id

        # Use environment variable fallback for user_id
        user_id = self.user_id or os.getenv("LMNR_USER_ID")
        if user_id:
            kwargs["user_id"] = user_id

        if self.metadata:
            kwargs["metadata"] = self.metadata
        if self.tags:
            kwargs["tags"] = self.tags
        return kwargs


# ---------------------------------------------------------------------------
# ``trace`` decorator
# ---------------------------------------------------------------------------


def _initialise_laminar() -> None:
    """Ensure Laminar is initialised once per process."""
    if settings.lmnr_project_api_key:
        Laminar.initialize(
            project_api_key=settings.lmnr_project_api_key,
            disabled_instruments=[Instruments.OPENAI] if Instruments.OPENAI else [],
        )


# Overload for calls like @trace(name="...", level="debug")
@overload
def trace(
    *,
    level: TraceLevel = "always",
    name: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    span_type: str | None = None,
    ignore_input: bool = False,
    ignore_output: bool = False,
    ignore_inputs: list[str] | None = None,
    input_formatter: Callable[..., str] | None = None,
    output_formatter: Callable[..., str] | None = None,
    ignore_exceptions: bool = False,
    preserve_global_context: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


# Overload for the bare @trace call
@overload
def trace(func: Callable[P, R]) -> Callable[P, R]: ...


# Actual implementation
def trace(
    func: Callable[P, R] | None = None,
    *,
    level: TraceLevel = "always",
    name: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    span_type: str | None = None,
    ignore_input: bool = False,
    ignore_output: bool = False,
    ignore_inputs: list[str] | None = None,
    input_formatter: Callable[..., str] | None = None,
    output_formatter: Callable[..., str] | None = None,
    ignore_exceptions: bool = False,
    preserve_global_context: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    """Decorator that wires Laminar tracing and observation into a function.

    Args:
        func: The function to be traced (when used as @trace)
        level: Trace level control:
            - "always": Always trace (default)
            - "debug": Only trace when LMNR_DEBUG environment variable is NOT set to "true"
            - "off": Never trace
        name: Custom name for the observation (defaults to function name)
        metadata: Additional metadata for the trace
        tags: Additional tags for the trace
        span_type: Type of span for the trace
        ignore_input: Ignore all inputs in the trace
        ignore_output: Ignore the output in the trace
        ignore_inputs: List of specific input parameter names to ignore
        input_formatter: Custom formatter for inputs (takes any arguments, returns string)
        output_formatter: Custom formatter for outputs (takes any arguments, returns string)
        ignore_exceptions: Whether to ignore exceptions in tracing
        preserve_global_context: Whether to preserve global context

    Returns:
        The decorated function with Laminar tracing enabled
    """

    if level == "off":
        if func:
            return func
        return lambda f: f

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        # Handle 'debug' level logic - only trace when LMNR_DEBUG is NOT "true"
        if level == "debug" and os.getenv("LMNR_DEBUG", "").lower() == "true":
            return f

        # --- Pre-computation (done once when the function is decorated) ---
        _initialise_laminar()
        sig = inspect.signature(f)
        is_coroutine = inspect.iscoroutinefunction(f)
        observe_name = name or f.__name__
        _observe = observe

        # Store the new parameters
        _session_id = session_id
        _user_id = user_id
        _metadata = metadata
        _tags = tags or []
        _span_type = span_type
        _ignore_input = ignore_input
        _ignore_output = ignore_output
        _ignore_inputs = ignore_inputs
        _input_formatter = input_formatter
        _output_formatter = output_formatter
        _ignore_exceptions = ignore_exceptions
        _preserve_global_context = preserve_global_context

        # --- Helper function for runtime logic ---
        def _prepare_and_get_observe_params(runtime_kwargs: dict[str, Any]) -> dict[str, Any]:
            """
            Inspects runtime args, manages TraceInfo, and returns params for lmnr.observe.
            Modifies runtime_kwargs in place to inject TraceInfo if the function expects it.
            """
            trace_info = runtime_kwargs.get("trace_info")
            if not isinstance(trace_info, TraceInfo):
                trace_info = TraceInfo()
                if "trace_info" in sig.parameters:
                    runtime_kwargs["trace_info"] = trace_info

            observe_params = trace_info.get_observe_kwargs()
            observe_params["name"] = observe_name

            # Override with decorator-level session_id and user_id if provided
            if _session_id:
                observe_params["session_id"] = _session_id
            if _user_id:
                observe_params["user_id"] = _user_id

            # Merge decorator-level metadata and tags
            if _metadata:
                observe_params["metadata"] = {**observe_params.get("metadata", {}), **_metadata}
            if _tags:
                observe_params["tags"] = observe_params.get("tags", []) + _tags
            if _span_type:
                observe_params["span_type"] = _span_type

            # Add the new Laminar parameters
            if _ignore_input:
                observe_params["ignore_input"] = _ignore_input
            if _ignore_output:
                observe_params["ignore_output"] = _ignore_output
            if _ignore_inputs is not None:
                observe_params["ignore_inputs"] = _ignore_inputs
            if _input_formatter is not None:
                observe_params["input_formatter"] = _input_formatter
            if _output_formatter is not None:
                observe_params["output_formatter"] = _output_formatter
            if _ignore_exceptions:
                observe_params["ignore_exceptions"] = _ignore_exceptions
            if _preserve_global_context:
                observe_params["preserve_global_context"] = _preserve_global_context

            return observe_params

        # --- The actual wrappers ---
        @wraps(f)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            observe_params = _prepare_and_get_observe_params(kwargs)
            observed_func = _observe(**observe_params)(f)
            return observed_func(*args, **kwargs)

        @wraps(f)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            observe_params = _prepare_and_get_observe_params(kwargs)
            observed_func = _observe(**observe_params)(f)
            return await observed_func(*args, **kwargs)

        wrapper = async_wrapper if is_coroutine else sync_wrapper

        # Preserve the original function signature
        try:
            wrapper.__signature__ = sig  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

        return cast(Callable[P, R], wrapper)

    if func:
        return decorator(func)  # Called as @trace
    else:
        return decorator  # Called as @trace(...)


__all__ = ["trace", "TraceLevel", "TraceInfo"]
