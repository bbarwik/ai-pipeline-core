"""Tracing utilities that integrate Laminar (``lmnr``) with our code-base.

@public

This module centralizes:
- ``TraceInfo`` - a small helper object for propagating contextual metadata.
- ``trace`` decorator - augments a callable with Laminar tracing, automatic
  ``observe`` instrumentation, and optional support for test runs.
"""

from __future__ import annotations

import inspect
import os
from functools import wraps
from typing import Any, Callable, Literal, ParamSpec, TypeVar, cast, overload

from lmnr import Attributes, Instruments, Laminar, observe
from pydantic import BaseModel

from ai_pipeline_core.settings import settings

# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------
P = ParamSpec("P")
R = TypeVar("R")

TraceLevel = Literal["always", "debug", "off"]
"""Control level for tracing activation.

@public

Values:
- "always": Always trace (default, production mode)
- "debug": Only trace when LMNR_DEBUG == "true"
- "off": Disable tracing completely
"""


# ---------------------------------------------------------------------------
# ``TraceInfo`` – metadata container
# ---------------------------------------------------------------------------
class TraceInfo(BaseModel):
    """Container for propagating trace context through the pipeline.

    TraceInfo provides a structured way to pass tracing metadata through
    function calls, ensuring consistent observability across the entire
    execution flow. It integrates with Laminar (LMNR) for distributed
    tracing and debugging.

    Attributes:
        session_id: Unique identifier for the current session/conversation.
                   Falls back to LMNR_SESSION_ID environment variable.
        user_id: Identifier for the user triggering the operation.
                Falls back to LMNR_USER_ID environment variable.
        metadata: Key-value pairs for additional trace context.
                 Useful for filtering and searching in LMNR dashboard.
        tags: List of tags for categorizing traces (e.g., ["production", "v2"]).

    Environment fallbacks:
        - LMNR_SESSION_ID: Default session_id if not explicitly set
        - LMNR_USER_ID: Default user_id if not explicitly set
        - LMNR_DEBUG: Controls debug-level tracing when set to "true"

        Note: These variables are read directly by the tracing layer and are
        not part of the Settings configuration.

    Example:
        >>> # Create trace context
        >>> trace_info = TraceInfo(
        ...     session_id="sess_123",
        ...     user_id="user_456",
        ...     metadata={"flow": "document_analysis", "version": "1.2"},
        ...     tags=["production", "high_priority"]
        ... )
        >>>
        >>> # Pass through function calls
        >>> @trace
        >>> async def process(data, trace_info: TraceInfo):
        ...     # TraceInfo automatically propagates to nested calls
        ...     result = await analyze(data, trace_info=trace_info)
        ...     return result

    Note:
        TraceInfo is typically created at the entry point of a flow
        and passed through all subsequent function calls for
        consistent tracing context.
    """

    session_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, str] = {}
    tags: list[str] = []

    def get_observe_kwargs(self) -> dict[str, Any]:
        """Convert TraceInfo to kwargs for Laminar's observe decorator.

        Transforms the TraceInfo fields into the format expected by
        the lmnr.observe() decorator, applying environment variable
        fallbacks for session_id and user_id.

        Returns:
            Dictionary with keys:
            - session_id: From field or LMNR_SESSION_ID env var
            - user_id: From field or LMNR_USER_ID env var
            - metadata: Dictionary of custom metadata (if set)
            - tags: List of tags (if set)

            Only non-empty values are included in the output.

        Example:
            >>> trace_info = TraceInfo(session_id="sess_123", tags=["test"])
            >>> kwargs = trace_info.get_observe_kwargs()
            >>> # Returns: {"session_id": "sess_123", "tags": ["test"]}

        Note:
            This method is called internally by the trace decorator
            to configure Laminar observation parameters.
        """
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
    """Initialize Laminar SDK with project configuration.

    Sets up the Laminar observability client with the project API key
    from settings. Disables automatic OpenAI instrumentation to avoid
    conflicts with our custom tracing.

    Configuration:
        - Uses settings.lmnr_project_api_key for authentication
        - Disables OPENAI instrument to prevent double-tracing
        - Called automatically by trace decorator on first use

    Note:
        This is an internal function called once per process.
        Multiple calls are safe (Laminar handles idempotency).
    """
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
    """Add Laminar observability tracing to any function.

    @public

    The trace decorator integrates functions with Laminar (LMNR) for
    distributed tracing, performance monitoring, and debugging. It
    automatically handles both sync and async functions, propagates
    trace context, and provides fine-grained control over what gets traced.

    USAGE GUIDELINE - Defaults First:
        In 90% of cases, use WITHOUT any parameters.
        The defaults are optimized for most use cases.

    Args:
        func: Function to trace (when used without parentheses: @trace).

        level: Controls when tracing is active:
            - "always": Always trace (default, production mode)
            - "debug": Only trace when LMNR_DEBUG == "true"
            - "off": Disable tracing completely

        name: Custom span name in traces (defaults to function.__name__).
             Use descriptive names for better trace readability.

        session_id: Override session ID for this function's traces.
                   Typically propagated via TraceInfo instead.

        user_id: Override user ID for this function's traces.
                Typically propagated via TraceInfo instead.

        metadata: Additional key-value metadata attached to spans.
                 Searchable in LMNR dashboard. Merged with TraceInfo metadata.

        tags: List of tags for categorizing spans (e.g., ["api", "critical"]).
             Merged with TraceInfo tags.

        span_type: Semantic type of the span (e.g., "LLM", "CHAIN", "TOOL").
                  Affects visualization in LMNR dashboard.

        ignore_input: Don't record function inputs in trace (privacy/size).

        ignore_output: Don't record function output in trace (privacy/size).

        ignore_inputs: List of parameter names to exclude from trace.
                      Useful for sensitive data like API keys.

        input_formatter: Custom function to format inputs for tracing.
                        Receives all function args, returns display string.

        output_formatter: Custom function to format output for tracing.
                         Receives function result, returns display string.

        ignore_exceptions: Don't record exceptions in traces (default False).

        preserve_global_context: Maintain Laminar's global context across
                                calls (default True). Set False for isolated traces.

    Returns:
        Decorated function with same signature but added tracing.

    TraceInfo propagation:
        If the decorated function has a 'trace_info' parameter, the decorator
        automatically creates or propagates a TraceInfo instance, ensuring
        consistent session/user tracking across the call chain.

    Example:
        >>> # RECOMMENDED - No parameters needed for most cases!
        >>> @trace
        >>> async def process_document(doc):
        ...     return await analyze(doc)
        >>>
        >>> # With parameters (RARE - only when specifically needed):
        >>> @trace(level="debug")  # Only for debug-specific tracing
        >>> async def debug_operation():
        ...     pass

        >>> @trace(ignore_inputs=["api_key"])  # Only for sensitive data
        >>> async def api_call(data, api_key):
        ...     return await external_api(data, api_key)
        >>>
        >>> # AVOID unnecessary configuration - defaults handle:
        >>> # - Automatic naming from function name
        >>> # - Standard trace level ("always")
        >>> # - Full input/output capture
        >>> # - Proper span type inference
        >>>
        >>> # Custom formatting
        >>> @trace(
        ...     input_formatter=lambda doc: f"Document: {doc.id}",
        ...     output_formatter=lambda res: f"Results: {len(res)} items"
        >>> )
        >>> def analyze(doc):
        ...     return results

    Environment variables:
        - LMNR_DEBUG: Set to "true" to enable debug-level traces
        - LMNR_SESSION_ID: Default session ID if not in TraceInfo
        - LMNR_USER_ID: Default user ID if not in TraceInfo
        - LMNR_PROJECT_API_KEY: Required for trace submission

    Performance:
        - Tracing overhead is minimal (~1-2ms per call)
        - When level="off", decorator returns original function unchanged
        - Large inputs/outputs can be excluded with ignore_* parameters

    Note:
        - Automatically initializes Laminar on first use
        - Works with both sync and async functions
        - Preserves function signature and metadata
        - Thread-safe and async-safe

    See Also:
        - TraceInfo: Container for trace metadata
        - pipeline_task: Task decorator with built-in tracing
        - pipeline_flow: Flow decorator with built-in tracing
    """
    if level == "off":
        if func:
            return func
        return lambda f: f

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        """Apply tracing to the target function.

        Returns:
            Wrapped function with LMNR observability.

        Raises:
            TypeError: If function is already decorated with @pipeline_task or @pipeline_flow.
        """
        # Check if this is already a traced pipeline_task or pipeline_flow
        # This happens when @trace is applied after @pipeline_task/@pipeline_flow
        if hasattr(f, "__is_traced__") and f.__is_traced__:  # type: ignore[attr-defined]
            # Check if it's a Prefect Task or Flow object (they have specific attributes)
            # Prefect objects have certain attributes that regular functions don't
            is_prefect_task = hasattr(f, "fn") and hasattr(f, "submit") and hasattr(f, "map")
            is_prefect_flow = hasattr(f, "fn") and hasattr(f, "serve")
            if is_prefect_task or is_prefect_flow:
                fname = getattr(f, "__name__", "function")
                raise TypeError(
                    f"Function '{fname}' is already decorated with @pipeline_task or "
                    f"@pipeline_flow. Remove the @trace decorator - pipeline decorators "
                    f"include tracing automatically."
                )

        # Handle 'debug' level logic - only trace when LMNR_DEBUG is "true"
        debug_value = settings.lmnr_debug or os.getenv("LMNR_DEBUG", "")
        if level == "debug" and debug_value.lower() != "true":
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
        _metadata = metadata or {}
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
            """Inspects runtime args, manages TraceInfo, and returns params for lmnr.observe.

            Modifies runtime_kwargs in place to inject TraceInfo if the function expects it.

            Returns:
                Dictionary of parameters for lmnr.observe decorator.
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
            if _metadata:
                observe_params["metadata"] = _metadata
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
            """Synchronous wrapper for traced function.

            Returns:
                The result of the wrapped function.
            """
            observe_params = _prepare_and_get_observe_params(kwargs)
            observed_func = _observe(**observe_params)(f)
            return observed_func(*args, **kwargs)

        @wraps(f)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """Asynchronous wrapper for traced function.

            Returns:
                The result of the wrapped function.
            """
            observe_params = _prepare_and_get_observe_params(kwargs)
            observed_func = _observe(**observe_params)(f)
            return await observed_func(*args, **kwargs)

        wrapper = async_wrapper if is_coroutine else sync_wrapper

        # Mark function as traced for detection by pipeline decorators
        wrapper.__is_traced__ = True  # type: ignore[attr-defined]

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


def set_trace_cost(cost: float | str) -> None:
    """Set cost attributes for the current trace span.

    Sets cost metadata in the current LMNR trace span for tracking expenses
    of custom operations. This function should be called within a traced
    function to dynamically set or update the cost associated with the
    current operation. Particularly useful for tracking costs of external
    API calls, compute resources, or custom billing scenarios.

    The cost is stored in three metadata fields for compatibility:
    - gen_ai.usage.output_cost: Standard OpenAI cost field
    - gen_ai.usage.cost: Alternative cost field
    - cost: Simple cost field

    Args:
        cost: The cost value to set. Can be:
              - float: Cost in dollars (e.g., 0.05 for 5 cents)
              - str: USD format with dollar sign (e.g., "$0.05" or "$1.25")
              Only positive values will be set; zero or negative values are ignored.

    Example:
        >>> # Track cost of external API call
        >>> @trace
        >>> async def call_translation_api(text: str) -> str:
        ...     # External API charges per character
        ...     char_count = len(text)
        ...     cost_per_char = 0.00001  # $0.00001 per character
        ...
        ...     result = await external_api.translate(text)
        ...
        ...     # Set the cost for this operation
        ...     set_trace_cost(char_count * cost_per_char)
        ...     return result
        >>>
        >>> # Track compute resource costs
        >>> @trace
        >>> def process_video(video_path: str) -> dict:
        ...     duration = get_video_duration(video_path)
        ...     cost_per_minute = 0.10  # $0.10 per minute
        ...
        ...     result = process_video_content(video_path)
        ...
        ...     # Set cost using string format
        ...     set_trace_cost(f"${duration * cost_per_minute:.2f}")
        ...     return result
        >>>
        >>> # Combine with LLM costs in pipeline
        >>> @pipeline_task
        >>> async def enriched_generation(prompt: str) -> str:
        ...     # LLM cost tracked automatically via ModelResponse
        ...     response = await llm.generate("gpt-5", messages=prompt)
        ...
        ...     # Add cost for post-processing
        ...     processing_cost = 0.02  # Fixed cost for enrichment
        ...     set_trace_cost(processing_cost)
        ...
        ...     return enrich_response(response.content)

    Raises:
        ValueError: If string format is invalid (not a valid USD amount).

    Note:
        - This function only works within a traced context (function decorated
          with @trace, @pipeline_task, or @pipeline_flow)
        - LLM costs are tracked automatically via ModelResponse; use this for non-LLM costs
        - Cost should be a positive number representing actual monetary cost in USD
        - The cost is added to the current span's attributes/metadata
        - Multiple calls overwrite the previous cost (not cumulative)
        - If called outside a traced context (no active span), it has no effect
          and does not raise an error

    See Also:
        - trace: Decorator for adding tracing to functions
        - ModelResponse.get_laminar_metadata: Access LLM generation costs
        - pipeline_task: Task decorator with built-in tracing and optional trace_cost parameter
        - pipeline_flow: Flow decorator with built-in tracing and optional trace_cost parameter
    """
    # Parse string format if provided
    if isinstance(cost, str):
        # Remove dollar sign and any whitespace
        cost_str = cost.strip()
        if not cost_str.startswith("$"):
            raise ValueError(f"Invalid USD format: {cost!r}. Must start with '$' (e.g., '$0.50')")

        try:
            # Remove $ and convert to float
            cost_value = float(cost_str[1:])
        except ValueError as e:
            raise ValueError(
                f"Invalid USD format: {cost!r}. Must be a valid number after '$'"
            ) from e
    else:
        cost_value = cost

    if cost_value > 0:
        # Build the attributes dictionary with cost metadata
        attributes: dict[Attributes | str, float] = {
            "gen_ai.usage.output_cost": cost_value,
            "gen_ai.usage.cost": cost_value,
            "cost": cost_value,
        }

        try:
            Laminar.set_span_attributes(attributes)
        except Exception:
            # Silently ignore if not in a traced context
            pass


__all__ = ["trace", "TraceLevel", "TraceInfo", "set_trace_cost"]
