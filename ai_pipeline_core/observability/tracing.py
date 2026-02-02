"""Tracing utilities that integrate Laminar (``lmnr``) with our code-base.

This module centralizes:
- ``TraceInfo`` - a small helper object for propagating contextual metadata.
- ``trace`` decorator - augments a callable with Laminar tracing, automatic
  ``observe`` instrumentation, and optional support for test runs.
"""

import contextlib
import inspect
import json
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal, ParamSpec, TypeVar, cast, overload

from lmnr import Attributes, Instruments, Laminar, observe
from pydantic import BaseModel, Field

from ai_pipeline_core.documents import Document
from ai_pipeline_core.llm import AIMessages, ModelResponse
from ai_pipeline_core.settings import settings

# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------
P = ParamSpec("P")
R = TypeVar("R")

TraceLevel = Literal["always", "debug", "off"]
"""Control level for tracing activation.

Values:
- "always": Always trace (default, production mode)
- "debug": Only trace when LMNR_DEBUG == "true"
- "off": Disable tracing completely
"""


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _serialize_for_tracing(obj: Any) -> Any:  # noqa: PLR0911
    """Convert objects to JSON-serializable format for tracing."""
    if isinstance(obj, Document):
        return obj.serialize_model()
    if isinstance(obj, list) and obj and isinstance(obj[0], Document):
        return [doc.serialize_model() for doc in cast(list[Document], obj)]
    if isinstance(obj, AIMessages):
        result: list[Any] = []
        for msg in obj:
            if isinstance(msg, Document):
                result.append(msg.serialize_model())
            else:
                result.append(msg)
        return result
    if isinstance(obj, ModelResponse):
        return obj.model_dump()
    if isinstance(obj, BaseModel):
        data: dict[str, Any] = {}
        for field_name, field_value in obj.__dict__.items():
            if isinstance(field_value, Document):
                data[field_name] = field_value.serialize_model()
            elif isinstance(field_value, BaseModel):
                data[field_name] = _serialize_for_tracing(field_value)
            else:
                data[field_name] = field_value
        return data
    try:
        return str(obj)  # pyright: ignore[reportUnknownArgumentType]
    except Exception:
        return f"<{type(obj).__name__}>"  # pyright: ignore[reportUnknownArgumentType]


# ---------------------------------------------------------------------------
# Document trimming utilities
# ---------------------------------------------------------------------------
def _trim_attachment_list(attachments: list[Any]) -> list[Any]:
    """Trim attachment content in a serialized attachment list.

    Always trims regardless of parent document type:
    - Binary (base64): replace content with placeholder
    - Text > 250 chars: keep first 100 + last 100
    """
    trimmed: list[Any] = []
    for raw_att in attachments:
        if not isinstance(raw_att, dict):
            trimmed.append(raw_att)
            continue
        att: dict[str, Any] = cast(dict[str, Any], raw_att)
        content_encoding: str = att.get("content_encoding", "utf-8")
        if content_encoding == "base64":
            att = att.copy()
            att["content"] = "[binary content removed]"
        elif isinstance(att.get("content"), str) and len(att["content"]) > 250:
            att = att.copy()
            c: str = att["content"]
            trimmed_chars = len(c) - 200
            att["content"] = c[:100] + f" ... [trimmed {trimmed_chars} chars] ... " + c[-100:]
        trimmed.append(att)
    return trimmed


def _trim_document_content(doc_dict: dict[str, Any]) -> dict[str, Any]:
    """Trim document content for traces. All documents trimmed equally."""
    if not isinstance(doc_dict, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        return doc_dict  # pyright: ignore[reportUnreachable]

    if "content" not in doc_dict or "class_name" not in doc_dict:
        return doc_dict

    doc_dict = doc_dict.copy()
    content = doc_dict.get("content", "")
    content_encoding = doc_dict.get("content_encoding", "utf-8")

    # Trim attachments
    if "attachments" in doc_dict and isinstance(doc_dict["attachments"], list):
        doc_dict["attachments"] = _trim_attachment_list(cast(list[Any], doc_dict["attachments"]))

    # Binary: remove content
    if content_encoding == "base64":
        doc_dict["content"] = "[binary content removed]"
        return doc_dict

    # Text: trim if > 250 chars
    if isinstance(content, str) and len(content) > 250:
        trimmed_chars = len(content) - 200
        doc_dict["content"] = content[:100] + f" ... [trimmed {trimmed_chars} chars] ... " + content[-100:]

    return doc_dict


def _trim_documents_in_data(data: Any) -> Any:
    """Recursively trim document content in nested data structures."""
    if isinstance(data, dict):
        data_dict = cast(dict[str, Any], data)
        if "class_name" in data_dict and "content" in data_dict:
            return _trim_document_content(data_dict)
        return {k: _trim_documents_in_data(v) for k, v in data_dict.items()}
    if isinstance(data, list):
        return [_trim_documents_in_data(item) for item in cast(list[Any], data)]
    if isinstance(data, tuple):
        return tuple(_trim_documents_in_data(item) for item in cast(tuple[Any, ...], data))
    return data


# ---------------------------------------------------------------------------
# ``TraceInfo`` - metadata container
# ---------------------------------------------------------------------------
class TraceInfo(BaseModel):
    """Container for propagating trace context through the pipeline.

    TraceInfo provides a structured way to pass tracing metadata through
    function calls, ensuring consistent observability across the entire
    execution flow. It integrates with Laminar (LMNR) for distributed
    tracing and debugging.

    Attributes:
        session_id: Unique identifier for the current session/conversation.
        user_id: Identifier for the user triggering the operation.
        metadata: Key-value pairs for additional trace context.
                 Useful for filtering and searching in LMNR dashboard.
        tags: List of tags for categorizing traces (e.g., ["production", "v2"]).

    Environment fallbacks:
        - LMNR_DEBUG: Controls debug-level tracing when set to "true"
        These variables are read directly by the tracing layer and are
        not part of the Settings configuration.

    TraceInfo is typically created at the entry point of a flow
    and passed through all subsequent function calls for
    consistent tracing context.
    """

    session_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    def get_observe_kwargs(self) -> dict[str, Any]:
        """Convert TraceInfo to kwargs for Laminar's observe decorator.

        Transforms the TraceInfo fields into the format expected by
        the lmnr.observe() decorator, applying environment variable
        fallbacks for session_id and user_id.

        Returns:
            Dictionary with keys:
            - session_id: From field or environment variable fallback
            - user_id: From field or environment variable fallback
            - metadata: Dictionary of custom metadata (if set)
            - tags: List of tags (if set)

            Only non-empty values are included in the output.

        Called internally by the trace decorator to configure Laminar
        observation parameters.
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

    Called once per process. Multiple calls are safe (Laminar handles idempotency).
    """
    if settings.lmnr_project_api_key:
        Laminar.initialize(
            project_api_key=settings.lmnr_project_api_key, disabled_instruments=[Instruments.OPENAI] if Instruments.OPENAI else [], export_timeout_seconds=15
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
    trim_documents: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


# Overload for the bare @trace call
@overload
def trace(func: Callable[P, R]) -> Callable[P, R]: ...  # noqa: UP047


# Actual implementation
def trace(  # noqa: UP047
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
    trim_documents: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    """Add Laminar observability tracing to any function.

    The trace decorator integrates functions with Laminar (LMNR) for
    distributed tracing, performance monitoring, and debugging. It
    automatically handles both sync and async functions, propagates
    trace context, and provides fine-grained control over what gets traced.

    USAGE GUIDELINE - Defaults First:
        By default, use WITHOUT any parameters unless instructed otherwise.
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

        trim_documents: Automatically trim document content in traces (default True).
                       When enabled, text content is trimmed to
                       first/last 100 chars, and all binary content is removed.
                       Binary content is removed, text content is trimmed.
                       Attachment content follows the same trimming rules.
                       Helps reduce trace size for large documents.

    Returns:
        Decorated function with same signature but added tracing.

    TraceInfo propagation:
        If the decorated function has a 'trace_info' parameter, the decorator
        automatically creates or propagates a TraceInfo instance, ensuring
        consistent session/user tracking across the call chain.

    Environment variables:
        - LMNR_DEBUG: Set to "true" to enable debug-level traces
        - LMNR_PROJECT_API_KEY: Required for trace submission

    Performance:
        - Tracing overhead is minimal (~1-2ms per call)
        - When level="off", decorator returns original function unchanged
        - Large inputs/outputs can be excluded with ignore_* parameters

    Automatically initializes Laminar on first use. Works with both sync and
    async functions. Preserves function signature and metadata. Thread-safe
    and async-safe.
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
        bound_observe = observe

        bound_session_id = session_id
        bound_user_id = user_id
        bound_metadata = metadata if metadata is not None else {}
        bound_tags = tags if tags is not None else []
        bound_span_type = span_type
        bound_ignore_input = ignore_input
        bound_ignore_output = ignore_output
        bound_ignore_inputs = ignore_inputs
        bound_input_formatter = input_formatter
        bound_output_formatter = output_formatter
        bound_ignore_exceptions = ignore_exceptions
        bound_preserve_global_context = preserve_global_context
        bound_trim_documents = trim_documents

        # Create document trimming formatters if needed
        def _create_trimming_input_formatter(*args: Any, **kwargs: Any) -> str:
            # First, let any custom formatter process the data
            if bound_input_formatter:
                result = bound_input_formatter(*args, **kwargs)
                # If formatter returns string, try to parse and trim
                if isinstance(result, str):  # type: ignore[reportUnknownArgumentType]
                    try:
                        data = json.loads(result)
                        trimmed = _trim_documents_in_data(data)
                        return json.dumps(trimmed)
                    except (json.JSONDecodeError, TypeError):
                        return result
                else:
                    # If formatter returns dict/list, trim it
                    trimmed = _trim_documents_in_data(result)
                    return json.dumps(trimmed) if not isinstance(trimmed, str) else trimmed
            else:
                # No custom formatter - mimic Laminar's get_input_from_func_args
                # Build a dict with parameter names as keys (like Laminar does)
                params = list(sig.parameters.keys())
                data: dict[str, Any] = {}

                # Map args to parameter names
                for i, arg in enumerate(args):
                    if i < len(params):
                        data[params[i]] = arg

                # Add kwargs
                data.update(kwargs)

                # Serialize with our helper function
                serialized = json.dumps(data, default=_serialize_for_tracing)
                parsed = json.loads(serialized)

                # Trim documents in the serialized data
                trimmed = _trim_documents_in_data(parsed)
                return json.dumps(trimmed)

        def _create_trimming_output_formatter(result: Any) -> str:
            # First, let any custom formatter process the data
            if bound_output_formatter:
                formatted = bound_output_formatter(result)
                # If formatter returns string, try to parse and trim
                if isinstance(formatted, str):  # type: ignore[reportUnknownArgumentType]
                    try:
                        data = json.loads(formatted)
                        trimmed = _trim_documents_in_data(data)
                        return json.dumps(trimmed)
                    except (json.JSONDecodeError, TypeError):
                        return formatted
                else:
                    # If formatter returns dict/list, trim it
                    trimmed = _trim_documents_in_data(formatted)
                    return json.dumps(trimmed) if not isinstance(trimmed, str) else trimmed
            else:
                # No custom formatter, serialize result with smart defaults
                # Serialize with our extracted helper function
                serialized = json.dumps(result, default=_serialize_for_tracing)
                parsed = json.loads(serialized)

                # Trim documents in the serialized data
                trimmed = _trim_documents_in_data(parsed)
                return json.dumps(trimmed)

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
            if bound_session_id:
                observe_params["session_id"] = bound_session_id
            if bound_user_id:
                observe_params["user_id"] = bound_user_id
            if bound_metadata:
                observe_params["metadata"] = bound_metadata
            if bound_tags:
                observe_params["tags"] = observe_params.get("tags", []) + bound_tags
            if bound_span_type:
                observe_params["span_type"] = bound_span_type

            # Add the new Laminar parameters
            if bound_ignore_input:
                observe_params["ignore_input"] = bound_ignore_input
            if bound_ignore_output:
                observe_params["ignore_output"] = bound_ignore_output
            if bound_ignore_inputs is not None:
                observe_params["ignore_inputs"] = bound_ignore_inputs

            # Use trimming formatters if trim_documents is enabled
            if bound_trim_documents:
                # Use the trimming formatters (which may wrap custom formatters)
                observe_params["input_formatter"] = _create_trimming_input_formatter
                observe_params["output_formatter"] = _create_trimming_output_formatter
            else:
                # Use custom formatters directly if provided
                if bound_input_formatter is not None:
                    observe_params["input_formatter"] = bound_input_formatter
                if bound_output_formatter is not None:
                    observe_params["output_formatter"] = bound_output_formatter

            if bound_ignore_exceptions:
                observe_params["ignore_exceptions"] = bound_ignore_exceptions
            if bound_preserve_global_context:
                observe_params["preserve_global_context"] = bound_preserve_global_context

            return observe_params

        # --- The actual wrappers ---
        @wraps(f)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """Synchronous wrapper for traced function.

            Returns:
                The result of the wrapped function.
            """
            observe_params = _prepare_and_get_observe_params(kwargs)
            observed_func = bound_observe(**observe_params)(f)
            return observed_func(*args, **kwargs)

        @wraps(f)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """Asynchronous wrapper for traced function.

            Returns:
                The result of the wrapped function.
            """
            observe_params = _prepare_and_get_observe_params(kwargs)
            observed_func = bound_observe(**observe_params)(f)
            return await observed_func(*args, **kwargs)  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]

        wrapper = async_wrapper if is_coroutine else sync_wrapper

        # Mark function as traced for detection by pipeline decorators
        wrapper.__is_traced__ = True  # type: ignore[attr-defined]

        # Preserve the original function signature
        with contextlib.suppress(AttributeError, ValueError):
            wrapper.__signature__ = sig  # type: ignore[attr-defined]

        return cast(Callable[P, R], wrapper)

    if func:
        return decorator(func)  # Called as @trace
    return decorator  # Called as @trace(...)


def set_trace_cost(cost: float | str) -> None:
    """Set cost attributes for the current trace span.

    Sets cost metadata in the current LMNR trace span for tracking expenses
    of custom operations. This function should be called within a traced
    function to dynamically set or update the cost associated with the
    current operation. Particularly useful for tracking costs of external
    API calls, compute resources, or custom billing scenarios.

    The cost is stored in three metadata fields for observability tool consumption:
    - gen_ai.usage.output_cost: OpenTelemetry GenAI semantic convention
    - gen_ai.usage.cost: Aggregated cost field
    - cost: Short-form cost field

    Args:
        cost: The cost value to set. Can be:
              - float: Cost in dollars (e.g., 0.05 for 5 cents)
              - str: USD format with dollar sign (e.g., "$0.05" or "$1.25")
              Only positive values will be set; zero or negative values are ignored.

    Raises:
        ValueError: If string format is invalid (not a valid USD amount).

    Only works within a traced context (function decorated with @trace,
    @pipeline_task, or @pipeline_flow). LLM costs are tracked automatically via
    ModelResponse; use this for non-LLM costs. Multiple calls overwrite the
    previous cost (not cumulative). If called outside a traced context, it has
    no effect and does not raise an error.
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
            raise ValueError(f"Invalid USD format: {cost!r}. Must be a valid number after '$'") from e
    else:
        cost_value = cost

    if cost_value > 0:
        # Build the attributes dictionary with cost metadata
        attributes: dict[Attributes | str, float] = {
            "gen_ai.usage.output_cost": cost_value,
            "gen_ai.usage.cost": cost_value,
            "cost": cost_value,
        }

        with contextlib.suppress(Exception):
            Laminar.set_span_attributes(attributes)


__all__ = ["TraceInfo", "TraceLevel", "set_trace_cost", "trace"]
