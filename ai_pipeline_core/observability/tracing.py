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
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, Literal, ParamSpec, TypeVar, cast, overload

from lmnr import Attributes, Instruments, Laminar, observe
from pydantic import BaseModel, Field

from ai_pipeline_core.documents import Document
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.settings import settings

from ._trimming import trim_documents_in_data

logger = get_pipeline_logger(__name__)

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
def _serialize_for_tracing(obj: Any) -> Any:
    """Convert objects to JSON-serializable format for tracing."""
    if isinstance(obj, Document):
        return obj.serialize_model()
    if isinstance(obj, list) and obj and all(isinstance(x, Document) for x in cast(list[Any], obj)):
        return [doc.serialize_model() for doc in cast(list[Document], obj)]
    if isinstance(obj, BaseModel):
        # Document fields need serialize_model(); use model_dump for other BaseModels
        return {
            k: v.serialize_model() if isinstance(v, Document) else _serialize_for_tracing(v) if isinstance(v, BaseModel) else v for k, v in obj.__dict__.items()
        }
    try:
        return str(obj)  # pyright: ignore[reportUnknownArgumentType]
    except Exception:
        return f"<{type(obj).__name__}>"  # pyright: ignore[reportUnknownArgumentType]


def _trim_formatted(formatted: Any) -> str:
    """Apply document trimming to formatter output (str or data)."""
    if isinstance(formatted, str):
        try:
            data = json.loads(formatted)
            return json.dumps(trim_documents_in_data(data))
        except (json.JSONDecodeError, TypeError):
            return formatted
    trimmed = trim_documents_in_data(formatted)
    return json.dumps(trimmed) if not isinstance(trimmed, str) else trimmed


def _serialize_and_trim(obj: Any) -> str:
    """Serialize to JSON, parse, trim documents, re-serialize."""
    serialized = json.dumps(obj, default=_serialize_for_tracing)
    return json.dumps(trim_documents_in_data(json.loads(serialized)))


@dataclass(frozen=True, slots=True)
class _TraceConfig:
    """Pre-computed parameters for a traced function."""

    observe_name: str
    sig: inspect.Signature
    session_id: str | None
    user_id: str | None
    metadata: dict[str, Any]
    tags: list[str]
    span_type: str | None
    ignore_input: bool
    ignore_output: bool
    ignore_inputs: list[str] | None
    input_formatter: Callable[..., str] | None
    output_formatter: Callable[..., str] | None
    ignore_exceptions: bool
    preserve_global_context: bool
    trim_documents: bool


def _make_trimming_input_formatter(cfg: _TraceConfig) -> Callable[..., str]:
    """Build input formatter that applies document trimming."""

    def formatter(*args: Any, **kwargs: Any) -> str:
        if cfg.input_formatter:
            return _trim_formatted(cfg.input_formatter(*args, **kwargs))
        params = list(cfg.sig.parameters.keys())
        data: dict[str, Any] = {params[i]: arg for i, arg in enumerate(args) if i < len(params)}
        data.update(kwargs)
        return _serialize_and_trim(data)

    return formatter


def _make_trimming_output_formatter(cfg: _TraceConfig) -> Callable[..., str]:
    """Build output formatter that applies document trimming."""

    def formatter(result: Any) -> str:
        if cfg.output_formatter:
            return _trim_formatted(cfg.output_formatter(result))
        return _serialize_and_trim(result)

    return formatter


def _prepare_observe_params(cfg: _TraceConfig, runtime_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Inspect runtime args, manage TraceInfo, and return params for lmnr.observe.

    Modifies runtime_kwargs in place to inject TraceInfo if the function expects it.
    """
    trace_info = runtime_kwargs.get("trace_info")
    if not isinstance(trace_info, TraceInfo):
        trace_info = TraceInfo()
        if "trace_info" in cfg.sig.parameters:
            runtime_kwargs["trace_info"] = trace_info

    observe_params = trace_info.get_observe_kwargs()
    observe_params["name"] = cfg.observe_name

    if cfg.session_id:
        observe_params["session_id"] = cfg.session_id
    if cfg.user_id:
        observe_params["user_id"] = cfg.user_id
    if cfg.metadata:
        observe_params["metadata"] = cfg.metadata
    if cfg.tags:
        observe_params["tags"] = observe_params.get("tags", []) + cfg.tags
    if cfg.span_type:
        observe_params["span_type"] = cfg.span_type

    if cfg.ignore_input:
        observe_params["ignore_input"] = cfg.ignore_input
    if cfg.ignore_output:
        observe_params["ignore_output"] = cfg.ignore_output
    if cfg.ignore_inputs is not None:
        observe_params["ignore_inputs"] = cfg.ignore_inputs

    if cfg.trim_documents:
        observe_params["input_formatter"] = _make_trimming_input_formatter(cfg)
        observe_params["output_formatter"] = _make_trimming_output_formatter(cfg)
    else:
        if cfg.input_formatter is not None:
            observe_params["input_formatter"] = cfg.input_formatter
        if cfg.output_formatter is not None:
            observe_params["output_formatter"] = cfg.output_formatter

    if cfg.ignore_exceptions:
        observe_params["ignore_exceptions"] = cfg.ignore_exceptions
    if cfg.preserve_global_context:
        observe_params["preserve_global_context"] = cfg.preserve_global_context

    return observe_params


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
        """Build kwargs for ``lmnr.observe()``, with env-var fallbacks for session/user ID."""
        kwargs: dict[str, Any] = {}
        if sid := (self.session_id or os.getenv("LMNR_SESSION_ID")):
            kwargs["session_id"] = sid
        if uid := (self.user_id or os.getenv("LMNR_USER_ID")):
            kwargs["user_id"] = uid
        if self.metadata:
            kwargs["metadata"] = self.metadata
        if self.tags:
            kwargs["tags"] = self.tags
        return kwargs


# ---------------------------------------------------------------------------
# ``trace`` decorator
# ---------------------------------------------------------------------------


_laminar_initialized = False
_laminar_init_lock = threading.Lock()


def _initialise_laminar() -> None:
    """Initialize Laminar SDK with project configuration (lazy, once per process).

    Sets up the Laminar observability client with the project API key
    from settings. Disables automatic OpenAI instrumentation to avoid
    conflicts with our custom tracing.

    IMPORTANT: This is called lazily at first trace execution (not at decoration time)
    to allow LMNR_SPAN_CONTEXT environment variable to be set before initialization.
    Laminar reads LMNR_SPAN_CONTEXT during initialize() to establish parent context
    for cross-process tracing.

    Uses double-checked locking pattern for thread safety. The flag is set AFTER
    successful initialization to prevent permanently disabled tracing on init failure.
    """
    global _laminar_initialized  # noqa: PLW0603

    # Fast path: already initialized (no lock needed)
    if _laminar_initialized:
        return

    with _laminar_init_lock:
        # Double-check inside lock
        if _laminar_initialized:
            return

        if settings.lmnr_project_api_key:
            disabled = [Instruments.OPENAI] if Instruments.OPENAI else []
            Laminar.initialize(project_api_key=settings.lmnr_project_api_key, disabled_instruments=disabled, export_timeout_seconds=15)

        # Set flag AFTER successful initialization
        _laminar_initialized = True


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

    Use WITHOUT parameters for most cases — defaults are production-ready.

    level controls when tracing is active: "always" (default), "debug" (only when
    LMNR_DEBUG=="true"), or "off" (returns original function unchanged).

    If the decorated function has a 'trace_info' parameter, the decorator
    automatically creates or propagates a TraceInfo instance for consistent
    session/user tracking across the call chain.

    Automatically initializes Laminar on first use. Works with both sync and
    async functions.
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
        if hasattr(f, "__is_traced__") and f.__is_traced__:  # type: ignore[attr-defined]
            is_prefect_task = hasattr(f, "fn") and hasattr(f, "submit") and hasattr(f, "map")
            is_prefect_flow = hasattr(f, "fn") and hasattr(f, "serve")
            if is_prefect_task or is_prefect_flow:
                fname = getattr(f, "__name__", "function")
                raise TypeError(
                    f"Function '{fname}' is already decorated with @pipeline_task or "
                    f"@pipeline_flow. Remove the @trace decorator - pipeline decorators "
                    f"include tracing automatically."
                )

        debug_value = settings.lmnr_debug or os.getenv("LMNR_DEBUG", "")
        if level == "debug" and debug_value.lower() != "true":
            return f

        cfg = _TraceConfig(
            observe_name=name or f.__name__,
            sig=inspect.signature(f),
            session_id=session_id,
            user_id=user_id,
            metadata=metadata if metadata is not None else {},
            tags=tags if tags is not None else [],
            span_type=span_type,
            ignore_input=ignore_input,
            ignore_output=ignore_output,
            ignore_inputs=ignore_inputs,
            input_formatter=input_formatter,
            output_formatter=output_formatter,
            ignore_exceptions=ignore_exceptions,
            preserve_global_context=preserve_global_context,
            trim_documents=trim_documents,
        )
        is_coroutine = inspect.iscoroutinefunction(f)

        @wraps(f)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _initialise_laminar()
            observe_params = _prepare_observe_params(cfg, kwargs)
            observed_func = observe(**observe_params)(f)
            return observed_func(*args, **kwargs)

        @wraps(f)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """Async wrapper with Laminar tracing."""
            _initialise_laminar()
            observe_params = _prepare_observe_params(cfg, kwargs)
            observed_func = observe(**observe_params)(f)
            return await observed_func(*args, **kwargs)  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]

        wrapper = async_wrapper if is_coroutine else sync_wrapper
        wrapper.__is_traced__ = True  # type: ignore[attr-defined]

        with contextlib.suppress(AttributeError, ValueError):
            wrapper.__signature__ = cfg.sig  # type: ignore[attr-defined]

        return cast(Callable[P, R], wrapper)

    if func:
        return decorator(func)  # Called as @trace
    return decorator  # Called as @trace(...)


def set_trace_cost(cost: float | str) -> None:
    """Set cost attributes for the current trace span.

    Only positive values are set; zero or negative values are ignored.
    Only works within a traced context (@trace, @pipeline_task, @pipeline_flow).
    Use for non-LLM costs; LLM costs are tracked automatically via ModelResponse.
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

        try:
            Laminar.set_span_attributes(attributes)
        except Exception as e:
            logger.debug("Failed to set trace cost attributes: %s", e)


__all__ = ["TraceInfo", "TraceLevel", "set_trace_cost", "trace"]
