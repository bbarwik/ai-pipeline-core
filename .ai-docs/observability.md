# MODULE: observability
# CLASSES: TraceInfo
# DEPENDS: BaseModel
# PURPOSE: Observability system for AI pipelines.
# VERSION: 0.10.0
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import TraceInfo, set_trace_cost, trace
```

## Public API

```python
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
consistent tracing context."""
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


```

## Functions

```python
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
        # NOTE: _initialise_laminar() is NOT called here (at decoration/import time)
        # to allow LMNR_SPAN_CONTEXT to be set before Laminar.initialize() runs.
        # It's called lazily in the wrapper functions at first execution.
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

        # Shared trimming logic for input/output formatters
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

        def _create_trimming_input_formatter(*args: Any, **kwargs: Any) -> str:
            if bound_input_formatter:
                return _trim_formatted(bound_input_formatter(*args, **kwargs))
            # No custom formatter — build dict from parameter names (like Laminar)
            params = list(sig.parameters.keys())
            data: dict[str, Any] = {params[i]: arg for i, arg in enumerate(args) if i < len(params)}
            data.update(kwargs)
            return _serialize_and_trim(data)

        def _create_trimming_output_formatter(result: Any) -> str:
            if bound_output_formatter:
                return _trim_formatted(bound_output_formatter(result))
            return _serialize_and_trim(result)

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
            # Lazy initialization: called at first execution, not at decoration time.
            # This allows LMNR_SPAN_CONTEXT to be set before Laminar.initialize().
            _initialise_laminar()
            observe_params = _prepare_and_get_observe_params(kwargs)
            observed_func = bound_observe(**observe_params)(f)
            return observed_func(*args, **kwargs)

        @wraps(f)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """Asynchronous wrapper for traced function.

            Returns:
                The result of the wrapped function.
            """
            # Lazy initialization: called at first execution, not at decoration time.
            # This allows LMNR_SPAN_CONTEXT to be set before Laminar.initialize().
            _initialise_laminar()
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

```

## Examples

**Set trace cost negative** (`tests/observability/test_trace_decorator.py:309`)

```python
@patch("ai_pipeline_core.observability.tracing.Laminar.set_span_attributes")
def test_set_trace_cost_negative(self, mock_set_attrs: Mock) -> None:
    """Test set_trace_cost with negative cost (should not call Laminar)."""
    set_trace_cost(-0.05)
    mock_set_attrs.assert_not_called()
```

**Set trace cost usd string negative** (`tests/observability/test_trace_decorator.py:343`)

```python
@patch("ai_pipeline_core.observability.tracing.Laminar.set_span_attributes")
def test_set_trace_cost_usd_string_negative(self, mock_set_attrs: Mock) -> None:
    """Test set_trace_cost with negative USD string (should not call Laminar)."""
    set_trace_cost("$-0.50")
    mock_set_attrs.assert_not_called()
```

**Set trace cost usd string zero** (`tests/observability/test_trace_decorator.py:337`)

```python
@patch("ai_pipeline_core.observability.tracing.Laminar.set_span_attributes")
def test_set_trace_cost_usd_string_zero(self, mock_set_attrs: Mock) -> None:
    """Test set_trace_cost with USD string format of zero (should not call Laminar)."""
    set_trace_cost("$0.00")
    mock_set_attrs.assert_not_called()
```

**Set trace cost zero** (`tests/observability/test_trace_decorator.py:303`)

```python
@patch("ai_pipeline_core.observability.tracing.Laminar.set_span_attributes")
def test_set_trace_cost_zero(self, mock_set_attrs: Mock) -> None:
    """Test set_trace_cost with zero cost (should not call Laminar)."""
    set_trace_cost(0.0)
    mock_set_attrs.assert_not_called()
```


## Error Examples

**Set trace cost invalid string empty after dollar** (`tests/observability/test_trace_decorator.py:358`)

```python
def test_set_trace_cost_invalid_string_empty_after_dollar(self) -> None:
    """Test set_trace_cost with invalid string (empty after dollar)."""
    with pytest.raises(ValueError, match=r"Invalid USD format.*Must be a valid number"):
        set_trace_cost("$")
```

**Set trace cost invalid string no dollar** (`tests/observability/test_trace_decorator.py:348`)

```python
def test_set_trace_cost_invalid_string_no_dollar(self) -> None:
    """Test set_trace_cost with invalid string (no dollar sign)."""
    with pytest.raises(ValueError, match=r"Invalid USD format.*Must start with"):
        set_trace_cost("0.50")
```

**Set trace cost invalid string not number** (`tests/observability/test_trace_decorator.py:353`)

```python
def test_set_trace_cost_invalid_string_not_number(self) -> None:
    """Test set_trace_cost with invalid string (not a number)."""
    with pytest.raises(ValueError, match=r"Invalid USD format.*Must be a valid number"):
        set_trace_cost("$abc")
```

**Config is frozen** (`tests/observability/test_initialization.py:26`)

```python
def test_config_is_frozen(self):
    config = ObservabilityConfig()
    with pytest.raises(ValidationError):
        config.clickhouse_host = "changed"  # type: ignore[misc]
```
