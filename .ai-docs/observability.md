# MODULE: observability
# CLASSES: TraceInfo
# DEPENDS: BaseModel
# PURPOSE: Observability system for AI pipelines.
# VERSION: 0.10.3
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import TraceInfo, TraceLevel, set_trace_cost, trace
```

## Types & Constants

```python
TraceLevel = Literal["always", "debug", "off"]

```

## Internal Types

```python
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
