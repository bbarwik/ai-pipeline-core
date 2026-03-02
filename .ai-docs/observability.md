# MODULE: observability
# CLASSES: TraceInfo
# DEPENDS: BaseModel
# PURPOSE: Observability system for AI pipelines.
# VERSION: 0.12.4
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import TraceInfo, TraceLevel, set_trace_cost, trace
```

## Types & Constants

```python
TraceLevel = Literal["always", "debug", "off"]

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


```

## Functions

```python
def main(argv: list[str] | None = None) -> int:
    """CLI entry point for trace download and inspection.

    Usage:
        ai-trace list --limit 10 --status completed
        ai-trace show 550e8400-e29b-41d4-a716-446655440000
        ai-trace download 550e8400-e29b-41d4-a716-446655440000 -o ./debug/
        ai-trace download my-run-id --children -o ./debug/
    """
    parser = argparse.ArgumentParser(
        prog="ai-trace",
        description="List, inspect, and download pipeline traces from ClickHouse",
    )
    subparsers = parser.add_subparsers(dest="command")

    # download
    dl = subparsers.add_parser(
        "download",
        parents=[_connection_parser],
        help="Fetch trace data and reconstruct .trace/ directory",
    )
    dl.add_argument("identifier", help="Execution UUID or run_id")
    dl.add_argument("-o", "--output", type=str, default=None, help="Output directory (default: ./{id[:8]}_trace/)")
    dl.add_argument("--children", action="store_true", help="Include child pipeline runs (matched by run_id prefix)")
    dl.add_argument("--no-docs", action="store_true", help="Skip downloading documents referenced by replay files")

    # list
    ls = subparsers.add_parser(
        "list",
        parents=[_connection_parser],
        help="List recent pipeline runs",
    )
    ls.add_argument("--limit", type=int, default=_DEFAULT_LIST_LIMIT, help="Number of runs (default: 20)")
    ls.add_argument("--status", choices=["running", "completed", "failed"], help="Filter by status")
    ls.add_argument("--flow", type=str, help="Filter by flow name")

    # show
    sh = subparsers.add_parser(
        "show",
        parents=[_connection_parser],
        help="Show trace summary",
    )
    sh.add_argument("execution_id", help="Execution UUID")

    args = parser.parse_args(argv)

    handlers: dict[str, Any] = {"download": _cmd_download, "list": _cmd_list, "show": _cmd_show}
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)

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
    WARNING: Do not use for LLM conversation costs — those are tracked automatically
    via ModelResponse. Using this for LLM costs causes double-counting.
    Use only for non-LLM costs (e.g. external API calls, search services).
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

**Generates summary and costs** (`tests/observability/test_materializer_unit.py:218`)

```python
def test_generates_summary_and_costs(self, tmp_path):
    mat = _make_materializer(tmp_path, generate_summary=True)
    span = _make_span_data(
        span_id="root",
        trace_id="t1",
        attributes={"gen_ai.usage.input_tokens": 100, "gen_ai.usage.output_tokens": 50, "gen_ai.usage.cost": 0.01},
    )
    mat.on_span_start(span)
    mat.add_span(span)
    # After add_span, trace should be auto-finalized
    assert (tmp_path / "summary.md").exists()
```

**Trace info immutability expectation** (`tests/observability/test_trace_mutable_defaults.py:172`)

```python
def test_trace_info_immutability_expectation(self):
    """Test that TraceInfo follows Pydantic patterns for immutability."""
    trace = TraceInfo(metadata={"key": "value"}, tags=["tag1"])

    # Pydantic models with mutable fields should still protect against
    # default parameter sharing issues
    original_metadata_id = id(trace.metadata)
    original_tags_id = id(trace.tags)

    # Create another instance
    trace2 = TraceInfo()

    # Verify different instances
    assert id(trace2.metadata) != original_metadata_id
    assert id(trace2.tags) != original_tags_id
```

**Default trace** (`tests/observability/test_span_data_unit.py:310`)

```python
def test_default_trace(self):
    assert _classify_span_type({}) == "trace"
```

**Empty trace dir** (`tests/observability/test_download_docs.py:172`)

```python
def test_empty_trace_dir(self, tmp_path: Path):
    trace = tmp_path / ".trace"
    trace.mkdir()
    refs = _collect_doc_refs(trace)
    assert refs == {}
```

**Root span detected when parent not in trace** (`tests/observability/test_materializer_unit.py:493`)

```python
def test_root_span_detected_when_parent_not_in_trace(self, tmp_path: Path) -> None:
    """A span whose parent_span_id is not in trace.spans should become root."""
    mat = _make_materializer(tmp_path / "r1")
    s = _make_span_data(span_id="s1", parent_span_id="external-laminar-id", span_order=1, status="running")
    mat.on_span_start(s)
    trace = mat._traces["trace1"]
    assert trace.root_span_id == "s1"
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
