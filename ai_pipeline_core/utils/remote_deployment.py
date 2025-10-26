"""Experimental remote deployment utilities.

EXPERIMENTAL: This module provides utilities for calling remotely deployed Prefect flows.
Subject to change in future versions.
"""

import inspect
from functools import wraps
from typing import Any, Callable, ParamSpec, Type, TypeVar

from prefect import get_client
from prefect.client.orchestration import PrefectClient
from prefect.client.schemas import FlowRun
from prefect.context import AsyncClientContext
from prefect.deployments.flow_runs import run_deployment
from prefect.exceptions import ObjectNotFound

from ai_pipeline_core import DocumentList, FlowDocument
from ai_pipeline_core.settings import settings
from ai_pipeline_core.tracing import TraceLevel, set_trace_cost, trace

# --------------------------------------------------------------------------- #
# Utility functions (copied from pipeline.py for consistency)
# --------------------------------------------------------------------------- #


def _callable_name(obj: Any, fallback: str) -> str:
    """Safely extract callable's name for error messages.

    Args:
        obj: Any object that might have a __name__ attribute.
        fallback: Default name if extraction fails.

    Returns:
        The callable's __name__ if available, fallback otherwise.

    Note:
        Internal helper that never raises exceptions.
    """
    try:
        n = getattr(obj, "__name__", None)
        return n if isinstance(n, str) else fallback
    except Exception:
        return fallback


def _is_already_traced(func: Callable[..., Any]) -> bool:
    """Check if a function has already been wrapped by the trace decorator.

    This checks both for the explicit __is_traced__ marker and walks
    the __wrapped__ chain to detect nested trace decorations.

    Args:
        func: Function to check for existing trace decoration.

    Returns:
        True if the function is already traced, False otherwise.
    """
    # Check for explicit marker
    if hasattr(func, "__is_traced__") and func.__is_traced__:  # type: ignore[attr-defined]
        return True

    # Walk the __wrapped__ chain to detect nested traces
    current = func
    depth = 0
    max_depth = 10  # Prevent infinite loops

    while hasattr(current, "__wrapped__") and depth < max_depth:
        wrapped = current.__wrapped__  # type: ignore[attr-defined]
        # Check if the wrapped function has the trace marker
        if hasattr(wrapped, "__is_traced__") and wrapped.__is_traced__:  # type: ignore[attr-defined]
            return True
        current = wrapped
        depth += 1

    return False


# --------------------------------------------------------------------------- #
# Remote deployment execution
# --------------------------------------------------------------------------- #


async def run_remote_deployment(deployment_name: str, parameters: dict[str, Any]) -> Any:
    """Run a remote Prefect deployment.

    Args:
        deployment_name: Name of the deployment to run.
        parameters: Parameters to pass to the deployment.

    Returns:
        Result from the deployment execution.

    Raises:
        ValueError: If deployment is not found in local or remote Prefect API.
    """

    async def _run(client: PrefectClient, as_subflow: bool) -> Any:
        fr: FlowRun = await run_deployment(
            client=client, name=deployment_name, parameters=parameters, as_subflow=as_subflow
        )  # type: ignore
        return await fr.state.result()  # type: ignore

    async with get_client() as client:
        try:
            await client.read_deployment_by_name(name=deployment_name)
            return await _run(client, True)
        except ObjectNotFound:
            pass

    if not settings.prefect_api_url:
        raise ValueError(f"{deployment_name} deployment not found, PREFECT_API_URL is not set")

    async with PrefectClient(
        api=settings.prefect_api_url,
        api_key=settings.prefect_api_key,
        auth_string=settings.prefect_api_auth_string,
    ) as client:
        try:
            await client.read_deployment_by_name(name=deployment_name)
            with AsyncClientContext.model_construct(
                client=client, _httpx_settings=None, _context_stack=0
            ):
                return await _run(client, False)
        except ObjectNotFound:
            pass

    raise ValueError(f"{deployment_name} deployment not found")


P = ParamSpec("P")
T = TypeVar("T")


def remote_deployment(
    output_document_type: Type[FlowDocument],
    *,
    # tracing
    name: str | None = None,
    trace_level: TraceLevel = "always",
    trace_ignore_input: bool = False,
    trace_ignore_output: bool = False,
    trace_ignore_inputs: list[str] | None = None,
    trace_input_formatter: Callable[..., str] | None = None,
    trace_output_formatter: Callable[..., str] | None = None,
    trace_cost: float | None = None,
    trace_trim_documents: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for calling remote Prefect deployments with automatic tracing.

    EXPERIMENTAL: Decorator for calling remote Prefect deployments with automatic
    parameter serialization, result deserialization, and LMNR tracing.

    IMPORTANT: Never combine with @trace decorator - this includes tracing automatically.
    The framework will raise TypeError if you try to use both decorators together.

    Best Practice - Use Defaults:
        For most use cases, only specify output_document_type. The defaults provide
        automatic tracing with optimal settings.

    Args:
        output_document_type: The FlowDocument type to deserialize results into.
        name: Custom trace name (defaults to function name).
        trace_level: When to trace ("always", "debug", "off").
                    - "always": Always trace (default)
                    - "debug": Only trace when LMNR_DEBUG="true"
                    - "off": Disable tracing
        trace_ignore_input: Don't trace input arguments.
        trace_ignore_output: Don't trace return value.
        trace_ignore_inputs: List of parameter names to exclude from tracing.
        trace_input_formatter: Custom formatter for input tracing.
        trace_output_formatter: Custom formatter for output tracing.
        trace_cost: Optional cost value to track in metadata. When provided and > 0,
             sets gen_ai.usage.output_cost, gen_ai.usage.cost, and cost metadata.
        trace_trim_documents: Trim document content in traces to first 100 chars (default True).
                             Reduces trace size with large documents.

    Returns:
        Decorator function that wraps the target function.

    Example:
        >>> # RECOMMENDED - Minimal usage
        >>> @remote_deployment(output_document_type=OutputDoc)
        >>> async def process_remotely(
        ...     project_name: str,
        ...     documents: DocumentList,
        ...     flow_options: FlowOptions
        >>> ) -> DocumentList:
        ...     pass  # This stub is replaced by remote call
        >>>
        >>> # With custom tracing
        >>> @remote_deployment(
        ...     output_document_type=OutputDoc,
        ...     trace_cost=0.05,  # Track cost of remote execution
        ...     trace_level="debug"  # Only trace in debug mode
        >>> )
        >>> async def debug_remote_flow(...) -> DocumentList:
        ...     pass

    Note:
        - Remote calls are automatically traced with LMNR
        - The decorated function's body is never executed - it serves as a signature template
        - Deployment name is auto-derived from function name
        - DocumentList parameters are automatically serialized/deserialized

    Raises:
        TypeError: If function is already decorated with @trace.
        ValueError: If deployment is not found.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        fname = _callable_name(func, "remote_deployment")

        # Check if function is already traced
        if _is_already_traced(func):
            raise TypeError(
                f"@remote_deployment target '{fname}' is already decorated "
                f"with @trace. Remove the @trace decorator - @remote_deployment includes "
                f"tracing automatically."
            )

        @wraps(func)
        async def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Serialize parameters, converting DocumentList to list[dict]
            parameters = {}
            for pname, value in bound.arguments.items():
                if isinstance(value, DocumentList):
                    parameters[pname] = [doc for doc in value]
                else:
                    parameters[pname] = value

            # Auto-derive deployment name
            deployment_name = f"{func.__name__.replace('_', '-')}/{func.__name__}"

            result = await run_remote_deployment(
                deployment_name=deployment_name, parameters=parameters
            )

            # Set trace cost if provided
            if trace_cost is not None and trace_cost > 0:
                set_trace_cost(trace_cost)

            assert isinstance(result, list), "Result must be a list"

            # Auto-handle return type conversion from list[dict] to DocumentList
            return_type = sig.return_annotation

            assert return_type is DocumentList, "Return type must be a DocumentList"
            return DocumentList([output_document_type(**item) for item in result])  # type: ignore

        # Apply trace decorator
        traced_wrapper = trace(
            level=trace_level,
            name=name or fname,
            ignore_input=trace_ignore_input,
            ignore_output=trace_ignore_output,
            ignore_inputs=trace_ignore_inputs,
            input_formatter=trace_input_formatter,
            output_formatter=trace_output_formatter,
            trim_documents=trace_trim_documents,
        )(_wrapper)

        return traced_wrapper  # type: ignore

    return decorator
