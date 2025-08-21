"""Pipeline decorators that combine Prefect functionality with tracing support.

These decorators extend the base Prefect decorators with automatic tracing capabilities.
"""

import datetime
import functools
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Dict,
    Iterable,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)

from prefect.assets import Asset
from prefect.cache_policies import CachePolicy
from prefect.context import TaskRunContext
from prefect.flows import Flow, FlowStateHook
from prefect.futures import PrefectFuture
from prefect.results import ResultSerializer, ResultStorage
from prefect.task_runners import TaskRunner
from prefect.tasks import (
    RetryConditionCallable,
    StateHookCallable,
    Task,
    TaskRunNameValueOrCallable,
)
from prefect.utilities.annotations import NotSet
from typing_extensions import Concatenate, ParamSpec

from ai_pipeline_core.documents import DocumentList
from ai_pipeline_core.flow.options import FlowOptions
from ai_pipeline_core.prefect import flow, task
from ai_pipeline_core.tracing import TraceLevel, trace

if TYPE_CHECKING:
    pass

P = ParamSpec("P")
R = TypeVar("R")

# ============================================================================
# PIPELINE TASK DECORATOR
# ============================================================================


@overload
def pipeline_task(__fn: Callable[P, R], /) -> Task[P, R]: ...


@overload
def pipeline_task(
    *,
    # Tracing parameters
    trace_level: TraceLevel = "always",
    trace_ignore_input: bool = False,
    trace_ignore_output: bool = False,
    trace_ignore_inputs: list[str] | None = None,
    trace_input_formatter: Optional[Callable[..., str]] = None,
    trace_output_formatter: Optional[Callable[..., str]] = None,
    # Prefect parameters
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    version: Optional[str] = None,
    cache_policy: Union[CachePolicy, type[NotSet]] = NotSet,
    cache_key_fn: Optional[Callable[[TaskRunContext, Dict[str, Any]], Optional[str]]] = None,
    cache_expiration: Optional[datetime.timedelta] = None,
    task_run_name: Optional[TaskRunNameValueOrCallable] = None,
    retries: Optional[int] = None,
    retry_delay_seconds: Optional[
        Union[float, int, list[float], Callable[[int], list[float]]]
    ] = None,
    retry_jitter_factor: Optional[float] = None,
    persist_result: Optional[bool] = None,
    result_storage: Optional[Union[ResultStorage, str]] = None,
    result_serializer: Optional[Union[ResultSerializer, str]] = None,
    result_storage_key: Optional[str] = None,
    cache_result_in_memory: bool = True,
    timeout_seconds: Union[int, float, None] = None,
    log_prints: Optional[bool] = False,
    refresh_cache: Optional[bool] = None,
    on_completion: Optional[list[StateHookCallable]] = None,
    on_failure: Optional[list[StateHookCallable]] = None,
    retry_condition_fn: Optional[RetryConditionCallable] = None,
    viz_return_value: Optional[bool] = None,
    asset_deps: Optional[list[Union[str, Asset]]] = None,
) -> Callable[[Callable[P, R]], Task[P, R]]: ...


def pipeline_task(
    __fn: Optional[Callable[P, R]] = None,
    /,
    *,
    # Tracing parameters
    trace_level: TraceLevel = "always",
    trace_ignore_input: bool = False,
    trace_ignore_output: bool = False,
    trace_ignore_inputs: list[str] | None = None,
    trace_input_formatter: Optional[Callable[..., str]] = None,
    trace_output_formatter: Optional[Callable[..., str]] = None,
    # Prefect parameters
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    version: Optional[str] = None,
    cache_policy: Union[CachePolicy, type[NotSet]] = NotSet,
    cache_key_fn: Optional[Callable[[TaskRunContext, Dict[str, Any]], Optional[str]]] = None,
    cache_expiration: Optional[datetime.timedelta] = None,
    task_run_name: Optional[TaskRunNameValueOrCallable] = None,
    retries: Optional[int] = None,
    retry_delay_seconds: Optional[
        Union[float, int, list[float], Callable[[int], list[float]]]
    ] = None,
    retry_jitter_factor: Optional[float] = None,
    persist_result: Optional[bool] = None,
    result_storage: Optional[Union[ResultStorage, str]] = None,
    result_serializer: Optional[Union[ResultSerializer, str]] = None,
    result_storage_key: Optional[str] = None,
    cache_result_in_memory: bool = True,
    timeout_seconds: Union[int, float, None] = None,
    log_prints: Optional[bool] = False,
    refresh_cache: Optional[bool] = None,
    on_completion: Optional[list[StateHookCallable]] = None,
    on_failure: Optional[list[StateHookCallable]] = None,
    retry_condition_fn: Optional[RetryConditionCallable] = None,
    viz_return_value: Optional[bool] = None,
    asset_deps: Optional[list[Union[str, Asset]]] = None,
) -> Union[Task[P, R], Callable[[Callable[P, R]], Task[P, R]]]:
    """
    Pipeline task decorator that combines Prefect task functionality with automatic tracing.

    This decorator applies tracing before the Prefect task decorator, allowing you to
    monitor task execution with LMNR while maintaining all Prefect functionality.

    Args:
        trace_level: Control tracing ("always", "debug", "off")
        trace_ignore_input: Whether to ignore input in traces
        trace_ignore_output: Whether to ignore output in traces
        trace_ignore_inputs: List of input parameter names to ignore
        trace_input_formatter: Custom formatter for inputs
        trace_output_formatter: Custom formatter for outputs

        Plus all standard Prefect task parameters...
    """

    def decorator(fn: Callable[P, R]) -> Task[P, R]:
        # Apply tracing first if enabled
        if trace_level != "off":
            traced_fn = trace(
                level=trace_level,
                name=name or fn.__name__,
                ignore_input=trace_ignore_input,
                ignore_output=trace_ignore_output,
                ignore_inputs=trace_ignore_inputs,
                input_formatter=trace_input_formatter,
                output_formatter=trace_output_formatter,
            )(fn)
        else:
            traced_fn = fn

        # Then apply Prefect task decorator
        return task(  # pyright: ignore[reportCallIssue,reportUnknownVariableType]
            traced_fn,  # pyright: ignore[reportArgumentType]
            name=name,
            description=description,
            tags=tags,
            version=version,
            cache_policy=cache_policy,
            cache_key_fn=cache_key_fn,
            cache_expiration=cache_expiration,
            task_run_name=task_run_name,
            retries=retries or 0,
            retry_delay_seconds=retry_delay_seconds,
            retry_jitter_factor=retry_jitter_factor,
            persist_result=persist_result,
            result_storage=result_storage,
            result_serializer=result_serializer,
            result_storage_key=result_storage_key,
            cache_result_in_memory=cache_result_in_memory,
            timeout_seconds=timeout_seconds,
            log_prints=log_prints,
            refresh_cache=refresh_cache,
            on_completion=on_completion,
            on_failure=on_failure,
            retry_condition_fn=retry_condition_fn,
            viz_return_value=viz_return_value,
            asset_deps=asset_deps,
        )

    if __fn:
        return decorator(__fn)
    return decorator


# ============================================================================
# PIPELINE FLOW DECORATOR WITH DOCUMENT PROCESSING
# ============================================================================

# Type aliases for document flow signatures
DocumentsFlowSig = Callable[
    Concatenate[str, DocumentList, FlowOptions, P],
    Union[DocumentList, Coroutine[Any, Any, DocumentList]],
]

DocumentsFlowResult = Flow[Concatenate[str, DocumentList, FlowOptions, P], DocumentList]


@overload
def pipeline_flow(
    __fn: DocumentsFlowSig[P],
    /,
) -> DocumentsFlowResult[P]: ...


@overload
def pipeline_flow(
    *,
    # Tracing parameters
    trace_level: TraceLevel = "always",
    trace_ignore_input: bool = False,
    trace_ignore_output: bool = False,
    trace_ignore_inputs: list[str] | None = None,
    trace_input_formatter: Optional[Callable[..., str]] = None,
    trace_output_formatter: Optional[Callable[..., str]] = None,
    # Prefect parameters
    name: Optional[str] = None,
    version: Optional[str] = None,
    flow_run_name: Optional[Union[Callable[[], str], str]] = None,
    retries: Optional[int] = None,
    retry_delay_seconds: Optional[Union[int, float]] = None,
    task_runner: Optional[TaskRunner[PrefectFuture[Any]]] = None,
    description: Optional[str] = None,
    timeout_seconds: Union[int, float, None] = None,
    validate_parameters: bool = True,
    persist_result: Optional[bool] = None,
    result_storage: Optional[Union[ResultStorage, str]] = None,
    result_serializer: Optional[Union[ResultSerializer, str]] = None,
    cache_result_in_memory: bool = True,
    log_prints: Optional[bool] = None,
    on_completion: Optional[list["FlowStateHook[..., Any]"]] = None,
    on_failure: Optional[list["FlowStateHook[..., Any]"]] = None,
    on_cancellation: Optional[list["FlowStateHook[..., Any]"]] = None,
    on_crashed: Optional[list["FlowStateHook[..., Any]"]] = None,
    on_running: Optional[list["FlowStateHook[..., Any]"]] = None,
) -> Callable[[DocumentsFlowSig[P]], DocumentsFlowResult[P]]: ...


def pipeline_flow(
    __fn: Optional[DocumentsFlowSig[P]] = None,
    /,
    *,
    # Tracing parameters
    trace_level: TraceLevel = "always",
    trace_ignore_input: bool = False,
    trace_ignore_output: bool = False,
    trace_ignore_inputs: list[str] | None = None,
    trace_input_formatter: Optional[Callable[..., str]] = None,
    trace_output_formatter: Optional[Callable[..., str]] = None,
    # Prefect parameters
    name: Optional[str] = None,
    version: Optional[str] = None,
    flow_run_name: Optional[Union[Callable[[], str], str]] = None,
    retries: Optional[int] = None,
    retry_delay_seconds: Optional[Union[int, float]] = None,
    task_runner: Optional[TaskRunner[PrefectFuture[Any]]] = None,
    description: Optional[str] = None,
    timeout_seconds: Union[int, float, None] = None,
    validate_parameters: bool = True,
    persist_result: Optional[bool] = None,
    result_storage: Optional[Union[ResultStorage, str]] = None,
    result_serializer: Optional[Union[ResultSerializer, str]] = None,
    cache_result_in_memory: bool = True,
    log_prints: Optional[bool] = None,
    on_completion: Optional[list["FlowStateHook[..., Any]"]] = None,
    on_failure: Optional[list["FlowStateHook[..., Any]"]] = None,
    on_cancellation: Optional[list["FlowStateHook[..., Any]"]] = None,
    on_crashed: Optional[list["FlowStateHook[..., Any]"]] = None,
    on_running: Optional[list["FlowStateHook[..., Any]"]] = None,
) -> Union[DocumentsFlowResult[P], Callable[[DocumentsFlowSig[P]], DocumentsFlowResult[P]]]:
    """
    Pipeline flow for document processing with standardized signature.

    This decorator enforces a specific signature for document processing flows:
    - First parameter: project_name (str)
    - Second parameter: documents (DocumentList)
    - Third parameter: flow_options (FlowOptions or subclass)
    - Additional parameters allowed
    - Must return DocumentList

    It includes automatic tracing and all Prefect flow functionality.

    Args:
        trace_level: Control tracing ("always", "debug", "off")
        trace_ignore_input: Whether to ignore input in traces
        trace_ignore_output: Whether to ignore output in traces
        trace_ignore_inputs: List of input parameter names to ignore
        trace_input_formatter: Custom formatter for inputs
        trace_output_formatter: Custom formatter for outputs

        Plus all standard Prefect flow parameters...
    """

    def decorator(func: DocumentsFlowSig[P]) -> DocumentsFlowResult[P]:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if len(params) < 3:
            raise TypeError(
                f"@pipeline_flow '{func.__name__}' must accept at least 3 arguments: "
                "(project_name, documents, flow_options)"
            )

        # Validate parameter types (optional but recommended)
        # We check names as a convention, not strict type checking at decoration time
        expected_names = ["project_name", "documents", "flow_options"]
        for i, expected in enumerate(expected_names):
            if i < len(params) and params[i].name != expected:
                print(
                    f"Warning: Parameter {i + 1} of '{func.__name__}' is named '{params[i].name}' "
                    f"but convention suggests '{expected}'"
                )

        # Create wrapper that ensures return type
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper(  # pyright: ignore[reportRedeclaration]
                project_name: str,
                documents: DocumentList,
                flow_options: FlowOptions,
                *args,  # pyright: ignore[reportMissingParameterType]
                **kwargs,  # pyright: ignore[reportMissingParameterType]
            ) -> DocumentList:
                result = await func(project_name, documents, flow_options, *args, **kwargs)
                # Runtime type checking
                DL = DocumentList  # Avoid recomputation
                if not isinstance(result, DL):
                    raise TypeError(
                        f"Flow '{func.__name__}' must return a DocumentList, "
                        f"but returned {type(result).__name__}"
                    )
                return result
        else:

            @functools.wraps(func)
            def wrapper(  # pyright: ignore[reportRedeclaration]
                project_name: str,
                documents: DocumentList,
                flow_options: FlowOptions,
                *args,  # pyright: ignore[reportMissingParameterType]
                **kwargs,  # pyright: ignore[reportMissingParameterType]
            ) -> DocumentList:
                result = func(project_name, documents, flow_options, *args, **kwargs)
                # Runtime type checking
                DL = DocumentList  # Avoid recomputation
                if not isinstance(result, DL):
                    raise TypeError(
                        f"Flow '{func.__name__}' must return a DocumentList, "
                        f"but returned {type(result).__name__}"
                    )
                return result

        # Apply tracing first if enabled
        if trace_level != "off":
            traced_wrapper = trace(
                level=trace_level,
                name=name or func.__name__,
                ignore_input=trace_ignore_input,
                ignore_output=trace_ignore_output,
                ignore_inputs=trace_ignore_inputs,
                input_formatter=trace_input_formatter,
                output_formatter=trace_output_formatter,
            )(wrapper)
        else:
            traced_wrapper = wrapper

        # Then apply Prefect flow decorator
        return cast(
            DocumentsFlowResult[P],
            flow(  # pyright: ignore[reportCallIssue,reportUnknownVariableType]
                traced_wrapper,  # pyright: ignore[reportArgumentType]
                name=name,
                version=version,
                flow_run_name=flow_run_name,
                retries=retries,
                retry_delay_seconds=retry_delay_seconds,
                task_runner=task_runner,
                description=description,
                timeout_seconds=timeout_seconds,
                validate_parameters=validate_parameters,
                persist_result=persist_result,
                result_storage=result_storage,
                result_serializer=result_serializer,
                cache_result_in_memory=cache_result_in_memory,
                log_prints=log_prints,
                on_completion=on_completion,
                on_failure=on_failure,
                on_cancellation=on_cancellation,
                on_crashed=on_crashed,
                on_running=on_running,
            ),
        )

    if __fn:
        return decorator(__fn)
    return decorator


__all__ = ["pipeline_task", "pipeline_flow"]
