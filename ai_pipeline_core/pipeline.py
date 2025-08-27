"""
ai_pipeline_core.pipeline
=========================

Tiny wrappers around Prefect's public ``@task`` and ``@flow`` that add our
``trace`` decorator and **require async functions**.

Why this exists
---------------
Prefect tasks/flows are awaitable at runtime, but their public type stubs
don't declare that clearly. We therefore:

1) Return the **real Prefect objects** (so you keep every Prefect method).
2) Type them as small Protocols that say "this is awaitable and has common
   helpers like `.submit`/`.map`”.

This keeps Pyright happy without altering runtime behavior and avoids
leaking advanced typing constructs (like ``ParamSpec``) that confuse tools
that introspect callables (e.g., Pydantic).

Quick start
-----------
from ai_pipeline_core.pipeline import pipeline_task, pipeline_flow
from ai_pipeline_core.documents import DocumentList
from ai_pipeline_core.flow.options import FlowOptions

@pipeline_task
async def add(x: int, y: int) -> int:
    return x + y

@pipeline_flow
async def my_flow(project_name: str, docs: DocumentList, opts: FlowOptions) -> DocumentList:
    await add(1, 2)  # awaitable and typed
    return docs

Rules
-----
• Your decorated function **must** be ``async def``.
• ``@pipeline_flow`` functions must accept at least:
  (project_name: str, documents: DocumentList, flow_options: FlowOptions | subclass).
• Both wrappers return the same Prefect objects you'd get from Prefect directly.
"""

from __future__ import annotations

import datetime
import inspect
from typing import (
    Any,
    Callable,
    Coroutine,
    Iterable,
    Protocol,
    TypeVar,
    Union,
    cast,
    overload,
)

from prefect.assets import Asset
from prefect.cache_policies import CachePolicy
from prefect.context import TaskRunContext
from prefect.flows import FlowStateHook
from prefect.flows import flow as _prefect_flow  # public import
from prefect.futures import PrefectFuture
from prefect.results import ResultSerializer, ResultStorage
from prefect.task_runners import TaskRunner
from prefect.tasks import task as _prefect_task  # public import
from prefect.utilities.annotations import NotSet
from typing_extensions import TypeAlias

from ai_pipeline_core.documents import DocumentList
from ai_pipeline_core.flow.options import FlowOptions
from ai_pipeline_core.tracing import TraceLevel, trace

# --------------------------------------------------------------------------- #
# Public callback aliases (Prefect stubs omit these exact types)
# --------------------------------------------------------------------------- #
RetryConditionCallable: TypeAlias = Callable[[Any, Any, Any], bool]
StateHookCallable: TypeAlias = Callable[[Any, Any, Any], None]
TaskRunNameValueOrCallable: TypeAlias = Union[str, Callable[[], str]]

# --------------------------------------------------------------------------- #
# Typing helpers
# --------------------------------------------------------------------------- #
R_co = TypeVar("R_co", covariant=True)
FO_contra = TypeVar("FO_contra", bound=FlowOptions, contravariant=True)
"""Flow options are an *input* type, so contravariant fits the callable model."""


class _TaskLike(Protocol[R_co]):
    """Minimal 'task-like' view: awaitable call + common helpers."""

    def __call__(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, R_co]: ...

    submit: Callable[..., Any]
    map: Callable[..., Any]
    name: str | None

    def __getattr__(self, name: str) -> Any: ...  # allow unknown helpers without type errors


class _DocumentsFlowCallable(Protocol[FO_contra]):
    """User async flow signature (first three params fixed)."""

    def __call__(
        self,
        project_name: str,
        documents: DocumentList,
        flow_options: FO_contra,
        *args: Any,
        **kwargs: Any,
    ) -> Coroutine[Any, Any, DocumentList]: ...


class _FlowLike(Protocol[FO_contra]):
    """Callable returned by Prefect ``@flow`` wrapper that we expose to users."""

    def __call__(
        self,
        project_name: str,
        documents: DocumentList,
        flow_options: FO_contra,
        *args: Any,
        **kwargs: Any,
    ) -> Coroutine[Any, Any, DocumentList]: ...

    name: str | None

    def __getattr__(self, name: str) -> Any: ...  # allow unknown helpers without type errors


# --------------------------------------------------------------------------- #
# Small helper: safely get a callable's name without upsetting the type checker
# --------------------------------------------------------------------------- #
def _callable_name(obj: Any, fallback: str) -> str:
    try:
        n = getattr(obj, "__name__", None)
        return n if isinstance(n, str) else fallback
    except Exception:
        return fallback


# --------------------------------------------------------------------------- #
# @pipeline_task — async-only, traced, returns Prefect's Task object
# --------------------------------------------------------------------------- #
@overload
def pipeline_task(__fn: Callable[..., Coroutine[Any, Any, R_co]], /) -> _TaskLike[R_co]: ...
@overload
def pipeline_task(
    *,
    # tracing
    trace_level: TraceLevel = "always",
    trace_ignore_input: bool = False,
    trace_ignore_output: bool = False,
    trace_ignore_inputs: list[str] | None = None,
    trace_input_formatter: Callable[..., str] | None = None,
    trace_output_formatter: Callable[..., str] | None = None,
    # prefect passthrough
    name: str | None = None,
    description: str | None = None,
    tags: Iterable[str] | None = None,
    version: str | None = None,
    cache_policy: CachePolicy | type[NotSet] = NotSet,
    cache_key_fn: Callable[[TaskRunContext, dict[str, Any]], str | None] | None = None,
    cache_expiration: datetime.timedelta | None = None,
    task_run_name: TaskRunNameValueOrCallable | None = None,
    retries: int | None = None,
    retry_delay_seconds: int | float | list[float] | Callable[[int], list[float]] | None = None,
    retry_jitter_factor: float | None = None,
    persist_result: bool | None = None,
    result_storage: ResultStorage | str | None = None,
    result_serializer: ResultSerializer | str | None = None,
    result_storage_key: str | None = None,
    cache_result_in_memory: bool = True,
    timeout_seconds: int | float | None = None,
    log_prints: bool | None = False,
    refresh_cache: bool | None = None,
    on_completion: list[StateHookCallable] | None = None,
    on_failure: list[StateHookCallable] | None = None,
    retry_condition_fn: RetryConditionCallable | None = None,
    viz_return_value: bool | None = None,
    asset_deps: list[str | Asset] | None = None,
) -> Callable[[Callable[..., Coroutine[Any, Any, R_co]]], _TaskLike[R_co]]: ...


def pipeline_task(
    __fn: Callable[..., Coroutine[Any, Any, R_co]] | None = None,
    /,
    *,
    # tracing
    trace_level: TraceLevel = "always",
    trace_ignore_input: bool = False,
    trace_ignore_output: bool = False,
    trace_ignore_inputs: list[str] | None = None,
    trace_input_formatter: Callable[..., str] | None = None,
    trace_output_formatter: Callable[..., str] | None = None,
    # prefect passthrough
    name: str | None = None,
    description: str | None = None,
    tags: Iterable[str] | None = None,
    version: str | None = None,
    cache_policy: CachePolicy | type[NotSet] = NotSet,
    cache_key_fn: Callable[[TaskRunContext, dict[str, Any]], str | None] | None = None,
    cache_expiration: datetime.timedelta | None = None,
    task_run_name: TaskRunNameValueOrCallable | None = None,
    retries: int | None = None,
    retry_delay_seconds: int | float | list[float] | Callable[[int], list[float]] | None = None,
    retry_jitter_factor: float | None = None,
    persist_result: bool | None = None,
    result_storage: ResultStorage | str | None = None,
    result_serializer: ResultSerializer | str | None = None,
    result_storage_key: str | None = None,
    cache_result_in_memory: bool = True,
    timeout_seconds: int | float | None = None,
    log_prints: bool | None = False,
    refresh_cache: bool | None = None,
    on_completion: list[StateHookCallable] | None = None,
    on_failure: list[StateHookCallable] | None = None,
    retry_condition_fn: RetryConditionCallable | None = None,
    viz_return_value: bool | None = None,
    asset_deps: list[str | Asset] | None = None,
) -> _TaskLike[R_co] | Callable[[Callable[..., Coroutine[Any, Any, R_co]]], _TaskLike[R_co]]:
    """Decorate an **async** function as a traced Prefect task."""
    task_decorator: Callable[..., Any] = _prefect_task  # helps the type checker

    def _apply(fn: Callable[..., Coroutine[Any, Any, R_co]]) -> _TaskLike[R_co]:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"@pipeline_task target '{_callable_name(fn, 'task')}' must be 'async def'"
            )

        fname = _callable_name(fn, "task")
        traced_fn = trace(
            level=trace_level,
            name=name or fname,
            ignore_input=trace_ignore_input,
            ignore_output=trace_ignore_output,
            ignore_inputs=trace_ignore_inputs,
            input_formatter=trace_input_formatter,
            output_formatter=trace_output_formatter,
        )(fn)

        return cast(
            _TaskLike[R_co],
            task_decorator(
                name=name or fname,
                description=description,
                tags=tags,
                version=version,
                cache_policy=cache_policy,
                cache_key_fn=cache_key_fn,
                cache_expiration=cache_expiration,
                task_run_name=task_run_name or name or fname,
                retries=0 if retries is None else retries,
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
            )(traced_fn),
        )

    return _apply(__fn) if __fn else _apply


# --------------------------------------------------------------------------- #
# @pipeline_flow — async-only, traced, returns Prefect's flow wrapper
# --------------------------------------------------------------------------- #
@overload
def pipeline_flow(__fn: _DocumentsFlowCallable[FO_contra], /) -> _FlowLike[FO_contra]: ...
@overload
def pipeline_flow(
    *,
    # tracing
    trace_level: TraceLevel = "always",
    trace_ignore_input: bool = False,
    trace_ignore_output: bool = False,
    trace_ignore_inputs: list[str] | None = None,
    trace_input_formatter: Callable[..., str] | None = None,
    trace_output_formatter: Callable[..., str] | None = None,
    # prefect passthrough
    name: str | None = None,
    version: str | None = None,
    flow_run_name: Union[Callable[[], str], str] | None = None,
    retries: int | None = None,
    retry_delay_seconds: int | float | None = None,
    task_runner: TaskRunner[PrefectFuture[Any]] | None = None,
    description: str | None = None,
    timeout_seconds: int | float | None = None,
    validate_parameters: bool = True,
    persist_result: bool | None = None,
    result_storage: ResultStorage | str | None = None,
    result_serializer: ResultSerializer | str | None = None,
    cache_result_in_memory: bool = True,
    log_prints: bool | None = None,
    on_completion: list[FlowStateHook[Any, Any]] | None = None,
    on_failure: list[FlowStateHook[Any, Any]] | None = None,
    on_cancellation: list[FlowStateHook[Any, Any]] | None = None,
    on_crashed: list[FlowStateHook[Any, Any]] | None = None,
    on_running: list[FlowStateHook[Any, Any]] | None = None,
) -> Callable[[_DocumentsFlowCallable[FO_contra]], _FlowLike[FO_contra]]: ...


def pipeline_flow(
    __fn: _DocumentsFlowCallable[FO_contra] | None = None,
    /,
    *,
    # tracing
    trace_level: TraceLevel = "always",
    trace_ignore_input: bool = False,
    trace_ignore_output: bool = False,
    trace_ignore_inputs: list[str] | None = None,
    trace_input_formatter: Callable[..., str] | None = None,
    trace_output_formatter: Callable[..., str] | None = None,
    # prefect passthrough
    name: str | None = None,
    version: str | None = None,
    flow_run_name: Union[Callable[[], str], str] | None = None,
    retries: int | None = None,
    retry_delay_seconds: int | float | None = None,
    task_runner: TaskRunner[PrefectFuture[Any]] | None = None,
    description: str | None = None,
    timeout_seconds: int | float | None = None,
    validate_parameters: bool = True,
    persist_result: bool | None = None,
    result_storage: ResultStorage | str | None = None,
    result_serializer: ResultSerializer | str | None = None,
    cache_result_in_memory: bool = True,
    log_prints: bool | None = None,
    on_completion: list[FlowStateHook[Any, Any]] | None = None,
    on_failure: list[FlowStateHook[Any, Any]] | None = None,
    on_cancellation: list[FlowStateHook[Any, Any]] | None = None,
    on_crashed: list[FlowStateHook[Any, Any]] | None = None,
    on_running: list[FlowStateHook[Any, Any]] | None = None,
) -> _FlowLike[FO_contra] | Callable[[_DocumentsFlowCallable[FO_contra]], _FlowLike[FO_contra]]:
    """Decorate an **async** flow.

    Required signature:
        async def flow_fn(
            project_name: str,
            documents: DocumentList,
            flow_options: FlowOptions,  # or any subclass
            *args,
            **kwargs
        ) -> DocumentList

    Returns the same callable object Prefect's ``@flow`` would return.
    """
    flow_decorator: Callable[..., Any] = _prefect_flow

    def _apply(fn: _DocumentsFlowCallable[FO_contra]) -> _FlowLike[FO_contra]:
        fname = _callable_name(fn, "flow")

        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@pipeline_flow '{fname}' must be declared with 'async def'")
        if len(inspect.signature(fn).parameters) < 3:
            raise TypeError(
                f"@pipeline_flow '{fname}' must accept "
                "'project_name, documents, flow_options' as its first three parameters"
            )

        async def _wrapper(
            project_name: str,
            documents: DocumentList,
            flow_options: FO_contra,
            *args: Any,
            **kwargs: Any,
        ) -> DocumentList:
            result = await fn(project_name, documents, flow_options, *args, **kwargs)
            if not isinstance(result, DocumentList):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(
                    f"Flow '{fname}' must return DocumentList, got {type(result).__name__}"
                )
            return result

        # Preserve the original function name for Prefect
        _wrapper.__name__ = fname
        _wrapper.__qualname__ = getattr(fn, "__qualname__", fname)

        traced = trace(
            level=trace_level,
            name=name or fname,
            ignore_input=trace_ignore_input,
            ignore_output=trace_ignore_output,
            ignore_inputs=trace_ignore_inputs,
            input_formatter=trace_input_formatter,
            output_formatter=trace_output_formatter,
        )(_wrapper)

        return cast(
            _FlowLike[FO_contra],
            flow_decorator(
                name=name or fname,
                version=version,
                flow_run_name=flow_run_name or name or fname,
                retries=0 if retries is None else retries,
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
            )(traced),
        )

    return _apply(__fn) if __fn else _apply


__all__ = ["pipeline_task", "pipeline_flow"]
