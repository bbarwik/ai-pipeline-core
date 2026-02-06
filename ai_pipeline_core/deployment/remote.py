"""Remote deployment utilities for calling PipelineDeployment flows via Prefect."""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from functools import wraps
from typing import Any, TypeVar, cast
from uuid import UUID

from prefect import get_client
from prefect.client.orchestration import PrefectClient
from prefect.client.schemas import FlowRun
from prefect.context import AsyncClientContext
from prefect.deployments.flow_runs import run_deployment
from prefect.exceptions import ObjectNotFound

from ai_pipeline_core.deployment import DeploymentContext, DeploymentResult, PipelineDeployment
from ai_pipeline_core.documents import Document
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability.tracing import TraceLevel, set_trace_cost, trace
from ai_pipeline_core.pipeline.options import FlowOptions
from ai_pipeline_core.settings import settings

logger = get_pipeline_logger(__name__)

TOptions = TypeVar("TOptions", bound=FlowOptions)
TResult = TypeVar("TResult", bound=DeploymentResult)

ProgressCallback = Callable[[float, str], Awaitable[None]]
"""Signature for remote deployment progress callbacks: (fraction, message) -> None."""


def _is_already_traced(func: Callable[..., Any]) -> bool:
    """Check if function or its __wrapped__ has __is_traced__ attribute."""
    if getattr(func, "__is_traced__", False):
        return True
    wrapped = getattr(func, "__wrapped__", None)
    return getattr(wrapped, "__is_traced__", False) if wrapped else False


_POLL_INTERVAL = 5.0


async def _poll_remote_flow_run(
    client: PrefectClient,
    flow_run_id: UUID,
    deployment_name: str,
    poll_interval: float = _POLL_INTERVAL,
    on_progress: ProgressCallback | None = None,
) -> Any:
    """Poll a remote flow run until final, invoking on_progress callback with progress.

    Reads the remote flow run's progress labels on each poll cycle and calls
    on_progress(fraction, message) if provided. Without a callback, no progress
    is reported. Only sends 1.0 on successful completion (not failure).
    """
    last_fraction = 0.0

    while True:
        try:
            flow_run = await client.read_flow_run(flow_run_id)
        except Exception:
            logger.warning("Failed to poll remote flow run %s", flow_run_id, exc_info=True)
            await asyncio.sleep(poll_interval)
            continue

        state = flow_run.state
        if state and state.is_final():
            if on_progress and state.is_completed():
                await on_progress(1.0, f"[{deployment_name}] Completed")
            return await state.result()  # type: ignore[union-attr]

        if on_progress:
            labels: dict[str, Any] = flow_run.labels or {}
            progress_val = labels.get("progress.progress")

            if progress_val is not None:
                fraction = max(float(progress_val), last_fraction)
                last_fraction = fraction
                flow_name = str(labels.get("progress.flow_name", ""))
                message = str(labels.get("progress.message", ""))
                display = f"[{deployment_name}] {flow_name}: {message}" if flow_name else f"[{deployment_name}] Running"
                await on_progress(fraction, display)
            else:
                await on_progress(last_fraction, f"[{deployment_name}] Waiting to start")

        await asyncio.sleep(poll_interval)


async def run_remote_deployment(
    deployment_name: str,
    parameters: dict[str, Any],
    on_progress: ProgressCallback | None = None,
) -> Any:
    """Run a remote Prefect deployment with optional progress callback.

    Creates the remote flow run immediately (timeout=0) then polls its state,
    invoking on_progress(fraction, message) on each poll cycle if provided.
    """

    async def _create_and_poll(client: PrefectClient, as_subflow: bool) -> Any:
        fr: FlowRun = await run_deployment(
            client=client,
            name=deployment_name,
            parameters=parameters,
            as_subflow=as_subflow,
            timeout=0,
        )  # type: ignore
        return await _poll_remote_flow_run(client, fr.id, deployment_name, on_progress=on_progress)

    async with get_client() as client:
        try:
            await client.read_deployment_by_name(name=deployment_name)
            return await _create_and_poll(client, True)  # noqa: FBT003
        except ObjectNotFound:
            pass

    if not settings.prefect_api_url:
        raise ValueError(f"{deployment_name} not found, PREFECT_API_URL not set")

    async with PrefectClient(
        api=settings.prefect_api_url,
        api_key=settings.prefect_api_key,
        auth_string=settings.prefect_api_auth_string,
    ) as client:
        try:
            await client.read_deployment_by_name(name=deployment_name)
            ctx = AsyncClientContext.model_construct(client=client, _httpx_settings=None, _context_stack=0)
            with ctx:
                return await _create_and_poll(client, False)  # noqa: FBT003
        except ObjectNotFound:
            pass

    raise ValueError(f"{deployment_name} deployment not found")


def remote_deployment(
    deployment_class: type[PipelineDeployment[TOptions, TResult]],
    *,
    deployment_name: str | None = None,
    name: str | None = None,
    trace_level: TraceLevel = "always",
    trace_cost: float | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Coroutine[Any, Any, TResult]]]:
    """Decorator to call PipelineDeployment flows remotely with automatic serialization.

    The decorated function's body is never executed — it serves as a typed stub.
    The wrapper enforces the deployment contract: (project_name, documents, options, context).
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Coroutine[Any, Any, TResult]]:
        fname = getattr(func, "__name__", deployment_class.name)

        if _is_already_traced(func):
            raise TypeError(f"@remote_deployment target '{fname}' already has @trace")

        @wraps(func)
        async def _wrapper(
            project_name: str,
            documents: list[Document],
            options: TOptions,
            context: DeploymentContext | None = None,
            on_progress: ProgressCallback | None = None,
        ) -> TResult:
            parameters: dict[str, Any] = {
                "project_name": project_name,
                "documents": documents,
                "options": options,
                "context": context if context is not None else DeploymentContext(),
            }

            full_name = f"{deployment_class.name}/{deployment_name or deployment_class.name.replace('-', '_')}"

            result = await run_remote_deployment(full_name, parameters, on_progress=on_progress)

            if trace_cost is not None and trace_cost > 0:
                set_trace_cost(trace_cost)

            if isinstance(result, DeploymentResult):
                return cast(TResult, result)
            if isinstance(result, dict):
                return cast(TResult, deployment_class.result_type(**cast(dict[str, Any], result)))
            raise TypeError(f"Expected DeploymentResult, got {type(result).__name__}")

        traced_wrapper = trace(
            level=trace_level,
            name=name or deployment_class.name,
        )(_wrapper)

        return traced_wrapper

    return decorator
