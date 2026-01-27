"""@public Remote deployment utilities for calling PipelineDeployment flows via Prefect."""

import inspect
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar, cast

from prefect import get_client
from prefect.client.orchestration import PrefectClient
from prefect.client.schemas import FlowRun
from prefect.context import AsyncClientContext
from prefect.deployments.flow_runs import run_deployment
from prefect.exceptions import ObjectNotFound

from ai_pipeline_core.deployment import DeploymentContext, DeploymentResult, PipelineDeployment
from ai_pipeline_core.flow.options import FlowOptions
from ai_pipeline_core.settings import settings
from ai_pipeline_core.tracing import TraceLevel, set_trace_cost, trace

P = ParamSpec("P")
TOptions = TypeVar("TOptions", bound=FlowOptions)
TResult = TypeVar("TResult", bound=DeploymentResult)


def _is_already_traced(func: Callable[..., Any]) -> bool:
    """Check if function or its __wrapped__ has __is_traced__ attribute."""
    if getattr(func, "__is_traced__", False):
        return True
    wrapped = getattr(func, "__wrapped__", None)
    return getattr(wrapped, "__is_traced__", False) if wrapped else False


async def run_remote_deployment(deployment_name: str, parameters: dict[str, Any]) -> Any:
    """Run a remote Prefect deployment, trying local client first then remote."""

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
        raise ValueError(f"{deployment_name} not found, PREFECT_API_URL not set")

    async with PrefectClient(
        api=settings.prefect_api_url,
        api_key=settings.prefect_api_key,
        auth_string=settings.prefect_api_auth_string,
    ) as client:
        try:
            await client.read_deployment_by_name(name=deployment_name)
            ctx = AsyncClientContext.model_construct(
                client=client, _httpx_settings=None, _context_stack=0
            )
            with ctx:
                return await _run(client, False)
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
) -> Callable[[Callable[P, TResult]], Callable[P, TResult]]:
    """@public Decorator to call PipelineDeployment flows remotely with automatic serialization."""

    def decorator(func: Callable[P, TResult]) -> Callable[P, TResult]:
        fname = getattr(func, "__name__", deployment_class.name)

        if _is_already_traced(func):
            raise TypeError(f"@remote_deployment target '{fname}' already has @trace")

        @wraps(func)
        async def _wrapper(*args: P.args, **kwargs: P.kwargs) -> TResult:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Pass parameters with proper types - Prefect handles Pydantic serialization
            parameters: dict[str, Any] = {}
            for pname, value in bound.arguments.items():
                if value is None and pname == "context":
                    parameters[pname] = DeploymentContext()
                else:
                    parameters[pname] = value

            full_name = f"{deployment_class.name}/{deployment_name or deployment_class.name}"

            result = await run_remote_deployment(full_name, parameters)

            if trace_cost is not None and trace_cost > 0:
                set_trace_cost(trace_cost)

            if isinstance(result, DeploymentResult):
                return cast(TResult, result)
            if isinstance(result, dict):
                return cast(TResult, deployment_class.result_type(**result))
            raise TypeError(f"Expected DeploymentResult, got {type(result).__name__}")

        traced_wrapper = trace(
            level=trace_level,
            name=name or deployment_class.name,
        )(_wrapper)

        return traced_wrapper  # type: ignore[return-value]

    return decorator
