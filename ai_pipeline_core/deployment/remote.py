"""Remote deployment utilities for calling PipelineDeployment flows via Prefect."""

import asyncio
import types
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, Generic, TypeVar, cast, final
from uuid import UUID

from prefect import get_client
from prefect.client.orchestration import PrefectClient
from prefect.client.schemas import FlowRun
from prefect.context import AsyncClientContext
from prefect.deployments.flow_runs import run_deployment
from prefect.exceptions import ObjectNotFound

from ai_pipeline_core.deployment import DeploymentContext, DeploymentResult
from ai_pipeline_core.deployment._helpers import class_name_to_deployment_name, extract_generic_params
from ai_pipeline_core.documents import Document
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability.tracing import TraceLevel, set_trace_cost, trace
from ai_pipeline_core.pipeline.options import FlowOptions
from ai_pipeline_core.settings import settings

logger = get_pipeline_logger(__name__)

TDoc = TypeVar("TDoc", bound=Document)
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
        if state is not None and state.is_final():
            if on_progress and state.is_completed():
                await on_progress(1.0, f"[{deployment_name}] Completed")
            return await state.result()

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
        fr: FlowRun = await run_deployment(  # type: ignore[assignment]
            client=client,
            name=deployment_name,
            parameters=parameters,
            as_subflow=as_subflow,
            timeout=0,
        )
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


# ---------------------------------------------------------------------------
# RemoteDeployment class
# ---------------------------------------------------------------------------


def _validate_document_type(cls_name: str, doc_type: Any) -> None:
    """Validate TDoc is a Document subclass or union of Document subclasses."""
    if isinstance(doc_type, types.UnionType):
        for member in doc_type.__args__:
            if not isinstance(member, type) or not issubclass(member, Document):
                raise TypeError(
                    f"{cls_name}: all TDoc union members must be Document subclasses, got {member.__name__ if isinstance(member, type) else member}"
                )
    elif isinstance(doc_type, type):
        if not issubclass(doc_type, Document):
            raise TypeError(f"{cls_name}: TDoc must be a Document subclass, got {doc_type.__name__}")
    else:
        raise TypeError(f"{cls_name}: TDoc must be a Document type or union of Document types, got {type(doc_type).__name__}")


class RemoteDeployment(Generic[TDoc, TOptions, TResult]):
    """Typed client for calling a remote PipelineDeployment via Prefect.

    Name your client class identically to the server's PipelineDeployment
    subclass so the auto-derived deployment name matches.

    Generic parameters:
        TDoc: Document types accepted as input (single type or union).
        TOptions: FlowOptions subclass for the deployment.
        TResult: DeploymentResult subclass returned by the deployment.

    Usage::

        class AiResearch(RemoteDeployment[
            ResearchTaskDocument | ContextDocument,
            FlowOptions,
            AiResearchResult,
        ]): pass

        _client = AiResearch()
        result = await _client.run("project", docs, FlowOptions())
    """

    name: ClassVar[str]
    options_type: ClassVar[type[FlowOptions]]
    result_type: ClassVar[type[DeploymentResult]]

    trace_level: ClassVar[TraceLevel] = "always"
    trace_cost: ClassVar[float | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Auto-derive name unless explicitly set in class body
        if "name" not in cls.__dict__:
            cls.name = class_name_to_deployment_name(cls.__name__)

        # Extract Generic params: (TDoc, TOptions, TResult)
        generic_args = extract_generic_params(cls, RemoteDeployment)
        if len(generic_args) < 3 or any(a is None for a in generic_args):
            raise TypeError(f"{cls.__name__} must specify 3 Generic parameters: class {cls.__name__}(RemoteDeployment[DocType, OptionsType, ResultType])")

        doc_type, options_type, result_type = generic_args

        _validate_document_type(cls.__name__, doc_type)

        if not isinstance(options_type, type) or not issubclass(options_type, FlowOptions):
            raise TypeError(f"{cls.__name__}: second Generic param must be a FlowOptions subclass, got {options_type}")
        if not isinstance(result_type, type) or not issubclass(result_type, DeploymentResult):
            raise TypeError(f"{cls.__name__}: third Generic param must be a DeploymentResult subclass, got {result_type}")

        cls.options_type = options_type
        cls.result_type = result_type

        # Apply @trace to _execute: combined guard prevents no-op and double-wrap
        trace_level = getattr(cls, "trace_level", "always")
        if trace_level != "off" and not _is_already_traced(cls._execute):
            cls._execute = trace(name=cls.name, level=trace_level)(cls._execute)  # type: ignore[assignment]

    @property
    def deployment_path(self) -> str:
        """Full Prefect deployment path: '{flow_name}/{deployment_name}'."""
        return f"{self.name}/{self.name.replace('-', '_')}"

    async def _execute(
        self,
        project_name: str,
        documents: list[TDoc],
        options: TOptions,
        context: DeploymentContext,
        on_progress: ProgressCallback | None,
    ) -> TResult:
        """Serialize, call Prefect, deserialize. Wrapped with @trace in __init_subclass__."""
        parameters: dict[str, Any] = {
            "project_name": project_name,
            "documents": [doc.serialize_model() for doc in documents],
            "options": options,
            "context": context,
        }

        result = await run_remote_deployment(
            self.deployment_path,
            parameters,
            on_progress=on_progress,
        )

        if self.trace_cost is not None and self.trace_cost > 0:
            set_trace_cost(self.trace_cost)

        if isinstance(result, DeploymentResult):
            return cast(TResult, result)
        if isinstance(result, dict):
            return cast(TResult, self.result_type.model_validate(result))
        raise TypeError(f"Remote deployment '{self.name}' returned unexpected type: {type(result).__name__}. Expected DeploymentResult or dict.")

    @final
    async def run(
        self,
        project_name: str,
        documents: list[TDoc],
        options: TOptions,
        context: DeploymentContext | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> TResult:
        """Execute the remote deployment via Prefect."""
        return await self._execute(
            project_name,
            documents,
            options,
            context if context is not None else DeploymentContext(),
            on_progress,
        )
