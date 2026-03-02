"""Remote deployment utilities for calling PipelineDeployment flows via Prefect."""

import asyncio
import hashlib
import json
import types
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, cast, final
from uuid import UUID

from opentelemetry import trace as otel_trace
from prefect import get_client
from prefect.client.orchestration import PrefectClient
from prefect.client.schemas import FlowRun
from prefect.context import AsyncClientContext
from prefect.deployments.flow_runs import run_deployment
from prefect.exceptions import ObjectNotFound

from ai_pipeline_core.deployment import DeploymentResult
from ai_pipeline_core.deployment._helpers import (
    _CLI_FIELDS,
    class_name_to_deployment_name,
    extract_generic_params,
    init_observability_best_effort,
    validate_run_id,
)
from ai_pipeline_core.deployment._resolve import AttachmentInput, DocumentInput
from ai_pipeline_core.deployment._task_results import ClickHouseTaskResultStore
from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents._context import get_run_context
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._span_data import ATTR_INPUT_DOC_SHA256S, ATTR_OUTPUT_DOC_SHA256S
from ai_pipeline_core.observability.tracing import TraceLevel, set_trace_cost, trace
from ai_pipeline_core.pipeline._type_validation import is_already_traced
from ai_pipeline_core.pipeline.options import FlowOptions
from ai_pipeline_core.settings import settings

logger = get_pipeline_logger(__name__)

__all__ = [
    "ProgressCallback",
    "RemoteDeployment",
    "run_remote_deployment",
]

TDoc = TypeVar("TDoc", bound=Document)
TOptions = TypeVar("TOptions", bound=FlowOptions)
TResult = TypeVar("TResult", bound=DeploymentResult)

ProgressCallback = Callable[[float, str], Awaitable[None]]
"""Signature for remote deployment progress callbacks: (fraction, message) -> None."""


def _strip_for_prefect(serialized: dict[str, Any]) -> dict[str, Any]:
    """Strip serialize_model() metadata for Prefect parameter validation.

    Prefect validates parameters against JSON schema server-side (additionalProperties: false)
    before DocumentInput._strip_serialize_metadata can run on the Python side.
    """
    d = {k: v for k, v in serialized.items() if k not in DocumentInput.STRIP_KEYS}
    if d.get("attachments"):
        d["attachments"] = [{k: v for k, v in att.items() if k not in AttachmentInput.STRIP_KEYS} for att in d["attachments"]]
    return d


_POLL_INTERVAL = 5.0
_REMOTE_RUN_ID_FINGERPRINT_LENGTH = 8


def _derive_remote_run_id(run_id: str, documents: Sequence[Document], options: FlowOptions) -> str:
    """Deterministic run_id from caller's run_id + input fingerprint.

    Same documents + options produce the same derived run_id (enables worker resume).
    Different inputs produce different derived run_id (prevents task_results collisions).
    CLI-specific fields (working_directory, start, end, etc.) are excluded from the
    fingerprint to match _compute_run_scope behavior.
    """
    sha256s = sorted(doc.sha256 for doc in documents)
    exclude = set(_CLI_FIELDS & set(type(options).model_fields))
    options_json = options.model_dump_json(exclude=exclude, exclude_none=True)
    fingerprint = hashlib.sha256(f"{':'.join(sha256s)}|{options_json}".encode()).hexdigest()[:_REMOTE_RUN_ID_FINGERPRINT_LENGTH]
    return f"{run_id}-{fingerprint}"


async def _get_completed_result(state: Any, *, run_id: str | None, deployment_name: str) -> Any:
    """Get result from a completed state, falling back to ClickHouse if Prefect storage fails."""
    try:
        return await state.result()
    except Exception:
        if run_id and settings.clickhouse_host:
            ch_result = await _read_from_task_results(run_id)
            if ch_result is not None:
                logger.info("Retrieved result from ClickHouse fallback for run_id=%s (deployment: %s)", run_id, deployment_name)
                return ch_result
            logger.warning(
                "ClickHouse fallback found no result for run_id=%s. Ensure the remote worker has CLICKHOUSE_HOST configured.",
                run_id,
            )
        raise


async def _poll_remote_flow_run(
    client: PrefectClient,
    flow_run_id: UUID,
    deployment_name: str,
    *,
    poll_interval: float = _POLL_INTERVAL,
    on_progress: ProgressCallback | None = None,
    run_id: str | None = None,
) -> Any:
    """Poll a remote flow run until final, invoking on_progress callback with progress.

    Reads the remote flow run's progress labels on each poll cycle and calls
    on_progress(fraction, message) if provided. Without a callback, no progress
    is reported. Only sends 1.0 on successful completion (not failure).

    When run_id is provided and the run completed successfully but state.result()
    fails (e.g. GCS credential error), falls back to reading from ClickHouse
    task_results table.
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
            if state.is_completed():
                return await _get_completed_result(state, run_id=run_id, deployment_name=deployment_name)
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


async def _read_from_task_results(run_id: str) -> dict[str, Any] | None:
    """Best-effort read from ClickHouse task_results as fallback for state.result() failures."""
    try:
        store = ClickHouseTaskResultStore(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            database=settings.clickhouse_database,
            username=settings.clickhouse_user,
            password=settings.clickhouse_password,
            secure=settings.clickhouse_secure,
            connect_timeout=settings.clickhouse_connect_timeout,
            send_receive_timeout=settings.clickhouse_send_receive_timeout,
        )
        try:
            record = await store.read_result(run_id)
            return json.loads(record.result) if record else None
        finally:
            store.shutdown()
    except Exception as e:
        logger.warning("ClickHouse fallback read failed for run_id=%s: %s", run_id, e)
        return None


async def run_remote_deployment(
    deployment_name: str,
    parameters: dict[str, Any],
    on_progress: ProgressCallback | None = None,
    run_id: str | None = None,
) -> Any:
    """Run a remote Prefect deployment with optional progress callback.

    Creates the remote flow run immediately (timeout=0) then polls its state,
    invoking on_progress(fraction, message) on each poll cycle if provided.

    When run_id is provided, it is passed to the polling function to enable
    ClickHouse fallback for result retrieval.
    """

    async def _create_and_poll(client: PrefectClient, as_subflow: bool) -> Any:
        fr: FlowRun = await run_deployment(  # type: ignore[assignment]
            client=client,
            name=deployment_name,
            parameters=parameters,
            as_subflow=as_subflow,
            timeout=0,
        )
        return await _poll_remote_flow_run(client, cast(UUID, fr.id), deployment_name, on_progress=on_progress, run_id=run_id)

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

    Derives worker run_id as ``{run_id}-{fingerprint[:8]}`` from input documents
    and options for resume and collision prevention.

    Name your client class identically to the server's PipelineDeployment
    subclass so the auto-derived deployment name matches.

    Generic parameters:
        TDoc: Document types accepted as input (single type or union).
        TOptions: FlowOptions subclass for the deployment.
        TResult: DeploymentResult subclass returned by the deployment.

    Mirror type contract:
        The client defines local Document subclasses ('mirror types') whose class_name must
        match the remote pipeline's document types exactly. When the remote returns documents,
        they are deserialized using the local mirror types. If class names don't match,
        documents are silently skipped with a warning log.

        Tasks that return mirror-typed remote results should use persist_result=False
        to avoid polluting the DocumentStore with unknown class_name entries.
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
        if len(generic_args) < 3:
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
        if trace_level != "off" and not is_already_traced(cls._execute):
            cls._execute = trace(name=cls.name, level=trace_level)(cls._execute)  # type: ignore[assignment]

    @property
    def deployment_path(self) -> str:
        """Full Prefect deployment path: '{flow_name}/{deployment_name}'."""
        return f"{self.name}/{self.name.replace('-', '_')}"

    async def _execute(
        self,
        run_id: str,
        documents: list[TDoc],
        options: TOptions,
        on_progress: ProgressCallback | None,
    ) -> TResult:
        """Serialize, call Prefect, deserialize, track document lineage.

        Wrapped with @trace in __init_subclass__. Tracks input/output document
        SHA256s on the OTel span, matching @pipeline_task/@pipeline_flow behavior.
        """
        validate_run_id(run_id)
        derived_run_id = _derive_remote_run_id(run_id, documents, options)
        validate_run_id(derived_run_id)

        # Extract parent lineage from RunContext and current OTel span
        run_ctx = get_run_context()
        parent_exec_id = str(run_ctx.execution_id) if run_ctx and run_ctx.execution_id else None
        parent_span_hex: str | None = None
        try:
            current_span = otel_trace.get_current_span()
            span_ctx = current_span.get_span_context() if current_span else None
            if span_ctx and isinstance(span_ctx.span_id, int) and span_ctx.span_id:
                parent_span_hex = format(span_ctx.span_id, "016x")
        except Exception:
            logger.debug("Failed to extract parent span context for lineage tracking")

        parameters: dict[str, Any] = {
            "run_id": derived_run_id,
            "documents": [_strip_for_prefect(doc.serialize_model()) for doc in documents],
            "options": options,
            "parent_execution_id": parent_exec_id,
            "parent_span_id": parent_span_hex,
        }

        result = await run_remote_deployment(
            self.deployment_path,
            parameters,
            on_progress=on_progress,
            run_id=derived_run_id,
        )

        if self.trace_cost is not None and self.trace_cost > 0:
            set_trace_cost(self.trace_cost)

        if isinstance(result, DeploymentResult):
            typed_result = cast(TResult, result)
        elif isinstance(result, dict):
            typed_result = cast(TResult, self.result_type.model_validate(result))
        else:
            raise TypeError(f"Remote deployment '{self.name}' returned unexpected type: {type(result).__name__}. Expected DeploymentResult or dict.")

        # Track document lineage on the current span (matching @pipeline_task/@pipeline_flow)
        try:
            span = otel_trace.get_current_span()
            input_sha256s = [doc.sha256 for doc in documents]
            if input_sha256s:
                span.set_attribute(ATTR_INPUT_DOC_SHA256S, input_sha256s)
            if typed_result.documents:
                output_sha256s = [d.sha256 for d in typed_result.documents]
                span.set_attribute(ATTR_OUTPUT_DOC_SHA256S, output_sha256s)
        except Exception:
            logger.debug("Failed to track document lineage", exc_info=True)

        return typed_result

    @final
    async def run(
        self,
        run_id: str,
        documents: list[TDoc],
        options: TOptions,
        on_progress: ProgressCallback | None = None,
    ) -> TResult:
        """Execute the remote deployment via Prefect."""
        return await self._execute(
            run_id,
            documents,
            options,
            on_progress,
        )

    @final
    async def run_traced(
        self,
        run_id: str,
        documents: list[TDoc],
        options: TOptions,
        output_dir: Path,
        on_progress: ProgressCallback | None = None,
    ) -> TResult:
        """Execute with local .trace/ debug tracing at output_dir/.trace.

        Sets up FilesystemBackend + PipelineSpanProcessor for local debug
        output, matching the CLI pipeline tracing pattern. Use when running
        RemoteDeployment standalone (not inside a caller pipeline with tracing).
        """
        from ai_pipeline_core.deployment._cli import _init_debug_tracing
        from ai_pipeline_core.observability._initialization import get_clickhouse_backend

        init_observability_best_effort()
        debug_backend = _init_debug_tracing(output_dir)
        try:
            return await self._execute(run_id, documents, options, on_progress)
        finally:
            ch_backend = get_clickhouse_backend()
            if ch_backend:
                ch_backend.shutdown()
            if debug_backend:
                debug_backend.shutdown()
