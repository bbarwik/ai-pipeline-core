"""Remote deployment utilities for calling PipelineDeployment flows via Prefect."""

import asyncio
import hashlib
import importlib
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, Generic, TypeVar, cast, final
from uuid import UUID, uuid4

from prefect import get_client
from prefect.client.orchestration import PrefectClient
from prefect.client.schemas import FlowRun
from prefect.context import AsyncClientContext
from prefect.deployments.flow_runs import run_deployment
from prefect.exceptions import ObjectNotFound

from ai_pipeline_core.database import Database, ExecutionNode, NodeKind, NodeStatus
from ai_pipeline_core.deployment._helpers import (
    _CLI_FIELDS,
    class_name_to_deployment_name,
    extract_generic_params,
    validate_run_id,
)
from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents._context import RunScope
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.pipeline._execution_context import get_execution_context, get_run_id
from ai_pipeline_core.pipeline.options import FlowOptions
from ai_pipeline_core.settings import settings

from ._resolve import DocumentInput
from .base import DeploymentResult

logger = get_pipeline_logger(__name__)

__all__ = [
    "RemoteDeployment",
]

TOptions = TypeVar("TOptions", bound=FlowOptions)
TResult = TypeVar("TResult", bound=DeploymentResult)

_POLL_INTERVAL = 5.0
_REMOTE_RUN_ID_FINGERPRINT_LENGTH = 8


def _derive_remote_run_id(run_id: str, documents: Sequence[Document], options: FlowOptions) -> str:
    """Deterministic run_id from caller's run_id + input fingerprint.

    Same documents + options produce the same derived run_id (enables worker resume).
    Different inputs produce different derived run_id (prevents collisions).
    """
    sha256s = sorted(doc.sha256 for doc in documents)
    exclude = set(_CLI_FIELDS & set(type(options).model_fields))
    options_json = options.model_dump_json(exclude=exclude, exclude_none=True)
    fingerprint = hashlib.sha256(f"{':'.join(sha256s)}|{options_json}".encode()).hexdigest()[:_REMOTE_RUN_ID_FINGERPRINT_LENGTH]
    return f"{run_id}-{fingerprint}"


def _serialize_document_inputs(documents: Sequence[Document]) -> list[dict[str, Any]]:
    """Normalize documents to the DocumentInput schema used by the Prefect flow."""
    return [DocumentInput.model_validate(doc.serialize_model()).model_dump(mode="json") for doc in documents]


class RemoteDeployment(Generic[TOptions, TResult]):
    """Typed client for calling a remote PipelineDeployment via Prefect.

    Generic parameters:
        TOptions: FlowOptions subclass for the deployment.
        TResult: DeploymentResult subclass returned by the deployment.

    Set ``deployment_class`` to enable inline mode (test/local):
        deployment_class = "module.path:ClassName"
    """

    name: ClassVar[str]
    options_type: ClassVar[type[FlowOptions]]
    result_type: ClassVar[type[DeploymentResult]]
    deployment_class: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Auto-derive name unless explicitly set in class body
        if "name" not in cls.__dict__:
            cls.name = class_name_to_deployment_name(cls.__name__)

        # Extract Generic params: (TOptions, TResult)
        generic_args = extract_generic_params(cls, RemoteDeployment)
        if len(generic_args) < 2:
            raise TypeError(f"{cls.__name__} must specify 2 Generic parameters: class {cls.__name__}(RemoteDeployment[OptionsType, ResultType])")

        options_type, result_type = generic_args[0], generic_args[1]

        if not isinstance(options_type, type) or not issubclass(options_type, FlowOptions):
            raise TypeError(f"{cls.__name__}: first Generic param must be a FlowOptions subclass, got {options_type}")
        if not isinstance(result_type, type) or not issubclass(result_type, DeploymentResult):
            raise TypeError(f"{cls.__name__}: second Generic param must be a DeploymentResult subclass, got {result_type}")

        cls.options_type = options_type
        cls.result_type = result_type

    @property
    def deployment_path(self) -> str:
        """Full Prefect deployment path: '{flow_name}/{deployment_name}'."""
        return f"{self.name}/{self.name.replace('-', '_')}"

    def _resolve_deployment_class(self) -> Any:
        """Import the actual PipelineDeployment class for inline execution."""
        if not self.deployment_class:
            raise ValueError(
                f"{type(self).__name__}.deployment_class is not set. Set deployment_class = 'module.path:ClassName' to enable inline/test execution."
            )
        module_path, class_name = self.deployment_class.rsplit(":", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @final
    async def run(
        self,
        documents: tuple[Document, ...],
        options: TOptions,
    ) -> TResult:
        """Execute the remote deployment.

        Uses inline mode when the active database backend cannot support remote execution,
        and Prefect remote mode when it can.
        """
        run_id = get_run_id()
        validate_run_id(run_id)
        derived_run_id = _derive_remote_run_id(run_id, documents, options)
        validate_run_id(derived_run_id)

        # Get execution context for DAG linking
        exec_ctx = get_execution_context()
        database = exec_ctx.database if exec_ctx else None
        deployment_id = exec_ctx.deployment_id if exec_ctx else None
        root_deployment_id = exec_ctx.root_deployment_id if exec_ctx else None

        # Pre-allocate child deployment ID
        child_deployment_id = uuid4()
        subtask_node_id = uuid4()

        # Create subtask node in caller's DAG
        if exec_ctx is not None and database is not None and deployment_id is not None and root_deployment_id is not None:
            current_node_id = exec_ctx.current_node_id
            sequence_no = exec_ctx.next_child_sequence(current_node_id) if current_node_id else 0
            subtask_node = ExecutionNode(
                node_id=subtask_node_id,
                node_kind=NodeKind.TASK,
                deployment_id=deployment_id,
                root_deployment_id=root_deployment_id,
                parent_node_id=current_node_id or deployment_id,
                run_id=run_id,
                run_scope=exec_ctx.run_scope if exec_ctx else RunScope(run_id),
                deployment_name=exec_ctx.deployment_name if exec_ctx else "",
                name=f"remote:{self.name}",
                sequence_no=sequence_no,
                task_class=type(self).__name__,
                status=NodeStatus.RUNNING,
                remote_child_deployment_id=child_deployment_id,
                input_document_shas=tuple(d.sha256 for d in documents),
            )
            try:
                await database.insert_node(subtask_node)
            except Exception as exc:
                logger.warning("Failed to insert remote subtask node: %s", exc)

        # Determine backend mode
        use_inline = database is None
        inline_reason = "no execution database context is active"
        if database is not None and not database.supports_remote:
            use_inline = True
            inline_reason = "the active database backend does not support remote execution"

        try:
            if use_inline:
                logger.warning(
                    "RemoteDeployment '%s' is falling back to inline execution because %s. "
                    "Configure a non-local deployment database to force Prefect remote execution.",
                    self.name,
                    inline_reason,
                )
                result = await self._run_inline(
                    derived_run_id,
                    documents,
                    options,
                    child_deployment_id=child_deployment_id,
                    root_deployment_id=root_deployment_id or child_deployment_id,
                    parent_deployment_task_id=subtask_node_id,
                    database=cast(Database, database) if database is not None else None,
                    publisher=exec_ctx.publisher if exec_ctx else None,
                    parent_execution_id=exec_ctx.execution_id if exec_ctx else None,
                )
            else:
                result = await self._run_remote(
                    derived_run_id,
                    documents,
                    options,
                    child_deployment_id=child_deployment_id,
                    root_deployment_id=root_deployment_id or child_deployment_id,
                    parent_deployment_task_id=subtask_node_id,
                    parent_execution_id=exec_ctx.execution_id if exec_ctx else None,
                )

            # Update subtask to COMPLETED
            if database is not None:
                try:
                    await database.update_node(
                        subtask_node_id,
                        status=NodeStatus.COMPLETED,
                        ended_at=datetime.now(UTC),
                    )
                except Exception as exc:
                    logger.warning("Failed to update remote subtask node: %s", exc)

            return result

        except (Exception, asyncio.CancelledError) as exc:
            # Update subtask to FAILED
            if database is not None:
                try:
                    await database.update_node(
                        subtask_node_id,
                        status=NodeStatus.FAILED,
                        ended_at=datetime.now(UTC),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                except Exception as update_exc:
                    logger.warning("Failed to update remote subtask node on failure: %s", update_exc)
            raise

    async def _run_inline(
        self,
        run_id: str,
        documents: Sequence[Document],
        options: TOptions,
        *,
        child_deployment_id: UUID,
        root_deployment_id: UUID,
        parent_deployment_task_id: UUID,
        database: Database | None,
        publisher: Any = None,
        parent_execution_id: UUID | None = None,
    ) -> TResult:
        """Run the deployment inline (same process) for test/local mode."""
        deployment_cls = self._resolve_deployment_class()
        deployment_instance = deployment_cls()

        result = await deployment_instance._run_with_context(
            run_id,
            documents,
            options,
            deployment_node_id=child_deployment_id,
            root_deployment_id=root_deployment_id,
            parent_deployment_task_id=parent_deployment_task_id,
            publisher=publisher,
            parent_execution_id=parent_execution_id,
            database=database,
        )

        if isinstance(result, DeploymentResult):
            return cast(TResult, result)
        if isinstance(result, dict):
            return cast(TResult, self.result_type.model_validate(result))
        raise TypeError(f"Inline deployment '{self.name}' returned unexpected type: {type(result).__name__}")

    async def _run_remote(
        self,
        run_id: str,
        documents: Sequence[Document],
        options: TOptions,
        *,
        child_deployment_id: UUID,
        root_deployment_id: UUID,
        parent_deployment_task_id: UUID,
        parent_execution_id: UUID | None = None,
    ) -> TResult:
        """Run the deployment remotely via Prefect."""
        parameters: dict[str, Any] = {
            "run_id": run_id,
            "document_inputs": _serialize_document_inputs(documents),
            "options": options,
            "parent_execution_id": str(parent_execution_id) if parent_execution_id is not None else None,
            "deployment_node_id": str(child_deployment_id),
            "parent_deployment_task_id": str(parent_deployment_task_id),
            "root_deployment_id": str(root_deployment_id),
            "input_document_sha256s": [doc.sha256 for doc in documents],
        }

        result = await _run_remote_deployment(
            self.deployment_path,
            parameters,
        )

        if isinstance(result, DeploymentResult):
            return cast(TResult, result)
        if isinstance(result, dict):
            return cast(TResult, self.result_type.model_validate(result))
        raise TypeError(f"Remote deployment '{self.name}' returned unexpected type: {type(result).__name__}")


async def _run_remote_deployment(
    deployment_name: str,
    parameters: dict[str, Any],
) -> Any:
    """Run a remote Prefect deployment and poll until completion."""
    async with get_client() as client:
        try:
            await client.read_deployment_by_name(name=deployment_name)
            fr: FlowRun = await run_deployment(  # type: ignore[assignment]
                client=client,
                name=deployment_name,
                parameters=parameters,
                as_subflow=True,
                timeout=0,
            )
            return await _poll_remote_flow_run(client, cast(UUID, fr.id))
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
                fr = await run_deployment(  # type: ignore[assignment]
                    client=client,
                    name=deployment_name,
                    parameters=parameters,
                    as_subflow=False,
                    timeout=0,
                )
                return await _poll_remote_flow_run(client, cast(UUID, fr.id))
        except ObjectNotFound:
            pass

    raise ValueError(f"{deployment_name} deployment not found")


async def _poll_remote_flow_run(
    client: PrefectClient,
    flow_run_id: UUID,
    *,
    poll_interval: float = _POLL_INTERVAL,
) -> Any:
    """Poll a remote flow run until final state."""
    while True:
        try:
            flow_run = await client.read_flow_run(flow_run_id)
        except Exception:
            logger.warning("Failed to poll remote flow run %s", flow_run_id, exc_info=True)
            await asyncio.sleep(poll_interval)
            continue

        state = flow_run.state
        if state is not None and state.is_final():
            return await state.result()

        await asyncio.sleep(poll_interval)
