"""Core classes for pipeline deployments.

@public

Provides the PipelineDeployment base class and related types for
creating unified, type-safe pipeline deployments with:
- Per-flow caching (skip if outputs exist)
- Per-flow uploads (immediate, not just at end)
- Prefect state hooks (on_running, on_completion, etc.)
- Smart storage provisioning (override provision_storage)
- Upload on failure (partial results saved)
"""

import asyncio
import os
import re
import sys
from abc import abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, ClassVar, Generic, Protocol, TypeVar, cast, final
from uuid import UUID

import httpx
from lmnr import Laminar
from prefect import get_client
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import CliPositionalArg, SettingsConfigDict

from ai_pipeline_core.documents import DocumentList
from ai_pipeline_core.flow.options import FlowOptions
from ai_pipeline_core.logging import get_pipeline_logger, setup_logging
from ai_pipeline_core.prefect import disable_run_logger, flow, prefect_test_harness
from ai_pipeline_core.settings import settings

from .contract import CompletedRun, DeploymentResultData, FailedRun, ProgressRun
from .helpers import (
    StatusPayload,
    class_name_to_deployment_name,
    download_documents,
    extract_generic_params,
    send_webhook,
    upload_documents,
)

logger = get_pipeline_logger(__name__)


class DeploymentContext(BaseModel):
    """@public Infrastructure configuration for deployments.

    Webhooks are optional - provide URLs to enable:
    - progress_webhook_url: Per-flow progress (started/completed/cached)
    - status_webhook_url: Prefect state transitions (RUNNING/FAILED/etc)
    - completion_webhook_url: Final result when deployment ends
    """

    input_documents_urls: list[str] = Field(default_factory=list)
    output_documents_urls: dict[str, str] = Field(default_factory=dict)

    progress_webhook_url: str = ""
    status_webhook_url: str = ""
    completion_webhook_url: str = ""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DeploymentResult(BaseModel):
    """@public Base class for deployment results."""

    success: bool
    error: str | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


TOptions = TypeVar("TOptions", bound=FlowOptions)
TResult = TypeVar("TResult", bound=DeploymentResult)


class FlowCallable(Protocol):
    """Protocol for @pipeline_flow decorated functions."""

    config: Any
    name: str
    __name__: str

    def __call__(
        self, project_name: str, documents: DocumentList, flow_options: FlowOptions
    ) -> Any: ...

    def with_options(self, **kwargs: Any) -> "FlowCallable":
        """Return a copy with overridden Prefect flow options."""
        ...


@dataclass(slots=True)
class _StatusWebhookHook:
    """Prefect hook that sends status webhooks on state transitions."""

    webhook_url: str
    flow_run_id: str
    project_name: str
    step: int
    total_steps: int
    flow_name: str

    async def __call__(self, flow: Any, flow_run: Any, state: Any) -> None:
        payload: StatusPayload = {
            "type": "status",
            "flow_run_id": str(flow_run.id),
            "project_name": self.project_name,
            "step": self.step,
            "total_steps": self.total_steps,
            "flow_name": self.flow_name,
            "state": state.type.value if hasattr(state.type, "value") else str(state.type),
            "state_name": state.name or "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(self.webhook_url, json=payload)
        except Exception as e:
            logger.warning(f"Status webhook failed: {e}")


class PipelineDeployment(Generic[TOptions, TResult]):
    """@public Base class for pipeline deployments.

    Features enabled by default when URLs/storage provided:
    - Per-flow caching: Skip flows if outputs exist in storage
    - Per-flow uploads: Upload documents after each flow
    - Prefect hooks: Attach state hooks if status_webhook_url provided
    - Upload on failure: Save partial results if pipeline fails
    """

    flows: ClassVar[list[FlowCallable]]
    name: ClassVar[str]
    options_type: ClassVar[type[FlowOptions]]
    result_type: ClassVar[type[DeploymentResult]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "flows"):
            return

        if cls.__name__.startswith("Test"):
            raise TypeError(f"Deployment class name cannot start with 'Test': {cls.__name__}")

        cls.name = class_name_to_deployment_name(cls.__name__)

        options_type, result_type = extract_generic_params(cls)
        if options_type is None or result_type is None:
            raise TypeError(
                f"{cls.__name__} must specify Generic parameters: "
                f"class {cls.__name__}(PipelineDeployment[MyOptions, MyResult])"
            )

        cls.options_type = options_type
        cls.result_type = result_type

        if not cls.flows:
            raise TypeError(f"{cls.__name__}.flows cannot be empty")

    @staticmethod
    @abstractmethod
    def build_result(project_name: str, documents: DocumentList, options: TOptions) -> TResult:
        """Extract typed result from accumulated pipeline documents."""
        ...

    async def provision_storage(
        self,
        project_name: str,
        documents: DocumentList,
        options: TOptions,
        context: DeploymentContext,
    ) -> str:
        """Provision GCS storage bucket based on project name and content hash.

        Default: Creates `{project}-{date}-{hash}` bucket on GCS.
        Returns empty string if GCS is unavailable or creation fails.
        Override for custom storage provisioning logic.
        """
        if not documents:
            return ""

        try:
            from ai_pipeline_core.storage.storage import GcsStorage  # noqa: PLC0415
        except ImportError:
            return ""

        content_hash = sha256(b"".join(sorted(d.content for d in documents))).hexdigest()[:6]
        base = re.sub(r"[^a-z0-9-]", "-", project_name.lower()).strip("-") or "project"
        today = datetime.now(timezone.utc).strftime("%y-%m-%d")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%y-%m-%d")

        today_bucket = f"{base[:30]}-{today}-{content_hash}"
        yesterday_bucket = f"{base[:30]}-{yesterday}-{content_hash}"

        # Try today's bucket, then yesterday's, then create new
        for bucket_name in (today_bucket, yesterday_bucket):
            try:
                storage = GcsStorage(bucket_name)
                if await storage.list(recursive=False):
                    logger.info(f"Using existing bucket: {bucket_name}")
                    return f"gs://{bucket_name}"
            except Exception:
                continue

        try:
            storage = GcsStorage(today_bucket)
            await storage.create_bucket()
            logger.info(f"Created new bucket: {today_bucket}")
            return f"gs://{today_bucket}"
        except Exception as e:
            logger.warning(f"Failed to provision GCS storage: {e}")
            return ""

    async def _load_cached_output(
        self, flow_fn: FlowCallable, storage_uri: str
    ) -> DocumentList | None:
        """Load cached outputs if they exist. Override for custom cache logic."""
        try:
            output_type = flow_fn.config.OUTPUT_DOCUMENT_TYPE
            docs = await flow_fn.config.load_documents_by_type(storage_uri, [output_type])
            return docs if docs else None
        except Exception:
            return None

    def _build_status_hooks(
        self,
        context: DeploymentContext,
        flow_run_id: str,
        project_name: str,
        step: int,
        total_steps: int,
        flow_name: str,
    ) -> dict[str, list[Callable[..., Any]]]:
        """Build Prefect hooks for status webhooks."""
        hook = _StatusWebhookHook(
            webhook_url=context.status_webhook_url,
            flow_run_id=flow_run_id,
            project_name=project_name,
            step=step,
            total_steps=total_steps,
            flow_name=flow_name,
        )
        return {
            "on_running": [hook],
            "on_completion": [hook],
            "on_failure": [hook],
            "on_crashed": [hook],
            "on_cancellation": [hook],
        }

    async def _send_progress(
        self,
        context: DeploymentContext,
        flow_run_id: str,
        project_name: str,
        storage_uri: str,
        step: int,
        total_steps: int,
        flow_name: str,
        status: str,
        step_progress: float = 0.0,
        message: str = "",
    ) -> None:
        """Send progress webhook and update flow run labels."""
        progress = round((step - 1 + step_progress) / total_steps, 4)

        if context.progress_webhook_url:
            payload = ProgressRun(
                flow_run_id=UUID(flow_run_id) if flow_run_id else UUID(int=0),
                project_name=project_name,
                state="RUNNING",
                timestamp=datetime.now(timezone.utc),
                storage_uri=storage_uri,
                step=step,
                total_steps=total_steps,
                flow_name=flow_name,
                status=status,
                progress=progress,
                step_progress=round(step_progress, 4),
                message=message,
            )
            try:
                await send_webhook(context.progress_webhook_url, payload)
            except Exception as e:
                logger.warning(f"Progress webhook failed: {e}")

        if flow_run_id:
            try:
                async with get_client() as client:
                    await client.update_flow_run_labels(
                        flow_run_id=UUID(flow_run_id),
                        labels={
                            "progress.step": step,
                            "progress.total_steps": total_steps,
                            "progress.flow_name": flow_name,
                            "progress.status": status,
                            "progress.progress": progress,
                            "progress.step_progress": round(step_progress, 4),
                            "progress.message": message,
                        },
                    )
            except Exception as e:
                logger.warning(f"Progress label update failed: {e}")

    async def _send_completion(
        self,
        context: DeploymentContext,
        flow_run_id: str,
        project_name: str,
        storage_uri: str,
        result: TResult | None,
        error: str | None,
    ) -> None:
        """Send completion webhook."""
        if not context.completion_webhook_url:
            return
        try:
            now = datetime.now(timezone.utc)
            frid = UUID(flow_run_id) if flow_run_id else UUID(int=0)
            payload: CompletedRun | FailedRun
            if result is not None:
                payload = CompletedRun(
                    flow_run_id=frid,
                    project_name=project_name,
                    timestamp=now,
                    storage_uri=storage_uri,
                    state="COMPLETED",
                    result=DeploymentResultData.model_validate(result.model_dump()),
                )
            else:
                payload = FailedRun(
                    flow_run_id=frid,
                    project_name=project_name,
                    timestamp=now,
                    storage_uri=storage_uri,
                    state="FAILED",
                    error=error or "Unknown error",
                )
            await send_webhook(context.completion_webhook_url, payload)
        except Exception as e:
            logger.warning(f"Completion webhook failed: {e}")

    @final
    async def run(
        self,
        project_name: str,
        documents: str | DocumentList,
        options: TOptions,
        context: DeploymentContext,
    ) -> TResult:
        """Execute flows with caching, uploads, and webhooks enabled by default."""
        from prefect import runtime  # noqa: PLC0415

        total_steps = len(self.flows)
        flow_run_id = str(runtime.flow_run.get_id()) if runtime.flow_run else ""  # pyright: ignore[reportAttributeAccessIssue]

        # Resolve storage URI and documents
        if isinstance(documents, str):
            storage_uri = documents
            docs = await self.flows[0].config.load_documents(storage_uri)
        else:
            docs = documents
            storage_uri = await self.provision_storage(project_name, docs, options, context)
            if storage_uri and docs:
                await self.flows[0].config.save_documents(
                    storage_uri, docs, validate_output_type=False
                )

        # Write identity labels for polling endpoint
        if flow_run_id:
            try:
                async with get_client() as client:
                    await client.update_flow_run_labels(
                        flow_run_id=UUID(flow_run_id),
                        labels={
                            "pipeline.project_name": project_name,
                            "pipeline.storage_uri": storage_uri,
                        },
                    )
            except Exception as e:
                logger.warning(f"Identity label update failed: {e}")

        # Download additional input documents
        if context.input_documents_urls:
            first_input_type = self.flows[0].config.INPUT_DOCUMENT_TYPES[0]
            downloaded = await download_documents(context.input_documents_urls, first_input_type)
            docs = DocumentList(list(docs) + list(downloaded))

        accumulated_docs = docs
        completion_sent = False

        try:
            for step, flow_fn in enumerate(self.flows, start=1):
                flow_name = getattr(flow_fn, "name", flow_fn.__name__)
                flow_run_id = str(runtime.flow_run.get_id()) if runtime.flow_run else ""  # pyright: ignore[reportAttributeAccessIssue]

                # Per-flow caching: check if outputs exist
                if storage_uri:
                    cached = await self._load_cached_output(flow_fn, storage_uri)
                    if cached is not None:
                        logger.info(f"[{step}/{total_steps}] Cache hit: {flow_name}")
                        accumulated_docs = DocumentList(list(accumulated_docs) + list(cached))
                        await self._send_progress(
                            context,
                            flow_run_id,
                            project_name,
                            storage_uri,
                            step,
                            total_steps,
                            flow_name,
                            "cached",
                            step_progress=1.0,
                            message=f"Loaded from cache: {flow_name}",
                        )
                        continue

                # Prefect state hooks
                active_flow = flow_fn
                if context.status_webhook_url:
                    hooks = self._build_status_hooks(
                        context, flow_run_id, project_name, step, total_steps, flow_name
                    )
                    active_flow = flow_fn.with_options(**hooks)

                # Progress: started
                await self._send_progress(
                    context,
                    flow_run_id,
                    project_name,
                    storage_uri,
                    step,
                    total_steps,
                    flow_name,
                    "started",
                    step_progress=0.0,
                    message=f"Starting: {flow_name}",
                )

                logger.info(f"[{step}/{total_steps}] Starting: {flow_name}")

                # Load documents for this flow
                if storage_uri:
                    current_docs = await flow_fn.config.load_documents(storage_uri)
                else:
                    current_docs = accumulated_docs

                try:
                    new_docs = await active_flow(project_name, current_docs, options)
                except Exception as e:
                    # Upload partial results on failure
                    if context.output_documents_urls:
                        await upload_documents(accumulated_docs, context.output_documents_urls)
                    await self._send_completion(
                        context, flow_run_id, project_name, storage_uri, result=None, error=str(e)
                    )
                    completion_sent = True
                    raise

                # Save to storage
                if storage_uri:
                    await flow_fn.config.save_documents(storage_uri, new_docs)

                accumulated_docs = DocumentList(list(accumulated_docs) + list(new_docs))

                # Per-flow upload
                if context.output_documents_urls:
                    await upload_documents(new_docs, context.output_documents_urls)

                # Progress: completed
                await self._send_progress(
                    context,
                    flow_run_id,
                    project_name,
                    storage_uri,
                    step,
                    total_steps,
                    flow_name,
                    "completed",
                    step_progress=1.0,
                    message=f"Completed: {flow_name}",
                )

                logger.info(f"[{step}/{total_steps}] Completed: {flow_name}")

            result = self.build_result(project_name, accumulated_docs, options)
            await self._send_completion(
                context, flow_run_id, project_name, storage_uri, result=result, error=None
            )
            return result

        except Exception as e:
            if not completion_sent:
                await self._send_completion(
                    context, flow_run_id, project_name, storage_uri, result=None, error=str(e)
                )
            raise

    @final
    def run_local(
        self,
        project_name: str,
        documents: str | DocumentList,
        options: TOptions,
        context: DeploymentContext | None = None,
        output_dir: Path | None = None,
    ) -> TResult:
        """Run locally with Prefect test harness."""
        if context is None:
            context = DeploymentContext()

        # If output_dir provided and documents is DocumentList, use output_dir as storage
        if output_dir and isinstance(documents, DocumentList):
            output_dir.mkdir(parents=True, exist_ok=True)
            documents = str(output_dir)

        with prefect_test_harness():
            with disable_run_logger():
                result = asyncio.run(self.run(project_name, documents, options, context))

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "result.json").write_text(result.model_dump_json(indent=2))

        return result

    @final
    def run_cli(
        self,
        initializer: Callable[[TOptions], tuple[str, DocumentList]] | None = None,
        trace_name: str | None = None,
    ) -> None:
        """Execute pipeline from CLI arguments with --start/--end step control."""
        if len(sys.argv) == 1:
            sys.argv.append("--help")

        setup_logging()
        try:
            Laminar.initialize()
            logger.info("LMNR tracing initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize LMNR: {e}")

        deployment = self

        class _CliOptions(
            deployment.options_type,
            cli_parse_args=True,
            cli_kebab_case=True,
            cli_exit_on_error=True,
            cli_prog_name=deployment.name,
            cli_use_class_docs_for_groups=True,
        ):
            working_directory: CliPositionalArg[Path]
            project_name: str | None = None
            start: int = 1
            end: int | None = None

            model_config = SettingsConfigDict(frozen=True, extra="ignore")

        opts = cast(TOptions, _CliOptions())  # type: ignore[reportCallIssue]

        wd: Path = getattr(opts, "working_directory")
        wd.mkdir(parents=True, exist_ok=True)

        project_name = getattr(opts, "project_name") or wd.name
        start_step = getattr(opts, "start", 1)
        end_step = getattr(opts, "end", None)

        # Initialize documents and save to working directory
        if initializer and start_step == 1:
            _, documents = initializer(opts)
            if documents and self.flows:
                first_config = getattr(self.flows[0], "config", None)
                if first_config:
                    asyncio.run(
                        first_config.save_documents(str(wd), documents, validate_output_type=False)
                    )

        context = DeploymentContext()

        with ExitStack() as stack:
            if trace_name:
                stack.enter_context(
                    Laminar.start_as_current_span(
                        name=f"{trace_name}-{project_name}",
                        input=[opts.model_dump_json()],
                    )
                )

            under_pytest = "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules
            if not settings.prefect_api_key and not under_pytest:
                stack.enter_context(prefect_test_harness())
                stack.enter_context(disable_run_logger())

            result = asyncio.run(
                self._run_with_steps(
                    project_name=project_name,
                    storage_uri=str(wd),
                    options=opts,
                    context=context,
                    start_step=start_step,
                    end_step=end_step,
                )
            )

        result_file = wd / "result.json"
        result_file.write_text(result.model_dump_json(indent=2))
        logger.info(f"Result saved to {result_file}")

    async def _run_with_steps(
        self,
        project_name: str,
        storage_uri: str,
        options: TOptions,
        context: DeploymentContext,
        start_step: int = 1,
        end_step: int | None = None,
    ) -> TResult:
        """Run pipeline with start/end step control for CLI resume support."""
        if end_step is None:
            end_step = len(self.flows)

        total_steps = len(self.flows)
        accumulated_docs = DocumentList([])

        for i in range(start_step - 1, end_step):
            step = i + 1
            flow_fn = self.flows[i]
            flow_name = getattr(flow_fn, "name", flow_fn.__name__)
            logger.info(f"--- [Step {step}/{total_steps}] {flow_name} ---")

            # Check cache
            cached = await self._load_cached_output(flow_fn, storage_uri)
            if cached is not None:
                logger.info(f"[{step}/{total_steps}] Cache hit: {flow_name}")
                accumulated_docs = DocumentList(list(accumulated_docs) + list(cached))
                continue

            current_docs = await flow_fn.config.load_documents(storage_uri)
            new_docs = await flow_fn(project_name, current_docs, options)
            await flow_fn.config.save_documents(storage_uri, new_docs)
            accumulated_docs = DocumentList(list(accumulated_docs) + list(new_docs))

        return self.build_result(project_name, accumulated_docs, options)

    @final
    def as_prefect_flow(self) -> Callable[..., Any]:
        """Generate Prefect flow for production deployment."""
        deployment = self

        @flow(  # pyright: ignore[reportUntypedFunctionDecorator]
            name=self.name,
            flow_run_name=f"{self.name}-{{project_name}}",
            persist_result=True,
            result_serializer="json",
        )
        async def _deployment_flow(
            project_name: str,
            documents: str | DocumentList,
            options: FlowOptions,
            context: DeploymentContext,
        ) -> DeploymentResult:
            return await deployment.run(project_name, documents, cast(Any, options), context)

        return _deployment_flow


__all__ = [
    "DeploymentContext",
    "DeploymentResult",
    "PipelineDeployment",
]
