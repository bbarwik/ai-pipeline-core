"""Helper functions for pipeline deployments."""

import asyncio
import hashlib
import json
import re
from typing import Any
from uuid import UUID

from lmnr import Laminar

from ai_pipeline_core.document_store._summary_worker import SummaryGenerator
from ai_pipeline_core.documents import Document, RunScope
from ai_pipeline_core.exceptions import LLMError, PipelineCoreError
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._initialization import get_pipeline_processors
from ai_pipeline_core.pipeline._flow import PipelineFlow
from ai_pipeline_core.pipeline.options import FlowOptions
from ai_pipeline_core.settings import Settings, settings

from ._types import ErrorCode, ResultPublisher, TaskResultStore, _NoopPublisher

logger = get_pipeline_logger(__name__)

__all__ = [
    "MAX_RUN_ID_LENGTH",
    "_CLI_FIELDS",
    "_HANDLE_CANCEL_GRACE_SECONDS",
    "_HEARTBEAT_INTERVAL_SECONDS",
    "_MILLISECONDS_PER_SECOND",
    "_build_summary_generator",
    "_classify_error",
    "_clear_run_context_on_processors",
    "_compute_run_scope",
    "_create_publisher",
    "_create_task_result_store",
    "_heartbeat_loop",
    "_set_flow_replay_payload",
    "_set_run_context_on_processors",
    "class_name_to_deployment_name",
    "extract_generic_params",
    "init_observability_best_effort",
    "validate_run_id",
]

_RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
MAX_RUN_ID_LENGTH = 100

# Fields added by run_cli()'s _CliOptions that should not affect fingerprints (run scope or remote run_id)
_CLI_FIELDS: frozenset[str] = frozenset({"working_directory", "run_id", "start", "end", "no_trace"})

_HEARTBEAT_INTERVAL_SECONDS = 30
_MILLISECONDS_PER_SECOND = 1000
_HANDLE_CANCEL_GRACE_SECONDS = 5


def validate_run_id(run_id: str) -> None:
    """Validate run_id: alphanumeric + underscore + hyphen, 1-100 chars.

    Must be called at deployment entry points (PipelineDeployment.run, RemoteDeployment._execute, CLI).
    """
    if not run_id:
        raise ValueError("run_id must not be empty")
    if len(run_id) > MAX_RUN_ID_LENGTH:
        raise ValueError(
            f"run_id '{run_id[:20]}...' is {len(run_id)} chars, max is {MAX_RUN_ID_LENGTH}. Shorten the base run_id before passing to the deployment."
        )
    if not _RUN_ID_PATTERN.match(run_id):
        raise ValueError(
            f"run_id '{run_id}' contains invalid characters. "
            f"Only alphanumeric characters, underscores, and hyphens are allowed (pattern: {_RUN_ID_PATTERN.pattern})."
        )


def init_observability_best_effort() -> None:
    """Best-effort observability initialization with Laminar fallback."""
    from ai_pipeline_core.observability import tracing
    from ai_pipeline_core.observability._initialization import initialize_observability

    try:
        initialize_observability()
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError, ImportError) as e:
        logger.warning("Failed to initialize observability: %s", e)
        try:
            tracing._initialise_laminar()
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError, ImportError) as e2:
            logger.warning("Laminar fallback initialization failed: %s", e2)


def class_name_to_deployment_name(class_name: str) -> str:
    """Convert PascalCase to kebab-case: ResearchPipeline -> research-pipeline."""
    name = re.sub(r"(?<!^)(?=[A-Z])", "-", class_name)
    return name.lower()


def extract_generic_params(cls: type, base_class: type) -> tuple[Any, ...]:
    """Extract Generic type arguments from a class's base.

    Works with any number of Generic parameters (2 for PipelineDeployment, 3 for RemoteDeployment).
    Returns () if the base class is not found in __orig_bases__.
    """
    for base in getattr(cls, "__orig_bases__", []):
        origin = getattr(base, "__origin__", None)
        if origin is base_class:
            args = getattr(base, "__args__", ())
            if args:
                return args

    return ()


def _set_flow_replay_payload(flow_instance: PipelineFlow, run_id: str, documents: list[Document], options: FlowOptions) -> None:
    """Attach flow replay payload to the current span for trace extraction."""
    try:
        payload = {
            "version": 1,
            "payload_type": "pipeline_flow",
            "function_path": f"{type(flow_instance).__module__}:{type(flow_instance).__qualname__}",
            "run_id": run_id,
            "documents": [{"$doc_ref": d.sha256, "class_name": type(d).__name__, "name": d.name} for d in documents],
            "flow_options": options.model_dump(exclude_defaults=True),
            "flow_params": flow_instance.get_params(),
        }
        Laminar.set_span_attributes({"replay.payload": json.dumps(payload)})
    except Exception:
        logger.debug("Failed to attach flow replay payload for '%s'", type(flow_instance).__name__, exc_info=True)


def _classify_error(exc: BaseException) -> ErrorCode:
    """Map exception to ErrorCode enum value."""
    if isinstance(exc, LLMError):
        return ErrorCode.PROVIDER_ERROR
    if isinstance(exc, asyncio.CancelledError):
        return ErrorCode.CANCELLED
    if isinstance(exc, TimeoutError):
        return ErrorCode.DURATION_EXCEEDED
    if isinstance(exc, (ValueError, TypeError)):
        return ErrorCode.INVALID_INPUT
    if isinstance(exc, PipelineCoreError):
        return ErrorCode.PIPELINE_ERROR
    return ErrorCode.UNKNOWN


def _create_publisher(settings_obj: Settings, service_type: str) -> ResultPublisher:
    """Create publisher based on environment and deployment configuration.

    Returns PubSubPublisher when Pub/Sub is configured and service_type is set,
    _NoopPublisher otherwise.
    """
    if not service_type:
        return _NoopPublisher()
    if settings_obj.pubsub_project_id and settings_obj.pubsub_topic_id:
        from ._pubsub import PubSubPublisher

        return PubSubPublisher(
            project_id=settings_obj.pubsub_project_id,
            topic_id=settings_obj.pubsub_topic_id,
            service_type=service_type,
        )
    return _NoopPublisher()


def _create_task_result_store(settings_obj: Settings) -> TaskResultStore | None:
    """Create ClickHouseTaskResultStore when ClickHouse is configured.

    Independent of Pub/Sub — results are persisted for remote caller fallback
    whenever ClickHouse is available.
    """
    if not settings_obj.clickhouse_host:
        return None
    from ._task_results import ClickHouseTaskResultStore

    return ClickHouseTaskResultStore(
        host=settings_obj.clickhouse_host,
        port=settings_obj.clickhouse_port,
        database=settings_obj.clickhouse_database,
        username=settings_obj.clickhouse_user,
        password=settings_obj.clickhouse_password,
        secure=settings_obj.clickhouse_secure,
        connect_timeout=settings_obj.clickhouse_connect_timeout,
        send_receive_timeout=settings_obj.clickhouse_send_receive_timeout,
    )


def _build_summary_generator() -> SummaryGenerator | None:
    """Build a summary generator callable from settings, or None if disabled/unavailable."""
    if not settings.doc_summary_enabled:
        return None

    from ai_pipeline_core.document_store._summary_llm import generate_document_summary

    model = settings.doc_summary_model

    async def _generator(name: str, excerpt: str) -> str:
        return await generate_document_summary(name, excerpt, model=model)

    return _generator


def _set_run_context_on_processors(execution_id: UUID, run_id: str, flow_name: str, run_scope: str) -> None:
    """Propagate pipeline run context to all registered PipelineSpanProcessors.

    Each processor injects these values into SpanData for child spans that
    don't have them set as span/resource attributes.
    """
    for processor in get_pipeline_processors():
        processor.set_run_context(
            execution_id=execution_id,
            run_id=run_id,
            flow_name=flow_name,
            run_scope=run_scope,
        )


def _clear_run_context_on_processors() -> None:
    """Clear pipeline run context from all registered PipelineSpanProcessors."""
    for processor in get_pipeline_processors():
        processor.clear_run_context()


def _compute_run_scope(run_id: str, documents: list[Document], options: FlowOptions) -> RunScope:
    """Compute a run scope that fingerprints inputs and options.

    Different inputs or options produce a different scope, preventing
    stale cache hits when re-running with the same run_id.
    """
    exclude = set(_CLI_FIELDS & set(type(options).model_fields))
    options_json = options.model_dump_json(exclude=exclude, exclude_none=True)

    if not documents:
        fingerprint = hashlib.sha256(options_json.encode()).hexdigest()[:16]
        return RunScope(f"{run_id}:{fingerprint}")

    sha256s = sorted(doc.sha256 for doc in documents)
    fingerprint = hashlib.sha256(f"{':'.join(sha256s)}|{options_json}".encode()).hexdigest()[:16]
    return RunScope(f"{run_id}:{fingerprint}")


async def _heartbeat_loop(publisher: ResultPublisher, run_id: str) -> None:
    """Publish heartbeat signals at regular intervals until cancelled."""
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
        try:
            await publisher.publish_heartbeat(run_id)
        except Exception as e:
            logger.warning("Heartbeat publish failed: %s", e)
