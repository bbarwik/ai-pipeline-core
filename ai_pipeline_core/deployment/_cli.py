"""CLI bootstrap for pipeline deployments.

Handles argument parsing, observability initialization, document store setup,
debug tracing, and Prefect test harness for local execution.
"""

import asyncio
import os
import sys
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

from lmnr import Laminar
from opentelemetry import trace as otel_trace
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness
from pydantic_settings import BaseSettings, CliPositionalArg, SettingsConfigDict

from ai_pipeline_core.document_store._dual_store import DualDocumentStore
from ai_pipeline_core.document_store._factory import create_document_store
from ai_pipeline_core.document_store._local import LocalDocumentStore
from ai_pipeline_core.document_store._protocol import get_document_store, set_document_store
from ai_pipeline_core.documents import Document
from ai_pipeline_core.logging import get_pipeline_logger, setup_logging
from ai_pipeline_core.observability._debug import LocalDebugSpanProcessor, LocalTraceWriter, TraceDebugConfig
from ai_pipeline_core.observability._initialization import get_tracking_service
from ai_pipeline_core.pipeline.options import FlowOptions
from ai_pipeline_core.settings import settings

from ._helpers import init_observability_best_effort, validate_run_id
from .base import DeploymentContext, DeploymentResult, PipelineDeployment, _build_summary_generator, _create_publisher, _create_task_result_store

logger = get_pipeline_logger(__name__)


def _init_debug_tracing(wd: Path) -> LocalDebugSpanProcessor | None:
    """Set up local debug tracing at <working_dir>/.trace. Returns processor or None on failure."""
    try:
        trace_path = wd / ".trace"
        trace_path.mkdir(parents=True, exist_ok=True)
        debug_config = TraceDebugConfig(path=trace_path)
        debug_writer = LocalTraceWriter(debug_config)
        processor = LocalDebugSpanProcessor(debug_writer, verbose=debug_config.verbose)
        provider: Any = otel_trace.get_tracer_provider()
        if hasattr(provider, "add_span_processor"):
            provider.add_span_processor(processor)
            logger.info("Local debug tracing enabled at %s", trace_path)
        return processor
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as e:
        logger.warning("Failed to set up local debug tracing: %s", e)
        return None


def _create_cli_document_store(wd: Path) -> Any:
    """Create document store for CLI: DualDocumentStore (ClickHouse + local) when configured, local-only otherwise."""
    summary_generator = _build_summary_generator()
    if settings.clickhouse_host:
        primary = create_document_store(settings)
        secondary = LocalDocumentStore(base_path=wd)
        return DualDocumentStore(primary=primary, secondary=secondary, summary_generator=summary_generator)
    return LocalDocumentStore(base_path=wd, summary_generator=summary_generator)


def run_cli_for_deployment[TOptions: FlowOptions, TResult: DeploymentResult](
    deployment: PipelineDeployment[TOptions, TResult],
    initializer: Callable[[TOptions], tuple[str, list[Document]]] | None = None,
    trace_name: str | None = None,
    cli_mixin: type[BaseSettings] | None = None,
) -> None:
    """Execute pipeline from CLI arguments with --start/--end step control."""
    if len(sys.argv) == 1:
        sys.argv.append("--help")

    setup_logging()
    init_observability_best_effort()

    options_base = deployment.options_type
    if cli_mixin is not None:
        options_base = type(deployment.options_type)(  # pyright: ignore[reportGeneralTypeIssues]
            "_OptionsBase",
            (cli_mixin, deployment.options_type),
            {"__module__": __name__, "__annotations__": {}},
        )

    class _CliOptions(
        options_base,
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_exit_on_error=True,
        cli_prog_name=deployment.name,
        cli_use_class_docs_for_groups=True,
    ):
        working_directory: CliPositionalArg[Path]
        run_id: str | None = None
        start: int = 1
        end: int | None = None
        no_trace: bool = False

        model_config = SettingsConfigDict(frozen=True, extra="ignore")

    opts = cast(TOptions, _CliOptions())  # pyright: ignore[reportCallIssue]

    wd = cast(Path, opts.working_directory)  # pyright: ignore[reportAttributeAccessIssue]
    wd.mkdir(parents=True, exist_ok=True)

    start_step = getattr(opts, "start", 1)
    end_step = getattr(opts, "end", None)

    debug_processor = _init_debug_tracing(wd) if not getattr(opts, "no_trace", False) else None

    store = _create_cli_document_store(wd)
    set_document_store(store)

    initial_documents: list[Document] = []
    if initializer:
        init_name, initial_documents = initializer(opts)
        run_id = cast(str | None, opts.run_id) or init_name or wd.name  # pyright: ignore[reportAttributeAccessIssue]
    else:
        run_id = cast(str, opts.run_id or wd.name)  # pyright: ignore[reportAttributeAccessIssue]

    validate_run_id(run_id)

    context = DeploymentContext()
    publisher = _create_publisher(settings, deployment.pubsub_service_type)
    task_result_store = _create_task_result_store(settings)

    with ExitStack() as stack:
        if trace_name:
            stack.enter_context(
                Laminar.start_as_current_span(
                    name=f"{trace_name}-{run_id}",
                    input=[opts.model_dump_json()],
                )
            )

        under_pytest = "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules
        if not settings.prefect_api_key and not under_pytest:
            stack.enter_context(prefect_test_harness())
            stack.enter_context(disable_run_logger())

        try:
            result = asyncio.run(
                deployment.run(
                    run_id=run_id,
                    documents=initial_documents,
                    options=opts,
                    context=context,
                    publisher=publisher,
                    start_step=start_step,
                    end_step=end_step,
                    task_result_store=task_result_store,
                )
            )
            if trace_name:
                Laminar.set_span_output(result.model_dump())
        finally:
            # Close publisher if it supports closing
            if hasattr(publisher, "close"):
                asyncio.run(publisher.close())
            if task_result_store:
                task_result_store.shutdown()
            # Shut down document store and tracking (but NOT debug processor —
            # it must shut down after ExitStack exits so the Laminar root span
            # gets its on_end before the writer finalizes)
            active_store = get_document_store()
            if active_store:
                active_store.shutdown()
                set_document_store(None)
            tracking_svc = get_tracking_service()
            if tracking_svc:
                tracking_svc.shutdown()

    # Shut down debug processor after ExitStack exits — the Laminar root span
    # context manager has now closed, so on_end was called for all spans.
    if debug_processor is not None:
        debug_processor.shutdown()

    result_file = wd / "result.json"
    result_file.write_text(result.model_dump_json(indent=2))
    logger.info("Result saved to %s", result_file)
