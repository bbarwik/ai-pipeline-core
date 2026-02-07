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
from typing import Any, TypeVar, cast

from lmnr import Laminar
from opentelemetry import trace as otel_trace
from pydantic_settings import BaseSettings, CliPositionalArg, SettingsConfigDict

from ai_pipeline_core.document_store import create_document_store, get_document_store, set_document_store
from ai_pipeline_core.document_store._dual_store import DualDocumentStore
from ai_pipeline_core.document_store._summary import SummaryGenerator
from ai_pipeline_core.document_store.local import LocalDocumentStore
from ai_pipeline_core.documents import Document
from ai_pipeline_core.logging import get_pipeline_logger, setup_logging
from ai_pipeline_core.observability import tracing
from ai_pipeline_core.observability._debug import LocalDebugSpanProcessor, LocalTraceWriter, TraceDebugConfig
from ai_pipeline_core.observability._initialization import get_tracking_service, initialize_observability
from ai_pipeline_core.pipeline.options import FlowOptions
from ai_pipeline_core.settings import settings
from ai_pipeline_core.testing import disable_run_logger, prefect_test_harness

from .base import DeploymentContext, DeploymentResult, PipelineDeployment, _build_summary_generator

logger = get_pipeline_logger(__name__)

TOptions = TypeVar("TOptions", bound=FlowOptions)
TResult = TypeVar("TResult", bound=DeploymentResult)


def _init_observability() -> None:
    """Best-effort observability initialization with Laminar fallback."""
    try:
        initialize_observability()
        logger.info("Observability initialized.")
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError, ImportError) as e:
        logger.warning("Failed to initialize observability: %s", e)
        try:
            tracing._initialise_laminar()
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError, ImportError) as e2:
            logger.warning("Laminar fallback initialization failed: %s", e2)


def _init_debug_tracing(wd: Path) -> LocalDebugSpanProcessor | None:
    """Set up local debug tracing at <working_dir>/.trace. Returns processor or None on failure."""
    try:
        trace_path = wd / ".trace"
        trace_path.mkdir(parents=True, exist_ok=True)
        debug_config = TraceDebugConfig(path=trace_path, max_traces=20)
        debug_writer = LocalTraceWriter(debug_config)
        processor = LocalDebugSpanProcessor(debug_writer)
        provider: Any = otel_trace.get_tracer_provider()
        if hasattr(provider, "add_span_processor"):
            provider.add_span_processor(processor)
            logger.info("Local debug tracing enabled at %s", trace_path)
        return processor
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as e:
        logger.warning("Failed to set up local debug tracing: %s", e)
        return None


def _create_document_store(wd: Path, summary_generator: SummaryGenerator | None) -> Any:
    """Create document store — DualDocumentStore (ClickHouse + local) when configured, local-only otherwise."""
    if settings.clickhouse_host:
        primary = create_document_store(settings)
        secondary = LocalDocumentStore(base_path=wd)
        return DualDocumentStore(primary=primary, secondary=secondary, summary_generator=summary_generator)
    return LocalDocumentStore(base_path=wd, summary_generator=summary_generator)


def _shutdown_workers(debug_processor: LocalDebugSpanProcessor | None) -> None:
    """Shut down background workers (debug tracing, document summaries, tracking)."""
    if debug_processor is not None:
        debug_processor.shutdown()
    store = get_document_store()
    if store:
        store.shutdown()
        set_document_store(None)
    tracking_svc = get_tracking_service()
    if tracking_svc:
        tracking_svc.shutdown()


def run_cli_for_deployment(
    deployment: PipelineDeployment[TOptions, TResult],
    initializer: Callable[[TOptions], tuple[str, list[Document]]] | None = None,
    trace_name: str | None = None,
    cli_mixin: type[BaseSettings] | None = None,
) -> None:
    """Execute pipeline from CLI arguments with --start/--end step control."""
    if len(sys.argv) == 1:
        sys.argv.append("--help")

    setup_logging()
    _init_observability()

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
        project_name: str | None = None
        start: int = 1
        end: int | None = None
        no_trace: bool = False

        model_config = SettingsConfigDict(frozen=True, extra="ignore")

    opts = cast(TOptions, _CliOptions())  # pyright: ignore[reportCallIssue]

    wd = cast(Path, opts.working_directory)  # pyright: ignore[reportAttributeAccessIssue]
    wd.mkdir(parents=True, exist_ok=True)

    project_name = cast(str, opts.project_name or wd.name)  # pyright: ignore[reportAttributeAccessIssue]
    start_step = getattr(opts, "start", 1)
    end_step = getattr(opts, "end", None)

    debug_processor = _init_debug_tracing(wd) if not getattr(opts, "no_trace", False) else None

    summary_generator = _build_summary_generator()
    store = _create_document_store(wd, summary_generator)
    set_document_store(store)

    initial_documents: list[Document] = []
    if initializer:
        _, initial_documents = initializer(opts)

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

        try:
            result = asyncio.run(
                deployment.run(
                    project_name=project_name,
                    documents=initial_documents,
                    options=opts,
                    context=context,
                    start_step=start_step,
                    end_step=end_step,
                )
            )
            if trace_name:
                Laminar.set_span_output(result.model_dump())
        finally:
            _shutdown_workers(debug_processor)

    result_file = wd / "result.json"
    result_file.write_text(result.model_dump_json(indent=2))
    logger.info("Result saved to %s", result_file)
