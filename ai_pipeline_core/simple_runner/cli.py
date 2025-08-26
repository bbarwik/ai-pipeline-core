from __future__ import annotations

import asyncio
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, Type, TypeVar, cast

from lmnr import Laminar
from pydantic import ValidationError
from pydantic_settings import CliPositionalArg, SettingsConfigDict

from ai_pipeline_core.documents import DocumentList
from ai_pipeline_core.flow.options import FlowOptions
from ai_pipeline_core.logging import get_pipeline_logger, setup_logging
from ai_pipeline_core.prefect import disable_run_logger, prefect_test_harness
from ai_pipeline_core.settings import settings

from .simple_runner import ConfigSequence, FlowSequence, run_pipelines, save_documents_to_directory

logger = get_pipeline_logger(__name__)

TOptions = TypeVar("TOptions", bound=FlowOptions)
InitializerFunc = Callable[[FlowOptions], tuple[str, DocumentList]] | None


def _initialize_environment() -> None:
    setup_logging()
    try:
        Laminar.initialize()
        logger.info("LMNR tracing initialized.")
    except Exception as e:
        logger.warning(f"Failed to initialize LMNR tracing: {e}")


def _running_under_pytest() -> bool:  # NEW
    """Return True when invoked by pytest (so fixtures will supply test contexts)."""
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules


def run_cli(
    *,
    flows: FlowSequence,
    flow_configs: ConfigSequence,
    options_cls: Type[TOptions],
    initializer: InitializerFunc = None,
    trace_name: str | None = None,
) -> None:
    """
    Parse CLI+env into options, then run the pipeline.

    - working_directory: required positional arg
    - --project-name: optional, defaults to directory name
    - --start/--end: optional, 1-based step bounds
    - all other flags come from options_cls (fields & Field descriptions)
    """
    # Check if no arguments provided before initialization
    if len(sys.argv) == 1:
        # Add --help to show usage
        sys.argv.append("--help")

    _initialize_environment()

    class _RunnerOptions(  # type: ignore[reportRedeclaration]
        options_cls,
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_exit_on_error=True,  # Let it exit normally on error
        cli_prog_name="ai-pipeline",
        cli_use_class_docs_for_groups=True,
    ):
        working_directory: CliPositionalArg[Path]
        project_name: str | None = None
        start: int = 1
        end: int | None = None

        model_config = SettingsConfigDict(frozen=True, extra="ignore")

    try:
        opts = cast(FlowOptions, _RunnerOptions())  # type: ignore[reportCallIssue]
    except ValidationError as e:
        print("\nError: Invalid command line arguments\n", file=sys.stderr)
        for error in e.errors():
            field = " -> ".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            value = error.get("input", "")

            # Format the field name nicely (convert from snake_case to kebab-case for CLI)
            cli_field = field.replace("_", "-")

            print(f"  --{cli_field}: {msg}", file=sys.stderr)
            if value:
                print(f"    Provided value: '{value}'", file=sys.stderr)

            # Add helpful hints for common errors
            if error["type"] == "float_parsing":
                print("    Hint: Please provide a valid number (e.g., 0.7)", file=sys.stderr)
            elif error["type"] == "int_parsing":
                print("    Hint: Please provide a valid integer (e.g., 10)", file=sys.stderr)
            elif error["type"] == "literal_error":
                ctx = error.get("ctx", {})
                expected = ctx.get("expected", "valid options")
                print(f"    Hint: Valid options are: {expected}", file=sys.stderr)
            elif error["type"] in [
                "less_than_equal",
                "greater_than_equal",
                "less_than",
                "greater_than",
            ]:
                ctx = error.get("ctx", {})
                if "le" in ctx:
                    print(f"    Hint: Value must be ≤ {ctx['le']}", file=sys.stderr)
                elif "ge" in ctx:
                    print(f"    Hint: Value must be ≥ {ctx['ge']}", file=sys.stderr)
                elif "lt" in ctx:
                    print(f"    Hint: Value must be < {ctx['lt']}", file=sys.stderr)
                elif "gt" in ctx:
                    print(f"    Hint: Value must be > {ctx['gt']}", file=sys.stderr)

        print("\nRun with --help to see all available options\n", file=sys.stderr)
        sys.exit(1)

    wd: Path = cast(Path, getattr(opts, "working_directory"))
    wd.mkdir(parents=True, exist_ok=True)

    # Get project name from options or use directory basename
    project_name = getattr(opts, "project_name", None)
    if not project_name:  # None or empty string
        project_name = wd.name

    # Ensure project_name is not empty
    if not project_name:
        raise ValueError("Project name cannot be empty")

    # Use initializer if provided, otherwise use defaults
    initial_documents = DocumentList([])
    if initializer:
        init_result = initializer(opts)
        # Always expect tuple format from initializer
        _, initial_documents = init_result  # Ignore project name from initializer

        if getattr(opts, "start", 1) == 1 and initial_documents:
            save_documents_to_directory(wd, initial_documents)

    # Setup context stack with optional test harness and tracing

    with ExitStack() as stack:
        if trace_name:
            stack.enter_context(
                Laminar.start_as_current_span(
                    name=f"{trace_name}-{project_name}", input=[opts.model_dump_json()]
                )
            )

        if not settings.prefect_api_key and not _running_under_pytest():
            stack.enter_context(prefect_test_harness())
            stack.enter_context(disable_run_logger())

        asyncio.run(
            run_pipelines(
                project_name=project_name,
                output_dir=wd,
                flows=flows,
                flow_configs=flow_configs,
                flow_options=opts,
                start_step=getattr(opts, "start", 1),
                end_step=getattr(opts, "end", None),
            )
        )
