from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Type, TypeVar, cast

from lmnr import Laminar
from pydantic_settings import CliPositionalArg, SettingsConfigDict

from ai_pipeline_core.documents import DocumentList
from ai_pipeline_core.flow.options import FlowOptions
from ai_pipeline_core.logging import get_pipeline_logger, setup_logging

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


def run_cli(
    *,
    flows: FlowSequence,
    flow_configs: ConfigSequence,
    options_cls: Type[TOptions],
    initializer: InitializerFunc = None,
) -> None:
    """
    Parse CLI+env into options, then run the pipeline.

    - working_directory: required positional arg
    - --project-name: optional, defaults to directory name
    - --start/--end: optional, 1-based step bounds
    - all other flags come from options_cls (fields & Field descriptions)
    """
    _initialize_environment()

    class _RunnerOptions(  # type: ignore[reportRedeclaration]
        options_cls,
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_exit_on_error=False,
    ):
        working_directory: CliPositionalArg[Path]
        project_name: str | None = None
        start: int = 1
        end: int | None = None

        model_config = SettingsConfigDict(frozen=True, extra="ignore")

    opts = cast(FlowOptions, _RunnerOptions())  # type: ignore[reportCallIssue]

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
