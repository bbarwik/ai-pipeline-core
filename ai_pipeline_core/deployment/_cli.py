"""CLI bootstrap for pipeline deployments.

Handles argument parsing and the Prefect test harness for local execution.
"""

import asyncio
import os
import sys
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import cast

from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness
from pydantic_settings import BaseSettings, CliPositionalArg, SettingsConfigDict

from ai_pipeline_core.documents import Document
from ai_pipeline_core.logging import get_pipeline_logger, setup_logging
from ai_pipeline_core.pipeline.options import FlowOptions
from ai_pipeline_core.settings import settings

from ._helpers import _create_publisher, validate_run_id
from .base import DeploymentResult, PipelineDeployment

logger = get_pipeline_logger(__name__)


def run_cli_for_deployment[TOptions: FlowOptions, TResult: DeploymentResult](
    deployment: PipelineDeployment[TOptions, TResult],
    initializer: Callable[[TOptions], tuple[str, tuple[Document, ...]]] | None = None,
    cli_mixin: type[BaseSettings] | None = None,
) -> None:
    """Execute pipeline from CLI arguments with --start/--end step control."""
    if len(sys.argv) == 1:
        sys.argv.append("--help")

    setup_logging()

    options_base = deployment.options_type
    if cli_mixin is not None:
        options_base = type(deployment.options_type)(  # pyright: ignore[reportGeneralTypeIssues]
            "_OptionsBase",
            (cli_mixin, deployment.options_type),
            {"__module__": __name__, "__annotations__": {}},
        )

    class _CliOptions(
        options_base,
        BaseSettings,
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

        model_config = SettingsConfigDict(frozen=True, extra="ignore")

    opts = cast(TOptions, _CliOptions())  # pyright: ignore[reportCallIssue]

    wd = cast(Path, opts.working_directory)  # pyright: ignore[reportAttributeAccessIssue]
    wd.mkdir(parents=True, exist_ok=True)

    start_step = getattr(opts, "start", 1)
    end_step = getattr(opts, "end", None)

    initial_documents: tuple[Document, ...] = ()
    if initializer:
        init_name, initial_documents = initializer(opts)
        run_id = cast(str | None, opts.run_id) or init_name or wd.name  # pyright: ignore[reportAttributeAccessIssue]
    else:
        run_id = cast(str, opts.run_id or wd.name)  # pyright: ignore[reportAttributeAccessIssue]

    validate_run_id(run_id)

    publisher = _create_publisher(settings, deployment.pubsub_service_type)

    with ExitStack() as stack:
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
                    publisher=publisher,
                    start_step=start_step,
                    end_step=end_step,
                )
            )
        finally:
            if hasattr(publisher, "close"):
                asyncio.run(publisher.close())

    result_file = wd / "result.json"
    result_file.write_text(result.model_dump_json(indent=2))
    logger.info("Result saved to %s", result_file)
