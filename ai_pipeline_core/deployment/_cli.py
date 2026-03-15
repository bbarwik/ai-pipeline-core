"""CLI bootstrap for pipeline deployments.

Handles argument parsing and the Prefect test harness for local execution.
"""

import asyncio
import os
import sys
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness
from pydantic_settings import BaseSettings, CliPositionalArg, SettingsConfigDict

from ai_pipeline_core.database._factory import Database
from ai_pipeline_core.database.filesystem._backend import FilesystemDatabase
from ai_pipeline_core.database.snapshot._download import generate_run_artifacts
from ai_pipeline_core.documents import Document
from ai_pipeline_core.logger import get_pipeline_logger, setup_logging
from ai_pipeline_core.pipeline.options import FlowOptions
from ai_pipeline_core.settings import settings

from ._helpers import _create_publisher, _create_span_database_from_settings, validate_run_id

logger = get_pipeline_logger(__name__)


async def _generate_run_artifacts(database: Database, output_dir: Path) -> None:
    """Generate summary.md, costs.md, llm_calls.jsonl, errors.md, documents.md after a CLI run."""
    if not isinstance(database, FilesystemDatabase):
        return
    deployments = await database.list_deployments(limit=1, status=None)
    if not deployments:
        return
    await generate_run_artifacts(database, deployments[0].root_deployment_id, output_dir)
    logger.info("Run artifacts saved to %s", output_dir)


async def _shutdown_database(database: Database) -> None:
    """Flush and shut down the CLI database after artifact generation completes."""
    try:
        await database.flush()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Database flush failed: %s", exc)
    try:
        await database.shutdown()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Database shutdown failed: %s", exc)


def run_cli_for_deployment(
    deployment: Any,
    initializer: Callable[..., tuple[str, tuple[Document, ...]]] | None = None,
    cli_mixin: type[BaseSettings] | None = None,
) -> None:
    """Execute pipeline from CLI arguments with --start/--end step control."""
    if len(sys.argv) == 1:
        sys.argv.append("--help")

    setup_logging()

    options_base = deployment.options_type
    if cli_mixin is not None:
        options_base = type(deployment.options_type)(
            "_OptionsBase",
            (cli_mixin, deployment.options_type),
            {"__module__": __name__, "__annotations__": {}},
        )

    cli_options_cls = type(
        "_CliOptions",
        (options_base, BaseSettings),
        {
            "__module__": __name__,
            "__annotations__": {
                "working_directory": CliPositionalArg[Path],
                "run_id": str | None,
                "start": int,
                "end": int | None,
            },
            "run_id": None,
            "start": 1,
            "end": None,
            "model_config": SettingsConfigDict(
                frozen=True,
                extra="ignore",
                cli_parse_args=True,
                cli_kebab_case=True,
                cli_exit_on_error=True,
                cli_prog_name=deployment.name,
                cli_use_class_docs_for_groups=True,
            ),
        },
    )

    cli_opts = cli_options_cls()
    # Project into the deployment's stable options type so the codec never sees the dynamic CLI class
    run_opts_payload = {name: getattr(cli_opts, name) for name in deployment.options_type.model_fields}
    opts = cast(FlowOptions, deployment.options_type.model_validate(run_opts_payload))

    wd = cast(Path, cli_opts.working_directory)
    wd.mkdir(parents=True, exist_ok=True)

    start_step = getattr(cli_opts, "start", 1)
    end_step = getattr(cli_opts, "end", None)

    initial_documents: tuple[Document, ...] = ()
    if initializer:
        init_name, initial_documents = initializer(opts)
        run_id = cast(str | None, cli_opts.run_id) or init_name or wd.name
    else:
        run_id = cast(str, cli_opts.run_id or wd.name)

    validate_run_id(run_id)

    publisher = _create_publisher(settings, deployment.pubsub_service_type)
    database = _create_span_database_from_settings(settings, base_path=wd)

    try:
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
                        database=database,
                    )
                )
            finally:
                if hasattr(publisher, "close"):
                    asyncio.run(publisher.close())

        result_file = wd / "result.json"
        result_file.write_text(result.model_dump_json(indent=2))
        logger.info("Result saved to %s", result_file)

        asyncio.run(_generate_run_artifacts(database, wd))
    finally:
        asyncio.run(_shutdown_database(database))
