"""Helper functions for pipeline deployments."""

import re
from typing import Any

from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

_RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
MAX_RUN_ID_LENGTH = 100

# Fields added by run_cli()'s _CliOptions that should not affect fingerprints (run scope or remote run_id)
_CLI_FIELDS: frozenset[str] = frozenset({"working_directory", "run_id", "start", "end", "no_trace"})


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
