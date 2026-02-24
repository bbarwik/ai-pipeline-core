"""Helper functions for pipeline deployments."""

import re
from typing import Any

from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)


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
