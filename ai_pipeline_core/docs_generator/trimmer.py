"""Size management for generated guides.

Measures rendered guide size and warns when exceeding 50KB limit.
No hard failure -- oversized guides are still written.
"""

from ai_pipeline_core.docs_generator.guide_builder import GuideData
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

MAX_GUIDE_SIZE = 51_200  # 50KB in bytes


def manage_guide_size(
    data: GuideData,
    rendered_content: str,
    max_size: int = MAX_GUIDE_SIZE,
) -> str:
    """Warn if rendered guide exceeds size limit. Returns content unchanged."""
    size = _measure(rendered_content)
    if size <= max_size:
        return rendered_content
    logger.warning(
        "%s guide is %s bytes (%dKB). Consider: move private helpers to _ prefixed functions, split large classes into separate modules",
        data.module_name,
        f"{size:,}",
        size // 1024,
    )
    return rendered_content


def _measure(content: str) -> int:
    """Measure guide size in UTF-8 bytes."""
    return len(content.encode("utf-8"))
