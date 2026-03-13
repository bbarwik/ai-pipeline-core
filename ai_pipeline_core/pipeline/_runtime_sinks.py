"""Runtime span sink construction helpers."""

from ai_pipeline_core.database._protocol import DatabaseWriter
from ai_pipeline_core.observability._laminar_sink import LaminarSpanSink
from ai_pipeline_core.pipeline._span_sink import DatabaseSpanSink, SpanSink
from ai_pipeline_core.settings import Settings

__all__ = ["build_runtime_sinks"]


def build_runtime_sinks(
    *,
    database: DatabaseWriter | None,
    settings_obj: Settings,
) -> tuple[SpanSink, ...]:
    """Build the span sinks for one runtime boundary."""
    sinks: list[SpanSink] = []
    if database is not None:
        sinks.append(DatabaseSpanSink(database))
    if settings_obj.lmnr_project_api_key:
        sinks.append(LaminarSpanSink(settings_obj))
    return tuple(sinks)
