"""LLM-powered auto-summary generation for trace debugging.

Separated from _summary.py to avoid circular imports: this module depends on
ai_pipeline_core.llm, which cannot be imported during the initial package load
chain that includes _debug/__init__.py.
"""

from pydantic import BaseModel, ConfigDict

from ai_pipeline_core.llm import generate_structured
from ai_pipeline_core.llm.ai_messages import AIMessages
from ai_pipeline_core.llm.model_options import ModelOptions

from ._types import TraceState


class AutoTraceSummary(BaseModel):
    """LLM-generated trace analysis."""

    model_config = ConfigDict(frozen=True)

    overview: str
    outcome: str
    error_analysis: str
    bottlenecks: tuple[str, ...] = ()
    cost_assessment: str
    recommendations: tuple[str, ...] = ()


async def generate_auto_summary(
    trace: TraceState,  # noqa: ARG001
    static_summary: str,
    model: str,
) -> str | None:
    """Generate LLM-powered auto-summary of the trace.

    Args:
        trace: Completed trace state with all span data.
        static_summary: Pre-generated static summary text used as LLM input context.
        model: LLM model name for summary generation.

    Returns:
        Formatted markdown auto-summary string, or None if generation fails.
    """
    messages = AIMessages()
    messages.append(static_summary)

    options = ModelOptions(
        system_prompt=(
            "You are analyzing an AI pipeline execution trace. "
            "Provide concise, actionable analysis based on the execution data. "
            "Focus on cost efficiency, performance bottlenecks, and errors."
        ),
    )

    result = await generate_structured(
        model=model,
        response_format=AutoTraceSummary,
        messages=messages,
        options=options,
        purpose="trace_auto_summary",
    )

    if not result or not result.parsed:
        return None

    summary = result.parsed
    lines = [
        "# Auto-Summary (LLM-Generated)",
        "",
        f"**Overview:** {summary.overview}",
        "",
        f"**Outcome:** {summary.outcome}",
        "",
    ]

    if summary.error_analysis:
        lines.append(f"**Error Analysis:** {summary.error_analysis}")
        lines.append("")

    if summary.bottlenecks:
        lines.append("**Bottlenecks:**")
        lines.extend(f"- {b}" for b in summary.bottlenecks)
        lines.append("")

    lines.append(f"**Cost Assessment:** {summary.cost_assessment}")
    lines.append("")

    if summary.recommendations:
        lines.append("**Recommendations:**")
        lines.extend(f"- {r}" for r in summary.recommendations)
        lines.append("")

    return "\n".join(lines)
