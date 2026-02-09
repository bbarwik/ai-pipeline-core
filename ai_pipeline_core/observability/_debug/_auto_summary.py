"""LLM-powered auto-summary generation for trace debugging.

Uses _llm_core directly to avoid Document dependency cycle.
"""

from pydantic import BaseModel, ConfigDict

from ai_pipeline_core._llm_core import CoreMessage, ModelOptions, Role, generate_structured


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
    static_summary: str,
    model: str,
) -> str | None:
    """Generate LLM-powered auto-summary of the trace.

    Args:
        static_summary: Pre-generated static summary text used as LLM input context.
        model: LLM model name for summary generation.

    Returns:
        Formatted markdown auto-summary string, or None if generation fails.
    """
    system_prompt = (
        "You are analyzing an AI pipeline execution trace. "
        "Provide concise, actionable analysis based on the execution data. "
        "Focus on cost efficiency, performance bottlenecks, and errors."
    )

    messages = [
        CoreMessage(role=Role.SYSTEM, content=system_prompt),
        CoreMessage(role=Role.USER, content=static_summary),
    ]

    options = ModelOptions()

    response = await generate_structured(
        messages,
        response_format=AutoTraceSummary,
        model=model,
        model_options=options,
        purpose="trace_auto_summary",
    )

    if not response.parsed:
        return None

    summary = response.parsed
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
