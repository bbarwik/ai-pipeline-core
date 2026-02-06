"""LLM-powered summary generation for tracked spans and documents.

Uses _llm_core directly to avoid Document dependency cycle.
"""

from pydantic import BaseModel, Field

from ai_pipeline_core._llm_core import CoreMessage, ModelOptions, Role, generate_structured
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._tracking._internal import internal_tracking_context

logger = get_pipeline_logger(__name__)

_SPAN_SUMMARY_SYSTEM_PROMPT = (
    "You summarize AI pipeline task results for non-technical users "
    "monitoring a research pipeline.\n"
    "Rules:\n"
    "- Describe the action and outcome, not the content\n"
    "- No internal names, function names, or technical details\n"
    "- No sensitive data (URLs, personal names, company details) from the output\n"
    "- Use present perfect tense"
)

_DOC_SUMMARY_SYSTEM_PROMPT = (
    "You generate metadata for documents in a research pipeline dashboard.\n"
    "Rules:\n"
    "- No sensitive data (URLs, personal names, company details)\n"
    "- Describe purpose and content type, not the content itself\n"
    "- For website documents: short_title must be 'domain.com: Page Title' (shorten title if needed to fit 50 chars)"
)


class SpanSummary(BaseModel):
    """Structured output for span/task summaries."""

    summary: str = Field(description="1-2 sentences (max 50 words) describing what the task accomplished in present perfect tense")


class DocumentSummary(BaseModel):
    """Structured output for document summaries."""

    short_title: str = Field(description="Document title proposition based on content, max 50 characters")
    summary: str = Field(description="1-2 sentences (max 50 words) describing the document's purpose and content type")


async def generate_span_summary(label: str, output_hint: str, model: str = "gemini-3-flash") -> str:
    """Generate a human-readable summary for a span/task output.

    Returns plain summary string (stored in tracked_spans.user_summary).
    """
    try:
        messages = [
            CoreMessage(role=Role.SYSTEM, content=_SPAN_SUMMARY_SYSTEM_PROMPT),
            CoreMessage(role=Role.USER, content=f"Task: {label}\nResult: {output_hint}"),
        ]
        options = ModelOptions(cache_ttl=None, retries=3, timeout=30)
        with internal_tracking_context():
            response = await generate_structured(
                messages,
                response_format=SpanSummary,
                model=model,
                model_options=options,
                purpose=f"span_summary: {label}",
            )
        return response.parsed.summary
    except Exception as e:
        logger.warning(f"Span summary failed for '{label}': {e}")
        return ""


async def generate_document_summary(name: str, excerpt: str, model: str = "gemini-3-flash") -> str:
    """Generate structured metadata for a document.

    Returns JSON-serialized DocumentSummary (stored in document_index.summary).
    """
    try:
        messages = [
            CoreMessage(role=Role.SYSTEM, content=_DOC_SUMMARY_SYSTEM_PROMPT),
            CoreMessage(role=Role.USER, content=f"Document: {name}\nContent excerpt:\n{excerpt}"),
        ]
        options = ModelOptions(cache_ttl=None, retries=3, timeout=30)
        with internal_tracking_context():
            response = await generate_structured(
                messages,
                response_format=DocumentSummary,
                model=model,
                model_options=options,
                purpose=f"document_summary: {name}",
            )
        return response.parsed.model_dump_json()
    except Exception as e:
        logger.warning(f"Document summary failed for '{name}': {e}")
        return ""
