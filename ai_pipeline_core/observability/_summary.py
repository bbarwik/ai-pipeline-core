"""LLM-powered summary generation for tracked spans and documents."""

from pydantic import BaseModel, Field

from ai_pipeline_core.llm import generate_structured
from ai_pipeline_core.llm.model_options import ModelOptions
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
        with internal_tracking_context():
            result = await generate_structured(
                model=model,
                response_format=SpanSummary,
                messages=f"Task: {label}\nResult: {output_hint}",
                options=ModelOptions(system_prompt=_SPAN_SUMMARY_SYSTEM_PROMPT, cache_ttl=None, retries=3, timeout=30),
                purpose=f"span_summary: {label}",
            )
        return result.parsed.summary
    except Exception as e:
        logger.warning(f"Span summary failed for '{label}': {e}")
        return ""


async def generate_document_summary(name: str, excerpt: str, model: str = "gemini-3-flash") -> str:
    """Generate structured metadata for a document.

    Returns JSON-serialized DocumentSummary (stored in document_index.summary).
    """
    try:
        with internal_tracking_context():
            result = await generate_structured(
                model=model,
                response_format=DocumentSummary,
                messages=f"Document: {name}\nContent excerpt:\n{excerpt}",
                options=ModelOptions(system_prompt=_DOC_SUMMARY_SYSTEM_PROMPT, cache_ttl=None, retries=3, timeout=30),
                purpose=f"document_summary: {name}",
            )
        return result.parsed.model_dump_json()
    except Exception as e:
        logger.warning(f"Document summary failed for '{name}': {e}")
        return ""
