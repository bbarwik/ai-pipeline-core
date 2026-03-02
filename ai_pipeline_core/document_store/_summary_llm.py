"""LLM-powered summary generation for stored documents.

Uses _llm_core directly. Decoupled from observability module.
"""

from pydantic import BaseModel, Field

from ai_pipeline_core._llm_core import CoreMessage, ModelOptions, Role, generate_structured
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

_DOC_SUMMARY_SYSTEM_PROMPT = "You generate metadata for documents in a research pipeline dashboard."

_DOC_SUMMARY_USER_TEMPLATE = """\
Generate a short_title and summary for the document below.

Rules:
- short_title: Propose a concise document title (max 50 chars) that reflects the document's purpose and goal based on its <class>, <description>, and <content>
- short_title: For single website page documents, use format 'domain.com: Page Title'. Only applies when the entire document is content from one website
- summary: 1-2 sentences (max 50 words) describing the document's purpose and content type
- No sensitive data (URLs, personal names, company details)
- Pay attention to the <class> and <description> tags — they indicate the document's role in the pipeline

{excerpt}"""


class DocumentSummary(BaseModel):
    """Structured output for document summaries."""

    short_title: str = Field(description="Concise document title reflecting its purpose and goal, max 50 characters")
    summary: str = Field(description="1-2 sentences (max 50 words) describing the document's purpose and content type")


async def generate_document_summary(name: str, excerpt: str, model: str = "gemini-3-flash") -> str:
    """Generate structured metadata for a document.

    Returns JSON-serialized DocumentSummary (stored in document_index.summary).
    """
    try:
        user_content = _DOC_SUMMARY_USER_TEMPLATE.format(excerpt=excerpt)
        messages = [
            CoreMessage(role=Role.SYSTEM, content=_DOC_SUMMARY_SYSTEM_PROMPT),
            CoreMessage(role=Role.USER, content=user_content),
        ]
        options = ModelOptions(cache_ttl=None, retries=3, timeout=30, reasoning_effort="low")
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
