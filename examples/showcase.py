#!/usr/bin/env python3
"""
Example: end‑to‑end showcase of ai_pipeline_core

This example demonstrates, in one file, how to:
  • Define typed Documents (FlowDocument)
  • Use PromptManager to render a Jinja2 prompt
  • Call LLMs with both raw and structured responses (generate / generate_structured)
  • Configure a Flow with FlowConfig and run a Prefect @flow of @task steps
  • Use tracing (@trace) and the unified pipeline logger

The script is intentionally self-contained: on first run it will create a
local prompts/analyze.jinja2 file next to this script so you can run it
without any extra setup beyond environment variables.

Prereqs (environment):
  - OPENAI_API_KEY (or compatible LiteLLM proxy key)
  - OPENAI_BASE_URL (LiteLLM proxy or OpenAI-compatible endpoint)

Run:
  python examples/showcase.py "Your text to analyze"

Tip: For richer logs, set PREFECT_LOGGING_LEVEL=INFO
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Iterable

from prefect import flow, task
from pydantic import BaseModel

from ai_pipeline_core import (
    PromptManager,
    settings,
    setup_logging,
    trace,
)
from ai_pipeline_core.documents import DocumentList, FlowDocument
from ai_pipeline_core.flow import FlowConfig
from ai_pipeline_core.llm import (
    AIMessages,
    ModelName,
    ModelOptions,
    StructuredModelResponse,
    generate,
    generate_structured,
)
from ai_pipeline_core.logging import get_pipeline_logger

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
setup_logging()  # use defaults; respects Prefect logging env vars
logger = get_pipeline_logger(__name__)


# -----------------------------------------------------------------------------
# Document types for this flow
# -----------------------------------------------------------------------------
class InputText(FlowDocument):
    """Plain text input to be analyzed."""


class AnalysisReport(FlowDocument):
    """JSON report produced by the flow (as UTF-8 bytes content)."""


# -----------------------------------------------------------------------------
# Flow configuration (input/output type checks & helpers)
# -----------------------------------------------------------------------------
class DemoFlowConfig(FlowConfig):
    INPUT_DOCUMENT_TYPES = [InputText]
    OUTPUT_DOCUMENT_TYPE = AnalysisReport


# -----------------------------------------------------------------------------
# Prompt template bootstrap (keeps example self-contained)
# -----------------------------------------------------------------------------
TEMPLATE_NAME = "analyze.jinja2"


def ensure_prompt_exists(base_dir: Path) -> Path:
    prompts_dir = base_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    template_path = prompts_dir / TEMPLATE_NAME

    if not template_path.exists():
        template_path.write_text(
            (
                """You are a senior analyst.\n"""
                """Summarize the following text clearly and concisely.\n"""
                """Return only what is asked for; do not add extra text.\n\n"""
                """Text:\n"""
                """{{ text }}\n\n"""
                """Constraints:\n"""
                """- Title: a punchy, 5–8 word headline\n"""
                """- Summary: 2–3 short sentences\n"""
                """- Bullets: 3–5 terse bullet points\n"""
            ),
            encoding="utf-8",
        )
        logger.info("Created example prompt at %s", template_path)

    return template_path


# -----------------------------------------------------------------------------
# Structured response schema (for generate_structured)
# -----------------------------------------------------------------------------
class ReportSchema(BaseModel):
    title: str
    summary: str
    bullets: list[str]


# -----------------------------------------------------------------------------
# Tasks
# -----------------------------------------------------------------------------
@task
@trace(name="render_prompt")
def render_prompt(doc: InputText) -> str:
    """Render a Jinja2 prompt using PromptManager.

    PromptManager looks in the script directory and in a sibling
    "prompts/" directory. We ensure the file exists beforehand.
    """
    # Ensure the prompt exists (first run convenience)
    ensure_prompt_exists(Path(__file__).parent)
    pm = PromptManager(__file__)
    # Render the template; your variables become available to Jinja2
    return pm.get(TEMPLATE_NAME, text=doc.as_text())


@task
@trace(name="llm_one_liner")
async def one_liner_summary(model: ModelName, doc: InputText) -> str:
    """Quick one-line summary using the raw text generation API."""
    context = AIMessages([doc])  # documents in context are cached by the client
    messages = AIMessages(["Give a single concise one-line summary of the provided document."])

    resp = await generate(
        model=model,
        context=context,
        messages=messages,
        options=ModelOptions(
            system_prompt="You summarize text clearly and tersely.",
            max_completion_tokens=256,
            timeout=60,
        ),
    )
    return resp.content.strip()


@task
@trace(name="llm_structured")
async def structured_report(model: ModelName, doc: InputText, rendered_prompt: str) -> ReportSchema:
    """Structured report using generate_structured -> ReportSchema."""
    response: StructuredModelResponse[ReportSchema] = await generate_structured(
        model=model,
        response_format=ReportSchema,
        context=AIMessages([doc]),
        messages=AIMessages([rendered_prompt]),
        options=ModelOptions(
            system_prompt=(
                "You are a precise, neutral analyst. Respond in the exact schema I require."
            ),
            max_completion_tokens=2048,
            timeout=120,
        ),
    )
    return response.parsed


@task
@trace(name="build_output_document")
def build_output_document(doc: InputText, one_liner: str, report: ReportSchema) -> AnalysisReport:
    """Pack the structured result into an output FlowDocument as JSON bytes."""
    payload = {
        "source_id": doc.id,
        "source_name": doc.name,
        "one_liner": one_liner,
        "report": report.model_dump(),
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return AnalysisReport(name="analysis_report.json", content=content)


# -----------------------------------------------------------------------------
# The flow
# -----------------------------------------------------------------------------
@flow(name="ai-pipeline-core-demo")
async def demo_flow(documents: DocumentList, model: ModelName = "gpt-5-mini") -> DocumentList:
    # Validate inputs
    cfg = DemoFlowConfig()
    input_docs = cfg.get_input_documents(documents)

    outputs = DocumentList(validate_same_type=True)

    # Simple serial composition for clarity in example
    for doc in input_docs:
        prompt = render_prompt(doc)
        one = await one_liner_summary(model, doc)  # type: ignore[misc]
        rep = await structured_report(model, doc, prompt)  # type: ignore[misc]
        out: AnalysisReport = build_output_document(doc, one, rep)  # type: ignore[misc]
        outputs.append(out)

    # Validate outputs
    cfg.validate_output_documents(outputs)
    return outputs


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------
async def _amain(args: list[str]) -> int:
    if not (settings.openai_api_key and settings.openai_base_url):
        logger.warning(
            "OPENAI_API_KEY / OPENAI_BASE_URL not set; real LLM calls will fail. "
            "Configure environment or a LiteLLM proxy before running."
        )

    # Compose input documents from CLI args or a default sample
    if args:
        texts: Iterable[str] = args
    else:
        texts = [
            """ai_pipeline_core focuses on minimal, typed, async building blocks for LLM flows\n"
            "with Prefect integration, structured outputs, and Jinja2 prompt management.""",
        ]

    docs = DocumentList(
        [
            InputText(name=f"input_{i + 1}.txt", content=t.encode("utf-8"))
            for i, t in enumerate(texts)
        ]
    )

    results = await demo_flow(docs)

    # Log the pretty JSON of the first result so users see output quickly
    if results:
        logger.info("First result (JSON):\n%s", results[0].content.decode("utf-8"))

    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(_amain(sys.argv[1:])))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
