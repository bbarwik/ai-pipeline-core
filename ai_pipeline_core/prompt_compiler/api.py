"""High-level API for sending prompt specifications to LLMs."""

import re
from typing import Any, cast, overload

from pydantic import BaseModel
from typing_extensions import TypeVar

from ai_pipeline_core._llm_core import ModelOptions
from ai_pipeline_core.documents import Document
from ai_pipeline_core.llm import Conversation

from .render import RESULT_CLOSE, RESULT_TAG, render_text
from .spec import PromptSpec

U = TypeVar("U", bound=BaseModel)

# Model families that support stop sequences. Add more as needed.
_STOP_SEQUENCE_MODELS: frozenset[str] = frozenset({"gemini"})

_EXTRACT_PATTERN = re.compile(rf"<{RESULT_TAG}>(.*?)(?:</{RESULT_TAG}>|$)", re.DOTALL)


def _supports_stop_sequence(model: str) -> bool:
    """Check if a model supports stop sequences based on its name."""
    model_lower = model.lower()
    return any(family in model_lower for family in _STOP_SEQUENCE_MODELS)


def extract_result(text: str) -> str:
    """Extract content from <result> tags. Returns text as-is if no tags found."""
    match = _EXTRACT_PATTERN.search(text)
    return match.group(1).strip() if match else text


@overload
async def send_spec(
    spec: PromptSpec[str],
    *,
    model: str | None = None,
    conversation: Conversation[Any] | None = None,
    documents: list[Document] | None = None,
    model_options: ModelOptions | None = None,
    include_input_documents: bool = True,
    purpose: str | None = None,
    expected_cost: float | None = None,
) -> Conversation[None]: ...


@overload
async def send_spec(  # noqa: UP047
    spec: PromptSpec[U],
    *,
    model: str | None = None,
    conversation: Conversation[Any] | None = None,
    documents: list[Document] | None = None,
    model_options: ModelOptions | None = None,
    include_input_documents: bool = True,
    purpose: str | None = None,
    expected_cost: float | None = None,
) -> "Conversation[U]": ...


async def send_spec(
    spec: PromptSpec[Any],
    *,
    model: str | None = None,
    conversation: Conversation[Any] | None = None,
    documents: list[Document] | None = None,
    model_options: ModelOptions | None = None,
    include_input_documents: bool = True,
    purpose: str | None = None,
    expected_cost: float | None = None,
) -> Conversation[Any]:
    r"""Send a PromptSpec to an LLM via Conversation.

    Either ``model`` or ``conversation`` must be provided. When ``conversation``
    is given, uses it directly (enables warmup+fork pattern). When ``model``
    is given, creates a fresh Conversation.

    Adds documents to context, renders the prompt, and sends it.
    Dispatches to send() or send_structured() based on the spec's output type.

    When spec.xml_wrapped is True and the model supports stop sequences,
    automatically sets stop=[\"</result>\"] to cut off after the response.
    Use extract_result(conv.content) to get clean content from xml_wrapped specs.
    """
    if conversation is not None:
        conv: Conversation[Any] = conversation
        if model_options is not None:
            conv = conv.with_model_options(model_options)
    elif model is not None:
        conv = Conversation(model=model, model_options=model_options)
    else:
        raise ValueError("Either 'model' or 'conversation' must be provided")

    spec_cls = type(spec)

    # Set stop sequence for xml_wrapped specs on supported models
    if spec_cls.xml_wrapped and _supports_stop_sequence(conv.model):
        current_options = conv.model_options or ModelOptions()
        existing_stop = current_options.stop
        if existing_stop is None:
            stop_list = [RESULT_CLOSE]
        elif isinstance(existing_stop, str):
            stop_list = [existing_stop, RESULT_CLOSE]
        else:
            stop_list = [*existing_stop, RESULT_CLOSE]
        conv = conv.with_model_options(current_options.model_copy(update={"stop": stop_list}))

    if documents:
        conv = conv.with_context(*documents)

    prompt_text = render_text(spec, documents=documents, include_input_documents=include_input_documents)
    trace_purpose = purpose or spec_cls.__name__

    if spec_cls.output_type is str:
        return await conv.send(prompt_text, purpose=trace_purpose, expected_cost=expected_cost)

    response_format = cast(type[BaseModel], spec_cls.output_type)
    return await conv.send_structured(prompt_text, response_format=response_format, purpose=trace_purpose, expected_cost=expected_cost)


__all__ = ["extract_result", "send_spec"]
