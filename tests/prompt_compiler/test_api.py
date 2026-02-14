"""Tests for prompt_compiler.api (send_spec, extract_result, stop sequences)."""

import pytest
from pydantic import BaseModel

import ai_pipeline_core.prompt_compiler.api as api
from ai_pipeline_core._llm_core import ModelOptions
from ai_pipeline_core.documents import Document
from ai_pipeline_core.prompt_compiler.components import Role
from ai_pipeline_core.prompt_compiler.spec import PromptSpec
from ai_pipeline_core.prompt_compiler.types import Phase


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


class ApiDoc(Document):
    """API input document."""


class ApiRole(Role):
    """API role."""

    text = "careful evaluator"


class StructuredResponse(BaseModel):
    """Structured output model."""

    answer: str


class PlainSpec(PromptSpec, phase=Phase("review")):
    """Plain text output spec."""

    input_documents = ()
    role = ApiRole
    task = "Respond briefly"


class XmlSpec(PromptSpec, phase=Phase("review")):
    """XML-wrapped text output spec."""

    input_documents = ()
    role = ApiRole
    task = "Respond in wrapped text"
    xml_wrapped = True


class StructuredSpec(PromptSpec[StructuredResponse], phase=Phase("review")):
    """Structured output spec."""

    input_documents = ()
    role = ApiRole
    task = "Return structured data"


class FakeConversation:
    """Test double for Conversation."""

    created: list["FakeConversation"] = []

    def __init__(self, *, model: str, model_options: ModelOptions | None = None) -> None:
        self.model = model
        self.model_options = model_options
        self.with_context_calls: list[tuple[Document, ...]] = []
        self.with_model_options_calls: list[ModelOptions] = []
        self.send_calls: list[dict[str, object]] = []
        self.send_structured_calls: list[dict[str, object]] = []
        type(self).created.append(self)

    def with_context(self, *documents: Document) -> "FakeConversation":
        self.with_context_calls.append(documents)
        return self

    def with_model_options(self, options: ModelOptions) -> "FakeConversation":
        self.with_model_options_calls.append(options)
        self.model_options = options
        return self

    async def send(self, prompt_text: str, *, purpose: str, expected_cost: float | None = None) -> "FakeConversation":
        self.send_calls.append({"prompt_text": prompt_text, "purpose": purpose})
        return self

    async def send_structured(
        self, prompt_text: str, *, response_format: type[BaseModel], purpose: str, expected_cost: float | None = None
    ) -> "FakeConversation":
        self.send_structured_calls.append({"prompt_text": prompt_text, "response_format": response_format, "purpose": purpose})
        return self


@pytest.fixture(autouse=True)
def _reset_fake_conversations() -> None:
    FakeConversation.created.clear()


@pytest.fixture
def fake_conversation(monkeypatch: pytest.MonkeyPatch) -> type[FakeConversation]:
    monkeypatch.setattr(api, "Conversation", FakeConversation)
    return FakeConversation


@pytest.fixture
def render_spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_render_text(spec: PromptSpec, *, documents: list[Document] | None = None, include_input_documents: bool = True) -> str:
        calls.append({"spec": spec, "documents": documents, "include_input_documents": include_input_documents})
        return "RENDERED_PROMPT"

    monkeypatch.setattr(api, "render_text", fake_render_text)
    return calls


# ---------------------------------------------------------------------------
# extract_result
# ---------------------------------------------------------------------------


@pytest.mark.ai_docs
def test_extract_result_with_closing_tag() -> None:
    text = "preamble <result>\n  hello world \n</result> trailing"
    assert api.extract_result(text) == "hello world"


def test_extract_result_without_tags() -> None:
    text = "plain response"
    assert api.extract_result(text) == text


def test_extract_result_with_incomplete_closing_tag() -> None:
    text = "prefix <result>partial content"
    assert api.extract_result(text) == "partial content"


def test_extract_result_multiline_content() -> None:
    text = "<result>\nline1\nline2\n</result>"
    assert api.extract_result(text) == "line1\nline2"


def test_extract_result_empty_content() -> None:
    text = "<result></result>"
    assert api.extract_result(text) == ""


def test_extract_result_whitespace_stripped() -> None:
    text = "<result>  \n  content  \n  </result>"
    assert api.extract_result(text) == "content"


# ---------------------------------------------------------------------------
# _supports_stop_sequence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-3-flash", True),
        ("GEMINI-2.5-PRO", True),
        ("openai/gemini-proxy", True),
        ("gpt-5.1", False),
        ("claude-4", False),
        ("grok-4.1-fast", False),
    ],
)
def test_supports_stop_sequence(model: str, expected: bool) -> None:
    assert api._supports_stop_sequence(model) is expected


# ---------------------------------------------------------------------------
# send_spec: text output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_spec_text_output(fake_conversation: type[FakeConversation], render_spy: list[dict[str, object]]) -> None:
    result = await api.send_spec(PlainSpec(), model="gpt-5")
    conv = fake_conversation.created[-1]

    assert result is conv
    assert conv.model == "gpt-5"
    assert conv.model_options is None
    assert conv.with_context_calls == []
    assert len(conv.send_calls) == 1
    assert conv.send_calls[0]["prompt_text"] == "RENDERED_PROMPT"
    assert conv.send_calls[0]["purpose"] == "PlainSpec"
    assert conv.send_structured_calls == []
    assert render_spy[-1]["documents"] is None
    assert render_spy[-1]["include_input_documents"] is True


@pytest.mark.asyncio
async def test_send_spec_custom_purpose(fake_conversation: type[FakeConversation], render_spy: list[dict[str, object]]) -> None:
    await api.send_spec(PlainSpec(), model="gpt-5", purpose="custom-purpose")
    conv = fake_conversation.created[-1]
    assert conv.send_calls[0]["purpose"] == "custom-purpose"


# ---------------------------------------------------------------------------
# send_spec: structured output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_spec_structured_output(fake_conversation: type[FakeConversation], render_spy: list[dict[str, object]]) -> None:
    doc = ApiDoc(name="input.txt", content=b"data", description="Doc desc")
    options = ModelOptions(timeout=1)

    result = await api.send_spec(
        StructuredSpec(),
        model="gpt-5",
        documents=[doc],
        model_options=options,
        include_input_documents=False,
        purpose="custom",
    )
    conv = fake_conversation.created[-1]

    assert result is conv
    assert conv.model_options is options
    assert conv.with_context_calls == [(doc,)]
    assert conv.send_calls == []
    assert conv.send_structured_calls[0]["response_format"] is StructuredResponse
    assert conv.send_structured_calls[0]["purpose"] == "custom"
    assert render_spy[-1]["documents"] == [doc]
    assert render_spy[-1]["include_input_documents"] is False


# ---------------------------------------------------------------------------
# send_spec: documents wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_spec_with_documents(fake_conversation: type[FakeConversation], render_spy: list[dict[str, object]]) -> None:
    doc = ApiDoc(name="input.txt", content=b"data")
    await api.send_spec(PlainSpec(), model="gpt-5", documents=[doc])
    conv = fake_conversation.created[-1]
    assert conv.with_context_calls == [(doc,)]
    assert render_spy[-1]["documents"] == [doc]


@pytest.mark.asyncio
async def test_send_spec_empty_documents_skips_with_context(fake_conversation: type[FakeConversation], render_spy: list[dict[str, object]]) -> None:
    await api.send_spec(PlainSpec(), model="gpt-5", documents=[])
    conv = fake_conversation.created[-1]
    assert conv.with_context_calls == []
    assert render_spy[-1]["documents"] == []


# ---------------------------------------------------------------------------
# send_spec: xml_wrapped + stop sequences
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_spec_xml_gemini_no_existing_options(fake_conversation: type[FakeConversation], render_spy: list[dict[str, object]]) -> None:
    """xml_wrapped + gemini + model_options=None -> creates ModelOptions with stop via with_model_options."""
    await api.send_spec(XmlSpec(), model="gemini-3-flash")
    conv = fake_conversation.created[-1]
    assert len(conv.with_model_options_calls) == 1
    assert conv.model_options.stop == [api.RESULT_CLOSE]


@pytest.mark.asyncio
async def test_send_spec_xml_gemini_existing_stop_none(fake_conversation: type[FakeConversation], render_spy: list[dict[str, object]]) -> None:
    """xml_wrapped + gemini + existing options with stop=None -> adds stop."""
    options = ModelOptions(stop=None)
    await api.send_spec(XmlSpec(), model="gemini-3-flash", model_options=options)
    conv = fake_conversation.created[-1]
    assert conv.model_options.stop == [api.RESULT_CLOSE]


@pytest.mark.asyncio
async def test_send_spec_xml_gemini_existing_stop_string(fake_conversation: type[FakeConversation], render_spy: list[dict[str, object]]) -> None:
    """xml_wrapped + gemini + existing stop as string -> merges both."""
    options = ModelOptions(stop="DONE")
    await api.send_spec(XmlSpec(), model="gemini-3-flash", model_options=options)
    conv = fake_conversation.created[-1]
    assert conv.model_options.stop == ["DONE", api.RESULT_CLOSE]


@pytest.mark.asyncio
async def test_send_spec_xml_gemini_existing_stop_list(fake_conversation: type[FakeConversation], render_spy: list[dict[str, object]]) -> None:
    """xml_wrapped + gemini + existing stop as list -> appends."""
    options = ModelOptions(stop=["A", "B"])
    await api.send_spec(XmlSpec(), model="gemini-3-flash", model_options=options)
    conv = fake_conversation.created[-1]
    assert conv.model_options.stop == ["A", "B", api.RESULT_CLOSE]


@pytest.mark.asyncio
async def test_send_spec_xml_non_gemini_no_stop_injection(fake_conversation: type[FakeConversation], render_spy: list[dict[str, object]]) -> None:
    """xml_wrapped + non-gemini model -> options not modified."""
    options = ModelOptions(stop=["END"])
    await api.send_spec(XmlSpec(), model="gpt-5", model_options=options)
    conv = fake_conversation.created[-1]
    assert conv.model_options is options
    assert conv.model_options.stop == ["END"]


@pytest.mark.asyncio
async def test_send_spec_non_xml_gemini_no_stop_injection(fake_conversation: type[FakeConversation], render_spy: list[dict[str, object]]) -> None:
    """Non-xml_wrapped spec + gemini -> no stop injection."""
    options = ModelOptions(stop=None)
    await api.send_spec(PlainSpec(), model="gemini-3-flash", model_options=options)
    conv = fake_conversation.created[-1]
    assert conv.model_options is options
    assert conv.model_options.stop is None


@pytest.mark.asyncio
async def test_send_spec_xml_non_gemini_no_options(fake_conversation: type[FakeConversation], render_spy: list[dict[str, object]]) -> None:
    """xml_wrapped + non-gemini + no options -> options stays None."""
    await api.send_spec(XmlSpec(), model="gpt-5")
    conv = fake_conversation.created[-1]
    assert conv.model_options is None


# ---------------------------------------------------------------------------
# send_spec: conversation parameter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_spec_with_conversation(fake_conversation: type[FakeConversation], render_spy: list[dict[str, object]]) -> None:
    """send_spec with conversation= uses existing conversation."""
    existing = FakeConversation(model="gemini-3-pro")
    result = await api.send_spec(PlainSpec(), conversation=existing)
    assert result is existing
    assert existing.send_calls[0]["prompt_text"] == "RENDERED_PROMPT"


@pytest.mark.asyncio
async def test_send_spec_requires_model_or_conversation(render_spy: list[dict[str, object]]) -> None:
    """send_spec without model or conversation raises ValueError."""
    with pytest.raises(ValueError, match="Either 'model' or 'conversation' must be provided"):
        await api.send_spec(PlainSpec())
