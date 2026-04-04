"""Unit tests for ai_pipeline_core._llm_core.client — pure helpers and generate()."""

import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from ai_pipeline_core._llm_core.client import (
    _apply_cache_control,
    _build_model_response,
    _compute_cache_key,
    _content_to_api_parts,
    _estimate_token_count,
    _extract_cost,
    _extract_usage,
    _messages_to_api,
    _model_name_to_openrouter_model,
    _remove_cache_control,
)
from ai_pipeline_core._llm_core.types import (
    TOKENS_PER_IMAGE,
    CoreMessage,
    ImageContent,
    PDFContent,
    Role,
    TextContent,
)
from ai_pipeline_core.exceptions import LLMError


# ---------------------------------------------------------------------------
# Helpers for building mock responses
# ---------------------------------------------------------------------------


def _make_response(
    content: str = "Hello",
    finish_reason: str = "stop",
    prompt: int = 10,
    completion: int = 5,
    reasoning_content: str | None = None,
    annotations: list[Any] | None = None,
) -> SimpleNamespace:
    msg = SimpleNamespace(
        content=content,
        role="assistant",
        reasoning_content=reasoning_content,
        annotations=annotations or [],
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    )
    return SimpleNamespace(id="resp-1", choices=[choice], usage=usage)


# ---------------------------------------------------------------------------
# _content_to_api_parts
# ---------------------------------------------------------------------------


class TestContentToApiParts:
    def test_string_input(self):
        result = _content_to_api_parts("hello")
        assert result == [{"type": "text", "text": "hello"}]

    def test_text_content(self):
        tc = TextContent(text="world")
        result = _content_to_api_parts(tc)
        assert result == [{"type": "text", "text": "world"}]

    def test_image_valid(self):
        # Create a minimal valid PNG — ImageContent uses Base64Bytes, so pass b64-encoded
        import struct
        import zlib

        def _make_png() -> bytes:
            sig = b"\x89PNG\r\n\x1a\n"
            ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
            ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
            raw = zlib.compress(b"\x00\x00\x00\x00")
            idat_crc = zlib.crc32(b"IDAT" + raw) & 0xFFFFFFFF
            idat = struct.pack(">I", len(raw)) + b"IDAT" + raw + struct.pack(">I", idat_crc)
            iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
            iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
            return sig + ihdr + idat + iend

        png_data = _make_png()
        b64_encoded = base64.b64encode(png_data)
        ic = ImageContent(data=b64_encoded, mime_type="image/png")
        result = _content_to_api_parts(ic)
        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        b64_str = base64.b64encode(png_data).decode("utf-8")
        assert f"data:image/png;base64,{b64_str}" == result[0]["image_url"]["url"]

    def test_image_invalid(self):
        # Base64Bytes needs valid base64 input — encode some invalid image bytes
        bad_bytes = b"not-a-real-image-data"
        ic = ImageContent(data=base64.b64encode(bad_bytes), mime_type="image/png")
        result = _content_to_api_parts(ic)
        assert result == []

    def test_pdf_valid(self):
        pdf_data = b"%PDF-1.4 fake pdf content"
        pc = PDFContent(data=base64.b64encode(pdf_data))
        result = _content_to_api_parts(pc)
        assert len(result) == 1
        assert result[0]["type"] == "file"

    def test_pdf_text_fallback(self):
        # PDFContent text fallback: data must not contain \x00 and must not start with %PDF-
        text_data = b"This is plain text content not PDF"
        pc = PDFContent(data=base64.b64encode(text_data))
        result = _content_to_api_parts(pc)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "This is plain text content not PDF"

    def test_pdf_empty(self):
        pc = PDFContent(data=base64.b64encode(b""))
        result = _content_to_api_parts(pc)
        assert result == []

    def test_tuple_of_parts(self):
        parts = (TextContent(text="a"), TextContent(text="b"))
        result = _content_to_api_parts(parts)
        assert len(result) == 2
        assert result[0]["text"] == "a"
        assert result[1]["text"] == "b"


# ---------------------------------------------------------------------------
# _messages_to_api
# ---------------------------------------------------------------------------


class TestMessagesToApi:
    def test_converts_messages(self):
        msgs = [CoreMessage(role=Role.USER, content="hi")]
        result = _messages_to_api(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_skips_empty(self):
        # Invalid image data (Base64Bytes) — b64-encode garbage bytes so pydantic accepts it
        ic = ImageContent(data=base64.b64encode(b"bad-image"), mime_type="image/png")
        msgs = [CoreMessage(role=Role.USER, content=ic)]
        result = _messages_to_api(msgs)
        assert result == []


# ---------------------------------------------------------------------------
# _apply_cache_control / _remove_cache_control
# ---------------------------------------------------------------------------


class TestCacheControl:
    def test_apply_cache_control(self):
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
        _apply_cache_control(msgs, "300s", 1)
        assert msgs[0]["cache_control"] == {"type": "ephemeral", "ttl": "300s"}
        assert msgs[0]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "300s"}
        assert "cache_control" not in msgs[1]

    def test_remove_cache_control(self):
        msgs: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
                "cache_control": {"type": "ephemeral"},
            },
        ]
        result = _remove_cache_control(msgs)
        assert "cache_control" not in result[0]
        assert "cache_control" not in result[0]["content"][0]  # type: ignore[index]


# ---------------------------------------------------------------------------
# _extract_usage
# ---------------------------------------------------------------------------


class TestExtractUsage:
    def test_basic(self):
        resp = _make_response(prompt=100, completion=50)
        usage = _extract_usage(resp)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cached_tokens == 0
        assert usage.reasoning_tokens == 0

    def test_cached_tokens(self):
        resp = _make_response()
        resp.usage.prompt_tokens_details = SimpleNamespace(cached_tokens=42)
        usage = _extract_usage(resp)
        assert usage.cached_tokens == 42

    def test_no_usage(self):
        resp = SimpleNamespace(usage=None, id="r", choices=[])
        usage = _extract_usage(resp)
        assert usage.prompt_tokens == 0
        assert usage.total_tokens == 0


# ---------------------------------------------------------------------------
# _extract_cost
# ---------------------------------------------------------------------------


class TestExtractCost:
    def test_usage_attr(self):
        resp = _make_response()
        resp.usage.cost = 0.05
        assert _extract_cost(resp) == 0.05

    def test_header_fallback(self):
        resp = _make_response()
        assert _extract_cost(resp, header_cost=0.10) == 0.10

    def test_none(self):
        resp = _make_response()
        assert _extract_cost(resp) is None


# ---------------------------------------------------------------------------
# _estimate_token_count
# ---------------------------------------------------------------------------


class TestEstimateTokenCount:
    def test_string_content(self):
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "a" * 400}]
        count = _estimate_token_count(msgs)
        assert count == 100

    def test_image_content(self):
        msgs: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "..."}}]}]
        count = _estimate_token_count(msgs)
        assert count == TOKENS_PER_IMAGE


# ---------------------------------------------------------------------------
# _compute_cache_key
# ---------------------------------------------------------------------------


class TestComputeCacheKey:
    def test_deterministic(self):
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "test"}]
        k1 = _compute_cache_key(msgs, "sys")
        k2 = _compute_cache_key(msgs, "sys")
        assert k1 == k2
        assert len(k1) == 64

    def test_differs_on_system(self):
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "test"}]
        k1 = _compute_cache_key(msgs, "sys1")
        k2 = _compute_cache_key(msgs, "sys2")
        assert k1 != k2


# ---------------------------------------------------------------------------
# _model_name_to_openrouter_model
# ---------------------------------------------------------------------------


class TestModelNameToOpenRouter:
    def test_sonar_pro_search(self):
        assert _model_name_to_openrouter_model("sonar-pro-search") == "perplexity/sonar-pro-search"

    def test_search_suffix_replaced(self):
        result = _model_name_to_openrouter_model("gemini-3-flash-search")
        assert ":online" in result
        assert "-search" not in result

    def test_known_provider_prefix(self):
        result = _model_name_to_openrouter_model("gpt-5.4-mini")
        assert result == "openai/gpt-5.4-mini"

    def test_unknown_model_passthrough(self):
        result = _model_name_to_openrouter_model("unknown-model-xyz")
        assert result == "unknown-model-xyz"


# ---------------------------------------------------------------------------
# _build_model_response
# ---------------------------------------------------------------------------


class TestBuildModelResponse:
    def test_basic(self):
        resp = _make_response(content="Answer", prompt=10, completion=5)
        mr = _build_model_response(resp, {"time_taken": 1.0}, None, "test-model", None)
        assert mr.content == "Answer"
        assert mr.parsed == "Answer"
        assert mr.usage.prompt_tokens == 10
        assert mr.model == "test-model"

    def test_empty_content_raises(self):
        from ai_pipeline_core.exceptions import EmptyResponseError

        resp = _make_response(content="")
        with pytest.raises(EmptyResponseError, match="Empty response content"):
            _build_model_response(resp, {}, None, "m", None)

    def test_thinking_stripped(self):
        resp = _make_response(content="<think>internal</think>Final answer")
        mr = _build_model_response(resp, {}, None, "m", None)
        assert mr.content == "Final answer"
        assert mr.reasoning_content == "<think>internal"

    def test_structured_output(self):
        class MyModel(BaseModel):
            value: int

        resp = _make_response(content='{"value": 42}')
        mr = _build_model_response(resp, {}, None, "m", MyModel)
        assert isinstance(mr.parsed, MyModel)
        assert mr.parsed.value == 42

    def test_citations(self):
        citation = SimpleNamespace(
            type="url_citation",
            url_citation=SimpleNamespace(title="Source", url="https://example.com", start_index=0, end_index=5),
        )
        resp = _make_response(content="Answer", annotations=[citation])
        mr = _build_model_response(resp, {}, None, "m", None)
        assert len(mr.citations) == 1
        assert mr.citations[0].title == "Source"

    def test_stream_usage_overrides(self):
        resp = _make_response(content="ok", prompt=10, completion=5)
        stream_usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        mr = _build_model_response(resp, {}, stream_usage, "m", None)
        assert mr.usage.prompt_tokens == 100
        assert mr.usage.completion_tokens == 50

    def test_invalid_finish_reason_normalized(self):
        resp = _make_response(content="ok", finish_reason="weird_reason")
        mr = _build_model_response(resp, {}, None, "m", None)
        assert mr.content == "ok"

    def test_wrong_role_normalized(self):
        resp = _make_response(content="ok")
        resp.choices[0].message.role = "tool"
        mr = _build_model_response(resp, {}, None, "m", None)
        assert mr.content == "ok"

    def test_cost_from_header(self):
        resp = _make_response(content="ok")
        mr = _build_model_response(resp, {"header_cost": 0.05}, None, "m", None)
        assert mr.cost == 0.05

    def test_thinking_blocks(self):
        resp = _make_response(content="answer")
        resp.choices[0].message.thinking_blocks = [{"type": "thinking", "thinking": "hmm"}]
        mr = _build_model_response(resp, {}, None, "m", None)
        assert mr.thinking_blocks is not None
        assert len(mr.thinking_blocks) == 1

    def test_provider_specific_fields(self):
        resp = _make_response(content="answer")
        resp.choices[0].message.provider_specific_fields = {"thought_signatures": "abc"}
        mr = _build_model_response(resp, {}, None, "m", None)
        assert mr.provider_specific_fields is not None
        assert mr.provider_specific_fields["thought_signatures"] == "abc"


# ---------------------------------------------------------------------------
# generate() — async tests with mocked AsyncOpenAI
# ---------------------------------------------------------------------------


class TestGenerateImpl:
    async def test_empty_messages_raises(self):
        from ai_pipeline_core._llm_core.client import _generate_impl
        from ai_pipeline_core._llm_core.types import ModelOptions

        with pytest.raises(ValueError, match="messages must not be empty"):
            await _generate_impl([], model="m", model_options=ModelOptions())

    async def test_empty_model_raises(self):
        from ai_pipeline_core._llm_core.client import _generate_impl
        from ai_pipeline_core._llm_core.types import ModelOptions

        msg = CoreMessage(role=Role.USER, content="hi")
        with pytest.raises(ValueError, match="model must be provided"):
            await _generate_impl([msg], model="", model_options=ModelOptions())

    @patch("ai_pipeline_core._llm_core.client.settings")
    @patch("ai_pipeline_core._llm_core.client.AsyncOpenAI")
    @patch("ai_pipeline_core._llm_core.client.detect_output_degeneration", return_value=None)
    async def test_generate_successful_non_streaming(self, mock_degen, mock_aoai, mock_settings):
        from ai_pipeline_core._llm_core.client import _generate_impl
        from ai_pipeline_core._llm_core.types import ModelOptions

        mock_settings.openai_api_key = "key"
        mock_settings.openai_base_url = "http://localhost:4000"

        resp = _make_response(content="OK", prompt=10, completion=5)
        raw_response = MagicMock()
        raw_response.parse.return_value = resp
        raw_response.headers = {}

        mock_client = AsyncMock()
        mock_client.chat.completions.with_raw_response.create = AsyncMock(return_value=raw_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_aoai.return_value = mock_client

        msg = CoreMessage(role=Role.USER, content="hi")
        opts = ModelOptions(retries=0, cache_ttl=None)
        result = await _generate_impl([msg], model="test-model", model_options=opts)
        assert result.content == "OK"

    @patch("ai_pipeline_core._llm_core.client.settings")
    @patch("ai_pipeline_core._llm_core.client.AsyncOpenAI")
    async def test_generate_retry_on_timeout(self, mock_aoai, mock_settings):
        from ai_pipeline_core._llm_core.client import _generate_impl
        from ai_pipeline_core._llm_core.types import ModelOptions

        mock_settings.openai_api_key = "key"
        mock_settings.openai_base_url = "http://localhost:4000"

        mock_client = AsyncMock()
        mock_client.chat.completions.with_raw_response.create = AsyncMock(side_effect=TimeoutError())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_aoai.return_value = mock_client

        msg = CoreMessage(role=Role.USER, content="hi")
        opts = ModelOptions(retries=0, retry_delay_seconds=0, cache_ttl=None)
        with pytest.raises(LLMError, match="Exhausted all retry attempts"):
            await _generate_impl([msg], model="test-model", model_options=opts)

    @patch("ai_pipeline_core._llm_core.client.settings")
    @patch("ai_pipeline_core._llm_core.client.AsyncOpenAI")
    async def test_generate_retry_on_generic_exception(self, mock_aoai, mock_settings):
        from ai_pipeline_core._llm_core.client import _generate_impl
        from ai_pipeline_core._llm_core.types import ModelOptions

        mock_settings.openai_api_key = "key"
        mock_settings.openai_base_url = "http://localhost:4000"

        mock_client = AsyncMock()
        mock_client.chat.completions.with_raw_response.create = AsyncMock(side_effect=RuntimeError("API error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_aoai.return_value = mock_client

        msg = CoreMessage(role=Role.USER, content="hi")
        opts = ModelOptions(retries=1, retry_delay_seconds=0, cache_ttl=None)
        with pytest.raises(LLMError, match="Exhausted all retry attempts"):
            await _generate_impl([msg], model="test-model", model_options=opts)
        # retries=1 means 1 original + 1 retry = 2 calls total
        assert mock_client.chat.completions.with_raw_response.create.call_count == 2

    @patch("ai_pipeline_core._llm_core.client.settings")
    @patch("ai_pipeline_core._llm_core.client.AsyncOpenAI")
    @patch("ai_pipeline_core._llm_core.client.detect_output_degeneration", return_value=None)
    async def test_generate_with_system_prompt(self, mock_degen, mock_aoai, mock_settings):
        from ai_pipeline_core._llm_core.client import _generate_impl
        from ai_pipeline_core._llm_core.types import ModelOptions

        mock_settings.openai_api_key = "key"
        mock_settings.openai_base_url = "http://localhost:4000"

        resp = _make_response(content="OK", prompt=10, completion=5)
        raw_response = MagicMock()
        raw_response.parse.return_value = resp
        raw_response.headers = {}

        mock_client = AsyncMock()
        mock_client.chat.completions.with_raw_response.create = AsyncMock(return_value=raw_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_aoai.return_value = mock_client

        msg = CoreMessage(role=Role.USER, content="hi")
        opts = ModelOptions(retries=0, cache_ttl=None, system_prompt="You are helpful")
        result = await _generate_impl([msg], model="test-model", model_options=opts)
        assert result.content == "OK"
        # Verify system prompt was included in the API call
        call_kwargs = mock_client.chat.completions.with_raw_response.create.call_args[1]
        messages = call_kwargs["messages"]
        system_messages = [m for m in messages if m.get("role") == "system"]
        assert len(system_messages) > 0, f"No system message found in: {messages}"
        # Content can be string or list of typed parts
        sys_content = system_messages[0]["content"]
        assert "You are helpful" in str(sys_content), f"System prompt not found in: {sys_content}"

    @patch("ai_pipeline_core._llm_core.client.settings")
    @patch("ai_pipeline_core._llm_core.client.AsyncOpenAI")
    @patch("ai_pipeline_core._llm_core.client.detect_output_degeneration", return_value=None)
    async def test_generate_openrouter_model_conversion(self, mock_degen, mock_aoai, mock_settings):
        from ai_pipeline_core._llm_core.client import _generate_impl
        from ai_pipeline_core._llm_core.types import ModelOptions

        mock_settings.openai_api_key = "key"
        mock_settings.openai_base_url = "https://openrouter.ai/api/v1"

        resp = _make_response(content="OK", prompt=10, completion=5)
        raw_response = MagicMock()
        raw_response.parse.return_value = resp
        raw_response.headers = {}

        mock_client = AsyncMock()
        mock_client.chat.completions.with_raw_response.create = AsyncMock(return_value=raw_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_aoai.return_value = mock_client

        msg = CoreMessage(role=Role.USER, content="hi")
        opts = ModelOptions(retries=0, cache_ttl=None)
        result = await _generate_impl([msg], model="gpt-5.4-mini", model_options=opts)
        assert result.content == "OK"

    @patch("ai_pipeline_core._llm_core.client.settings")
    @patch("ai_pipeline_core._llm_core.client.AsyncOpenAI")
    @patch("ai_pipeline_core._llm_core.client.detect_output_degeneration", return_value="repetition detected")
    async def test_generate_degeneration_detected(self, mock_degen, mock_aoai, mock_settings):
        from ai_pipeline_core._llm_core.client import _generate_impl
        from ai_pipeline_core._llm_core.types import ModelOptions

        mock_settings.openai_api_key = "key"
        mock_settings.openai_base_url = "http://localhost:4000"

        resp = _make_response(content="repeated text " * 100, prompt=10, completion=200)
        raw_response = MagicMock()
        raw_response.parse.return_value = resp
        raw_response.headers = {}

        mock_client = AsyncMock()
        mock_client.chat.completions.with_raw_response.create = AsyncMock(return_value=raw_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_aoai.return_value = mock_client

        msg = CoreMessage(role=Role.USER, content="hi")
        opts = ModelOptions(retries=0, retry_delay_seconds=0, cache_ttl=None)
        with pytest.raises(LLMError):
            await _generate_impl([msg], model="test-model", model_options=opts)

    @patch("ai_pipeline_core._llm_core.client.settings")
    @patch("ai_pipeline_core._llm_core.client.AsyncOpenAI")
    @patch("ai_pipeline_core._llm_core.client.detect_output_degeneration", return_value=None)
    async def test_generate_with_cache_ttl_and_context(self, mock_degen, mock_aoai, mock_settings):
        from ai_pipeline_core._llm_core.client import _generate_impl
        from ai_pipeline_core._llm_core.types import ModelOptions

        mock_settings.openai_api_key = "key"
        mock_settings.openai_base_url = "http://localhost:4000"

        resp = _make_response(content="OK", prompt=10, completion=5)
        raw_response = MagicMock()
        raw_response.parse.return_value = resp
        raw_response.headers = {}

        mock_client = AsyncMock()
        mock_client.chat.completions.with_raw_response.create = AsyncMock(return_value=raw_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_aoai.return_value = mock_client

        ctx_msg = CoreMessage(role=Role.SYSTEM, content="You are a system." * 1000)
        user_msg = CoreMessage(role=Role.USER, content="hi")
        opts = ModelOptions(retries=0, cache_ttl="300s")
        result = await _generate_impl([ctx_msg, user_msg], model="test-model", model_options=opts, context_count=1)
        assert result.content == "OK"

    @patch("ai_pipeline_core._llm_core.client.settings")
    @patch("ai_pipeline_core._llm_core.client.AsyncOpenAI")
    @patch("ai_pipeline_core._llm_core.client.detect_output_degeneration", return_value=None)
    async def test_generate_retries_when_provider_returns_tool_calls_without_tools(self, mock_degen, mock_aoai, mock_settings):
        from ai_pipeline_core._llm_core.client import _generate_impl
        from ai_pipeline_core._llm_core.types import ModelOptions

        mock_settings.openai_api_key = "key"
        mock_settings.openai_base_url = "http://localhost:4000"

        bad_resp = _make_response(content="")
        bad_resp.choices[0].message.tool_calls = [
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="unexpected_tool", arguments="{}"),
            )
        ]
        bad_raw = MagicMock()
        bad_raw.parse.return_value = bad_resp
        bad_raw.headers = {}

        good_resp = _make_response(content="Recovered answer", prompt=10, completion=5)
        good_raw = MagicMock()
        good_raw.parse.return_value = good_resp
        good_raw.headers = {}

        mock_client = AsyncMock()
        mock_client.chat.completions.with_raw_response.create = AsyncMock(side_effect=[bad_raw, good_raw])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_aoai.return_value = mock_client

        msg = CoreMessage(role=Role.USER, content="hi")
        opts = ModelOptions(retries=1, retry_delay_seconds=0, cache_ttl=None)
        result = await _generate_impl([msg], model="test-model", model_options=opts)

        assert result.content == "Recovered answer"
        assert mock_client.chat.completions.with_raw_response.create.call_count == 2
