"""Integration tests for multi-turn reasoning content preservation.

Approach: Make a first call that triggers reasoning, then a second call
with a tiny follow-up ("Hi"). If reasoning is forwarded, the second call's
prompt_tokens will be significantly larger than call1's prompt_tokens +
call1's completion text tokens + a small overhead. If reasoning is stripped,
prompt_tokens will be close to just prompt + text completion.
"""

import random

import pytest

from ai_pipeline_core.llm import AIMessages, ModelOptions, generate
from ai_pipeline_core.settings import settings

HAS_API_KEYS = bool(settings.openai_api_key and settings.openai_base_url)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_API_KEYS, reason="OpenAI API keys not configured"),
]


@pytest.mark.asyncio
async def test_reasoning_content_forwarded_to_second_call() -> None:
    """Verify reasoning tokens are included in the second call's prompt.

    Strategy:
    - Call 1: Hard problem with reasoning_effort=high → generates reasoning
    - Call 2: Append response + "Hi" (1 token follow-up)
    - If call2.prompt_tokens > call1.prompt_tokens + call1.text_completion_tokens + 50,
      then reasoning was forwarded (those extra tokens are the reasoning).
    """
    model = "gemini-3-flash"
    options = ModelOptions(reasoning_effort="high", cache_ttl=None)

    cache_buster = random.randint(1000, 99999)
    prompt1 = (
        f"[ID:{cache_buster}] Solve this problem step by step:\n\n"
        "A ship has a rope ladder hanging over its side with rungs 12 inches apart. "
        "At low tide, 10 rungs are above the water. The tide rises at 6 inches per hour. "
        "After 3 hours, how many rungs will be above water?\n\n"
        "Think through this carefully - there's a common mistake people make."
    )

    messages = AIMessages([prompt1])
    response1 = await generate(model, messages=messages, options=options)

    assert response1.usage is not None, "Response1 missing usage data"
    assert response1.usage.prompt_tokens is not None, "Response1 missing prompt_tokens"
    assert response1.usage.completion_tokens is not None, "Response1 missing completion_tokens"

    call1_prompt_tokens = response1.usage.prompt_tokens
    call1_completion_tokens = response1.usage.completion_tokens
    call1_reasoning_tokens = 0
    if response1.usage.completion_tokens_details:
        call1_reasoning_tokens = response1.usage.completion_tokens_details.reasoning_tokens or 0
    call1_text_tokens = call1_completion_tokens - call1_reasoning_tokens

    print("\n--- Call 1 ---")
    print(f"  prompt_tokens: {call1_prompt_tokens}")
    print(f"  completion_tokens: {call1_completion_tokens}")
    print(f"  reasoning_tokens: {call1_reasoning_tokens}")
    print(f"  text_tokens (completion - reasoning): {call1_text_tokens}")
    print(f"  total_tokens: {response1.usage.total_tokens}")

    # Call 2: tiny follow-up
    messages.append(response1)
    messages.append("Hi")

    response2 = await generate(model, messages=messages, options=options)

    assert response2.usage is not None, "Response2 missing usage data"
    assert response2.usage.prompt_tokens is not None, "Response2 missing prompt_tokens"

    call2_prompt_tokens = response2.usage.prompt_tokens

    print("\n--- Call 2 ---")
    print(f"  prompt_tokens: {call2_prompt_tokens}")

    baseline = call1_prompt_tokens + call1_text_tokens + 50

    print("\n--- Verification ---")
    print(f"  baseline (prompt + text + 50): {baseline}")
    print(f"  call2_prompt_tokens: {call2_prompt_tokens}")
    print(f"  difference: {call2_prompt_tokens - baseline}")

    assert call2_prompt_tokens > baseline, (
        f"BUG: Reasoning was NOT forwarded to second call!\n\n"
        f"Call 1: prompt_tokens={call1_prompt_tokens}, text_tokens={call1_text_tokens}\n"
        f"Call 2: prompt_tokens={call2_prompt_tokens}\n"
        f"Expected call2 prompt_tokens > {baseline} (prompt + text + 50 margin)\n"
        f"Reasoning was stripped from the conversation."
    )
