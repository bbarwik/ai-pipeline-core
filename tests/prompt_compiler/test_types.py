"""Tests for prompt_compiler.types (Phase type)."""

import ai_pipeline_core as core
import ai_pipeline_core.prompt_compiler as pc
from ai_pipeline_core.prompt_compiler import Phase


def test_phase_is_str_subclass() -> None:
    phase = Phase("review")
    assert phase == "review"
    assert isinstance(phase, str)
    assert isinstance(phase, Phase)


def test_prompt_compiler_public_api_exports() -> None:
    expected = [
        "Guide",
        "OutputRule",
        "OutputT",
        "Phase",
        "PromptSpec",
        "Role",
        "Rule",
        "extract_result",
        "render_preview",
        "render_text",
        "send_spec",
    ]
    assert pc.__all__ == expected
    for name in expected:
        assert hasattr(pc, name)


def test_top_level_exports_include_prompt_compiler_symbols() -> None:
    names = ["Guide", "OutputT", "OutputRule", "Phase", "PromptSpec", "Role", "Rule", "extract_result", "render_preview", "render_text", "send_spec"]
    for name in names:
        assert name in core.__all__
        assert getattr(core, name) is getattr(pc, name)
