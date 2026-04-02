"""Regression tests for verified database metrics bugs."""

from __future__ import annotations


from ai_pipeline_core.database._types import TOKENS_INPUT_KEY, get_token_count


def test_get_token_count_ignores_boolean_values() -> None:
    assert get_token_count({TOKENS_INPUT_KEY: True}, TOKENS_INPUT_KEY) == 0
