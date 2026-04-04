"""Tests for _list_wrappers.py — wrapper model creation, key derivation, unwrapping."""

import json

import pytest
from pydantic import BaseModel

from typing import Any

from ai_pipeline_core._llm_core.model_response import ModelResponse
from ai_pipeline_core._llm_core.types import TokenUsage
from ai_pipeline_core.llm._list_wrappers import (
    _derive_list_field_name,
    get_list_field_name,
    get_list_item_type,
    get_or_create_wrapper,
    is_list_output_type,
    unwrap_list_response,
)


class TaskGroupModel(BaseModel):
    name: str
    priority: int


class Finding(BaseModel):
    text: str


class AnalysisEntry(BaseModel):
    score: float


class Category(BaseModel):
    label: str


# --- is_list_output_type ---


def test_is_list_output_type_accepts_list_basemodel() -> None:
    assert is_list_output_type(list[TaskGroupModel]) is True


def test_is_list_output_type_rejects_plain_basemodel() -> None:
    assert is_list_output_type(TaskGroupModel) is False


def test_is_list_output_type_rejects_list_str() -> None:
    assert is_list_output_type(list[str]) is False


def test_is_list_output_type_rejects_list_int() -> None:
    assert is_list_output_type(list[int]) is False


def test_is_list_output_type_rejects_none() -> None:
    assert is_list_output_type(None) is False


def test_is_list_output_type_rejects_bare_list() -> None:
    assert is_list_output_type(list) is False


def test_is_list_output_type_rejects_str() -> None:
    assert is_list_output_type(str) is False


# --- get_list_item_type ---


def test_get_list_item_type_extracts_inner() -> None:
    assert get_list_item_type(list[TaskGroupModel]) is TaskGroupModel


# --- _derive_list_field_name ---


def test_derive_field_name_strips_model_suffix() -> None:
    assert _derive_list_field_name(TaskGroupModel) == "task_groups"


def test_derive_field_name_no_suffix() -> None:
    assert _derive_list_field_name(Finding) == "findings"


def test_derive_field_name_entry_suffix() -> None:
    # AnalysisEntry -> analysis_entries (y->ies not applicable, just +s after stripping)
    name = _derive_list_field_name(AnalysisEntry)
    assert name == "analysis_entries"


def test_derive_field_name_y_pluralization() -> None:
    assert _derive_list_field_name(Category) == "categories"


# --- get_or_create_wrapper ---


def test_wrapper_has_correct_field_name() -> None:
    wrapper = get_or_create_wrapper(TaskGroupModel)
    assert "task_groups" in wrapper.model_fields


def test_wrapper_is_cached() -> None:
    w1 = get_or_create_wrapper(TaskGroupModel)
    w2 = get_or_create_wrapper(TaskGroupModel)
    assert w1 is w2


def test_wrapper_is_frozen() -> None:
    wrapper = get_or_create_wrapper(TaskGroupModel)
    instance = wrapper(task_groups=(TaskGroupModel(name="a", priority=1),))
    with pytest.raises(Exception):
        instance.task_groups = ()  # type: ignore[misc]


def test_wrapper_validates_items() -> None:
    wrapper = get_or_create_wrapper(TaskGroupModel)
    items = (TaskGroupModel(name="a", priority=1), TaskGroupModel(name="b", priority=2))
    instance = wrapper(task_groups=items)
    assert len(instance.task_groups) == 2


def test_wrapper_accepts_empty_tuple() -> None:
    wrapper = get_or_create_wrapper(TaskGroupModel)
    instance = wrapper(task_groups=())
    assert instance.task_groups == ()


# --- get_list_field_name ---


def test_get_list_field_name_returns_first_field() -> None:
    wrapper = get_or_create_wrapper(Finding)
    assert get_list_field_name(wrapper) == "findings"


# --- unwrap_list_response ---


def _make_wrapper_response(item_type: type[BaseModel], items: list[BaseModel]) -> ModelResponse:
    wrapper = get_or_create_wrapper(item_type)
    field_name = get_list_field_name(wrapper)
    parsed_wrapper = wrapper(**{field_name: tuple(items)})
    wrapper_json = json.dumps({field_name: [item.model_dump(mode="json") for item in items]})
    return ModelResponse[Any](
        content=wrapper_json,
        parsed=parsed_wrapper,
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="test",
    )


def test_unwrap_list_response_returns_list() -> None:
    items = [TaskGroupModel(name="a", priority=1), TaskGroupModel(name="b", priority=2)]
    response = _make_wrapper_response(TaskGroupModel, items)
    unwrapped = unwrap_list_response(response, TaskGroupModel)
    assert isinstance(unwrapped.parsed, list)
    assert len(unwrapped.parsed) == 2
    assert unwrapped.parsed[0].name == "a"
    assert unwrapped.parsed[1].priority == 2


def test_unwrap_list_response_rewrites_content_to_array_json() -> None:
    items = [Finding(text="found it")]
    response = _make_wrapper_response(Finding, items)
    unwrapped = unwrap_list_response(response, Finding)
    parsed_content = json.loads(unwrapped.content)
    assert isinstance(parsed_content, list)
    assert parsed_content[0]["text"] == "found it"


def test_unwrap_list_response_empty_list() -> None:
    response = _make_wrapper_response(Finding, [])
    unwrapped = unwrap_list_response(response, Finding)
    assert unwrapped.parsed == []
    assert json.loads(unwrapped.content) == []
