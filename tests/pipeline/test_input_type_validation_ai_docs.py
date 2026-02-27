"""AI-doc example for @pipeline_task input type validation."""

from enum import Enum
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import BaseModel

from ai_pipeline_core import Document, FlowOptions, pipeline_task


class _InputDoc(Document):
    """Test input document."""


class _OutputDoc(Document):
    """Test output document."""


@pytest.mark.ai_docs
def test_pipeline_task_allowed_input_types() -> None:
    """@pipeline_task validates input types at decoration time."""

    class Priority(Enum):
        HIGH = "high"

    class FrozenConfig(BaseModel):
        model_config = {"frozen": True}
        value: str = "x"

    @pipeline_task
    async def accepted(
        text: str,
        count: int,
        ratio: float,
        flag: bool,
        uid: UUID,
        file_path: Path,
        priority: Priority,
        doc: _InputDoc,
        config: FrozenConfig,
        options: FlowOptions,
        items: list[str],
        pair: tuple[str, int],
        mapping: dict[str, int],
        optional: str | None = None,
    ) -> _OutputDoc:
        return _OutputDoc(name="out.txt", content=b"ok")

    assert callable(accepted)
