"""Regression tests for verified pipeline parallel execution bugs."""

from __future__ import annotations

import asyncio

import pytest

from ai_pipeline_core.documents import Document
from ai_pipeline_core.pipeline import PipelineTask, pipeline_test_context, run_tasks_until


class _ParallelInputDoc(Document):
    """Input document for parallel regression tests."""


class _ParallelOutputDoc(Document):
    """Output document for parallel regression tests."""


class _DuplicateTask(PipelineTask):
    """Task that intentionally returns the same document twice."""

    @classmethod
    async def run(
        cls,
    ) -> tuple[_ParallelOutputDoc, ...]:
        doc = _ParallelOutputDoc.create_root(name="same.txt", content="same", reason="test")
        return (doc, doc)


@pytest.mark.asyncio
async def test_run_tasks_until_cancels_started_tasks_on_bind_failure() -> None:
    class _SlowTask(PipelineTask):
        """Task that blocks until cancellation."""

        @classmethod
        async def run(
            cls,
            documents: tuple[_ParallelInputDoc, ...],
        ) -> tuple[_ParallelOutputDoc, ...]:
            await asyncio.sleep(10)
            return ()

    doc = _ParallelInputDoc.create_root(name="in.txt", content="x", reason="test")

    with pipeline_test_context():
        with pytest.raises(TypeError):
            await run_tasks_until(
                _SlowTask,
                [
                    (((doc,),), {}),
                    (((doc,),), {}),
                    (("not-a-document",), {}),
                ],
            )

    await asyncio.sleep(0.05)
    lingering_tasks = [task for task in asyncio.all_tasks() if "_SlowTask.run" in repr(task.get_coro()) and not task.done()]
    assert lingering_tasks == []


@pytest.mark.asyncio
async def test_task_duplicate_outputs_keep_original_tuple_shape() -> None:
    with pipeline_test_context():
        result = await _DuplicateTask.run()

    assert len(result) == 2
    assert result[0].sha256 == result[1].sha256
