"""Tests for handle-based parallel execution primitives."""

import asyncio
from typing import Any

import pytest

from ai_pipeline_core.documents import Document
from ai_pipeline_core.pipeline import PipelineTask, TaskBatch, TaskHandle, as_task_completed, collect_tasks, run_tasks_until
from ai_pipeline_core.pipeline._execution_context import FlowFrame, pipeline_test_context, reset_execution_context, set_execution_context


class _PDoc(Document):
    """Parallel test document."""


class _FastTask(PipelineTask):
    @classmethod
    async def run(cls, documents: list[_PDoc]) -> list[_PDoc]:
        return [_PDoc(name="fast.txt", content=b"fast")]


class _SlowTask(PipelineTask):
    @classmethod
    async def run(cls, documents: list[_PDoc]) -> list[_PDoc]:
        await asyncio.sleep(60)
        return [_PDoc(name="slow.txt", content=b"slow")]


class _FailTask(PipelineTask):
    @classmethod
    async def run(cls, documents: list[_PDoc]) -> list[_PDoc]:
        raise ValueError("run fail")


def _make_flow_frame() -> FlowFrame:
    return FlowFrame(
        name="parallel-flow",
        flow_class_name="PFlow",
        step=1,
        total_steps=1,
        flow_minutes=(1.0,),
        completed_minutes=0.0,
        flow_params={},
    )


def _make_doc(content: str = "test") -> _PDoc:
    return _PDoc(name="input.txt", content=content.encode())


@pytest.mark.asyncio
async def test_run_returns_task_handle() -> None:
    with pipeline_test_context() as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        try:
            handle: Any = _FastTask.run([_make_doc()])
            assert isinstance(handle, TaskHandle)
            result = await handle.result()
            assert len(result) == 1
        finally:
            reset_execution_context(token)


@pytest.mark.asyncio
async def test_task_handle_done_property() -> None:
    with pipeline_test_context() as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        try:
            handle = _FastTask.run([_make_doc()])
            await handle.result()
            assert handle.done is True
        finally:
            reset_execution_context(token)


@pytest.mark.asyncio
async def test_task_handle_cancel() -> None:
    with pipeline_test_context() as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        try:
            handle = _SlowTask.run([_make_doc()])
            handle.cancel()
            with pytest.raises((asyncio.CancelledError, Exception)):
                await handle.result()
        finally:
            reset_execution_context(token)


@pytest.mark.asyncio
async def test_collect_tasks_all_complete() -> None:
    with pipeline_test_context() as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        try:
            h1 = _FastTask.run([_make_doc("a")])
            h2 = _FastTask.run([_make_doc("b")])
            batch = await collect_tasks(h1, h2)
        finally:
            reset_execution_context(token)

    assert isinstance(batch, TaskBatch)
    assert len(batch.completed) == 2
    assert batch.incomplete == []


@pytest.mark.asyncio
async def test_collect_tasks_with_deadline_splits() -> None:
    with pipeline_test_context() as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        slow: Any = None
        try:
            fast = _FastTask.run([_make_doc()])
            slow = _SlowTask.run([_make_doc()])
            batch = await collect_tasks(fast, slow, deadline_seconds=0.5)
        finally:
            if slow is not None:
                slow.cancel()
            reset_execution_context(token)

    assert len(batch.completed) == 1
    assert len(batch.incomplete) == 1


@pytest.mark.asyncio
async def test_collect_tasks_failed_goes_to_incomplete() -> None:
    with pipeline_test_context() as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        try:
            good = _FastTask.run([_make_doc()])
            bad = _FailTask.run([_make_doc()])
            batch = await collect_tasks(good, bad)
        finally:
            reset_execution_context(token)

    assert len(batch.completed) == 1
    assert len(batch.incomplete) == 1


@pytest.mark.asyncio
async def test_collect_tasks_accepts_list() -> None:
    with pipeline_test_context() as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        try:
            handles = [_FastTask.run([_make_doc("x")]) for _ in range(3)]
            batch = await collect_tasks(handles)
        finally:
            reset_execution_context(token)

    assert len(batch.completed) == 3
    assert batch.incomplete == []


@pytest.mark.asyncio
async def test_as_task_completed_yields_handles() -> None:
    with pipeline_test_context() as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        try:
            h1 = _FastTask.run([_make_doc("1")])
            h2 = _FastTask.run([_make_doc("2")])
            yielded = [handle async for handle in as_task_completed(h1, h2)]
        finally:
            reset_execution_context(token)

    assert len(yielded) == 2
    assert all(isinstance(handle, TaskHandle) for handle in yielded)


@pytest.mark.asyncio
async def test_run_tasks_until_dispatches_and_collects() -> None:
    with pipeline_test_context() as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        try:
            groups = [
                (([_make_doc("a")],), {}),
                (([_make_doc("b")],), {}),
                (([_make_doc("c")],), {}),
            ]
            batch = await run_tasks_until(_FastTask, groups)
        finally:
            reset_execution_context(token)

    assert len(batch.completed) == 3
    assert batch.incomplete == []
