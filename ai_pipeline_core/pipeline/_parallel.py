"""Parallel execution primitives for pipeline tasks."""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeIs, cast

from ai_pipeline_core.documents import Document

__all__ = [
    "TaskBatch",
    "TaskHandle",
    "as_task_completed",
    "collect_tasks",
    "run_tasks_until",
]


@dataclass(frozen=True, slots=True, eq=False)
class TaskHandle[T]:
    """Handle for an executing pipeline task."""

    _task: asyncio.Task[T]
    task_class: type[Any] | None
    input_arguments: Mapping[str, Any]

    def __await__(self):
        return self._task.__await__()

    @property
    def done(self) -> bool:
        """Whether the underlying task has finished."""
        return self._task.done()

    async def result(self) -> T:
        """Await the underlying task result."""
        return await self._task

    def cancel(self) -> None:
        """Cancel the underlying task."""
        self._task.cancel()


TaskAwaitable = TaskHandle[tuple[Document[Any], ...]] | Awaitable[tuple[Document[Any], ...]]
TaskAwaitableGroup = TaskAwaitable | Sequence[TaskAwaitable]


@dataclass(frozen=True, slots=True)
class TaskBatch:
    """Collected task results and handles that did not complete successfully."""

    completed: list[tuple[Document[Any], ...]]
    incomplete: list[TaskHandle[tuple[Document[Any], ...]]]


def _empty_arguments() -> Mapping[str, Any]:
    return MappingProxyType({})


def _to_handle(awaitable: TaskAwaitable) -> TaskHandle[tuple[Document[Any], ...]]:
    if isinstance(awaitable, TaskHandle):
        return awaitable
    task = asyncio.ensure_future(awaitable)
    return TaskHandle(
        _task=task,
        task_class=None,
        input_arguments=_empty_arguments(),
    )


def _is_handle_sequence(value: TaskAwaitableGroup) -> TypeIs[Sequence[TaskAwaitable]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, TaskHandle))


def _normalize_handles(handles: tuple[TaskAwaitableGroup, ...]) -> list[TaskHandle[tuple[Document[Any], ...]]]:
    if not handles:
        return []
    if len(handles) == 1 and _is_handle_sequence(handles[0]):
        return [_to_handle(handle) for handle in handles[0]]

    flattened: list[TaskAwaitable] = []
    for handle in handles:
        if _is_handle_sequence(handle):
            flattened.extend(handle)
            continue
        flattened.append(cast("TaskAwaitable", handle))
    return [_to_handle(handle) for handle in flattened]


async def collect_tasks(
    *handles: TaskAwaitableGroup,
    deadline_seconds: float | None = None,
) -> TaskBatch:
    """Await task handles with an optional deadline and split completed/incomplete."""
    ordered_handles = _normalize_handles(handles)
    if not ordered_handles:
        return TaskBatch(completed=[], incomplete=[])

    completed: list[tuple[Document[Any], ...]] = []
    incomplete: list[TaskHandle[tuple[Document[Any], ...]]] = []
    by_task: dict[asyncio.Task[tuple[Document[Any], ...]], TaskHandle[tuple[Document[Any], ...]]] = {handle._task: handle for handle in ordered_handles}
    pending: set[asyncio.Task[tuple[Document[Any], ...]]] = set(by_task.keys())
    deadline_at = (time.monotonic() + deadline_seconds) if deadline_seconds is not None else None

    while pending:
        timeout: float | None = None
        if deadline_at is not None:
            timeout = max(0.0, deadline_at - time.monotonic())
            if timeout <= 0.0:
                break
        done, pending = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if not done:
            break
        for finished in done:
            handle = by_task[finished]
            outcome = (await asyncio.gather(handle.result(), return_exceptions=True))[0]
            if isinstance(outcome, BaseException):
                incomplete.append(handle)
                continue
            completed.append(outcome)

    incomplete.extend(by_task[still_pending] for still_pending in pending)
    return TaskBatch(completed=completed, incomplete=incomplete)


async def as_task_completed(*handles: TaskAwaitableGroup) -> AsyncIterator[TaskHandle[tuple[Document[Any], ...]]]:
    """Yield task handles in completion order."""
    ordered_handles = _normalize_handles(handles)
    if not ordered_handles:
        return

    by_task: dict[asyncio.Task[tuple[Document[Any], ...]], TaskHandle[tuple[Document[Any], ...]]] = {handle._task: handle for handle in ordered_handles}
    pending: set[asyncio.Task[tuple[Document[Any], ...]]] = set(by_task.keys())
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for finished in done:
            yield by_task[finished]


async def run_tasks_until(
    task_cls: type[Any],
    argument_groups: Sequence[tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    deadline_seconds: float | None = None,
) -> TaskBatch:
    """Launch ``task_cls.run(*args, **kwargs)`` for each argument group and collect the handles."""
    handles: list[TaskAwaitable] = []
    try:
        for args, kwargs in argument_groups:
            handles.append(task_cls.run(*args, **kwargs))
    except Exception:
        normalized_handles = _normalize_handles(tuple(handles))
        for handle in normalized_handles:
            handle.cancel()
        if normalized_handles:
            await asyncio.gather(*(handle.result() for handle in normalized_handles), return_exceptions=True)
        raise
    return await collect_tasks(handles, deadline_seconds=deadline_seconds)
