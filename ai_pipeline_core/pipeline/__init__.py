"""Pipeline framework primitives."""

from ai_pipeline_core.pipeline._execution_context import get_run_id, pipeline_test_context
from ai_pipeline_core.pipeline._flow import PipelineFlow
from ai_pipeline_core.pipeline._parallel import TaskBatch, TaskHandle, as_task_completed, collect_tasks, run_tasks_until
from ai_pipeline_core.pipeline._task import PipelineTask
from ai_pipeline_core.pipeline.gather import safe_gather, safe_gather_indexed
from ai_pipeline_core.pipeline.limits import LimitKind, PipelineLimit, pipeline_concurrency
from ai_pipeline_core.pipeline.options import FlowOptions

__all__ = [
    "FlowOptions",
    "LimitKind",
    "PipelineFlow",
    "PipelineLimit",
    "PipelineTask",
    "TaskBatch",
    "TaskHandle",
    "as_task_completed",
    "collect_tasks",
    "get_run_id",
    "pipeline_concurrency",
    "pipeline_test_context",
    "run_tasks_until",
    "safe_gather",
    "safe_gather_indexed",
]
