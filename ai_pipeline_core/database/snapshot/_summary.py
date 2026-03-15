"""Render span execution summaries and cost reports as Markdown strings."""

from uuid import UUID

from ai_pipeline_core.database._protocol import DatabaseReader
from ai_pipeline_core.database.snapshot._spans import generate_costs_from_tree, generate_summary_from_tree

__all__ = [
    "generate_costs",
    "generate_summary",
]


async def generate_summary(reader: DatabaseReader, deployment_id: UUID) -> str:
    """Generate summary.md content from a deployment span tree."""
    tree = await reader.get_deployment_tree(deployment_id)
    return generate_summary_from_tree(tree, deployment_id)


async def generate_costs(reader: DatabaseReader, deployment_id: UUID) -> str:
    """Generate costs.md content from deployment spans."""
    tree = await reader.get_deployment_tree(deployment_id)
    return generate_costs_from_tree(tree, deployment_id)
