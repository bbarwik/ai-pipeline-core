"""Tests for deployment resume/skip logic with DocumentStore.

Proves that the resume/skip logic correctly handles:
- Same inputs + same options → skip (cache hit)
- Different inputs → no skip (cache miss)
- Different options → no skip (cache miss)
"""

# pyright: reportPrivateUsage=false

import pytest

from ai_pipeline_core import DeploymentResult, Document, FlowOptions, PipelineDeployment, pipeline_flow
from ai_pipeline_core.deployment import DeploymentContext
from ai_pipeline_core.document_store.memory import MemoryDocumentStore
from ai_pipeline_core.document_store.protocol import set_document_store
from pydantic import Field


# --- Module-level test infrastructure ---


class ResumeInputDoc(Document):
    """Input document for resume tests."""


class ResumeOutputDoc(Document):
    """Output document for resume tests."""


class ResumeOptions(FlowOptions):
    """Options with a mode field to test option-change detection."""

    mode: str = Field(default="default")


# Mutable list tracking flow invocations across runs
_flow_executions: list[str] = []


@pipeline_flow()
async def resume_flow(
    project_name: str,
    documents: list[ResumeInputDoc],
    flow_options: ResumeOptions,
) -> list[ResumeOutputDoc]:
    """Flow that tracks each execution and produces input-dependent output."""
    _flow_executions.append("called")
    content = "|".join(sorted(doc.name for doc in documents))
    return [ResumeOutputDoc(name="output.txt", content=f"result:{content}:{flow_options.mode}".encode())]


class ResumeResult(DeploymentResult):
    """Minimal result for resume tests."""


class ResumeDeployment(PipelineDeployment[ResumeOptions, ResumeResult]):
    """Single-flow deployment for resume tests."""

    flows = [resume_flow]  # type: ignore[reportAssignmentType]

    @staticmethod
    def build_result(project_name: str, documents: list[Document], options: ResumeOptions) -> ResumeResult:
        return ResumeResult(success=True)


# --- Tests ---


class TestResumeLogic:
    """Verify that deployment skip/resume logic accounts for inputs and options."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        """Clear execution tracker and document store before each test."""
        _flow_executions.clear()
        set_document_store(MemoryDocumentStore())
        yield
        set_document_store(None)

    async def _run(
        self,
        project_name: str,
        documents: list[Document],
        options: ResumeOptions | None = None,
    ) -> ResumeResult:
        """Run pipeline through run() with shared store."""
        return await ResumeDeployment().run(
            project_name=project_name,
            documents=documents,
            options=options or ResumeOptions(),
            context=DeploymentContext(),
        )

    @pytest.mark.asyncio
    async def test_skip_when_same_inputs_and_options(self):
        """Re-running with identical project, inputs, and options → flow skipped."""
        docs = [ResumeInputDoc(name="a.txt", content=b"aaa")]
        await self._run("proj", docs)
        assert len(_flow_executions) == 1

        await self._run("proj", docs)
        assert len(_flow_executions) == 1  # skipped

    @pytest.mark.asyncio
    async def test_no_skip_when_inputs_change(self):
        """Re-running with different input documents → flow must re-execute."""
        docs_a = [ResumeInputDoc(name="a.txt", content=b"aaa")]
        docs_b = [ResumeInputDoc(name="b.txt", content=b"bbb")]
        await self._run("proj", docs_a)
        assert len(_flow_executions) == 1

        await self._run("proj", docs_b)
        assert len(_flow_executions) == 2  # must re-execute

    @pytest.mark.asyncio
    async def test_no_skip_when_options_change(self):
        """Re-running with different options → flow must re-execute."""
        docs = [ResumeInputDoc(name="a.txt", content=b"aaa")]
        await self._run("proj", docs, ResumeOptions(mode="fast"))
        assert len(_flow_executions) == 1

        await self._run("proj", docs, ResumeOptions(mode="thorough"))
        assert len(_flow_executions) == 2  # must re-execute

    @pytest.mark.asyncio
    async def test_skip_still_works_with_identical_run(self):
        """Regression guard: identical inputs + options → flow still skipped after fix."""
        docs = [ResumeInputDoc(name="a.txt", content=b"aaa")]
        opts = ResumeOptions(mode="exact")
        await self._run("proj", docs, opts)
        assert len(_flow_executions) == 1

        await self._run("proj", docs, opts)
        assert len(_flow_executions) == 1  # still skipped
