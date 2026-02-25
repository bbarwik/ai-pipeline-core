"""Regression tests for resume document loading bugs.

These tests are written TDD-style: they assert the CORRECT behavior.
Tests marked xfail prove the bug exists (they fail until the fix is applied).
Once fixed, remove the xfail markers — the tests become regression guards.

Bug 1: Resume loads all documents by type instead of using FlowCompletion.output_sha256s.
Bug 2: FlowCompletion.output_sha256s recording is contaminated by preceding flows' documents.
Bug 3: chain_context output_document_refs includes documents from all flows, not just the last.
Bug 4: load_by_sha256s forces single-type construction, losing subclass identity.
"""

# pyright: reportPrivateUsage=false

import pytest

from ai_pipeline_core import DeploymentResult, Document, FlowOptions, PipelineDeployment, pipeline_flow, pipeline_task
from ai_pipeline_core.deployment import DeploymentContext
from ai_pipeline_core.deployment.base import _compute_run_scope
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.document_store._protocol import set_document_store
from ai_pipeline_core.documents.types import DocumentSha256


# ---------------------------------------------------------------------------
# Document types
# ---------------------------------------------------------------------------


class InputDoc(Document):
    """Pipeline entry document."""


class PlanDoc(Document):
    """Plan document — produced by flow 1, consumed by flow 2."""


class AnalysisDoc(Document):
    """Analysis result — produced by flow 2."""


class IntermediateDoc(Document):
    """Intermediate document to bridge flows without type overlap."""


class ReportDoc(Document):
    """Report document shared across flows as output type."""


# ---------------------------------------------------------------------------
# Common result type
# ---------------------------------------------------------------------------


class SimpleResult(DeploymentResult):
    """Minimal deployment result for tests."""


# ---------------------------------------------------------------------------
# Bug 1: Zombie docs from crashed flow pollute retry inputs on resume
#
# Flow 1 → PlanDoc (plan.json)
# Flow 2 → internally creates PlanDoc (progress.md) via @pipeline_task, then crashes
# Resume: flow 1 skipped, flow 2 re-runs
# BUG: flow 2 receives plan.json + zombie progress.md from the crashed run
# CORRECT: flow 2 receives only plan.json (flow 1's declared output)
# ---------------------------------------------------------------------------

_zombie_flow2_crash_flag: list[bool] = [False]
_zombie_flow2_received: list[Document] = []


@pipeline_flow()
async def zombie_flow1(
    run_id: str,
    documents: list[InputDoc],
    flow_options: FlowOptions,
) -> list[PlanDoc]:
    """Flow 1: produces a single official PlanDoc."""
    return [PlanDoc.create(name="plan.json", content='{"topic": "AI"}', derived_from=(documents[0].sha256,))]


@pipeline_task
async def save_progress(plans: list[PlanDoc]) -> PlanDoc:
    """Internal task: creates a progress PlanDoc (persisted by @pipeline_task)."""
    return PlanDoc.create(name="progress.md", content="# Progress notes", derived_from=(plans[0].sha256,))


@pipeline_flow()
async def zombie_flow2(
    run_id: str,
    documents: list[PlanDoc],
    flow_options: FlowOptions,
) -> list[AnalysisDoc]:
    """Flow 2: saves internal progress doc, then crashes (if flag set)."""
    _zombie_flow2_received.extend(documents)
    await save_progress(documents)
    if _zombie_flow2_crash_flag[0]:
        raise RuntimeError("Simulated crash after saving progress")
    return [AnalysisDoc.create(name="analysis.json", content='{"done": true}', derived_from=(documents[0].sha256,))]


class ZombieBugDeployment(PipelineDeployment[FlowOptions, SimpleResult]):
    """Two-flow pipeline for zombie document bug reproduction."""

    flows = [zombie_flow1, zombie_flow2]  # type: ignore[reportAssignmentType]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> SimpleResult:
        return SimpleResult(success=True)


class TestBug1ResumeZombieDocuments:
    """Bug 1: On resume after a crash, the retried flow receives zombie documents
    from its own previous crashed execution.

    Root cause: base.py:538 uses store.load(run_scope, input_types) which returns
    ALL documents of matching types in the store, including zombies persisted by
    @pipeline_task before the crash. Should use FlowCompletion.output_sha256s instead.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        _zombie_flow2_crash_flag[0] = False
        _zombie_flow2_received.clear()
        set_document_store(MemoryDocumentStore())
        yield
        set_document_store(None)

    @pytest.mark.asyncio
    async def test_resume_excludes_zombie_docs_from_crashed_flow(self):
        """On resume, flow 2 should receive only flow 1's outputs (plan.json),
        not zombie progress.md from the crashed run.
        """
        input_doc = InputDoc.create_root(name="input.txt", content="test", reason="test")
        deployment = ZombieBugDeployment()
        ctx = DeploymentContext()
        opts = FlowOptions()
        store = MemoryDocumentStore()
        set_document_store(store)

        # Run 1: flow 1 completes, flow 2 saves progress.md then crashes
        _zombie_flow2_crash_flag[0] = True
        _zombie_flow2_received.clear()
        with pytest.raises(RuntimeError, match="Simulated crash after saving progress"):
            await deployment.run("proj", [input_doc], opts, ctx)

        run_scope = _compute_run_scope("proj", [input_doc], opts)

        # Verify preconditions: flow 1 completed, flow 2 did not
        assert await store.get_flow_completion(run_scope, "zombie_flow1") is not None
        assert await store.get_flow_completion(run_scope, "zombie_flow2") is None

        # Store has 2 PlanDoc: plan.json (flow 1 output) + progress.md (zombie)
        all_plans = await store.load(run_scope, [PlanDoc])
        assert sorted(d.name for d in all_plans) == ["plan.json", "progress.md"]

        # Run 2: resume — flow 1 skipped, flow 2 re-executes
        _zombie_flow2_crash_flag[0] = False
        _zombie_flow2_received.clear()
        await deployment.run("proj", [input_doc], opts, ctx)

        # CORRECT: flow 2 receives only plan.json (flow 1's output)
        received_names = sorted(d.name for d in _zombie_flow2_received)
        assert received_names == ["plan.json"], (
            f"Flow 2 should receive only flow 1's output [plan.json], got {received_names}. Zombie progress.md from crashed run should NOT appear in inputs."
        )

    @pytest.mark.asyncio
    async def test_flow_completion_output_sha256s_is_populated(self):
        """FlowCompletion.output_sha256s is correctly populated — the data exists
        for the fix but is currently unused on resume.
        """
        input_doc = InputDoc.create_root(name="input.txt", content="test", reason="test")
        deployment = ZombieBugDeployment()
        ctx = DeploymentContext()
        opts = FlowOptions()
        store = MemoryDocumentStore()
        set_document_store(store)

        # Run 1: flow 1 completes, flow 2 crashes
        _zombie_flow2_crash_flag[0] = True
        with pytest.raises(RuntimeError, match="Simulated crash"):
            await deployment.run("proj", [input_doc], opts, ctx)

        run_scope = _compute_run_scope("proj", [input_doc], opts)
        completion1 = await store.get_flow_completion(run_scope, "zombie_flow1")
        assert completion1 is not None
        assert len(completion1.output_sha256s) > 0, "output_sha256s should be populated"


# ---------------------------------------------------------------------------
# Bug 2: output_sha256s recording is contaminated by preceding flows
#
# Flow 1: InputDoc → [IntermediateDoc, ReportDoc]
# Flow 2: IntermediateDoc → [ReportDoc]
# After flow 2, store.load(run_scope, [ReportDoc]) returns BOTH flows' ReportDoc
# BUG: flow 2's FlowCompletion.output_sha256s includes flow 1's ReportDoc
# CORRECT: flow 2's output_sha256s should have exactly 1 (its own ReportDoc)
# ---------------------------------------------------------------------------


@pipeline_flow()
async def bug2_flow1(
    run_id: str,
    documents: list[InputDoc],
    flow_options: FlowOptions,
) -> list[IntermediateDoc | ReportDoc]:
    """Flow 1: produces both IntermediateDoc and ReportDoc."""
    return [
        IntermediateDoc.create(name="intermediate.txt", content="bridge data", derived_from=(documents[0].sha256,)),
        ReportDoc.create(name="report_from_flow1.txt", content="flow1 report", derived_from=(documents[0].sha256,)),
    ]


@pipeline_flow()
async def bug2_flow2(
    run_id: str,
    documents: list[IntermediateDoc],
    flow_options: FlowOptions,
) -> list[ReportDoc]:
    """Flow 2: takes IntermediateDoc, outputs ReportDoc (same output type as flow 1)."""
    return [ReportDoc.create(name="report_from_flow2.txt", content="flow2 report", derived_from=(documents[0].sha256,))]


class Bug2Deployment(PipelineDeployment[FlowOptions, SimpleResult]):
    """Two-flow pipeline where both flows output ReportDoc."""

    flows = [bug2_flow1, bug2_flow2]  # type: ignore[reportAssignmentType]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> SimpleResult:
        return SimpleResult(success=True)


class TestBug2OutputSha256sContamination:
    """Bug 2: FlowCompletion.output_sha256s for flow 2 includes flow 1's documents
    when output types overlap.

    Root cause: base.py:561 uses store.load(run_scope, output_types) which returns
    all documents of matching types, not just the current flow's outputs. Should use
    the flow's actual return value instead.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        set_document_store(MemoryDocumentStore())
        yield
        set_document_store(None)

    @pytest.mark.asyncio
    async def test_flow2_output_sha256s_contains_only_own_outputs(self):
        """Flow 2 produced 1 ReportDoc → output_sha256s should have exactly 1 SHA256."""
        input_doc = InputDoc.create_root(name="input.txt", content="test", reason="test")
        deployment = Bug2Deployment()
        ctx = DeploymentContext()
        opts = FlowOptions()
        store = MemoryDocumentStore()
        set_document_store(store)

        await deployment.run("proj", [input_doc], opts, ctx)

        run_scope = _compute_run_scope("proj", [input_doc], opts)
        completion2 = await store.get_flow_completion(run_scope, "bug2_flow2")
        assert completion2 is not None

        assert len(completion2.output_sha256s) == 1, (
            f"Flow 2 produced 1 ReportDoc but output_sha256s has {len(completion2.output_sha256s)} entries (contaminated with flow 1's ReportDoc)"
        )

    @pytest.mark.asyncio
    async def test_flow2_output_sha256s_does_not_overlap_flow1(self):
        """Flow 2's output_sha256s should not contain any of flow 1's SHA256s."""
        input_doc = InputDoc.create_root(name="input.txt", content="test", reason="test")
        deployment = Bug2Deployment()
        ctx = DeploymentContext()
        opts = FlowOptions()
        store = MemoryDocumentStore()
        set_document_store(store)

        await deployment.run("proj", [input_doc], opts, ctx)

        run_scope = _compute_run_scope("proj", [input_doc], opts)
        completion1 = await store.get_flow_completion(run_scope, "bug2_flow1")
        completion2 = await store.get_flow_completion(run_scope, "bug2_flow2")
        assert completion1 is not None
        assert completion2 is not None

        contamination = set(completion1.output_sha256s) & set(completion2.output_sha256s)
        assert len(contamination) == 0, f"Flow 2's output_sha256s should not overlap with flow 1's. Contaminated SHA256s: {contamination}"


# ---------------------------------------------------------------------------
# Bug 3: chain_context output_document_refs contamination
#
# Same flows as Bug 2. chain_context is built from store.load(run_scope, last_output_types).
# BUG: includes ReportDoc from both flows
# CORRECT: includes only the last flow's ReportDoc
# ---------------------------------------------------------------------------

_bug3_published_events: list[object] = []


class _CapturingPublisher:
    """Captures published events for assertion."""

    async def publish_started(self, event: object) -> None:
        pass

    async def publish_progress(self, event: object) -> None:
        pass

    async def publish_completed(self, event: object) -> None:
        _bug3_published_events.append(event)

    async def publish_failed(self, event: object) -> None:
        pass

    async def publish_heartbeat(self, run_id: str) -> None:
        pass

    async def close(self) -> None:
        pass


class Bug3Deployment(PipelineDeployment[FlowOptions, SimpleResult]):
    """Same flow chain as Bug 2 — both flows output ReportDoc."""

    flows = [bug2_flow1, bug2_flow2]  # type: ignore[reportAssignmentType]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> SimpleResult:
        return SimpleResult(success=True)


class TestBug3ChainContextContamination:
    """Bug 3: chain_context.output_document_refs includes docs from all flows
    when the last flow's output type is shared with a preceding flow.

    Root cause: base.py:614 uses store.load(run_scope, last_output_types). Should use
    the last flow's actual return value instead.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        _bug3_published_events.clear()
        set_document_store(MemoryDocumentStore())
        yield
        set_document_store(None)

    @pytest.mark.asyncio
    async def test_chain_context_has_only_last_flow_outputs(self):
        """chain_context.output_document_refs should contain only the last flow's outputs."""
        input_doc = InputDoc.create_root(name="input.txt", content="test", reason="test")
        deployment = Bug3Deployment()
        ctx = DeploymentContext()
        opts = FlowOptions()
        publisher = _CapturingPublisher()

        await deployment.run("proj", [input_doc], opts, ctx, publisher=publisher)

        assert len(_bug3_published_events) == 1
        event = _bug3_published_events[0]
        chain_context = event.chain_context  # pyright: ignore[reportAttributeAccessIssue]
        output_refs = chain_context["output_document_refs"]

        # Flow 2 (last flow) produced exactly 1 ReportDoc
        assert len(output_refs) == 1, (
            f"Last flow produced 1 ReportDoc but chain_context has {len(output_refs)} refs (contaminated with preceding flow's ReportDoc)"
        )


# ---------------------------------------------------------------------------
# Bug 4: load_by_sha256s single-type construction loses subclass identity
#
# Not a runtime bug but an API gap that blocks a straightforward fix for Bug 1.
# The workaround is to call load_nodes_by_sha256s first (get class names),
# then load_by_sha256s per type. These tests document the API behavior.
# ---------------------------------------------------------------------------


class TypeADoc(Document):
    """Document type A for load_by_sha256s test."""


class TypeBDoc(Document):
    """Document type B for load_by_sha256s test."""


class TestBug4LoadBySha256sSingleType:
    """Bug 4: load_by_sha256s constructs all documents as a single type.

    MemoryDocumentStore happens to preserve original types via variant fallback.
    ClickHouse/Local would NOT — they force construction as the provided type.

    The workaround for Bug 1's fix: call load_nodes_by_sha256s to get class names,
    group SHA256s by type, then call load_by_sha256s once per type.
    """

    @pytest.mark.asyncio
    async def test_memory_store_preserves_type_on_fallback(self):
        """MemoryDocumentStore preserves original type when document_type doesn't match."""
        store = MemoryDocumentStore()
        scope = "test-scope"

        doc_a = TypeADoc.create_root(name="a.txt", content="type A", reason="test")
        doc_b = TypeBDoc.create_root(name="b.txt", content="type B", reason="test")

        await store.save(doc_a, scope)
        await store.save(doc_b, scope)

        sha256s = [DocumentSha256(doc_a.sha256), DocumentSha256(doc_b.sha256)]
        result = await store.load_by_sha256s(sha256s, Document)

        assert len(result) == 2
        assert isinstance(result[DocumentSha256(doc_a.sha256)], TypeADoc)
        assert isinstance(result[DocumentSha256(doc_b.sha256)], TypeBDoc)

    @pytest.mark.asyncio
    async def test_per_type_calls_preserve_identity_across_all_backends(self):
        """Calling load_by_sha256s once per type works correctly everywhere."""
        store = MemoryDocumentStore()
        scope = "test-scope"

        doc_a = TypeADoc.create_root(name="a.txt", content="A", reason="test")
        doc_b = TypeBDoc.create_root(name="b.txt", content="B", reason="test")
        await store.save(doc_a, scope)
        await store.save(doc_b, scope)

        result_a = await store.load_by_sha256s([DocumentSha256(doc_a.sha256)], TypeADoc)
        result_b = await store.load_by_sha256s([DocumentSha256(doc_b.sha256)], TypeBDoc)

        assert isinstance(result_a[DocumentSha256(doc_a.sha256)], TypeADoc)
        assert isinstance(result_b[DocumentSha256(doc_b.sha256)], TypeBDoc)

    @pytest.mark.asyncio
    async def test_nodes_then_per_type_load_workaround(self):
        """Demonstrates the workaround for multi-type SHA256 loading:
        1. load_nodes_by_sha256s to get class_name metadata
        2. Group SHA256s by class_name
        3. load_by_sha256s once per type

        This is the pattern the Bug 1 fix should use.
        """
        store = MemoryDocumentStore()
        scope = "test-scope"

        doc_a = TypeADoc.create_root(name="a.txt", content="A", reason="test")
        doc_b = TypeBDoc.create_root(name="b.txt", content="B", reason="test")
        await store.save(doc_a, scope)
        await store.save(doc_b, scope)

        all_sha256s = [DocumentSha256(doc_a.sha256), DocumentSha256(doc_b.sha256)]
        type_map = {"TypeADoc": TypeADoc, "TypeBDoc": TypeBDoc}

        # Step 1: get metadata
        nodes = await store.load_nodes_by_sha256s(all_sha256s)
        assert len(nodes) == 2

        # Step 2: group by type
        by_type: dict[type[Document], list[DocumentSha256]] = {}
        for sha, node in nodes.items():
            doc_type = type_map.get(node.class_name)
            assert doc_type is not None, f"Unknown class_name: {node.class_name}"
            by_type.setdefault(doc_type, []).append(sha)

        assert len(by_type) == 2

        # Step 3: load per type
        all_docs: list[Document] = []
        for doc_type, sha256s in by_type.items():
            loaded = await store.load_by_sha256s(sha256s, doc_type, scope)
            all_docs.extend(loaded.values())

        assert len(all_docs) == 2
        types = {type(d).__name__ for d in all_docs}
        assert types == {"TypeADoc", "TypeBDoc"}
