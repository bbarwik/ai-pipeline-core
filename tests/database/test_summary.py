"""Tests for summary and cost report generation."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from ai_pipeline_core.database import ExecutionNode, MemoryDatabase, NodeKind, NodeStatus
from ai_pipeline_core.database._summary import _TreeIndex, _build_parent_chain, _build_tree_lines, generate_costs, generate_summary


def _make_node(**kwargs: object) -> ExecutionNode:
    did = kwargs.pop("_deployment_id", None) or uuid4()
    rid = kwargs.pop("_root_deployment_id", None) or did
    defaults: dict[str, object] = {
        "node_id": uuid4(),
        "node_kind": NodeKind.TASK,
        "deployment_id": did,
        "root_deployment_id": rid,
        "run_id": "test-run",
        "run_scope": "test-run/scope",
        "deployment_name": "test-pipeline",
        "name": "TestTask",
        "sequence_no": 0,
    }
    defaults.update(kwargs)
    return ExecutionNode(**defaults)  # type: ignore[arg-type]


async def _build_test_tree(db: MemoryDatabase) -> tuple[UUID, list[ExecutionNode]]:
    """Build a realistic execution tree for testing."""
    deploy_id = uuid4()
    now = datetime.now(UTC)
    nodes: list[ExecutionNode] = []

    # Deployment
    deploy = _make_node(
        node_kind=NodeKind.DEPLOYMENT,
        _deployment_id=deploy_id,
        name="research-pipeline",
        deployment_name="research-pipeline",
        run_id="run-abc123",
        status=NodeStatus.COMPLETED,
        started_at=now - timedelta(minutes=7),
        ended_at=now,
        sequence_no=0,
    )
    nodes.append(deploy)

    # Flow 1 (completed)
    flow1 = _make_node(
        node_kind=NodeKind.FLOW,
        _deployment_id=deploy_id,
        parent_node_id=deploy.node_id,
        name="CreatePlanFlow",
        sequence_no=1,
        status=NodeStatus.COMPLETED,
        started_at=now - timedelta(minutes=7),
        ended_at=now - timedelta(minutes=6),
        flow_id=None,
    )
    nodes.append(flow1)

    # Task in flow 1
    task1 = _make_node(
        node_kind=NodeKind.TASK,
        _deployment_id=deploy_id,
        parent_node_id=flow1.node_id,
        name="CreatePlanTask",
        sequence_no=1,
        status=NodeStatus.COMPLETED,
        started_at=now - timedelta(minutes=7),
        ended_at=now - timedelta(minutes=6, seconds=30),
        flow_id=flow1.node_id,
    )
    nodes.append(task1)

    # Conversation turn in task1
    turn1 = _make_node(
        node_kind=NodeKind.CONVERSATION_TURN,
        _deployment_id=deploy_id,
        parent_node_id=task1.node_id,
        name="turn_0",
        sequence_no=0,
        status=NodeStatus.COMPLETED,
        started_at=now - timedelta(minutes=7),
        ended_at=now - timedelta(minutes=6, seconds=30),
        model="gemini-3-pro",
        cost_usd=0.182,
        tokens_input=12000,
        tokens_output=1100,
        flow_id=flow1.node_id,
        task_id=task1.node_id,
    )
    nodes.append(turn1)

    # Flow 2 (failed)
    flow2 = _make_node(
        node_kind=NodeKind.FLOW,
        _deployment_id=deploy_id,
        parent_node_id=deploy.node_id,
        name="ResearchLoopFlow",
        sequence_no=2,
        status=NodeStatus.FAILED,
        started_at=now - timedelta(minutes=6),
        ended_at=now - timedelta(seconds=6),
        flow_id=None,
    )
    nodes.append(flow2)

    # Task in flow 2 (failed)
    task2 = _make_node(
        node_kind=NodeKind.TASK,
        _deployment_id=deploy_id,
        parent_node_id=flow2.node_id,
        name="AnalyzeResultsTask",
        sequence_no=1,
        status=NodeStatus.FAILED,
        started_at=now - timedelta(minutes=5),
        ended_at=now - timedelta(seconds=6),
        error_type="OutputDegenerationError",
        error_message="substring '...' repeated 84 times",
        flow_id=flow2.node_id,
    )
    nodes.append(task2)

    # Turn in task2
    turn2 = _make_node(
        node_kind=NodeKind.CONVERSATION_TURN,
        _deployment_id=deploy_id,
        parent_node_id=task2.node_id,
        name="turn_0",
        sequence_no=0,
        status=NodeStatus.COMPLETED,
        started_at=now - timedelta(minutes=5),
        ended_at=now - timedelta(minutes=2),
        model="gpt-5.1",
        cost_usd=2.7712,
        tokens_input=48000,
        tokens_output=3900,
        flow_id=flow2.node_id,
        task_id=task2.node_id,
    )
    nodes.append(turn2)

    for node in nodes:
        await db.insert_node(node)

    return deploy_id, nodes


class TestGenerateSummary:
    async def test_summary_contains_header(self) -> None:
        db = MemoryDatabase()
        deploy_id, _ = await _build_test_tree(db)
        summary = await generate_summary(db, deploy_id)
        assert "# research-pipeline / run-abc123" in summary

    async def test_summary_contains_status(self) -> None:
        db = MemoryDatabase()
        deploy_id, _ = await _build_test_tree(db)
        summary = await generate_summary(db, deploy_id)
        assert "Failed" in summary

    async def test_summary_contains_flow_plan(self) -> None:
        db = MemoryDatabase()
        deploy_id, _ = await _build_test_tree(db)
        summary = await generate_summary(db, deploy_id)
        assert "CreatePlanFlow" in summary
        assert "ResearchLoopFlow" in summary

    async def test_summary_contains_failures(self) -> None:
        db = MemoryDatabase()
        deploy_id, _ = await _build_test_tree(db)
        summary = await generate_summary(db, deploy_id)
        assert "AnalyzeResultsTask" in summary
        assert "OutputDegenerationError" in summary

    async def test_summary_contains_execution_tree(self) -> None:
        db = MemoryDatabase()
        deploy_id, _ = await _build_test_tree(db)
        summary = await generate_summary(db, deploy_id)
        assert "## Execution Tree" in summary

    async def test_summary_contains_cost(self) -> None:
        db = MemoryDatabase()
        deploy_id, _ = await _build_test_tree(db)
        summary = await generate_summary(db, deploy_id)
        # Total cost should be sum of turn costs
        assert "$" in summary

    async def test_summary_navigation_references_database_snapshot_layout(self) -> None:
        db = MemoryDatabase()
        deploy_id, _ = await _build_test_tree(db)
        summary = await generate_summary(db, deploy_id)
        assert "`runs/`" in summary
        assert "`logs.jsonl`" in summary
        assert "`costs.md`" in summary
        assert "`blobs/`" in summary
        assert "`llm_calls.yaml`" not in summary
        assert "`errors.yaml`" not in summary

    async def test_empty_deployment_returns_placeholder(self) -> None:
        db = MemoryDatabase()
        summary = await generate_summary(db, uuid4())
        assert "No execution data found" in summary


class TestGenerateCosts:
    async def test_costs_by_flow(self) -> None:
        db = MemoryDatabase()
        deploy_id, _ = await _build_test_tree(db)
        costs = await generate_costs(db, deploy_id)
        assert "# Cost by Flow" in costs
        assert "CreatePlanFlow" in costs
        assert "ResearchLoopFlow" in costs

    async def test_costs_by_task(self) -> None:
        db = MemoryDatabase()
        deploy_id, _ = await _build_test_tree(db)
        costs = await generate_costs(db, deploy_id)
        assert "## Cost by Task" in costs

    async def test_total_cost(self) -> None:
        db = MemoryDatabase()
        deploy_id, _ = await _build_test_tree(db)
        costs = await generate_costs(db, deploy_id)
        assert "**Total**" in costs

    async def test_empty_returns_empty_string(self) -> None:
        db = MemoryDatabase()
        costs = await generate_costs(db, uuid4())
        assert costs == ""


class TestSummaryCycleGuards:
    async def test_generate_summary_survives_cycle_in_tree_index(self) -> None:
        db = MemoryDatabase()
        deployment_id = uuid4()
        deployment = _make_node(
            node_kind=NodeKind.DEPLOYMENT,
            _deployment_id=deployment_id,
            name="CycleDeployment",
            deployment_name="CycleDeployment",
            status=NodeStatus.FAILED,
        )
        first = _make_node(
            node_kind=NodeKind.TASK,
            _deployment_id=deployment_id,
            parent_node_id=deployment.node_id,
            name="FirstTask",
            sequence_no=1,
            status=NodeStatus.FAILED,
        )
        second = _make_node(
            node_kind=NodeKind.TASK,
            _deployment_id=deployment_id,
            parent_node_id=first.node_id,
            name="SecondTask",
            sequence_no=2,
            status=NodeStatus.FAILED,
        )
        await db.insert_node(deployment)
        await db.insert_node(first)
        await db.insert_node(second)
        db._nodes[first.node_id] = replace(first, parent_node_id=second.node_id)

        summary = await generate_summary(db, deployment_id)

        assert "## Execution Tree" in summary
        assert "FirstTask" in summary
        assert "SecondTask" in summary

    async def test_generate_costs_survives_cycle_in_tree_index(self) -> None:
        db = MemoryDatabase()
        deployment_id = uuid4()
        deployment = _make_node(
            node_kind=NodeKind.DEPLOYMENT,
            _deployment_id=deployment_id,
            name="CycleDeployment",
            deployment_name="CycleDeployment",
            status=NodeStatus.COMPLETED,
        )
        flow = _make_node(
            node_kind=NodeKind.FLOW,
            _deployment_id=deployment_id,
            parent_node_id=deployment.node_id,
            flow_id=None,
            name="LoopFlow",
            sequence_no=1,
            status=NodeStatus.COMPLETED,
        )
        task = _make_node(
            node_kind=NodeKind.TASK,
            _deployment_id=deployment_id,
            parent_node_id=flow.node_id,
            flow_id=flow.node_id,
            name="LoopTask",
            sequence_no=1,
            status=NodeStatus.COMPLETED,
        )
        await db.insert_node(deployment)
        await db.insert_node(flow)
        await db.insert_node(task)
        db._nodes[flow.node_id] = replace(flow, parent_node_id=task.node_id)

        costs = await generate_costs(db, deployment_id)

        assert "# Cost by Flow" in costs
        assert "LoopFlow" in costs

    def test_build_parent_chain_stops_on_cycle(self) -> None:
        deployment_id = uuid4()
        first_id = uuid4()
        second_id = uuid4()
        first = _make_node(
            node_id=first_id,
            node_kind=NodeKind.TASK,
            _deployment_id=deployment_id,
            parent_node_id=second_id,
            name="FirstTask",
        )
        second = _make_node(
            node_id=second_id,
            node_kind=NodeKind.TASK,
            _deployment_id=deployment_id,
            parent_node_id=first_id,
            name="SecondTask",
        )

        chain = _build_parent_chain(first, {first.node_id: first, second.node_id: second})
        assert chain == ["FirstTask", "SecondTask"]

    def test_build_tree_lines_reports_cycle_once(self) -> None:
        deployment_id = uuid4()
        root = _make_node(
            node_kind=NodeKind.DEPLOYMENT,
            _deployment_id=deployment_id,
            name="CycleDeployment",
        )
        first = _make_node(
            node_kind=NodeKind.TASK,
            _deployment_id=deployment_id,
            parent_node_id=root.node_id,
            name="FirstTask",
            sequence_no=1,
        )
        second = _make_node(
            node_kind=NodeKind.TASK,
            _deployment_id=deployment_id,
            parent_node_id=first.node_id,
            name="SecondTask",
            sequence_no=2,
        )
        index = _TreeIndex(
            nodes_by_id={
                root.node_id: root,
                first.node_id: first,
                second.node_id: second,
            },
            children_map={
                root.node_id: [first.node_id],
                first.node_id: [second.node_id],
                second.node_id: [first.node_id],
            },
            tasks_by_flow_id={},
            turns_by_flow_id={},
            flow_names_by_id={},
            descendant_turn_costs={},
            descendant_turn_counts={},
        )

        lines: list[str] = []
        _build_tree_lines(root, index, 0, lines)

        assert any("cycle detected" in line for line in lines)
