"""Smoke tests verifying showcase examples import without errors and have valid flow chain structure."""


def test_showcase_imports() -> None:
    """examples/showcase.py imports without errors."""
    import examples.showcase as m

    assert hasattr(m, "ShowcasePipeline")


def test_showcase_database_imports() -> None:
    """examples/showcase_database.py imports without errors."""
    import examples.showcase_database as m

    assert hasattr(m, "DatabaseShowcasePipeline")


def test_showcase_replay_imports() -> None:
    """examples/showcase_replay.py imports without errors."""
    import examples.showcase_replay as m

    assert hasattr(m, "ReplayUppercaseTask")


def test_showcase_builds_flows() -> None:
    """ShowcasePipeline.build_flows() returns PipelineFlow instances."""
    from examples.showcase import ShowcasePipeline
    from ai_pipeline_core.pipeline import PipelineFlow, FlowOptions

    deployment = ShowcasePipeline()
    flows = deployment.build_flows(FlowOptions())
    assert len(flows) >= 1
    assert all(isinstance(f, PipelineFlow) for f in flows)


def test_database_showcase_builds_flows() -> None:
    """DatabaseShowcasePipeline.build_flows() returns PipelineFlow instances."""
    from examples.showcase_database import DatabaseShowcasePipeline
    from ai_pipeline_core.pipeline import PipelineFlow, FlowOptions

    deployment = DatabaseShowcasePipeline()
    flows = deployment.build_flows(FlowOptions())
    assert len(flows) >= 1
    assert all(isinstance(f, PipelineFlow) for f in flows)
