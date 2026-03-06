"""Smoke tests verifying showcase examples import without errors and have valid flow chain structure."""


def test_showcase_imports() -> None:
    """examples/showcase.py imports without errors."""
    import examples.showcase as m

    assert hasattr(m, "ShowcasePipeline")


def test_showcase_document_store_imports() -> None:
    """examples/showcase_document_store.py imports without errors."""
    import examples.showcase_document_store as m

    assert hasattr(m, "StoreShowcasePipeline")


def test_showcase_builds_flows() -> None:
    """ShowcasePipeline.build_flows() returns PipelineFlow instances."""
    from examples.showcase import ShowcasePipeline
    from ai_pipeline_core.pipeline import PipelineFlow, FlowOptions

    deployment = ShowcasePipeline()
    flows = deployment.build_flows(FlowOptions())
    assert len(flows) >= 1
    assert all(isinstance(f, PipelineFlow) for f in flows)


def test_store_showcase_builds_flows() -> None:
    """StoreShowcasePipeline.build_flows() returns PipelineFlow instances."""
    from examples.showcase_document_store import StoreShowcasePipeline
    from ai_pipeline_core.pipeline import PipelineFlow, FlowOptions

    deployment = StoreShowcasePipeline()
    flows = deployment.build_flows(FlowOptions())
    assert len(flows) >= 1
    assert all(isinstance(f, PipelineFlow) for f in flows)
