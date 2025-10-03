"""Tests to detect mutable default parameter issues in Tracing module."""

from ai_pipeline_core.tracing import TraceInfo


class TestTraceInfoMutableDefaults:
    """Test that TraceInfo doesn't use mutable defaults for metadata and tags."""

    def test_multiple_instances_have_independent_metadata_dicts(self):
        """Verify that creating multiple TraceInfo instances doesn't share metadata dicts."""
        # Create first TraceInfo without metadata (using default)
        trace1 = TraceInfo()

        # Create second TraceInfo without metadata (using default)
        trace2 = TraceInfo()

        # Both should have empty metadata dicts
        assert trace1.metadata == {}
        assert trace2.metadata == {}

        # But they should NOT be the same dict instance
        assert trace1.metadata is not trace2.metadata, (
            "TraceInfo instances are sharing the same metadata dict! "
            "This indicates a mutable default parameter bug."
        )

    def test_multiple_instances_have_independent_tags_lists(self):
        """Verify that creating multiple TraceInfo instances doesn't share tags lists."""
        # Create first TraceInfo without tags (using default)
        trace1 = TraceInfo()

        # Create second TraceInfo without tags (using default)
        trace2 = TraceInfo()

        # Both should have empty tags lists
        assert trace1.tags == []
        assert trace2.tags == []

        # But they should NOT be the same list instance
        assert trace1.tags is not trace2.tags, (
            "TraceInfo instances are sharing the same tags list! "
            "This indicates a mutable default parameter bug."
        )

    def test_metadata_dict_not_shared_across_calls(self):
        """Verify metadata default doesn't persist across TraceInfo creations."""
        # Create multiple TraceInfo instances
        traces = [TraceInfo() for _ in range(5)]

        # All should have independent metadata dicts
        metadata_ids = [id(trace.metadata) for trace in traces]
        assert len(set(metadata_ids)) == 5, "Some TraceInfo instances share the same metadata dict"

    def test_tags_list_not_shared_across_calls(self):
        """Verify tags default doesn't persist across TraceInfo creations."""
        # Create multiple TraceInfo instances
        traces = [TraceInfo() for _ in range(5)]

        # All should have independent tags lists
        tags_ids = [id(trace.tags) for trace in traces]
        assert len(set(tags_ids)) == 5, "Some TraceInfo instances share the same tags list"

    def test_explicit_metadata_not_affected_by_default(self):
        """Verify that explicitly passing metadata works correctly."""
        # Create TraceInfo with explicit metadata
        trace1 = TraceInfo(metadata={"key1": "value1", "key2": "value2"})

        # Create TraceInfo without metadata (default)
        trace2 = TraceInfo()

        # Verify they are independent
        assert len(trace1.metadata) == 2
        assert len(trace2.metadata) == 0
        assert trace1.metadata is not trace2.metadata

    def test_explicit_tags_not_affected_by_default(self):
        """Verify that explicitly passing tags works correctly."""
        # Create TraceInfo with explicit tags
        trace1 = TraceInfo(tags=["tag1", "tag2", "tag3"])

        # Create TraceInfo without tags (default)
        trace2 = TraceInfo()

        # Verify they are independent
        assert len(trace1.tags) == 3
        assert len(trace2.tags) == 0
        assert trace1.tags is not trace2.tags

    def test_mixing_default_and_explicit_metadata(self):
        """Test mixing TraceInfo instances with default and explicit metadata."""
        traces = []

        # Create with defaults
        traces.append(TraceInfo())
        traces.append(TraceInfo())

        # Create with explicit metadata
        traces.append(TraceInfo(metadata={"key": "value"}))

        # Create another with default
        traces.append(TraceInfo())

        # Verify all metadata dicts are independent
        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                assert traces[i].metadata is not traces[j].metadata, (
                    f"TraceInfo instances {i} and {j} share the same metadata dict"
                )

        # Verify the explicit one has correct content
        assert len(traces[2].metadata) == 1
        assert traces[2].metadata["key"] == "value"

    def test_mixing_default_and_explicit_tags(self):
        """Test mixing TraceInfo instances with default and explicit tags."""
        traces = []

        # Create with defaults
        traces.append(TraceInfo())
        traces.append(TraceInfo())

        # Create with explicit tags
        traces.append(TraceInfo(tags=["tag1", "tag2"]))

        # Create another with default
        traces.append(TraceInfo())

        # Verify all tags lists are independent
        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                assert traces[i].tags is not traces[j].tags, (
                    f"TraceInfo instances {i} and {j} share the same tags list"
                )

        # Verify the explicit one has correct content
        assert len(traces[2].tags) == 2
        assert "tag1" in traces[2].tags

    def test_trace_info_with_all_parameters(self):
        """Test TraceInfo with all parameters explicitly set."""
        trace1 = TraceInfo(
            session_id="session1",
            user_id="user1",
            metadata={"key1": "val1"},
            tags=["tag1"],
        )

        trace2 = TraceInfo(
            session_id="session2",
            user_id="user2",
            metadata={"key2": "val2"},
            tags=["tag2"],
        )

        # Verify independence
        assert trace1.metadata is not trace2.metadata
        assert trace1.tags is not trace2.tags
        assert trace1.metadata["key1"] == "val1"
        assert trace2.metadata["key2"] == "val2"
        assert trace1.tags[0] == "tag1"
        assert trace2.tags[0] == "tag2"

    def test_trace_info_get_observe_kwargs(self):
        """Test that get_observe_kwargs creates fresh wrapper dicts."""
        trace = TraceInfo(
            session_id="test_session",
            metadata={"key": "value"},
            tags=["tag1"],
        )

        # Call get_observe_kwargs multiple times
        kwargs1 = trace.get_observe_kwargs()
        kwargs2 = trace.get_observe_kwargs()

        # Should return equal but independent wrapper dicts
        assert kwargs1 == kwargs2
        assert kwargs1 is not kwargs2, "get_observe_kwargs() is returning the same dict instance"

        # Note: The metadata and tags inside are references to the TraceInfo's
        # internal objects - this is existing behavior and not a bug

    def test_trace_info_immutability_expectation(self):
        """Test that TraceInfo follows Pydantic patterns for immutability."""
        trace = TraceInfo(metadata={"key": "value"}, tags=["tag1"])

        # Pydantic models with mutable fields should still protect against
        # default parameter sharing issues
        original_metadata_id = id(trace.metadata)
        original_tags_id = id(trace.tags)

        # Create another instance
        trace2 = TraceInfo()

        # Verify different instances
        assert id(trace2.metadata) != original_metadata_id
        assert id(trace2.tags) != original_tags_id
