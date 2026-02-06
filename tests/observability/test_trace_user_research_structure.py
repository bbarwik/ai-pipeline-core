"""Test the specific user case: SingleResearchResults with Document fields.

This test validates that functions accepting list[SingleResearchResults] and
returning SingleResearchResults work correctly with the tracing system.
"""

from typing import Any
from unittest.mock import patch

from pydantic import BaseModel

from ai_pipeline_core.documents import Document
from ai_pipeline_core.observability.tracing import trace


# User's exact document structure
class SingleResearchReportDocument(Document):
    """Main research report."""


class SingleResearchAdditionalReportDocument(Document):
    """Additional research report."""


class SingleResearchFurtherResearchPlanDocument(Document):
    """Further research plan."""


class SingleResearchResults(BaseModel):
    """User's Pydantic model containing multiple document types."""

    research_report: SingleResearchReportDocument
    additional_research_report: SingleResearchAdditionalReportDocument
    further_research_plan: SingleResearchFurtherResearchPlanDocument


class TestUserResearchStructure:
    """Test the user's specific SingleResearchResults structure works with tracing."""

    @patch("ai_pipeline_core.observability.tracing.observe")
    def test_list_input_single_output(self, mock_observe):
        """Test list[SingleResearchResults] input and SingleResearchResults output."""

        # Setup mock
        def mock_decorator(f):
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        mock_observe.return_value = mock_decorator

        # Define the user's function pattern
        @trace(trim_documents=True)
        def process_research_list(results: list[SingleResearchResults]) -> SingleResearchResults:
            """Process list of research and return the best one."""
            if not results:
                # Create a default result
                return SingleResearchResults(
                    research_report=SingleResearchReportDocument(name="empty.md", content=b"No data"),
                    additional_research_report=SingleResearchAdditionalReportDocument(name="empty.md", content=b"No data"),
                    further_research_plan=SingleResearchFurtherResearchPlanDocument(name="empty.md", content=b"No data"),
                )
            # Return the first result
            return results[0]

        # Create test data with realistic content
        long_research = "This research finding is very important. " * 50  # ~1900 chars
        medium_content = "Supporting data and evidence. " * 20  # ~620 chars
        short_content = "Brief summary"

        # Create multiple research results
        result1 = SingleResearchResults(
            research_report=SingleResearchReportDocument(
                name="primary_research.md",
                content=long_research.encode(),
                description="Main findings from AI model analysis",
            ),
            additional_research_report=SingleResearchAdditionalReportDocument(
                name="supporting_data.json",
                content=medium_content.encode(),
                description="Additional supporting evidence",
            ),
            further_research_plan=SingleResearchFurtherResearchPlanDocument(
                name="next_steps.md",
                content=short_content.encode(),
                description="Plans for future research",
            ),
        )

        result2 = SingleResearchResults(
            research_report=SingleResearchReportDocument(
                name="secondary_research.md",
                content=b"Secondary findings from analysis.",
            ),
            additional_research_report=SingleResearchAdditionalReportDocument(
                name="extra_evidence.txt",
                content=("Extra data point " * 30).encode(),  # ~510 chars
            ),
            further_research_plan=SingleResearchFurtherResearchPlanDocument(
                name="future_work.md",
                content=("Detailed future plan " * 25).encode(),  # ~525 chars
            ),
        )

        # Call the function
        output = process_research_list([result1, result2])

        # Verify the function works correctly
        assert isinstance(output, SingleResearchResults)
        assert output.research_report.name == "primary_research.md"
        assert output.additional_research_report.name == "supporting_data.json"
        assert output.further_research_plan.name == "next_steps.md"

        # Verify observe was called with formatters
        observe_kwargs = mock_observe.call_args[1]
        assert "input_formatter" in observe_kwargs
        assert "output_formatter" in observe_kwargs

        # Get the formatters
        input_formatter = observe_kwargs["input_formatter"]
        output_formatter = observe_kwargs["output_formatter"]

        # Test input formatting for list[SingleResearchResults]
        import json

        formatted_input = input_formatter([result1, result2])
        input_data = json.loads(formatted_input)

        # Verify structure - parameter name as key
        assert "results" in input_data  # parameter name
        assert isinstance(input_data["results"], list)
        assert len(input_data["results"]) == 2

        # Check first result in list
        first_result = input_data["results"][0]

        # All documents use class_name and are trimmed equally - content is {v, e} format
        assert first_result["research_report"]["class_name"] == "SingleResearchReportDocument"
        # Long content (>250 chars) is trimmed for ALL documents
        assert "[trimmed" in first_result["research_report"]["content"]["v"]
        assert "chars]" in first_result["research_report"]["content"]["v"]

        # Additional report is also trimmed (>250 chars)
        assert first_result["additional_research_report"]["class_name"] == "SingleResearchAdditionalReportDocument"
        assert "[trimmed" in first_result["additional_research_report"]["content"]["v"]
        assert "chars]" in first_result["additional_research_report"]["content"]["v"]

        # Short content (<250 chars) is NOT trimmed - preserves {v, e} format
        assert first_result["further_research_plan"]["class_name"] == "SingleResearchFurtherResearchPlanDocument"
        assert first_result["further_research_plan"]["content"]["v"] == short_content

        # Check second result in list
        second_result = input_data["results"][1]

        # Short text - not trimmed - content is {v, e} format
        assert second_result["research_report"]["class_name"] == "SingleResearchReportDocument"
        assert second_result["research_report"]["content"]["v"] == "Secondary findings from analysis."

        # Long content - should be trimmed - content is {v, e} format
        assert second_result["additional_research_report"]["class_name"] == "SingleResearchAdditionalReportDocument"
        assert "[trimmed" in second_result["additional_research_report"]["content"]["v"]
        assert "chars]" in second_result["additional_research_report"]["content"]["v"]

        # Test output formatting for SingleResearchResults
        formatted_output = output_formatter(output)
        output_data = json.loads(formatted_output)

        # Verify output structure matches input
        assert "research_report" in output_data
        assert "additional_research_report" in output_data
        assert "further_research_plan" in output_data

        # Output should have same trimming rules applied (all long content trimmed) - content is {v, e} format
        assert output_data["research_report"]["class_name"] == "SingleResearchReportDocument"
        assert "[trimmed" in output_data["research_report"]["content"]["v"]

        assert output_data["additional_research_report"]["class_name"] == "SingleResearchAdditionalReportDocument"
        assert "[trimmed" in output_data["additional_research_report"]["content"]["v"]
        assert "chars]" in output_data["additional_research_report"]["content"]["v"]

    @patch("ai_pipeline_core.observability.tracing.observe")
    def test_complex_processing_pipeline(self, mock_observe):
        """Test a more complex pipeline with multiple processing steps."""

        # Setup mock
        def mock_decorator(f):
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        mock_observe.return_value = mock_decorator

        # Simulate a multi-step research pipeline
        @trace(trim_documents=True)
        def aggregate_research(
            batch1: list[SingleResearchResults],
            batch2: list[SingleResearchResults],
        ) -> SingleResearchResults:
            """Aggregate multiple batches of research."""
            all_results = batch1 + batch2
            if not all_results:
                raise ValueError("No research to aggregate")
            # Return the first one (in real code, would merge/aggregate)
            return all_results[0]

        # Create test batches
        batch1 = [
            SingleResearchResults(
                research_report=SingleResearchReportDocument(
                    name="batch1_research.md",
                    content=b"Batch 1 findings",
                ),
                additional_research_report=SingleResearchAdditionalReportDocument(
                    name="batch1_data.json",
                    content=b'{"data": "batch1"}',
                ),
                further_research_plan=SingleResearchFurtherResearchPlanDocument(
                    name="batch1_plan.md",
                    content=b"Next steps for batch 1",
                ),
            )
        ]

        batch2 = [
            SingleResearchResults(
                research_report=SingleResearchReportDocument(
                    name="batch2_research.md",
                    content=("Extensive batch 2 research " * 100).encode(),  # Long content
                ),
                additional_research_report=SingleResearchAdditionalReportDocument(
                    name="batch2_data.csv",
                    content=("row,col1,col2\n" * 50).encode(),  # CSV data
                ),
                further_research_plan=SingleResearchFurtherResearchPlanDocument(
                    name="batch2_plan.md",
                    content=("Detailed plan " * 50).encode(),  # Long plan
                ),
            )
        ]

        # Process
        result = aggregate_research(batch1, batch2)
        assert result.research_report.name == "batch1_research.md"

        # Verify formatters handle multiple list arguments
        observe_kwargs = mock_observe.call_args[1]
        input_formatter = observe_kwargs["input_formatter"]

        import json

        formatted = input_formatter(batch1, batch2)
        data = json.loads(formatted)

        # Should have two arguments with parameter names
        assert "batch1" in data
        assert "batch2" in data
        assert len(data["batch1"]) == 1  # batch1 has 1 item
        assert len(data["batch2"]) == 1  # batch2 has 1 item

        # Verify batch2's long content is trimmed (all documents trimmed equally) - content is {v, e} format
        batch2_data = data["batch2"][0]

        # All long documents are trimmed - content is {v, e} format
        assert "[trimmed" in batch2_data["research_report"]["content"]["v"]
        assert "[trimmed" in batch2_data["additional_research_report"]["content"]["v"]
        assert "chars]" in batch2_data["additional_research_report"]["content"]["v"]
        assert "[trimmed" in batch2_data["further_research_plan"]["content"]["v"]
        assert "chars]" in batch2_data["further_research_plan"]["content"]["v"]

    @patch("ai_pipeline_core.observability.tracing.observe")
    def test_with_binary_documents(self, mock_observe):
        """Test handling of binary content in research documents."""

        # Setup mock
        def mock_decorator(f):
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        mock_observe.return_value = mock_decorator

        @trace(trim_documents=True)
        def process_with_binary(research: SingleResearchResults) -> dict[str, Any]:
            """Process research containing binary data."""
            return {"status": "processed", "research": research}

        # Create research with binary content (e.g., images, PDFs)

        # Simulate binary data - use truly invalid UTF-8 bytes
        # PDF header with invalid UTF-8 bytes
        fake_pdf_bytes = b"%PDF-1.4\xff\xfe" + b"\x00" * 10
        fake_image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10  # PNG header (invalid UTF-8)

        research = SingleResearchResults(
            research_report=SingleResearchReportDocument(
                name="report.pdf",
                content=fake_pdf_bytes,  # Binary PDF
            ),
            additional_research_report=SingleResearchAdditionalReportDocument(
                name="chart.png",
                content=fake_image_bytes,  # Binary image
            ),
            further_research_plan=SingleResearchFurtherResearchPlanDocument(
                name="plan.txt",
                content=b"Text-based plan",  # Regular text
            ),
        )

        # Process
        result = process_with_binary(research)
        assert result["status"] == "processed"

        # Check formatting
        observe_kwargs = mock_observe.call_args[1]
        input_formatter = observe_kwargs["input_formatter"]

        import json

        formatted = input_formatter(research)
        data = json.loads(formatted)

        research_data = data["research"]  # parameter name

        # All binary content should be removed (regardless of document type) - content is {v, e} format
        assert research_data["research_report"]["content"]["e"] == "utf-8"  # Binary replaced with text marker
        assert research_data["research_report"]["content"]["v"] == "[binary content removed]"

        assert research_data["additional_research_report"]["content"]["e"] == "utf-8"  # Binary replaced with text marker
        assert research_data["additional_research_report"]["content"]["v"] == "[binary content removed]"

        # Text content should be preserved - content is {v, e} format
        assert research_data["further_research_plan"]["content"]["e"] == "utf-8"
        assert research_data["further_research_plan"]["content"]["v"] == "Text-based plan"

    @patch("ai_pipeline_core.observability.tracing.observe")
    def test_empty_and_none_handling(self, mock_observe):
        """Test edge cases with empty lists and None values."""

        # Setup mock
        def mock_decorator(f):
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        mock_observe.return_value = mock_decorator

        @trace(trim_documents=True)
        def handle_optional_research(
            results: list[SingleResearchResults] | None,
            fallback: SingleResearchResults | None = None,
        ) -> SingleResearchResults | None:
            """Handle optional research data."""
            if results:
                return results[0]
            return fallback

        # Test with None
        result = handle_optional_research(None, None)
        assert result is None

        # Test with empty list
        result = handle_optional_research([], None)
        assert result is None

        # Create a fallback
        fallback = SingleResearchResults(
            research_report=SingleResearchReportDocument(name="default.md", content=b"Default"),
            additional_research_report=SingleResearchAdditionalReportDocument(name="default.json", content=b"{}"),
            further_research_plan=SingleResearchFurtherResearchPlanDocument(name="default.txt", content=b"TBD"),
        )

        # Test with empty list but fallback provided
        result = handle_optional_research([], fallback)
        assert result == fallback

        # Verify formatters handle None correctly
        observe_kwargs = mock_observe.call_args[1]
        input_formatter = observe_kwargs["input_formatter"]
        output_formatter = observe_kwargs["output_formatter"]

        import json

        # Format None inputs
        formatted = input_formatter(None, None)
        data = json.loads(formatted)
        assert data["results"] is None
        assert data["fallback"] is None

        # Format None output
        formatted = output_formatter(None)
        assert json.loads(formatted) is None

        # Format with fallback
        formatted = input_formatter([], fallback)
        data = json.loads(formatted)
        assert data["results"] == []
        assert "research_report" in data["fallback"]

    def test_integration_with_real_trace(self):
        """Integration test with actual trace execution (no mocks)."""
        # This test verifies the full integration works end-to-end

        @trace(trim_documents=False)  # Disable trimming for comparison
        def without_trimming(results: list[SingleResearchResults]) -> SingleResearchResults | None:
            return results[0] if results else None

        @trace(trim_documents=True)  # Enable trimming
        def with_trimming(results: list[SingleResearchResults]) -> SingleResearchResults | None:
            return results[0] if results else None

        # Create test data
        long_content = "x" * 1000
        result = SingleResearchResults(
            research_report=SingleResearchReportDocument(
                name="test.md",
                content=long_content.encode(),
            ),
            additional_research_report=SingleResearchAdditionalReportDocument(
                name="test.json",
                content=long_content.encode(),
            ),
            further_research_plan=SingleResearchFurtherResearchPlanDocument(
                name="test.txt",
                content=b"short",
            ),
        )

        # Both should work without errors
        output1 = without_trimming([result])
        output2 = with_trimming([result])

        # Both should return the same result
        assert output1 and output2
        assert output1.research_report.name == output2.research_report.name
        assert output1.research_report.content == output2.research_report.content

        # The actual documents are unchanged - only tracing is affected
        assert len(output2.additional_research_report.content) == 1000
