"""Test document trimming with Pydantic models containing Documents."""

from typing import Any
from unittest.mock import patch

from pydantic import BaseModel

from ai_pipeline_core.documents import Document
from ai_pipeline_core.observability.tracing import trace


class ResearchReportDocument(Document):
    """Main research report - Document (important, keep full text)."""


class AdditionalReportDocument(Document):
    """Additional research - Document (less important, can trim)."""


class FurtherResearchPlanDocument(Document):
    """Further research plan - Document (less important, can trim)."""


class SingleResearchResults(BaseModel):
    """Pydantic model containing multiple document types."""

    research_report: ResearchReportDocument
    additional_research_report: AdditionalReportDocument
    further_research_plan: FurtherResearchPlanDocument


class TestPydanticModelTrimming:
    """Test that Pydantic models with Document fields are properly handled."""

    @patch("ai_pipeline_core.observability.tracing.observe")
    def test_pydantic_model_with_documents_as_input(self, mock_observe):
        """Test function accepting list[SingleResearchResults]."""

        # Create mock that captures formatters and properly wraps function
        def mock_decorator(f):
            # Return a wrapper that properly handles arguments
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        mock_observe.return_value = mock_decorator

        # Create traced function
        @trace(trim_documents=True)
        def process_research_list(
            results: list[SingleResearchResults],
        ) -> SingleResearchResults | None:
            return results[0] if results else None

        # Create test data with long content
        long_text = "x" * 500  # Long text for testing trimming

        # Create documents
        main_report = ResearchReportDocument(
            name="main_report.md",
            content=long_text.encode(),
            description="Main research findings",
        )
        additional_report = AdditionalReportDocument(
            name="additional.md",
            content=long_text.encode(),
            description="Supporting research",
        )
        research_plan = FurtherResearchPlanDocument(name="plan.md", content=long_text.encode(), description="Next steps")

        # Create SingleResearchResults
        research_results = SingleResearchResults(
            research_report=main_report,
            additional_research_report=additional_report,
            further_research_plan=research_plan,
        )

        # Call function with list of results
        result = process_research_list([research_results])

        # Verify function works
        assert result == research_results

        # Verify observe was called with formatters
        observe_kwargs = mock_observe.call_args[1]
        assert "input_formatter" in observe_kwargs
        assert "output_formatter" in observe_kwargs

        # Test the input formatter
        input_formatter = observe_kwargs["input_formatter"]
        formatted_input = input_formatter([research_results])

        # Parse the JSON to verify structure
        import json

        input_data = json.loads(formatted_input)

        # Check that the structure is preserved - parameter name as key
        assert "results" in input_data  # parameter name is 'results'
        assert len(input_data["results"]) == 1  # List with one item

        # Get the serialized research results
        serialized_results = input_data["results"][0]

        # All documents are trimmed equally (>250 chars)
        assert "[trimmed" in serialized_results["research_report"]["content"]
        assert "chars]" in serialized_results["research_report"]["content"]
        assert "[trimmed" in serialized_results["additional_research_report"]["content"]
        assert "chars]" in serialized_results["additional_research_report"]["content"]
        assert "[trimmed" in serialized_results["further_research_plan"]["content"]
        assert "chars]" in serialized_results["further_research_plan"]["content"]

        # Test the output formatter
        output_formatter = observe_kwargs["output_formatter"]
        formatted_output = output_formatter(research_results)

        output_data = json.loads(formatted_output)

        # Verify output structure
        assert "research_report" in output_data
        assert "additional_research_report" in output_data
        assert "further_research_plan" in output_data

        # Same trimming rules apply to output (all trimmed equally)
        assert "[trimmed" in output_data["research_report"]["content"]
        assert "[trimmed" in output_data["additional_research_report"]["content"]
        assert "chars]" in output_data["additional_research_report"]["content"]
        assert "[trimmed" in output_data["further_research_plan"]["content"]
        assert "chars]" in output_data["further_research_plan"]["content"]

    @patch("ai_pipeline_core.observability.tracing.observe")
    def test_nested_pydantic_models(self, mock_observe):
        """Test deeply nested Pydantic models with documents."""

        class ResearchBatch(BaseModel):
            """A batch of research results."""

            batch_id: str
            results: list[SingleResearchResults]
            metadata: dict[str, str]

        # Create mock that properly wraps function
        def mock_decorator(f):
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        mock_observe.return_value = mock_decorator

        @trace(trim_documents=True)
        def process_batch(batch: ResearchBatch) -> dict[str, Any]:
            return {"processed": True, "batch": batch}

        # Create test data
        long_text = "y" * 400

        research1 = SingleResearchResults(
            research_report=ResearchReportDocument(name="r1.md", content=long_text.encode()),
            additional_research_report=AdditionalReportDocument(name="a1.md", content=long_text.encode()),
            further_research_plan=FurtherResearchPlanDocument(name="p1.md", content=long_text.encode()),
        )

        research2 = SingleResearchResults(
            research_report=ResearchReportDocument(name="r2.md", content=long_text.encode()),
            additional_research_report=AdditionalReportDocument(name="a2.md", content=long_text.encode()),
            further_research_plan=FurtherResearchPlanDocument(name="p2.md", content=long_text.encode()),
        )

        batch = ResearchBatch(
            batch_id="batch-001",
            results=[research1, research2],
            metadata={"project": "AI Research", "version": "1.0"},
        )

        # Process the batch
        result = process_batch(batch)
        assert result["processed"] is True

        # Get formatters
        observe_kwargs = mock_observe.call_args[1]
        input_formatter = observe_kwargs["input_formatter"]

        # Test input formatting
        formatted_input = input_formatter(batch)
        import json

        input_data = json.loads(formatted_input)

        # Navigate to nested documents
        batch_data = input_data["batch"]  # parameter name
        results_data = batch_data["results"]

        # Verify both results are present
        assert len(results_data) == 2

        # Check first result - all documents trimmed equally
        result1 = results_data[0]
        assert "[trimmed" in result1["research_report"]["content"]
        assert "[trimmed" in result1["additional_research_report"]["content"]
        assert "chars]" in result1["additional_research_report"]["content"]

        # Check second result - all documents trimmed equally
        result2 = results_data[1]
        assert "[trimmed" in result2["research_report"]["content"]
        assert "[trimmed" in result2["further_research_plan"]["content"]
        assert "chars]" in result2["further_research_plan"]["content"]

        # Verify metadata is preserved
        assert batch_data["batch_id"] == "batch-001"
        assert batch_data["metadata"]["project"] == "AI Research"

    @patch("ai_pipeline_core.observability.tracing.observe")
    def test_binary_content_in_pydantic_model(self, mock_observe):
        """Test that binary content in Pydantic model documents is removed."""

        # Create mock that properly wraps function
        def mock_decorator(f):
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        mock_observe.return_value = mock_decorator

        @trace(trim_documents=True)
        def process_with_images(results: SingleResearchResults) -> str:
            return "processed"

        # Create documents with binary content (images)

        # Create a small fake "image" (just some bytes)
        image_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"

        # Create documents - mix of text and binary
        # Note: Documents auto-detect encoding, don't pass content_encoding
        main_report = ResearchReportDocument(
            name="report_with_image.pdf",
            content=image_bytes,  # Pass raw binary (will be base64 encoded internally)
        )
        additional = AdditionalReportDocument(name="data.json", content=b'{"key": "value"}')
        plan = FurtherResearchPlanDocument(
            name="diagram.png",
            content=image_bytes,  # Pass raw binary
        )

        research = SingleResearchResults(
            research_report=main_report,
            additional_research_report=additional,
            further_research_plan=plan,
        )

        # Process
        result = process_with_images(research)
        assert result == "processed"

        # Get formatter
        observe_kwargs = mock_observe.call_args[1]
        input_formatter = observe_kwargs["input_formatter"]

        # Test formatting
        formatted = input_formatter(research)
        import json

        data = json.loads(formatted)
        research_data = data["results"]  # parameter name

        # Binary content should be removed (PDFs and images are base64 encoded)
        # Documents with image extensions are automatically base64 encoded
        assert research_data["research_report"]["content"] == "[binary content removed]"
        assert research_data["further_research_plan"]["content"] == "[binary content removed]"

        # Text content (JSON) should be preserved (short, so not trimmed)
        assert research_data["additional_research_report"]["content"] == '{"key": "value"}'

    @patch("ai_pipeline_core.observability.tracing.observe")
    def test_mixed_types_with_pydantic(self, mock_observe):
        """Test function with mixed types including Pydantic models."""

        # Create mock that properly wraps function
        def mock_decorator(f):
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        mock_observe.return_value = mock_decorator

        @trace(trim_documents=True)
        def complex_function(
            text: str,
            research: SingleResearchResults,
            config: dict[str, Any],
            count: int,
        ) -> tuple[SingleResearchResults, str]:
            return research, f"Processed {count} items"

        # Create test data
        research = SingleResearchResults(
            research_report=ResearchReportDocument(name="r.md", content=b"short content"),
            additional_research_report=AdditionalReportDocument(name="a.md", content=b"data"),
            further_research_plan=FurtherResearchPlanDocument(name="p.md", content=b"plan"),
        )

        # Call function
        result = complex_function("input text", research, {"setting": "value"}, 42)

        assert result[0] == research
        assert result[1] == "Processed 42 items"

        # Verify formatters handle all types
        observe_kwargs = mock_observe.call_args[1]
        input_formatter = observe_kwargs["input_formatter"]
        output_formatter = observe_kwargs["output_formatter"]

        # Test input formatting
        formatted_input = input_formatter("input text", research, {"setting": "value"}, 42)
        import json

        input_data = json.loads(formatted_input)

        # Verify all arguments are present
        # Verify all arguments are present with parameter names
        assert input_data["text"] == "input text"
        assert "research_report" in input_data["research"]  # Pydantic model serialized
        assert input_data["config"] == {"setting": "value"}
        assert input_data["count"] == 42

        # Test output formatting (tuple with Pydantic model)
        formatted_output = output_formatter(result)
        output_data = json.loads(formatted_output)

        # Tuple converted to list
        assert len(output_data) == 2
        assert "research_report" in output_data[0]  # Pydantic model serialized
        assert output_data[1] == "Processed 42 items"

    @patch("ai_pipeline_core.observability.tracing.observe")
    def test_empty_and_none_values(self, mock_observe):
        """Test that empty lists, None values, and edge cases are handled."""

        # Create mock that properly wraps function
        def mock_decorator(f):
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        mock_observe.return_value = mock_decorator

        @trace(trim_documents=True)
        def handle_edge_cases(
            results: list[SingleResearchResults] | None,
        ) -> SingleResearchResults | None:
            return results[0] if results else None

        # Test with None
        result = handle_edge_cases(None)
        assert result is None

        # Test with empty list
        result = handle_edge_cases([])
        assert result is None

        # Get formatter
        observe_kwargs = mock_observe.call_args[1]
        input_formatter = observe_kwargs["input_formatter"]
        output_formatter = observe_kwargs["output_formatter"]

        # Test formatting None
        formatted = input_formatter(None)
        import json

        data = json.loads(formatted)
        assert data["results"] is None  # parameter name

        # Test formatting empty list
        formatted = input_formatter([])
        data = json.loads(formatted)
        assert data["results"] == []  # parameter name

        # Test output formatter with None
        formatted = output_formatter(None)
        assert json.loads(formatted) is None
