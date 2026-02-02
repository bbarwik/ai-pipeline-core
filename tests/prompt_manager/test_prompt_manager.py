"""Tests for PromptManager."""

from pathlib import Path

import pytest

from ai_pipeline_core.exceptions import PromptNotFoundError, PromptRenderError
from ai_pipeline_core.prompt_manager import PromptManager


class TestPromptManager:
    """Test PromptManager functionality."""

    def test_search_paths_construction(self, tmp_path: Path) -> None:
        """Test search path construction logic."""
        # Create nested package structure
        parent = tmp_path / "parent_pkg"
        parent.mkdir()
        (parent / "__init__.py").touch()

        child = parent / "child_pkg"
        child.mkdir()
        (child / "__init__.py").touch()

        # Create prompts directories
        parent_prompts = parent / "prompts"
        parent_prompts.mkdir()

        child_prompts = child / "prompts"
        child_prompts.mkdir()

        # Create PromptManager from child directory
        pm = PromptManager(str(child))

        # Should include both child and parent prompts, plus child dir itself
        assert child_prompts in pm.search_paths
        assert child in pm.search_paths
        assert parent_prompts in pm.search_paths

    def test_local_template_loading(self, tmp_path: Path) -> None:
        """Test loading template from same directory as caller."""
        # Create template in same directory
        template_content = "Hello {{ name }}!"
        template_file = tmp_path / "greeting.jinja2"
        template_file.write_text(template_content)

        pm = PromptManager(str(tmp_path))
        result = pm.get("greeting.jinja2", name="World")

        assert result == "Hello World!"

    def test_prompts_dir_template_loading(self, tmp_path: Path) -> None:
        """Test loading template from prompts subdirectory."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        template_file = prompts_dir / "template.jinja2"
        template_file.write_text("Value: {{ value }}")

        pm = PromptManager(str(tmp_path))
        result = pm.get("template.jinja2", value=42)

        assert result == "Value: 42"

    def test_extension_fallback_jinja2(self, tmp_path: Path) -> None:
        """Test automatic .jinja2 extension addition."""
        template_file = tmp_path / "test.jinja2"
        template_file.write_text("Content: {{ content }}")

        pm = PromptManager(str(tmp_path))

        # Should work without extension
        result = pm.get("test", content="data")
        assert result == "Content: data"

    def test_extension_fallback_jinja(self, tmp_path: Path) -> None:
        """Test automatic .jinja extension addition."""
        template_file = tmp_path / "test.jinja"
        template_file.write_text("Content: {{ content }}")

        pm = PromptManager(str(tmp_path))

        # Should work without extension
        result = pm.get("test", content="data")
        assert result == "Content: data"

    def test_template_not_found(self, tmp_path: Path) -> None:
        """Test PromptNotFoundError when template doesn't exist."""
        pm = PromptManager(str(tmp_path))

        with pytest.raises(PromptNotFoundError) as exc_info:
            pm.get("nonexistent.jinja2")

        assert "not found" in str(exc_info.value)
        assert str(tmp_path) in str(exc_info.value)  # Shows search paths

    def test_template_render_error(self, tmp_path: Path) -> None:
        """Test PromptRenderError on template errors."""
        # Template with syntax error (unclosed tag)
        template_file = tmp_path / "bad.jinja2"
        template_file.write_text("{% if true %} missing endif")

        pm = PromptManager(str(tmp_path))

        with pytest.raises(PromptRenderError) as exc_info:
            pm.get("bad.jinja2")

        assert "Template error" in str(exc_info.value)

    def test_template_with_filter_error(self, tmp_path: Path) -> None:
        """Test error with invalid filter."""
        template_file = tmp_path / "filter.jinja2"
        template_file.write_text("{{ value | nonexistent_filter }}")

        pm = PromptManager(str(tmp_path))

        with pytest.raises(PromptRenderError) as exc_info:
            pm.get("filter.jinja2", value="test")

        assert "Template error" in str(exc_info.value)

    def test_directory_precedence(self, tmp_path: Path) -> None:
        """Test that nearer directories take precedence."""
        # Create parent and child directories
        parent = tmp_path / "parent"
        parent.mkdir()
        (parent / "__init__.py").touch()

        child = parent / "child"
        child.mkdir()
        (child / "__init__.py").touch()

        # Create same-named template in both
        parent_prompts = parent / "prompts"
        parent_prompts.mkdir()
        (parent_prompts / "test.jinja2").write_text("Parent")

        child_prompts = child / "prompts"
        child_prompts.mkdir()
        (child_prompts / "test.jinja2").write_text("Child")

        # PromptManager from child should use child's template
        pm = PromptManager(str(child))
        result = pm.get("test.jinja2")

        assert result == "Child"

    def test_complex_template(self, tmp_path: Path) -> None:
        """Test complex template with loops and conditions."""
        template = """
        {% for item in items %}
        {% if item.active %}
        - {{ item.name }}: {{ item.value }}
        {% endif %}
        {% endfor %}
        """

        template_file = tmp_path / "complex.jinja2"
        template_file.write_text(template)

        pm = PromptManager(str(tmp_path))
        result = pm.get(
            "complex.jinja2",
            items=[
                {"name": "Item1", "value": 10, "active": True},
                {"name": "Item2", "value": 20, "active": False},
                {"name": "Item3", "value": 30, "active": True},
            ],
        )

        assert "Item1: 10" in result
        assert "Item2" not in result  # Inactive
        assert "Item3: 30" in result

    def test_template_with_includes(self, tmp_path: Path) -> None:
        """Test template that includes other templates."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        # Base template
        (prompts_dir / "base.jinja2").write_text("Header: {{ title }}")

        # Main template that includes base
        (prompts_dir / "main.jinja2").write_text("{% include 'base.jinja2' %}\nBody: {{ body }}")

        pm = PromptManager(str(tmp_path))
        result = pm.get("main.jinja2", title="Test", body="Content")

        assert "Header: Test" in result
        assert "Body: Content" in result

    def test_no_prompts_dir_fallback(self, tmp_path: Path) -> None:
        """Test that PromptManager works without prompts directory."""
        # No prompts directory, just local templates
        template_file = tmp_path / "local.jinja2"
        template_file.write_text("Local template")

        pm = PromptManager(str(tmp_path))
        result = pm.get("local.jinja2")

        assert result == "Local template"

    def test_max_depth_limit(self, tmp_path: Path) -> None:
        """Test that search stops at max depth."""
        # Create deeply nested structure
        current = tmp_path
        for i in range(6):  # More than max_depth of 4
            current = current / f"level{i}"
            current.mkdir()
            (current / "__init__.py").touch()

        # Put prompts at various levels
        (tmp_path / "prompts").mkdir()
        (tmp_path / "prompts" / "root.jinja2").write_text("Root")

        deep_prompts = current / "prompts"
        deep_prompts.mkdir()
        (deep_prompts / "deep.jinja2").write_text("Deep")

        # PromptManager from deepest should only search up to max_depth
        pm = PromptManager(str(current))

        # Should find deep template
        assert pm.get("deep.jinja2") == "Deep"

        # Should not find root template (beyond max_depth)
        with pytest.raises(PromptNotFoundError):
            pm.get("root.jinja2")

    def test_trim_blocks_and_lstrip(self, tmp_path: Path) -> None:
        """Test that Jinja2 trim_blocks and lstrip_blocks are enabled."""
        template = """
        {% if true %}
            Line 1
            Line 2
        {% endif %}
        """

        template_file = tmp_path / "trim.jinja2"
        template_file.write_text(template)

        pm = PromptManager(str(tmp_path))
        result = pm.get("trim.jinja2")

        # Should have trimmed whitespace
        lines = [line for line in result.split("\n") if line.strip()]
        assert len(lines) == 2
        assert "Line 1" in lines[0]
        assert "Line 2" in lines[1]

    def test_real_template_simple(self):
        """Test loading real simple.j2 template."""
        test_dir = Path(__file__).parent
        pm = PromptManager(str(test_dir))

        result = pm.get("templates/simple.j2", name="Alice")
        assert result == "Hello Alice!"

    def test_real_template_with_context(self):
        """Test loading real with_context.jinja2 template."""
        test_dir = Path(__file__).parent
        pm = PromptManager(str(test_dir))

        result = pm.get(
            "templates/with_context.jinja2",
            role="helpful",
            context=["fact1", "fact2", "fact3"],
            task="summarize the facts",
        )

        assert "You are a helpful assistant" in result
        assert "- fact1" in result
        assert "- fact2" in result
        assert "- fact3" in result
        assert "Task: summarize the facts" in result

    def test_real_template_nested_analysis(self):
        """Test loading real nested analysis template."""
        test_dir = Path(__file__).parent
        pm = PromptManager(str(test_dir))

        result = pm.get(
            "templates/nested/analysis.j2",
            document_name="Report.pdf",
            summary="This is a summary",
            key_points=["Point A", "Point B", "Point C"],
            recommendations=["Rec 1", "Rec 2"],
        )

        assert "## Analysis for Report.pdf" in result
        assert "### Summary" in result
        assert "This is a summary" in result
        assert "1. Point A" in result
        assert "2. Point B" in result
        assert "3. Point C" in result
        assert "- Rec 1" in result
        assert "- Rec 2" in result

    def test_real_template_nested_analysis_partial(self):
        """Test nested analysis template with optional fields."""
        test_dir = Path(__file__).parent
        pm = PromptManager(str(test_dir))

        # Without summary and recommendations
        result = pm.get("templates/nested/analysis.j2", document_name="Data.csv", key_points=["Only key points"])

        assert "## Analysis for Data.csv" in result
        assert "### Summary" not in result
        assert "### Recommendations" not in result
        assert "1. Only key points" in result

    def test_current_date_global(self, tmp_path: Path) -> None:
        """Test that current_date global variable is available in templates."""
        from datetime import datetime

        # Create template that uses current_date
        template_content = "Today is {{ current_date }}"
        template_file = tmp_path / "date_test.jinja2"
        template_file.write_text(template_content)

        pm = PromptManager(str(tmp_path))
        result = pm.get("date_test.jinja2")

        # Check format matches expected "03 January 2025" pattern
        expected_date = datetime.now().strftime("%d %B %Y")
        assert result == f"Today is {expected_date}"

        # Test in combination with other variables
        template_content2 = "{{ greeting }} on {{ current_date }}!"
        template_file2 = tmp_path / "date_test2.jinja2"
        template_file2.write_text(template_content2)

        result = pm.get("date_test2.jinja2", greeting="Hello")
        assert result == f"Hello on {expected_date}!"
