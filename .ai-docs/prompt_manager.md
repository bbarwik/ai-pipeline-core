# MODULE: prompt_manager
# CLASSES: PromptManager
# SIZE: ~12KB
# === PUBLIC API ===

class PromptManager:
    """Manages Jinja2 prompt templates with smart path resolution.

PromptManager provides a convenient interface for loading and rendering
Jinja2 templates used as prompts for LLMs. It automatically searches for
templates in multiple locations, supporting both local (module-specific)
and shared (project-wide) templates.

Search hierarchy:
    1. Same directory as the calling module (for local templates)
    2. 'prompts' subdirectory in the calling module's directory
    3. 'prompts' directories in parent packages (search ascends parent packages up to the
       package boundary or after 4 parent levels, whichever comes first)

Attributes:
    search_paths: List of directories where templates are searched.
    env: Jinja2 Environment configured for prompt rendering.

Template format:
    Templates use standard Jinja2 syntax:
    ```jinja2
    Analyze the following document:
    {{ document.name }}

    {% if instructions %}
    Instructions: {{ instructions }}
    {% endif %}

    Date: {{ current_date }}  # Current date in format "03 January 2025"
    ```

Autoescape is disabled for prompts (raw text output).
Whitespace control is enabled (trim_blocks, lstrip_blocks).

Template Inheritance:
    Templates support standard Jinja2 inheritance. Templates are searched
    in order of search_paths, so templates in earlier paths override later ones.
    Precedence (first match wins):
    1. Same directory as module
    2. Module's prompts/ subdirectory
    3. Parent prompts/ directories (nearest to farthest)
    - Templates are cached by Jinja2 for performance"""
    def __init__(self, current_file: str, prompts_dir: str = "prompts"):
        """Initialize PromptManager with smart template discovery.

        Sets up the Jinja2 environment with a FileSystemLoader that searches
        multiple directories for templates. The search starts from the calling
        module's location and extends to parent package directories.

        Args:
            current_file: The __file__ path of the calling module. Must be
                         a valid file path (not __name__). Used as the
                         starting point for template discovery.
            prompts_dir: Name of the prompts subdirectory to search for
                        in each package level. Defaults to "prompts".
                        Do not pass prompts_dir='prompts' because it is already the default.

        Raises:
            PromptError: If current_file is not a valid file path (e.g.,
                        if __name__ was passed instead of __file__).

        Search behavior - Given a module at /project/tasks/my_task.py:
            1. /project/tasks/ (local templates)
            2. /project/tasks/prompts/ (if exists)
            3. /project/prompts/ (if /project has __init__.py)
        Search ascends parent packages up to the package boundary or after 4 parent
        levels, whichever comes first.
        """
        search_paths: list[Path] = []

        # Start from the directory containing the calling file
        current_path = Path(current_file).resolve()
        if not current_path.exists():
            raise PromptError(f"PromptManager expected __file__ (a valid file path), but got {current_file!r}. Did you pass __name__ instead?")

        if current_path.is_file():
            current_path = current_path.parent

        # First, add the immediate directory if it has a prompts subdirectory
        local_prompts = current_path / prompts_dir
        if local_prompts.is_dir():
            search_paths.append(local_prompts)

        # Also add the current directory itself for local templates
        search_paths.append(current_path)

        # Search for prompts directory in parent directories
        # Stop when we can't find __init__.py (indicating we've left the package)
        parent_path = current_path.parent
        max_depth = 4  # Reasonable limit to prevent infinite searching
        depth = 0

        while depth < max_depth:
            # Check if we're still within a Python package
            if not (parent_path / "__init__.py").exists():
                break

            # Check if this directory has a prompts subdirectory
            parent_prompts = parent_path / prompts_dir
            if parent_prompts.is_dir():
                search_paths.append(parent_prompts)

            # Move to the next parent
            parent_path = parent_path.parent
            depth += 1

        # If no prompts directories were found, that's okay - we can still use local templates
        if not search_paths:
            search_paths = [current_path]

        self.search_paths = search_paths

        # Create Jinja2 environment with all found search paths
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.search_paths),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,  # Important for prompt engineering
        )

    def get(self, prompt_path: str, **kwargs: Any) -> str:
        """Load and render a Jinja2 template with the given context.

        Searches for the template in all configured search paths and renders
        it with the provided context variables. Automatically tries adding
        .jinja2 or .jinja extensions if the file is not found.

        Args:
            prompt_path: Path to the template file, relative to any search
                        directory. Can be a simple filename ("analyze")
                        or include subdirectories ("tasks/summarize").
                        Extensions (.jinja2, .jinja) are optional.
            **kwargs: Context variables passed to the template. These become
                     available as variables within the Jinja2 template.

        Returns:
            The rendered template as a string, ready to be sent to an LLM.

        Raises:
            PromptNotFoundError: If the template file cannot be found in
                               any search path.
            PromptRenderError: If the template contains errors or if
                              rendering fails (e.g., missing variables,
                              syntax errors).

        Template resolution - Given prompt_path="analyze":
            1. Try "analyze" as-is
            2. Try "analyze.jinja2"
            3. Try "analyze.jinja"
        The first matching file is used.

        Template example:
            ```jinja2
            Summarize the following text in {{ max_length }} words:

            {{ text }}

            {% if style %}
            Style: {{ style }}
            {% endif %}
            ```

        All Jinja2 features are available: loops, conditionals,
        filters, macros, inheritance, etc.
        """
        kwargs.setdefault("current_date", datetime.now().strftime("%d %B %Y"))

        # Build candidate list
        candidates = [prompt_path, *(prompt_path + ext for ext in (".jinja2", ".jinja", ".j2"))]

        # Try each candidate
        for name in candidates:
            try:
                template = self.env.get_template(name)
                return template.render(**kwargs)
            except jinja2.TemplateNotFound:
                continue
            except jinja2.TemplateError as e:
                raise PromptRenderError(f"Template error in '{prompt_path}': {e}") from e
            except (OSError, KeyError, TypeError, AttributeError, ValueError) as e:
                logger.error(f"Unexpected error rendering '{prompt_path}'", exc_info=True)
                raise PromptRenderError(f"Failed to render prompt '{prompt_path}': {e}") from e

        raise PromptNotFoundError(f"Prompt template '{prompt_path}' not found (searched in {self.search_paths}).")


# === EXAMPLES (from tests/) ===

# Example: Real template simple
# Source: tests/prompt_manager/test_prompt_manager.py:251
def test_real_template_simple(self):
    """Test loading real simple.j2 template."""
    test_dir = Path(__file__).parent
    pm = PromptManager(str(test_dir))

    result = pm.get("templates/simple.j2", name="Alice")
    assert result == "Hello Alice!"

# Example: No prompts dir fallback
# Source: tests/prompt_manager/test_prompt_manager.py:192
def test_no_prompts_dir_fallback(self, tmp_path: Path) -> None:
    """Test that PromptManager works without prompts directory."""
    # No prompts directory, just local templates
    template_file = tmp_path / "local.jinja2"
    template_file.write_text("Local template")

    pm = PromptManager(str(tmp_path))
    result = pm.get("local.jinja2")

    assert result == "Local template"

# Example: Directory precedence
# Source: tests/prompt_manager/test_prompt_manager.py:122
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

# Example: Extension fallback jinja
# Source: tests/prompt_manager/test_prompt_manager.py:76
def test_extension_fallback_jinja(self, tmp_path: Path) -> None:
    """Test automatic .jinja extension addition."""
    template_file = tmp_path / "test.jinja"
    template_file.write_text("Content: {{ content }}")

    pm = PromptManager(str(tmp_path))

    # Should work without extension
    result = pm.get("test", content="data")
    assert result == "Content: data"

# === ERROR EXAMPLES (What NOT to Do) ===

# Error: Template not found
# Source: tests/prompt_manager/test_prompt_manager.py:87
def test_template_not_found(self, tmp_path: Path) -> None:
    """Test PromptNotFoundError when template doesn't exist."""
    pm = PromptManager(str(tmp_path))

    with pytest.raises(PromptNotFoundError) as exc_info:
        pm.get("nonexistent.jinja2")

    assert "not found" in str(exc_info.value)
    assert str(tmp_path) in str(exc_info.value)  # Shows search paths

# Error: Max depth limit
# Source: tests/prompt_manager/test_prompt_manager.py:203
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

# Error: Template render error
# Source: tests/prompt_manager/test_prompt_manager.py:97
def test_template_render_error(self, tmp_path: Path) -> None:
    """Test PromptRenderError on template errors."""
    # Template with syntax error (unclosed tag)
    template_file = tmp_path / "bad.jinja2"
    template_file.write_text("{% if true %} missing endif")

    pm = PromptManager(str(tmp_path))

    with pytest.raises(PromptRenderError) as exc_info:
        pm.get("bad.jinja2")

    assert "Template error" in str(exc_info.value)

# Error: Template with filter error
# Source: tests/prompt_manager/test_prompt_manager.py:110
def test_template_with_filter_error(self, tmp_path: Path) -> None:
    """Test error with invalid filter."""
    template_file = tmp_path / "filter.jinja2"
    template_file.write_text("{{ value | nonexistent_filter }}")

    pm = PromptManager(str(tmp_path))

    with pytest.raises(PromptRenderError) as exc_info:
        pm.get("filter.jinja2", value="test")

    assert "Template error" in str(exc_info.value)
