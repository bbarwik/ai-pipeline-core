"""Jinja2-based prompt template management system.

This module provides the PromptManager class for loading and rendering
Jinja2 templates used as prompts for language models. It implements a
smart search strategy that looks for templates in both local and shared
directories.

Search strategy:
    1. Local directory (same as calling module)
    2. Local 'prompts' subdirectory
    3. Parent 'prompts' directories (search ascends parent packages up to the package
       boundary or after 4 parent levels, whichever comes first)

Template organization:
    project/
    ├── my_module.py        # Can use local templates
    ├── analyze.jinja2      # Local template (same directory)
    └── prompts/           # Shared prompts directory
        ├── summarize.jinja2
        └── extract.jinja2

Templates should use .jinja2 or .jinja extension.
The extension can be omitted when calling get().
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2

from ai_pipeline_core.logging import get_pipeline_logger

from .exceptions import PromptError, PromptNotFoundError, PromptRenderError

logger = get_pipeline_logger(__name__)


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
        - Templates are cached by Jinja2 for performance
    """

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
