from pathlib import Path
from typing import Any

import jinja2

from ai_pipeline_core.logging import get_pipeline_logger

from .exceptions import PromptError, PromptNotFoundError, PromptRenderError

logger = get_pipeline_logger(__name__)


class PromptManager:
    """A utility to load and render prompts from a structured directory.

    Searches for 'prompts' directory in the current directory and parent directories
    (as long as __init__.py exists in parent directories).
    """

    def __init__(self, current_dir: str, prompts_dir: str = "prompts"):
        """Initialize PromptManager with the current file path.

        Args:
            current_dir: The __file__ path of the calling module (required)
            prompts_dir: Name of the prompts directory to search for (default: "prompts")
        """
        search_paths: list[Path] = []

        # Start from the directory containing the calling file
        current_path = Path(current_dir).resolve()
        if not current_path.exists():
            raise PromptError(
                f"PromptManager expected __file__ (a valid file path), "
                f"but got {current_dir!r}. Did you pass __name__ instead?"
            )

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
        """
        Renders a specific prompt with the given context.

        Args:
            prompt_path: The path to the prompt file relative to the `prompts`
                         directory (e.g., 'step_01_process_inputs/summarize_document.jinja2').
                         The .jinja2 extension will be added automatically if missing.
            **kwargs: Variables to be injected into the template.

        Returns:
            The rendered prompt string.
        """
        try:
            template = self.env.get_template(prompt_path)
            return template.render(**kwargs)
        except jinja2.TemplateNotFound:
            # If the template wasn't found and doesn't end with .jinja2, try adding the extension
            if not prompt_path.endswith(".jinja2"):
                try:
                    template = self.env.get_template(prompt_path + ".jinja2")
                    return template.render(**kwargs)
                except jinja2.TemplateNotFound:
                    pass  # Fall through to the original error
            if not prompt_path.endswith(".jinja"):
                try:
                    template = self.env.get_template(prompt_path + ".jinja")
                    return template.render(**kwargs)
                except jinja2.TemplateNotFound:
                    pass  # Fall through to the original error
            raise PromptNotFoundError(
                f"Prompt template '{prompt_path}' not found (searched in {self.search_paths})."
            )
        except jinja2.TemplateError as e:
            raise PromptRenderError(f"Template error in '{prompt_path}': {e}") from e
        except PromptNotFoundError:
            raise  # Re-raise our custom exception
        except (KeyError, TypeError, AttributeError, IOError, ValueError) as e:
            logger.error(f"Unexpected error rendering '{prompt_path}'", exc_info=True)
            raise PromptRenderError(f"Failed to render prompt '{prompt_path}': {e}") from e
