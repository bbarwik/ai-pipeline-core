"""File-level structural rules for pipeline application code.

Enforces at import time:
- One PipelineFlow per file
- One PipelineTask per file
- Flows and tasks not in same file
- Flows and specs not in same file
- Tasks and specs not in same file
- PromptSpec co-location (follows= chains)
- Non-empty docstrings on flows, tasks, specs

Called from __init_subclass__ of PipelineTask, PipelineFlow, and PromptSpec.
Framework-internal code, tests, and examples are exempt.
"""

import re

__all__ = [
    "is_exempt",
    "register_flow",
    "register_spec",
    "register_task",
    "require_docstring",
    "reset_registries",
]

# module_name -> class_name
_flows: dict[str, str] = {}  # nosemgrep: no-mutable-module-globals — import-time registry, reset via reset_registries()
_tasks: dict[str, str] = {}  # nosemgrep: no-mutable-module-globals — import-time registry, reset via reset_registries()
# module_name -> list of (class_name, follows_class_name | None)
_specs: dict[str, list[tuple[str, str | None]]] = {}  # nosemgrep: no-mutable-module-globals — import-time registry, reset via reset_registries()
# module_name -> class_name for specs that follow a spec from a different module
_cross_file_follows: dict[str, str] = {}  # nosemgrep: no-mutable-module-globals — import-time registry, reset via reset_registries()

_SNAKE_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _snake(name: str) -> str:
    """PascalCase to snake_case for path suggestions in error messages."""
    return _SNAKE_RE.sub("_", name).lower()


def is_exempt(cls: type) -> bool:
    """Framework internals, tests, examples, __main__, and function-local classes are exempt."""
    module = getattr(cls, "__module__", "") or ""

    # Framework-internal code
    if module.startswith("ai_pipeline_core."):
        return True

    # Scripts run directly
    if module == "__main__":
        return True

    # Function-local classes (also caught by existing semgrep rule)
    qualname = getattr(cls, "__qualname__", "")
    if "<locals>" in qualname:
        return True

    # Test modules and examples
    parts = module.split(".")
    return any(part in {"tests", "conftest", "examples"} or part.startswith("test_") for part in parts)


def require_docstring(cls: type, *, kind: str) -> None:
    """Require a non-empty docstring on application flows, tasks, and specs."""
    if cls.__doc__ and cls.__doc__.strip():
        return
    raise TypeError(
        f"{kind} '{cls.__name__}' must have a docstring explaining its purpose, "
        f"expected input, and expected output.\n"
        f"FIX: Add a docstring:\n"
        f"    class {cls.__name__}({kind}):\n"
        f'        """<Purpose of this {kind.lower()}>.\n'
        f"\n"
        f"        Input: <what documents/data this receives>.\n"
        f"        Output: <what documents/data this produces>.\n"
        f'        """'
    )


def register_flow(cls: type) -> None:
    """Enforce: one flow per file, no tasks or specs in same file."""
    module = cls.__module__

    # One flow per file
    existing = _flows.get(module)
    if existing is not None and existing != cls.__name__:
        raise TypeError(
            f"Module '{module}' defines PipelineFlow '{cls.__name__}' but already contains "
            f"PipelineFlow '{existing}'. Each file may contain at most one PipelineFlow.\n"
            f"FIX: Move '{cls.__name__}' to its own file:\n"
            f"  flows/{_snake(cls.__name__)}/{_snake(cls.__name__)}.py"
        )

    # No tasks in same file
    task = _tasks.get(module)
    if task is not None:
        raise TypeError(
            f"Module '{module}' defines PipelineFlow '{cls.__name__}' but already contains "
            f"PipelineTask '{task}'. Flows and tasks must be in separate files.\n"
            f"FIX: Keep the flow in flows/{_snake(cls.__name__)}/{_snake(cls.__name__)}.py\n"
            f"     Move the task to flows/<flow>/tasks/{_snake(task)}/{_snake(task)}.py"
        )

    # No specs in same file
    specs = _specs.get(module)
    if specs:
        spec_names = [name for name, _ in specs]
        raise TypeError(
            f"Module '{module}' defines PipelineFlow '{cls.__name__}' but already contains "
            f"PromptSpec(s): {', '.join(spec_names)}. Flows and specs must be in separate files.\n"
            f"FIX: Move specs to flows/<flow>/tasks/<task>/specs/<category>.py"
        )

    _flows[module] = cls.__name__


def register_task(cls: type) -> None:
    """Enforce: one task per file, no flows or specs in same file."""
    module = cls.__module__

    # One task per file
    existing = _tasks.get(module)
    if existing is not None and existing != cls.__name__:
        raise TypeError(
            f"Module '{module}' defines PipelineTask '{cls.__name__}' but already contains "
            f"PipelineTask '{existing}'. Each file may contain at most one PipelineTask.\n"
            f"FIX: Move '{cls.__name__}' to its own file:\n"
            f"  flows/<flow>/tasks/{_snake(cls.__name__)}/{_snake(cls.__name__)}.py"
        )

    # No flows in same file
    flow = _flows.get(module)
    if flow is not None:
        raise TypeError(
            f"Module '{module}' defines PipelineTask '{cls.__name__}' but already contains "
            f"PipelineFlow '{flow}'. Flows and tasks must be in separate files.\n"
            f"FIX: Keep the task in flows/<flow>/tasks/{_snake(cls.__name__)}/{_snake(cls.__name__)}.py\n"
            f"     Keep the flow in flows/{_snake(flow)}/{_snake(flow)}.py"
        )

    # No specs in same file
    specs = _specs.get(module)
    if specs:
        spec_names = [name for name, _ in specs]
        raise TypeError(
            f"Module '{module}' defines PipelineTask '{cls.__name__}' but already contains "
            f"PromptSpec(s): {', '.join(spec_names)}. Tasks and specs must be in separate files.\n"
            f"FIX: Move specs to flows/<flow>/tasks/<task>/specs/<category>.py"
        )

    _tasks[module] = cls.__name__


def register_spec(cls: type, follows: type | None) -> None:
    """Enforce PromptSpec co-location rules.

    - No flows or tasks in same file
    - At most one standalone spec (follows=None) per file
    - A follows= spec targeting another file must be alone in its file
    - A follows= spec targeting the same file is allowed (co-located chain)
    """
    module = cls.__module__
    follows_name = follows.__name__ if follows is not None else None
    follows_module = getattr(follows, "__module__", None) if follows is not None else None

    # No flows in same file
    flow = _flows.get(module)
    if flow is not None:
        raise TypeError(
            f"Module '{module}' defines PromptSpec '{cls.__name__}' but already contains "
            f"PipelineFlow '{flow}'. Specs and flows must be in separate files.\n"
            f"FIX: Move specs to flows/<flow>/tasks/<task>/specs/<category>.py"
        )

    # No tasks in same file
    task = _tasks.get(module)
    if task is not None:
        raise TypeError(
            f"Module '{module}' defines PromptSpec '{cls.__name__}' but already contains "
            f"PipelineTask '{task}'. Specs and tasks must be in separate files.\n"
            f"FIX: Move specs to flows/<flow>/tasks/<task>/specs/<category>.py"
        )

    entries = _specs.setdefault(module, [])

    # Check if a previously registered cross-file follow-up requires isolation
    existing_cross_file = _cross_file_follows.get(module)
    if existing_cross_file is not None:
        raise TypeError(
            f"Module '{module}' already contains PromptSpec '{existing_cross_file}' which follows "
            f"a spec from a different file and must be the only spec in its file. "
            f"Cannot add '{cls.__name__}' to the same module.\n"
            f"FIX: Move '{existing_cross_file}' to its own file, or move '{cls.__name__}' elsewhere."
        )

    if follows is not None and follows_module != module:
        # Cross-file follows: must be alone in its file
        if entries:
            existing_names = [name for name, _ in entries]
            raise TypeError(
                f"PromptSpec '{cls.__name__}' follows '{follows_name}' from module "
                f"'{follows_module}', but module '{module}' already contains: "
                f"{', '.join(existing_names)}. A spec that follows a spec from a different "
                f"file must be the only spec in its file.\n"
                f"FIX: Move '{cls.__name__}' to its own file in the specs/ directory."
            )
    elif follows is None:
        # Standalone spec: at most one per file
        existing_standalone = [name for name, f in entries if f is None]
        if existing_standalone:
            raise TypeError(
                f"Module '{module}' defines standalone PromptSpec '{cls.__name__}' but already "
                f"contains standalone PromptSpec '{existing_standalone[0]}'. Each file may "
                f"contain at most one standalone spec (without follows=). Additional specs "
                f"must use follows= to chain from a spec defined in the same file.\n"
                f"FIX: Move '{cls.__name__}' to its own file:\n"
                f"  flows/<flow>/tasks/<task>/specs/{_snake(cls.__name__)}.py"
            )
    # else: follows a spec in the same file — always allowed

    entries.append((cls.__name__, follows_name))

    # Track cross-file follow-ups for reverse isolation check
    if follows is not None and follows_module != module:
        _cross_file_follows[module] = cls.__name__


def reset_registries() -> None:
    """Clear all registries. For framework tests only."""
    _flows.clear()
    _tasks.clear()
    _specs.clear()
    _cross_file_follows.clear()
