"""AST-based symbol extraction from Python source files.

Extracts class/function signatures, inheritance chains,
and builds a symbol table for dependency resolution.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MethodInfo:
    """Extracted method/property metadata from a class body."""

    name: str
    signature: str
    docstring: str
    source: str
    is_property: bool
    is_classmethod: bool
    is_abstract: bool
    line_count: int
    is_inherited: bool = False
    inherited_from: str | None = None


@dataclass(frozen=True)
class ClassInfo:
    """Extracted class metadata including methods, validators, and class variables."""

    name: str
    bases: tuple[str, ...]
    docstring: str
    is_public: bool
    class_vars: tuple[tuple[str, str, str], ...]  # (name, type_annotation, default_value)
    methods: tuple[MethodInfo, ...]
    validators: tuple[MethodInfo, ...]
    module_path: str
    decorators: tuple[str, ...] = ()


@dataclass(frozen=True)
class FunctionInfo:
    """Extracted module-level function metadata."""

    name: str
    signature: str
    docstring: str
    source: str
    is_public: bool
    is_async: bool
    line_count: int
    module_path: str


@dataclass(frozen=True)
class ModuleInfo:
    """Parsed module containing its classes and functions."""

    name: str
    path: Path
    docstring: str
    is_public: bool
    classes: tuple[ClassInfo, ...]
    functions: tuple[FunctionInfo, ...]


@dataclass
class SymbolTable:
    """Mutable during construction, used read-only after building.

    Maps class and function names to their ClassInfo/FunctionInfo objects,
    and provides class_to_module/function_to_module lookups for dependency resolution.
    """

    classes: dict[str, ClassInfo] = field(default_factory=dict)
    functions: dict[str, FunctionInfo] = field(default_factory=dict)
    class_to_module: dict[str, str] = field(default_factory=dict)
    function_to_module: dict[str, str] = field(default_factory=dict)


# Known external base classes that get stub representations
EXTERNAL_STUBS: dict[str, str] = {
    "BaseModel": "Pydantic base model. Fields are typed class attributes.",
    "BaseSettings": "Pydantic settings model. Loads values from environment variables.",
    "ABC": "Python abstract base class marker.",
    "Generic": "Python generic base class for parameterized types.",
    "list": "Python built-in list.",
    "dict": "Python built-in dictionary.",
    "StrEnum": "String enumeration base class.",
}


def is_public_name(name: str) -> bool:
    """Determine if a symbol is public based on Python naming convention."""
    if name.startswith("__") and name.endswith("__"):
        return True
    return not name.startswith("_")


def parse_module(path: Path) -> ModuleInfo:
    """Parse a single .py file and return all extracted symbols."""
    source = path.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    tree = ast.parse(source)

    module_doc = ast.get_docstring(tree) or ""
    module_path = _module_path(path)

    classes: list[ClassInfo] = []
    functions: list[FunctionInfo] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(_extract_class(node, source_lines, module_path))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_extract_function(node, source_lines, module_path))

    module_public = any(c.is_public for c in classes) or any(f.is_public for f in functions)

    return ModuleInfo(
        name=path.stem,
        path=path,
        docstring=module_doc,
        is_public=module_public,
        classes=tuple(classes),
        functions=tuple(functions),
    )


def build_symbol_table(source_dir: Path) -> SymbolTable:
    """Parse all .py files under source_dir and build a unified symbol table."""
    table = SymbolTable()

    for py_file in sorted(source_dir.rglob("*.py")):
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue
        module = parse_module(py_file)

        relative = py_file.relative_to(source_dir)
        if len(relative.parts) > 1:
            package_name = relative.parts[0]
        else:
            package_name = relative.stem

        for cls in module.classes:
            table.classes[cls.name] = cls
            table.class_to_module[cls.name] = package_name
        for func in module.functions:
            table.functions[func.name] = func
            table.function_to_module[func.name] = package_name

    return table


def resolve_dependencies(
    root_classes: list[str],
    table: SymbolTable,
) -> tuple[list[ClassInfo], set[str]]:
    """Resolve transitive dependencies for a set of root classes.

    Returns (resolved ClassInfo list in topological order, external base names).
    """
    resolved: list[ClassInfo] = []
    external_bases: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)

        if name in EXTERNAL_STUBS or name not in table.classes:
            external_bases.add(name)
            return

        cls = table.classes[name]
        for base in cls.bases:
            visit(base.split("[")[0])

        resolved.append(cls)

    for root in root_classes:
        visit(root)

    return resolved, external_bases


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _module_path(path: Path) -> str:
    """Convert filesystem path to dotted module path.

    e.g. ai_pipeline_core/documents/document.py -> ai_pipeline_core.documents.document
    """
    parts = list(path.with_suffix("").parts)
    # Find the package root (ai_pipeline_core)
    for i, part in enumerate(parts):
        if part == "ai_pipeline_core":
            return ".".join(parts[i:])
    return ".".join(parts)


def _decorator_name(decorator: ast.expr) -> str:
    if isinstance(decorator, ast.Call):
        return _decorator_name(decorator.func)
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    if isinstance(decorator, ast.Name):
        return decorator.id
    return ""


def _body_line_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    if not node.body:
        return 0
    first = node.body[0]
    body_nodes = node.body
    is_docstring = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
    if is_docstring:
        body_nodes = node.body[1:]
    if not body_nodes:
        return 0
    start = body_nodes[0].lineno
    end = body_nodes[-1].end_lineno or body_nodes[-1].lineno
    return end - start + 1


def _extract_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args_str = ast.unparse(node.args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"({args_str}){ret}"


def _get_source(source_lines: list[str], node: ast.AST) -> str:
    decoratable = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    if isinstance(node, decoratable) and node.decorator_list:
        start = node.decorator_list[0].lineno - 1
    else:
        start: int = node.lineno - 1  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    end: int = node.end_lineno or node.lineno  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    return "\n".join(source_lines[start:end])


def _is_validator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    validator_names = ("field_validator", "model_validator")
    return any(_decorator_name(d) in validator_names for d in node.decorator_list)


def _extract_method(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: list[str],
) -> MethodInfo:
    decorator_names = {_decorator_name(d) for d in node.decorator_list}
    return MethodInfo(
        name=node.name,
        signature=_extract_signature(node),
        docstring=ast.get_docstring(node) or "",
        source=_get_source(source_lines, node),
        is_property="property" in decorator_names,
        is_classmethod="classmethod" in decorator_names,
        is_abstract="abstractmethod" in decorator_names,
        line_count=_body_line_count(node),
    )


def _extract_class(node: ast.ClassDef, source_lines: list[str], module_path: str) -> ClassInfo:
    docstring = ast.get_docstring(node) or ""
    bases = [ast.unparse(base) for base in node.bases]

    methods: list[MethodInfo] = []
    validators: list[MethodInfo] = []
    class_vars: list[tuple[str, str, str]] = []

    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method = _extract_method(item, source_lines)
            methods.append(method)
            if _is_validator(item):
                validators.append(method)
        elif isinstance(item, ast.AnnAssign) and item.target and isinstance(item.target, ast.Name):
            name = item.target.id
            if is_public_name(name):
                type_ann = ast.unparse(item.annotation) if item.annotation else ""
                default = ast.unparse(item.value) if item.value else ""
                class_vars.append((name, type_ann, default))
        elif isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
            name = item.targets[0].id
            if is_public_name(name):
                default = ast.unparse(item.value)
                class_vars.append((name, "", default))

    return ClassInfo(
        name=node.name,
        bases=tuple(bases),
        docstring=docstring,
        is_public=is_public_name(node.name),
        class_vars=tuple(class_vars),
        methods=tuple(methods),
        validators=tuple(validators),
        module_path=module_path,
        decorators=tuple(ast.unparse(d) for d in node.decorator_list),
    )


def _extract_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: list[str],
    module_path: str,
) -> FunctionInfo:
    return FunctionInfo(
        name=node.name,
        signature=_extract_signature(node),
        docstring=ast.get_docstring(node) or "",
        source=_get_source(source_lines, node),
        is_public=is_public_name(node.name),
        is_async=isinstance(node, ast.AsyncFunctionDef),
        line_count=_body_line_count(node),
        module_path=module_path,
    )
