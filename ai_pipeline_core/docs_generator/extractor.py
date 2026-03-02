"""AST-based symbol extraction from Python source files.

Extracts class/function signatures, inheritance chains,
and builds a symbol table for dependency resolution.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "EXTERNAL_STUBS",
    "ClassInfo",
    "FunctionInfo",
    "MethodInfo",
    "ModuleInfo",
    "SymbolTable",
    "ValueInfo",
    "build_symbol_table",
    "format_class_field",
    "get_source",
    "is_public_name",
    "parse_module",
    "resolve_dependencies",
    "unpack_class_field",
]


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class ClassInfo:
    """Extracted class metadata including methods, validators, and class variables."""

    name: str
    bases: tuple[str, ...]
    docstring: str
    is_public: bool
    class_vars: tuple[tuple[str, str, str, str], ...]  # (name, type_annotation, default_value, description)
    methods: tuple[MethodInfo, ...]
    validators: tuple[MethodInfo, ...]
    module_path: str
    decorators: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class ValueInfo:
    """Extracted module-level value: NewType, type alias, or constant."""

    name: str
    source: str
    kind: str  # "NewType", "TypeAlias", "Constant"
    is_public: bool
    module_path: str


@dataclass(frozen=True, slots=True)
class ModuleInfo:
    """Parsed module containing its classes and functions."""

    name: str
    path: Path
    docstring: str
    is_public: bool
    classes: tuple[ClassInfo, ...]
    functions: tuple[FunctionInfo, ...]
    values: tuple[ValueInfo, ...] = ()


@dataclass
class SymbolTable:
    """Mutable during construction, used read-only after building.

    Maps class and function names to their ClassInfo/FunctionInfo objects,
    and provides class_to_module/function_to_module lookups for dependency resolution.
    """

    classes: dict[str, ClassInfo] = field(default_factory=dict)
    functions: dict[str, FunctionInfo] = field(default_factory=dict)
    values: dict[str, ValueInfo] = field(default_factory=dict)
    class_to_module: dict[str, str] = field(default_factory=dict)
    function_to_module: dict[str, str] = field(default_factory=dict)
    value_to_module: dict[str, str] = field(default_factory=dict)


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
    values: list[ValueInfo] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(_extract_class(node, source_lines, module_path))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_extract_function(node, source_lines, module_path))
        elif value := _extract_value(node, source_lines, module_path):
            values.append(value)

    module_public = any(c.is_public for c in classes) or any(f.is_public for f in functions) or any(v.is_public for v in values)

    return ModuleInfo(
        name=path.stem,
        path=path,
        docstring=module_doc,
        is_public=module_public,
        classes=tuple(classes),
        functions=tuple(functions),
        values=tuple(values),
    )


def _dedup_key(name: str, package_name: str, symbols: dict[str, Any], module_map: dict[str, str]) -> str:
    """Return a unique dict key for a symbol, qualifying with module name on collision.

    When two modules define the same public symbol (e.g. ``main`` in multiple
    cli.py files), the first is re-keyed as ``existing_module:name`` and the
    new one is stored as ``package_name:name``.
    """
    if name not in symbols:
        return name
    existing_module = module_map.get(name, "")
    if existing_module == package_name:
        return name
    # First collision: re-key the existing entry
    qualified_existing = f"{existing_module}:{name}"
    if qualified_existing not in symbols:
        symbols[qualified_existing] = symbols.pop(name)
        module_map[qualified_existing] = module_map.pop(name)
    return f"{package_name}:{name}"


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
            key = _dedup_key(cls.name, package_name, table.classes, table.class_to_module)
            table.classes[key] = cls
            table.class_to_module[key] = package_name
        for func in module.functions:
            key = _dedup_key(func.name, package_name, table.functions, table.function_to_module)
            table.functions[key] = func
            table.function_to_module[key] = package_name
        for val in module.values:
            key = _dedup_key(val.name, package_name, table.values, table.value_to_module)
            table.values[key] = val
            table.value_to_module[key] = package_name

    _remap_private_symbols(table, source_dir)
    return table


def _remap_private_symbols(table: SymbolTable, source_dir: Path) -> None:  # noqa: C901, PLR0912
    """Remap symbols defined in private modules to public modules that re-export them via __all__.

    Also discovers symbols from private files (e.g. _types.py) that are
    re-exported via a public module's __all__ but were skipped during initial parsing.
    """
    if not source_dir.is_dir():
        return
    public_exports: dict[str, set[str]] = {}
    for subdir in sorted(source_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue
        init_file = subdir / "__init__.py"
        if not init_file.exists():
            continue
        all_names = _parse_dunder_all(init_file)
        if all_names:
            public_exports[subdir.name] = all_names

    # Remap already-discovered symbols from private to public modules
    for symbol_map in (table.class_to_module, table.function_to_module, table.value_to_module):
        for name, current_module in list(symbol_map.items()):
            if not current_module.startswith("_"):
                continue
            for public_module, exports in public_exports.items():
                if name in exports:
                    symbol_map[name] = public_module
                    break

    # Discover symbols from private files that are re-exported via __all__
    # but were skipped during initial parsing (build_symbol_table skips _-prefixed files).
    # Include bare names from qualified keys ("module:Name" → "Name") so that
    # cross-module collisions handled by _dedup_key are not re-discovered as "missing".
    known_symbols: set[str] = set()
    for key in (*table.classes, *table.functions, *table.values):
        known_symbols.add(key)
        if ":" in key:
            known_symbols.add(key.split(":", 1)[1])
    for public_module, exports in public_exports.items():
        missing = exports - known_symbols
        if not missing:
            continue
        subdir = source_dir / public_module
        for py_file in sorted(subdir.rglob("*.py")):
            if py_file.name == "__init__.py":
                continue
            module = parse_module(py_file)
            for cls in module.classes:
                if cls.name in missing:
                    table.classes[cls.name] = cls
                    table.class_to_module[cls.name] = public_module
            for func in module.functions:
                if func.name in missing:
                    table.functions[func.name] = func
                    table.function_to_module[func.name] = public_module
            for val in module.values:
                if val.name in missing:
                    table.values[val.name] = val
                    table.value_to_module[val.name] = public_module

    # Discover _CapitalizedName classes from private files so _collect_internal_types
    # can resolve them when they appear in public API field annotations or signatures.
    for subdir in sorted(source_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue
        for py_file in sorted(subdir.rglob("*.py")):
            if py_file.name == "__init__.py" or not py_file.name.startswith("_"):
                continue
            module = parse_module(py_file)
            for cls in module.classes:
                if cls.name not in table.classes and cls.name.startswith("_") and len(cls.name) > 1 and cls.name[1].isupper():
                    table.classes[cls.name] = cls
                    table.class_to_module[cls.name] = subdir.name


def _parse_dunder_all(init_file: Path) -> set[str]:
    """Extract __all__ symbol names from an __init__.py file."""
    tree = ast.parse(init_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__" and isinstance(node.value, ast.List | ast.Tuple):
                return {elt.value for elt in node.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)}
    return set()


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


def get_source(source_lines: list[str], node: ast.AST) -> str:
    """Extract source code text for an AST node, including decorators."""
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
        source=get_source(source_lines, node),
        is_property="property" in decorator_names,
        is_classmethod="classmethod" in decorator_names,
        is_abstract="abstractmethod" in decorator_names,
        line_count=_body_line_count(node),
    )


def _extract_value(node: ast.stmt, source_lines: list[str], module_path: str) -> ValueInfo | None:
    """Extract module-level NewType, type alias, or UPPER_CASE constant."""
    # NewType("Name", base) — ast.Assign with Call to NewType
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        if isinstance(node.value, ast.Call) and _is_newtype_call(node.value):
            return ValueInfo(name=name, source=get_source(source_lines, node), kind="NewType", is_public=is_public_name(name), module_path=module_path)
        if name.isupper() and is_public_name(name) and len(name) > 1:
            return ValueInfo(name=name, source=get_source(source_lines, node), kind="Constant", is_public=True, module_path=module_path)
        if is_public_name(name) and name[0].isupper() and _looks_like_type_alias(node.value):
            return ValueInfo(name=name, source=get_source(source_lines, node), kind="TypeAlias", is_public=True, module_path=module_path)
    # PEP 695 type alias: type Name = ...
    if isinstance(node, ast.TypeAlias):
        name = node.name.id
        return ValueInfo(name=name, source=get_source(source_lines, node), kind="TypeAlias", is_public=is_public_name(name), module_path=module_path)
    # Annotated type alias: Name: TypeAlias = ...
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
        ann = ast.unparse(node.annotation)
        if "TypeAlias" in ann:
            name = node.target.id
            return ValueInfo(name=name, source=get_source(source_lines, node), kind="TypeAlias", is_public=is_public_name(name), module_path=module_path)
    return None


def _looks_like_type_alias(node: ast.expr) -> bool:
    """Check if an expression looks like a type alias (Subscript or union via |)."""
    return isinstance(node, ast.Subscript) or (isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr))


def _is_newtype_call(node: ast.Call) -> bool:
    """Check if a Call node is NewType(...)."""
    return (isinstance(node.func, ast.Name) and node.func.id == "NewType") or (isinstance(node.func, ast.Attribute) and node.func.attr == "NewType")


def _extract_class(node: ast.ClassDef, source_lines: list[str], module_path: str) -> ClassInfo:
    docstring = ast.get_docstring(node) or ""
    bases = [ast.unparse(base) for base in node.bases]

    methods: list[MethodInfo] = []
    validators: list[MethodInfo] = []
    class_vars: list[tuple[str, str, str, str]] = []

    for index, item in enumerate(node.body):
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
                description = _extract_class_var_description(item, index, node.body, source_lines)
                class_vars.append((name, type_ann, default, description))
        elif isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
            name = item.targets[0].id
            if is_public_name(name):
                default = ast.unparse(item.value)
                description = _extract_class_var_description(item, index, node.body, source_lines)
                class_vars.append((name, "", default, description))

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


def _extract_class_var_description(
    item: ast.AnnAssign | ast.Assign,
    index: int,
    class_body: list[ast.stmt],
    source_lines: list[str],
) -> str:
    """Extract one-line class field description from supported sources."""
    field_description = _extract_field_description(item)
    if field_description:
        return field_description

    inline_comment = _extract_inline_comment(item, source_lines)
    if inline_comment:
        return inline_comment

    docstring_below = _extract_docstring_below(index, class_body)
    if docstring_below:
        return docstring_below

    return ""


def _extract_field_description(item: ast.AnnAssign | ast.Assign) -> str:
    """Extract Pydantic Field(description=...) text from a class field default."""
    value = item.value
    if not isinstance(value, ast.Call):
        return ""
    if not _is_field_call(value):
        return ""

    for keyword in value.keywords:
        if keyword.arg != "description":
            continue
        literal = _as_string_literal(keyword.value)
        if literal:
            return _first_line(literal)
    return ""


def _is_field_call(node: ast.Call) -> bool:
    """Return True when node is a call to Field(...)."""
    if isinstance(node.func, ast.Name):
        return node.func.id == "Field"
    if isinstance(node.func, ast.Attribute):
        return node.func.attr == "Field"
    return False


def _as_string_literal(node: ast.expr) -> str:
    """Convert an AST expression to a plain string when possible."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _first_line(text: str) -> str:
    """Return the first non-empty line from text, stripped."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _extract_inline_comment(item: ast.AnnAssign | ast.Assign, source_lines: list[str]) -> str:
    """Extract inline # comment on any line of the field declaration."""
    start = item.lineno
    end = item.end_lineno or item.lineno
    for line_no in range(start, end + 1):
        line = source_lines[line_no - 1]
        hash_index = line.find("#")
        if hash_index < 0:
            continue
        comment = line[hash_index + 1 :].strip()
        if comment:
            return _first_line(comment)
    return ""


def _extract_docstring_below(index: int, class_body: list[ast.stmt]) -> str:
    """Extract first line from an immediate docstring-style literal below a field."""
    next_index = index + 1
    if next_index >= len(class_body):
        return ""
    next_item = class_body[next_index]
    if not isinstance(next_item, ast.Expr):
        return ""
    if not isinstance(next_item.value, ast.Constant):
        return ""
    if not isinstance(next_item.value.value, str):
        return ""
    return _first_line(next_item.value.value)


def _extract_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: list[str],
    module_path: str,
) -> FunctionInfo:
    return FunctionInfo(
        name=node.name,
        signature=_extract_signature(node),
        docstring=ast.get_docstring(node) or "",
        source=get_source(source_lines, node),
        is_public=is_public_name(node.name),
        is_async=isinstance(node, ast.AsyncFunctionDef),
        line_count=_body_line_count(node),
        module_path=module_path,
    )


# ---------------------------------------------------------------------------
# Class field formatting helpers
# ---------------------------------------------------------------------------


def unpack_class_field(field: tuple[str, ...]) -> tuple[str, str, str, str]:
    """Unpack class field tuple and support legacy 3-item tuples."""
    var_name = field[0] if len(field) > 0 else ""
    type_ann = field[1] if len(field) > 1 else ""
    default = field[2] if len(field) > 2 else ""
    description = field[3] if len(field) > 3 else ""
    return var_name, type_ann, default, description


def format_class_field(var_name: str, type_ann: str, default: str, description: str) -> str:
    """Render one class field declaration with optional inline description."""
    if type_ann and default:
        line = f"    {var_name}: {type_ann} = {default}"
    elif type_ann:
        line = f"    {var_name}: {type_ann}"
    elif default:
        line = f"    {var_name} = {default}"
    else:
        line = f"    {var_name}"
    if description:
        return f"{line}  # {description}"
    return line
