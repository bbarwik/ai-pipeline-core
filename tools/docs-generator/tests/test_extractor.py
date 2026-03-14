from docs_generator.extractor import (
    EXTERNAL_STUBS,
    ClassInfo,
    SymbolTable,
    build_symbol_table,
    is_public_name,
    parse_module,
    resolve_dependencies,
)


def _write_py(path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# ---------------------------------------------------------------------------
# is_public_name
# ---------------------------------------------------------------------------


def test_is_public_name_regular():
    assert is_public_name("Foo") is True
    assert is_public_name("generate") is True


def test_is_public_name_underscore_prefix():
    assert is_public_name("_private") is False
    assert is_public_name("_InternalHelper") is False


def test_is_public_name_dunder():
    assert is_public_name("__init__") is True
    assert is_public_name("__init_subclass__") is True


def test_is_public_name_mangled():
    assert is_public_name("__mangled") is False


# ---------------------------------------------------------------------------
# parse_module — classes
# ---------------------------------------------------------------------------


def test_parse_module_finds_public_class(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('class Foo:\n    """A class."""\n    pass\n')
    module = parse_module(src)
    assert len(module.classes) == 1
    assert module.classes[0].name == "Foo"
    assert module.classes[0].is_public is True


def test_parse_module_non_public_class(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('class _Bar:\n    """Internal."""\n    pass\n')
    module = parse_module(src)
    assert module.classes[0].is_public is False


def test_parse_module_extracts_bases(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('class Child(Parent, Mixin):\n    """A child class."""\n    pass\n')
    module = parse_module(src)
    assert module.classes[0].bases == ("Parent", "Mixin")


def test_parse_module_extracts_docstring(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('class Doc:\n    """Detailed docs."""\n    pass\n')
    module = parse_module(src)
    assert "Detailed docs" in module.classes[0].docstring


def test_parse_module_extracts_class_variables(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('class Cfg:\n    """Config class."""\n    name: str = "default"\n    count: int\n')
    module = parse_module(src)
    cls = module.classes[0]
    assert len(cls.class_vars) == 2
    assert cls.class_vars[0] == ("name", "str", "'default'", "")
    assert cls.class_vars[1] == ("count", "int", "", "")


def test_parse_module_extracts_method_info(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('class Svc:\n    """Service class."""\n    def process(self, data: str) -> bool:\n        """Process data."""\n        return True\n')
    module = parse_module(src)
    method = module.classes[0].methods[0]
    assert method.name == "process"
    assert "data: str" in method.signature
    assert method.docstring == "Process data."
    assert method.line_count == 1


def test_parse_module_method_decorators(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text(
        "class Svc:\n"
        '    """Service class."""\n'
        "    @property\n"
        "    def name(self) -> str:\n"
        '        return "x"\n'
        "    @classmethod\n"
        '    def create(cls) -> "Svc":\n'
        "        return cls()\n"
        "    @abstractmethod\n"
        "    def run(self) -> None:\n"
        "        pass\n"
    )
    module = parse_module(src)
    methods = {m.name: m for m in module.classes[0].methods}
    assert methods["name"].is_property is True
    assert methods["create"].is_classmethod is True
    assert methods["run"].is_abstract is True


# ---------------------------------------------------------------------------
# parse_module — functions
# ---------------------------------------------------------------------------


def test_parse_module_finds_public_sync_function(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('def helper(x: int) -> int:\n    """Compute helper."""\n    return x + 1\n')
    module = parse_module(src)
    func = module.functions[0]
    assert func.name == "helper"
    assert func.is_public is True
    assert func.is_async is False


def test_parse_module_finds_public_async_function(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('async def fetch(url: str) -> str:\n    """Fetch data."""\n    return ""\n')
    module = parse_module(src)
    func = module.functions[0]
    assert func.name == "fetch"
    assert func.is_async is True


def test_parse_module_non_public_function(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('def _private():\n    """Internal."""\n    pass\n')
    module = parse_module(src)
    assert module.functions[0].is_public is False


def test_parse_module_function_signature_and_line_count(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('def compute(a: int, b: int) -> int:\n    """Compute sum."""\n    x = a + b\n    return x\n')
    module = parse_module(src)
    func = module.functions[0]
    assert "a: int" in func.signature
    assert "-> int" in func.signature
    assert func.line_count == 2


# ---------------------------------------------------------------------------
# parse_module — module level
# ---------------------------------------------------------------------------


def test_parse_module_public_when_has_public_class(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('"""Module doc."""\nclass Foo:\n    pass\n')
    module = parse_module(src)
    assert "Module doc" in module.docstring
    assert module.is_public is True


def test_parse_module_detects_module_not_public(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('"""Just a module."""\nx = 1\n')
    module = parse_module(src)
    assert module.is_public is False


def test_parse_module_not_public_when_only_private_members(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('"""A module."""\ndef _internal():\n    pass\n')
    module = parse_module(src)
    assert module.is_public is False


def test_parse_module_not_public_when_no_members(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('"""A module."""\nx = 1\n')
    module = parse_module(src)
    assert module.is_public is False


def test_validators_in_validators_tuple(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text(
        "class M:\n"
        '    """Model class."""\n'
        '    @model_validator(mode="before")\n'
        "    @classmethod\n"
        "    def check(cls, v):\n"
        "        return v\n"
        "    def normal(self):\n"
        "        pass\n"
    )
    cls = parse_module(src).classes[0]
    assert len(cls.validators) == 1
    assert cls.validators[0].name == "check"
    assert len(cls.methods) == 2


# ---------------------------------------------------------------------------
# build_symbol_table
# ---------------------------------------------------------------------------


def test_build_symbol_table_populates_classes(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write_py(pkg / "mod.py", 'class Foo:\n    """A class."""\n    pass\n')
    table = build_symbol_table(tmp_path)
    assert "Foo" in table.classes


def test_build_symbol_table_populates_functions(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write_py(pkg / "mod.py", 'def helper() -> int:\n    """A helper."""\n    return 1\n')
    table = build_symbol_table(tmp_path)
    assert "helper" in table.functions


def test_build_symbol_table_class_to_module_mapping(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write_py(pkg / "mod.py", 'class Foo:\n    """A class."""\n    pass\n')
    table = build_symbol_table(tmp_path)
    assert table.class_to_module["Foo"] == "pkg"


def test_build_symbol_table_function_to_module_mapping(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write_py(pkg / "mod.py", 'def helper():\n    """A helper."""\n    pass\n')
    table = build_symbol_table(tmp_path)
    assert table.function_to_module["helper"] == "pkg"


def test_build_symbol_table_root_file_uses_stem(tmp_path):
    _write_py(tmp_path / "root.py", 'class Root:\n    """Root class."""\n    pass\n')
    table = build_symbol_table(tmp_path)
    assert table.class_to_module["Root"] == "root"


def test_build_symbol_table_skips_underscore_files(tmp_path):
    _write_py(tmp_path / "_private.py", 'class Secret:\n    """Secret class."""\n    pass\n')
    table = build_symbol_table(tmp_path)
    assert "Secret" not in table.classes


def test_build_symbol_table_includes_init_py(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write_py(pkg / "__init__.py", 'class InitClass:\n    """Init class."""\n    pass\n')
    table = build_symbol_table(tmp_path)
    assert "InitClass" in table.classes


# ---------------------------------------------------------------------------
# resolve_dependencies
# ---------------------------------------------------------------------------


def _make_cls(name, bases=(), **kwargs):
    return ClassInfo(
        name=name,
        bases=tuple(bases),
        docstring=kwargs.get("docstring", ""),
        is_public=True,
        class_vars=(),
        methods=(),
        validators=(),
        module_path="m",
    )


def test_resolve_dependencies_topological_order():
    table = SymbolTable()
    table.classes["Parent"] = _make_cls("Parent")
    table.classes["Child"] = _make_cls("Child", bases=("Parent",))

    resolved, external = resolve_dependencies(["Child"], table)
    names = [c.name for c in resolved]
    assert names.index("Parent") < names.index("Child")
    assert len(external) == 0


def test_resolve_dependencies_identifies_external_bases():
    table = SymbolTable()
    table.classes["MyModel"] = _make_cls("MyModel", bases=("BaseModel",))

    resolved, external = resolve_dependencies(["MyModel"], table)
    assert "BaseModel" in external
    assert all(c.name != "BaseModel" for c in resolved)


def test_resolve_dependencies_known_stubs_in_external_set():
    table = SymbolTable()
    table.classes["MyEnum"] = _make_cls("MyEnum", bases=("StrEnum",))

    _, external = resolve_dependencies(["MyEnum"], table)
    assert "StrEnum" in external
    # Verify the constant has expected stubs
    for stub in ("BaseModel", "ABC", "StrEnum"):
        assert stub in EXTERNAL_STUBS


def test_resolve_dependencies_unknown_base_also_external():
    table = SymbolTable()
    table.classes["Custom"] = _make_cls("Custom", bases=("SomeLibBase",))

    _, external = resolve_dependencies(["Custom"], table)
    assert "SomeLibBase" in external


# ---------------------------------------------------------------------------
# parse_module — plain ast.Assign class variables
# ---------------------------------------------------------------------------


def test_parse_module_extracts_plain_assign_class_vars(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('class Color:\n    """Color enum."""\n    RED = "red"\n    GREEN = "green"\n')
    cls = parse_module(src).classes[0]
    assert len(cls.class_vars) == 2
    assert cls.class_vars[0] == ("RED", "", "'red'", "")
    assert cls.class_vars[1] == ("GREEN", "", "'green'", "")


def test_parse_module_extracts_plain_assign_with_call(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('class Cfg:\n    """Config."""\n    model_config = SettingsConfigDict(env_prefix="APP_")\n')
    cls = parse_module(src).classes[0]
    assert len(cls.class_vars) == 1
    name, type_ann, default, description = cls.class_vars[0]
    assert name == "model_config"
    assert type_ann == ""
    assert "SettingsConfigDict" in default
    assert description == ""


def test_parse_module_skips_private_plain_assign(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('class Foo:\n    """Foo."""\n    PUBLIC = 1\n    _private = 2\n')
    cls = parse_module(src).classes[0]
    assert len(cls.class_vars) == 1
    assert cls.class_vars[0][0] == "PUBLIC"


def test_parse_module_mixed_annotated_and_plain_assigns(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('class Mixed:\n    """Mixed."""\n    name: str = "default"\n    CONST = 42\n    count: int\n')
    cls = parse_module(src).classes[0]
    assert len(cls.class_vars) == 3
    assert cls.class_vars[0] == ("name", "str", "'default'", "")
    assert cls.class_vars[1] == ("CONST", "", "42", "")
    assert cls.class_vars[2] == ("count", "int", "", "")


def test_parse_module_skips_multi_target_assign(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('class Foo:\n    """Foo."""\n    a = b = 1\n')
    cls = parse_module(src).classes[0]
    assert len(cls.class_vars) == 0


# ---------------------------------------------------------------------------
# parse_module — class decorators
# ---------------------------------------------------------------------------


def test_parse_module_extracts_class_decorators(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('from dataclasses import dataclass\n\n@dataclass(frozen=True)\nclass Model:\n    """A model."""\n    name: str\n')
    cls = parse_module(src).classes[0]
    assert cls.decorators == ("dataclass(frozen=True)",)


def test_parse_module_class_no_decorators(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('class Plain:\n    """Plain class."""\n    pass\n')
    cls = parse_module(src).classes[0]
    assert cls.decorators == ()


def test_parse_module_class_multiple_decorators(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('@decorator_a\n@decorator_b(x=1)\nclass Multi:\n    """Multi-decorated."""\n    pass\n')
    cls = parse_module(src).classes[0]
    assert cls.decorators == ("decorator_a", "decorator_b(x=1)")


# ---------------------------------------------------------------------------
# ValueInfo extraction (NewType, TypeAlias, Constants)
# ---------------------------------------------------------------------------


def test_parse_module_extracts_newtype(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('from typing import NewType\nDocSha = NewType("DocSha", str)\n')
    module = parse_module(src)
    assert len(module.values) == 1
    val = module.values[0]
    assert val.name == "DocSha"
    assert val.kind == "NewType"
    assert val.is_public is True


def test_parse_module_extracts_uppercase_constant(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text("MAX_RETRIES = 3\n")
    module = parse_module(src)
    assert len(module.values) == 1
    assert module.values[0].name == "MAX_RETRIES"
    assert module.values[0].kind == "Constant"


def test_parse_module_skips_private_constant(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text("_INTERNAL = 42\nPUBLIC = 1\n")
    module = parse_module(src)
    public_vals = [v for v in module.values if v.is_public]
    assert len(public_vals) == 1
    assert public_vals[0].name == "PUBLIC"


def test_parse_module_skips_lowercase_assigns(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('logger = get_logger()\nname = "test"\n')
    module = parse_module(src)
    assert len(module.values) == 0


def test_parse_module_extracts_type_alias_pep695(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text("type ConversationContent = str | list[dict]\n")
    module = parse_module(src)
    assert len(module.values) == 1
    assert module.values[0].name == "ConversationContent"
    assert module.values[0].kind == "TypeAlias"


def test_parse_module_extracts_annotated_type_alias(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text("from typing import TypeAlias\nModelName: TypeAlias = str\n")
    module = parse_module(src)
    assert len(module.values) == 1
    assert module.values[0].name == "ModelName"
    assert module.values[0].kind == "TypeAlias"


def test_build_symbol_table_includes_values(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write_py(pkg / "types.py", 'from typing import NewType\nDocSha = NewType("DocSha", str)\n')
    table = build_symbol_table(tmp_path)
    assert "DocSha" in table.values
    assert table.value_to_module["DocSha"] == "pkg"


def test_build_symbol_table_deduplicates_cross_module_functions(tmp_path):
    """Same-named functions in different modules must all be stored, not overwritten."""
    pkg_a = tmp_path / "alpha"
    pkg_a.mkdir()
    _write_py(pkg_a / "cli.py", 'def main():\n    """Alpha entry."""\n    pass\n')
    pkg_b = tmp_path / "beta"
    pkg_b.mkdir()
    _write_py(pkg_b / "cli.py", 'def main():\n    """Beta entry."""\n    pass\n')

    table = build_symbol_table(tmp_path)

    # Both functions must be present (under any key)
    all_functions = list(table.functions.values())
    main_functions = [f for f in all_functions if f.name == "main"]
    assert len(main_functions) == 2, f"Expected 2 main functions, got {len(main_functions)}: {[f.module_path for f in main_functions]}"

    # Module mappings must be correct
    modules = {table.function_to_module[k] for k, f in table.functions.items() if f.name == "main"}
    assert modules == {"alpha", "beta"}


def test_parse_module_values_make_module_public(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text("MAX_SIZE = 1024\n")
    module = parse_module(src)
    assert module.is_public is True


def test_remap_does_not_duplicate_qualified_class_in_build_guide(tmp_path):
    """When two modules define a class with the same name and the class is re-exported
    via __all__, build_guide must not produce duplicate entries.

    Scenario: _internal/types.py defines Role (processed first, stored as key "Role"),
    then prompt/components.py defines Role (collision → re-keyed to "_internal:Role"
    and "prompt:Role"). Phase 2 of _remap_private_symbols sees "Role" in prompt's
    __all__ but not in known_symbols (which has "prompt:Role"), so it re-adds Role
    under the simple key "Role" → duplicate.
    """
    from docs_generator.guide_builder import build_guide

    # Module A: _internal package with a public types.py defining Role
    internal = tmp_path / "_internal"
    internal.mkdir()
    _write_py(internal / "__init__.py", "")
    _write_py(internal / "types.py", 'class Role:\n    """Internal role enum."""\n    USER = "user"\n')

    # Module B: prompt package with components.py defining Role + __init__ re-exporting it
    prompt = tmp_path / "prompt"
    prompt.mkdir()
    _write_py(prompt / "__init__.py", '__all__ = ["Role", "Rule"]\nfrom .components import Role, Rule\n')
    _write_py(prompt / "components.py", 'class Role:\n    """Prompt role."""\n    text = "analyst"\n\nclass Rule:\n    """A rule."""\n    text = "be nice"\n')

    table = build_symbol_table(tmp_path)

    # Build guide for "prompt" module
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    guide = build_guide("prompt", tmp_path, tests_dir, table)

    # Role must appear exactly once
    role_classes = [c for c in guide.classes if c.name == "Role"]
    assert len(role_classes) == 1, f"Expected 1 Role class, got {len(role_classes)}"

    src = tmp_path / "sample.py"
    src.write_text(
        "from pydantic import BaseModel, Field\n"
        "class Model(BaseModel):\n"
        '    from_field: str = Field(description="Field description value")\n'
        "    inline_comment: int = 1  # Inline comment value\n"
        "    below_docstring: bool = False\n"
        '    """Below docstring value.\n'
        '    Extra detail that should not be rendered."""\n'
    )
    cls = parse_module(src).classes[0]

    assert cls.class_vars[0][0] == "from_field"
    assert cls.class_vars[0][3] == "Field description value"
    assert cls.class_vars[1][0] == "inline_comment"
    assert cls.class_vars[1][3] == "Inline comment value"
    assert cls.class_vars[2][0] == "below_docstring"
    assert cls.class_vars[2][3] == "Below docstring value."


# ---------------------------------------------------------------------------
# MethodInfo.is_async extraction
# ---------------------------------------------------------------------------


def test_parse_module_extracts_async_method(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text(
        "class Svc:\n"
        '    """Service class."""\n'
        "    async def fetch(self, url: str) -> str:\n"
        '        """Fetch data."""\n'
        '        return ""\n'
        "    def sync_method(self) -> None:\n"
        '        """Sync method."""\n'
        "        pass\n"
    )
    module = parse_module(src)
    methods = {m.name: m for m in module.classes[0].methods}
    assert methods["fetch"].is_async is True
    assert methods["sync_method"].is_async is False


# ---------------------------------------------------------------------------
# cached_property detection
# ---------------------------------------------------------------------------


def test_parse_module_cached_property_detected(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text(
        "from functools import cached_property\n"
        "class Doc:\n"
        '    """Document class."""\n'
        "    @cached_property\n"
        "    def mime_type(self) -> str:\n"
        '        return "text/plain"\n'
    )
    module = parse_module(src)
    method = module.classes[0].methods[0]
    assert method.is_property is True


# ---------------------------------------------------------------------------
# staticmethod detection
# ---------------------------------------------------------------------------


def test_parse_module_extracts_staticmethod(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text(
        'class Svc:\n    """Service class."""\n    @staticmethod\n    def create(name: str) -> str:\n        """Create something."""\n        return name\n'
    )
    module = parse_module(src)
    method = module.classes[0].methods[0]
    assert method.is_staticmethod is True


# ---------------------------------------------------------------------------
# Re-exported symbols from private modules
# ---------------------------------------------------------------------------


def test_build_symbol_table_discovers_root_level_reexports(tmp_path):
    """Root-level .py files that import from private modules and re-export via __all__ must appear."""
    # Private module defines the class
    _write_py(tmp_path / "_base.py", "class BaseError(Exception):\n    pass\n")

    # Root-level public file re-exports it
    _write_py(tmp_path / "exceptions.py", 'from _base import BaseError\n__all__ = ["BaseError", "SpecificError"]\nclass SpecificError(BaseError):\n    pass\n')

    table = build_symbol_table(tmp_path)
    # SpecificError is defined directly and should be found
    assert "SpecificError" in table.classes
    # BaseError is re-exported from a private module — should also be found
    assert "BaseError" in table.classes
