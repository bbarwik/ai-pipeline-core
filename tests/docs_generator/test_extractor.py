from ai_pipeline_core.docs_generator.extractor import (
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
    assert cls.class_vars[0] == ("name", "str", "'default'")
    assert cls.class_vars[1] == ("count", "int", "")


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
    assert cls.class_vars[0] == ("RED", "", "'red'")
    assert cls.class_vars[1] == ("GREEN", "", "'green'")


def test_parse_module_extracts_plain_assign_with_call(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text('class Cfg:\n    """Config."""\n    model_config = SettingsConfigDict(env_prefix="APP_")\n')
    cls = parse_module(src).classes[0]
    assert len(cls.class_vars) == 1
    name, type_ann, default = cls.class_vars[0]
    assert name == "model_config"
    assert type_ann == ""
    assert "SettingsConfigDict" in default


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
    assert cls.class_vars[0] == ("name", "str", "'default'")
    assert cls.class_vars[1] == ("CONST", "", "42")
    assert cls.class_vars[2] == ("count", "int", "")


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
