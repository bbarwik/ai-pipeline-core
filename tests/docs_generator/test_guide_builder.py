from pathlib import Path

from ai_pipeline_core.docs_generator.extractor import (
    ClassInfo,
    FunctionInfo,
    MethodInfo,
    SymbolTable,
    build_symbol_table,
)
from ai_pipeline_core.docs_generator.guide_builder import (
    GuideData,
    TestExample,
    build_guide,
    discover_tests,
    extract_rules,
    flatten_methods,
    render_guide,
    score_test,
    select_examples,
)


def _write_py(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _make_class(
    name: str = "Cls",
    bases: tuple[str, ...] = (),
    docstring: str = "",
    is_public: bool = True,
    methods: tuple[MethodInfo, ...] = (),
    class_vars: tuple[tuple[str, str, str], ...] = (),
    validators: tuple[MethodInfo, ...] = (),
    module_path: str = "mod",
    decorators: tuple[str, ...] = (),
) -> ClassInfo:
    return ClassInfo(
        name=name,
        bases=bases,
        docstring=docstring,
        is_public=is_public,
        class_vars=class_vars,
        methods=methods,
        validators=validators,
        module_path=module_path,
        decorators=decorators,
    )


def _make_function(
    name: str = "func",
    signature: str = "()",
    docstring: str = "",
    is_public: bool = True,
    is_async: bool = False,
    module_path: str = "mod",
) -> FunctionInfo:
    return FunctionInfo(
        name=name,
        signature=signature,
        docstring=docstring,
        source=f"{'async ' if is_async else ''}def {name}{signature}: pass",
        is_public=is_public,
        is_async=is_async,
        line_count=1,
        module_path=module_path,
    )


def _make_method(
    name: str = "method",
    signature: str = "(self)",
    docstring: str = "",
    source: str = "",
    is_property: bool = False,
    is_classmethod: bool = False,
    is_abstract: bool = False,
    line_count: int = 1,
    is_inherited: bool = False,
    inherited_from: str | None = None,
) -> MethodInfo:
    return MethodInfo(
        name=name,
        signature=signature,
        docstring=docstring,
        source=source or f"def {name}{signature}: pass",
        is_property=is_property,
        is_classmethod=is_classmethod,
        is_abstract=is_abstract,
        line_count=line_count,
        is_inherited=is_inherited,
        inherited_from=inherited_from,
    )


def _make_test_example(
    name: str = "test_foo",
    code: str = "def test_foo(): pass",
    score: int = 0,
    is_error: bool = False,
    is_marked: bool = False,
) -> TestExample:
    return TestExample(
        name=name,
        source_file="tests/test_mod.py",
        line_number=1,
        code=code,
        score=score,
        is_error_example=is_error,
        is_marked=is_marked,
    )


# ---------------------------------------------------------------------------
# discover_tests
# ---------------------------------------------------------------------------


def test_discover_tests_finds_subdir_tests(tmp_path):
    tests_dir = tmp_path / "tests"
    subdir = tests_dir / "mymod"
    subdir.mkdir(parents=True)
    (subdir / "test_basic.py").write_text("def test_one(): pass\n")

    results = discover_tests("mymod", tests_dir)
    assert len(results) == 1
    assert results[0].name == "test_one"


def test_discover_tests_finds_root_level_tests(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod_extra.py").write_text("def test_extra(): pass\n")

    results = discover_tests("mymod", tests_dir)
    assert len(results) == 1
    assert results[0].name == "test_extra"


def test_discover_tests_respects_overrides(tmp_path):
    tests_dir = tmp_path / "tests"
    custom = tests_dir / "custom_dir"
    custom.mkdir(parents=True)
    (custom / "test_x.py").write_text("def test_x(): pass\n")

    results = discover_tests("mymod", tests_dir, test_dir_overrides={"mymod": "custom_dir"})
    assert len(results) == 1
    assert results[0].name == "test_x"


def test_discover_tests_returns_correct_fields(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("def test_hello():\n    assert True\n")

    results = discover_tests("mymod", tests_dir)
    assert len(results) == 1
    ex = results[0]
    assert ex.name == "test_hello"
    assert "test_mymod.py" in ex.source_file
    assert ex.line_number == 1
    assert "def test_hello" in ex.code


def test_discover_tests_detects_error_examples(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("import pytest\ndef test_error():\n    with pytest.raises(ValueError):\n        raise ValueError\n")
    results = discover_tests("mymod", tests_dir)
    assert results[0].is_error_example is True


# ---------------------------------------------------------------------------
# score_test
# ---------------------------------------------------------------------------


def test_score_test_exact_name_match():
    ex = _make_test_example(name="test_Foo", code="x = 1")
    score = score_test(ex, ["Foo"])
    # Exact match +5, simplicity bonus for short code
    assert score >= 5


def test_score_test_exact_name_prefix_match():
    ex = _make_test_example(name="test_Foo_with_args", code="x = 1")
    score = score_test(ex, ["Foo"])
    # subject = "foo_with_args", starts with "foo_" -> +5
    assert score >= 5


def test_score_test_partial_name_match():
    ex = _make_test_example(name="test_something_Foo_related", code="x = 1")
    score = score_test(ex, ["Foo"])
    # "foo" in "test_something_foo_related" -> +3
    assert score >= 3


def test_score_test_body_occurrences():
    ex = _make_test_example(name="test_stuff", code="Foo()\nFoo()\nFoo()")
    score = score_test(ex, ["Foo"])
    # No name match, body count=3 capped at +2
    assert score >= 2


def test_score_test_error_example_bonus():
    ex_err = _make_test_example(name="test_stuff", code="short", is_error=True)
    ex_normal = _make_test_example(name="test_stuff", code="short", is_error=False)
    score_err = score_test(ex_err, ["Unrelated"])
    score_normal = score_test(ex_normal, ["Unrelated"])
    assert score_err == score_normal + 2


def test_score_test_short_test_simplicity_bonus():
    short_code = "def test_x(): pass"
    long_code = "\n".join(f"line {i}" for i in range(25))
    ex_short = _make_test_example(name="test_x", code=short_code)
    ex_long = _make_test_example(name="test_x", code=long_code)
    score_short = score_test(ex_short, ["Unrelated"])
    score_long = score_test(ex_long, ["Unrelated"])
    assert score_short > score_long


def test_score_test_pattern_bonus():
    ex = _make_test_example(name="test_basic_creation", code="short line")
    score = score_test(ex, ["Unrelated"])
    # "creation" pattern +1, simplicity +2
    assert score >= 3


# ---------------------------------------------------------------------------
# select_examples
# ---------------------------------------------------------------------------


def test_select_examples_returns_tuple():
    tests = [_make_test_example(name="test_a")]
    normal, errors = select_examples(tests, ["a"])
    assert isinstance(normal, list)
    assert isinstance(errors, list)


def test_select_examples_error_cap():
    errs = [_make_test_example(name=f"test_e{i}", is_error=True) for i in range(10)]
    _, selected_errors = select_examples(errs, ["e"], max_total=8)
    assert len(selected_errors) <= 4  # max_total // 2


def test_select_examples_normal_fills_remaining():
    tests = [
        _make_test_example(name="test_err1", is_error=True),
        _make_test_example(name="test_err2", is_error=True),
        *[_make_test_example(name=f"test_n{i}") for i in range(10)],
    ]
    normal, errors = select_examples(tests, ["n"], max_total=6)
    assert len(normal) + len(errors) <= 6


def test_select_examples_sorted_by_score():
    tests = [
        _make_test_example(name="test_MyClass", code="MyClass()"),
        _make_test_example(name="test_unrelated", code="pass"),
    ]
    normal, _ = select_examples(tests, ["MyClass"], max_total=8)
    assert len(normal) == 2
    assert normal[0].name == "test_MyClass"


# ---------------------------------------------------------------------------
# flatten_methods
# ---------------------------------------------------------------------------


def test_flatten_methods_child_takes_precedence():
    child_m = _make_method(name="run", source="def run(self): return 'child'")
    parent_m = _make_method(name="run", source="def run(self): return 'parent'")
    child = _make_class(name="Child", bases=("Parent",), methods=(child_m,))
    parent = _make_class(name="Parent", methods=(parent_m,))

    table = SymbolTable()
    table.classes["Child"] = child
    table.classes["Parent"] = parent

    flat = flatten_methods(child, table)
    run_methods = [m for m in flat if m.name == "run"]
    assert len(run_methods) == 1
    assert "child" in run_methods[0].source
    assert run_methods[0].is_inherited is False


def test_flatten_methods_inherited_marked():
    parent_m = _make_method(name="helper", source="def helper(self): pass")
    child = _make_class(name="Child", bases=("Parent",))
    parent = _make_class(name="Parent", methods=(parent_m,))

    table = SymbolTable()
    table.classes["Child"] = child
    table.classes["Parent"] = parent

    flat = flatten_methods(child, table)
    helper = [m for m in flat if m.name == "helper"]
    assert len(helper) == 1
    assert helper[0].is_inherited is True
    assert helper[0].inherited_from == "Parent"


def test_flatten_methods_external_stubs_skipped():
    ext_m = _make_method(name="model_dump")
    parent = _make_class(name="BaseModel", methods=(ext_m,))
    child = _make_class(name="MyModel", bases=("BaseModel",))

    table = SymbolTable()
    table.classes["BaseModel"] = parent
    table.classes["MyModel"] = child

    flat = flatten_methods(child, table)
    # BaseModel is in EXTERNAL_STUBS; its methods are not visited
    assert all(m.name != "model_dump" for m in flat)


# ---------------------------------------------------------------------------
# extract_rules
# ---------------------------------------------------------------------------


def test_extract_rules_finds_constraint_keywords():
    cls = _make_class(docstring="Must validate input.\nCannot be empty.\nNormal text here.")
    rules = extract_rules([cls])
    assert len(rules) == 2
    assert "Must validate input." in rules
    assert "Cannot be empty." in rules


def test_extract_rules_deduplicates():
    c1 = _make_class(name="A", docstring="Must be unique.")
    c2 = _make_class(name="B", docstring="Must be unique.")
    rules = extract_rules([c1, c2])
    assert rules.count("Must be unique.") == 1


def test_extract_rules_empty_docstring():
    cls = _make_class(docstring="")
    rules = extract_rules([cls])
    assert rules == []


# ---------------------------------------------------------------------------
# build_guide
# ---------------------------------------------------------------------------


def test_build_guide_returns_guide_data(tmp_path):
    src = tmp_path / "src"
    pkg = src / "mymod"
    pkg.mkdir(parents=True)
    _write_py(pkg / "core.py", 'class MyClass:\n    """My class."""\n    pass\n')
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    table = build_symbol_table(src)
    guide = build_guide("mymod", src, tests_dir, table)
    assert isinstance(guide, GuideData)
    assert guide.module_name == "mymod"


def test_build_guide_uses_function_to_module(tmp_path):
    src = tmp_path / "src"
    pkg = src / "mymod"
    pkg.mkdir(parents=True)
    _write_py(pkg / "funcs.py", 'def helper() -> int:\n    """A helper."""\n    return 1\n')
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    table = build_symbol_table(src)
    guide = build_guide("mymod", src, tests_dir, table)
    assert len(guide.functions) == 1
    assert guide.functions[0].name == "helper"


def test_build_guide_empty_module(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    table = SymbolTable()
    guide = build_guide("nonexistent", src, tests_dir, table)
    assert guide.classes == []
    assert guide.functions == []


# ---------------------------------------------------------------------------
# render_guide
# ---------------------------------------------------------------------------


def _empty_guide(**overrides) -> GuideData:
    defaults = dict(
        module_name="mymod",
        classes=[],
        functions=[],
        rules=[],
        external_bases=set(),
        normal_examples=[],
        error_examples=[],
    )
    defaults.update(overrides)
    return GuideData(**defaults)


def test_render_guide_has_module_header():
    data = _empty_guide(classes=[_make_class(name="Foo")])
    output = render_guide(data)
    assert "# MODULE: mymod" in output
    assert "# CLASSES: Foo" in output


def test_render_guide_includes_function_section():
    data = _empty_guide(functions=[_make_function(name="do_stuff", is_async=True)])
    output = render_guide(data)
    assert "# === FUNCTIONS ===" in output
    assert "async def do_stuff" in output


def test_render_guide_includes_example_sections():
    err_ex = _make_test_example(name="test_err", code="def test_err(): raise", is_error=True)
    data = _empty_guide(
        normal_examples=[_make_test_example(name="test_basic", code="def test_basic(): pass")],
        error_examples=[err_ex],
    )
    output = render_guide(data)
    assert "# === EXAMPLES (from tests/) ===" in output
    assert "# === ERROR EXAMPLES (What NOT to Do) ===" in output


def test_render_guide_size_header():
    data = _empty_guide()
    output = render_guide(data)
    assert "# SIZE:" in output
    assert "KB" in output


def test_render_guide_includes_method_body():
    method = _make_method(
        name="compute",
        signature="(self, x: int) -> int",
        source="def compute(self, x: int) -> int:\n    return x + 1",
        line_count=1,
    )
    cls = _make_class(name="Calc", methods=(method,))
    data = _empty_guide(classes=[cls])
    output = render_guide(data)
    assert "return x + 1" in output


def test_render_guide_excludes_private_methods():
    public_m = _make_method(name="run", source="def run(self): pass")
    private_m = _make_method(name="_internal", source="def _internal(self): pass")
    cls = _make_class(name="Svc", methods=(public_m, private_m))
    data = _empty_guide(classes=[cls])
    output = render_guide(data)
    assert "def run" in output
    assert "_internal" not in output


def test_render_guide_includes_dunder_methods():
    dunder = _make_method(name="__init__", source="def __init__(self):\n    self.x = 1")
    cls = _make_class(name="Foo", methods=(dunder,))
    data = _empty_guide(classes=[cls])
    output = render_guide(data)
    assert "def __init__" in output
    assert "self.x = 1" in output


# ---------------------------------------------------------------------------
# @pytest.mark.ai_docs marker detection
# ---------------------------------------------------------------------------


def test_discover_tests_detects_ai_docs_marker(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("import pytest\n\n@pytest.mark.ai_docs\ndef test_marked():\n    pass\n\ndef test_unmarked():\n    pass\n")
    results = discover_tests("mymod", tests_dir)
    by_name = {t.name: t for t in results}
    assert by_name["test_marked"].is_marked is True
    assert by_name["test_unmarked"].is_marked is False


def test_discover_tests_detects_ai_docs_marker_with_call(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("import pytest\n\n@pytest.mark.ai_docs()\ndef test_called():\n    pass\n")
    results = discover_tests("mymod", tests_dir)
    assert results[0].is_marked is True


def test_discover_tests_detects_mark_import_form(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("from pytest import mark\n\n@mark.ai_docs\ndef test_short_form():\n    pass\n")
    results = discover_tests("mymod", tests_dir)
    assert results[0].is_marked is True


def test_discover_tests_marker_not_in_extracted_code(tmp_path):
    """The @pytest.mark.ai_docs decorator should not appear in extracted code."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("import pytest\n\n@pytest.mark.ai_docs\ndef test_marked():\n    assert True\n")
    results = discover_tests("mymod", tests_dir)
    assert "@pytest.mark.ai_docs" not in results[0].code
    assert "def test_marked" in results[0].code


def test_discover_tests_stacked_decorators(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("import pytest\n\n@pytest.mark.parametrize('x', [1, 2])\n@pytest.mark.ai_docs\ndef test_stacked(x):\n    pass\n")
    results = discover_tests("mymod", tests_dir)
    assert results[0].is_marked is True


def test_discover_tests_async_function_with_marker(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("import pytest\n\n@pytest.mark.ai_docs\nasync def test_async_marked():\n    pass\n")
    results = discover_tests("mymod", tests_dir)
    assert results[0].is_marked is True
    assert results[0].name == "test_async_marked"


def test_discover_tests_unrelated_marker_not_detected(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("import pytest\n\n@pytest.mark.integration\ndef test_not_ai_docs():\n    pass\n")
    results = discover_tests("mymod", tests_dir)
    assert results[0].is_marked is False


# ---------------------------------------------------------------------------
# select_examples with marked tests
# ---------------------------------------------------------------------------


def test_select_examples_marked_priority_over_score():
    marked = _make_test_example(name="test_marked_thing", code="pass", is_marked=True)
    unmarked = _make_test_example(name="test_MyClass", code="MyClass()\nMyClass()\nMyClass()")
    normal, _ = select_examples([marked, unmarked], ["MyClass"], max_total=1)
    assert len(normal) == 1
    assert normal[0].name == "test_marked_thing"


def test_select_examples_marked_override_cap():
    marked = [_make_test_example(name=f"test_m{i}", code="pass", is_marked=True) for i in range(10)]
    normal, errors = select_examples(marked, ["x"], max_total=8)
    assert len(normal) + len(errors) == 10


def test_select_examples_marked_errors_in_error_list():
    marked_err = _make_test_example(name="test_err", code="x", is_error=True, is_marked=True)
    marked_normal = _make_test_example(name="test_ok", code="x", is_marked=True)
    normal, errors = select_examples([marked_err, marked_normal], ["x"], max_total=8)
    assert marked_normal in normal
    assert marked_err in errors


def test_select_examples_auto_fills_remaining_after_marked():
    marked = [_make_test_example(name="test_m1", code="pass", is_marked=True)]
    auto = [_make_test_example(name=f"test_MyClass_{i}", code="MyClass()") for i in range(10)]
    normal, errors = select_examples(marked + auto, ["MyClass"], max_total=4)
    total = len(normal) + len(errors)
    assert total == 4
    assert any(t.name == "test_m1" for t in normal)


def test_select_examples_no_marked_same_as_before():
    tests = [
        _make_test_example(name="test_MyClass", code="MyClass()"),
        _make_test_example(name="test_unrelated", code="pass"),
    ]
    normal, _ = select_examples(tests, ["MyClass"], max_total=8)
    assert normal[0].name == "test_MyClass"


def test_select_examples_auto_error_cap_with_marked():
    marked = [
        _make_test_example(name="test_m1", code="pass", is_marked=True),
        _make_test_example(name="test_m2", code="pass", is_marked=True),
    ]
    auto_errors = [_make_test_example(name=f"test_e{i}", code="x", is_error=True) for i in range(10)]
    auto_normal = [_make_test_example(name=f"test_n{i}", code="x") for i in range(10)]
    normal, errors = select_examples(marked + auto_errors + auto_normal, ["x"], max_total=8)
    auto_error_count = len([t for t in errors if not t.is_marked])
    assert auto_error_count == 3  # remaining(6) // 2


def test_select_examples_marked_errors_bypass_cap():
    marked_errors = [_make_test_example(name=f"test_me{i}", code="x", is_error=True, is_marked=True) for i in range(6)]
    normal, errors = select_examples(marked_errors, ["x"], max_total=8)
    assert len(errors) == 6


def test_select_examples_empty_input():
    normal, errors = select_examples([], ["x"], max_total=8)
    assert normal == []
    assert errors == []


def test_select_examples_all_marked_all_errors():
    tests = [_make_test_example(name=f"test_e{i}", is_error=True, is_marked=True) for i in range(10)]
    normal, errors = select_examples(tests, ["x"], max_total=8)
    assert len(normal) == 0
    assert len(errors) == 10


# ---------------------------------------------------------------------------
# Zero-overlap warning
# ---------------------------------------------------------------------------


def test_marked_zero_overlap_warning(tmp_path, caplog):
    src = tmp_path / "src"
    pkg = src / "mymod"
    pkg.mkdir(parents=True)
    _write_py(pkg / "core.py", "class MyClass:\n    pass\n")

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("import pytest\n\n@pytest.mark.ai_docs\ndef test_completely_unrelated():\n    assert 1 + 1 == 2\n")

    import logging

    table = build_symbol_table(src)
    with caplog.at_level(logging.WARNING):
        build_guide("mymod", src, tests_dir, table)
    assert "no symbol overlap" in caplog.text.lower()


# ---------------------------------------------------------------------------
# render_guide — unannotated class variables
# ---------------------------------------------------------------------------


def test_render_guide_class_var_without_type():
    cls = _make_class(
        name="Cfg",
        class_vars=(("MODEL_CONFIG", "", "SettingsConfigDict()"),),
    )
    data = _empty_guide(classes=[cls])
    output = render_guide(data)
    assert "MODEL_CONFIG = SettingsConfigDict()" in output
    assert "MODEL_CONFIG:" not in output


def test_render_guide_mixed_class_vars():
    cls = _make_class(
        name="Mixed",
        class_vars=(
            ("name", "str", "'default'"),
            ("CONST", "", "42"),
            ("count", "int", ""),
        ),
    )
    data = _empty_guide(classes=[cls])
    output = render_guide(data)
    assert "name: str = 'default'" in output
    assert "CONST = 42" in output
    assert "CONST:" not in output
    assert "count: int" in output


# ---------------------------------------------------------------------------
# Test decorator inclusion in extracted test examples (Part A)
# ---------------------------------------------------------------------------


def test_discover_tests_includes_parametrize_decorator(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("import pytest\n\n@pytest.mark.parametrize('x', [1, 2])\ndef test_params(x):\n    pass\n")
    results = discover_tests("mymod", tests_dir)
    assert "@pytest.mark.parametrize" in results[0].code
    assert "def test_params" in results[0].code


def test_discover_tests_strips_ai_docs_keeps_other_decorators(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("import pytest\n\n@pytest.mark.parametrize('y', [3])\n@pytest.mark.ai_docs\ndef test_mixed(y):\n    pass\n")
    results = discover_tests("mymod", tests_dir)
    code = results[0].code
    assert "@pytest.mark.ai_docs" not in code
    assert "@pytest.mark.parametrize" in code
    assert "def test_mixed" in code


def test_discover_tests_strips_ai_docs_call_form(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("import pytest\n\n@pytest.mark.ai_docs()\ndef test_called():\n    pass\n")
    results = discover_tests("mymod", tests_dir)
    assert "pytest.mark.ai_docs" not in results[0].code


def test_discover_tests_strips_mark_ai_docs_import_form(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("from pytest import mark\n\n@mark.ai_docs\ndef test_short():\n    pass\n")
    results = discover_tests("mymod", tests_dir)
    assert "mark.ai_docs" not in results[0].code


# ---------------------------------------------------------------------------
# Class decorator rendering (Part B)
# ---------------------------------------------------------------------------


def test_render_guide_includes_class_decorators():
    cls = _make_class(
        name="Data",
        decorators=("dataclass(frozen=True)",),
    )
    data = _empty_guide(classes=[cls])
    output = render_guide(data)
    assert "@dataclass(frozen=True)" in output
    assert output.index("@dataclass(frozen=True)") < output.index("class Data")


def test_render_guide_class_without_decorators():
    cls = _make_class(name="Plain")
    data = _empty_guide(classes=[cls])
    output = render_guide(data)
    lines = output.splitlines()
    class_idx = next(i for i, line in enumerate(lines) if "class Plain" in line)
    # No decorator line immediately before
    assert not lines[class_idx - 1].startswith("@")


# ---------------------------------------------------------------------------
# Dedenting class method tests (Issue 2)
# ---------------------------------------------------------------------------


def test_discover_tests_dedents_class_methods(tmp_path):
    """Test methods inside classes are dedented to column 0."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("class TestFoo:\n    def test_bar(self):\n        assert True\n")
    results = discover_tests("mymod", tests_dir)
    assert len(results) == 1
    lines = results[0].code.splitlines()
    assert lines[0] == "def test_bar(self):"
    assert lines[1] == "    assert True"


def test_discover_tests_dedents_class_method_with_decorator(tmp_path):
    """Decorators on class methods are dedented along with the def line."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text(
        "import pytest\n\nclass TestFoo:\n    @pytest.mark.parametrize('x', [1])\n    def test_bar(self, x):\n        assert x == 1\n"
    )
    results = discover_tests("mymod", tests_dir)
    lines = results[0].code.splitlines()
    assert lines[0] == "@pytest.mark.parametrize('x', [1])"
    assert lines[1] == "def test_bar(self, x):"
    assert lines[2] == "    assert x == 1"


def test_discover_tests_standalone_function_unchanged(tmp_path):
    """Top-level test functions (already at column 0) are unaffected."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("def test_simple():\n    assert True\n")
    results = discover_tests("mymod", tests_dir)
    assert results[0].code == "def test_simple():\n    assert True"


def test_discover_tests_dedents_class_method_with_ai_docs(tmp_path):
    """ai_docs marker is stripped AND the result is dedented."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("import pytest\n\nclass TestFoo:\n    @pytest.mark.ai_docs\n    def test_marked(self):\n        pass\n")
    results = discover_tests("mymod", tests_dir)
    assert "ai_docs" not in results[0].code
    assert results[0].code.startswith("def test_marked")


# ---------------------------------------------------------------------------
# Relative source paths (Issue 1)
# ---------------------------------------------------------------------------


def test_discover_tests_uses_relative_paths(tmp_path):
    """source_file is relative to repo_root when provided."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("def test_one(): pass\n")

    results = discover_tests("mymod", tests_dir, repo_root=tmp_path)
    assert results[0].source_file == "tests/test_mymod.py"


def test_discover_tests_relative_paths_in_subdir(tmp_path):
    """Subdirectory test files also get relative paths."""
    tests_dir = tmp_path / "tests"
    subdir = tests_dir / "mymod"
    subdir.mkdir(parents=True)
    (subdir / "test_core.py").write_text("def test_core(): pass\n")

    results = discover_tests("mymod", tests_dir, repo_root=tmp_path)
    assert results[0].source_file == "tests/mymod/test_core.py"


def test_discover_tests_fallback_to_absolute_when_outside_root(tmp_path):
    """Falls back to absolute path when test file is outside repo_root."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("def test_one(): pass\n")

    unrelated_root = tmp_path / "other"
    unrelated_root.mkdir()
    results = discover_tests("mymod", tests_dir, repo_root=unrelated_root)
    assert str(tests_dir) in results[0].source_file


def test_build_guide_produces_relative_source_paths(tmp_path):
    """build_guide defaults repo_root to source_dir.parent, producing relative paths."""
    src = tmp_path / "src"
    pkg = src / "mymod"
    pkg.mkdir(parents=True)
    _write_py(pkg / "core.py", 'class MyClass:\n    """My class."""\n    pass\n')
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mymod.py").write_text("def test_hello():\n    assert True\n")

    table = build_symbol_table(src)
    guide = build_guide("mymod", src, tests_dir, table)
    for ex in guide.normal_examples + guide.error_examples:
        assert not ex.source_file.startswith("/")


# ---------------------------------------------------------------------------
# Internal types (Issue 6)
# ---------------------------------------------------------------------------


def test_build_guide_collects_internal_types(tmp_path):
    """Private classes referenced in public function signatures appear in internal_types."""
    src = tmp_path / "src"
    pkg = src / "mymod"
    pkg.mkdir(parents=True)
    _write_py(
        pkg / "core.py",
        "from typing import Protocol\n"
        "class _TaskLike(Protocol):\n"
        '    """Internal protocol."""\n'
        "    def __call__(self) -> None: ...\n"
        "def pipeline_task() -> _TaskLike:\n"
        '    """Public func."""\n'
        "    ...\n",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    table = build_symbol_table(src)
    guide = build_guide("mymod", src, tests_dir, table)
    assert len(guide.internal_types) == 1
    assert guide.internal_types[0].name == "_TaskLike"


def test_build_guide_no_internal_types_when_none_referenced(tmp_path):
    """No internal types when public functions don't reference private classes."""
    src = tmp_path / "src"
    pkg = src / "mymod"
    pkg.mkdir(parents=True)
    _write_py(pkg / "core.py", 'def helper() -> int:\n    """Public func."""\n    return 1\n')
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    table = build_symbol_table(src)
    guide = build_guide("mymod", src, tests_dir, table)
    assert guide.internal_types == []


def test_render_guide_includes_internal_types_section():
    """Rendered guide includes INTERNAL TYPES section when present."""
    private_cls = _make_class(
        name="_MyProtocol",
        is_public=False,
        methods=(_make_method(name="__call__", signature="(self) -> None"),),
    )
    data = _empty_guide(internal_types=[private_cls])
    output = render_guide(data)
    assert "# === INTERNAL TYPES (referenced by public API) ===" in output
    assert "class _MyProtocol" in output
    assert "def __call__" in output


def test_render_guide_no_internal_types_section_when_empty():
    """No INTERNAL TYPES section when internal_types is empty."""
    data = _empty_guide()
    output = render_guide(data)
    assert "INTERNAL TYPES" not in output


def test_internal_types_section_between_dependencies_and_public_api():
    """INTERNAL TYPES section appears between DEPENDENCIES and PUBLIC API."""
    private_cls = _make_class(name="_Proto", is_public=False)
    data = _empty_guide(
        external_bases={"BaseModel"},
        internal_types=[private_cls],
        classes=[_make_class(name="MyClass")],
    )
    output = render_guide(data)
    deps_idx = output.index("# === DEPENDENCIES")
    internal_idx = output.index("# === INTERNAL TYPES")
    api_idx = output.index("# === PUBLIC API")
    assert deps_idx < internal_idx < api_idx
