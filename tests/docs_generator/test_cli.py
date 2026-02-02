from pathlib import Path

from ai_pipeline_core.docs_generator.cli import (
    EXCLUDED_MODULES,
    TEST_DIR_OVERRIDES,
    _discover_modules,
    _render_index,
    _run_check,
    _run_generate,
    main,
)
from ai_pipeline_core.docs_generator.extractor import SymbolTable
from ai_pipeline_core.docs_generator.guide_builder import GuideData
from ai_pipeline_core.docs_generator.validator import HASH_FILE


def _make_repo(tmp_path):
    src = tmp_path / "ai_pipeline_core"
    src.mkdir()
    tests = tmp_path / "tests"
    tests.mkdir()
    output = tmp_path / ".ai-docs"
    (src / "__init__.py").write_text('"""Module."""\n')
    (src / "mod.py").write_text('"""A module."""\ndef foo():\n    """A func."""\n    pass\n')
    (tests / "test_mod.py").write_text("def test_foo(): pass\n")
    return src, tests, output


def _empty_guide_data(module_name):
    return GuideData(
        module_name=module_name,
        classes=[],
        functions=[],
        rules=[],
        external_bases=set(),
        normal_examples=[],
        error_examples=[],
    )


def test_generate_creates_output_dir(tmp_path):
    src, tests, output = _make_repo(tmp_path)
    assert not output.exists()
    _run_generate(src, tests, output, tmp_path)
    assert output.is_dir()


def test_generate_writes_guides(tmp_path, monkeypatch):
    src, tests, output = _make_repo(tmp_path)
    table = SymbolTable()

    from ai_pipeline_core.docs_generator.extractor import FunctionInfo

    func = FunctionInfo(
        name="foo",
        signature="()",
        docstring="A func.",
        source="def foo(): ...",
        is_public=True,
        is_async=False,
        line_count=3,
        module_path="test",
    )

    def mock_build_symbol_table(source_dir):
        return table

    def mock_build_guide(module_name, source_dir, tests_dir, tbl, overrides, repo_root=None):
        data = _empty_guide_data(module_name)
        if module_name == "mod":
            data.functions = [func]
        return data

    def mock_render(data):
        return f"# GUIDE: {data.module_name}\n"

    def mock_manage(data, rendered_content, max_size=51200):
        return rendered_content

    monkeypatch.setattr(
        "ai_pipeline_core.docs_generator.cli.build_symbol_table",
        mock_build_symbol_table,
    )
    monkeypatch.setattr("ai_pipeline_core.docs_generator.cli.build_guide", mock_build_guide)
    monkeypatch.setattr("ai_pipeline_core.docs_generator.cli.render_guide", mock_render)
    monkeypatch.setattr("ai_pipeline_core.docs_generator.cli.manage_guide_size", mock_manage)

    result = _run_generate(src, tests, output, tmp_path)
    assert result == 0
    guide_files = [f for f in output.glob("*.md") if f.name != "INDEX.md"]
    assert len(guide_files) > 0
    assert (output / "INDEX.md").exists()
    assert (output / HASH_FILE).exists()


def test_generate_writes_index(tmp_path, monkeypatch):
    src, tests, output = _make_repo(tmp_path)
    table = SymbolTable()

    def mock_build_symbol_table(source_dir):
        return table

    def mock_build_guide(module_name, source_dir, tests_dir, tbl, overrides, repo_root=None):
        return _empty_guide_data(module_name)

    monkeypatch.setattr(
        "ai_pipeline_core.docs_generator.cli.build_symbol_table",
        mock_build_symbol_table,
    )
    monkeypatch.setattr("ai_pipeline_core.docs_generator.cli.build_guide", mock_build_guide)

    _run_generate(src, tests, output, tmp_path)
    assert (output / "INDEX.md").exists()


def test_generate_writes_hash(tmp_path, monkeypatch):
    src, tests, output = _make_repo(tmp_path)
    table = SymbolTable()

    def mock_build_symbol_table(source_dir):
        return table

    def mock_build_guide(module_name, source_dir, tests_dir, tbl, overrides, repo_root=None):
        return _empty_guide_data(module_name)

    monkeypatch.setattr(
        "ai_pipeline_core.docs_generator.cli.build_symbol_table",
        mock_build_symbol_table,
    )
    monkeypatch.setattr("ai_pipeline_core.docs_generator.cli.build_guide", mock_build_guide)

    _run_generate(src, tests, output, tmp_path)
    assert (output / HASH_FILE).exists()
    content = (output / HASH_FILE).read_text().strip()
    assert len(content) == 64


def test_generate_skips_empty_modules(tmp_path, monkeypatch):
    src, tests, output = _make_repo(tmp_path)
    table = SymbolTable()

    def mock_build_symbol_table(source_dir):
        return table

    def mock_build_guide(module_name, source_dir, tests_dir, tbl, overrides, repo_root=None):
        return _empty_guide_data(module_name)

    monkeypatch.setattr(
        "ai_pipeline_core.docs_generator.cli.build_symbol_table",
        mock_build_symbol_table,
    )
    monkeypatch.setattr("ai_pipeline_core.docs_generator.cli.build_guide", mock_build_guide)

    _run_generate(src, tests, output, tmp_path)
    # All modules are empty, so no guide .md files (only INDEX.md)
    guide_files = [f for f in output.glob("*.md") if f.name != "INDEX.md"]
    assert guide_files == []


def test_generate_cleans_stale_files(tmp_path, monkeypatch):
    src, tests, output = _make_repo(tmp_path)
    output.mkdir()
    (output / "old_module.md").write_text("stale content")

    table = SymbolTable()

    def mock_build_symbol_table(source_dir):
        return table

    def mock_build_guide(module_name, source_dir, tests_dir, tbl, overrides, repo_root=None):
        return _empty_guide_data(module_name)

    monkeypatch.setattr(
        "ai_pipeline_core.docs_generator.cli.build_symbol_table",
        mock_build_symbol_table,
    )
    monkeypatch.setattr("ai_pipeline_core.docs_generator.cli.build_guide", mock_build_guide)

    _run_generate(src, tests, output, tmp_path)
    assert not (output / "old_module.md").exists()


def test_check_passes_valid(tmp_path, monkeypatch):
    src, tests, output = _make_repo(tmp_path)
    output.mkdir()

    from ai_pipeline_core.docs_generator.validator import compute_source_hash

    h = compute_source_hash(src, tests)
    (output / HASH_FILE).write_text(h + "\n")
    (output / "mod.md").write_text("def foo():\n    pass\n")

    result = _run_check(src, tests, output)
    assert result == 0


def test_check_fails_stale(tmp_path):
    src, tests, output = _make_repo(tmp_path)
    output.mkdir()
    (output / HASH_FILE).write_text("wrong_hash\n")
    (output / "mod.md").write_text("def foo():\n    pass\n")

    result = _run_check(src, tests, output)
    assert result == 1


def test_check_fails_missing_dir(tmp_path):
    src, tests, output = _make_repo(tmp_path)
    result = _run_check(src, tests, output)
    assert result == 1


def test_path_auto_detection():
    result = main([
        "--source-dir",
        "/nonexistent/src",
        "--tests-dir",
        "/nonexistent/tests",
        "--output-dir",
        "/tmp/test_ai_docs_out",
        "generate",
    ])
    # Nonexistent dirs produce 0 guides but no crash — returns success
    assert result == 0


def test_path_override(tmp_path):
    src, tests, output = _make_repo(tmp_path)
    result = main([
        "--source-dir",
        str(src),
        "--tests-dir",
        str(tests),
        "--output-dir",
        str(output),
        "generate",
    ])
    assert isinstance(result, int)


def test_module_auto_discovery():
    src_dir = Path(__file__).resolve().parent.parent.parent / "ai_pipeline_core"
    discovered = _discover_modules(src_dir)
    assert "documents" in discovered
    assert "document_store" in discovered
    assert "llm" in discovered
    assert "observability" in discovered
    assert "pipeline" in discovered
    assert "tracing" not in discovered  # moved into observability/
    # Excluded modules must not appear
    for excluded in EXCLUDED_MODULES:
        assert excluded not in discovered


def test_test_dir_overrides_correctness():
    tests_dir = Path(__file__).resolve().parent.parent
    for module_name, override in TEST_DIR_OVERRIDES.items():
        override_dir = tests_dir / override
        assert override_dir.is_dir(), f"Override dir {override_dir} for module {module_name} does not exist"


def test_render_index_content():
    generated = [("documents", 5000), ("llm", 3000)]
    content = _render_index(generated)
    assert "# AI Documentation Index" in content
    assert "documents" in content
    assert "llm" in content
    assert "5,000" in content
