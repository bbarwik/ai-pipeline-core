import dataclasses

import pytest

from ai_pipeline_core.docs_generator.validator import (
    ValidationResult,
    validate_all,
    validate_completeness,
    validate_size,
)


def _write_py(path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def repo(tmp_path):
    src = tmp_path / "ai_pipeline_core"
    src.mkdir()
    tests = tmp_path / "tests"
    tests.mkdir()
    _write_py(src / "mod.py", 'class Foo:\n    """A class."""\n')
    _write_py(tests / "test_mod.py", "def test_foo(): pass\n")
    return tmp_path, src, tests


def test_validate_completeness_all_covered(repo):
    tmp_path, src, tests = repo
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    (ai_docs / "mod.md").write_text("class Foo:\n    pass\n")
    missing = validate_completeness(ai_docs, src)
    assert missing == []


def test_validate_completeness_missing_symbols(repo):
    tmp_path, src, tests = repo
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    (ai_docs / "mod.md").write_text("nothing here\n")
    missing = validate_completeness(ai_docs, src)
    assert "Foo" in missing


def test_validate_completeness_skips_private_files(repo):
    tmp_path, src, tests = repo
    _write_py(src / "_internal.py", "class Hidden:\n    pass\n")
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    (ai_docs / "mod.md").write_text("class Foo:\n    pass\n")
    missing = validate_completeness(ai_docs, src)
    assert "Hidden" not in missing


def test_validate_completeness_skips_private_symbols(repo):
    tmp_path, src, tests = repo
    _write_py(src / "extra.py", "class _Secret:\n    pass\nclass Visible:\n    pass\n")
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    (ai_docs / "mod.md").write_text("class Foo:\nclass Visible:\n")
    missing = validate_completeness(ai_docs, src)
    assert "_Secret" not in missing
    assert "Visible" not in missing


def test_validate_completeness_includes_init_py(repo):
    tmp_path, src, tests = repo
    _write_py(src / "__init__.py", "class InitPublic:\n    pass\n")
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    (ai_docs / "mod.md").write_text("class Foo:\n    pass\n")
    missing = validate_completeness(ai_docs, src)
    assert "InitPublic" in missing


def test_validate_size_all_ok(tmp_path):
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    (ai_docs / "small.md").write_text("x" * 100)
    violations = validate_size(ai_docs)
    assert violations == []


def test_validate_size_violations(tmp_path):
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    (ai_docs / "big.md").write_text("x" * 60_000)
    violations = validate_size(ai_docs, max_size=50_000)
    assert len(violations) == 1
    assert violations[0][0] == "big.md"


def test_validate_all_pass(repo):
    tmp_path, src, tests = repo
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    (ai_docs / "mod.md").write_text("class Foo:\n    pass\n")
    result = validate_all(ai_docs, src)
    assert result.is_valid is True


def test_validate_all_fail_incomplete(repo):
    tmp_path, src, tests = repo
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    (ai_docs / "mod.md").write_text("nothing\n")
    result = validate_all(ai_docs, src)
    assert len(result.missing_symbols) > 0
    assert result.is_valid is False


def test_validate_all_oversized_still_valid(repo):
    tmp_path, src, tests = repo
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    (ai_docs / "mod.md").write_text("class Foo:\n" + "x" * 60_000)
    result = validate_all(ai_docs, src)
    assert len(result.size_violations) > 0
    assert result.is_valid is True


def test_validation_result_frozen():
    result = ValidationResult(missing_symbols=(), size_violations=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.missing_symbols = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Completeness checks for NewType/Constants (#2)
# ---------------------------------------------------------------------------


def test_validate_completeness_finds_newtype(tmp_path):
    src = tmp_path / "ai_pipeline_core"
    src.mkdir()
    _write_py(src / "types.py", 'from typing import NewType\nDocSha = NewType("DocSha", str)\n')
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    (ai_docs / "types.md").write_text("nothing\n")
    missing = validate_completeness(ai_docs, src)
    assert "DocSha" in missing


def test_validate_completeness_covered_newtype(tmp_path):
    src = tmp_path / "ai_pipeline_core"
    src.mkdir()
    _write_py(src / "types.py", 'from typing import NewType\nDocSha = NewType("DocSha", str)\n')
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    (ai_docs / "types.md").write_text('DocSha = NewType("DocSha", str)\n')
    missing = validate_completeness(ai_docs, src)
    assert "DocSha" not in missing


def test_validate_completeness_finds_uppercase_constant(tmp_path):
    src = tmp_path / "ai_pipeline_core"
    src.mkdir()
    _write_py(src / "config.py", "MAX_RETRIES = 3\n")
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    (ai_docs / "config.md").write_text("nothing\n")
    missing = validate_completeness(ai_docs, src)
    assert "MAX_RETRIES" in missing


# ---------------------------------------------------------------------------
# validate_private_reexports tests
# ---------------------------------------------------------------------------


from ai_pipeline_core.docs_generator.validator import (
    _find_public_symbols,
    _parse_init_all,
    _read_all_guides,
    validate_private_reexports,
)


class TestValidatePrivateReexports:
    def test_flags_private_import(self, tmp_path):
        src = tmp_path / "pkg"
        src.mkdir()
        _write_py(src / "__init__.py", '__all__ = ["Foo"]\nfrom ._internal import Foo\n')
        _write_py(src / "_internal.py", "class Foo: pass\n")
        violations = validate_private_reexports(src)
        assert len(violations) == 1
        assert "'Foo'" in violations[0]
        assert "_internal" in violations[0]

    def test_public_ok(self, tmp_path):
        src = tmp_path / "pkg"
        src.mkdir()
        _write_py(src / "__init__.py", '__all__ = ["Bar"]\nfrom .public import Bar\n')
        _write_py(src / "public.py", "class Bar: pass\n")
        violations = validate_private_reexports(src)
        assert violations == []

    def test_non_init_exempt_via_parent_all(self, tmp_path):
        src = tmp_path / "pkg"
        src.mkdir()
        _write_py(src / "__init__.py", '__all__ = ["Sym"]\nfrom ._internal import Sym\n')
        _write_py(src / "_internal.py", "class Sym: pass\n")
        _write_py(src / "types.py", '__all__ = ["Sym"]\nfrom ._internal import Sym\n')
        violations = validate_private_reexports(src)
        # Only __init__.py should flag it; types.py is exempt because Sym is in parent __all__
        assert len(violations) == 1
        assert "__init__.py" in violations[0]

    def test_skips_private_pkg(self, tmp_path):
        src = tmp_path / "pkg"
        src.mkdir()
        _write_py(src / "__init__.py", "")
        private_pkg = src / "_private_pkg"
        private_pkg.mkdir()
        _write_py(private_pkg / "__init__.py", '__all__ = ["X"]\nfrom ._mod import X\n')
        _write_py(private_pkg / "_mod.py", "class X: pass\n")
        violations = validate_private_reexports(src)
        assert violations == []

    def test_excluded_modules(self, tmp_path):
        src = tmp_path / "pkg"
        src.mkdir()
        _write_py(src / "__init__.py", "")
        sub = src / "excluded_mod"
        sub.mkdir()
        _write_py(sub / "__init__.py", '__all__ = ["Y"]\nfrom ._priv import Y\n')
        _write_py(sub / "_priv.py", "class Y: pass\n")
        violations = validate_private_reexports(src, excluded_modules=frozenset({"excluded_mod"}))
        assert violations == []


class TestValidateAllIncludesReexportViolations:
    def test_includes_private_reexports(self, tmp_path):
        src = tmp_path / "pkg"
        src.mkdir()
        _write_py(src / "__init__.py", '__all__ = ["Foo"]\nfrom ._priv import Foo\n')
        _write_py(src / "_priv.py", "class Foo: pass\n")
        ai_docs = tmp_path / ".ai-docs"
        ai_docs.mkdir()
        (ai_docs / "mod.md").write_text("class Foo:\n    pass\n")
        result = validate_all(ai_docs, src)
        assert len(result.private_reexports) > 0
        assert result.is_valid is False


class TestParseInitAll:
    def test_annotated_assign(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('__all__: list[str] = ["Alpha", "Beta"]\n')
        result = _parse_init_all(f)
        assert result == {"Alpha", "Beta"}

    def test_syntax_error(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def unclosed(:\n")
        result = _parse_init_all(f)
        assert result == set()

    def test_no_all(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("x = 1\n")
        result = _parse_init_all(f)
        assert result == set()


class TestFindPublicSymbols:
    def test_type_alias(self, tmp_path):
        src = tmp_path / "pkg"
        src.mkdir()
        _write_py(src / "types.py", "type MyAlias = str | int\n")
        symbols = _find_public_symbols(src)
        assert "MyAlias" in symbols

    def test_syntax_error_skipped(self, tmp_path):
        src = tmp_path / "pkg"
        src.mkdir()
        _write_py(src / "bad.py", "def broken(:\n")
        symbols = _find_public_symbols(src)
        assert symbols == set()

    def test_excludes_main(self, tmp_path):
        src = tmp_path / "pkg"
        src.mkdir()
        _write_py(src / "cli.py", "def main(): pass\n")
        symbols = _find_public_symbols(src)
        assert "main" not in symbols

    def test_skips_private_package(self, tmp_path):
        src = tmp_path / "pkg"
        src.mkdir()
        _write_py(src / "__init__.py", "")
        priv = src / "_internal"
        priv.mkdir()
        _write_py(priv / "__init__.py", "class Hidden: pass\n")
        symbols = _find_public_symbols(src)
        assert "Hidden" not in symbols


class TestReadAllGuides:
    def test_nonexistent_dir(self, tmp_path):
        result = _read_all_guides(tmp_path / "nonexistent")
        assert result == ""

    def test_concatenates(self, tmp_path):
        d = tmp_path / "guides"
        d.mkdir()
        (d / "a.md").write_text("AAA")
        (d / "b.md").write_text("BBB")
        result = _read_all_guides(d)
        assert "AAA" in result
        assert "BBB" in result


class TestValidateSizeSkipsReadme:
    def test_skips_readme(self, tmp_path):
        ai_docs = tmp_path / ".ai-docs"
        ai_docs.mkdir()
        (ai_docs / "README.md").write_text("x" * 100_000)
        violations = validate_size(ai_docs, max_size=50_000)
        assert violations == []


class TestValidateCompletenessExcludesMain:
    def test_main_excluded(self, tmp_path):
        src = tmp_path / "pkg"
        src.mkdir()
        _write_py(src / "cli.py", "def main(): pass\nclass PublicThing: pass\n")
        ai_docs = tmp_path / ".ai-docs"
        ai_docs.mkdir()
        (ai_docs / "cli.md").write_text("nothing here\n")
        missing = validate_completeness(ai_docs, src)
        assert "main" not in missing
        assert "PublicThing" in missing
