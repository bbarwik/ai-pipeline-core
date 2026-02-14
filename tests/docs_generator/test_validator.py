import dataclasses

import pytest

from ai_pipeline_core.docs_generator.validator import (
    HASH_FILE,
    ValidationResult,
    compute_source_hash,
    validate_all,
    validate_completeness,
    validate_freshness,
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


def test_compute_source_hash_deterministic(repo):
    _, src, tests = repo
    h1 = compute_source_hash(src, tests)
    h2 = compute_source_hash(src, tests)
    assert h1 == h2


def test_compute_source_hash_changes_on_content(repo):
    _, src, tests = repo
    h1 = compute_source_hash(src, tests)
    (src / "mod.py").write_text("# changed\n")
    h2 = compute_source_hash(src, tests)
    assert h1 != h2


def test_compute_source_hash_changes_on_path(repo):
    _, src, tests = repo
    h1 = compute_source_hash(src, tests)
    _write_py(src / "new.py", "x = 1\n")
    h2 = compute_source_hash(src, tests)
    assert h1 != h2


def test_compute_source_hash_includes_tests(repo):
    _, src, tests = repo
    h1 = compute_source_hash(src, tests)
    _write_py(tests / "test_extra.py", "def test_bar(): pass\n")
    h2 = compute_source_hash(src, tests)
    assert h1 != h2


def test_compute_source_hash_uses_relative_paths(tmp_path):
    src = tmp_path / "ai_pipeline_core"
    src.mkdir()
    tests = tmp_path / "tests"
    tests.mkdir()
    _write_py(src / "a.py", "x = 1\n")
    h = compute_source_hash(src, tests)
    assert isinstance(h, str) and len(h) == 64


def test_validate_freshness_matching(repo):
    tmp_path, src, tests = repo
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    h = compute_source_hash(src, tests)
    (ai_docs / HASH_FILE).write_text(h + "\n")
    assert validate_freshness(ai_docs, src, tests) is True


def test_validate_freshness_stale(repo):
    tmp_path, src, tests = repo
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    (ai_docs / HASH_FILE).write_text("oldhash\n")
    assert validate_freshness(ai_docs, src, tests) is False


def test_validate_freshness_missing_hash_file(repo):
    tmp_path, src, tests = repo
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    assert validate_freshness(ai_docs, src, tests) is False


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
    h = compute_source_hash(src, tests)
    (ai_docs / HASH_FILE).write_text(h + "\n")
    (ai_docs / "mod.md").write_text("class Foo:\n    pass\n")
    result = validate_all(ai_docs, src, tests)
    assert result.is_valid is True


def test_validate_all_fail_stale(repo):
    tmp_path, src, tests = repo
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    (ai_docs / HASH_FILE).write_text("wrong\n")
    (ai_docs / "mod.md").write_text("class Foo:\n    pass\n")
    result = validate_all(ai_docs, src, tests)
    assert result.is_fresh is False
    assert result.is_valid is False


def test_validate_all_fail_incomplete(repo):
    tmp_path, src, tests = repo
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    h = compute_source_hash(src, tests)
    (ai_docs / HASH_FILE).write_text(h + "\n")
    (ai_docs / "mod.md").write_text("nothing\n")
    result = validate_all(ai_docs, src, tests)
    assert len(result.missing_symbols) > 0
    assert result.is_valid is False


def test_validate_all_oversized_still_valid(repo):
    tmp_path, src, tests = repo
    ai_docs = tmp_path / ".ai-docs"
    ai_docs.mkdir()
    h = compute_source_hash(src, tests)
    (ai_docs / HASH_FILE).write_text(h + "\n")
    (ai_docs / "mod.md").write_text("class Foo:\n" + "x" * 60_000)
    result = validate_all(ai_docs, src, tests)
    assert len(result.size_violations) > 0
    assert result.is_valid is True


def test_validation_result_frozen():
    result = ValidationResult(is_fresh=True, missing_symbols=(), size_violations=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.is_fresh = False  # type: ignore[misc]


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
