.PHONY: install install-dev test test-clickhouse test-cov lint format typecheck
.PHONY: clean pre-commit docstrings-cover docs-ai-build docs-ai-check deadcode semgrep check
.PHONY: filesize duplicates exports hygiene

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest

test-clickhouse:
	pytest -m clickhouse

test-cov:
	pytest --cov=ai_pipeline_core --cov-report=html --cov-report=term --cov-fail-under=80

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	basedpyright --level warning
	basedpyright --level error -p pyrightconfig.tests.json

docstrings-cover:
	interrogate -v --fail-under 100 ai_pipeline_core

docs-ai-build:
	python -m ai_pipeline_core.docs_generator generate

docs-ai-check: docs-ai-build
	@git diff --quiet -- .ai-docs/ || (echo ".ai-docs/ is stale. Commit regenerated files."; exit 1)

deadcode:
	vulture ai_pipeline_core/ .vulture_whitelist.py --min-confidence 80

semgrep:
	semgrep --config .semgrep/ ai_pipeline_core/ tests/ --error

# File size limits: 500 lines warning, 1000 lines error (excluding blanks and comments)
filesize:
	@echo "Checking file sizes..."
	@error=0; \
	for f in $$(find ai_pipeline_core -name "*.py" -type f); do \
		lines=$$(grep -cvE '^[[:space:]]*$$|^[[:space:]]*#' "$$f" 2>/dev/null || echo 0); \
		if [ "$$lines" -gt 1000 ]; then \
			echo "ERROR: $$f has $$lines lines (max 1000)"; \
			error=1; \
		elif [ "$$lines" -gt 500 ]; then \
			echo "WARNING: $$f has $$lines lines (soft limit 500)"; \
		fi; \
	done; \
	exit $$error

# Duplicate code detection using pylint
duplicates:
	@echo "Checking for duplicate code..."
	pylint --disable=all --enable=duplicate-code ai_pipeline_core/ || true

# Export discipline: public modules with public symbols should define __all__
# Skips: internal modules (_*.py), __init__.py, test files
# NOTE: Advisory only - prints warnings but does not fail build
exports:
	@echo "Checking __all__ exports (advisory)..."
	@for f in $$(find ai_pipeline_core -name "*.py" -type f ! -name "_*" ! -name "__init__.py" ! -path "*/__pycache__/*" ! -path "*/_*/*"); do \
		if grep -qE '^(def|class) [^_]' "$$f" 2>/dev/null && ! grep -q '^__all__' "$$f" 2>/dev/null; then \
			echo "Advisory: Missing __all__: $$f"; \
		fi; \
	done

# Run all code hygiene checks
hygiene: filesize duplicates exports
	@echo "Code hygiene checks completed"

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .ruff_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

pre-commit:
	pre-commit run --all-files

check: lint typecheck deadcode semgrep docstrings-cover filesize docs-ai-check test
	@echo "All checks passed"
