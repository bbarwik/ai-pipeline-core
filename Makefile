SHELL := /usr/bin/env bash

# Dynamically extract Makefile stages from .PHONY declarations.
_MAKEFILE_TARGETS := $(shell grep -h '^.PHONY:' $(firstword $(MAKEFILE_LIST)) | sed 's/^.PHONY: //' | tr ' ' '\n' | sort -u)
EXTRA_ARGS = $(filter-out $(_MAKEFILE_TARGETS),$(MAKECMDGOALS))

.PHONY: install
install:
	@uv sync --locked

.PHONY: install-dev
install-dev:
	@uv sync --locked --extra dev
	@pre-commit install

.PHONY: test
test:
	@uv run pytest $(EXTRA_ARGS)

.PHONY: test-clickhouse
test-clickhouse:
	@uv run pytest -m clickhouse $(EXTRA_ARGS)

.PHONY: test-pubsub
test-pubsub:
	@uv run pytest -m pubsub $(EXTRA_ARGS)

.PHONY: test-cov
test-cov:
	@uv run pytest \
		--cov=ai_pipeline_core \
		--cov-report=html \
		--cov-report=term \
		--cov-fail-under=80 \
		$(EXTRA_ARGS)

.PHONY: test-collect
test-collect:
	@uv run pytest --collect-only -q --no-header

.PHONY: lint
lint:
	@uv run ruff check $(or $(EXTRA_ARGS),.)
	@uv run ruff format --check $(or $(EXTRA_ARGS),.)

.PHONY: format
format:
	@uv run ruff format $(or $(EXTRA_ARGS),.)
	@uv run ruff check --fix $(or $(EXTRA_ARGS),.)

.PHONY: typecheck
typecheck:
	@uv run basedpyright --level warning $(EXTRA_ARGS)
	@uv run basedpyright --level error -p pyrightconfig.tests.json $(EXTRA_ARGS)

.PHONY: docstrings-cover
docstrings-cover:
	@uv run interrogate -v --fail-under 100 ai_pipeline_core

.PHONY: docs-ai-build
docs-ai-build:
	@uv run python -m ai_pipeline_core.docs_generator generate

.PHONY: docs-ai-check
docs-ai-check: docs-ai-build
	@git diff --quiet -- .ai-docs/ \
	|| (echo ".ai-docs/ is stale. Commit regenerated files."; exit 1)

.PHONY: deadcode
deadcode:
	@uv run vulture ai_pipeline_core/ .vulture_whitelist.py --min-confidence 80

.PHONY: semgrep
semgrep:
	@uvx semgrep --config .semgrep/ ai_pipeline_core/ tests/ --error

check-claude-md:
	python scripts/check_claude_md_symbols.py

# File size limits: 500 lines warning, 1000 lines error (excluding blanks and comments)
.PHONY: filesize
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

.PHONY: duplicates
duplicates:
	@echo "Checking for duplicate code..."
	@pylint --disable=all --enable=duplicate-code ai_pipeline_core/ || true

# Export discipline: public modules with public symbols should define __all__
# Skips: internal modules (_*.py), __init__.py, test files
# NOTE: Advisory only - prints warnings but does not fail build
.PHONY: exports
exports:
	@echo "Checking __all__ exports (advisory)..."
	@for f in $$(find ai_pipeline_core -name "*.py" -type f ! -name "_*" ! -name "__init__.py" ! -path "*/__pycache__/*" ! -path "*/_*/*"); do \
		if grep -qE '^(def|class) [^_]' "$$f" 2>/dev/null && ! grep -q '^__all__' "$$f" 2>/dev/null; then \
			echo "Advisory: Missing __all__: $$f"; \
		fi; \
	done

# Run all code hygiene checks
.PHONY: hygiene
hygiene: filesize duplicates exports
	@echo "Code hygiene checks completed"


.PHONY: clean
clean:
	@rm -rf build/ dist/ *.egg-info .pytest_cache/ .ruff_cache/ htmlcov/ .coverage
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete

.PHONY: pre-commit
pre-commit:
	@pre-commit run --all-files

.PHONY: lint-pre-commit-config
lint-pre-commit-config:
	@pre-commit validate-config .pre-commit-config.yaml

.PHONY: check
check: lint typecheck deadcode semgrep docstrings-cover filesize check-claude-md docs-ai-check test-collect test
	@echo "All checks passed"
