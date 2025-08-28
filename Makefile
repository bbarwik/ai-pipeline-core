.PHONY: help install install-dev test test-cov test-showcase lint format typecheck clean pre-commit
.PHONY: docs-check docs-build docs-open docstrings-check docstrings-cover

help:
	@echo "Available commands:"
	@echo "  install        Install package in production mode"
	@echo "  install-dev    Install package with development dependencies"
	@echo "  test           Run tests"
	@echo "  test-cov       Run tests with coverage"
	@echo "  test-showcase  Test the showcase.py CLI example"
	@echo "  lint           Run linting checks"
	@echo "  format         Format code with ruff"
	@echo "  typecheck      Run type checking with basedpyright"
	@echo "  clean          Remove build artifacts and cache files"
	@echo "  pre-commit     Run pre-commit hooks on all files"
	@echo ""
	@echo "Documentation commands:"
	@echo "  docstrings-check  Check docstring style and correctness (D/DOC)"
	@echo "  docstrings-cover  Check docstring coverage (100% required)"
	@echo "  docs-build        Generate API.md documentation"
	@echo "  docs-check        Verify API.md is up-to-date"
	@echo "  docs-open         Generate and open API.md"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest

test-cov:
	pytest --cov=ai_pipeline_core --cov-report=html --cov-report=term --cov-fail-under=80

test-showcase:
	@echo "Testing showcase.py CLI example..."
	@TEST_DIR=$$(mktemp -d); \
	echo "Using temp directory: $$TEST_DIR"; \
	python examples/showcase.py "$$TEST_DIR" --temperature 0.5 --batch-size 2 || true; \
	if [ -d "$$TEST_DIR/input" ] && [ -d "$$TEST_DIR/analysis" ] && [ -d "$$TEST_DIR/enhanced" ]; then \
		echo "✓ Directory structure created successfully"; \
		echo "Contents:"; \
		find "$$TEST_DIR" -type f | sort; \
		rm -rf "$$TEST_DIR"; \
		echo "✓ Test passed"; \
	else \
		echo "✗ Expected directories not created"; \
		rm -rf "$$TEST_DIR"; \
		exit 1; \
	fi

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	basedpyright --level warning

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

pre-commit:
	pre-commit run --all-files

docstrings-check:
	ruff check --select D,DOC .

docstrings-cover:
	interrogate -v --fail-under 100 ai_pipeline_core

docs-build:
	pydoc-markdown

docs-check: docs-build
	@git diff --quiet -- API.md || (echo "API.md is stale. Commit regenerated file."; exit 1)

docs-open: docs-build
	@command -v xdg-open >/dev/null && xdg-open API.md || open API.md || true
