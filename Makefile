.PHONY: install install-dev test test-clickhouse test-cov lint format typecheck
.PHONY: clean pre-commit docstrings-cover docs-ai-build docs-ai-check deadcode semgrep check

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

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .ruff_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

pre-commit:
	pre-commit run --all-files

check: lint typecheck deadcode semgrep docstrings-cover test
	@echo "All checks passed"
