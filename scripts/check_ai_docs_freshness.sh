#!/usr/bin/env bash
# Pre-commit hook: regenerate .ai-docs/ and fail if files changed.
set -e

python -m ai_pipeline_core.docs_generator generate > /dev/null 2>&1 || true

if ! git diff --quiet -- .ai-docs/; then
    echo "ERROR: .ai-docs/ is stale after regeneration."
    echo "Run: make docs-ai-build && git add .ai-docs/"
    exit 1
fi
