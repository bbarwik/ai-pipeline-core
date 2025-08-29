#!/bin/bash

# This script lists all files in the current repository that are not ignored by git,
# and prints their contents. It is intended to be used when repository files need to be
# provided within an AI prompt.
# Some files and directories are excluded entirely via IGNORE_PATTERNS defined below.

# Check if git is installed
if ! command -v git &> /dev/null
then
    echo "git could not be found. Please install git to run this script."
    exit 1
fi

# Check if inside a git repository
if ! git rev-parse --is-inside-work-tree &> /dev/null
then
    echo "Not inside a git repository."
    exit 1
fi

SEPARATOR="========================================"

# Patterns to ignore completely (files will not be listed)
IGNORE_PATTERNS=(
    ".claude/*"
    ".github/*"
    ".devcontainer/*"
    "tests/test_data/*"
    "test_data/*"
    "License"
    "LICENSE"
    ".gitignore"
    ".gitattributes"
    ".env.example"
    "scripts/*"
    "dependencies_docs/*"
    "tests/*"
    "CLAUDE.md"
    "Makefile"
    "API.md"
)

# Helper: return 0 if the given file path should be ignored
should_ignore() {
    local file_path_to_check="$1"
    for pattern in "${IGNORE_PATTERNS[@]}"; do
        if [[ "$file_path_to_check" == $pattern ]]; then
            return 0
        fi
    done
    return 1
}

# Get all tracked and untracked files, excluding standard gitignores.
# The files are piped to a while loop to handle file paths with spaces correctly.
git ls-files --cached --others --exclude-standard | sort | while IFS= read -r file_path; do
    # This check is important because ls-files can list files that are staged for deletion but no longer exist on disk.
    if [ ! -f "$file_path" ]; then
        continue
    fi

    # Skip files matching ignore patterns
    if should_ignore "$file_path"; then
        continue
    fi

    echo "" # for spacing
    echo "$SEPARATOR"
    echo "FILE: $file_path"
    echo "$SEPARATOR"

    if [[ "$file_path" == *test_data* || "$file_path" == *scripts/* || "$file_path" == *dependencies_docs/* ]]; then
        echo "[contents skipped]"
    # Check if the file is binary. The `file` command is a good heuristic.
    # The `-b` option prevents printing the filename.
    elif file -b --mime-encoding "$file_path" | grep -q "binary"; then
        echo "[Binary file - contents not shown]"
    else
        # Check if file is empty, like the python script does.
        if [ -s "$file_path" ]; then
            cat "$file_path"
        else
            echo "[Empty file]"
        fi
    fi
    echo "$SEPARATOR"
done
