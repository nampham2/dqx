#!/bin/bash
# Wrapper script for commitizen to work with pre-commit commit-msg hook
# Pre-commit sets PRE_COMMIT_COMMIT_MSG_FILENAME for commit-msg hooks

set -e

# For commit-msg hooks, pre-commit provides the file via environment variable
# or as the first argument
if [ -n "$PRE_COMMIT_COMMIT_MSG_FILENAME" ]; then
  COMMIT_MSG_FILE="$PRE_COMMIT_COMMIT_MSG_FILENAME"
elif [ -n "$1" ]; then
  COMMIT_MSG_FILE="$1"
else
  # Default to .git/COMMIT_EDITMSG
  COMMIT_MSG_FILE=".git/COMMIT_EDITMSG"
fi

if [ ! -f "$COMMIT_MSG_FILE" ]; then
  echo "Error: Commit message file not found at: $COMMIT_MSG_FILE"
  exit 1
fi

# Run commitizen check with the commit message file
uv run cz check --commit-msg-file "$COMMIT_MSG_FILE"
