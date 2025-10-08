#!/bin/bash
# Run pre-commit hooks manually with options
# Usage: ./bin/run-hooks.sh [options] [files...]
#
# Options:
#   --all     Run on all files
#   --fast    Skip slow hooks (mypy)
#   --fix     Only run hooks that auto-fix issues

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse options
RUN_ALL=false
SKIP_HOOKS=""
HOOK_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --fast)
            SKIP_HOOKS="mypy"
            shift
            ;;
        --fix)
            HOOK_ID="ruff-format,ruff-check,trailing-whitespace,end-of-file-fixer"
            shift
            ;;
        *)
            break
            ;;
    esac
done

echo "🔍 Running pre-commit hooks..."

# Build command
CMD="uv run pre-commit run"

if [ -n "$SKIP_HOOKS" ]; then
    export SKIP="$SKIP_HOOKS"
    echo -e "${YELLOW}⚡ Skipping slow hooks: $SKIP_HOOKS${NC}"
fi

if [ -n "$HOOK_ID" ]; then
    # Run specific hooks
    IFS=',' read -ra HOOKS <<< "$HOOK_ID"
    for hook in "${HOOKS[@]}"; do
        echo -e "Running $hook..."
        if [ "$RUN_ALL" = true ]; then
            $CMD "$hook" --all-files || true
        elif [ $# -eq 0 ]; then
            $CMD "$hook" || true
        else
            $CMD "$hook" --files "$@" || true
        fi
    done
else
    # Run all hooks
    if [ "$RUN_ALL" = true ]; then
        $CMD --all-files
    elif [ $# -eq 0 ]; then
        echo "Checking staged files..."
        $CMD
    else
        echo "Checking files: $@"
        $CMD --files "$@"
    fi
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All hooks passed!${NC}"
else
    echo -e "${RED}✗ Some hooks failed. Please fix the issues above.${NC}"
    echo -e "${YELLOW}Tip: Use './bin/run-hooks.sh --fix' to run only auto-fixing hooks${NC}"
    exit 1
fi
