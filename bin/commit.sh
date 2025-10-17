#!/bin/bash
# Helper script for creating conventional commits
# Usage: ./bin/commit.sh

set -e

echo "🎯 Creating a conventional commit..."
echo ""
echo "This will guide you through creating a properly formatted commit message."
echo "For more options, you can also use: uv run cz commit"
echo ""

# Run commitizen in interactive mode
uv run cz commit

echo ""
echo "✅ Commit created successfully!"
echo ""
echo "Tip: You can also commit manually with conventional format:"
echo '  git commit -m "type(scope): description"'
echo ""
