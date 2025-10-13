#!/bin/bash
# Setup development environment for DQX
# This script sets up pre-commit hooks and verifies the environment

set -e # Exit on error

echo "🚀 Setting up DQX development environment..."

# Check Python version
echo "Checking Python version..."
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
required_version="3.11"

if ! python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
  echo "❌ Error: Python $required_version or higher is required (found $python_version)"
  exit 1
fi
echo "✓ Python $python_version is compatible"

# Check uv is installed
if ! command -v uv &>/dev/null; then
  echo "❌ Error: 'uv' is not installed!"
  echo "Please install uv from: https://astral.sh/uv/install.sh"
  exit 1
fi

echo "✓ uv is installed: $(uv --version)"

# Check git repository
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "❌ Error: Not in a git repository!"
  echo "Please run this from the DQX project root"
  exit 1
fi

# Check for uncommitted changes
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "⚠️  Warning: You have uncommitted changes"
  echo "It's recommended to commit or stash them before setup"
  read -p "Continue anyway? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled"
    exit 1
  fi
fi

# Sync dependencies
echo "Installing project dependencies..."
uv sync

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
uv run pre-commit install

# Install pre-commit for CI
echo "Setting up pre-commit for CI..."
uv run pre-commit install --install-hooks

# Verify VS Code settings if VS Code is used
if [ -d ".vscode" ]; then
  echo "📝 VS Code detected. Ensure you have the Ruff extension installed."
  echo "   Extension ID: charliermarsh.ruff"
fi

# Run hooks on all files to verify setup
echo ""
echo "Verifying setup by running hooks on all files..."
echo "(This may take a minute on first run...)"
uv run pre-commit run --all-files || {
  echo ""
  echo "⚠️  Some files need fixing. This is normal for first-time setup."
  echo "Review the changes with: git diff"
  echo "Then stage and commit them."
}

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "Pre-commit hooks are now installed and will run automatically on git commit."
echo ""
echo "Useful commands:"
echo "  ./bin/run-hooks.sh          # Run on staged files"
echo "  ./bin/run-hooks.sh --all    # Run on all files"
echo "  ./bin/run-hooks.sh --fast   # Skip slow checks (mypy)"
echo "  ./bin/run-hooks.sh --fix    # Run only auto-fixing hooks"
echo ""
echo "To temporarily skip hooks (not recommended):"
echo "  git commit --no-verify"
echo "  SKIP=mypy git commit       # Skip specific hook"
echo ""
