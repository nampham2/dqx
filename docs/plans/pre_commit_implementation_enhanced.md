# Pre-commit Hooks Implementation Plan for DQX (Enhanced Version)

## Overview
This enhanced guide provides a comprehensive, step-by-step approach to implement pre-commit hooks in the DQX project with performance optimizations, additional quality checks, and developer experience improvements.

**Time estimate**: 2-3 hours for full implementation
**Improvements**: Performance optimizations, CI/CD integration, VS Code setup, enhanced scripts

## Background

### What are pre-commit hooks?
Pre-commit hooks are scripts that run automatically before each git commit. They check your code for issues and can prevent commits that don't meet quality standards. Think of them as automated code reviewers that catch problems before they enter the codebase.

### Why do we need them?
- **Consistency**: Ensures all code follows the same standards
- **Early error detection**: Catches issues before code review
- **Time saving**: Automates repetitive checks
- **Clean git history**: Prevents "fix linting" commits
- **Team alignment**: Everyone follows the same quality standards automatically

### Current DQX Quality Tools
- **ruff**: Python linter and formatter (configured in pyproject.toml)
- **mypy**: Static type checker for Python
- **uv**: Fast Python package manager (replacement for pip/poetry)

## Prerequisites

### Required Knowledge
- Basic git commands (`git add`, `git commit`, `git status`)
- Basic Python and command line usage
- Understanding of virtual environments

### Environment Setup
1. Ensure you have `uv` installed:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Verify your environment:
   ```bash
   cd /path/to/dqx
   uv --version  # Should show a version number
   python --version  # Should show Python 3.11.x or 3.12.x

   # Verify Python version is compatible
   python -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ required'"
   ```

3. Sync project dependencies:
   ```bash
   uv sync
   ```

## Implementation Tasks

### Task 1: Add pre-commit to project dependencies

**Objective**: Add pre-commit package to development dependencies

**Files to modify**:
- `pyproject.toml`

**Test First (TDD)**:
```bash
# This should fail initially
uv run pre-commit --version
```

**Implementation**:
1. Open `pyproject.toml`
2. Find the `[dependency-groups]` section
3. In the `dev` list, add `"pre-commit>=3.5.0",` after the mypy line
4. The dev section should look like:
   ```toml
   dev = [
       "faker>=37.5.3",
       "mypy>=1.17.1",
       "pre-commit>=3.5.0",  # Add this line
       "pytest>=8.4.1",
       # ... rest of dependencies
   ]
   ```

**Verify**:
```bash
# Sync dependencies
uv sync

# Test - this should now work
uv run pre-commit --version
```

**Commit**:
```bash
git add pyproject.toml uv.lock
git commit -m "build: add pre-commit to dev dependencies"
```

### Task 2: Create optimized pre-commit configuration

**Objective**: Create pre-commit config with performance optimizations and proper hook ordering

**Files to create**:
- `.pre-commit-config.yaml` (root directory)

**Test First (TDD)**:
Create a test file to verify hooks work:
```bash
# Create a deliberately bad Python file
echo 'import os;import sys;x=1' > test_hooks.py

# This should fail (no config yet)
uv run pre-commit run --files test_hooks.py
```

**Implementation**:
Create `.pre-commit-config.yaml` in the project root:
```yaml
# .pre-commit-config.yaml
# See https://pre-commit.com for more information

# Don't run on files in these directories
exclude: '^(\.git|\.mypy_cache|\.pytest_cache|\.ruff_cache|\.venv|dist|build|uv\.lock)/'

# Run all hooks on all files by default
fail_fast: false

# Set minimum pre-commit version for consistency
minimum_pre_commit_version: '3.5.0'

repos:
  # Pre-commit hooks for general file quality (fast checks first)
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      # Fast syntax checks
      - id: check-ast
        name: Check Python syntax

      # Security - no private keys
      - id: detect-private-key

      # Check for merge conflicts
      - id: check-merge-conflict

      # Debug statements
      - id: debug-statements
        name: Check for debug statements

      # File quality checks
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-case-conflict
      - id: mixed-line-ending
        args: ['--fix=lf']
      - id: fix-byte-order-marker

      # Content validation
      - id: check-yaml
        args: ['--allow-multiple-documents']
      - id: check-toml
      - id: check-json
        # Note: This will fail on JSON with comments (like .vscode/settings.json)
        exclude: '.vscode/.*\.json$'

      # Prevent large files (increased limit for lock files)
      - id: check-added-large-files
        args: ['--maxkb=1000']
        exclude: 'uv\.lock$'

      # Python specific
      - id: check-docstring-first

  # Local hooks that use our project's virtual environment
  - repo: local
    hooks:
      # Ruff - Format first (modifies files)
      - id: ruff-format
        name: ruff format
        entry: uv run ruff format
        language: system
        types: [python]
        require_serial: true

      # Ruff - Then lint (may modify files)
      - id: ruff-check
        name: ruff check
        entry: uv run ruff check --fix
        language: system
        types: [python]
        require_serial: true

      # MyPy - Static type checker (doesn't modify files)
      - id: mypy
        name: mypy type check
        entry: uv run mypy
        language: system
        types: [python]
        require_serial: true
        pass_filenames: false
        args: [src, tests]
        # Note: We use full project checking to ensure type consistency
        # For faster commits, developers can use SKIP=mypy
```

**Verify**:
```bash
# Install pre-commit hooks
uv run pre-commit install

# Test on our bad file (should show errors)
uv run pre-commit run --files test_hooks.py

# Clean up test file
rm test_hooks.py
```

**Clean up existing codebase** (optional but recommended):
```bash
# Run on all files to establish baseline
uv run pre-commit run --all-files

# Review and commit any automatic fixes
git diff
git add -u
git commit -m "style: apply pre-commit hooks to existing code"
```

**Commit**:
```bash
git add .pre-commit-config.yaml
git commit -m "feat: add optimized pre-commit config with formatting and linting"
```

### Task 3: Test pre-commit hooks are working

**Objective**: Verify hooks prevent bad commits

**Test First (TDD)**:
Create test scenarios:

1. **Test linting errors are caught**:
   ```bash
   # Create a file with linting issues
   cat > test_lint.py << 'EOF'
   import os
   import sys
   def bad_function( x,y ):
       return x+y
   EOF

   # Try to stage and commit (should fail)
   git add test_lint.py
   git commit -m "test: bad lint" || echo "Good! Commit was blocked"

   # Clean up
   git reset test_lint.py
   rm test_lint.py
   ```

2. **Test type errors are caught**:
   ```bash
   # Create a file with type errors
   cat > test_types.py << 'EOF'
   def add_numbers(a: int, b: int) -> int:
       return a + b

   result: str = add_numbers(1, 2)  # Type error!
   EOF

   # This should also fail
   git add test_types.py
   git commit -m "test: bad types" || echo "Good! Commit was blocked"

   # Clean up
   git reset test_types.py
   rm test_types.py
   ```

3. **Test debug statements are caught**:
   ```bash
   # Create a file with debug statements
   cat > test_debug.py << 'EOF'
   def calculate(x, y):
       print("Debug: calculating", x, y)  # Debug print
       result = x + y
       breakpoint()  # Debug breakpoint
       return result
   EOF

   # This should fail
   git add test_debug.py
   git commit -m "test: debug statements" || echo "Good! Commit was blocked"

   # Clean up
   git reset test_debug.py
   rm test_debug.py
   ```

**Expected behavior**: All commits should be blocked by pre-commit hooks.

**Commit**:
No commit needed for this task - it's just verification.

### Task 4: Configure VS Code integration

**Objective**: Ensure VS Code works seamlessly with pre-commit hooks

**Files to create/modify**:
- `.vscode/settings.json`

**Implementation**:
Create or update `.vscode/settings.json`:
```bash
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll.ruff": "explicit",
      "source.organizeImports.ruff": "explicit"
    }
  },
  "ruff.format.args": [],
  "ruff.lint.args": [],
  "python.analysis.typeCheckingMode": "strict",
  "python.analysis.autoImportCompletions": true,
  "python.analysis.diagnosticMode": "workspace"
}
EOF
```

**Verify**:
- Open VS Code in the project directory
- Install the Ruff extension if not already installed
- Save a Python file and verify it gets formatted automatically

**Commit**:
```bash
git add .vscode/settings.json
git commit -m "feat: configure VS Code to use ruff formatter"
```

### Task 5: Add general file quality hooks

**Objective**: Already covered in Task 2 with the optimized configuration

This task has been integrated into Task 2 for better organization and performance.

### Task 6: Create enhanced hook runner script

**Objective**: Create script with more options for running hooks

**Files to create**:
- `bin/run-hooks.sh`

**Implementation**:
```bash
cat > bin/run-hooks.sh << 'EOF'
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
EOF

chmod +x bin/run-hooks.sh
```

**Verify**:
```bash
# Test the script with different options
./bin/run-hooks.sh --all
./bin/run-hooks.sh --fast
./bin/run-hooks.sh --fix src/dqx/api.py
```

**Commit**:
```bash
git add bin/run-hooks.sh
git commit -m "feat: add enhanced hook runner script with options"
```

### Task 7: Create comprehensive setup script

**Objective**: Enhanced setup script with more checks and better user experience

**Files to create**:
- `bin/setup-dev-env.sh`

**Implementation**:
```bash
cat > bin/setup-dev-env.sh << 'EOF'
#!/bin/bash
# Setup development environment for DQX
# This script sets up pre-commit hooks and verifies the environment

set -e  # Exit on error

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
if ! command -v uv &> /dev/null; then
    echo "❌ Error: 'uv' is not installed!"
    echo "Please install uv from: https://astral.sh/uv/install.sh"
    exit 1
fi

echo "✓ uv is installed: $(uv --version)"

# Check git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
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
EOF

chmod +x bin/setup-dev-env.sh
```

**Verify**:
```bash
# Test the setup script
./bin/setup-dev-env.sh
```

**Commit**:
```bash
git add bin/setup-dev-env.sh
git commit -m "feat: add comprehensive developer setup script"
```

### Task 8: Add CI/CD integration

**Objective**: Ensure pre-commit runs in CI pipelines

**Files to create**:
- `.github/workflows/pre-commit.yml`

**Implementation**:
```bash
mkdir -p .github/workflows
cat > .github/workflows/pre-commit.yml << 'EOF'
name: Pre-commit

on:
  pull_request:
  push:
    branches: [main, develop]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for better hook performance

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install uv
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          echo "$HOME/.local/bin" >> $GITHUB_PATH

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.cache/uv
            .venv
          key: ${{ runner.os }}-uv-${{ hashFiles('pyproject.toml', 'uv.lock') }}
          restore-keys: |
            ${{ runner.os }}-uv-

      - name: Install dependencies
        run: uv sync

      - name: Cache pre-commit hooks
        uses: actions/cache@v3
        with:
          path: ~/.cache/pre-commit
          key: ${{ runner.os }}-pre-commit-${{ hashFiles('.pre-commit-config.yaml') }}
          restore-keys: |
            ${{ runner.os }}-pre-commit-

      - name: Run pre-commit
        run: uv run pre-commit run --all-files --show-diff-on-failure
EOF
```

**Verify**:
This will be tested when you push to a branch or create a pull request.

**Commit**:
```bash
git add .github/workflows/pre-commit.yml
git commit -m "ci: add pre-commit workflow with caching"
```

### Task 9: Update documentation

**Objective**: Document pre-commit usage for developers

**Files to modify**:
- `README.md`

**Implementation**:
Add a new section to README.md after the "Development Setup" section:

```markdown
### Pre-commit Hooks

This project uses pre-commit hooks to maintain code quality. Hooks run automatically before each commit.

#### Quick Setup

```bash
# Run the setup script (recommended)
./bin/setup-dev-env.sh

# Or manually install hooks
uv run pre-commit install
```

#### What Gets Checked

- **Code formatting**: Ruff automatically formats Python code
- **Linting**: Ruff checks for code quality issues
- **Type checking**: MyPy validates type annotations
- **Debug detection**: Catches forgotten print/breakpoint statements
- **File quality**: Trailing whitespace, file endings, large files
- **Security**: Detects accidentally committed private keys
- **Syntax validation**: Python, YAML, TOML, JSON files

#### Manual Usage

```bash
# Run on all files
./bin/run-hooks.sh --all

# Run on specific files
./bin/run-hooks.sh src/dqx/api.py tests/test_api.py

# Run only formatting hooks (fast)
./bin/run-hooks.sh --fix

# Skip slow hooks like mypy
./bin/run-hooks.sh --fast

# Skip hooks temporarily (not recommended)
git commit --no-verify -m "emergency fix"
```

#### Fixing Issues

If pre-commit blocks your commit:

1. **Read the error message** - it tells you exactly what's wrong
2. **Let hooks auto-fix** - many issues are fixed automatically
3. **Review changes** - check what was fixed with `git diff`
4. **Re-stage files** - `git add` the fixed files
5. **Commit again** - your commit should now succeed

Example:
```bash
$ git commit -m "feat: add new feature"
ruff.....................................................................Failed
- hook id: ruff-check
- exit code: 1
  Fixed 1 error:
  - src/dqx/new_feature.py:
    1 × I001 (unsorted-imports)

# Ruff fixed the import order. Check and re-commit:
$ git add src/dqx/new_feature.py
$ git commit -m "feat: add new feature"
```

#### VS Code Integration

VS Code will automatically use the same formatting rules if you have the Ruff extension installed:

1. Install the Ruff extension (ID: `charliermarsh.ruff`)
2. Your code will be formatted on save
3. Linting errors will appear inline

#### Performance Tips

- For faster commits during development:
  ```bash
  SKIP=mypy git commit -m "wip: quick save"  # Skip type checking
  ./bin/run-hooks.sh --fast                   # Run without mypy
  ```

- Pre-commit caches results, so subsequent runs on unchanged files are instant

#### CI/CD Integration

Pre-commit runs automatically in CI on pull requests. To run the same checks locally:
```bash
./bin/run-hooks.sh --all
```
```

**Verify**:
Read through the documentation to ensure it's clear and comprehensive.

**Commit**:
```bash
git add README.md
git commit -m "docs: add comprehensive pre-commit usage documentation"
```

### Task 10: Create enhanced troubleshooting guide

**Objective**: Help developers solve common pre-commit issues

**Files to create**:
- `docs/pre_commit_troubleshooting.md`

**Implementation**:
```bash
cat > docs/pre_commit_troubleshooting.md << 'EOF'
# Pre-commit Troubleshooting Guide

## Common Issues and Solutions

### 1. "pre-commit: command not found"

**Problem**: Pre-commit is not installed or not in PATH.

**Solution**:
```bash
# Ensure you're using uv to run pre-commit
uv sync
uv run pre-commit --version
```

### 2. MyPy fails with "Cannot find module"

**Problem**: MyPy can't find project modules.

**Solution**:
```bash
# MyPy needs to run from project root
cd /path/to/dqx
uv run mypy src tests

# If still failing, check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

### 3. Hooks are slow

**Problem**: Hooks take too long to run.

**Solutions**:
- Run hooks only on changed files (default behavior)
- Use performance options:
  ```bash
  # Skip slow hooks
  ./bin/run-hooks.sh --fast

  # Skip specific hook
  SKIP=mypy git commit -m "wip: testing"

  # Run only formatting
  ./bin/run-hooks.sh --fix
  ```
- Pre-commit caches results - first run is slowest

### 4. "Fixed X errors" but commit still fails

**Problem**: Ruff fixed issues but files need to be re-staged.

**Solution**:
```bash
# After hooks run and fix issues:
git add -u  # Stage all modified files
git commit  # Try commit again
```

### 5. Large file detected

**Problem**: Trying to commit a file larger than 1MB.

**Solutions**:
- Add file to `.gitignore` if it shouldn't be committed
- Use Git LFS for large files that must be versioned
- Compress the file if possible
- Override for specific files in `.pre-commit-config.yaml`:
  ```yaml
  - id: check-added-large-files
    args: ['--maxkb=1000']
    exclude: 'path/to/large/file.ext'
  ```

### 6. Conflicts with IDE formatting

**Problem**: Your IDE reformats code differently than ruff.

**Solution for VS Code**:
1. Install Ruff extension: `charliermarsh.ruff`
2. Ensure `.vscode/settings.json` exists (created in setup)
3. Restart VS Code

**Solution for PyCharm**:
1. Configure external tool:
   - File → Settings → Tools → External Tools
   - Add new tool: `ruff format`
   - Program: `$ProjectFileDir$/bin/run-hooks.sh`
   - Arguments: `--fix $FilePath$`

### 7. Debug statements detected

**Problem**: Pre-commit found print() or breakpoint() statements.

**Solution**:
```bash
# Find all debug statements
grep -rn "print(" src/
grep -rn "breakpoint(" src/

# Remove or convert to proper logging
# Replace print() with logging.debug()
```

### 8. JSON validation fails on VS Code settings

**Problem**: check-json fails on `.vscode/settings.json` with comments.

**Solution**: Already handled in config - VS Code JSON files are excluded.

### 9. Hooks not running on commit

**Problem**: Git commit doesn't trigger hooks.

**Solutions**:
```bash
# Check if hooks are installed
ls -la .git/hooks/pre-commit

# Reinstall hooks
uv run pre-commit install

# Check git configuration
git config core.hooksPath
```

### 10. Different behavior in CI vs local

**Problem**: Pre-commit passes locally but fails in CI.

**Solutions**:
- Ensure you're running on all files locally:
  ```bash
  ./bin/run-hooks.sh --all
  ```
- Check Python version matches CI
- Update pre-commit hooks:
  ```bash
  uv run pre-commit autoupdate
  ```

## Performance Optimization

### For Large Codebases

1. **Use file filtering**:
   ```bash
   # Check only Python files in src/
   ./bin/run-hooks.sh src/**/*.py
   ```

2. **Parallel execution** (if available):
   ```bash
   # Some hooks support parallel execution
   uv run pre-commit run --all-files --jobs 4
   ```

3. **Skip unchanged files**:
   Pre-commit automatically caches results for unchanged files.

### During Active Development

1. **Use --fast flag**:
   ```bash
   ./bin/run-hooks.sh --fast  # Skips mypy
   ```

2. **Fix-only mode**:
   ```bash
   ./bin/run-hooks.sh --fix  # Only formatting
   ```

3. **Defer to CI**:
   ```bash
   git commit --no-verify -m "wip"  # Emergency only!
   git push  # Let CI run full checks
   ```

## Disabling Hooks (Emergency Only!)

In emergencies, you can bypass hooks:
```bash
# Skip all hooks
git commit --no-verify -m "emergency: fix production issue"

# Skip specific hook
SKIP=mypy git commit -m "wip: debugging"

# Disable temporarily
uv run pre-commit uninstall
# ... make commits ...
uv run pre-commit install  # Re-enable!
```

**Note**: Always run `./bin/run-hooks.sh --all` before pushing to ensure code quality.

## Getting Help

1. Check the error message - it usually tells you exactly what to fix
2. Run hooks on just the problem file: `uv run pre-commit run --files <file>`
3. Check hook configuration in `.pre-commit-config.yaml`
4. Enable verbose output: `uv run pre-commit run --verbose`
5. Ask team members or check pre-commit documentation: https://pre-commit.com/
EOF
```

**Verify**:
```bash
# Ensure the file was created
ls -la docs/pre_commit_troubleshooting.md
```

**Commit**:
```bash
git add docs/pre_commit_troubleshooting.md
git commit -m "docs: add comprehensive pre-commit troubleshooting guide"
```

### Task 11: Final verification

**Objective**: Ensure everything works end-to-end

**Test scenarios**:

1. **Test fresh setup**:
   ```bash
   # Uninstall hooks
   uv run pre-commit uninstall

   # Run setup script
   ./bin/setup-dev-env.sh
   ```

2. **Test actual commit**:
   ```bash
   # Make a small change
   echo "# Test comment" >> src/dqx/__init__.py

   # Commit should work
   git add src/dqx/__init__.py
   git commit -m "test: verify pre-commit hooks work"

   # Revert the test
   git reset --soft HEAD~1
   git checkout src/dqx/__init__.py
   ```

3. **Test all hooks**:
   ```bash
   ./bin/run-hooks.sh --all
   ```

4. **Test CI simulation**:
   ```bash
   # Run exactly what CI will run
   uv run pre-commit run --all-files --show-diff-on-failure
   ```

## Summary

You've successfully created an enhanced pre-commit implementation plan for DQX!

### Key Improvements Over Original:
1. ✅ **Performance optimizations**: Better hook ordering, mypy configuration
2. ✅ **Ruff formatting**: Separate format hook for consistency
3. ✅ **Debug detection**: Catches forgotten debugging code
4. ✅ **Enhanced scripts**: Flexible options (--fast, --fix, --all)
5. ✅ **VS Code integration**: Seamless editor experience
6. ✅ **CI/CD ready**: GitHub Actions with caching
7. ✅ **Better limits**: Practical file size limits
8. ✅ **Comprehensive troubleshooting**: Detailed problem-solving guide

### What was implemented:
1. ✅ Pre-commit package added to dependencies
2. ✅ Optimized configuration with ruff format + check
3. ✅ Debug statement detection
4. ✅ VS Code configuration for consistency
5. ✅ Enhanced runner script with options
6. ✅ Comprehensive setup script with checks
7. ✅ CI/CD workflow with caching
8. ✅ Updated documentation with performance tips
9. ✅ Detailed troubleshooting guide

### Key files created/modified:
- `.pre-commit-config.yaml` - Optimized hook configuration
- `pyproject.toml` - Added pre-commit dependency
- `.vscode/settings.json` - VS Code integration
- `bin/run-hooks.sh` - Enhanced manual hook runner
- `bin/setup-dev-env.sh` - Comprehensive developer setup
- `.github/workflows/pre-commit.yml` - CI/CD integration
- `README.md` - Usage documentation
- `docs/pre_commit_troubleshooting.md` - Troubleshooting guide

### For the implementing developer:
1. Follow tasks in order
2. Test each step before proceeding
3. Commit frequently as indicated
4. Run `./bin/setup-dev-env.sh` to activate hooks
5. Use `./bin/run-hooks.sh --fast` during active development

## Testing Checklist

Before considering this complete, verify:
- [ ] `git commit` triggers pre-commit hooks
- [ ] Badly formatted Python code gets auto-fixed
- [ ] Type errors prevent commits
- [ ] Debug statements are caught
- [ ] Large files are rejected
- [ ] VS Code formats on save
- [ ] CI workflow runs on push
- [ ] Documentation is clear and helpful
- [ ] All scripts are executable and work
- [ ] Performance options work (--fast, --fix)

## Additional Resources

- Pre-commit documentation: https://pre-commit.com/
- Ruff documentation: https://docs.astral.sh/ruff/
- MyPy documentation: https://mypy-lang.org/
- GitHub Actions: https://docs.github.com/en/actions

---

This enhanced implementation plan provides a robust, performant, and developer-friendly pre-commit setup that will significantly improve code quality and consistency in the DQX project.
