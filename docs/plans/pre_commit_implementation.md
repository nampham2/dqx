# Pre-commit Hooks Implementation Plan for DQX

## Overview
This document provides a comprehensive, step-by-step guide to implement pre-commit hooks in the DQX project. Follow each task in order, committing frequently as indicated.

**Time estimate**: 2-3 hours for full implementation

## Background

### What are pre-commit hooks?
Pre-commit hooks are scripts that run automatically before each git commit. They check your code for issues and can prevent commits that don't meet quality standards. Think of them as automated code reviewers that catch problems before they enter the codebase.

### Why do we need them?
- **Consistency**: Ensures all code follows the same standards
- **Early error detection**: Catches issues before code review
- **Time saving**: Automates repetitive checks
- **Clean git history**: Prevents "fix linting" commits

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

### Task 2: Create basic pre-commit configuration

**Objective**: Create minimal pre-commit config with ruff and mypy

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
exclude: '^(\.git|\.mypy_cache|\.pytest_cache|\.ruff_cache|\.venv|dist|build)/'

# Run all hooks on all files by default
fail_fast: false

repos:
  # Local hooks that use our project's virtual environment
  - repo: local
    hooks:
      # Ruff - Python linter and formatter
      - id: ruff
        name: ruff lint
        entry: uv run ruff check --fix
        language: system
        types: [python]
        require_serial: true

      # MyPy - Static type checker
      - id: mypy
        name: mypy type check
        entry: uv run mypy
        language: system
        types: [python]
        require_serial: true
        pass_filenames: false
        args: [src, tests]
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

**Commit**:
```bash
git add .pre-commit-config.yaml
git commit -m "feat: add basic pre-commit config with ruff and mypy"
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

**Expected behavior**: Both commits should be blocked by pre-commit hooks.

**Commit**:
No commit needed for this task - it's just verification.

### Task 4: Add general file quality hooks

**Objective**: Add hooks for whitespace, file endings, and syntax

**Files to modify**:
- `.pre-commit-config.yaml`

**Test First (TDD)**:
Create test files with various issues:
```bash
# File with trailing whitespace
echo "hello world   " > test_whitespace.txt

# File without newline at end
printf "no newline at end" > test_no_newline.txt

# Invalid YAML
cat > test_bad.yaml << 'EOF'
invalid yaml content
  - this is: not valid
    yaml file
EOF
```

**Implementation**:
Add to `.pre-commit-config.yaml` before the `local` repo section:
```yaml
repos:
  # Pre-commit hooks for general file quality
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      # Remove trailing whitespace
      - id: trailing-whitespace
      
      # Ensure files end with newline
      - id: end-of-file-fixer
      
      # Check for merge conflicts
      - id: check-merge-conflict
      
      # Validate YAML files
      - id: check-yaml
      
      # Validate TOML files
      - id: check-toml
      
      # Check JSON files
      - id: check-json
      
      # Prevent large files
      - id: check-added-large-files
        args: ['--maxkb=500']
      
      # Check Python syntax
      - id: check-ast
      
      # Security - no private keys
      - id: detect-private-key
      
      # Fix byte order markers
      - id: fix-byte-order-marker
      
      # Check case conflicts
      - id: check-case-conflict
      
      # Fix line endings
      - id: mixed-line-ending
        args: ['--fix=lf']

  # Local hooks (existing content)
  - repo: local
    # ... rest of your existing local hooks
```

**Verify**:
```bash
# Re-install to get new hooks
uv run pre-commit install

# Test whitespace fix
uv run pre-commit run trailing-whitespace --files test_whitespace.txt
cat test_whitespace.txt  # Should have no trailing spaces

# Test newline fix
uv run pre-commit run end-of-file-fixer --files test_no_newline.txt
tail -c 5 test_no_newline.txt | od -c  # Should show \n at end

# Test YAML check
uv run pre-commit run check-yaml --files test_bad.yaml || echo "Good! Invalid YAML detected"

# Clean up
rm -f test_whitespace.txt test_no_newline.txt test_bad.yaml
```

**Commit**:
```bash
git add .pre-commit-config.yaml
git commit -m "feat: add file quality pre-commit hooks

- trailing whitespace removal
- end of file fixer
- merge conflict detection
- yaml/toml/json validation
- large file prevention
- security checks"
```

### Task 5: Add Python-specific quality hooks

**Objective**: Add docstring and import order checks

**Files to modify**:
- `.pre-commit-config.yaml`

**Test First (TDD)**:
```bash
# Create file with docstring after code
cat > test_docstring.py << 'EOF'
def my_function():
    x = 1
    """This docstring is in the wrong place!"""
    return x
EOF

# Should fail
uv run pre-commit run check-docstring-first --files test_docstring.py || echo "Good! Bad docstring detected"

rm test_docstring.py
```

**Implementation**:
Add to the pre-commit-hooks section in `.pre-commit-config.yaml`:
```yaml
      # Ensure docstrings come first
      - id: check-docstring-first
```

**Verify**:
Run all hooks on the entire project:
```bash
uv run pre-commit run --all-files
```

**Commit**:
```bash
git add .pre-commit-config.yaml
git commit -m "feat: add Python docstring order check"
```

### Task 6: Create hook runner script

**Objective**: Create convenient script to run hooks manually

**Files to create**:
- `bin/run-hooks.sh`

**Implementation**:
```bash
cat > bin/run-hooks.sh << 'EOF'
#!/bin/bash
# Run pre-commit hooks manually
# Usage: ./bin/run-hooks.sh [files...]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Running pre-commit hooks..."

if [ $# -eq 0 ]; then
    # No arguments - run on all files
    echo "Checking all files..."
    uv run pre-commit run --all-files
else
    # Arguments provided - run on specific files
    echo "Checking files: $@"
    uv run pre-commit run --files "$@"
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All hooks passed!${NC}"
else
    echo -e "${RED}✗ Some hooks failed. Please fix the issues above.${NC}"
    exit 1
fi
EOF

chmod +x bin/run-hooks.sh
```

**Verify**:
```bash
# Test the script
./bin/run-hooks.sh

# Test with specific file
./bin/run-hooks.sh src/dqx/api.py
```

**Commit**:
```bash
git add bin/run-hooks.sh
git commit -m "feat: add convenient hook runner script"
```

### Task 7: Create setup script for new developers

**Objective**: Make it easy for new developers to set up pre-commit

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
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "Found Python $python_version"

# Check uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: 'uv' is not installed!"
    echo "Please install uv from: https://astral.sh/uv/install.sh"
    exit 1
fi

echo "✓ uv is installed: $(uv --version)"

# Sync dependencies
echo "Installing project dependencies..."
uv sync

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
uv run pre-commit install

# Run hooks on all files to verify setup
echo "Verifying setup by running hooks..."
uv run pre-commit run --all-files || true

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "Pre-commit hooks are now installed and will run automatically on git commit."
echo "To run hooks manually, use: ./bin/run-hooks.sh"
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
git commit -m "feat: add developer setup script"
```

### Task 8: Update documentation

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
- **File quality**: Trailing whitespace, file endings, large files
- **Security**: Detects accidentally committed private keys
- **Syntax validation**: Python, YAML, TOML, JSON files

#### Manual Usage

```bash
# Run on all files
./bin/run-hooks.sh

# Run on specific files
./bin/run-hooks.sh src/dqx/api.py tests/test_api.py

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
- hook id: ruff
- exit code: 1
  Fixed 1 error:
  - src/dqx/new_feature.py:
    1 × I001 (unsorted-imports)

# Ruff fixed the import order. Check and re-commit:
$ git add src/dqx/new_feature.py
$ git commit -m "feat: add new feature"
```
```

**Verify**:
Read through the documentation to ensure it's clear.

**Commit**:
```bash
git add README.md
git commit -m "docs: add pre-commit usage documentation"
```

### Task 9: Create troubleshooting guide

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
```

### 3. Hooks are slow

**Problem**: Hooks take too long to run.

**Solutions**:
- Run hooks only on changed files (default behavior)
- For faster feedback during development:
  ```bash
  # Run only ruff on specific file
  uv run ruff check src/dqx/api.py
  
  # Skip mypy temporarily while iterating
  SKIP=mypy git commit -m "wip: testing"
  ```

### 4. "Fixed X errors" but commit still fails

**Problem**: Ruff fixed issues but files need to be re-staged.

**Solution**:
```bash
# After hooks run and fix issues:
git add -u  # Stage all modified files
git commit  # Try commit again
```

### 5. Large file detected

**Problem**: Trying to commit a file larger than 500KB.

**Solutions**:
- Add file to `.gitignore` if it shouldn't be committed
- Use Git LFS for large files that must be versioned
- Compress the file if possible

### 6. Conflicts with IDE formatting

**Problem**: Your IDE reformats code differently than ruff.

**Solution**:
Configure your IDE to use ruff:
- VS Code: Install the Ruff extension
- PyCharm: Configure external tool to run ruff

### Disabling Hooks (Emergency Only!)

In emergencies, you can bypass hooks:
```bash
# Skip all hooks
git commit --no-verify -m "emergency: fix production issue"

# Skip specific hook
SKIP=mypy git commit -m "wip: debugging"
```

**Note**: Always run `./bin/run-hooks.sh` before pushing to ensure code quality.

## Getting Help

1. Check the error message - it usually tells you exactly what to fix
2. Run hooks on just the problem file: `uv run pre-commit run --files <file>`
3. Check hook configuration in `.pre-commit-config.yaml`
4. Ask team members or check pre-commit documentation
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
git commit -m "docs: add pre-commit troubleshooting guide"
```

### Task 10: Final verification

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
   ./bin/run-hooks.sh
   ```

## Summary

You've successfully implemented pre-commit hooks for DQX! 

### What was implemented:
1. ✅ Pre-commit package added to dependencies
2. ✅ Configuration with ruff and mypy hooks
3. ✅ General file quality hooks (whitespace, endings, etc.)
4. ✅ Python-specific checks
5. ✅ Convenient runner script
6. ✅ Developer setup script
7. ✅ Documentation in README
8. ✅ Troubleshooting guide

### Key files created/modified:
- `.pre-commit-config.yaml` - Hook configuration
- `pyproject.toml` - Added pre-commit dependency
- `bin/run-hooks.sh` - Manual hook runner
- `bin/setup-dev-env.sh` - Developer setup script
- `README.md` - Usage documentation
- `docs/pre_commit_troubleshooting.md` - Troubleshooting guide

### For the implementing developer:
Remember to run `./bin/setup-dev-env.sh` on your machine to activate the hooks!

## Testing Checklist

Before considering this complete, verify:
- [ ] `git commit` triggers pre-commit hooks
- [ ] Badly formatted Python code gets auto-fixed
- [ ] Type errors prevent commits
- [ ] Large files are rejected
- [ ] Documentation is clear and helpful
- [ ] All scripts are executable and work
