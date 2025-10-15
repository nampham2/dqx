# Move Rich to Main Dependencies - Implementation Plan v1

## Overview
The `rich` library is currently incorrectly placed in dev dependencies but is actively used in production code. This causes ImportError when the package is installed without dev dependencies. This plan moves `rich` to main dependencies where it belongs.

## Background for Engineers

### What is DQX?
DQX (Data Quality eXcellence) is a high-performance data quality framework for validating large datasets. It uses a graph-based architecture to efficiently compute metrics and run assertions on data.

### What is Rich?
Rich is a Python library for rich text and beautiful formatting in the terminal. DQX uses it to:
- Display graph structures as trees
- Show assertion results in formatted tables
- Present symbol values in tabular format

### Tooling Context
- **uv**: Modern Python package manager (replaces pip/poetry). Commands: `uv sync`, `uv run`, `uv add`
- **pyproject.toml**: Python project configuration (PEP 621 standard)
- **pytest**: Testing framework
- **mypy**: Static type checker
- **ruff**: Fast Python linter and formatter
- **pre-commit**: Git hooks for code quality

## Problem Statement
1. `rich>=14.1.0` is in `[dependency-groups]` dev section
2. Production code imports rich:
   - `src/dqx/display.py`: Uses Console, Tree, and Table
   - `src/dqx/analyzer.py`: Uses Console
3. Result: Production installations fail with ImportError

## Implementation Tasks

### Task 1: Setup and Branch Creation
```bash
# Ensure you're in the project root
cd /Users/npham/git-tree/dqx

# Create feature branch
git checkout -b fix/move-rich-to-main-deps

# Verify branch
git branch --show-current
```
**Commit**: `git commit --allow-empty -m "chore: create branch for moving rich to main deps"`

### Task 2: Write Failing Test (TDD)
Create `tests/test_rich_dependency.py`:

```python
"""Test that rich is available as a main dependency."""

import pytest
from pathlib import Path


def test_rich_is_main_dependency():
    """Verify rich is in main dependencies, not dev dependencies."""
    # Read pyproject.toml
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()

    # Parse dependencies section
    lines = content.split('\n')
    in_main_deps = False
    in_dev_deps = False
    found_in_main = False
    found_in_dev = False

    for i, line in enumerate(lines):
        # Check if we're in main dependencies
        if line.strip() == 'dependencies = [':
            in_main_deps = True
            in_dev_deps = False
        elif line.strip() == 'dev = [':
            in_dev_deps = True
            in_main_deps = False
        elif line.strip() == ']':
            in_main_deps = False
            in_dev_deps = False

        # Check for rich
        if in_main_deps and 'rich>=' in line:
            found_in_main = True
        if in_dev_deps and 'rich>=' in line:
            found_in_dev = True

    assert found_in_main, "rich should be in main dependencies"
    assert not found_in_dev, "rich should NOT be in dev dependencies"


def test_production_imports_work():
    """Test that production code can import rich."""
    # These imports should work without dev dependencies
    from dqx.display import print_graph, print_assertion_results, print_symbols
    from dqx.analyzer import Analyzer

    # Verify the imports have rich components
    assert hasattr(print_graph, '__code__')
    assert hasattr(print_assertion_results, '__code__')
    assert hasattr(print_symbols, '__code__')
    assert hasattr(Analyzer, '__init__')
```

Run test to verify it fails:
```bash
uv run pytest tests/test_rich_dependency.py -v
```

**Commit**: `git add tests/test_rich_dependency.py && git commit -m "test: add failing test for rich dependency location"`

### Task 3: Update pyproject.toml

Edit `pyproject.toml`:

1. Locate the main dependencies section (around line 8-17)
2. Add `"rich>=14.1.0",` after `"returns>=0.26.0",` (maintaining alphabetical order)
3. Remove `"rich>=14.1.0",` from the dev dependencies (around line 26-35)

The dependencies section should look like:
```toml
dependencies = [
    "datasketches>=5.2.0",
    "duckdb>=1.3.2",
    "msgpack>=1.1.1",
    "numpy>=2.3.2",
    "pyarrow>=21.0.0",
    "returns>=0.26.0",
    "rich>=14.1.0",
    "sqlalchemy>=2.0.43",
    "sympy>=1.14.0",
]
```

**Commit**: `git add pyproject.toml && git commit -m "fix: move rich from dev to main dependencies"`

### Task 4: Update Lock File
```bash
# Sync dependencies
uv sync

# Verify rich is installed
uv run python -c "import rich; print(f'Rich version: {rich.__version__}')"
```

**Commit**: `git add uv.lock && git commit -m "chore: update lock file for rich dependency move"`

### Task 5: Verify Test Now Passes
```bash
# Run our new test
uv run pytest tests/test_rich_dependency.py -v

# Should see green/passing tests
```

### Task 6: Run Full Test Suite
```bash
# Run all tests
uv run pytest tests/ -v

# Focus on display-related tests
uv run pytest tests/test_display.py tests/test_analyzer.py -v

# Check test coverage
uv run pytest --cov=dqx --cov-report=term-missing
```

### Task 7: Run Code Quality Checks
```bash
# Run mypy
uv run mypy src/

# Run ruff
uv run ruff check src/ tests/

# Run all pre-commit hooks
uv run pre-commit run --all-files

# Alternative: use project script
./bin/run-hooks.sh
```

### Task 8: Test Production Installation
```bash
# Create temporary test directory
mkdir /tmp/test-dqx-prod && cd /tmp/test-dqx-prod

# Create fresh virtual environment
python3 -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Build the package
cd /Users/npham/git-tree/dqx
pip install build
python -m build

# Install without dev dependencies
cd /tmp/test-dqx-prod
pip install /Users/npham/git-tree/dqx/dist/dqx-*.whl

# Test imports
python -c "
from dqx.display import print_graph
from dqx.analyzer import Analyzer
print('✓ Production imports successful!')
"

# Cleanup
deactivate
cd /Users/npham/git-tree/dqx
rm -rf /tmp/test-dqx-prod
```

### Task 9: Documentation Check
Search for any installation docs that might need updating:
```bash
# Search for installation instructions
grep -r "pip install" docs/ README.md
grep -r "dev.*dep" docs/ README.md
grep -r "rich" docs/ README.md
```

If any docs mention dev dependencies or installation, update them.

### Task 10: Final Verification Checklist
- [ ] `tests/test_rich_dependency.py` passes
- [ ] All existing tests pass
- [ ] mypy passes
- [ ] ruff passes
- [ ] pre-commit hooks pass
- [ ] Production installation includes rich
- [ ] No ImportError in production

**Final Commit**: `git add -A && git commit -m "docs: update any documentation for rich dependency"`

## Testing Guide

### Unit Test
```bash
uv run pytest tests/test_rich_dependency.py -v
```

### Integration Tests
```bash
# Display functionality
uv run pytest tests/test_display.py -v

# Analyzer functionality
uv run pytest tests/test_analyzer.py -v
```

### Manual Testing
```python
# In Python REPL
from dqx.display import SimpleNodeFormatter, print_assertion_results
from dqx.analyzer import Analyzer

# Should work without errors
print("✓ Manual import test passed")
```

## Troubleshooting

### Problem: "Module 'rich' not found"
**Solution**: Run `uv sync` to update your environment

### Problem: Lock file conflicts
**Solution**:
```bash
rm uv.lock
uv sync
```

### Problem: Tests fail after changes
**Solution**: Ensure you're in the virtual environment: `uv run pytest ...`

### Problem: Pre-commit hooks fail
**Solution**:
```bash
uv run pre-commit clean
uv run pre-commit install
uv run pre-commit run --all-files
```

## Success Criteria
✓ rich is in main dependencies in pyproject.toml
✓ rich is NOT in dev dependencies
✓ New test `test_rich_dependency.py` passes
✓ All existing tests pass
✓ Code quality checks pass
✓ Production installation works without dev dependencies

## Notes
- This is a bug fix, not a feature
- No version bump needed (internal dependency management)
- No breaking changes for users
- Estimated time: 30-45 minutes
