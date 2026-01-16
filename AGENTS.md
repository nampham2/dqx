# DQX Agent Guidelines

This document provides coding guidelines and commands for AI agents working on the DQX codebase.

## Project Overview

DQX (Data Quality Excellence) is a Python 3.11+ library for data quality validation. The package name is `dqlib` but the module name is `dqx`. It allows writing validation logic as testable Python functions that execute efficiently on SQL backends (DuckDB, BigQuery, Snowflake).

**Key Technologies:** Python 3.11+, DuckDB, PyArrow, SymPy, SQLAlchemy, Lark parser, Rich (terminal UI)

## Build, Test, and Lint Commands

### Package Management
All commands use `uv` (fast Python package manager):

```bash
# Install development environment
uv sync

# Install dependencies only
uv pip install -e .

# Install with dev dependencies
uv sync --all-extras
```

### Testing

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_graph_display.py

# Run a single test function/class
uv run pytest tests/test_graph_display.py::TestGraphPrintTree::test_print_tree_with_default_formatter

# Run tests with coverage
uv run pytest --cov=src/dqx --cov-report=html --cov-report=xml

# Run tests with coverage report in terminal
uv run pytest --cov=src/dqx --cov-report=term-missing

# Run tests matching a pattern
uv run pytest -k "test_parser"

# Run with verbose output
uv run pytest -v

# Run with print statements visible
uv run pytest -s

# Run demo tests (marked with @pytest.mark.demo)
uv run pytest -m demo -s
```

### IMPORTANT: 100% Test Coverage Required

- All code changes MUST maintain 100% test coverage
- After implementation, always verify coverage with: `uv run pytest --cov=src/dqx --cov-report=term-missing`
- If any lines are uncovered, either:
  1. Add tests to cover them, OR
  2. Add `# pragma: no cover` comment for unreachable/defensive code
- Coverage target: 100% (no exceptions)

### Code Quality

```bash
# Format code (run this first)
uv run ruff format

# Lint and auto-fix issues
uv run ruff check --fix

# Type check
uv run mypy src tests

# Run all pre-commit hooks
uv run pre-commit run --all-files

# Install pre-commit hooks
uv run pre-commit install
```

### Custom Commands

```bash
# Run coverage with HTML report
uv run coverage

# Set up pre-commit hooks
uv run hooks

# Clean up cache and build artifacts
uv run cleanup
```

### Documentation

```bash
# Serve docs locally
uv run mkdocs serve

# Build documentation
uv run mkdocs build
```

## Code Style Guidelines

### Imports

**Order:** Standard library, third-party, local imports (separated by blank lines)

```python
from __future__ import annotations  # Always first for forward references

import logging
import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pyarrow as pa
import sympy as sp

from dqx import functions, setup_logger
from dqx.analyzer import AnalysisReport, Analyzer
from dqx.common import DQXError, ResultKey

if TYPE_CHECKING:
    from dqx.orm.repositories import MetricDB
```

**Key patterns:**
- Use `from __future__ import annotations` at the top of every module
- Import `TYPE_CHECKING` and use it for circular dependency resolution
- Use `collections.abc` for abstract types (Callable, Sequence, etc.)
- Prefer specific imports over wildcard imports

### Type Hints

**Strict typing is enforced:** All functions must have complete type annotations.

```python
def validate_tags(tags: set[str] | frozenset[str] | None) -> frozenset[str]:
    """Validate and convert tags to frozenset."""
    if tags is None:
        return frozenset()
    # ...


# Use Protocol for structural typing
@runtime_checkable
class SqlDataSource(Protocol):
    def execute(self, query: str) -> pa.Table: ...
```

**Type configuration:**
- MyPy strict mode: `disallow_untyped_defs = true`
- Target Python 3.11
- Use `|` for union types (not `Union`)
- Use `Protocol` for structural subtyping

### Formatting

**Ruff configuration:**
- Line length: 120 characters
- Indentation: 4 spaces
- Target: Python 3.11
- Docstring convention: Google style

```python
def where(
    self,
    *,
    name: str,
    severity: SeverityLevel = "P1",
    tags: frozenset[str] | set[str] | None = None,
    experimental: bool = False,
) -> AssertionReady:
    """Create an AssertionReady bound to this expression.

    Args:
        name: Descriptive name for the assertion (1–255 characters).
        severity: Severity level (e.g., "P0", "P1", "P2", "P3").
        tags: Optional set of tags for categorization.
        experimental: If True, marks assertion as removable by algorithms.

    Returns:
        AssertionReady: A ready-to-use assertion object.

    Raises:
        ValueError: If name is empty or longer than 255 characters.
    """
```

### Naming Conventions

- **Classes:** PascalCase (`MetricProvider`, `AnalysisReport`)
- **Functions/methods:** snake_case (`validate_tags`, `add_assertion`)
- **Constants:** UPPER_SNAKE_CASE (`TAG_PATTERN`)
- **Private:** Leading underscore (`_actual`, `_context`)
- **Type aliases:** PascalCase (`DatasetName`, `ExecutionId`)

### Error Handling

Use `returns` library for functional error handling:

```python
from returns.result import Result


def compute_metric() -> Result[float, str]:
    """Return Success(value) or Failure(error_msg)."""
    try:
        value = perform_calculation()
        return Success(value)
    except Exception as e:
        return Failure(f"Calculation failed: {e}")
```

**Custom exceptions:**
```python
class DQXError(Exception): ...


# Raise with descriptive messages
raise DQXError(f"Invalid tag '{tag}': must contain only alphanumerics")
```

### Dataclasses

Use frozen dataclasses for immutable data:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ResultKey:
    yyyy_mm_dd: datetime.date
    tags: Tags

    def __hash__(self) -> int:
        return hash((self.yyyy_mm_dd, tuple(self.tags)))
```

## Testing Guidelines

- Test files mirror source structure: `src/dqx/foo.py` → `tests/test_foo.py`
- Use pytest fixtures in `tests/fixtures/`
- Organize tests in classes: `class TestGraphPrintTree:`
- Use descriptive test names: `test_print_tree_with_default_formatter`
- Mock external dependencies with `unittest.mock`
- Use `pytest.mark.demo` for demonstration tests requiring visual output

## Implementation Workflow

After implementing code changes, **always** follow this checklist before reporting completion:

1. **Run all tests**: `uv run pytest`
   - All tests must pass

2. **Check coverage**: `uv run pytest --cov=src/dqx --cov-report=term-missing`
   - Coverage must be 100%
   - Add tests for uncovered lines or use `# pragma: no cover` for unreachable code

3. **Run pre-commit checks**: `uv run pre-commit run --all-files`
   - All hooks must pass (formatting, linting, type checking, commit message validation)
   - Pre-commit runs: ruff format, ruff check, mypy, commitizen check

4. **Verify no issues**:
   - No linting errors
   - No type errors
   - All formatting applied

**Summary**: Implementation is only complete when tests pass, coverage is 100%, and pre-commit checks pass.

## Project Structure

```
src/dqx/           # Main source code
├── api.py         # User-facing API (VerificationSuite, MetricProvider, Context)
├── validator.py   # Validation logic and suite execution
├── analyzer.py    # SQL analysis engine
├── dialect.py     # SQL dialect implementations (DuckDB, BigQuery, Snowflake)
├── provider.py    # Metric provider implementation
├── common.py      # Shared types and protocols
├── display.py     # Result visualization with Rich
├── graph/         # Graph processing for dependency analysis
├── dql/           # Data Quality Language parser
└── orm/           # Object-relational mapping for metrics

tests/             # Test suite (110+ files)
├── dql/           # DQL tests
├── e2e/           # End-to-end tests
├── graph/         # Graph processing tests
└── fixtures/      # Shared fixtures

docs/              # MkDocs documentation
bin/               # Development scripts
```

## Commit Message Convention

Use Conventional Commits format (enforced by pre-commit hook):

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Examples
```
feat(analyzer): add query optimization for large datasets
fix(evaluator): resolve type inference issue
docs(api): update API reference for MetricProvider
test(parser): add coverage for edge cases
refactor(graph): simplify traversal algorithm
perf(analyzer): optimize SQL generation
```

### Commit Types
- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Code formatting changes (no logic changes)
- **refactor**: Code changes that neither fix bugs nor add features
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **build**: Changes to build system or dependencies
- **ci**: Changes to CI configuration
- **chore**: Other changes (maintenance, etc.)
- **revert**: Reverts a previous commit

### Scopes (optional)
- analyzer, api, graph, evaluator, provider, specs, validator
- display, orm, extensions, common, dialect, functions
- models, ops, states, utils, dql

### Rules
- Subject line: max 72 characters, imperative mood ("add" not "added")
- Body: optional, provide context and motivation
- Footer: optional, reference issues (e.g., "Closes #123")
- Breaking changes: use "BREAKING CHANGE:" in footer or commit body

## Version Management

- Tool: Commitizen with conventional commits
- Versioning: PEP 440 with major_version_zero
- Current version: 0.5.11
- Automated via GitHub Actions release workflow

## Additional Notes

- Coverage target: 100% (green), 95% minimum (orange)
- Pre-commit hooks run format → lint → type check
- Use `SKIP=mypy` to skip type checking in pre-commit for faster iteration
- Documentation uses Google-style docstrings
- SQL backends supported: DuckDB (primary), BigQuery, Snowflake
