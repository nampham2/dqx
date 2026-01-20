# DQX Agent Guidelines

This document provides coding guidelines and commands for AI agents working on the DQX codebase.

## Table of Contents

- [Project Overview](#project-overview)
- [Commands Reference](#commands-reference)
  - [Package Management](#package-management)
  - [Testing Commands](#testing-commands)
  - [Code Quality Commands](#code-quality-commands)
  - [Documentation Commands](#documentation-commands)
  - [Custom Commands](#custom-commands)
- [Code Standards](#code-standards) (Single Source of Truth)
  - [Import Order](#import-order)
  - [Type Hints](#type-hints)
  - [Docstrings](#docstrings)
  - [Formatting](#formatting)
  - [Naming Conventions](#naming-conventions)
  - [Error Handling](#error-handling)
  - [Dataclasses](#dataclasses)
- [Testing Standards](#testing-standards)
  - [Test Structure](#test-structure)
  - [Coverage Requirements](#coverage-requirements)
  - [Testing Patterns](#testing-patterns)
  - [Fixtures and Mocking](#fixtures-and-mocking)
- [Quality Gates](#quality-gates)
  - [Implementation Checklist](#implementation-checklist)
  - [Pre-commit Requirements](#pre-commit-requirements)
- [Commit Conventions](#commit-conventions)
- [Feature Development Workflow](#feature-development-workflow)
- [Project Structure](#project-structure)
- [Version Management](#version-management)

---

## Project Overview

DQX (Data Quality Excellence) is a Python 3.11+ library for data quality validation. The package name is `dqlib` but the module name is `dqx`. It allows writing validation logic as testable Python functions that execute efficiently on SQL backends (DuckDB, BigQuery, Snowflake).

**Key Technologies:** Python 3.11+, DuckDB, PyArrow, SymPy, SQLAlchemy, Lark parser, Rich (terminal UI)

---

## Commands Reference

All commands use `uv` (fast Python package manager).

**Command Quick Reference Card:**
```bash
# Most common (memorize these)
uv run pytest                                    # All tests
uv run pytest --cov=src/dqx --cov-report=term  # Coverage check
uv run ruff format                               # Format code
uv run mypy src tests                            # Type check
uv run pre-commit run --all-files                # All hooks
```

### Package Management

```bash
# Install development environment
uv sync

# Install dependencies only
uv pip install -e .

# Install with dev dependencies
uv sync --all-extras
```

### Testing Commands

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

### Code Quality Commands

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

### Documentation Commands

```bash
# Serve docs locally
uv run mkdocs serve

# Build documentation
uv run mkdocs build
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

---

## Code Standards

> **Single Source of Truth**: This section defines ALL code standards for DQX. All agent files reference these standards.

> **For Agents**: Use the `dqx-code-standards` skill for quick reference without loading this full document.
> ```javascript
> skill({ name: "dqx-code-standards" })
> ```
> The skill provides concise, actionable guidance (170 lines) instead of loading the full AGENTS.md (1000+ lines).
> This section remains the authoritative source for humans and comprehensive documentation.

### Import Order

**Order:** Standard library → third-party → local imports (separated by blank lines)

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

### Docstrings

**Convention:** Google style (required for all public APIs)

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

### Formatting

**Ruff configuration:**
- Line length: 120 characters
- Indentation: 4 spaces
- Target: Python 3.11
- Docstring convention: Google style

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

---

## Testing Standards

### Test Structure

- Test files mirror source structure: `src/dqx/foo.py` → `tests/test_foo.py`
- Organize tests in classes: `class TestGraphPrintTree:`
- Use descriptive test names: `test_print_tree_with_default_formatter`

### Coverage Requirements

#### Target: 100% coverage (no exceptions)

- All code changes MUST maintain 100% test coverage
- After implementation, always verify coverage with: `uv run pytest --cov=src/dqx --cov-report=term-missing`
- If any lines are uncovered, either:
  1. Add tests to cover them, OR
  2. Add `# pragma: no cover` comment for unreachable/defensive code

**When to use `# pragma: no cover`:**
- Defensive code that should never execute
- Abstract methods that must be overridden
- Trivial `__repr__` or `__str__` methods

### Testing Patterns

```python
# Test class organization
class TestFeatureName:
    """Tests for feature functionality."""

    def test_creation_with_valid_input(self) -> None:
        """Test creating instance with valid input."""
        feature = MyFeature(valid_arg="test")
        assert feature.property == "test"

    def test_creation_with_invalid_input(self) -> None:
        """Test error handling for invalid input."""
        with pytest.raises(ValueError, match="Invalid arg"):
            MyFeature(valid_arg=None)

    def test_edge_case_empty(self) -> None:
        """Test behavior with empty input."""
        feature = MyFeature(valid_arg="")
        assert feature.is_empty()
```

### Fixtures and Mocking

- Use pytest fixtures in `tests/fixtures/`
- Mock external dependencies with `unittest.mock`
- Use `pytest.mark.demo` for demonstration tests requiring visual output

```python
from unittest.mock import Mock, patch


def test_with_mock() -> None:
    """Test with mocked database."""
    mock_db = Mock(spec=MetricDB)
    mock_db.fetch.return_value = expected_data

    feature = MyFeature(db=mock_db)
    result = feature.fetch_data()

    mock_db.fetch.assert_called_once()
    assert result == expected_data
```

---

## Quality Gates

### Implementation Checklist

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

### Pre-commit Requirements

Pre-commit hooks run in this order:
1. **ruff format** - Auto-fixes formatting
2. **ruff check** - Auto-fixes linting issues
3. **mypy** - Type checking (must fix manually)
4. **commitizen** - Validates commit messages

All hooks must pass before committing.

---

## Commit Conventions

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

---

## Feature Development Workflow

DQX uses specialized subagents for different phases of feature development to optimize context usage and efficiency.

### Workflow Overview

```text
User Request → Plan Agent → Implementation Agent → PR Agent → Feedback Agent
                  ↓               ↓                    ↓            ↓
            Design Docs      Code + Tests          PR Created   Fixes Applied
```

### Phase 1: Planning (Automatic)

**When**: User requests a new feature

**Agent**: `dqx-plan`

**Process**:
```text
User: "I want to add {feature}"

Core Agent:
  1. Launch dqx-plan agent
  2. dqx-plan creates 3 modular design docs:
     - Technical Specification (architecture, APIs)
     - Implementation Guide (TDD phases)
     - Context Document (DQX patterns, background)
  3. Present summary to user
  4. Wait for approval
```

**Output**: 3 design documents in `docs/plans/` (~600-1100 lines total)

**Key Benefits**:
- Modular docs avoid context overload
- Clear separation: what (spec) vs how (guide) vs why (context)
- Documents are reusable across sessions
- Specialized agents handle exploration efficiently

### Phase 2: Implementation (Automatic after approval)

**When**: User approves design

**Agent**: `dqx-implement`

**Process**:
```text
User: "Proceed with implementation" or "Implement it"

Core Agent:
  1. Launch dqx-implement agent
  2. dqx-implement executes each phase automatically:
     For each phase:
       a. Write tests FIRST (TDD)
       b. Implement code to pass tests
       c. Verify 100% coverage
       d. Pass pre-commit hooks
       e. Commit with conventional message
       f. Brief progress report
  3. Present completion summary
  4. Wait for PR approval
```

**Output**:
- Working code with 100% test coverage
- Multiple atomic commits (one per phase)
- All quality gates passed

**Key Benefits**:
- Fully automated TDD cycle
- Quality enforcement (coverage, pre-commit) built-in
- Loads only relevant docs per phase (reduces context)
- Independent, committable phases

### Phase 3: PR Creation (Automatic after approval)

**When**: User approves PR creation

**Agent**: `dqx-pr`

**Process**:
```text
User: "Create PR"

Core Agent:
  1. Launch dqx-pr agent
  2. dqx-pr:
     a. Analyzes all commits in branch
     b. Verifies quality gates (tests, coverage, pre-commit)
     c. Generates comprehensive PR description
     d. Creates PR with gh CLI
     e. Returns PR URL
  3. Present PR details to user
```

**Output**: GitHub PR with comprehensive description

**Key Benefits**:
- Automatic quality gate verification
- PR includes design doc links (not full content)
- Phase breakdown for reviewers
- Conventional commit format

### Phase 4: Feedback Iteration (On-demand)

**When**: User needs to address review feedback

**Agent**: `dqx-feedback`

**Process**:
```text
User: "Address CodeRabbit feedback" or "Fix review comments"

Core Agent:
  1. Launch dqx-feedback agent
  2. dqx-feedback:
     a. Fetches all review comments from GitHub
     b. Groups and prioritizes (P0 → P1 → P2)
     c. For each comment:
        - Load only relevant file + context
        - Make targeted fix
        - Run affected tests
        - Commit with descriptive message
     d. Final verification (full test suite)
     e. Push all fixes
  3. Present fix summary to user
```

**Output**:
- Targeted fixes for each comment
- Clear commit history
- All quality gates still passing

**Key Benefits**:
- Minimal context reload per fix
- Fast iteration cycle
- Clear commit history for reviewers
- Prioritized fixes (blockers first)

---

### Agent Context Management

To avoid context window overload, each agent loads only necessary documents:

#### Planning Phase (dqx-plan)
**Load**:
- AGENTS.md (coding guidelines)
- Exploration results (from dqx-explore, dqx-api agents)

**Avoid**:
- Implementation details
- Full code examples
- Test implementations

#### Implementation Phase (dqx-implement)
**Load per phase**:
- Technical spec (relevant sections only)
- Implementation guide (current phase only)
- Context doc (relevant patterns only)

**Avoid**:
- Other phases from implementation guide
- Full technical spec
- Full context doc
- Unrelated files

#### PR Phase (dqx-pr)
**Load**:
- Design docs (summary extraction only)
- Git commit history
- Quality gate results

**Avoid**:
- Full implementation code
- Test files
- Full design doc content

#### Feedback Phase (dqx-feedback)
**Load per comment**:
- Specific file with comment
- Related tests for that file
- Relevant technical spec section (if needed)

**Avoid**:
- Design docs (already implemented)
- Unrelated files
- Other comments' context

**Context Reduction**: ~60% less context per phase compared to monolithic approach

---

### Subagent Specialization

#### Core Workflow Agents (New)

**dqx-plan** (Proactive: Use for all feature requests)
- **Role**: Create modular design documents
- **Input**: Feature request, exploration results
- **Output**: 3 design documents (spec, guide, context)
- **When**: User requests new feature
- **Context**: Minimal - AGENTS.md + exploration results

**dqx-implement** (Proactive: Use after plan approval)
- **Role**: Execute TDD implementation phases
- **Input**: Design documents, current phase
- **Output**: Code, tests, commits per phase
- **When**: After design approval
- **Context**: Minimal - current phase + relevant patterns

**dqx-pr** (Proactive: Use after implementation)
- **Role**: Create comprehensive pull requests
- **Input**: Design docs, git history
- **Output**: GitHub PR with description
- **When**: After implementation complete
- **Context**: Minimal - doc summaries + commits

**dqx-feedback** (On-demand: Use when requested)
- **Role**: Address review feedback efficiently
- **Input**: PR number, review comments
- **Output**: Targeted fixes, commits
- **When**: User requests feedback iteration
- **Context**: Minimal - specific files per comment

#### Supporting Agents (Use as needed)

##### dqx-explore
- **Role**: Fast codebase exploration
- **When**: During planning to find similar patterns
- **Usage**: Always use in planning, avoid in implementation

##### dqx-sql
- **Role**: SQL dialect and parser work
- **When**: Feature touches SQL generation
- **Usage**: Delegate SQL-specific implementation

##### dqx-docs
- **Role**: API documentation and examples
- **When**: After implementation complete
- **Usage**: Update docs after code is done

##### dqx-test
- **Role**: Test generation and coverage analysis
- **When**: Coverage gaps need detailed analysis
- **Usage**: During implementation for missing tests

##### dqx-api
- **Role**: API design patterns and consistency
- **When**: During planning for API features
- **Usage**: Ensure consistency with existing APIs

##### dqx-quality
- **Role**: Pre-commit hooks, linting, type checking
- **When**: Pre-commit failures need fixing
- **Usage**: During implementation for quality issues

##### dqx-graph
- **Role**: Graph processing and dependency analysis
- **When**: Features involving graphs or traversal
- **Usage**: Delegate graph-specific code

---

### Example: Complete Feature Workflow

#### User Request
```text
User: "I want to add metric caching to avoid recomputing expensive metrics"
```

#### Step 1: Planning

```text
Core Agent: "I'll help you plan metric caching."

[Launches dqx-plan agent]

dqx-plan:
  - Explores codebase (delegates to dqx-explore)
  - Reviews API patterns (delegates to dqx-api)
  - Creates 3 design docs:
    ✓ metric_caching_technical_spec.md (312 lines)
    ✓ metric_caching_implementation_guide.md (478 lines)
    ✓ metric_caching_context.md (245 lines)

Core Agent: "Design complete. Key decisions:
- LRU cache with TTL support
- Protocol-based cache backend
- 4 implementation phases
- Estimated: ~5 hours

Ready to implement?"
```

#### Step 2: Implementation

```text
User: "Yes, implement it"

Core Agent: "Starting implementation..."

[Launches dqx-implement agent]

dqx-implement:
  Phase 1/4: Core cache data structures
    ✓ Tests written (8 tests)
    ✓ Implementation complete
    ✓ Coverage: 100%
    ✓ Pre-commit: passing
    ✓ Committed: feat(cache): add core cache data structures

  Phase 2/4: Cache backend protocol
    ✓ Tests written (12 tests)
    ✓ Implementation complete
    ✓ Coverage: 100%
    ✓ Pre-commit: passing
    ✓ Committed: feat(cache): implement cache backend protocol

  Phase 3/4: MetricProvider integration
    ✓ Tests written (15 tests)
    ✓ Implementation complete
    ✓ Coverage: 100%
    ✓ Pre-commit: passing
    ✓ Committed: feat(cache): integrate caching with MetricProvider

  Phase 4/4: E2E tests + docs
    ✓ Tests written (6 tests)
    ✓ Implementation complete
    ✓ Coverage: 100%
    ✓ Pre-commit: passing
    ✓ Committed: docs(cache): add caching documentation

Core Agent: "Implementation complete!
- 4 phases, 41 tests, 100% coverage
- 4 commits created

Ready to create PR?"
```

#### Step 3: PR Creation

```text
User: "Create PR"

Core Agent: "Creating pull request..."

[Launches dqx-pr agent]

dqx-pr:
  ✓ Quality gates verified
  ✓ PR description generated
  ✓ Created PR #123

Core Agent: "PR created: https://github.com/repo/pull/123

Summary:
- 4 commits
- 41 tests added
- Coverage: 100%
- Design docs: 3 files linked

CodeRabbit will review automatically."
```

#### Step 4: Feedback Iteration

```text
User: "Address CodeRabbit feedback"

Core Agent: "Analyzing review comments..."

[Launches dqx-feedback agent]

dqx-feedback:
  ✓ Fetched 8 review comments
  ✓ Prioritized: 2 P0, 4 P1, 2 P2

  Fixing P0 issues:
    ✓ src/dqx/cache.py:78 - Added type hint
    ✓ tests/test_cache.py:45 - Added edge case test

  Fixing P1 issues:
    ✓ src/dqx/cache.py:102 - Improved naming
    ✓ src/dqx/cache.py:156 - Added docstring
    ✓ tests/test_cache.py:89 - Added integration test
    ✓ docs/plans/cache_technical_spec.md - Clarified decision

  Fixing P2 issues:
    ✓ src/dqx/cache.py:134 - Refactored helper method
    ✓ tests/test_cache.py:112 - Added performance test

  ✓ Full test suite: passing
  ✓ Coverage: 100%
  ✓ Pushed 8 fix commits

Core Agent: "All 8 review comments addressed!
- P0 (blockers): 2 fixed
- P1 (important): 4 fixed
- P2 (nice-to-have): 2 fixed

Ready for next review cycle."
```

---

### Best Practices for Using Workflow

#### DO:
- ✓ Use dqx-plan for all new features (automatic)
- ✓ Review design docs before implementation
- ✓ Let dqx-implement run phases automatically
- ✓ Trust quality gates (100% coverage, pre-commit)
- ✓ Use dqx-feedback for iterative improvements
- ✓ Keep commits atomic and focused

#### DON'T:
- ✗ Skip planning phase for features
- ✗ Modify design docs during implementation (finish first, then update)
- ✗ Manually implement without using agents (less efficient)
- ✗ Skip quality gates (coverage, pre-commit)
- ✗ Combine multiple phases in one commit
- ✗ Address feedback without using dqx-feedback agent

#### When to Deviate:
- **Small bug fixes**: Skip planning, implement directly
- **Documentation updates**: Skip planning, edit docs directly
- **Simple refactors**: Skip planning if scope is clear
- **Urgent hotfixes**: Streamline process, but maintain quality gates

---

## Project Structure

```text
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

---

## Version Management

- Tool: Commitizen with conventional commits
- Versioning: PEP 440 with major_version_zero
- Current version: 0.5.11
- Automated via GitHub Actions release workflow

---

## Additional Notes

- Coverage target: 100% (no exceptions)
- Pre-commit hooks run format → lint → type check
- Use `SKIP=mypy` only for local exploration; do not skip mypy for commits or PRs
- Documentation uses Google-style docstrings
- SQL backends supported: DuckDB (primary), BigQuery, Snowflake

---

## Quick Reference

For a quick lookup of common commands and standards, see [docs/quick_reference.md](docs/quick_reference.md).

For a complete end-to-end workflow example, see [docs/workflow_example.md](docs/workflow_example.md).
