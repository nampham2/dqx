# Verification Suite Validator Implementation Plan (Version 2)

## Overview

You'll be implementing a validation system that checks DQX verification suites for common configuration errors BEFORE they run. Think of it as a "linter" for data quality checks that catches issues like duplicate names, empty checks, and other problems early.

This version incorporates architectural feedback to:
- Simplify the design by removing severity from ValidationIssue
- Use visitor pattern (consistent with DQX patterns)
- Remove unnecessary rules (UnnamedAssertionRule)
- Improve error handling

## Background Context

### What is DQX?
DQX is a data quality framework. Users define "checks" (functions that validate data) and group them into "suites". A suite runs these checks against datasets to ensure data quality.

### Current Problem
Right now, users can create invalid suites with issues like:
- Two checks with the same name
- Assertions with duplicate names within a check
- Empty checks (no assertions)

These issues only surface during execution, making debugging harder.

### Solution
Build a validator that runs AFTER suite creation but BEFORE execution to catch these issues early.

## Architecture Overview

```
VerificationSuite (existing)
    |
    ├── Has checks (functions)
    ├── Has a graph structure (nodes representing checks/assertions)
    └── NEW: Has a validator that inspects the graph using visitor pattern
```

## Development Principles

1. **TDD (Test-Driven Development)**: Write tests FIRST, then code
2. **DRY (Don't Repeat Yourself)**: Reuse code, avoid duplication
3. **YAGNI (You Aren't Gonna Need It)**: Only implement what's specified
4. **Frequent Commits**: Commit after each passing test

## Implementation Tasks

### Task 1: Create the Validation Module Structure

**Goal**: Set up the basic module structure

**Files to create**:
```
src/dqx/validators/
├── __init__.py
├── base.py
├── report.py
└── visitors.py  # Changed from rules.py to reflect visitor pattern
```

**Steps**:
1. Create the directory: `mkdir -p src/dqx/validators`
2. Create empty `__init__.py`
3. Commit: `git commit -m "feat: create validators module structure"`

**How to verify**:
```bash
ls -la src/dqx/validators/
# Should show all 4 files
```

---

### Task 2: Define Core Data Structures (TDD)

**Goal**: Create ValidationIssue (simplified without severity)

**Test file**: `tests/test_validators_base.py`

**Write this test FIRST**:
```python
import pytest
from dqx.validators.base import ValidationIssue


def test_validation_issue_creation():
    """Test creating a validation issue."""
    issue = ValidationIssue(
        rule="test_rule", message="Something went wrong", node_path=["root", "check1"]
    )

    assert issue.rule == "test_rule"
    assert issue.message == "Something went wrong"
    assert issue.node_path == ["root", "check1"]
```

**Now implement in `src/dqx/validators/base.py`**:
```python
"""Base classes for validation system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValidationIssue:
    """Represents a single validation issue found in the graph."""

    rule: str  # Name of the rule that found this issue
    message: str  # Human-readable description
    node_path: list[str]  # Path to the problematic node
```

**Update `src/dqx/validators/__init__.py`**:
```python
"""Validation system for DQX verification suites."""

from dqx.validators.base import ValidationIssue

__all__ = ["ValidationIssue"]
```

**Run test again (should pass)**:
```bash
uv run pytest tests/test_validators_base.py -v
```

**Commit**:
```bash
git add -A
git commit -m "feat: add ValidationIssue dataclass"
```

---

### Task 3: Create ValidationReport (TDD)

**Goal**: Build a class to collect and format validation issues

**Test file** `tests/test_validators_report.py`:

```python
import pytest
from dqx.validators.report import ValidationReport
from dqx.validators.base import ValidationIssue


def test_validation_report_empty():
    """Test empty validation report."""
    report = ValidationReport()

    assert not report.has_errors()
    assert not report.has_warnings()
    assert len(report.errors) == 0
    assert len(report.warnings) == 0


def test_validation_report_add_issues():
    """Test adding issues to report."""
    report = ValidationReport()

    # Add an error from a rule that produces errors
    report.add_error(
        ValidationIssue(
            rule="duplicate_check", message="Duplicate found", node_path=["root"]
        )
    )

    # Add a warning from a rule that produces warnings
    report.add_warning(
        ValidationIssue(
            rule="empty_check",
            message="Empty check found",
            node_path=["root", "check1"],
        )
    )

    assert report.has_errors()
    assert report.has_warnings()
    assert len(report.errors) == 1
    assert len(report.warnings) == 1


def test_validation_report_string_format():
    """Test report string formatting."""
    report = ValidationReport()

    report.add_error(
        ValidationIssue(
            rule="duplicate_check",
            message="Duplicate check name: 'test_check'",
            node_path=["root", "check:test_check"],
        )
    )

    report.add_warning(
        ValidationIssue(
            rule="empty_check",
            message="Check 'test' has no assertions",
            node_path=["root", "check:test"],
        )
    )

    report_str = str(report)
    assert "ERROR" in report_str
    assert "WARNING" in report_str
    assert "Duplicate check name: 'test_check'" in report_str
    assert "Check 'test' has no assertions" in report_str
```

**Implement in `src/dqx/validators/report.py`**:
```python
"""Validation report for collecting and formatting issues."""

from __future__ import annotations

from dqx.validators.base import ValidationIssue


class ValidationReport:
    """Collects validation issues and formats them for display."""

    def __init__(self) -> None:
        """Initialize empty report."""
        self._errors: list[ValidationIssue] = []
        self._warnings: list[ValidationIssue] = []

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get all error issues."""
        return self._errors

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get all warning issues."""
        return self._warnings

    def add_error(self, issue: ValidationIssue) -> None:
        """Add an error issue to the report."""
        self._errors.append(issue)

    def add_warning(self, issue: ValidationIssue) -> None:
        """Add a warning issue to the report."""
        self._warnings.append(issue)

    def has_errors(self) -> bool:
        """Check if report contains any errors."""
        return len(self._errors) > 0

    def has_warnings(self) -> bool:
        """Check if report contains any warnings."""
        return len(self._warnings) > 0

    def __str__(self) -> str:
        """Format report as human-readable string."""
        if not self._errors and not self._warnings:
            return "No validation issues found."

        lines = []

        if self._errors:
            lines.append(f"\n{len(self._errors)} ERROR(S):")
            for issue in self._errors:
                lines.append(f"  [{issue.rule}] {issue.message}")
                lines.append(f"    Path: {' > '.join(issue.node_path)}")

        if self._warnings:
            lines.append(f"\n{len(self._warnings)} WARNING(S):")
            for issue in self._warnings:
                lines.append(f"  [{issue.rule}] {issue.message}")
                lines.append(f"    Path: {' > '.join(issue.node_path)}")

        return "\n".join(lines)
```

**Update `src/dqx/validators/__init__.py`**:
```python
"""Validation system for DQX verification suites."""

from dqx.validators.base import ValidationIssue
from dqx.validators.report import ValidationReport

__all__ = ["ValidationIssue", "ValidationReport"]
```

**Test and commit**:
```bash
uv run pytest tests/test_validators_report.py -v
git add -A
git commit -m "feat: add ValidationReport with separate error/warning tracking"
```

---

### Task 4: Implement First Validation Visitor - DuplicateCheckNameVisitor (TDD)

**Goal**: Detect duplicate check names using visitor pattern

**Test file** `tests/test_validators_visitors.py`:

```python
import pytest
from dqx.validators.visitors import DuplicateCheckNameVisitor
from dqx.graph.traversal import Graph
from dqx.graph.nodes import RootNode


def test_duplicate_check_name_visitor_no_duplicates():
    """Test visitor with no duplicate check names."""
    root = RootNode("test_suite")
    check1 = root.add_check("Check One")
    check2 = root.add_check("Check Two")

    graph = Graph(root)
    visitor = DuplicateCheckNameVisitor()

    # Traverse the graph
    graph.bfs(visitor)

    # Get issues after traversal
    issues = visitor.get_issues()
    assert len(issues) == 0


def test_duplicate_check_name_visitor_with_duplicates():
    """Test visitor with duplicate check names."""
    root = RootNode("test_suite")
    check1 = root.add_check("Duplicate Check")
    check2 = root.add_check("Duplicate Check")
    check3 = root.add_check("Unique Check")

    graph = Graph(root)
    visitor = DuplicateCheckNameVisitor()

    # Traverse the graph
    graph.bfs(visitor)

    # Get issues after traversal
    issues = visitor.get_issues()

    assert len(issues) == 1
    assert "Duplicate check name" in issues[0].message
    assert "Duplicate Check" in issues[0].message


def test_duplicate_check_name_visitor_multiple_duplicates():
    """Test visitor with multiple sets of duplicates."""
    root = RootNode("test_suite")
    root.add_check("Check A")
    root.add_check("Check A")
    root.add_check("Check B")
    root.add_check("Check B")
    root.add_check("Check B")  # Triple duplicate

    graph = Graph(root)
    visitor = DuplicateCheckNameVisitor()

    # Traverse the graph
    graph.bfs(visitor)

    # Get issues after traversal
    issues = visitor.get_issues()

    # Should find issues for both A and B
    assert len(issues) >= 2
```

**Implement in `src/dqx/validators/visitors.py`**:
```python
"""Validation visitors for DQX graph traversal."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from dqx.graph.base import NodeVisitor, BaseNode
from dqx.graph.nodes import RootNode, CheckNode, AssertionNode
from dqx.validators.base import ValidationIssue


class DuplicateCheckNameVisitor(NodeVisitor):
    """Visitor that detects duplicate check names."""

    name = "duplicate_check_names"
    is_error = True  # This rule produces errors

    def __init__(self) -> None:
        """Initialize visitor."""
        self._issues: list[ValidationIssue] = []
        self._check_names: dict[str, list[CheckNode]] = defaultdict(list)

    def visit(self, node: BaseNode) -> Any:
        """Visit a node in the graph."""
        if isinstance(node, CheckNode):
            self._check_names[node.name].append(node)

    def get_issues(self) -> list[ValidationIssue]:
        """Get validation issues after traversal."""
        # Process duplicates
        for name, nodes in self._check_names.items():
            if len(nodes) > 1:
                self._issues.append(
                    ValidationIssue(
                        rule=self.name,
                        message=f"Duplicate check name: '{name}' appears {len(nodes)} times",
                        node_path=["root", f"check:{name}"],
                    )
                )

        return self._issues
```

**Test and commit**:
```bash
uv run pytest tests/test_validators_visitors.py::test_duplicate_check_name -v
git add -A
git commit -m "feat: add DuplicateCheckNameVisitor using visitor pattern"
```

---

### Task 5: Add More Validation Visitors (TDD for each)

**Add to `tests/test_validators_visitors.py`**:

```python
import sympy as sp  # Add at top


def test_empty_check_visitor():
    """Test visitor that detects checks with no assertions."""
    root = RootNode("test_suite")

    # Empty check
    empty_check = root.add_check("Empty Check")

    # Check with assertions
    normal_check = root.add_check("Normal Check")
    normal_check.add_assertion(actual=sp.Symbol("x"), name="Test assertion")

    graph = Graph(root)
    visitor = EmptyCheckVisitor()

    graph.bfs(visitor)
    issues = visitor.get_issues()

    assert len(issues) == 1
    assert "Empty Check" in issues[0].message


def test_duplicate_assertion_name_visitor():
    """Test visitor that detects duplicate assertion names within a check."""
    root = RootNode("test_suite")
    check = root.add_check("Test Check")

    # Add assertions with duplicate names
    check.add_assertion(sp.Symbol("x"), name="Same Name")
    check.add_assertion(sp.Symbol("y"), name="Same Name")
    check.add_assertion(sp.Symbol("z"), name="Different Name")

    graph = Graph(root)
    visitor = DuplicateAssertionNameVisitor()

    graph.bfs(visitor)
    issues = visitor.get_issues()

    assert len(issues) == 1
    assert "Same Name" in issues[0].message
    assert "Test Check" in issues[0].message
```

**Add implementations to `src/dqx/validators/visitors.py`**:

```python
class EmptyCheckVisitor(NodeVisitor):
    """Visitor that detects checks with no assertions."""

    name = "empty_checks"
    is_error = False  # This rule produces warnings

    def __init__(self) -> None:
        """Initialize visitor."""
        self._issues: list[ValidationIssue] = []
        self._current_check: CheckNode | None = None

    def visit(self, node: BaseNode) -> Any:
        """Visit a node in the graph."""
        if isinstance(node, CheckNode):
            # Check if this check has no children
            if len(node.children) == 0:
                self._issues.append(
                    ValidationIssue(
                        rule=self.name,
                        message=f"Check '{node.name}' has no assertions",
                        node_path=["root", f"check:{node.name}"],
                    )
                )

    def get_issues(self) -> list[ValidationIssue]:
        """Get validation issues after traversal."""
        return self._issues


class DuplicateAssertionNameVisitor(NodeVisitor):
    """Visitor that detects duplicate assertion names within each check."""

    name = "duplicate_assertion_names"
    is_error = True  # This rule produces errors

    def __init__(self) -> None:
        """Initialize visitor."""
        self._issues: list[ValidationIssue] = []
        self._current_check: CheckNode | None = None
        self._assertion_names: dict[str, int] = {}

    def visit(self, node: BaseNode) -> Any:
        """Visit a node in the graph."""
        if isinstance(node, CheckNode):
            # Start tracking assertions for this check
            self._current_check = node
            self._assertion_names = defaultdict(int)

            # Count assertion names
            for assertion in node.children:
                if isinstance(assertion, AssertionNode) and assertion.name:
                    self._assertion_names[assertion.name] += 1

            # Report duplicates
            for name, count in self._assertion_names.items():
                if count > 1:
                    self._issues.append(
                        ValidationIssue(
                            rule=self.name,
                            message=(
                                f"Assertion name '{name}' appears {count} times "
                                f"in check '{node.name}'"
                            ),
                            node_path=[
                                "root",
                                f"check:{node.name}",
                                f"assertion:{name}",
                            ],
                        )
                    )

    def get_issues(self) -> list[ValidationIssue]:
        """Get validation issues after traversal."""
        return self._issues
```

**Commit after testing**:
```bash
uv run pytest tests/test_validators_visitors.py -v
git add -A
git commit -m "feat: add EmptyCheckVisitor and DuplicateAssertionNameVisitor"
```

---

### Task 6: Create the Main SuiteValidator (TDD)

**Goal**: Orchestrate all visitors and produce a final report

**Test file** `tests/test_suite_validator.py`:

```python
import pytest
from dqx.validators import SuiteValidator
from dqx.graph.traversal import Graph
from dqx.graph.nodes import RootNode
import sympy as sp


def test_suite_validator_clean_suite():
    """Test validator with a clean suite (no issues)."""
    root = RootNode("clean_suite")
    check = root.add_check("Good Check")
    check.add_assertion(sp.Symbol("x"), name="X is positive")

    graph = Graph(root)
    validator = SuiteValidator()

    report = validator.validate(graph)
    assert not report.has_errors()
    assert not report.has_warnings()


def test_suite_validator_with_issues():
    """Test validator with multiple issues."""
    root = RootNode("problematic_suite")

    # Duplicate check names
    root.add_check("Duplicate")
    root.add_check("Duplicate")

    # Empty check
    root.add_check("Empty Check")

    # Check with duplicate assertions
    check = root.add_check("Check with Issues")
    check.add_assertion(sp.Symbol("x"), name="Same")
    check.add_assertion(sp.Symbol("y"), name="Same")

    graph = Graph(root)
    validator = SuiteValidator()

    report = validator.validate(graph)

    # Should have errors and warnings
    assert report.has_errors()  # Duplicate names are errors
    assert report.has_warnings()  # Empty check is warning

    # Verify the report contains expected issues
    report_str = str(report)
    assert "Duplicate check name" in report_str
    assert "Empty Check" in report_str
    assert "Same" in report_str


def test_suite_validator_thread_safety():
    """Test validator with concurrent graph building (thread safety)."""
    import threading

    root = RootNode("thread_test_suite")

    def add_checks():
        for i in range(10):
            check = root.add_check(f"Thread Check {i}")
            check.add_assertion(sp.Symbol(f"x{i}"), name=f"Assert {i}")

    # Create multiple threads
    threads = []
    for _ in range(5):
        t = threading.Thread(target=add_checks)
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Validate - should not crash
    graph = Graph(root)
    validator = SuiteValidator()
    report = validator.validate(graph)

    # Should have duplicate check names from concurrent adds
    assert report.has_errors()


def test_suite_validator_edge_cases():
    """Test validator with edge cases."""
    # Empty suite
    empty_root = RootNode("empty_suite")
    empty_graph = Graph(empty_root)
    validator = SuiteValidator()

    report = validator.validate(empty_graph)
    assert not report.has_errors()
    assert not report.has_warnings()

    # Single check suite
    single_root = RootNode("single_suite")
    single_root.add_check("Only Check")
    single_graph = Graph(single_root)

    report = validator.validate(single_graph)
    assert not report.has_errors()
    assert report.has_warnings()  # Empty check warning
```

**Create `src/dqx/validators/suite_validator.py`**:

```python
"""Main validator that orchestrates all validation visitors."""

from __future__ import annotations

from dqx.graph.traversal import Graph
from dqx.validators.report import ValidationReport
from dqx.validators.visitors import (
    DuplicateAssertionNameVisitor,
    DuplicateCheckNameVisitor,
    EmptyCheckVisitor,
)


class SuiteValidator:
    """Main validator that runs all validation rules using visitors."""

    def __init__(self) -> None:
        """Initialize with built-in visitors."""
        # Store visitors with their error/warning classification
        self._error_visitors = [
            DuplicateCheckNameVisitor(),
            DuplicateAssertionNameVisitor(),
        ]
        self._warning_visitors = [
            EmptyCheckVisitor(),
        ]

    def validate(self, graph: Graph) -> ValidationReport:
        """Run all validation visitors and collect issues."""
        report = ValidationReport()

        # Run error-producing visitors
        for visitor in self._error_visitors:
            graph.bfs(visitor)
            for issue in visitor.get_issues():
                report.add_error(issue)

        # Run warning-producing visitors
        for visitor in self._warning_visitors:
            graph.bfs(visitor)
            for issue in visitor.get_issues():
                report.add_warning(issue)

        return report
```

**Update `src/dqx/validators/__init__.py`**:
```python
"""Validation system for DQX verification suites."""

from dqx.validators.base import ValidationIssue
from dqx.validators.report import ValidationReport
from dqx.validators.suite_validator import SuiteValidator

__all__ = ["ValidationIssue", "ValidationReport", "SuiteValidator"]
```

**Test and commit**:
```bash
uv run pytest tests/test_suite_validator.py -v
git add -A
git commit -m "feat: add SuiteValidator with visitor orchestration"
```

---

### Task 7: Integrate with VerificationSuite (TDD)

**Goal**: Make validation run automatically during suite collection

**Test file** `tests/test_api_validation_integration.py`:

```python
import pytest
from dqx.api import VerificationSuiteBuilder, check
from dqx.orm.repositories import InMemoryMetricDB
from dqx.common import DQXError, ResultKey
from dqx import get_logger

# Capture log output for testing
import logging


def test_suite_validation_on_collect_success(caplog):
    """Test that validation runs during collect with valid suite."""
    db = InMemoryMetricDB()

    @check(name="Valid Check 1")
    def check1(mp, ctx):
        ctx.assert_that(mp.num_rows()).where(name="Has data").is_gt(0)

    @check(name="Valid Check 2")
    def check2(mp, ctx):
        ctx.assert_that(mp.average("price")).where(name="Price check").is_positive()

    suite = (
        VerificationSuiteBuilder("Valid Suite", db)
        .add_check(check1)
        .add_check(check2)
        .build()
    )

    # Should not raise any errors
    suite.collect(suite._context, ResultKey())


def test_suite_validation_on_collect_failure():
    """Test that validation fails with duplicate check names."""
    db = InMemoryMetricDB()

    @check(name="Duplicate Name")
    def check1(mp, ctx):
        ctx.assert_that(mp.num_rows()).where(name="Test").is_gt(0)

    @check(name="Duplicate Name")  # Same name!
    def check2(mp, ctx):
        ctx.assert_that(mp.average("price")).where(name="Test").is_positive()

    suite = (
        VerificationSuiteBuilder("Invalid Suite", db)
        .add_check(check1)
        .add_check(check2)
        .build()
    )

    # Should raise DQXError with validation message
    with pytest.raises(DQXError) as exc_info:
        suite.collect(suite._context, ResultKey())

    assert "validation failed" in str(exc_info.value).lower()
    assert "Duplicate check name" in str(exc_info.value)


def test_suite_validation_warnings_logged(caplog):
    """Test that validation warnings are logged but don't fail."""
    db = InMemoryMetricDB()

    @check(name="Empty Check")
    def empty_check(mp, ctx):
        pass  # No assertions!

    suite = VerificationSuiteBuilder("Test Suite", db).add_check(empty_check).build()

    # Enable logging
    with caplog.at_level(logging.WARNING):
        # Should not raise error
        suite.collect(suite._context, ResultKey())

    # Check that warning was logged
    assert any(
        "validation warnings" in record.message.lower() for record in caplog.records
    )
    assert any("Empty Check" in record.message for record in caplog.records)


def test_explicit_validate_method():
    """Test explicit validation method."""
    db = InMemoryMetricDB()

    @check(name="Empty Check")
    def empty_check(mp, ctx):
        pass  # No assertions!

    suite = VerificationSuiteBuilder("Test Suite", db).add_check(empty_check).build()

    # Call validate explicitly
    report = suite.validate()

    assert report.has_warnings()
    assert "Empty Check" in str(report)
```

**Modify `src/dqx/api.py`**:

Add imports at top:
```python
from dqx import get_logger
from dqx.validators import SuiteValidator, ValidationReport

logger = get_logger(__name__)
```

In `VerificationSuite.__init__`, add:
```python
# Add after self._context initialization
self._validator = SuiteValidator()
```

Add new method to `VerificationSuite`:
```python
def validate(self) -> ValidationReport:
    """
    Explicitly validate the suite configuration.

    Returns:
        ValidationReport containing any issues found
    """
    # Create temporary context to collect checks
    temp_context = Context(suite=self._name, db=self._db)

    # Execute all checks to build graph
    for check_fn in self._checks:
        check_fn(self.provider, temp_context)

    # Run validation on the graph
    return self._validator.validate(temp_context._graph)
```

Modify `VerificationSuite.collect()` method:
```python
def collect(self, context: Context, key: ResultKey) -> None:
    """
    Collect all checks and build the dependency graph without executing analysis.

    MODIFIED: Now includes automatic validation
    """
    # Execute all checks to collect assertions
    for check_fn in self._checks:
        check_fn(self.provider, context)

    # NEW: Run validation
    report = self._validator.validate(context._graph)

    # Only raise on errors, log warnings
    if report.has_errors():
        raise DQXError(f"Suite validation failed:\n{report}")
    elif report.has_warnings():
        logger.warning(f"Suite validation warnings:\n{report}")
```

**Test and commit**:
```bash
uv run pytest tests/test_api_validation_integration.py -v
git add -A
git commit -m "feat: integrate validation with improved error handling"
```

---

### Task 8: Run Full Test Suite and Fix Issues

**Goal**: Ensure all existing tests still pass

**Steps**:

1. **Run existing tests to check for regressions**:
```bash
# Run all tests
uv run pytest -v

# If any fail, fix them before proceeding
```

2. **Run linting and type checking**:
```bash
# Type checking
uv run mypy src/dqx/validators/

# Linting
uv run ruff check src/dqx/validators/
uv run ruff format src/dqx/validators/
```

3. **Run the specific validator tests**:
```bash
uv run pytest tests/test_validators* -v
uv run pytest tests/test_api_validation_integration.py -v
```

4. **Check test coverage**:
```bash
uv run pytest tests/test_validators* --cov=dqx.validators --cov-report=term-missing
```

**Commit final changes**:
```bash
git add -A
git commit -m "feat: complete verification suite validator v2 implementation"
```

---

## Testing the Implementation

### Manual Testing Script

Create `test_validator_manually.py`:

```python
from dqx.api import VerificationSuiteBuilder, check
from dqx.orm.repositories import InMemoryMetricDB
from dqx.common import ResultKey

db = InMemoryMetricDB()


# Create problematic checks
@check(name="Duplicate Check")
def check1(mp, ctx):
    ctx.assert_that(mp.num_rows()).where(name="Test").is_gt(0)


@check(name="Duplicate Check")  # Duplicate!
def check2(mp, ctx):
    ctx.assert_that(mp.average("price")).where(name="Test").is_positive()


@check(name="Empty Check")
def check3(mp, ctx):
    pass  # No assertions!


@check(name="Good Check")
def check4(mp, ctx):
    # Duplicate assertion names
    ctx.assert_that(mp.sum("amount")).where(name="Same").is_gt(0)
    ctx.assert_that(mp.sum("tax")).where(name="Same").is_gt(0)


# Build suite
suite = (
    VerificationSuiteBuilder("Test Suite", db)
    .add_check(check1)
    .add_check(check2)
    .add_check(check3)
    .add_check(check4)
    .build()
)

# Try to validate explicitly
print("=== Explicit Validation ===")
report = suite.validate()
print(report)
print(f"\nHas errors: {report.has_errors()}")
print(f"Has warnings: {report.has_warnings()}")

# Try to run (should fail during collect due to errors)
print("\n=== Automatic Validation ===")
try:
    suite.collect(suite._context, ResultKey())
except Exception as e:
    print(f"Failed as expected: {e}")
```

Run it:
```bash
uv run python test_validator_manually.py
```

Expected output:
- Explicit validation shows both errors and warnings
- Automatic validation fails only on errors (duplicates)
- Warnings are logged but don't cause failure

---

## Key Improvements in Version 2

### 1. Simplified Design
- Removed severity from ValidationIssue
- Rules declare if they produce errors or warnings via `is_error` property
- Cleaner separation of concerns

### 2. Visitor Pattern
- Consistent with existing DQX patterns
- Better separation of traversal logic from validation logic
- Easier to test and maintain

### 3. Removed Unnecessary Rule
- UnnamedAssertionRule removed (API already enforces this)
- Focus on rules that can actually detect issues

### 4. Better Error Handling
- Only raises DQXError for actual errors
- Warnings are logged but don't fail execution
- Clearer user experience

### 5. Enhanced Testing
- Thread safety tests
- Edge case tests (empty suite, single check)
- Log output verification for warnings

---

## Final Checklist

Before considering the task complete:

- [ ] All test files pass
- [ ] Linting passes (`ruff check`)
- [ ] Type checking passes (`mypy`)
- [ ] Manual test script works as expected
- [ ] Integration doesn't break existing functionality
- [ ] Code follows DRY principle (no duplication)
- [ ] Each validation visitor is simple and focused (YAGNI)
- [ ] All changes are committed with descriptive messages
- [ ] Thread safety tests pass
- [ ] Performance is acceptable for large suites

---

## Summary

You've implemented an improved validation system that:
1. Checks for duplicate check names (error)
2. Checks for duplicate assertion names within checks (error)
3. Warns about empty checks (warning)
4. Uses visitor pattern consistent with DQX architecture
5. Only fails on errors, logs warnings
6. Runs automatically during suite collection
7. Can be run explicitly via `validate()` method
8. Produces clear, actionable error messages
9. Is thread-safe and handles edge cases
10. Is simpler and more maintainable than v1

The implementation follows TDD, commits frequently, and maintains backward compatibility while adding valuable new functionality.
