# Verification Suite Validator Implementation Plan (DEPRECATED - See V2)

**NOTE: This plan has been superseded by verification_suite_validator_plan_v2.md which incorporates architectural feedback for a better design.**

## Overview

You'll be implementing a validation system that checks DQX verification suites for common configuration errors BEFORE they run. Think of it as a "linter" for data quality checks that catches issues like duplicate names, empty checks, and other problems early.

## Background Context

### What is DQX?
DQX is a data quality framework. Users define "checks" (functions that validate data) and group them into "suites". A suite runs these checks against datasets to ensure data quality.

### Current Problem
Right now, users can create invalid suites with issues like:
- Two checks with the same name
- Assertions with duplicate names within a check
- Empty checks (no assertions)
- Other configuration problems

These issues only surface during execution, making debugging harder.

### Solution
Build a validator that runs AFTER suite creation but BEFORE execution to catch these issues early.

## Architecture Overview

```
VerificationSuite (existing)
    |
    ├── Has checks (functions)
    ├── Has a graph structure (nodes representing checks/assertions)
    └── NEW: Has a validator that inspects the graph
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
└── rules.py
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

**Goal**: Create ValidationIssue and ValidationRule protocol

**Test file**: `tests/test_validators_base.py`

**Write this test FIRST**:
```python
import pytest
from dqx.validators.base import ValidationIssue, ValidationRule
from dqx.graph.traversal import Graph
from dqx.graph.nodes import RootNode

def test_validation_issue_creation():
    """Test creating a validation issue."""
    issue = ValidationIssue(
        rule="test_rule",
        severity="error",
        message="Something went wrong",
        node_path=["root", "check1"]
    )

    assert issue.rule == "test_rule"
    assert issue.severity == "error"
    assert issue.message == "Something went wrong"
    assert issue.node_path == ["root", "check1"]

def test_validation_rule_protocol():
    """Test that ValidationRule is a proper protocol."""
    # This tests the protocol definition
    class MockRule:
        name = "mock_rule"

        def validate(self, graph: Graph) -> list[ValidationIssue]:
            return []

    rule = MockRule()
    assert hasattr(rule, 'name')
    assert hasattr(rule, 'validate')
```

**Run test (it will fail)**:
```bash
uv run pytest tests/test_validators_base.py -v
```

**Now implement in `src/dqx/validators/base.py`**:
```python
"""Base classes and protocols for validation system."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from dqx.graph.traversal import Graph


@dataclass
class ValidationIssue:
    """Represents a single validation issue found in the graph."""
    rule: str  # Name of the rule that found this issue
    severity: Literal["error", "warning", "info"]
    message: str  # Human-readable description
    node_path: list[str]  # Path to the problematic node


class ValidationRule(Protocol):
    """Protocol for validation rules."""
    name: str

    def validate(self, graph: Graph) -> list[ValidationIssue]:
        """Run validation and return any issues found."""
        ...
```

**Update `src/dqx/validators/__init__.py`**:
```python
"""Validation system for DQX verification suites."""
from dqx.validators.base import ValidationIssue, ValidationRule

__all__ = ["ValidationIssue", "ValidationRule"]
```

**Run test again (should pass)**:
```bash
uv run pytest tests/test_validators_base.py -v
```

**Commit**:
```bash
git add -A
git commit -m "feat: add ValidationIssue and ValidationRule base classes"
```

---

### Task 3: Create ValidationReport (TDD)

**Goal**: Build a class to collect and format validation issues

**Add to test file** `tests/test_validators_report.py`:

```python
import pytest
from dqx.validators.report import ValidationReport
from dqx.validators.base import ValidationIssue


def test_validation_report_empty():
    """Test empty validation report."""
    report = ValidationReport()

    assert not report.has_errors()
    assert not report.has_warnings()
    assert len(report.issues) == 0


def test_validation_report_add_issues():
    """Test adding issues to report."""
    report = ValidationReport()

    error = ValidationIssue(
        rule="test", severity="error",
        message="Error found", node_path=["root"]
    )
    warning = ValidationIssue(
        rule="test", severity="warning",
        message="Warning found", node_path=["root"]
    )

    report.add_issue(error)
    report.add_issue(warning)

    assert report.has_errors()
    assert report.has_warnings()
    assert len(report.issues) == 2


def test_validation_report_string_format():
    """Test report string formatting."""
    report = ValidationReport()

    report.add_issue(ValidationIssue(
        rule="duplicate_check",
        severity="error",
        message="Duplicate check name: 'test_check'",
        node_path=["root", "check:test_check"]
    ))

    report_str = str(report)
    assert "ERROR" in report_str
    assert "Duplicate check name: 'test_check'" in report_str
    assert "duplicate_check" in report_str
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
        self._issues: list[ValidationIssue] = []

    @property
    def issues(self) -> list[ValidationIssue]:
        """Get all issues."""
        return self._issues

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the report."""
        self._issues.append(issue)

    def add_issues(self, issues: list[ValidationIssue]) -> None:
        """Add multiple issues to the report."""
        self._issues.extend(issues)

    def has_errors(self) -> bool:
        """Check if report contains any errors."""
        return any(issue.severity == "error" for issue in self._issues)

    def has_warnings(self) -> bool:
        """Check if report contains any warnings."""
        return any(issue.severity == "warning" for issue in self._issues)

    def __str__(self) -> str:
        """Format report as human-readable string."""
        if not self._issues:
            return "No validation issues found."

        lines = []

        # Group by severity
        errors = [i for i in self._issues if i.severity == "error"]
        warnings = [i for i in self._issues if i.severity == "warning"]
        infos = [i for i in self._issues if i.severity == "info"]

        if errors:
            lines.append(f"\n{len(errors)} ERROR(S):")
            for issue in errors:
                lines.append(f"  [{issue.rule}] {issue.message}")
                lines.append(f"    Path: {' > '.join(issue.node_path)}")

        if warnings:
            lines.append(f"\n{len(warnings)} WARNING(S):")
            for issue in warnings:
                lines.append(f"  [{issue.rule}] {issue.message}")
                lines.append(f"    Path: {' > '.join(issue.node_path)}")

        if infos:
            lines.append(f"\n{len(infos)} INFO(S):")
            for issue in infos:
                lines.append(f"  [{issue.rule}] {issue.message}")
                lines.append(f"    Path: {' > '.join(issue.node_path)}")

        return "\n".join(lines)
```

**Update `src/dqx/validators/__init__.py`**:
```python
"""Validation system for DQX verification suites."""
from dqx.validators.base import ValidationIssue, ValidationRule
from dqx.validators.report import ValidationReport

__all__ = ["ValidationIssue", "ValidationRule", "ValidationReport"]
```

**Test and commit**:
```bash
uv run pytest tests/test_validators_report.py -v
git add -A
git commit -m "feat: add ValidationReport for collecting issues"
```

---

### Task 4: Implement First Validation Rule - DuplicateCheckNameRule (TDD)

**Goal**: Detect when two checks have the same name

**Understanding the domain**:
- A "check" is a node in the graph
- Each check has a name
- Names should be unique across all checks

**Test file** `tests/test_validators_rules.py`:

```python
import pytest
from dqx.validators.rules import DuplicateCheckNameRule
from dqx.graph.traversal import Graph
from dqx.graph.nodes import RootNode


def test_duplicate_check_name_rule_no_duplicates():
    """Test rule with no duplicate check names."""
    # Create a graph with unique check names
    root = RootNode("test_suite")
    check1 = root.add_check("Check One")
    check2 = root.add_check("Check Two")

    graph = Graph(root)
    rule = DuplicateCheckNameRule()

    issues = rule.validate(graph)
    assert len(issues) == 0


def test_duplicate_check_name_rule_with_duplicates():
    """Test rule with duplicate check names."""
    # Create a graph with duplicate names
    root = RootNode("test_suite")
    check1 = root.add_check("Duplicate Check")
    check2 = root.add_check("Duplicate Check")
    check3 = root.add_check("Unique Check")

    graph = Graph(root)
    rule = DuplicateCheckNameRule()

    issues = rule.validate(graph)

    # Should find one issue (for the duplicate)
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert "Duplicate check name" in issues[0].message
    assert "Duplicate Check" in issues[0].message


def test_duplicate_check_name_rule_multiple_duplicates():
    """Test rule with multiple sets of duplicates."""
    root = RootNode("test_suite")
    root.add_check("Check A")
    root.add_check("Check A")
    root.add_check("Check B")
    root.add_check("Check B")
    root.add_check("Check B")  # Triple duplicate

    graph = Graph(root)
    rule = DuplicateCheckNameRule()

    issues = rule.validate(graph)

    # Should find issues for both A and B
    assert len(issues) >= 2
    assert all(issue.severity == "error" for issue in issues)
```

**Implement in `src/dqx/validators/rules.py`**:
```python
"""Built-in validation rules for DQX suites."""
from __future__ import annotations

from collections import defaultdict

from dqx.graph.traversal import Graph
from dqx.validators.base import ValidationIssue


class DuplicateCheckNameRule:
    """Detects duplicate check names in the suite."""

    name = "duplicate_check_names"

    def validate(self, graph: Graph) -> list[ValidationIssue]:
        """Find checks with duplicate names."""
        issues = []
        name_counts = defaultdict(int)

        # Count occurrences of each check name
        for check in graph.root.children:
            name_counts[check.name] += 1

        # Report duplicates
        for name, count in name_counts.items():
            if count > 1:
                issues.append(ValidationIssue(
                    rule=self.name,
                    severity="error",
                    message=f"Duplicate check name: '{name}' appears {count} times",
                    node_path=["root", f"check:{name}"]
                ))

        return issues
```

**Test and commit**:
```bash
uv run pytest tests/test_validators_rules.py::test_duplicate_check_name -v
git add -A
git commit -m "feat: add DuplicateCheckNameRule"
```

---

### Task 5: Add More Validation Rules (TDD for each)

**Add to `tests/test_validators_rules.py`**:

```python
def test_empty_check_rule():
    """Test rule that detects checks with no assertions."""
    root = RootNode("test_suite")

    # Empty check
    empty_check = root.add_check("Empty Check")

    # Check with assertions
    normal_check = root.add_check("Normal Check")
    normal_check.add_assertion(
        actual=sp.Symbol("x"),
        name="Test assertion"
    )

    graph = Graph(root)
    rule = EmptyCheckRule()

    issues = rule.validate(graph)
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert "Empty Check" in issues[0].message


def test_duplicate_assertion_name_rule():
    """Test rule that detects duplicate assertion names within a check."""
    root = RootNode("test_suite")
    check = root.add_check("Test Check")

    # Add assertions with duplicate names
    check.add_assertion(sp.Symbol("x"), name="Same Name")
    check.add_assertion(sp.Symbol("y"), name="Same Name")
    check.add_assertion(sp.Symbol("z"), name="Different Name")

    graph = Graph(root)
    rule = DuplicateAssertionNameRule()

    issues = rule.validate(graph)
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert "Same Name" in issues[0].message
    assert "Test Check" in issues[0].message
```

**Add implementations to `src/dqx/validators/rules.py`**:

```python
import sympy as sp  # Add this import at top

class EmptyCheckRule:
    """Detects checks with no assertions."""

    name = "empty_checks"

    def validate(self, graph: Graph) -> list[ValidationIssue]:
        """Find checks with no assertions."""
        issues = []

        for check in graph.root.children:
            if len(check.children) == 0:
                issues.append(ValidationIssue(
                    rule=self.name,
                    severity="warning",
                    message=f"Check '{check.name}' has no assertions",
                    node_path=["root", f"check:{check.name}"]
                ))

        return issues


class DuplicateAssertionNameRule:
    """Detects duplicate assertion names within each check."""

    name = "duplicate_assertion_names"

    def validate(self, graph: Graph) -> list[ValidationIssue]:
        """Find assertions with duplicate names within checks."""
        issues = []

        for check in graph.root.children:
            # Track assertion names within this check
            name_counts = defaultdict(int)

            for assertion in check.children:
                if assertion.name:  # Only count named assertions
                    name_counts[assertion.name] += 1

            # Report duplicates
            for name, count in name_counts.items():
                if count > 1:
                    issues.append(ValidationIssue(
                        rule=self.name,
                        severity="error",
                        message=(
                            f"Assertion name '{name}' appears {count} times "
                            f"in check '{check.name}'"
                        ),
                        node_path=["root", f"check:{check.name}", f"assertion:{name}"]
                    ))

        return issues


class UnnamedAssertionRule:
    """Detects assertions without descriptive names."""

    name = "unnamed_assertions"

    def validate(self, graph: Graph) -> list[ValidationIssue]:
        """Find assertions without names."""
        issues = []

        for check in graph.root.children:
            for i, assertion in enumerate(check.children):
                if not assertion.name:
                    issues.append(ValidationIssue(
                        rule=self.name,
                        severity="warning",
                        message=(
                            f"Unnamed assertion #{i+1} in check '{check.name}'. "
                            "Consider adding a descriptive name."
                        ),
                        node_path=["root", f"check:{check.name}", f"assertion:{i+1}"]
                    ))

        return issues
```

**Commit each rule after testing**:
```bash
uv run pytest tests/test_validators_rules.py -v
git add -A
git commit -m "feat: add EmptyCheckRule, DuplicateAssertionNameRule, UnnamedAssertionRule"
```

---

### Task 6: Create the Main SuiteValidator (TDD)

**Goal**: Orchestrate all rules and produce a final report

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
    check.add_assertion(sp.Symbol("z"))  # Unnamed

    graph = Graph(root)
    validator = SuiteValidator()

    report = validator.validate(graph)

    # Should have errors and warnings
    assert report.has_errors()  # Duplicate names are errors
    assert report.has_warnings()  # Empty check and unnamed assertion are warnings

    # Verify the report contains expected issues
    report_str = str(report)
    assert "Duplicate check name" in report_str
    assert "Empty Check" in report_str
    assert "Same" in report_str
    assert "Unnamed assertion" in report_str
```

**Create `src/dqx/validators/suite_validator.py`**:

```python
"""Main validator that orchestrates all validation rules."""
from __future__ import annotations

from dqx.graph.traversal import Graph
from dqx.validators.base import ValidationRule
from dqx.validators.report import ValidationReport
from dqx.validators.rules import (
    DuplicateAssertionNameRule,
    DuplicateCheckNameRule,
    EmptyCheckRule,
    UnnamedAssertionRule,
)


class SuiteValidator:
    """Main validator that runs all validation rules."""

    def __init__(self) -> None:
        """Initialize with built-in rules."""
        self.rules: list[ValidationRule] = [
            DuplicateCheckNameRule(),
            DuplicateAssertionNameRule(),
            EmptyCheckRule(),
            UnnamedAssertionRule(),
        ]

    def validate(self, graph: Graph) -> ValidationReport:
        """Run all validation rules and collect issues."""
        report = ValidationReport()

        for rule in self.rules:
            issues = rule.validate(graph)
            report.add_issues(issues)

        return report
```

**Update `src/dqx/validators/__init__.py`**:
```python
"""Validation system for DQX verification suites."""
from dqx.validators.base import ValidationIssue, ValidationRule
from dqx.validators.report import ValidationReport
from dqx.validators.suite_validator import SuiteValidator

__all__ = ["ValidationIssue", "ValidationRule", "ValidationReport", "SuiteValidator"]
```

**Test and commit**:
```bash
uv run pytest tests/test_suite_validator.py -v
git add -A
git commit -m "feat: add SuiteValidator orchestrator"
```

---

### Task 7: Integrate with VerificationSuite (TDD)

**Goal**: Make validation run automatically during suite collection

**Test file** `tests/test_api_validation_integration.py`:

```python
import pytest
from dqx.api import VerificationSuiteBuilder, check
from dqx.orm.repositories import InMemoryMetricDB
from dqx.common import DQXError


def test_suite_validation_on_collect_success():
    """Test that validation runs during collect with valid suite."""
    db = InMemoryMetricDB()

    @check(name="Valid Check 1")
    def check1(mp, ctx):
        ctx.assert_that(mp.num_rows()).where(name="Has data").is_gt(0)

    @check(name="Valid Check 2")
    def check2(mp, ctx):
        ctx.assert_that(mp.average("price")).where(name="Price check").is_positive()

    suite = VerificationSuiteBuilder("Valid Suite", db)\
        .add_check(check1)\
        .add_check(check2)\
        .build()

    # Should not raise any errors
    # Create a dummy context to test collection
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

    suite = VerificationSuiteBuilder("Invalid Suite", db)\
        .add_check(check1)\
        .add_check(check2)\
        .build()

    # Should raise DQXError with validation message
    with pytest.raises(DQXError) as exc_info:
        suite.collect(suite._context, ResultKey())

    assert "validation failed" in str(exc_info.value).lower()
    assert "Duplicate check name" in str(exc_info.value)


def test_explicit_validate_method():
    """Test explicit validation method."""
    db = InMemoryMetricDB()

    @check(name="Empty Check")
    def empty_check(mp, ctx):
        pass  # No assertions!

    suite = VerificationSuiteBuilder("Test Suite", db)\
        .add_check(empty_check)\
        .build()

    # Call validate explicitly
    report = suite.validate()

    assert report.has_warnings()
    assert "Empty Check" in str(report)
```

**Modify `src/dqx/api.py`**:

Add import at top:
```python
from dqx.validators import SuiteValidator, ValidationReport
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
    if report.has_errors():
        raise DQXError(f"Suite validation failed:\n{report}")
```

**Important Notes for Integration**:
- The `ResultKey` import already exists, so no need to add it
- The validation happens AFTER checks are collected but BEFORE analysis
- We use existing `DQXError` instead of creating new exception types

**Test and commit**:
```bash
uv run pytest tests/test_api_validation_integration.py -v
git add -A
git commit -m "feat: integrate validation with VerificationSuite"
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
git commit -m "feat: complete verification suite validator implementation"
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
    # Unnamed assertion
    ctx.assert_that(mp.num_rows()).is_gt(100)  # Missing where()!

# Build suite
suite = VerificationSuiteBuilder("Test Suite", db)\
    .add_check(check1)\
    .add_check(check2)\
    .add_check(check3)\
    .add_check(check4)\
    .build()

# Try to validate explicitly
print("=== Explicit Validation ===")
report = suite.validate()
print(report)

# Try to run (should fail during collect)
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

Expected output should show all validation errors clearly formatted.

---

## Common Pitfalls and How to Avoid Them

### 1. Import Errors
**Problem**: "Cannot import ValidationIssue"
**Solution**: Make sure all __init__.py files have proper imports

### 2. Type Errors
**Problem**: "Graph has no attribute 'root'"
**Solution**: Check that you're importing from `dqx.graph.traversal` not `dqx.graph`

### 3. Test Failures
**Problem**: Existing tests fail after integration
**Solution**: The validation might catch real issues in test fixtures. Either:
- Fix the test fixtures to be valid
- Or check if validation is too strict

### 4. Missing Imports in Tests
**Problem**: "Symbol 'sp' not found"
**Solution**: Add `import sympy as sp` at the top of test files

### 5. Validation Not Running
**Problem**: Invalid suites don't raise errors
**Solution**: Check that you modified the `collect()` method correctly

---

## Understanding Key Concepts

### Graph Structure
```
RootNode (the suite)
├── CheckNode ("Order validation")
│   ├── AssertionNode ("price > 0")
│   └── AssertionNode ("quantity > 0")
└── CheckNode ("Customer validation")
    └── AssertionNode ("email not null")
```

### Validation Flow
1. User creates checks with `@check` decorator
2. Suite is built with `VerificationSuiteBuilder`
3. During `collect()`, checks are executed to build graph
4. Validator inspects graph for issues
5. If errors found, raises `DQXError`
6. Otherwise, suite continues to execution

### Severity Levels
- **error**: Must be fixed (duplicates, conflicts)
- **warning**: Should be fixed (empty checks, unnamed assertions)
- **info**: Nice to fix (style issues)

---

## Final Checklist

Before considering the task complete:

- [ ] All test files pass
- [ ] Linting passes (`ruff check`)
- [ ] Type checking passes (`mypy`)
- [ ] Manual test script works as expected
- [ ] Integration doesn't break existing functionality
- [ ] Code follows DRY principle (no duplication)
- [ ] Each validation rule is simple and focused (YAGNI)
- [ ] All changes are committed with descriptive messages

---

## Summary

You've implemented a validation system that:
1. Checks for duplicate check names (error)
2. Checks for duplicate assertion names within checks (error)
3. Warns about empty checks (warning)
4. Warns about unnamed assertions (warning)
5. Runs automatically during suite collection
6. Can be run explicitly via `validate()` method
7. Produces clear, actionable error messages
8. Is extensible for future validation rules

The implementation follows TDD, commits frequently, and maintains backward compatibility while adding valuable new functionality.
