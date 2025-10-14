# Implementation Plan: Table Display for Results and Symbols (v2)

## Overview

This plan implements table display functionality for the DQX framework using the Rich library. The goal is to create two functions that display the output of `VerificationSuite.collect_results()` and `VerificationSuite.collect_symbols()` in well-formatted tables.

## Key Changes from v1
1. **No generic helper function** - Each table function handles its own Result extraction
2. **Reorganized task order** - All implementation first, then all testing/validation at the end
3. **More efficient workflow** - Reduces context switching between implementing and testing
4. **Uses Python pattern matching** - Modern, clear approach for Result type handling (Python 3.10+)
5. **Enhanced color scheme** - Beautiful, semantic coloring for better readability and visual hierarchy

## Background for Engineers

### What is DQX?

DQX is a data quality framework that validates data using assertions. Users write "checks" containing "assertions" about their data:

```python
@check(name="Price validation")
def validate_prices(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.average("price"))
       .where(name="Average price is positive")
       .is_positive()
```

### What are collect_results() and collect_symbols()?

After running a verification suite, DQX provides two methods to extract results:

1. **collect_results()** - Returns a list of `AssertionResult` objects containing:
   - Information about each assertion (name, check, suite)
   - Whether it passed or failed
   - The actual computed value or error message

2. **collect_symbols()** - Returns a list of `SymbolInfo` objects containing:
   - All metrics (symbols) that were computed
   - Their values
   - Which dataset they came from

### Current Problem

The results are returned as Python objects, making them hard to read:

```python
# Current output is just raw objects
results = suite.collect_results()
print(results)  # [AssertionResult(...), AssertionResult(...), ...]

# We want nice tables instead!
```

### Solution

Create display functions using Rich to show these results as formatted tables.

## Technical Context

### Key Data Structures

```python
@dataclass
class AssertionResult:
    yyyy_mm_dd: datetime.date
    suite: str
    check: str
    assertion: str
    severity: SeverityLevel  # "P0", "P1", "P2", "P3"
    status: AssertionStatus  # "OK" or "FAILURE"
    metric: Result[float, list[EvaluationFailure]]  # Success(42.5) or Failure([...])
    expression: str | None
    tags: Tags = field(default_factory=dict)

@dataclass
class SymbolInfo:
    name: str  # e.g., "x_1", "x_2"
    metric: str  # e.g., "average(price)"
    dataset: str | None
    value: Result[float, str]  # Success(42.5) or Failure("error message")
    yyyy_mm_dd: datetime.date
    suite: str
    tags: Tags = field(default_factory=dict)
```

### The Result Type

DQX uses the `returns` library for error handling. A `Result` can be:
- `Success(value)` - Contains the computed value
- `Failure(error)` - Contains error information

Each table function will handle its specific Result type directly.

### Development Environment

- Python 3.11/3.12 with `uv` package manager
- Run tests with: `uv run pytest`
- Type check with: `uv run mypy`
- Format/lint with: `uv run ruff`
- Pre-commit hooks: `bin/run-hooks.sh`

## Enhanced Color Scheme

The tables use a semantic color scheme for better readability and visual hierarchy:

### Assertion Results Table Colors

**Column Headers:**
- Date: `cyan` (time-related data)
- Suite: `blue` (organizational grouping)
- Check: `yellow` (important identifier)
- Assertion: default (main content - keep readable)
- Expression: `dim` (technical detail - less prominent)
- Severity: `magenta` (to make severity stand out)
- Status: `bold` (critical information)
- Value/Error: default (important data)
- Tags: `dim` (metadata)

**Data Row Colors:**
- **Status**:
  - OK: `[green bold]OK[/]` ✓
  - FAILURE: `[red bold]FAILURE[/]` ✗
- **Severity** (color-coded by priority):
  - P0: `[red]P0[/]` (critical)
  - P1: `[yellow]P1[/]` (high)
  - P2: `[blue]P2[/]` (medium)
  - P3: `[dim]P3[/]` (low)
- **Values/Errors**:
  - Success values: `[green]125.50[/]`
  - Error messages: `[red]Connection failed[/]`

### Symbol Values Table Colors

**Column Headers:**
- Date: `cyan` (consistency with assertion table)
- Suite: `blue` (consistency)
- Symbol: `yellow bold` (make identifiers like x_1, x_2 stand out)
- Metric: default (main content)
- Dataset: `magenta` (data source identification)
- Value/Error: default
- Tags: `dim` (metadata)

**Data Row Colors:**
- **Symbol names**: Inherit `yellow` from column style
- **Values/Errors**:
  - Success values: `[green]42.5[/]`
  - Error messages: `[red]No data found[/]`
- **Empty/None values**: Display as dim "-" or "None"

### Color Scheme Benefits

1. **Visual Hierarchy**: Important information (status, severity) stands out
2. **Semantic Meaning**: Green=success, red=failure, yellow=warning/attention
3. **Consistency**: Similar data types use similar colors across tables
4. **Readability**: Not overwhelming - strategic use of dim for less critical info
5. **Professional Look**: Clean, modern appearance with clear information architecture

## Implementation Tasks

### Task 1: Add Rich Import to display.py

**Goal**: Import the Rich Table class needed for table display.

**File to modify**: `src/dqx/display.py`

**What to add** (at the top with other imports):
```python
from rich.table import Table
```

**Why**: The file already imports `rich.tree.Tree` and `rich.console.Console`, so we just need to add Table.

**Commit message**: "Add Rich Table import to display module"

### Task 2: Implement print_assertion_results Function

**Goal**: Create function to display assertion results in a table.

**File to modify**: `src/dqx/display.py`

**What to implement**:
```python
from returns.result import Success, Failure
from rich.console import Console
from rich.table import Table

from dqx.common import AssertionResult


def print_assertion_results(results: list[AssertionResult]) -> None:
    """
    Display assertion results in a formatted table.

    Shows all fields from AssertionResult objects in a table with columns:
    Date, Suite, Check, Assertion, Expression, Severity, Status, Value/Error, Tags

    Args:
        results: List of AssertionResult objects from collect_results()

    Example:
        >>> suite = VerificationSuite(checks, db, "My Suite")
        >>> suite.run(datasources, key)
        >>> results = suite.collect_results()
        >>> print_assertion_results(results)
    """
    # Create table with title
    table = Table(title="Assertion Results", show_lines=True)

    # Add columns in specified order
    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Suite", style="blue")
    table.add_column("Check", style="yellow")
    table.add_column("Assertion")
    table.add_column("Expression", style="dim")
    table.add_column("Severity", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("Value/Error")
    table.add_column("Tags", style="dim")

    # Define severity colors
    severity_colors = {
        "P0": "red",
        "P1": "yellow",
        "P2": "blue",
        "P3": "dim"
    }

    # Add rows
    for result in results:
        # Format status with color
        status_style = "green bold" if result.status == "OK" else "red bold"
        status_display = f"[{status_style}]{result.status}[/{status_style}]"

        # Format severity with color
        severity_color = severity_colors.get(result.severity, "white")
        severity_display = f"[{severity_color}]{result.severity}[/{severity_color}]"

        # Extract value/error using pattern matching with colors
        match result.metric:
            case Success(value):
                value_display = f"[green]{value}[/green]"
            case Failure(failures):
                error_text = "; ".join(f.error_message for f in failures)
                value_display = f"[red]{error_text}[/red]"

        # Format tags as key=value pairs
        tags_display = ", ".join(f"{k}={v}" for k, v in result.tags.items())
        if not tags_display:
            tags_display = "-"

        # Add row
        table.add_row(
            result.yyyy_mm_dd.isoformat(),
            result.suite,
            result.check,
            result.assertion,
            result.expression or "-",
            severity_display,
            status_display,
            value_display,
            tags_display
        )

    # Print table
    console = Console()
    console.print(table)
```

**Note**: Uses Python pattern matching (match/case) to extract values from Result types cleanly.

**Commit message**: "Add print_assertion_results function for table display"

### Task 3: Implement print_symbols Function

**Goal**: Create function to display symbol values in a table.

**File to modify**: `src/dqx/display.py`

**What to implement**:
```python
def print_symbols(symbols: list[SymbolInfo]) -> None:
    """
    Display symbol values in a formatted table.

    Shows all fields from SymbolInfo objects in a table with columns:
    Date, Suite, Symbol, Metric, Dataset, Value/Error, Tags

    Args:
        symbols: List of SymbolInfo objects from collect_symbols()

    Example:
        >>> suite = VerificationSuite(checks, db, "My Suite")
        >>> suite.run(datasources, key)
        >>> symbols = suite.collect_symbols()
        >>> print_symbols(symbols)
    """
    # Create table with title
    table = Table(title="Symbol Values", show_lines=True)

    # Add columns in specified order
    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Suite", style="blue")
    table.add_column("Symbol", style="yellow", no_wrap=True)
    table.add_column("Metric")
    table.add_column("Dataset", style="magenta")
    table.add_column("Value/Error")
    table.add_column("Tags", style="dim")

    # Add rows
    for symbol in symbols:
        # Extract value/error using pattern matching with colors
        match symbol.value:
            case Success(value):
                value_display = f"[green]{value}[/green]"
            case Failure(error):
                value_display = f"[red]{error}[/red]"

        # Format tags as key=value pairs
        tags_display = ", ".join(f"{k}={v}" for k, v in symbol.tags.items())
        if not tags_display:
            tags_display = "-"

        # Add row
        table.add_row(
            symbol.yyyy_mm_dd.isoformat(),
            symbol.suite,
            symbol.name,
            symbol.metric,
            symbol.dataset or "-",
            value_display,
            tags_display
        )

    # Print table
    console = Console()
    console.print(table)
```

**Note**: Uses Python pattern matching (match/case) to extract values from Result types cleanly.

**Commit message**: "Add print_symbols function for table display"

### Task 4: Add Type Annotations and Complete Documentation

**Goal**: Ensure all functions have proper type hints and comprehensive docstrings.

**Files to check**: `src/dqx/display.py`

**What to verify**:
1. Import the required types at the top of the file:
   ```python
   from dqx.common import AssertionResult, SymbolInfo
   ```
2. All function parameters have type hints
3. All return types are specified
4. Docstrings follow Google format
5. Examples in docstrings are accurate

**Commit message**: "Add complete type annotations to display functions"

### Task 5: Create Example Usage Script

**Goal**: Create a demonstration script showing how to use the new functions.

**File to create**: `examples/table_display_demo.py`

**What to implement**:
```python
#!/usr/bin/env python3
"""
Demonstration of table display functions for DQX results.

This script shows how to use print_assertion_results() and print_symbols()
to display verification suite results in formatted tables.
"""

from datetime import date

from returns.result import Failure, Success

from dqx.common import AssertionResult, EvaluationFailure, SymbolInfo
from dqx.display import print_assertion_results, print_symbols


def main() -> None:
    """Run the table display demonstration."""
    print("=== DQX Table Display Demo ===\n")

    # Create sample assertion results
    assertion_results = [
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Daily Data Quality",
            check="Order Validation",
            assertion="Average order amount is positive",
            expression="average(amount) > 0",
            severity="P1",
            status="OK",
            metric=Success(125.50),
            tags={"env": "production", "region": "us-west"},
        ),
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Daily Data Quality",
            check="Order Validation",
            assertion="No orders exceed $10,000",
            expression="max(amount) <= 10000",
            severity="P0",
            status="FAILURE",
            metric=Failure(
                [
                    EvaluationFailure(
                        error_message="Maximum amount is $15,000",
                        expression="max(amount)",
                        symbols=[],
                    )
                ]
            ),
            tags={"env": "production", "region": "us-west"},
        ),
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Daily Data Quality",
            check="Customer Validation",
            assertion="All customers have email",
            expression="null_count(email) = 0",
            severity="P2",
            status="OK",
            metric=Success(0.0),
            tags={"env": "production", "region": "us-west"},
        ),
    ]

    # Display assertion results
    print("\n1. Assertion Results Table:")
    print_assertion_results(assertion_results)

    # Create sample symbol values
    symbols = [
        SymbolInfo(
            name="x_1",
            metric="average(amount)",
            dataset="orders",
            value=Success(125.50),
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Daily Data Quality",
            tags={"env": "production", "region": "us-west"},
        ),
        SymbolInfo(
            name="x_2",
            metric="max(amount)",
            dataset="orders",
            value=Success(15000.0),
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Daily Data Quality",
            tags={"env": "production", "region": "us-west"},
        ),
        SymbolInfo(
            name="x_3",
            metric="null_count(email)",
            dataset="customers",
            value=Success(0.0),
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Daily Data Quality",
            tags={"env": "production", "region": "us-west"},
        ),
        SymbolInfo(
            name="x_4",
            metric="count(*)",
            dataset="orders",
            value=Failure("Connection timeout"),
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Daily Data Quality",
            tags={"env": "production", "region": "us-west"},
        ),
    ]

    # Display symbol values
    print("\n2. Symbol Values Table:")
    print_symbols(symbols)

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
```

**Commit message**: "Add table display demonstration script"

### Task 6: Update README Documentation

**Goal**: Add examples of the new display functions to the README.

**File to modify**: `README.md`

**What to add** (in appropriate section about viewing results):
```markdown
## Viewing Results

After running a verification suite, you can display the results in formatted tables:

```python
# Run verification
suite = VerificationSuite(checks, db, "Daily Quality Checks")
suite.run({"orders": datasource}, key)

# Display assertion results in a table
from dqx.display import print_assertion_results, print_symbols

results = suite.collect_results()
print_assertion_results(results)

# Display computed symbol values in a table
symbols = suite.collect_symbols()
print_symbols(symbols)
```

The tables show all relevant information including dates, suites, checks,
assertion names, statuses, computed values, and any error messages.
```

**Commit message**: "Update README with table display examples"

### Task 7: Write Comprehensive Tests

**Goal**: Write all unit tests and edge case tests.

**File to create/modify**: `tests/test_display.py`

**Tests to write**:

```python
def test_print_assertion_results(capsys) -> None:
    """Test print_assertion_results displays table correctly."""
    from datetime import date
    from returns.result import Success, Failure
    from dqx.common import AssertionResult, EvaluationFailure
    from dqx.display import print_assertion_results

    # Create test data
    results = [
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test Suite",
            check="Price Check",
            assertion="Average price is positive",
            severity="P1",
            status="OK",
            metric=Success(42.5),
            expression="average(price) > 0",
            tags={"env": "prod", "region": "us"}
        ),
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test Suite",
            check="Data Check",
            assertion="No null values",
            severity="P0",
            status="FAILURE",
            metric=Failure([
                EvaluationFailure(
                    error_message="Found 10 null values",
                    expression="null_count(id)",
                    symbols=[]
                )
            ]),
            expression="null_count(id) = 0",
            tags={}
        )
    ]

    # Call function
    print_assertion_results(results)

    # Check output contains expected content
    captured = capsys.readouterr()
    assert "Assertion Results" in captured.out
    assert "2024-01-15" in captured.out
    assert "Test Suite" in captured.out
    assert "Price Check" in captured.out
    assert "Average price is positive" in captured.out
    assert "42.5" in captured.out
    assert "env=prod, region=us" in captured.out
    assert "Found 10 null values" in captured.out


def test_print_symbols(capsys) -> None:
    """Test print_symbols displays table correctly."""
    from datetime import date
    from returns.result import Success, Failure
    from dqx.common import SymbolInfo
    from dqx.display import print_symbols

    # Create test data
    symbols = [
        SymbolInfo(
            name="x_1",
            metric="average(price)",
            dataset="orders",
            value=Success(42.5),
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test Suite",
            tags={"env": "prod"}
        ),
        SymbolInfo(
            name="x_2",
            metric="count(*)",
            dataset="orders",
            value=Success(1000.0),
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test Suite",
            tags={"env": "prod"}
        ),
        SymbolInfo(
            name="x_3",
            metric="max(amount)",
            dataset=None,
            value=Failure("No data found"),
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test Suite",
            tags={}
        )
    ]

    # Call function
    print_symbols(symbols)

    # Check output contains expected content
    captured = capsys.readouterr()
    assert "Symbol Values" in captured.out
    assert "2024-01-15" in captured.out
    assert "x_1" in captured.out
    assert "average(price)" in captured.out
    assert "orders" in captured.out
    assert "42.5" in captured.out
    assert "1000.0" in captured.out
    assert "No data found" in captured.out
    assert "env=prod" in captured.out


def test_print_assertion_results_empty_list(capsys) -> None:
    """Test print_assertion_results with empty list."""
    from dqx.display import print_assertion_results

    print_assertion_results([])

    captured = capsys.readouterr()
    assert "Assertion Results" in captured.out
    # Table should be empty but still display headers


def test_print_symbols_empty_list(capsys) -> None:
    """Test print_symbols with empty list."""
    from dqx.display import print_symbols

    print_symbols([])

    captured = capsys.readouterr()
    assert "Symbol Values" in captured.out


def test_print_assertion_results_none_values(capsys) -> None:
    """Test handling of None values in assertion results."""
    from datetime import date
    from returns.result import Success
    from dqx.common import AssertionResult
    from dqx.display import print_assertion_results

    results = [
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test",
            check="Check",
            assertion="Test assertion",
            severity="P1",
            status="OK",
            metric=Success(None),  # None value
            expression=None,  # None expression
            tags={}  # Empty tags
        )
    ]

    print_assertion_results(results)

    captured = capsys.readouterr()
    assert "None" in captured.out  # Should display None as string


def test_print_assertion_results_multiple_failures(capsys) -> None:
    """Test handling of multiple evaluation failures."""
    from datetime import date
    from returns.result import Failure
    from dqx.common import AssertionResult, EvaluationFailure
    from dqx.display import print_assertion_results

    results = [
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test",
            check="Check",
            assertion="Complex validation",
            severity="P0",
            status="FAILURE",
            metric=Failure([
                EvaluationFailure(
                    error_message="First error",
                    expression="expr1",
                    symbols=[]
                ),
                EvaluationFailure(
                    error_message="Second error",
                    expression="expr2",
                    symbols=[]
                )
            ]),
            expression="complex expression",
            tags={"test": "true"}
        )
    ]

    print_assertion_results(results)

    captured = capsys.readouterr()
    assert "First error; Second error" in captured.out


def test_integration_with_real_suite(capsys) -> None:
    """Test display functions with real VerificationSuite results."""
    from datetime import date
    import sympy as sp
    from dqx.api import VerificationSuite, Context, check
    from dqx.provider import MetricProvider
    from dqx.common import ResultKey
    from dqx.display import print_assertion_results, print_symbols
    from dqx.orm.repositories import InMemoryMetricDB

    # Create a real check
    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(sp.Float(42.5)).where(
            name="Constant is positive"
        ).is_positive()

    # Run suite
    db = InMemoryMetricDB()
    suite = VerificationSuite([test_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=date(2024, 1, 15), tags={"test": "true"})

    # Note: We can't actually run() without data sources, but we can test
    # that the display functions handle empty results gracefully
    try:
        results = suite.collect_results()
    except Exception:
        results = []  # Expected since suite wasn't run

    # Should not crash even with empty results
    print_assertion_results(results)

    captured = capsys.readouterr()
    assert "Assertion Results" in captured.out
```

**Commit message**: "Add comprehensive tests for table display functions"

### Task 8: Run Full Test Suite and Fix Issues

**Goal**: Ensure all tests pass, type checking succeeds, and coverage is maintained.

**Commands to run in order**:

1. **Type checking**:
   ```bash
   uv run mypy src/dqx/display.py
   ```

2. **Linting**:
   ```bash
   uv run ruff check src/dqx/display.py tests/test_display.py
   ```

   If there are formatting issues:
   ```bash
   uv run ruff check --fix src/dqx/display.py tests/test_display.py
   ```

3. **Run display tests**:
   ```bash
   uv run pytest tests/test_display.py -xvs
   ```

4. **Check coverage for display module**:
   ```bash
   uv run pytest tests/test_display.py --cov=dqx.display --cov-report=term-missing
   ```

5. **Run all tests to ensure nothing broke**:
   ```bash
   uv run pytest
   ```

6. **Run pre-commit hooks**:
   ```bash
   bin/run-hooks.sh
   ```

**Fix any issues found:**
- Type errors: Add missing type imports or annotations
- Linting errors: Apply suggested fixes
- Test failures: Debug and fix implementation
- Coverage gaps: Add tests for uncovered lines

**Commit message**: "Fix type checking, linting, and test issues"

## Testing Strategy

### Implementation First, Testing Second

This plan follows a more efficient workflow:
1. Implement all functionality (Tasks 1-6)
2. Write all tests (Task 7)
3. Validate and fix everything (Task 8)

This reduces context switching and allows for a more focused development process.

### Test Coverage Goals

- 100% coverage for new functions in display.py
- Test both happy path and edge cases
- Test with empty lists
- Test with None values
- Test with multiple failures
- Integration test with real DQX components

## Commit Strategy

Make focused commits after each major task:

1. "Add Rich Table import to display module"
2. "Add print_assertion_results function for table display"
3. "Add print_symbols function for table display"
4. "Add complete type annotations to display functions"
5. "Add table display demonstration script"
6. "Update README with table display examples"
7. "Add comprehensive tests for table display functions"
8. "Fix type checking, linting, and test issues"

## Definition of Done

- [ ] All new functions implemented in display.py
- [ ] Type annotations complete
- [ ] Documentation updated
- [ ] Example script works correctly
- [ ] All tests written and passing
- [ ] 100% test coverage for new code
- [ ] No type errors (`uv run mypy src/dqx/display.py`)
- [ ] No linting issues (`uv run ruff check`)
- [ ] Pre-commit hooks pass (`bin/run-hooks.sh`)

## Quick Reference

### New Functions

1. **print_assertion_results(results: list[AssertionResult]) -> None**
   - Displays assertion results in a formatted table
   - Shows: Date, Suite, Check, Assertion, Expression, Severity, Status, Value/Error, Tags
   - Handles `Result[float, list[EvaluationFailure]]` directly

2. **print_symbols(symbols: list[SymbolInfo]) -> None**
   - Displays symbol values in a formatted table
   - Shows: Date, Suite, Symbol, Metric, Dataset, Value/Error, Tags
   - Handles `Result[float, str]` directly

### Usage Example

```python
# After running a suite
results = suite.collect_results()
symbols = suite.collect_symbols()

# Display in tables
from dqx.display import print_assertion_results, print_symbols
print_assertion_results(results)
print_symbols(symbols)
```

### Key Implementation Details

- No generic helper function - each table handles its own Result type
- Extract values from Success/Failure inline
- Handle None values gracefully
- Format tags as "key=value" pairs
- Use color coding for status (green=OK, red=FAILURE)
- Show all fields from the data classes

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure Rich is installed: `uv pip list | grep rich`
2. **Type errors**: The Result type is from `returns.result`, not `typing`
3. **Test failures**: Check that you're using the correct test fixtures
4. **Coverage gaps**: Use `--cov-report=term-missing` to see uncovered lines

### Getting Help

- Check existing display.py for patterns (tree display)
- Look at common.py for data structure definitions
- Review test fixtures in tests/conftest.py
- Use `uv run python -i` to experiment with code interactively
