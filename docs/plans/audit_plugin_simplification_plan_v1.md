# Audit Plugin Simplification Implementation Plan

## Overview
This plan details the simplification of the AuditPlugin in the DQX project, replacing the Rich table-based output with a simpler text-based format. No backward compatibility is required.

## Objectives
1. Replace Rich table-based output with text-based format
2. Maintain visual clarity using Rich color markup
3. Add support for displaying tags with proper formatting
4. Simplify the code by removing table-related logic
5. Follow TDD principles throughout implementation

## Current State
The AuditPlugin currently uses Rich tables to display:
- Header table with suite information
- Statistics table with assertion/symbol counts

## Target State
Text-based output format:
```
═══ DQX Audit Report ═══
Suite: Simple test suite
Date: 2025-01-15
Tags: none  (or "env=prod, region=us-east" if tags exist)
Duration: 22.95ms
Datasets: ds1, ds2

Execution Summary:
  Assertions: 5 total, 3 passed (60.0%), 2 failed (40.0%)
  Symbols: 3 total, 2 successful, 1 failed
══════════════════════
```

## Implementation Tasks

### Task Group 1: Write Failing Tests (TDD Red Phase)
**Goal**: Create comprehensive tests that define the expected behavior

#### Task 1.1: Create test for basic text format
- **File**: `tests/test_plugin_manager.py`
- **Method**: Update `test_audit_plugin_process()`
- **Changes**:
  ```python
  def test_audit_plugin_process(monkeypatch):
      # Remove Table import checks
      # Check for specific text output
      assert "═══ DQX Audit Report ═══" in captured_output
      assert "Suite: Test Suite" in captured_output
      assert "Date: 2025-01-15" in captured_output
      assert "Duration:" in captured_output
  ```

#### Task 1.2: Create test for tag formatting
- **File**: `tests/test_plugin_manager.py`
- **Method**: Add `test_audit_plugin_with_tags()` and `test_audit_plugin_no_tags()`
- **Changes**:
  ```python
  def test_audit_plugin_with_tags(monkeypatch):
      # Test with tags: {"env": "prod", "region": "us-east"}
      # Expect: "Tags: env=prod, region=us-east"

  def test_audit_plugin_no_tags(monkeypatch):
      # Test with empty tags
      # Expect: "Tags: none"
  ```

#### Task 1.3: Update statistics test
- **File**: `tests/test_plugin_manager.py`
- **Method**: Update `test_audit_plugin_with_statistics()`
- **Changes**:
  ```python
  def test_audit_plugin_with_statistics(monkeypatch):
      # Remove Table expectations
      # Check for merged summary format:
      assert "Assertions: 5 total, 3 passed (60.0%), 2 failed (40.0%)" in output
      assert "Symbols: 3 total, 2 successful, 1 failed" in output
  ```

#### Task 1.4: Run tests to confirm failures
- **Command**: `uv run pytest tests/test_plugin_manager.py::TestAuditPlugin -v`
- **Expected**: All updated tests should fail
- **Verify**: Review failure messages to ensure they align with expectations

### Task Group 2: Implement Minimal Code (TDD Green Phase)
**Goal**: Make the tests pass with minimal implementation

#### Task 2.1: Remove table imports and update process method header
- **File**: `src/dqx/plugins.py`
- **Class**: `AuditPlugin`
- **Changes**:
  ```python
  # Remove imports:
  # from rich.table import Table
  # from rich import box

  def process(self, context: PluginExecutionContext) -> None:
      self.console.print()
      self.console.print("[bold blue]═══ DQX Audit Report ═══[/bold blue]")
      self.console.print(f"[cyan]Suite:[/cyan] {context.suite_name}")
      self.console.print(f"[cyan]Date:[/cyan] {context.key.yyyy_mm_dd}")
  ```

#### Task 2.2: Implement tag formatting
- **File**: `src/dqx/plugins.py`
- **Method**: Continue in `process()`
- **Changes**:
  ```python
  # Tags handling
  if context.key.tags:
      sorted_tags = ", ".join(f"{k}={v}" for k, v in sorted(context.key.tags.items()))
      self.console.print(f"[cyan]Tags:[/cyan] {sorted_tags}")
  else:
      self.console.print("[cyan]Tags:[/cyan] none")

  self.console.print(f"[cyan]Duration:[/cyan] {context.duration_ms:.2f}ms")

  if context.datasources:
      self.console.print(f"[cyan]Datasets:[/cyan] {', '.join(context.datasources)}")
  ```

#### Task 2.3: Implement execution summary
- **File**: `src/dqx/plugins.py`
- **Method**: Continue in `process()`
- **Changes**:
  ```python
  # Calculate statistics
  total = context.total_assertions()
  passed = context.assertions_by_severity[Severity.P1]
  failed = context.failed_assertions()
  pass_rate = (passed / total * 100) if total > 0 else 0.0

  self.console.print()
  self.console.print("[cyan]Execution Summary:[/cyan]")

  # Assertions line
  if total > 0:
      self.console.print(f"  Assertions: {total} total, [green]{passed} passed ({pass_rate:.1f}%)[/green], [red]{failed} failed ({100-pass_rate:.1f}%)[/red]")
  else:
      self.console.print("  Assertions: 0 total, 0 passed (0.0%), 0 failed (0.0%)")

  # Symbols line (only if symbols exist)
  if context.symbols:
      successful_symbols = context.total_symbols() - context.failed_symbols()
      failed_symbols = context.failed_symbols()
      self.console.print(f"  Symbols: {context.total_symbols()} total, {successful_symbols} successful, {failed_symbols} failed")

  self.console.print("[bold blue]══════════════════════[/bold blue]")
  self.console.print()
  ```

#### Task 2.4: Run tests iteratively
- **Command**: `uv run pytest tests/test_plugin_manager.py::TestAuditPlugin -v`
- **Action**: Fix any remaining test failures
- **Goal**: All tests should pass

### Task Group 3: Refactor and Clean (TDD Refactor Phase)
**Goal**: Improve code quality while keeping tests green

#### Task 3.1: Extract tag formatting logic (optional)
- **File**: `src/dqx/plugins.py`
- **Method**: Add helper method if beneficial
- **Changes**:
  ```python
  def _format_tags(self, tags: dict[str, str] | None) -> str:
      """Format tags for display."""
      if not tags:
          return "none"
      return ", ".join(f"{k}={v}" for k, v in sorted(tags.items()))
  ```

#### Task 3.2: Clean up code
- **File**: `src/dqx/plugins.py`
- **Actions**:
  - Remove all unused imports
  - Remove any commented-out table code
  - Ensure consistent formatting
  - Update docstrings

#### Task 3.3: Run quality checks
- **Commands**:
  ```bash
  uv run mypy src/dqx/plugins.py
  uv run ruff check src/dqx/plugins.py --fix
  uv run pytest tests/test_plugin_manager.py -v
  ```
- **Expected**: All checks should pass

#### Task 3.4: Commit Task Groups 1-3
- **Command**: `git add -A && git commit -m "test: add failing tests for simplified audit plugin format"`
- **Command**: `git add -A && git commit -m "feat: implement text-based audit plugin output"`
- **Command**: `git add -A && git commit -m "refactor: clean up audit plugin implementation"`

### Task Group 4: Documentation and Examples
**Goal**: Update documentation and create examples

#### Task 4.1: Update plugin docstrings
- **File**: `src/dqx/plugins.py`
- **Class**: `AuditPlugin`
- **Changes**:
  - Update class docstring to mention "text-based audit report"
  - Update `process()` method docstring
  - Update metadata description

#### Task 4.2: Create example demonstrating new format
- **File**: `examples/audit_plugin_demo.py`
- **Content**:
  ```python
  """Demonstration of the AuditPlugin text-based output format."""

  from dqx import VerificationSuite, ResultKey
  from dqx.plugins import PluginManager

  # Example with tags
  key_with_tags = ResultKey(
      yyyy_mm_dd="2025-01-15",
      tags={"env": "prod", "region": "us-east"}
  )

  # Example without tags
  key_no_tags = ResultKey(yyyy_mm_dd="2025-01-15")

  # Create and run suite demonstrations...
  ```

#### Task 4.3: Search for other examples using AuditPlugin
- **Command**: `uv run grep -r "AuditPlugin" examples/`
- **Action**: Update any found examples to work with new format

#### Task 4.4: Run all examples to verify
- **Commands**:
  ```bash
  uv run python examples/audit_plugin_demo.py
  uv run python examples/plugin_demo.py  # if it uses AuditPlugin
  ```
- **Expected**: Examples should run without errors

### Task Group 5: Integration Testing
**Goal**: Ensure the plugin works correctly in real scenarios

#### Task 5.1: Create integration test
- **File**: `tests/test_plugin_integration.py`
- **Method**: Add test for full suite execution with AuditPlugin
- **Changes**:
  ```python
  def test_audit_plugin_full_integration():
      # Create a real VerificationSuite
      # Add checks and assertions
      # Run with AuditPlugin enabled
      # Verify output format
  ```

#### Task 5.2: Test empty results edge case
- **File**: `tests/test_plugin_manager.py`
- **Method**: Update `test_audit_plugin_empty_results()`
- **Changes**: Ensure it expects the new text format with zeros

#### Task 5.3: Run full test suite
- **Command**: `uv run pytest tests/ -v`
- **Expected**: All tests should pass
- **Action**: Fix any unexpected failures

#### Task 5.4: Check code coverage
- **Command**: `uv run pytest tests/test_plugin_manager.py -v --cov=dqx.plugins`
- **Expected**: High coverage for AuditPlugin
- **Action**: Add tests if coverage gaps exist

### Task Group 6: Final Verification
**Goal**: Ensure all quality standards are met

#### Task 6.1: Run pre-commit hooks
- **Command**: `bin/run-hooks.sh`
- **Expected**: All hooks should pass
- **Action**: Fix any issues found

#### Task 6.2: Run full pytest suite
- **Command**: `uv run pytest tests/ -v`
- **Expected**: 100% pass rate
- **Action**: Fix any failures

#### Task 6.3: Final commit
- **Command**: `git add -A && git commit -m "docs: update documentation and examples for simplified audit plugin"`

## Success Criteria
1. All tests pass (100% pass rate)
2. No linting or type checking errors
3. Code coverage maintained or improved
4. Examples run successfully
5. Documentation is accurate and complete

## Risk Mitigation
- No backward compatibility concerns (confirmed by user)
- Tests written first ensure behavior is well-defined
- Incremental commits allow easy rollback if needed
- Full test suite run catches any unexpected side effects

## Notes
- The simplified format is more maintainable and easier to understand
- Rich color markup is retained for visual appeal
- Tag sorting ensures consistent output
- The design allows for easy future extensions
