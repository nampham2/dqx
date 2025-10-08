# Implementation Plan: Fix Type Annotations in test_api.py

## Overview

This plan addresses the issue where `test_api.py` uses `Any` type annotations for function parameters in `@check` decorated functions. These should use proper types: `MetricProvider` for `mp` and `Context` for `ctx`.

**IMPORTANT**: This task ONLY modifies `tests/test_api.py`. No other files should be changed.

## Background for Engineers

### What is DQX?
DQX is a data quality framework that uses a graph-based architecture to manage dependencies between checks, metrics, and assertions. The key components relevant to this task are:

- **MetricProvider**: A class that provides access to metrics and extended metrics
- **Context**: A class that provides execution context for data quality checks, including assertion utilities
- **@check decorator**: A decorator that marks functions as data quality checks

### Current Problem
The test file uses `Any` type for parameters, which:
- Makes code harder to understand
- Loses type safety benefits
- Makes IDE autocomplete less helpful
- Can hide potential bugs

## Step-by-Step Implementation

### Step 1: Understand the Current State

1. Open `tests/test_api.py`
2. Look for all occurrences of functions with parameters `mp: Any, ctx: Any`
3. Note that these appear inside `@check` decorated functions within test cases

Current problematic pattern:
```python
@check
def test_check(mp: Any, ctx: Any) -> None:
    # function body
```

### Step 2: Add Required Imports

At the top of `tests/test_api.py`, after the existing imports, add:
```python
from dqx.api import MetricProvider
```

Note: `Context` is already imported in the file, so you don't need to add it again.

### Step 3: Replace Type Annotations

Find every occurrence of:
```python
def test_check(mp: Any, ctx: Any) -> None:
```

And replace with:
```python
def test_check(mp: MetricProvider, ctx: Context) -> None:
```

Based on the search results, there are 3 occurrences to fix:
1. Inside `test_assertion_methods_return_none()`
2. Inside `test_no_assertion_chaining()`
3. Inside `test_multiple_assertions_on_same_metric()`

### Step 4: Run Type Checking

Before making changes:
```bash
uv run mypy tests/test_api.py
```

After making changes:
```bash
uv run mypy tests/test_api.py
```

The type checking should pass without errors.

### Step 5: Run Tests

Execute the tests to ensure nothing broke:
```bash
uv run pytest tests/test_api.py -v
```

All tests should pass.

### Step 6: Run Code Quality Checks

Check with ruff:
```bash
uv run ruff check tests/test_api.py
```

If there are any issues:
```bash
uv run ruff check tests/test_api.py --fix
```

### Step 7: Verify Your Changes

Use git to see exactly what changed:
```bash
git diff tests/test_api.py
```

You should see:
- One import added: `from dqx.api import MetricProvider`
- Three replacements of `Any` with proper types

## Complete Code Changes

Here's exactly what needs to change:

### Import Section
Add this import (if not already present):
```python
from dqx.api import MetricProvider
```

### Function Signatures
Replace these three occurrences:

1. In `test_assertion_methods_return_none()`:
```python
# OLD:
def test_check(mp: Any, ctx: Any) -> None:
# NEW:
def test_check(mp: MetricProvider, ctx: Context) -> None:
```

2. In `test_no_assertion_chaining()`:
```python
# OLD:
def test_check(mp: Any, ctx: Any) -> None:
# NEW:
def test_check(mp: MetricProvider, ctx: Context) -> None:
```

3. In `test_multiple_assertions_on_same_metric()`:
```python
# OLD:
def test_check(mp: Any, ctx: Any) -> None:
# NEW:
def test_check(mp: MetricProvider, ctx: Context) -> None:
```

## Testing Procedure

### 1. Unit Tests
```bash
# Run only the modified test file
uv run pytest tests/test_api.py -v

# Expected: All tests pass
```

### 2. Type Checking
```bash
# Check types in the modified file
uv run mypy tests/test_api.py

# Expected: No type errors
```

### 3. Linting
```bash
# Check code style
uv run ruff check tests/test_api.py

# Expected: No issues
```

### 4. Full Test Suite (Optional but Recommended)
```bash
# Run all tests to ensure no regressions
uv run pytest

# Expected: All tests pass
```

## Common Pitfalls to Avoid

1. **Don't modify other files**: Even if you see similar issues in other test files, this task is specifically for `test_api.py` only
2. **Don't remove the Any import**: The file imports `Any` from typing - leave this import as it may be used elsewhere
3. **Don't change function logic**: Only change type annotations, not the function implementations
4. **Check import order**: Make sure imports follow the project's convention (standard library, third-party, local)

## Verification Checklist

- [ ] Added `MetricProvider` import
- [ ] Found all 3 occurrences of `mp: Any, ctx: Any`
- [ ] Replaced all 3 occurrences with proper types
- [ ] Tests still pass
- [ ] Type checking passes
- [ ] Linting passes
- [ ] Git diff shows only expected changes

## Why This Matters

1. **Type Safety**: Catches errors at development time rather than runtime
2. **IDE Support**: Better autocomplete and inline documentation
3. **Code Clarity**: Makes it clear what types are expected
4. **Maintenance**: Easier for future developers to understand the code

## Commit Message

When committing, use:
```
fix: add proper type annotations to test_api.py check functions

- Replace Any with MetricProvider for mp parameter
- Replace Any with Context for ctx parameter
- Improves type safety and code clarity
```

## Notes

- The `MetricProvider` and `Context` classes are defined in `src/dqx/api.py`
- These types are the standard parameter types for all `@check` decorated functions
- This change aligns the test code with the production code patterns
