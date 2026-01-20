# Tolerance Modifier Fix - Context for Implementation

This document provides background context for implementing tolerance modifier support across all DQL comparison operators.

## DQX Architecture Overview

### Relevant Components

**DQL Parser** (`src/dqx/dql/parser.py`, `src/dqx/dql/grammar.lark`)
- Purpose: Parses DQL source files into AST (Abstract Syntax Tree)
- Key methods: `parse()`, `tolerance_mod()`
- How tolerance fix relates: Parser already handles tolerance correctly; no changes needed
- Grammar rule: `tolerance_mod: TOLERANCE_KW NUMBER` (line 59) applies to all assertions

**DQL AST** (`src/dqx/dql/ast.py`)
- Purpose: Dataclass representations of parsed DQL structures
- Key types: `Assertion`, `Collection`, `Check`, `Suite`
- How tolerance fix relates: `Assertion.tolerance` field already exists; stores parsed tolerance value
- Relevant: Line 65 - `tolerance: float | None = None`

**API Layer** (`src/dqx/api.py`)
- Purpose: Bridges DQL AST to Python API execution (MetricProvider, Context, assertions)
- Key methods: `_apply_condition()` (lines 1780-1848), `_eval_simple_expr()`
- How tolerance fix relates: **This is the primary file to modify** - currently only passes tolerance for `==` operator
- Pattern: Each condition branch calls `ready.is_*()` method to create assertion node

**Comparison Functions** (`src/dqx/functions.py`)
- Purpose: Core floating-point comparison logic with tolerance
- Key functions: `is_gt`, `is_geq`, `is_lt`, `is_leq`, `is_eq`, `is_neq`, `is_between` (lines 13-193)
- How tolerance fix relates: All functions already support `tol` parameter; DQL just needs to pass it through
- Tolerance default: `EPSILON = 1e-9` (line 7)

**AssertionReady** (`src/dqx/api.py`, lines 213-450)
- Purpose: Fluent API for creating assertions after `.where()` call
- Key methods: `is_gt()`, `is_geq()`, `is_lt()`, `is_leq()`, `is_eq()`, `is_neq()`, `is_between()`
- How tolerance fix relates: All methods already accept `tol` parameter; DQL needs to call them with correct argument

## Code Patterns to Follow

**IMPORTANT**: All patterns reference AGENTS.md standards.

### Pattern 1: Conditional Method Call with Optional Parameter

**When to use**: When a method parameter is optional and comes from AST field that may be `None`.

**Example from DQX** (api.py:1816-1821):
```python
if assertion_ast.tolerance:
    ready.is_eq(threshold, tol=assertion_ast.tolerance)
else:
    ready.is_eq(threshold)
```

**Reference**: AGENTS.MD §type-hints - handle `None` explicitly, don't pass `None` as parameter value when method has default.

**Apply to tolerance fix**: Use this exact pattern for all six comparison operators (`>`, `>=`, `<`, `<=`, `!=`, `between`).

**Why not call with `tol=None`?**
```python
# WRONG - passes None explicitly, which may not match default behavior
ready.is_gt(threshold, tol=assertion_ast.tolerance)  # if tolerance is None

# RIGHT - uses method's default parameter (EPSILON)
if assertion_ast.tolerance:
    ready.is_gt(threshold, tol=assertion_ast.tolerance)
else:
    ready.is_gt(threshold)
```

### Pattern 2: Tolerance Semantics for Comparisons

**Example** (functions.py:42-55):
```python
def is_gt(a: float, b: float, tol: float = EPSILON) -> bool:
    return a > b + tol
```

**Reference**: AGENTS.md §testing-patterns - verify behavior with values at boundaries.

**Apply to tolerance fix**: When writing tests, use values that are exactly `threshold ± tolerance` to verify boundary behavior.

**Key insight**: Different operators handle tolerance differently:
- `>` / `<`: Tolerance makes comparison *stricter* (must exceed by more)
- `>=` / `<=`: Tolerance makes comparison *more lenient* (allows undershoot/overshoot)
- `==` / `!=`: Tolerance creates symmetric buffer zone

### Pattern 3: Between with Dual Bounds

**Example** (functions.py:180-193):
```python
def is_between(a: float, lower: float, upper: float, tol: float = EPSILON) -> bool:
    return is_geq(a, lower, tol) and is_leq(a, upper, tol)
```

**Reference**: AGENTS.md §type-hints - multiple parameters of same type clearly named.

**Apply to tolerance fix**: For `between` operator, apply single tolerance value to both bounds.

**Implementation in _apply_condition**:
```python
elif cond == "between":
    # ... existing checks ...
    lower = self._eval_simple_expr(assertion_ast.threshold, tunables)
    upper = self._eval_simple_expr(assertion_ast.threshold_upper, tunables)
    if assertion_ast.tolerance:
        ready.is_between(lower, upper, tol=assertion_ast.tolerance)
    else:
        ready.is_between(lower, upper)
```

### Pattern 4: Test Organization for Operators

**Example** (tests/test_api.py:966-990):
```python
class TestAssertionReady:
    def test_is_neq_with_tolerance(self) -> None:
        """Test is_neq with custom tolerance."""
        # ... setup ...

        # Test with default tolerance
        # ... assertion ...

        # Test with custom tolerance
        # ... assertion ...
```

**Reference**: AGENTS.md §test-structure - organize in classes, descriptive names.

**Apply to tolerance fix**: Create test class `TestToleranceOnAllOperators` with one test method per operator.

## Code Standards Reference

**All code must follow AGENTS.md standards**:
- **Import Order**: AGENTS.md §import-order (stdlib → third-party → local)
- **Type Hints**: AGENTS.md §type-hints (strict mode, all functions typed)
- **Docstrings**: AGENTS.md §docstrings (Google style, not needed for test functions)
- **Testing**: AGENTS.md §testing-standards (100% coverage, organized in classes)
- **Coverage**: AGENTS.md §coverage-requirements (100%, no exceptions)

### Specific Standards for This Fix

**Imports**: No new imports needed - all necessary modules already imported in api.py.

**Type hints**: No new functions - modifying existing method with no signature change.

**Docstrings**: No docstring changes needed - tolerance behavior already documented in `AssertionReady.is_*` methods.

**Testing**:
- Test file: `tests/dql/test_dql.py` or create new file `tests/dql/test_tolerance_operators.py`
- Organization: One test class, multiple test methods
- Naming: `test_tolerance_on_<operator>` pattern
- Coverage: Must test with and without tolerance for each operator

## Testing Patterns

**Reference**: AGENTS.md §testing-patterns

Test organization:
- Mirror source structure: `src/dqx/api.py` → `tests/dql/test_tolerance_operators.py`
- Organize in classes: `class TestToleranceOperators:`
- Use fixtures from `tests/fixtures/` if needed (e.g., `InMemoryMetricDB`)

**For this fix**:
- **Fixtures to use**: `InMemoryMetricDB` (lightweight in-memory database for tests)
- **New fixtures needed**: None
- **Test pattern**: Create DQL string with tolerance, parse, execute, verify pass/fail status

**Example test structure**:
```python
class TestToleranceOperators:
    """Tests for tolerance modifier on all DQL comparison operators."""

    def test_tolerance_on_greater_than(self) -> None:
        """Test tolerance modifier on > operator."""
        db = InMemoryMetricDB()
        # Register test data...

        dql = """
        suite "Test" {
            check "Test" on dataset {
                assert average(value) > 100
                    tolerance 5
                    name "Greater than with tolerance"
            }
        }
        """
        suite = VerificationSuite(dql=StringIO(dql), db=db)
        # Execute and verify...

    # ... more tests for other operators ...
```

## Common Pitfalls

### Pitfall 1: Passing `None` as Tolerance Parameter

**Problem**: Calling `ready.is_gt(threshold, tol=None)` when tolerance is not specified.

**Why it's wrong**: The default parameter value is `functions.EPSILON` (1e-9), not `None`. Passing `None` explicitly may cause type errors or unexpected behavior.

**Solution**: Use conditional call - only pass `tol=...` when `assertion_ast.tolerance` is not `None`.

**Correct code**:
```python
if assertion_ast.tolerance:
    ready.is_gt(threshold, tol=assertion_ast.tolerance)
else:
    ready.is_gt(threshold)  # Uses default EPSILON
```

### Pitfall 2: Misunderstanding Tolerance Direction

**Problem**: Assuming tolerance always "relaxes" a constraint.

**Reality**: For strict inequalities (`>`, `<`), tolerance makes the constraint *stricter*.

**Example**:
```python
# assert value > 100 tolerance 5
# Means: value > 105 (NOT value > 95)
# Tolerance adds to the threshold for >, subtracts for <
```

**Solution**: Refer to functions.py implementation:
- `is_gt(a, b, tol)` → `a > b + tol` (stricter)
- `is_geq(a, b, tol)` → `a > b - tol` (more lenient)

### Pitfall 3: Copy-Paste Errors

**Problem**: Copying the `if/else` block but forgetting to update the method name.

**Example of wrong code**:
```python
# Copied from > branch but forgot to change method:
elif cond == "<":
    threshold = self._eval_simple_expr(assertion_ast.threshold, tunables)
    if assertion_ast.tolerance:
        ready.is_gt(threshold, tol=assertion_ast.tolerance)  # WRONG! Should be is_lt
    else:
        ready.is_gt(threshold)  # WRONG!
```

**Solution**: After copy-paste, immediately change the method name. Use IDE find/replace with caution.

### Pitfall 4: Forgetting to Test Both Branches

**Problem**: Only testing assertions with tolerance, not verifying default behavior without tolerance.

**Solution**: Each test should verify:
1. Behavior **with** tolerance (pass/fail as expected)
2. Behavior **without** tolerance (uses default EPSILON)
3. Regression: Existing assertions without tolerance still work

### Pitfall 5: Insufficient Edge Case Testing

**Problem**: Only testing values clearly inside or outside the range.

**Solution**: Test boundary values:
- Exactly at `threshold - tolerance`
- Exactly at `threshold`
- Exactly at `threshold + tolerance`

**Example**:
```python
# For: assert value > 100 tolerance 5
# Test values: 105.0 (boundary), 105.1 (pass), 104.9 (fail)
```

## Related PRs and Issues

**Similar features**: None directly, but related to:
- PR #XXX (if any): Tolerance implementation in Python API
- Issue #XXX (if any): DQL parity with Python API

**Relevant history**:
- Tolerance was originally implemented only for `==` in DQL
- Python API has always supported tolerance on all operators (since functions.py creation)
- This fix closes the parity gap between DQL and Python API

## DQL Language Design Context

### Why Tolerance Exists

**Background**: Floating-point arithmetic is inherently imprecise. Values like `0.1 + 0.2` may not exactly equal `0.3` due to IEEE 754 representation limits.

**DQX approach**: All comparisons use tolerance (epsilon) to avoid spurious failures from floating-point precision errors.

**Default tolerance**: `1e-9` (EPSILON) - suitable for most cases.

**User-specified tolerance**: When data has known measurement error or when comparing aggregates that accumulate rounding errors.

**Example use case**:
```dql
# Banking reconciliation - totals may differ by cents due to rounding
assert sum(amount, dataset=transactions) == sum(amount, dataset=settlements)
    tolerance 0.02
    name "Amount reconciliation"
```

### DQL Modifier System

**Modifiers** in DQL are optional clauses that customize assertion behavior:
- `name`: Human-readable identifier
- `severity`: P0/P1/P2/P3 priority
- `tags`: Categorization for filtering
- `tolerance`: Numerical comparison buffer

**Grammar design**: Modifiers can appear in any order after the condition (lines 52-55 of grammar.lark).

**Consistency principle**: All modifiers should work with all applicable assertions. The tolerance modifier was inadvertently restricted to `==` in implementation, despite grammar allowing it everywhere.

## Functions.py Deep Dive

### Tolerance Parameter Semantics

**Standard pattern** (all comparison functions):
```python
def is_<op>(a: float, b: float, tol: float = EPSILON) -> bool:
    """Compare a and b with tolerance tol."""
    return <expression using tol>
```

**Why `tol` is always absolute**: Relative tolerance (e.g., 1% of threshold) would require different semantics per operator. Absolute tolerance is simpler and more predictable.

**Comparison implementations**:
```python
is_gt(a, b, tol):  return a > b + tol   # "a exceeds b by more than tol"
is_geq(a, b, tol): return a > b - tol   # "a is at least b, minus tol buffer"
is_lt(a, b, tol):  return a < b - tol   # "a is below b by more than tol"
is_leq(a, b, tol): return a < b + tol   # "a is at most b, plus tol buffer"
is_eq(a, b, tol):  return abs(a-b) < tol  # "a and b differ by less than tol"
is_neq(a, b, tol): return abs(a-b) >= tol # "a and b differ by at least tol"
```

### Special Function: `is_between`

**Implementation** (lines 180-193):
```python
def is_between(a: float, lower: float, upper: float, tol: float = EPSILON) -> bool:
    return is_geq(a, lower, tol) and is_leq(a, upper, tol)
```

**Key insight**: Tolerance is applied independently to both bounds via `is_geq` and `is_leq`.

**Effect**: Range expands symmetrically:
- Without tolerance: `[lower, upper]`
- With tolerance: `[lower - tol, upper + tol]`

**Example**:
```python
is_between(89, 90, 110, tol=5)
# Equivalent to: is_geq(89, 90, 5) and is_leq(89, 110, 5)
# Which is: (89 > 85) and (89 < 115)
# Result: True
```

## Implementation Checklist Reference

**Reference**: AGENTS.md §implementation-checklist

After completing all phases:

1. **Run all tests**: `uv run pytest`
2. **Check coverage**: `uv run pytest --cov=src/dqx --cov-report=term-missing`
   - Must show 100% coverage for modified lines in `src/dqx/api.py`
3. **Run pre-commit**: `uv run pre-commit run --all-files`
   - All hooks must pass (format, lint, mypy)
4. **Manual verification**: Create test DQL file, execute with verbose output

**Quality gate**: Implementation is complete when all four steps pass without errors.

## Summary

**What's changing**: The `_apply_condition` method in `src/dqx/api.py` will be updated to pass `tolerance` to all comparison operators, not just `==`.

**What's staying the same**:
- Parser and grammar (already correct)
- AST structure (already has tolerance field)
- Comparison functions (already support tolerance)
- Python API (already exposes tolerance everywhere)

**Scope**: ~30 lines of code changes, ~200 lines of test code additions.

**Risk level**: Low - leveraging existing, tested functionality; backward compatible.

**Expected outcome**: DQL achieves parity with Python API for tolerance support.
