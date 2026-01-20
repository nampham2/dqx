# Tolerance Modifier Fix - Implementation Guide

## Overview

This guide provides step-by-step TDD implementation to fix the tolerance modifier discrepancy in DQL. Currently, tolerance only applies to `==` operator, but all comparison operators in the Python API support tolerance. We'll extend DQL to expose this existing functionality uniformly.

**Goal**: Enable tolerance on all DQL comparison operators: `>`, `>=`, `<`, `<=`, `==`, `!=`, `between`.

## Prerequisites

**Files to read before starting**:
- `src/dqx/api.py` (lines 1800-1850) - Current `_apply_condition` implementation
- `src/dqx/functions.py` (lines 13-193) - Tolerance semantics for each comparison
- `src/dqx/dql/ast.py` (lines 60-68) - Assertion AST structure with tolerance field
- `tests/dql/test_dql.py` (line 78-100) - Existing tolerance test pattern

**Related components**:
- **Parser** (`src/dqx/dql/parser.py`): Already parses tolerance correctly - no changes needed
- **Grammar** (`src/dqx/dql/grammar.lark`): Already supports tolerance on all assertions - no changes needed
- **Functions** (`src/dqx/functions.py`): All comparison functions support `tol` parameter - no changes needed

## Phase Breakdown

### Phase 1: Add Tolerance Support to Basic Comparison Operators

**Goal**: Enable tolerance for `>`, `>=`, `<`, `<=` operators in DQL assertions.

**Duration estimate**: 1 hour

**Files to modify**:
- `src/dqx/api.py` - Update `_apply_condition` method (lines 1804-1815)

**Tests to write**:
```python
# In tests/dql/test_dql.py or new test file

def test_tolerance_on_greater_than():
    """Test tolerance modifier on > operator."""
    ...

def test_tolerance_on_greater_or_equal():
    """Test tolerance modifier on >= operator."""
    ...

def test_tolerance_on_less_than():
    """Test tolerance modifier on < operator."""
    ...

def test_tolerance_on_less_or_equal():
    """Test tolerance modifier on <= operator."""
    ...

def test_basic_operators_without_tolerance_unchanged():
    """Verify operators without tolerance still use default EPSILON."""
    ...
```

**Implementation notes**:
- Follow the pattern already used for `==` operator (lines 1816-1821)
- For each operator, add conditional: `if assertion_ast.tolerance: ... else: ...`
- Pass `tol=assertion_ast.tolerance` to the respective `is_*` method
- Maintain backward compatibility - when tolerance is `None`, call without `tol` parameter

**Success criteria**:
- [ ] Tests pass for `>`, `>=`, `<`, `<=` with tolerance
- [ ] Tests pass for same operators without tolerance (default behavior)
- [ ] Coverage: 100% for modified lines
- [ ] Pre-commit hooks: passing

**Commit message**: `feat(dql): add tolerance support for basic comparison operators (>, >=, <, <=)`

---

### Phase 2: Add Tolerance Support to `!=` Operator

**Goal**: Enable tolerance for `!=` operator, ensuring inverse semantics of `==`.

**Duration estimate**: 30 minutes

**Files to modify**:
- `src/dqx/api.py` - Update `_apply_condition` for `!=` condition (lines 1822-1824)

**Tests to write**:
```python
def test_tolerance_on_not_equal():
    """Test tolerance modifier on != operator."""
    ...

def test_not_equal_inverse_of_equal_with_tolerance():
    """Verify != with tolerance is inverse of == with same tolerance."""
    ...

def test_not_equal_without_tolerance_unchanged():
    """Verify != without tolerance uses default EPSILON."""
    ...
```

**Implementation notes**:
- Follow same if/else pattern as other operators
- Ensure tolerance semantics match `functions.is_neq`: `abs(a - b) >= tol`
- Test edge case: value exactly at tolerance boundary

**Success criteria**:
- [ ] Tests pass for `!=` with tolerance
- [ ] Inverse relationship with `==` verified
- [ ] Coverage: 100% for modified lines
- [ ] Pre-commit hooks: passing

**Commit message**: `feat(dql): add tolerance support for not-equal operator (!=)`

---

### Phase 3: Add Tolerance Support to `between` Operator

**Goal**: Enable tolerance for `between` operator, applying same tolerance to both bounds.

**Duration estimate**: 1 hour

**Files to modify**:
- `src/dqx/api.py` - Update `_apply_condition` for `between` condition (lines 1825-1833)

**Tests to write**:
```python
def test_tolerance_on_between():
    """Test tolerance modifier on between operator."""
    ...

def test_between_tolerance_expands_range():
    """Verify tolerance widens acceptable range symmetrically."""
    ...

def test_between_with_negative_bounds_and_tolerance():
    """Test between with negative values and tolerance."""
    ...

def test_between_without_tolerance_unchanged():
    """Verify between without tolerance uses default EPSILON."""
    ...

def test_between_tolerance_at_boundaries():
    """Test edge cases where value is exactly at tolerance boundary."""
    ...
```

**Implementation notes**:
- Apply same tolerance to both `lower` and `upper` bounds
- Implementation: `ready.is_between(lower, upper, tol=assertion_ast.tolerance)`
- Semantic: Expands range from `[lower, upper]` to `[lower - tol, upper + tol]`
- Verify against `functions.is_between` implementation (line 180-193)

**Success criteria**:
- [ ] Tests pass for `between` with tolerance
- [ ] Range expansion verified with test data
- [ ] Edge cases (boundaries) handled correctly
- [ ] Coverage: 100% for modified lines
- [ ] Pre-commit hooks: passing

**Commit message**: `feat(dql): add tolerance support for between operator`

---

### Phase 4: End-to-End Integration Tests

**Goal**: Verify tolerance works correctly in real DQL suites with actual data.

**Duration estimate**: 1.5 hours

**Files to modify**:
- `tests/e2e/test_dql_verification_suite_e2e.py` - Add comprehensive e2e test

**Tests to write**:
```python
def test_tolerance_all_operators_e2e():
    """End-to-end test for tolerance on all comparison operators."""
    ...

def test_tolerance_with_real_data_edge_cases():
    """Test tolerance with realistic data scenarios."""
    ...

def test_tolerance_mixed_with_other_modifiers():
    """Test tolerance combined with tags, severity, annotations."""
    ...

def test_existing_tolerance_behavior_unchanged():
    """Regression test: existing == with tolerance still works."""
    ...
```

**Implementation notes**:
- Create a DQL suite exercising all operators with tolerance
- Use actual data with floating-point values near thresholds
- Verify pass/fail status matches expected tolerance semantics
- Include edge cases: tolerance causing pass, tolerance causing fail
- Test tolerance with tunables (static tolerance + dynamic threshold)

**Success criteria**:
- [ ] All e2e tests pass
- [ ] Real-world scenarios validated
- [ ] Existing tolerance tests still pass (regression check)
- [ ] Coverage: 100% for new test code
- [ ] Pre-commit hooks: passing

**Commit message**: `test(dql): add e2e tests for tolerance on all operators`

---

## Phase Dependencies

**Sequential execution required**:
1. Phase 1 must complete before Phase 2 (builds on same pattern)
2. Phase 2 must complete before Phase 3 (verifies pattern consistency)
3. Phases 1-3 must complete before Phase 4 (e2e tests require all operators)

**Rationale**: Each phase adds operator support incrementally, allowing independent verification of each operator's tolerance behavior.

## Detailed Test Scenarios

### Phase 1 Test Details

**Test: `test_tolerance_on_greater_than`**
```python
# DQL:
# assert average(price) > 100 tolerance 5
#
# Test data scenarios:
# - price_avg = 106 → PASS (106 > 105)
# - price_avg = 105 → FAIL (105 > 105 is false, needs strict >)
# - price_avg = 104 → FAIL (104 > 105)
```

**Test: `test_tolerance_on_greater_or_equal`**
```python
# DQL:
# assert average(price) >= 100 tolerance 5
#
# Test data scenarios:
# - price_avg = 96 → PASS (96 > 95)
# - price_avg = 95 → FAIL (95 > 95 is false)
# - price_avg = 94 → FAIL (94 > 95)
```

### Phase 2 Test Details

**Test: `test_not_equal_inverse_of_equal_with_tolerance`**
```python
# Verify that != is inverse of ==:
# If: assert x == 100 tolerance 1 → PASS (e.g., x=100.5)
# Then: assert x != 100 tolerance 1 → FAIL (same x=100.5)
#
# Test with same data, verify opposite results
```

### Phase 3 Test Details

**Test: `test_between_tolerance_expands_range`**
```python
# DQL:
# assert num_rows() between 90 and 110 tolerance 5
#
# Without tolerance: acceptable range [90, 110]
# With tolerance 5: acceptable range [85, 115]
#
# Test data scenarios:
# - num_rows = 86 → PASS (86 is in [85, 115])
# - num_rows = 85 → FAIL (boundary, needs > 85)
# - num_rows = 114 → PASS (114 is in [85, 115])
# - num_rows = 115 → FAIL (boundary, needs < 115)
```

### Phase 4 Test Details

**Test: `test_tolerance_all_operators_e2e`**
```python
# Create complete DQL suite:
"""
suite "Tolerance Test Suite" {
    check "Tolerance Check" on test_data {
        assert average(price) > 100
            tolerance 5
            name "Price greater than"

        assert null_count(col) <= 0.05
            tolerance 0.01
            name "Null rate"

        assert sum(amount) == 1000
            tolerance 10
            name "Total amount"

        assert stddev(value) != 0
            tolerance 0.01
            name "Non-zero variance"

        assert num_rows() between 90 and 110
            tolerance 5
            name "Row count range"
    }
}
"""
# Execute with real data, verify all assertions evaluate correctly
```

## Rollback Strategy

**If issues arise**:

1. **Phase 1-3 issues**: Revert commit for specific phase, fix tests, recommit.
2. **Integration issues in Phase 4**: All phases are independently committed, so revert only Phase 4 commit.
3. **Production issues**: Revert all commits in reverse order (Phase 4 → 3 → 2 → 1).

**Safe revert command**:
```bash
git revert <commit-hash>  # Revert specific phase commit
```

**No database migrations or schema changes**, so rollback is purely code-level.

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting to Handle `None` Tolerance

**Problem**: Calling `ready.is_gt(threshold, tol=None)` when tolerance is not specified.

**Solution**: Use if/else pattern to call method without `tol` parameter when `assertion_ast.tolerance is None`.

**Correct pattern**:
```python
if assertion_ast.tolerance:
    ready.is_gt(threshold, tol=assertion_ast.tolerance)
else:
    ready.is_gt(threshold)  # Uses default EPSILON
```

### Pitfall 2: Misunderstanding Tolerance Semantics

**Problem**: Assuming tolerance always makes comparisons "more lenient."

**Reality**: For `>` and `<`, tolerance makes comparisons *stricter*.

**Solution**: Refer to technical spec Decision 2 table. Write explicit test cases for each operator.

### Pitfall 3: Copy-Paste Errors Between Operators

**Problem**: Copying code for `>` but forgetting to change method call to `is_lt`.

**Solution**: After implementation, verify each operator branch calls the correct method. Use IDE find/replace carefully.

### Pitfall 4: Not Testing Edge Cases (Boundary Values)

**Problem**: Tests only check values clearly inside/outside tolerance range.

**Solution**: Add tests for exact boundary values (e.g., `threshold ± tolerance`).

## Verification Checklist

Before marking each phase complete:

- [ ] All new tests pass
- [ ] All existing tests still pass (regression check)
- [ ] Coverage report shows 100% for modified lines
- [ ] Pre-commit hooks pass (ruff format, ruff check, mypy)
- [ ] Manual verification: Create minimal DQL file, run with `pytest -s` to see output
- [ ] Code review self-check: Compare implementation to functions.py semantics

## Estimated Total Time

- Phase 1: 1 hour
- Phase 2: 30 minutes
- Phase 3: 1 hour
- Phase 4: 1.5 hours
- **Total**: ~4 hours

## Additional Notes

### Testing Philosophy

- **Write tests first** (TDD): For each operator, write failing test showing expected behavior, then implement.
- **Test both positive and negative cases**: Value passes with tolerance, value fails even with tolerance.
- **Test boundary conditions**: Exact threshold values, threshold ± tolerance.
- **Use realistic data**: Floating-point values, edge cases like negative numbers, zero.

### Code Review Focus Areas

1. **Consistency**: All operators follow same if/else pattern.
2. **Correctness**: Each `ready.is_*` call matches operator condition.
3. **Completeness**: All six comparison operators + between handled.
4. **Testing**: Each operator has dedicated test with multiple scenarios.

### Post-Implementation

After all phases complete:

1. Run full test suite: `uv run pytest`
2. Check coverage: `uv run pytest --cov=src/dqx --cov-report=term-missing`
3. Verify no regressions: `uv run pytest tests/dql/ tests/e2e/`
4. Manual smoke test: Create simple DQL file with all operators, execute and verify

**Example smoke test DQL**:
```dql
suite "Tolerance Smoke Test" {
    check "All Operators" on test {
        assert average(x) > 100 tolerance 5 name "GT"
        assert average(x) >= 100 tolerance 5 name "GEQ"
        assert average(x) < 200 tolerance 5 name "LT"
        assert average(x) <= 200 tolerance 5 name "LEQ"
        assert average(x) == 150 tolerance 5 name "EQ"
        assert average(x) != 0 tolerance 1 name "NEQ"
        assert average(x) between 90 and 210 tolerance 10 name "BETWEEN"
    }
}
```

Save as `smoke_test.dql`, create test with representative data, verify all assertions evaluate as expected.
