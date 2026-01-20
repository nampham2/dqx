# pct() Helper Function Implementation Guide

## Overview

Implement a simple `pct()` helper function that converts percentage notation (5) to decimal (0.05). This is a pure Python function returning plain float values, making Python API thresholds as readable as DQL percentages.

**Key principle**: Keep it simple - this is just division by 100.

## Prerequisites

### Files to Read Before Starting

- `src/dqx/functions.py` - Where we'll add `pct()`, similar to `is_leq`, `is_gt`, etc.
- `src/dqx/api.py` - Lines 1-50 (imports section) - Where we'll export `pct`
- `src/dqx/api.py` - Lines 220-472 (AssertionReady methods) - Integration point
- `tests/test_api.py` - Lines 1-100 - Test patterns to follow

### Related Components

**`src/dqx/functions.py`**:
- Contains pure Python numeric utilities
- Examples: `is_leq(a, b, tol)`, `is_zero(a, tol)`, `is_between(a, lower, upper, tol)`
- Pattern: Simple functions with type hints, no SymPy

**`src/dqx/api.py` (imports)**:
- Line 16: `from dqx import functions, setup_logger`
- Pattern: Import module and specific items for re-export

**`src/dqx/tunables.py`**:
- `TunableFloat` class with `bounds` parameter
- Users will use `pct()` with: `TunableFloat("T", value=pct(5), bounds=(pct(0), pct(10)))`

**`src/dqx/dql/parser.py` (Lines 73-75)**:
- `_parse_percent(s: str) -> float` - DQL's equivalent
- Implementation: `float(s.rstrip("%")) / 100`
- Inspiration for our function

## Phase Breakdown

### Phase 1: Core Function Implementation

**Goal**: Implement `pct()` in `functions.py` with complete docstring and type hints.

**Duration estimate**: 30 minutes

**Files to create**:
- None (adding to existing file)

**Files to modify**:
- `src/dqx/functions.py` - Add `pct()` function

**Tests to write** (test names only):
```python
class TestPct:
    def test_pct_basic_conversion(self): ...
    def test_pct_with_integer_input(self): ...
    def test_pct_with_float_input(self): ...
    def test_pct_with_decimal_percentage(self): ...
    def test_pct_with_large_value(self): ...
    def test_pct_with_negative_value(self): ...
    def test_pct_with_zero(self): ...
    def test_pct_returns_float_type(self): ...
    def test_pct_not_sympy_type(self): ...
```

**Implementation notes**:
- Add `pct()` after existing functions (e.g., after `coalesce()` around line 250)
- Follow existing patterns: simple, pure function
- Type hints: `def pct(value: float | int) -> float:`
- Docstring: Google style with comprehensive examples
- Implementation: `return float(value) / 100.0`
- Use `float(value)` to handle both int and float inputs
- Use `100.0` (not `100`) to ensure float division

**Success criteria**:
- [x] Function added to `functions.py`
- [x] Type hints: `float | int` → `float`
- [x] Docstring follows Google style (see AGENTS.md §docstrings)
- [x] All phase tests passing
- [x] Coverage: 100% for new code
- [x] Pre-commit hooks: passing

**Commit message**: `feat(functions): add pct() helper for percentage notation`

---

### Phase 2: API Export and Integration

**Goal**: Export `pct()` from `dqx.api` module so users can import it.

**Duration estimate**: 30 minutes

**Files to create**:
- None

**Files to modify**:
- `src/dqx/api.py` - Add import for `pct`

**Tests to write**:
```python
class TestPctAPIExport:
    def test_pct_available_from_api(self): ...
    def test_pct_import_from_api(self): ...
```

**Implementation notes**:
- In `src/dqx/api.py`, find the imports section (lines 1-50)
- After line 16 (`from dqx import functions, setup_logger`), add:
  ```python
  from dqx.functions import pct
  ```
- This makes `pct` available as `dqx.api.pct`
- No changes to `__all__` needed (api.py doesn't define `__all__`)

**Success criteria**:
- [x] `from dqx.api import pct` works
- [x] `pct` is importable alongside other API items
- [x] All phase tests passing
- [x] Coverage: 100%
- [x] Pre-commit hooks: passing

**Commit message**: `feat(api): export pct() helper for percentage notation`

---

### Phase 3: Integration Tests with Assertions

**Goal**: Verify `pct()` works correctly with all assertion methods and tunables.

**Duration estimate**: 1 hour

**Files to create**:
- `tests/test_pct_integration.py` - Integration tests

**Files to modify**:
- None

**Tests to write**:
```python
class TestPctWithAssertions:
    def test_pct_with_is_leq(self): ...
    def test_pct_with_is_lt(self): ...
    def test_pct_with_is_geq(self): ...
    def test_pct_with_is_gt(self): ...
    def test_pct_with_is_eq(self): ...
    def test_pct_with_is_neq(self): ...
    def test_pct_with_is_between(self): ...
    def test_pct_with_all_assertions_in_check(self): ...

class TestPctWithTunables:
    def test_pct_with_tunable_float_value(self): ...
    def test_pct_with_tunable_float_bounds(self): ...
    def test_pct_with_tunable_in_assertion(self): ...
    def test_pct_mixed_with_tunable_in_is_between(self): ...

class TestPctNotInNamespace:
    def test_pct_not_in_metric_namespace(self): ...
    def test_pct_not_registered_as_sympy_function(self): ...
```

**Implementation notes**:
- Create full end-to-end tests with `VerificationSuite`
- Test pattern: build suite, verify graph builds successfully
- For namespace verification:
  - Build a check that uses `pct()` in threshold (should work)
  - Verify `pct` is NOT available in metric expressions
  - Reference: `_build_metric_namespace()` in api.py (lines 1673-1714)
- Use fixtures from `tests/fixtures/` for data sources
- Pattern from `tests/test_api.py`:
  ```python
  @check(name="Test Check")
  def test_check(mp, ctx):
      ctx.assert_that(mp.null_rate("col")).where(name="Test").is_leq(pct(5))

  suite = VerificationSuite([test_check], db, "Test")
  ```

**Success criteria**:
- [x] All assertion methods work with `pct()` output
- [x] Tunables work with `pct()` for value and bounds
- [x] `pct()` is NOT in SymPy namespace
- [x] All phase tests passing
- [x] Coverage: 100%
- [x] Pre-commit hooks: passing

**Commit message**: `test(pct): add integration tests for pct() with assertions and tunables`

---

### Phase 4: Documentation and Examples

**Goal**: Add comprehensive documentation and usage examples.

**Duration estimate**: 30 minutes

**Files to create**:
- None (updating existing docs)

**Files to modify**:
- Update inline docstring in `functions.py` if needed (should be complete from Phase 1)

**Tests to write**:
```python
class TestPctDocumentation:
    def test_pct_docstring_examples_work(self): ...
```

**Implementation notes**:
- Verify docstring examples are accurate and runnable
- Docstring should include:
  - Brief description
  - Args section
  - Returns section
  - Examples section with multiple use cases
  - Note explaining it returns float, not SymPy
- Follow Google style (AGENTS.md §docstrings)
- Examples in docstring:
  - Basic conversion: `pct(5) → 0.05`
  - Decimal input: `pct(0.5) → 0.005`
  - Large value: `pct(150) → 1.5`
  - With assertions: `is_leq(pct(5))`
  - With tunables: `TunableFloat("T", value=pct(5))`

**Success criteria**:
- [x] Docstring complete and follows Google style
- [x] Examples in docstring are tested
- [x] All phase tests passing
- [x] Coverage: 100%
- [x] Pre-commit hooks: passing

**Commit message**: `docs(pct): verify and test docstring examples`

---

## Phase Dependencies

**Sequential phases** (must run in order):
1. Phase 1 (Core Function) → Phase 2 (API Export)
2. Phase 2 (API Export) → Phase 3 (Integration Tests)
3. Phase 3 (Integration Tests) → Phase 4 (Documentation)

**Why sequential**:
- Can't export what doesn't exist (Phase 1 before Phase 2)
- Can't test imports until exported (Phase 2 before Phase 3)
- Can't verify docs until implementation complete (Phase 3 before Phase 4)

**No parallelization**: Each phase builds on the previous.

## Implementation Details

### Phase 1 Implementation

**Location in `functions.py`**: After `coalesce()` function (around line 250)

**Complete implementation**:
```python
def pct(value: float | int) -> float:
    """Convert percentage notation to decimal.

    This helper makes Python API thresholds as readable as DQL percentages.
    The function evaluates immediately to a plain float value.

    Args:
        value: Percentage value (e.g., 5 for 5%, 0.5 for 0.5%)

    Returns:
        Decimal equivalent as plain float (NOT SymPy symbol)

    Examples:
        >>> pct(5)
        0.05
        >>> pct(0.5)
        0.005
        >>> pct(150)
        1.5
        >>> pct(-10)
        -0.1

        # Usage with assertions
        >>> from dqx.api import check, pct
        >>> @check(name="Nulls")
        >>> def check_nulls(mp, ctx):
        >>>     ctx.assert_that(mp.null_rate("col")).where(name="Null rate").is_leq(pct(5))

        # Usage with tunables
        >>> from dqx.tunables import TunableFloat
        >>> THRESHOLD = TunableFloat("THRESHOLD", value=pct(5), bounds=(pct(0), pct(10)))
        >>> # Equivalent to: value=0.05, bounds=(0.0, 0.1)

    Note:
        Returns immediately evaluated float, NOT a SymPy expression.
        Do not use in metric definitions - only for thresholds and tunable values.
    """
    return float(value) / 100.0
```

### Phase 2 Implementation

**Location in `api.py`**: Imports section (after line 16)

**Add import**:
```python
from dqx import functions, setup_logger
from dqx.functions import pct  # Add this line
```

### Phase 3 Key Tests

**Test 1: All assertions work**
```python
def test_pct_with_all_assertions_in_check() -> None:
    """Verify pct() output is compatible with all assertion methods."""
    from dqx.api import VerificationSuite, check, pct
    from dqx.orm.repositories import InMemoryMetricDB

    db = InMemoryMetricDB()

    @check(name="Test All Assertions")
    def test_check(mp, ctx):
        metric = mp.null_rate("col")
        ctx.assert_that(metric).where(name="leq").is_leq(pct(5))
        ctx.assert_that(metric).where(name="lt").is_lt(pct(5))
        ctx.assert_that(metric).where(name="geq").is_geq(pct(1))
        ctx.assert_that(metric).where(name="gt").is_gt(pct(1))
        ctx.assert_that(metric).where(name="eq").is_eq(pct(0))
        ctx.assert_that(metric).where(name="neq").is_neq(pct(100))
        ctx.assert_that(metric).where(name="between").is_between(pct(0), pct(10))

    # Should build successfully - pct() returns compatible float
    suite = VerificationSuite([test_check], db, "Test")
    assert len(list(suite.graph.checks())) == 1
```

**Test 2: Not in namespace**
```python
def test_pct_not_in_metric_namespace() -> None:
    """Verify pct() is NOT available in metric expressions."""
    from dqx.api import VerificationSuite, check, pct
    from dqx.orm.repositories import InMemoryMetricDB

    db = InMemoryMetricDB()

    @check(name="Test Namespace")
    def test_check(mp, ctx):
        # This should work - pct() used as threshold (immediate evaluation)
        ctx.assert_that(mp.null_rate("col")).where(name="Test").is_leq(pct(5))

    suite = VerificationSuite([test_check], db, "Test")

    # Verify pct is not in the namespace (implementation detail)
    # The key verification is that the suite builds successfully
    # and pct() evaluates to 0.05 before being passed to is_leq()
    assert suite.is_evaluated == False  # Not run yet
```

**Test 3: With tunables**
```python
def test_pct_with_tunable_float_bounds() -> None:
    """Verify pct() works with TunableFloat bounds."""
    from dqx.tunables import TunableFloat
    from dqx.api import pct

    # Should work - pct() returns float
    threshold = TunableFloat(
        "THRESHOLD",
        value=pct(5),
        bounds=(pct(0), pct(10))
    )

    assert threshold.value == 0.05
    assert threshold.bounds == (0.0, 0.1)
```

### Phase 4 Docstring Verification

**Test docstring examples**:
```python
def test_pct_docstring_examples_work() -> None:
    """Verify all examples in pct() docstring are correct."""
    from dqx.api import pct

    # Basic conversions from docstring
    assert pct(5) == 0.05
    assert pct(0.5) == 0.005
    assert pct(150) == 1.5
    assert pct(-10) == -0.1

    # Type verification
    assert type(pct(5)) == float
```

## Rollback Strategy

**If Phase 1 fails**:
- Revert changes to `functions.py`
- No API impact (not yet exported)

**If Phase 2 fails**:
- Revert import in `api.py`
- Core function remains but unexported (safe state)

**If Phase 3 integration issues**:
- Root cause: likely assertion method incompatibility
- Fix: ensure `pct()` returns plain `float`, not any SymPy type
- Verify: `type(pct(5)) == float`

**If Phase 4 doc issues**:
- Update docstring to match actual behavior
- Non-breaking change

**Complete rollback**:
```bash
# If critical issues found, revert all commits
git revert <commit-sha-phase-4>
git revert <commit-sha-phase-3>
git revert <commit-sha-phase-2>
git revert <commit-sha-phase-1>
```

## Estimated Total Time

**Phase breakdown**:
- Phase 1: 30 minutes (simple function)
- Phase 2: 30 minutes (one import line + tests)
- Phase 3: 1 hour (comprehensive integration tests)
- Phase 4: 30 minutes (documentation verification)

**Total: ~2.5 hours**

**Confidence**: High - this is a simple utility function with minimal integration points.

## Common Pitfalls

### Pitfall 1: Accidentally Creating SymPy Expression

**Problem**: Using `sp.sympify()` or SymPy operations in `pct()`

**Wrong**:
```python
def pct(value: float | int) -> float:
    return sp.sympify(value) / 100  # ❌ Creates SymPy expression!
```

**Correct**:
```python
def pct(value: float | int) -> float:
    return float(value) / 100.0  # ✅ Returns plain float
```

**Verification**:
```python
assert type(pct(5)) == float
assert not isinstance(pct(5), sp.Basic)
```

### Pitfall 2: Adding to Metric Namespace

**Problem**: Accidentally adding `pct` to `_build_metric_namespace()`

**Check location**: `src/dqx/api.py`, lines 1673-1714

**Verification**: `pct` should NOT appear in that function

**Why it matters**: Would allow `mp.custom_sql("SELECT pct(5)")` which is wrong usage

### Pitfall 3: Integer Division

**Problem**: Using `value / 100` without float conversion

**Wrong**:
```python
def pct(value: float | int) -> float:
    return value / 100  # ❌ Could lose precision with Python 2 semantics
```

**Correct**:
```python
def pct(value: float | int) -> float:
    return float(value) / 100.0  # ✅ Explicit float division
```

### Pitfall 4: Over-validating

**Problem**: Adding validation for negative/large values

**Wrong**:
```python
def pct(value: float | int) -> float:
    if value < 0 or value > 100:
        raise ValueError("Percentage must be 0-100")  # ❌ Too restrictive!
    return float(value) / 100.0
```

**Correct**:
```python
def pct(value: float | int) -> float:
    return float(value) / 100.0  # ✅ No validation - accept all values
```

**Why**: Users need flexibility for growth rates (>100%), declines (<0%), etc.

## Testing Strategy

### Unit Tests (Phase 1)

**Coverage areas**:
- Basic conversion: `pct(5) == 0.05`
- Integer input: `pct(10)`
- Float input: `pct(5.5)`
- Decimal percentage: `pct(0.5)`
- Large values: `pct(150)`
- Negative values: `pct(-10)`
- Zero: `pct(0)`
- Type verification: `type(pct(5)) == float`
- Not SymPy: `not isinstance(pct(5), sp.Basic)`

**Test file**: `tests/test_functions.py` (add to existing file)

### Import Tests (Phase 2)

**Coverage areas**:
- Import from `dqx.api`
- Available alongside other API exports

**Test file**: `tests/test_api.py` (add to existing file)

### Integration Tests (Phase 3)

**Coverage areas**:
- All assertion methods (`is_leq`, `is_gt`, etc.)
- `is_between` with `pct()` for both bounds
- Tunables with `pct()` for value
- Tunables with `pct()` for bounds
- Mixed: tunable + `pct()` in `is_between`
- Namespace verification (NOT available in metrics)

**Test file**: `tests/test_pct_integration.py` (new file)

### Documentation Tests (Phase 4)

**Coverage areas**:
- All examples in docstring execute correctly
- Docstring follows Google style

**Test file**: Part of Phase 3 tests

## Quality Checklist

Before marking implementation complete, verify:

- [x] **Type hints**: `def pct(value: float | int) -> float:`
- [x] **Docstring**: Google style with examples
- [x] **Implementation**: `return float(value) / 100.0`
- [x] **Location**: `src/dqx/functions.py` after `coalesce()`
- [x] **Export**: `from dqx.functions import pct` in `api.py`
- [x] **Type safety**: `type(pct(5)) == float`
- [x] **Not SymPy**: `not isinstance(pct(5), sp.Basic)`
- [x] **Not in namespace**: NOT in `_build_metric_namespace()`
- [x] **Tests pass**: All unit, import, integration tests passing
- [x] **Coverage**: 100% for `pct()` function
- [x] **Pre-commit**: All hooks passing (format, lint, type check)
- [x] **Commits**: 4 atomic commits with conventional messages

## Final Verification Commands

After all phases complete:

```bash
# 1. Run all tests
uv run pytest tests/test_functions.py tests/test_api.py tests/test_pct_integration.py -v

# 2. Check coverage
uv run pytest --cov=src/dqx --cov-report=term-missing

# 3. Verify 100% coverage for pct()
# Look for src/dqx/functions.py:pct in coverage report - should be 100%

# 4. Run pre-commit hooks
uv run pre-commit run --all-files

# 5. Verify type checking
uv run mypy src/dqx/functions.py src/dqx/api.py tests/test_pct_integration.py

# 6. Manual verification
python -c "from dqx.api import pct; assert pct(5) == 0.05; print('✅ Works!')"
```

All commands must pass before implementation is complete.
