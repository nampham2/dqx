# pct() Helper Function Technical Specification

## Problem Statement

Python API users must write decimal notation (`0.05` for 5%) while DQL users enjoy native percentage literals (`5%`). This creates a readability gap and increases error risk - it's easy to accidentally write `5` instead of `0.05`.

**Current state:**
```python
# Python API - error-prone decimal notation
ctx.assert_that(mp.null_rate("col")).where(name="Nulls").is_leq(0.05)  # Is this 5% or 0.05%?

# DQL - natural percentage notation
assert null_rate("col") <= 5% as "Nulls"
```

**Desired state:**
```python
from dqx.api import pct

# Python API - clear percentage notation
ctx.assert_that(mp.null_rate("col")).where(name="Nulls").is_leq(pct(5))  # Clearly 5%
```

This helper function bridges the notation gap, making Python API code as readable as DQL.

## Architecture Decisions

### Decision 1: Pure Python Function (NOT SymPy Symbol)

**Rationale**: `pct()` must evaluate immediately to a plain `float`, never becoming a SymPy symbol.

**Why this matters**:
- Assertion methods (`is_leq`, `is_gt`, etc.) accept `float | NumericTunable`
- If `pct()` returned a SymPy expression, it would break type contracts
- DQL's percentage parsing happens at parse time, converting `5%` → `0.05` before SymPy
- Consistency: both DQL and Python API should pass plain floats to assertion methods

**Implementation approach**:
```python
def pct(value: float | int) -> float:
    """Convert percentage to decimal. Returns plain float, NOT SymPy symbol."""
    return float(value) / 100.0
```

**What we avoid**:
```python
# ❌ DON'T DO THIS - would create SymPy expression
def pct(value: float | int) -> sp.Expr:
    return sp.sympify(value) / 100  # Wrong! Creates symbolic expression
```

### Decision 2: Implementation in `src/dqx/functions.py`

**Rationale**: `functions.py` contains pure Python utility functions for numeric operations.

**Existing similar functions in `functions.py`**:
- `is_geq`, `is_leq`, `is_gt`, `is_lt` - comparison utilities
- `is_zero`, `is_positive`, `is_negative` - numeric predicates
- `within_tol` - tolerance-based comparison

**Why `functions.py` over `api.py`**:
- `api.py` contains complex classes (`VerificationSuite`, `Context`, `AssertionReady`)
- `functions.py` is the established home for simple numeric helpers
- Separation of concerns: `functions.py` = utilities, `api.py` = orchestration

**Alternative considered**: Implement directly in `api.py`
- **Rejected**: Would clutter the already large `api.py` (2000+ lines)
- `functions.py` is cleaner and more maintainable

### Decision 3: Export via `dqx.api` Module

**Rationale**: Users should import from `dqx.api`, the primary user-facing module.

**Implementation**:
```python
# In src/dqx/api.py (top of file)
from dqx import functions
from dqx.functions import pct  # Add this import

# Later in api.py, around line 16 (with other imports)
# The import makes pct() available as dqx.api.pct
```

**Why not export from `dqx` root**:
- `dqx/__init__.py` currently only exports profiles, tunables, logging
- API functions (`check`, `VerificationSuite`, etc.) live in `dqx.api`
- Consistency: if users import `check` from `dqx.api`, they should import `pct` from there too

**User import pattern**:
```python
from dqx.api import VerificationSuite, check, pct  # All from same module
```

### Decision 4: No Validation or Bounds

**Rationale**: Maximum flexibility for all use cases.

**Accepted values**:
- ✅ `pct(5)` → `0.05` (typical percentage)
- ✅ `pct(0.5)` → `0.005` (sub-percent precision)
- ✅ `pct(150)` → `1.5` (over 100% is valid for growth rates)
- ✅ `pct(-10)` → `-0.1` (negative percentages for changes)
- ✅ `pct(0)` → `0.0` (zero is valid)

**Why no validation**:
- Percentages > 100 are common (e.g., 150% growth rate, 200% target achievement)
- Negative percentages are valid (e.g., -5% decline, -10% tolerance)
- Zero is meaningful (0% null rate)
- Validation would require context-specific rules, which don't belong in a helper

**Where validation happens**:
- `TunableFloat` has bounds validation (e.g., `bounds=(0.0, 1.0)`)
- User code validates the threshold makes sense for their metric
- `pct()` is just notation conversion, not business logic

### Decision 5: No Reverse Function

**Rationale**: YAGNI (You Aren't Gonna Need It) - keep it simple.

**Considered**: `pct_to_decimal(5)` and `decimal_to_pct(0.05)`
- **Rejected**: No use case identified for reverse conversion
- Users can write `value * 100` if needed
- Adding unused functions increases maintenance burden

## API Design

### Function Signature

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
        >>> ctx.assert_that(mp.null_rate("col")).where(name="Nulls").is_leq(pct(5))
        # Equivalent to: is_leq(0.05)

        # Usage with tunables
        >>> THRESHOLD = TunableFloat("THRESHOLD", value=pct(5), bounds=(pct(0), pct(10)))
        # Equivalent to: value=0.05, bounds=(0.0, 0.1)

    Note:
        Returns immediately evaluated float, NOT a SymPy expression.
        Do not use in metric definitions - only for thresholds.
    """
    return float(value) / 100.0
```

### Type Behavior

```python
# Runtime type verification
assert type(pct(5)) == float  # ✅ Plain float
assert pct(5) == 0.05  # ✅ Correct value

# NOT a SymPy type
import sympy as sp
assert not isinstance(pct(5), sp.Basic)  # ✅ Not SymPy
```

## Integration Points

### Integration 1: Assertion Methods (Primary Use Case)

**Files affected**: None (just usage)

**How it integrates**:
```python
# AssertionReady methods accept float | NumericTunable
def is_leq(self, other: float | NumericTunable, tol: float = functions.EPSILON) -> None:
    # ...

# pct() returns float, so it's compatible
ctx.assert_that(mp.null_rate("col")).where(name="Nulls").is_leq(pct(5))
# Equivalent to: is_leq(0.05)
```

**Verification**: All assertion methods (`is_leq`, `is_gt`, `is_lt`, `is_geq`, `is_eq`, `is_neq`) work with `pct()` output.

### Integration 2: Tunables

**Files affected**: None (just usage)

**How it integrates**:
```python
from dqx.tunables import TunableFloat

# Use pct() for readable tunable initialization
THRESHOLD = TunableFloat(
    "THRESHOLD",
    value=pct(5),      # 0.05
    bounds=(pct(0), pct(10))  # (0.0, 0.1)
)

# Tunable validates the float value as normal
# pct() just provides clearer notation for the initial values
```

### Integration 3: `is_between()` Method

**Files affected**: None (just usage)

**How it integrates**:
```python
ctx.assert_that(mp.null_rate("col")).where(name="Range").is_between(pct(5), pct(10))
# Equivalent to: is_between(0.05, 0.1)

# Can mix with tunables
LOWER = TunableFloat("LOWER", value=pct(5), bounds=(pct(0), pct(50)))
UPPER = TunableFloat("UPPER", value=pct(10), bounds=(pct(0), pct(50)))
ctx.assert_that(mp.null_rate("col")).where(name="Range").is_between(LOWER, UPPER)
```

## Performance Considerations

**Execution time**: Negligible (single division operation)
- `pct(5)` compiles to one float division: `5.0 / 100.0`
- No SymPy overhead (not creating symbolic expressions)
- Pure Python, no external calls

**Memory impact**: Zero
- No state, no caching, no allocations beyond the returned float
- Function is stateless and side-effect free

**Build time impact**: None
- Immediate evaluation at graph-build time
- DQL's `5%` also evaluates at parse time (equivalent performance)

**Comparison with DQL**:
```python
# Python API with pct()
pct(5)  # One division at runtime

# DQL parser
_parse_percent("5%")  # One division at parse time: float("5") / 100

# Equivalent performance, different timing
```

## Non-Goals

Explicitly out of scope:

1. **NOT a SymPy function**: Will not be registered in metric namespace
   - Must NOT appear in `_build_metric_namespace()` (api.py lines 1673-1714)
   - Cannot be used in metric expressions: `mp.custom_sql("SELECT pct(5)")` ❌

2. **NOT for DQL**: DQL already has native `%` syntax
   - Do not modify DQL parser or grammar
   - `pct()` is Python-only

3. **NOT for metric calculations**: Only for thresholds
   - ❌ Wrong: `mp.null_count("col") * pct(100)` (doesn't make sense)
   - ✅ Right: `.is_leq(pct(5))` (threshold)

4. **NOT for automatic inference**: User explicitly calls `pct()`
   - No magic: won't auto-detect `5` should be `pct(5)`
   - User chooses: `is_leq(5)` vs `is_leq(pct(5))`

5. **NOT for display/formatting**: Only for input conversion
   - Won't format `0.05` as "5%" in output
   - Display formatting is a separate concern (if needed later)

## Verification Requirements

### Verification 1: Not in SymPy Namespace

**Critical**: Ensure `pct()` is NOT added to `_build_metric_namespace()`.

**Check location**: `src/dqx/api.py`, lines 1673-1714

**Current namespace includes**:
- Math functions: `abs`, `sqrt`, `log`, `exp`, `min`, `max`
- Metrics: `num_rows`, `null_count`, `average`, `sum`, etc.
- SQL functions: `coalesce`

**Verification test**:
```python
def test_pct_not_in_metric_namespace():
    """Verify pct() is NOT available in metric expressions."""
    from dqx.api import VerificationSuite

    @check(name="Test")
    def test_check(mp, ctx):
        # This should work - pct() used for threshold
        ctx.assert_that(mp.null_rate("col")).where(name="Test").is_leq(pct(5))

    # Build suite - should succeed
    suite = VerificationSuite([test_check], db, "Test")

    # Verify pct is NOT in the namespace used for metric parsing
    # (Implementation detail: check namespace built in _build_metric_namespace)
```

### Verification 2: Type Safety

**Test that return type is plain float**:
```python
def test_pct_returns_float():
    """Verify pct() returns plain float, not SymPy type."""
    result = pct(5)

    assert type(result) == float  # Exact type check
    assert result == 0.05
    assert not isinstance(result, sp.Basic)  # Not SymPy
```

### Verification 3: Integration with Assertions

**Test with all assertion methods**:
```python
def test_pct_with_all_assertions():
    """Verify pct() works with all assertion methods."""
    # Should not raise any errors during graph building
    @check(name="Test")
    def test_check(mp, ctx):
        metric = mp.null_rate("col")
        ctx.assert_that(metric).where(name="1").is_leq(pct(5))
        ctx.assert_that(metric).where(name="2").is_lt(pct(5))
        ctx.assert_that(metric).where(name="3").is_geq(pct(1))
        ctx.assert_that(metric).where(name="4").is_gt(pct(1))
        ctx.assert_that(metric).where(name="5").is_eq(pct(0))
        ctx.assert_that(metric).where(name="6").is_neq(pct(100))
        ctx.assert_that(metric).where(name="7").is_between(pct(0), pct(10))

    suite = VerificationSuite([test_check], db, "Test")
    # Graph builds successfully = pct() returns compatible float
```

## Migration Path

**100% backward compatible** - no migration needed.

**Existing code continues working**:
```python
# Old style (still works)
ctx.assert_that(mp.null_rate("col")).where(name="Nulls").is_leq(0.05)

# New style (optional upgrade)
ctx.assert_that(mp.null_rate("col")).where(name="Nulls").is_leq(pct(5))
```

**Users can adopt incrementally**:
- Start using `pct()` for new code
- Leave existing decimal literals as-is
- No breaking changes, no deprecation period needed

## Documentation Requirements

### API Reference

Add entry to `docs/api-reference.md`:

```markdown
### `pct(value: float | int) -> float`

Convert percentage notation to decimal.

**Parameters:**
- `value`: Percentage value (e.g., 5 for 5%, 0.5 for 0.5%)

**Returns:**
- `float`: Decimal equivalent (e.g., pct(5) = 0.05)

**Examples:**
```python
from dqx.api import pct

# Basic conversion
pct(5)     # 0.05 (5%)
pct(0.5)   # 0.005 (0.5%)
pct(150)   # 1.5 (150%)

# With assertions
ctx.assert_that(mp.null_rate("col")).where(name="Nulls").is_leq(pct(5))

# With tunables
THRESHOLD = TunableFloat("THRESHOLD", value=pct(5), bounds=(pct(0), pct(10)))
```
```

### Inline Docstring

Already included in function signature above (Google style).

### Usage Examples

Add to example suite files or tutorials showing `pct()` usage patterns.

## Related Files

### Files to Modify

1. `src/dqx/functions.py` - Add `pct()` function
2. `src/dqx/api.py` - Import and export `pct` (add to imports at top)

### Files to Reference (No Changes)

1. `src/dqx/api.py` - Lines 220-472 (assertion methods that will use `pct()`)
2. `src/dqx/tunables.py` - `TunableFloat` class (users will use `pct()` with it)
3. `src/dqx/dql/parser.py` - Lines 73-75 (`_parse_percent()` - DQL equivalent)

### Critical Verification Files

1. `src/dqx/api.py` - Lines 1673-1714 (`_build_metric_namespace()` - must NOT include `pct`)

## Open Questions

None - requirements are clear and complete.

## Success Metrics

1. **Type safety**: `type(pct(5)) == float` ✓
2. **Not symbolic**: `not isinstance(pct(5), sp.Basic)` ✓
3. **Available via API**: `from dqx.api import pct` ✓
4. **Not in namespace**: `pct` not in `_build_metric_namespace()` result ✓
5. **Works with assertions**: All assertion methods accept `pct()` output ✓
6. **Works with tunables**: `TunableFloat(value=pct(5))` succeeds ✓
7. **100% test coverage**: All branches and edge cases covered ✓
8. **Documentation complete**: Docstring + API reference + examples ✓
