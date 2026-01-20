# pct() Helper Function Context for Implementation

This document provides background context for implementing the `pct()` helper function.

## DQX Architecture Overview

### Relevant Components

#### `src/dqx/functions.py`

**Purpose**: Pure Python numeric utility functions for validation and comparison

**Key functions**:
- `is_leq(a, b, tol)` - Less than or equal comparison with tolerance
- `is_gt(a, b, tol)` - Greater than comparison with tolerance
- `is_between(a, lower, upper, tol)` - Range validation
- `is_zero(a, tol)` - Zero check with tolerance
- `within_tol(a, b, rel_tol, abs_tol)` - Tolerance-based equality

**How `pct()` relates**:
- Fits naturally alongside these numeric utilities
- Same pattern: simple, pure Python function
- No SymPy dependencies (unlike `coalesce()` which is a SymPy function)
- Returns plain Python types (`float`)

**Pattern to follow**:
```python
def is_leq(a: float, b: float, tol: float = EPSILON) -> bool:
    """Check if a <= b within tolerance."""
    return a < b + tol
```

**Our function**:
```python
def pct(value: float | int) -> float:
    """Convert percentage to decimal."""
    return float(value) / 100.0
```

#### `src/dqx/api.py` (Exports)

**Purpose**: User-facing API module that orchestrates data quality checks

**Key exports** (imported by users):
- `VerificationSuite` - Main suite class
- `Context` - Execution context for checks
- `check` - Decorator for check functions
- `MetricProvider` - Metric access (via `from dqx.api import MetricProvider`)

**How `pct()` relates**:
- Will be exported from this module: `from dqx.api import pct`
- Users import multiple items together: `from dqx.api import check, pct`
- Pattern: Import from `dqx.functions`, re-export from `dqx.api`

**Export pattern** (line ~16):
```python
from dqx import functions, setup_logger
from dqx.functions import pct  # Add this
```

#### `src/dqx/api.py` (AssertionReady class, lines 177-580)

**Purpose**: Provides assertion methods for validation

**Key methods** (our integration points):
- `is_leq(other: float | NumericTunable, tol: float)` - Line 273
- `is_gt(other: float | NumericTunable, tol: float)` - Line 256
- `is_lt(other: float | NumericTunable, tol: float)` - Line 317
- `is_geq(other: float | NumericTunable, tol: float)` - Line 227
- `is_eq(other: float | NumericTunable, tol: float)` - Line 334
- `is_neq(other: float | NumericTunable, tol: float)` - Line 361
- `is_between(lower: float | NumericTunable, upper: float | NumericTunable, tol: float)` - Line 389

**How `pct()` relates**:
- These methods accept `float` as first parameter
- `pct()` returns `float`, making it compatible
- Usage: `ready.is_leq(pct(5))` → `ready.is_leq(0.05)`
- The assertion method receives a plain float, not aware `pct()` was used

**Type compatibility verification**:
```python
# Assertion method signature
def is_leq(self, other: float | NumericTunable, tol: float = functions.EPSILON) -> None:
    # ...

# pct() return type
def pct(value: float | int) -> float:
    # ...

# Compatible: float is accepted by float | NumericTunable
ctx.assert_that(metric).where(name="Test").is_leq(pct(5))  # ✅ Works
```

#### `src/dqx/tunables.py` (TunableFloat class)

**Purpose**: Dynamic parameters for threshold tuning by RL agents

**Key attributes**:
- `value: float` - Current threshold value
- `bounds: tuple[float, float]` - Min/max allowed values

**How `pct()` relates**:
- Users can use `pct()` for initial values: `TunableFloat("T", value=pct(5))`
- Users can use `pct()` for bounds: `bounds=(pct(0), pct(10))`
- `pct()` evaluates immediately, tunable receives plain float

**Usage pattern**:
```python
from dqx.tunables import TunableFloat
from dqx.api import pct

THRESHOLD = TunableFloat(
    "NULL_THRESHOLD",
    value=pct(5),        # Evaluates to 0.05
    bounds=(pct(0), pct(20))  # Evaluates to (0.0, 0.2)
)
```

#### `src/dqx/dql/parser.py` (Lines 73-75)

**Purpose**: Parse DQL syntax, including percentage literals

**Key function**:
```python
def _parse_percent(s: str) -> float:
    """Parse a percentage like '5%' to decimal 0.05."""
    return float(s.rstrip("%")) / 100
```

**How `pct()` relates**:
- DQL's `5%` and Python's `pct(5)` both produce `0.05`
- Different entry points (parse time vs runtime), same result
- Inspiration for our implementation
- `pct()` is the Python API equivalent of DQL's `%` syntax

**Key difference**:
- DQL: `5%` → Parser converts to `0.05` → Passes to assertion
- Python: `pct(5)` → Runtime converts to `0.05` → Passes to assertion

#### `src/dqx/api.py` (_build_metric_namespace, lines 1673-1714)

**Purpose**: Build SymPy namespace for parsing metric expressions

**Contents**:
- Math functions: `abs`, `sqrt`, `log`, `exp`, `min`, `max`
- Metrics: `num_rows`, `null_count`, `average`, `sum`, `count_distinct`, etc.
- SQL functions: `coalesce`

**How `pct()` relates** (CRITICAL):
- `pct()` must NOT be added to this namespace
- This namespace is for metric expressions, not threshold helpers
- `pct()` is for immediate evaluation, not symbolic manipulation

**Verification**:
```python
# ❌ WRONG - Do NOT do this
namespace = _build_metric_namespace(...)
namespace['pct'] = pct  # NO! Don't add pct to namespace

# ✅ CORRECT - pct() stays outside namespace
# Users call pct() directly for immediate float result
threshold = pct(5)  # Evaluates to 0.05
ctx.assert_that(metric).where(name="Test").is_leq(threshold)
```

## Code Patterns to Follow

**IMPORTANT**: All patterns reference AGENTS.md standards.

### Pattern 1: Simple Utility Function

**When to use**: Pure functions with no state, simple logic

**Example from DQX** (2-3 lines):
```python
def is_zero(a: float, tol: float = EPSILON) -> bool:
    """Check if a is effectively zero."""
    return abs(a) < tol
```

**Reference**: See AGENTS.md §code-standards for function patterns

**Apply to pct()**:
```python
def pct(value: float | int) -> float:
    """Convert percentage to decimal."""
    return float(value) / 100.0
```

### Pattern 2: Type Hints (Strict Mode)

**Example**:
```python
def is_leq(a: float, b: float, tol: float = EPSILON) -> bool:
    # ...
```

**Reference**: AGENTS.md §type-hints (strict mode required)

**Apply to pct()**:
- Input: `float | int` (accept both)
- Output: `float` (always return float)
- Use `|` not `Union` (Python 3.11+ syntax)

### Pattern 3: Google-Style Docstrings

**Example**:
```python
def is_between(a: float, lower: float, upper: float, tol: float = EPSILON) -> bool:
    """Determine whether a value lies within the closed interval [lower, upper].

    Args:
        a: Value to test.
        lower: Lower bound of the interval.
        upper: Upper bound of the interval.
        tol: Tolerance applied to both bound comparisons.

    Returns:
        bool: True if lower <= a <= upper within tol, False otherwise.
    """
```

**Reference**: AGENTS.md §docstrings (Google style)

**Apply to pct()**:
- One-line summary
- Args section (describe `value` parameter)
- Returns section (explain plain float return)
- Examples section (multiple use cases)
- Note section (explain NOT SymPy)

### Pattern 4: No Validation in Simple Helpers

**DQX pattern**: Simple helpers don't validate, callers do

**Examples in DQX**:
- `is_gt(a, b)` - No validation that `a` and `b` are reasonable
- `is_between(a, lower, upper)` - Validates `lower <= upper`, but not value ranges

**Apply to pct()**:
- Don't validate `value >= 0` or `value <= 100`
- Accept any numeric value (negative, > 100, zero, decimals)
- Let callers decide what's valid for their use case

### Pattern 5: Plain Python, Not SymPy

**DQX has two types of functions**:

1. **Plain Python** (in `functions.py`): Return Python types
   ```python
   def is_leq(a: float, b: float, tol: float) -> bool:
       return a < b + tol  # Returns bool
   ```

2. **SymPy Functions** (also in `functions.py`): Return SymPy expressions
   ```python
   class Coalesce(sp.Function):
       # Returns sp.Expr
   ```

**Apply to pct()**:
- Plain Python (type 1)
- Return `float`, not `sp.Expr`
- No SymPy imports needed in function body

## Code Standards Reference

**All code must follow AGENTS.md standards**:

### Import Order

**Reference**: AGENTS.md §import-order

**For pct() implementation**: No imports needed in function body

**For tests**:
```python
from __future__ import annotations

import pytest  # Standard library
import sympy as sp  # Third-party

from dqx.api import pct, check, VerificationSuite  # Local
from dqx.tunables import TunableFloat
```

### Type Hints

**Reference**: AGENTS.md §type-hints (strict mode)

**Apply to pct()**:
```python
def pct(value: float | int) -> float:
    # Use | for union (not Union)
    # Complete type coverage required
```

### Docstrings

**Reference**: AGENTS.md §docstrings (Google style)

**Required sections for pct()**:
- Summary line
- Args
- Returns
- Examples (multiple)
- Note (explain NOT SymPy)

### Formatting

**Reference**: AGENTS.md §formatting

- Line length: 120 chars
- Indentation: 4 spaces
- Ruff will auto-format

### Testing

**Reference**: AGENTS.md §testing-standards

**For pct()**: Organize in test class

```python
class TestPct:
    """Tests for pct() helper function."""

    def test_basic_conversion(self) -> None:
        """Test pct(5) returns 0.05."""
        # ...
```

### Coverage

**Reference**: AGENTS.md §coverage-requirements (100%)

**For pct()**: Simple function, easy to reach 100%
- Test integer input
- Test float input
- Test edge cases (negative, zero, large)
- Test type verification

## Testing Patterns

**Reference**: AGENTS.md §testing-patterns

### Pattern: Test Class Organization

```python
class TestPct:
    """Tests for pct() helper function."""

    def test_basic_conversion(self) -> None:
        """Test typical percentage conversion."""
        from dqx.api import pct
        assert pct(5) == 0.05

    def test_edge_case_negative(self) -> None:
        """Test negative percentage."""
        from dqx.api import pct
        assert pct(-10) == -0.1
```

### Pattern: Integration Tests with VerificationSuite

```python
def test_pct_with_assertions() -> None:
    """Test pct() works with assertion methods."""
    from dqx.api import VerificationSuite, check, pct
    from dqx.orm.repositories import InMemoryMetricDB

    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp, ctx):
        ctx.assert_that(mp.null_rate("col")).where(name="Nulls").is_leq(pct(5))

    suite = VerificationSuite([test_check], db, "Test")
    assert len(list(suite.graph.checks())) == 1
```

**For pct()**: Use fixtures from `tests/fixtures/` for data sources if needed

## Common Pitfalls

### Pitfall 1: Creating SymPy Expression

**Problem**: Using SymPy operations in `pct()`

**Wrong approach**:
```python
import sympy as sp

def pct(value: float | int) -> float:
    return sp.sympify(value) / 100  # ❌ Returns sp.Expr, not float!
```

**Solution**: Use plain Python arithmetic
```python
def pct(value: float | int) -> float:
    return float(value) / 100.0  # ✅ Returns float
```

**Reference**: See section "Pattern 5: Plain Python, Not SymPy" above

### Pitfall 2: Adding to Metric Namespace

**Problem**: Adding `pct` to `_build_metric_namespace()`

**Wrong approach**:
```python
# In _build_metric_namespace() - DON'T DO THIS
namespace['pct'] = pct  # ❌ Makes pct() available in metric expressions
```

**Solution**: Keep `pct()` outside the namespace
- Only add to `functions.py` and export from `api.py`
- Do NOT register in namespace

**Why this matters**: `pct()` is for thresholds, not metric expressions

**Reference**: See "Integration Point: _build_metric_namespace" above

### Pitfall 3: Over-Validation

**Problem**: Restricting valid percentage ranges

**Wrong approach**:
```python
def pct(value: float | int) -> float:
    if value < 0 or value > 100:
        raise ValueError("Invalid percentage")  # ❌ Too restrictive
    return float(value) / 100.0
```

**Solution**: Accept all numeric values
```python
def pct(value: float | int) -> float:
    return float(value) / 100.0  # ✅ No validation
```

**Why**: Users need flexibility for:
- Growth rates over 100% (e.g., `pct(150)` → 1.5)
- Negative changes (e.g., `pct(-10)` → -0.1)
- Sub-percent precision (e.g., `pct(0.5)` → 0.005)

**Reference**: See AGENTS.md §error-handling for when to validate

### Pitfall 4: Forgetting Float Conversion

**Problem**: Not converting input to float

**Wrong approach**:
```python
def pct(value: float | int) -> float:
    return value / 100  # ❌ If value is int, result could be int in some contexts
```

**Solution**: Explicit float conversion
```python
def pct(value: float | int) -> float:
    return float(value) / 100.0  # ✅ Guarantees float return
```

### Pitfall 5: Type Checking Issues

**Problem**: Using `TYPE_CHECKING` when not needed

**For pct()**: No imports needed in function body, so no `TYPE_CHECKING` block needed

**When TYPE_CHECKING is needed** (not applicable here):
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dqx.orm.repositories import MetricDB  # Avoid circular imports
```

**Reference**: AGENTS.md §type-hints

## Related PRs and Issues

### DQL Percentage Support

**Context**: DQL already supports `5%` syntax

**Relevant code**: `src/dqx/dql/parser.py`, lines 73-75

**Implementation**:
```python
def _parse_percent(s: str) -> float:
    """Parse a percentage like '5%' to decimal 0.05."""
    return float(s.rstrip("%")) / 100
```

**Relation to pct()**: Same conversion, different trigger
- DQL: Parse time conversion
- Python API: Runtime conversion

### Tunables Integration

**Context**: Tunables accept float values

**Relevant code**: `src/dqx/tunables.py`, `TunableFloat` class

**How they work together**:
```python
# User code
THRESHOLD = TunableFloat("T", value=pct(5), bounds=(pct(0), pct(10)))

# What happens:
# 1. pct(5) evaluates to 0.05
# 2. pct(0) evaluates to 0.0
# 3. pct(10) evaluates to 0.1
# 4. TunableFloat receives: value=0.05, bounds=(0.0, 0.1)
```

**Key insight**: Tunable is unaware `pct()` was used - it just receives floats

## Usage Examples

### Example 1: Basic Threshold

```python
from dqx.api import VerificationSuite, check, pct

@check(name="Data Quality")
def check_quality(mp, ctx):
    # Null rate must be <= 5%
    ctx.assert_that(mp.null_rate("email")).where(
        name="Email nulls"
    ).is_leq(pct(5))
```

### Example 2: With Tunables

```python
from dqx.api import VerificationSuite, check, pct
from dqx.tunables import TunableFloat

# Define tunable threshold with readable percentage notation
MAX_NULL = TunableFloat(
    "MAX_NULL_RATE",
    value=pct(5),       # 5% null rate
    bounds=(pct(0), pct(20))  # 0-20% range
)

@check(name="Nulls")
def check_nulls(mp, ctx):
    null_rate = mp.null_count("col") / mp.num_rows()
    ctx.assert_that(null_rate).where(name="Null rate").is_leq(MAX_NULL)
```

### Example 3: Range Validation

```python
from dqx.api import check, pct

@check(name="Ranges")
def check_ranges(mp, ctx):
    # Completion rate should be 80-100%
    ctx.assert_that(mp.completeness("col")).where(
        name="Completion"
    ).is_between(pct(80), pct(100))
```

### Example 4: Sub-Percent Precision

```python
from dqx.api import check, pct

@check(name="Precision")
def check_precision(mp, ctx):
    # Error rate must be < 0.5% (very strict)
    ctx.assert_that(mp.error_rate("col")).where(
        name="Error rate"
    ).is_lt(pct(0.5))  # 0.5% = 0.005
```

### Example 5: Growth Rates (> 100%)

```python
from dqx.api import check, pct

@check(name="Growth")
def check_growth(mp, ctx):
    # Growth should be at least 150%
    ctx.assert_that(mp.growth_rate("sales")).where(
        name="Sales growth"
    ).is_geq(pct(150))  # 150% = 1.5
```

## Documentation

After implementation, ensure:

### Inline Docstring Complete

Function docstring in `functions.py` must have:
- [x] Summary line
- [x] Args section describing `value` parameter
- [x] Returns section (emphasize "plain float")
- [x] Examples section (5+ examples)
- [x] Note section (explain NOT SymPy, usage context)

**Reference**: AGENTS.md §docstrings

### Type Hints Complete

- [x] Input: `float | int`
- [x] Output: `float`
- [x] No defaults needed

**Reference**: AGENTS.md §type-hints

## Summary

**Key takeaways for implementation**:

1. **Location**: `src/dqx/functions.py` after `coalesce()`
2. **Implementation**: `return float(value) / 100.0`
3. **Export**: Add `from dqx.functions import pct` to `api.py`
4. **NOT in namespace**: Do not add to `_build_metric_namespace()`
5. **No validation**: Accept all numeric values
6. **Plain Python**: Return `float`, not `sp.Expr`
7. **Type hints**: `(float | int) -> float`
8. **Docstring**: Google style with examples
9. **Tests**: Unit + integration + namespace verification
10. **Coverage**: 100% (easy to achieve for simple function)

**Complexity**: Low - this is a simple utility function with clear requirements and minimal integration surface.
