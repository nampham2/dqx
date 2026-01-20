# Tolerance Modifier Fix - Technical Specification

## Problem Statement

The DQL `tolerance` modifier is currently only applied to the `==` operator in the `_apply_condition` method (api.py:1816-1821). However, the Python API's `tol` parameter works with all comparison methods (`is_gt`, `is_geq`, `is_lt`, `is_leq`, `is_eq`, `is_neq`, `is_between`). This creates a semantic inconsistency where DQL users cannot leverage tolerance for non-equality comparisons, despite the underlying implementation fully supporting it.

**Current limitation**: Only this works:
```dql
assert average(price) == 100
    tolerance 0.05
```

**Expected behavior**: All operators should support tolerance:
```dql
assert average(price) > 100
    tolerance 5

assert null_count(col) <= 0.05
    tolerance 0.01

assert num_rows() between 90 and 110
    tolerance 5
```

## Architecture Decisions

### Decision 1: Uniform Tolerance Semantics Across All Operators

**Rationale**: The Python API already implements tolerance consistently across all comparison functions (see `src/dqx/functions.py`). The DQL layer should expose this existing functionality without semantic changes. This ensures DQL is a complete representation of Python API capabilities.

**Alternatives considered**:
- **Option A** (rejected): Only enable tolerance for equality-like operators (`==`, `between`). This creates confusion about when tolerance applies.
- **Option B** (rejected): Use different tolerance semantics for different operators. This violates principle of least surprise.
- **Option C** (chosen): Apply tolerance uniformly to all comparison operators using the existing functions.py semantics.

### Decision 2: Tolerance Application Semantics

**For each operator** (from functions.py:13-193):

| Operator | Function | Semantic with tolerance `tol` | Example |
|----------|----------|-------------------------------|---------|
| `>` | `is_gt(a, b, tol)` | `a > b + tol` | `101 > 100 tol=5` → `101 > 105` → FAIL |
| `>=` | `is_geq(a, b, tol)` | `a > b - tol` | `95 >= 100 tol=5` → `95 > 95` → PASS |
| `<` | `is_lt(a, b, tol)` | `a < b - tol` | `99 < 100 tol=5` → `99 < 95` → FAIL |
| `<=` | `is_leq(a, b, tol)` | `a < b + tol` | `105 <= 100 tol=5` → `105 < 105` → PASS |
| `==` | `is_eq(a, b, tol)` | `abs(a - b) < tol` | `100.5 == 100 tol=1` → `abs(0.5) < 1` → PASS |
| `!=` | `is_neq(a, b, tol)` | `abs(a - b) >= tol` | `100.5 != 100 tol=1` → `abs(0.5) >= 1` → FAIL |
| `between` | `is_between(a, lo, hi, tol)` | `is_geq(a, lo, tol) && is_leq(a, hi, tol)` | `89 between 90 and 110 tol=5` → `89 > 85 && 89 < 115` → PASS |

**Key insight**: Tolerance creates a "buffer zone" around thresholds:
- For `>` and `<`: makes the comparison *stricter* (must exceed threshold by `tol`)
- For `>=` and `<=`: makes the comparison *more lenient* (allows undershoot/overshoot by `tol`)
- For `==`: standard epsilon-based equality (symmetric buffer)
- For `!=`: inverse of `==` (must differ by at least `tol`)
- For `between`: applies tolerance to both bounds independently

**Rationale**: This matches IEEE 754 floating-point comparison best practices and is already battle-tested in the Python API.

### Decision 3: Implementation Approach

**Change location**: `src/dqx/api.py`, method `_apply_condition` (lines 1804-1833)

**Current structure**:
```python
if cond == ">":
    threshold = self._eval_simple_expr(assertion_ast.threshold, tunables)
    ready.is_gt(threshold)  # No tolerance parameter
elif cond == "==":
    threshold = self._eval_simple_expr(assertion_ast.threshold, tunables)
    if assertion_ast.tolerance:  # Only handles tolerance here
        ready.is_eq(threshold, tol=assertion_ast.tolerance)
    else:
        ready.is_eq(threshold)
```

**Target structure**:
```python
if cond == ">":
    threshold = self._eval_simple_expr(assertion_ast.threshold, tunables)
    if assertion_ast.tolerance:
        ready.is_gt(threshold, tol=assertion_ast.tolerance)
    else:
        ready.is_gt(threshold)
# ... same pattern for all operators
```

**Alternatives considered**:
- **Option A** (rejected): Extract tolerance handling to a helper function. Adds indirection for minimal code reuse (2 lines per operator).
- **Option B** (chosen): Apply consistent if/else pattern inline for each operator. Clear, explicit, easy to verify.

### Decision 4: Special Handling for `between`

**Challenge**: `between` has two thresholds (`lower`, `upper`), but only one `tolerance` value.

**Solution**: Apply the same tolerance to both bounds, as implemented in `functions.is_between`:
```python
def is_between(a: float, lower: float, upper: float, tol: float = EPSILON) -> bool:
    return is_geq(a, lower, tol) and is_leq(a, upper, tol)
```

**DQL Example**:
```dql
assert num_rows() between 90 and 110
    tolerance 5
# Equivalent to: is_geq(num_rows(), 90, 5) AND is_leq(num_rows(), 110, 5)
# Acceptable range: [85, 115]
```

**Rationale**: Single tolerance value is simpler and covers 99% of use cases. If users need asymmetric tolerance, they can use two separate assertions.

### Decision 5: `is positive` / `is negative` - No Tolerance

**Rationale**: These keywords check `a > EPSILON` and `a < -EPSILON` respectively (functions.py:152-177). They already have built-in tolerance (EPSILON). Adding user tolerance would be confusing:
- What does "is positive tolerance 10" mean? `a > 10`? Or `a > EPSILON - 10`?

**Decision**: Keep current behavior - ignore tolerance for `is positive` / `is negative`. These operators should not accept tolerance in DQL.

**Implementation**: No changes needed in the `is` condition branch (lines 1834-1843).

## API Design

### Modified Method Signature (No Changes)

The `_apply_condition` method signature remains unchanged:
```python
def _apply_condition(
    self,
    ready: AssertionReady,
    assertion_ast: Assertion,
    tunables: dict[str, TunableValue],
) -> None:
    """Apply the parsed condition to the assertion."""
```

### Internal Logic Changes

**Pattern for each comparison operator**:
```python
elif cond == "<operator>":
    threshold = self._eval_simple_expr(assertion_ast.threshold, tunables)
    if assertion_ast.tolerance:
        ready.is_<operator>(threshold, tol=assertion_ast.tolerance)
    else:
        ready.is_<operator>(threshold)
```

**Special case for `between`**:
```python
elif cond == "between":
    if assertion_ast.threshold_upper is None:  # pragma: no cover
        raise DQLError("Condition 'between' requires upper threshold", assertion_ast.loc)
    lower = self._eval_simple_expr(assertion_ast.threshold, tunables)
    upper = self._eval_simple_expr(assertion_ast.threshold_upper, tunables)
    if assertion_ast.tolerance:
        ready.is_between(lower, upper, tol=assertion_ast.tolerance)
    else:
        ready.is_between(lower, upper)
```

## Data Structures

**No changes required**. The AST already has `tolerance` field:
```python
@dataclass(frozen=True)
class Assertion:
    # ... other fields ...
    tolerance: float | None = None
```

Grammar already supports tolerance modifier on all assertions:
```lark
modifiers: name_mod
         | tolerance_mod
         | severity_mod
         | tags_mod

tolerance_mod: TOLERANCE_KW NUMBER
```

## Integration Points

### 1. Parser (`src/dqx/dql/parser.py`)
**No changes needed**. Already parses tolerance modifier for all assertions.

### 2. API (`src/dqx/api.py`)
**Changes required**: `_apply_condition` method (lines 1804-1833).

### 3. Functions (`src/dqx/functions.py`)
**No changes needed**. All comparison functions already support `tol` parameter.

### 4. Tests
**Changes required**: Add comprehensive test coverage in:
- `tests/dql/test_dql.py` - DQL parsing and execution
- `tests/e2e/test_dql_verification_suite_e2e.py` - End-to-end validation

## Performance Considerations

**Impact**: Negligible. Adding tolerance parameter to function calls has zero overhead when tolerance is `None` (default path) and trivial computation cost when present (one extra floating-point operation per comparison).

**Memory**: No additional memory allocation.

## Backward Compatibility

**100% backward compatible**:
- Existing DQL without tolerance: unchanged behavior (uses default EPSILON)
- Existing DQL with `tolerance` on `==`: unchanged behavior (same code path)
- New DQL with tolerance on other operators: new capability, no breaking changes

**Verification**: All existing tests must pass without modification.

## Edge Cases

### 1. Tolerance on `is positive` / `is negative`
**Behavior**: Ignored (no implementation change).
**User impact**: DQL will parse but tolerance has no effect.
**Documentation**: Should warn users that tolerance is not applicable to `is` conditions.

### 2. Negative Tolerance
**Current behavior**: Parser accepts any NUMBER token (including negatives).
**Desired behavior**: Negative tolerance is semantically meaningless but technically allowed by functions.py.
**Decision**: Accept negative tolerance (let functions.py handle it). If it causes issues, add validation in a future PR.

### 3. Tolerance Larger Than Threshold
**Example**: `assert average(price) > 10 tolerance 100`
**Behavior**: Comparison becomes `actual > 110`, which may never pass.
**Decision**: Accept as user error. DQL is expressive; validation is the user's responsibility.

### 4. Tolerance with Tunables
**Example**:
```dql
tunable MAX_PRICE = 100 bounds [50, 200]
assert average(price) < MAX_PRICE tolerance 5
```
**Behavior**: Tolerance is static, tunable threshold is dynamic.
**Implementation**: Works correctly - `ready.is_lt(tunable_value, tol=5)`.

## Non-Goals

1. **Relative tolerance**: Only absolute tolerance is supported (matches Python API).
2. **Per-bound tolerance for `between`**: Single tolerance applies to both bounds.
3. **Tolerance validation**: No restrictions on tolerance values (user responsibility).
4. **Tolerance on `is` conditions**: Not semantically meaningful; ignored.

## Testing Strategy

### Unit Tests (tests/dql/test_dql.py)
1. Parse DQL with tolerance on all operators
2. Verify AST contains correct tolerance values

### Integration Tests (tests/e2e/test_dql_verification_suite_e2e.py)
1. Execute assertions with tolerance on `>`, `>=`, `<`, `<=`, `!=`, `between`
2. Verify pass/fail behavior matches expected semantics
3. Test edge cases (tolerance causing pass, tolerance causing fail)

### Regression Tests
1. Verify existing tolerance tests for `==` still pass
2. Run full test suite - all existing tests must pass

## Documentation Updates

**No documentation changes required** as part of this fix. DQL language documentation (`docs/design/dql-language.md`) and API reference already describe tolerance modifier abstractly. If documentation update is desired, it should be a separate PR.

## Risk Assessment

**Low risk**:
- Small, localized change (one method, ~30 lines)
- Leverages existing, tested functions (functions.py)
- 100% backward compatible
- Easy to verify correctness (compare to functions.py semantics)

**Potential issues**:
- User confusion about tolerance semantics (mitigated by clear examples)
- Edge cases with extreme tolerance values (accepted as user error)
