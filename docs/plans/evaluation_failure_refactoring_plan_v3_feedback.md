# Feedback on EvaluationFailure Refactoring Plan v3

## Executive Summary

The plan to refactor evaluation failures from `dict[SymbolicMetric | sp.Expr, str]` to structured dataclasses is a good improvement. However, there are several critical implementation issues that need to be addressed before proceeding.

## Critical Issues

### 1. Existing Bug in Current Implementation

There is a type mismatch bug in the current `evaluator.py` that must be fixed:

```python
# In _gather method:
failures: dict[SymbolicMetric, str] = {}
# ...
case Failure(err):
    failures[sym] = err  # BUG: sym is sp.Symbol, not SymbolicMetric!
```

The type annotation declares `failures` as mapping `SymbolicMetric` to `str`, but the code uses `sp.Symbol` as the key. This will cause runtime errors and should be addressed as part of or before the refactoring.

### 2. Logic Error in Task 3

The error handling in the proposed `_gather` implementation has a flaw:

```python
if sym not in self.metrics:
    sm = self.metric_for_symbol(sym)  # This could fail!
    raise DQXError(f"Symbol {sm.name} not found in collected metrics.")
```

**Problem**: If the symbol isn't in `self.metrics`, calling `self.metric_for_symbol(sym)` might also fail since it searches through `self.provider.symbolic_metrics`. This could raise an exception before the intended error message.

**Suggested Fix**:
```python
if sym not in self.metrics:
    try:
        sm = self.metric_for_symbol(sym)
        raise DQXError(f"Symbol {sm.name} not found in collected metrics.")
    except DQXError:
        raise DQXError(f"Symbol {sym} not found in provider.")
```

### 3. Inefficiency in Task 4

The proposed implementation rebuilds symbol information in the `evaluate()` method even though it was already built in `_gather()`:

```python
# In evaluate() for NaN/infinity handling:
symbol_infos = []
for sym in expr.free_symbols:
    sm = self.metric_for_symbol(sym)
    symbol_infos.append(SymbolInfo(...))
```

This is redundant since `_gather()` already builds this information. Consider passing the symbol info from `_gather()` to avoid duplication.

## Missing Test Coverage

The test plan in Task 2 should include additional edge cases:

1. **Symbol lookup failures**: Test what happens when `metric_for_symbol()` fails
2. **Complex expressions**: Test expressions with multiple operators like `(a + b) / (c - d)`
3. **Empty expressions**: Test expressions with no free symbols
4. **Provider inconsistencies**: Test when a symbol exists in the expression but not in the provider

## Implementation Recommendations

### 1. Fix the Existing Bug First

Before implementing the refactoring, create a separate commit to fix the type mismatch bug in `_gather()`. This ensures a clean baseline.

### 2. Task Ordering

Add a new task between Task 2 and 3:

**Task 2.5: Identify Affected Tests**
Run the full test suite to identify all tests that will break with the new return types:
```bash
uv run pytest tests/ -v
```
Document which tests need updates to better plan the work in Task 8.

### 3. Documentation Updates

The plan mentions updating `README.md` in Task 10, but more importantly, the docstrings in the following methods need updates:
- `Evaluator.evaluate()` - Update return type documentation
- `Evaluator._gather()` - Update return type documentation
- `AssertionNode._value` - Update type annotation documentation

### 4. Consider Efficiency Improvements

To avoid rebuilding symbol information:

```python
def _gather(self, expr: sp.Expr) -> Result[tuple[dict[sp.Symbol, float], list[SymbolInfo]], EvaluationFailure]:
    # Return both successes dict and symbol_infos list on success
    # This way evaluate() can reuse the symbol_infos for NaN/infinity cases
```

## Positive Aspects

The plan has several strong points:
- Clean dataclass design with proper use of `Result` types
- Single `EvaluationFailure` from `_gather()` simplifies the API
- Comprehensive TDD approach with tests written first
- Good commit organization and granularity
- Serialization utilities for database persistence

## Conclusion

The refactoring plan is well-structured but needs adjustments to address the implementation issues identified above. With these fixes, the plan will result in a cleaner, more maintainable error handling system for DQX.
