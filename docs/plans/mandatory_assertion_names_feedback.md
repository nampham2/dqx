# Architectural Feedback on Mandatory Assertion Names Plan

## Overview

This document provides architectural feedback on the proposed implementation plan for mandatory assertion names in DQX. The plan introduces a two-stage assertion building process (AssertionDraft → AssertionReady) to enforce required names at the type system level.

## Overall Assessment

The plan is well-structured and the two-stage assertion building approach is a solid design pattern that will effectively enforce mandatory names at compile time through the type system. However, there are several important considerations and potential improvements to discuss.

## Key Observations

### Current Usage Analysis
- Many existing assertions don't have names (found ~10 instances in e2e tests alone)
- One test has an empty `.where()` call which would break with the new API
- Some assertions already follow good naming practices

### Design Strengths
- Clear separation of concerns with two distinct classes
- Type-safe enforcement of the requirement
- Good use of the builder pattern progression
- Comprehensive test coverage plan

## Critical Feedback & Questions

### 1. Alternative Design Consideration

Instead of two separate classes, consider making the name a required parameter in `assert_that()`. This would be simpler:

```python
# Alternative approach
ctx.assert_that(mp.average("price"), name="Average price is positive").is_positive()

# vs. Proposed approach
ctx.assert_that(mp.average("price")).where(name="Average price is positive").is_positive()
```

**Pros of alternative approach:**
- Simpler API with one less method call
- No intermediate stage to manage
- More direct and intuitive

**Pros of proposed approach:**
- Better separation of concerns
- Allows future extension of the `where()` method
- Clear distinction between expression setup and assertion configuration

### 2. Empty `where()` Call Issue

Found in the e2e tests:
```python
ctx.assert_that(tax_avg / tax_avg_lag).where().is_eq(1.0, tol=0.01)
```

This will break with the new API since `where()` will require a name parameter. Options:
- Force migration (recommended for consistency)
- Provide a migration script to automatically add placeholder names
- Temporarily support empty where() with deprecation warning

### 3. Migration Strategy

The plan doesn't discuss migration strategy. With ~22 assertions to update:

**Option A: Hard Breaking Change**
- Clear and immediate enforcement
- Forces all code to be updated at once
- Risk of blocking deployment if not all code is updated

**Option B: Deprecation Path**
- Keep AssertBuilder but emit warnings when used without names
- Gradual migration over several releases
- More complex to maintain both APIs

**Option C: Automated Migration**
- Provide a script to automatically add placeholder names
- Quick but may result in low-quality names like "Assertion 1"
- Could combine with manual review process

### 4. Error Messages

The plan should specify clear error messages for common mistakes:

```python
# When trying to call assertion methods on AssertionDraft:
AttributeError: 'AssertionDraft' object has no attribute 'is_positive'.
Did you forget to call .where(name="...") first?

# When where() is called without name:
TypeError: where() missing 1 required keyword-only argument: 'name'
```

### 5. Implementation Detail Questions

- Should `AssertionReady._create_assertion_node()` continue to handle `context=None` gracefully?
- Should we validate that names are non-empty strings? Current `.where()` accepts `None`
- Should there be a maximum length for assertion names?
- Should we check for duplicate names within a check?

## Suggestions for Improvement

### 1. Add Validation for Name Quality

```python
def where(self, *, name: str, severity: SeverityLevel | None = None) -> AssertionReady:
    if not name or not name.strip():
        raise ValueError("Assertion name cannot be empty")
    if len(name.strip()) < 10:
        logger.warning(f"Assertion name '{name}' is very short. Consider a more descriptive name.")
    if len(name) > 200:
        raise ValueError("Assertion name is too long (max 200 characters)")
    return AssertionReady(
        actual=self._actual,
        name=name.strip(),
        severity=severity,
        context=self._context
    )
```

### 2. Consider Helper Methods

For common assertion patterns:

```python
class Context:
    def assert_positive(self, expr: sp.Expr, metric_name: str) -> None:
        """Helper for common positive value assertions."""
        self.assert_that(expr).where(
            name=f"{metric_name} must be positive"
        ).is_positive()

    def assert_not_null(self, column: str) -> None:
        """Helper for null count assertions."""
        self.assert_that(self.provider.null_count(column)).where(
            name=f"Column '{column}' has no nulls"
        ).is_eq(0)
```

### 3. Documentation Enhancement

Include examples of good vs bad assertion names:

**Good Names:**
- "Average order value must be positive"
- "Customer ID null count is less than 100"
- "Daily revenue increase is within 20%"

**Bad Names:**
- "Check 1"
- "Assert"
- "Test"
- "Validation"

### 4. Type Hints Enhancement

Consider using `@overload` for better IDE support:

```python
from typing import overload

class AssertionDraft:
    @overload
    def where(self, *, name: str) -> AssertionReady: ...

    @overload
    def where(self, *, name: str, severity: SeverityLevel) -> AssertionReady: ...
```

## Technical Implementation Notes

### 1. Import Organization

Update `__all__` in api.py:
```python
__all__ = [
    "AssertionDraft",
    "AssertionReady",
    # Remove: "AssertBuilder",
    "Context",
    "VerificationSuite",
    # ... other exports
]
```

### 2. Backward Compatibility Option

If a gentler migration is desired:

```python
class AssertBuilder:
    """Deprecated: Use assert_that().where(name=...) pattern instead."""

    def __init__(self, actual: sp.Expr, context: Context | None = None) -> None:
        warnings.warn(
            "AssertBuilder is deprecated. Assertions now require names. "
            "Use: ctx.assert_that(expr).where(name='...').is_positive()",
            DeprecationWarning,
            stacklevel=2
        )
        # ... rest of implementation
```

### 3. Additional Test Coverage

The plan should include tests for:
- Type checking that AssertionDraft doesn't have assertion methods
- Clear error messages when misusing the API
- Validation of empty/whitespace-only names
- Very long assertion names
- Special characters in assertion names
- Thread safety of the two-stage process

## Migration Checklist

- [ ] Update all tests to use new assertion pattern
- [ ] Update documentation and examples
- [ ] Create migration guide for users
- [ ] Consider automated migration script
- [ ] Update any code generation tools
- [ ] Plan announcement for breaking change
- [ ] Update type stubs if published separately

## Summary

The plan is solid and will achieve the goal of mandatory assertion names. The two-stage approach provides excellent compile-time safety but is more complex than alternatives.

**Key Recommendations:**
1. Add name validation to prevent empty/low-quality names
2. Clarify the migration strategy (recommend hard break with good tooling support)
3. Enhance error messages for better developer experience
4. Consider the simpler alternative of required parameters
5. Provide helper methods for common patterns

The approach will significantly improve debugging and test failure analysis by ensuring every assertion has a clear, descriptive name.
