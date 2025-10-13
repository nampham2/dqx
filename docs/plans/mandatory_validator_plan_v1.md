# Make Validator Mandatory in AssertionNode - Implementation Plan v1

## Overview

Currently, the `validator` field in `AssertionNode` is optional (`SymbolicValidator | None`). This plan details how to make it mandatory, ensuring every assertion must validate something specific rather than just computing metrics.

**Key Principle**: Every assertion must assert something. An assertion without validation is just metric computation.

## Background Context

### What is an AssertionNode?
- Part of a graph-based data quality framework (DQX)
- Represents a validation rule to be evaluated against data
- Contains a symbolic expression (`actual`) and a validator function
- Lives in the graph hierarchy: RootNode → CheckNode → AssertionNode

### What is a SymbolicValidator?
- A named validation function that checks if a computed value passes certain criteria
- Contains a `name` (e.g., "> 50") and a `fn` (lambda function that returns True/False)
- Example: `SymbolicValidator("> 50", lambda x: x > 50)`

### Current Problem
- Validator is optional, allowing assertions that don't actually validate anything
- This contradicts the purpose of assertions
- Test code has many assertions without validators, making tests less meaningful

## Implementation Tasks

### Task 1: Update Core AssertionNode Definition
**Files to modify**: `src/dqx/graph/nodes.py`

**What to change**:
1. In `AssertionNode.__init__`, change parameter from:
   ```python
   validator: SymbolicValidator | None = None
   ```
   to:
   ```python
   validator: SymbolicValidator
   ```

2. Remove the `None` default value - validator becomes a required parameter

**Commit message**: `refactor: make validator mandatory in AssertionNode constructor`

### Task 2: Update CheckNode Factory Method
**Files to modify**: `src/dqx/graph/nodes.py`

**What to change**:
1. In `CheckNode.add_assertion`, change parameter from:
   ```python
   validator: SymbolicValidator | None = None
   ```
   to:
   ```python
   validator: SymbolicValidator
   ```

2. Remove the `None` default value

**Commit message**: `refactor: make validator mandatory in CheckNode.add_assertion`

### Task 3: Simplify Evaluator Logic
**Files to modify**: `src/dqx/evaluator.py`

**What to change**:
1. Find the section in `Evaluator.visit()` that handles AssertionNode
2. Change the conditional check from:
   ```python
   if node.validator and node.validator.fn:
       try:
           passed = node.validator.fn(value)
           node._result = "OK" if passed else "FAILURE"
       except Exception as e:
           raise DQXError(f"Validator execution failed: {str(e)}") from e
   else:
       # No validator means no validation - just checking if metric computes
       node._result = "OK"
   ```
   to:
   ```python
   try:
       passed = node.validator.fn(value)
       node._result = "OK" if passed else "FAILURE"
   except Exception as e:
       raise DQXError(f"Validator execution failed: {str(e)}") from e
   ```

3. Remove the else branch since validator will always exist

**Commit message**: `refactor: simplify evaluator logic - validator always exists`

### Task 4: Clean Up API Assertion
**Files to modify**: `src/dqx/api.py`

**What to change**:
1. In `VerificationSuite.collect_results()`, find and remove the line:
   ```python
   assert assertion.validator is not None
   ```

2. This assertion is no longer needed since validator is always present

**Commit message**: `refactor: remove redundant validator assertion in API`

### Task 5: Update Test Files
**Files to modify**: Multiple test files (see list below)

**General pattern for fixing tests**:
```python
# Before:
assertion = check.add_assertion(x1, name="some name")

# After:
validator = SymbolicValidator("> 0", lambda x: x > 0)
assertion = check.add_assertion(x1, name="some name", validator=validator)
```

**Files and number of changes needed**:

1. `tests/test_api.py` - 2 occurrences
   - Search for `add_assertion(` and add validators

2. `tests/test_dataset_validator.py` - 6 occurrences
   - Add validators that make sense for dataset validation tests

3. `tests/test_validator.py` - 4 occurrences
   - Add simple validators for suite validation tests

4. `tests/test_display.py` - 3 occurrences
   - Add validators for display/visualization tests

5. `tests/test_evaluator_integration.py` - 3 occurrences
   - Add validators appropriate for integration tests

6. `tests/test_evaluator_validation.py` - 1 occurrence
   - Remove the test `test_no_validator_results_in_ok_status` entirely
   - Update the one call with `validator=None`

7. `tests/graph/test_visitor.py` - 9 occurrences
   - Add validators for graph traversal tests

8. `tests/graph/test_typed_parents.py` - 1 occurrence
   - Add validator for parent type testing

9. `tests/test_graph_display.py` - 2 occurrences
   - Add validators for graph display tests

**Suggested validators for tests**:
- For general tests: `SymbolicValidator("> 0", lambda x: x > 0)`
- For boundary tests: `SymbolicValidator("< 100", lambda x: x < 100)`
- For equality tests: `SymbolicValidator("= 42", lambda x: x == 42)`
- For non-null tests: `SymbolicValidator("not None", lambda x: x is not None)`

**Commit after each file**: `test: add validators to assertions in [filename]`

### Task 6: Run Tests and Fix Issues
**Commands to run**:
```bash
# Run type checking
uv run mypy src/

# Run tests for modified files
uv run pytest tests/test_api.py -v
uv run pytest tests/test_dataset_validator.py -v
# ... repeat for each modified test file

# Run all tests
uv run pytest -v

# Check code quality
uv run ruff check src/ tests/
```

**What to look for**:
- All tests should pass
- No type errors from mypy
- No linting issues from ruff

**Final commit**: `test: verify all tests pass with mandatory validators`

## Testing Strategy

### How to Test Your Changes

1. **Unit Testing**: Each modified test file should pass individually
   ```bash
   uv run pytest tests/[specific_test_file.py] -v
   ```

2. **Integration Testing**: Run the full test suite
   ```bash
   uv run pytest -v
   ```

3. **Type Checking**: Ensure no type errors
   ```bash
   uv run mypy src/
   ```

4. **Code Quality**: Check formatting and linting
   ```bash
   uv run ruff check src/ tests/ --fix
   ```

### What Makes a Good Test Validator

When adding validators to tests, consider:
- **Meaningful validation**: The validator should test something relevant to the test case
- **Simple predicates**: Use simple lambda functions that are easy to understand
- **Clear names**: The validator name should describe what's being checked

Examples:
```python
# Good - clear and relevant
SymbolicValidator("> 0", lambda x: x > 0)  # For positive value tests
SymbolicValidator("in range [0, 1]", lambda x: 0 <= x <= 1)  # For probability tests

# Bad - too complex or unclear
SymbolicValidator("valid", lambda x: True)  # Always passes - meaningless
SymbolicValidator("x", lambda x: x > 0 and x < 100 and x % 2 == 0)  # Too complex
```

## Rollback Plan

If issues arise:
1. Git revert the commits in reverse order
2. The changes are isolated and can be reverted independently

## Success Criteria

- [ ] All tests pass
- [ ] No type errors from mypy
- [ ] No linting issues
- [ ] Every AssertionNode creation requires a validator
- [ ] Code is cleaner with no None checks for validators

## Notes

- **No backward compatibility** is required - this is a breaking change
- **YAGNI**: We're not adding any new features, just enforcing existing design
- **DRY**: The validator pattern is consistent across all tests
- **TDD**: Run tests after each change to ensure nothing breaks
