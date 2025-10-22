# Implementation Plan: Remove Tags Parameter from @check Decorator

## Overview
Remove the unused `tags` parameter from the `@check` decorator. This parameter is not used anywhere in the codebase except in tests and documentation.

**Key Requirements:**
- No backward compatibility needed (as specified by user)
- Follow TDD principles
- Ensure all tests pass before committing
- Single atomic commit with all changes

## Background
The `tags` parameter in the `@check` decorator is currently unused in the production code. Analysis shows:
- It's stored in `CheckNode.tags` but never accessed for any functionality
- Only tests and documentation reference it
- Removing it simplifies the API without losing functionality

## Task Groups

### Task Group 1: Remove Tags from Core Classes
**Objective**: Remove tags parameter from graph node classes

1. **Update CheckNode class** (`src/dqx/graph/nodes.py`)
   - Remove `tags` parameter from `__init__` signature
   - Remove `self.tags = tags or []` assignment
   - Keep only `name` and `datasets` parameters

2. **Update RootNode.add_check()** (`src/dqx/graph/nodes.py`)
   - Remove `tags` parameter from method signature
   - Remove `tags` from CheckNode instantiation

**Verification**:
- Run `uv run mypy src/dqx/graph/`
- Run `uv run ruff check src/dqx/graph/`

### Task Group 2: Remove Tags from API Layer
**Objective**: Remove tags parameter from decorator and internal functions

1. **Update _create_check()** (`src/dqx/api.py`)
   - Remove `tags` parameter from function signature
   - Remove `tags` from docstring
   - Update `add_check()` call to remove tags parameter

2. **Update check() decorator** (`src/dqx/api.py`)
   - Remove `tags` parameter from decorator signature
   - Remove `tags` from docstring example (change to show only name and datasets)
   - Remove `tags` from functools.partial call

**Verification**:
- Run `uv run mypy src/dqx/api.py`
- Run `uv run ruff check src/dqx/api.py`

### Task Group 3: Update Tests
**Objective**: Remove all tags usage from test files

1. **Update test_api_coverage.py**
   - Line ~156: Change `@check(name="Check 2", tags=["important"])` to `@check(name="Check 2")`
   - Lines ~179-180: Remove these two lines completely:
     ```python
     check2_node = next(c for c in checks if c.name == "Check 2")
     assert check2_node.tags == ["important"]
     ```

2. **Update test_api.py**
   - Find and change `@check(name="Order Validation Check", tags=["critical"])` to `@check(name="Order Validation Check")`

3. **Update test_typed_parents.py**
   - Line ~55: Change `check = root.add_check("my_check", tags=["important"])` to `check = root.add_check("my_check")`

**Verification**:
- Run `uv run pytest tests/test_api_coverage.py -v`
- Run `uv run pytest tests/test_api.py -v`
- Run `uv run pytest tests/graph/test_typed_parents.py -v`

### Task Group 4: Update Documentation
**Objective**: Remove tags references from comments and docstrings

1. **Update graph/traversal.py comments**
   - Find and update/remove the example showing `print(f"Check: {check.name}, Tags: {check.tags}")`
   - Remove the `tag_counts` example that counts tags

2. **Verify no other documentation references**
   - Search for any remaining "tags" references related to checks
   - Ensure examples in docstrings are updated

**Verification**:
- Run `grep -r "check.*tags" docs/` to ensure no docs reference check tags
- Run `grep -r "\.tags" src/` to ensure no code references check.tags

### Task Group 5: Final Verification and Commit
**Objective**: Ensure all changes work together and commit

1. **Run comprehensive tests**
   ```bash
   uv run pytest tests/ -v
   ```

2. **Run type checking**
   ```bash
   uv run mypy src/
   ```

3. **Run linting**
   ```bash
   uv run ruff check --fix
   ```

4. **Run pre-commit hooks**
   ```bash
   bin/run-hooks.sh
   ```

5. **Commit with breaking change message**
   ```
   refactor!: remove unused tags parameter from @check decorator

   BREAKING CHANGE: The tags parameter has been removed from the @check
   decorator. This parameter was not used anywhere in the codebase except
   in tests and documentation.

   Migration guide:
   Before: @check(name="My Check", tags=["critical"])
   After: @check(name="My Check")

   - Remove tags from CheckNode class and constructor
   - Remove tags from RootNode.add_check() method
   - Remove tags from _create_check() function
   - Remove tags parameter from check() decorator
   - Update all tests to remove tags usage
   - Update documentation and examples
   ```

## Implementation Notes

### Files to Modify
1. `src/dqx/graph/nodes.py` - Remove tags from CheckNode and RootNode.add_check
2. `src/dqx/api.py` - Remove tags from decorator and _create_check
3. `tests/test_api_coverage.py` - Remove tags usage and assertion
4. `tests/test_api.py` - Remove tags from decorator
5. `tests/graph/test_typed_parents.py` - Remove tags from add_check call
6. `src/dqx/graph/traversal.py` - Update/remove tag examples in comments

### Testing Strategy
- After each task group, run relevant tests to ensure no regressions
- Use mypy and ruff after each code change
- Final comprehensive test run before committing
- All tests must pass with 100% coverage maintained

### Risk Mitigation
- Single atomic commit ensures easy rollback if needed
- Comprehensive test coverage ensures no functionality is broken
- Type checking catches any interface mismatches
- Pre-commit hooks ensure code quality standards are met
