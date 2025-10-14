# Implementation Plan: Remove VerificationSuiteBuilder

## Overview

This plan removes the `VerificationSuiteBuilder` class from the DQX framework to simplify the API. The builder pattern adds unnecessary complexity for the simple task of collecting checks into a list.

## Background for Engineers

### What is DQX?

DQX is a data quality framework that validates data using assertions. Users write "checks" containing "assertions" about their data:

```python
@check(name="Price validation")
def validate_prices(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.average("price"))
       .where(name="Average price is positive")
       .is_positive()
```

### What is VerificationSuite?

`VerificationSuite` is the main class that collects checks and executes them against data sources. It currently supports two ways of instantiation:

1. **Direct instantiation** (preferred):
   ```python
   suite = VerificationSuite([check1, check2], db, "My Suite")
   ```

2. **Builder pattern** (to be removed):
   ```python
   suite = VerificationSuiteBuilder("My Suite", db)
       .add_check(check1)
       .add_check(check2)
       .build()
   ```

### Why Remove the Builder?

The builder pattern is over-engineered for this use case because:
- It only collects checks into a list (no complex configuration)
- Direct instantiation is clearer and more concise
- Reduces API surface area
- Follows YAGNI principle (You Aren't Gonna Need It)

### Development Environment

- **Python**: 3.11/3.12 with `uv` package manager
- **Testing**: `uv run pytest`
- **Type checking**: `uv run mypy`
- **Linting**: `uv run ruff check --fix`
- **Pre-commit**: `bin/run-hooks.sh`

## Implementation Tasks

### Task 1: Write Failing Test First (TDD)

**Goal**: Create a test that ensures VerificationSuiteBuilder no longer exists after removal.

**File to create**: `tests/test_verification_suite_builder_removal.py`

**What to implement**:
```python
"""
Test to ensure VerificationSuiteBuilder has been properly removed.

This test will fail initially and pass once the builder is removed.
"""
import pytest


def test_verification_suite_builder_does_not_exist():
    """Test that VerificationSuiteBuilder class no longer exists."""
    with pytest.raises(ImportError):
        from dqx.api import VerificationSuiteBuilder


def test_verification_suite_builder_not_in_module():
    """Test that VerificationSuiteBuilder is not accessible via module attribute."""
    import dqx.api

    assert not hasattr(dqx.api, 'VerificationSuiteBuilder'), \
        "VerificationSuiteBuilder should not be accessible in dqx.api module"


def test_direct_verification_suite_still_works():
    """Test that direct VerificationSuite instantiation still works."""
    from dqx.api import VerificationSuite, check, Context, MetricProvider
    from dqx.orm.repositories import InMemoryMetricDB

    # Create a simple check for testing
    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        pass

    # This should work without the builder
    db = InMemoryMetricDB()
    suite = VerificationSuite([test_check], db, "Test Suite")

    assert suite is not None
    assert len(suite._checks) == 1
    assert suite._name == "Test Suite"
```

**Commands to run**:
```bash
# This test should FAIL initially (builder still exists)
uv run pytest tests/test_verification_suite_builder_removal.py -xvs
```

**Expected result**: Test fails because VerificationSuiteBuilder still exists.

**Commit message**: "test: add failing test for VerificationSuiteBuilder removal (TDD)"

### Task 2: Remove VerificationSuiteBuilder Class

**Goal**: Delete the VerificationSuiteBuilder class from api.py.

**File to modify**: `src/dqx/api.py`

**What to remove**: Lines containing the VerificationSuiteBuilder class (approximately lines 530-570):

```python
class VerificationSuiteBuilder:
    """
    Builder pattern for creating VerificationSuite instances with fluent configuration.

    Example:
        >>> builder = VerificationSuiteBuilder("My Suite", db)
        >>> suite = builder.add_check(check1).add_checks([check2, check3]).build()
    """

    def __init__(self, name: str, db: MetricDB) -> None:
        """
        Initialize the builder.

        Args:
            name: Name for the verification suite
            db: Database for metrics storage
        """
        self._name = name
        self._db = db
        self._checks: list[CheckProducer | DecoratedCheck] = []

    def add_check(self, check: CheckProducer | DecoratedCheck) -> Self:
        """Add a single check to the suite."""
        self._checks.append(check)
        return self

    def add_checks(self, checks: Sequence[CheckProducer | DecoratedCheck]) -> Self:
        """Add multiple checks to the suite."""
        self._checks.extend(checks)
        return self

    def build(self) -> VerificationSuite:
        """Build and return the configured VerificationSuite."""
        return VerificationSuite(self._checks, self._db, self._name)
```

**Steps**:
1. Open `src/dqx/api.py`
2. Find the `VerificationSuiteBuilder` class
3. Delete the entire class definition
4. Remove any imports of `Self` if it's no longer used elsewhere in the file
5. Check if VerificationSuiteBuilder is in any `__all__` list and remove it

**Commands to run**:
```bash
# Check the file structure
grep -n "class VerificationSuiteBuilder" src/dqx/api.py

# After removal, verify it's gone
grep -n "VerificationSuiteBuilder" src/dqx/api.py

# Run the failing test - it should now pass!
uv run pytest tests/test_verification_suite_builder_removal.py -xvs
```

**Expected result**: Test now passes because VerificationSuiteBuilder no longer exists.

**Commit message**: "feat: remove VerificationSuiteBuilder class from api.py"

### Task 3: Update Example File

**Goal**: Replace VerificationSuiteBuilder usage with direct VerificationSuite instantiation.

**File to modify**: `examples/result_collection_demo.py`

**Current code** (find this pattern):
```python
from dqx.api import Context, MetricProvider, VerificationSuiteBuilder, check

# Later in the file:
suite = (
    VerificationSuiteBuilder("E-commerce Data Quality with Failures", db)
    .add_check(basic_validations)
    # ... possibly more .add_check() calls
    .build()
)
```

**Replace with**:
```python
from dqx.api import Context, MetricProvider, VerificationSuite, check

# Later in the file:
suite = VerificationSuite(
    [basic_validations],  # Put all checks in a list
    db,
    "E-commerce Data Quality with Failures"
)
```

**Steps**:
1. Open `examples/result_collection_demo.py`
2. Change the import from `VerificationSuiteBuilder` to `VerificationSuite`
3. Find the suite creation code
4. Collect all checks that were added via `.add_check()` into a list
5. Replace builder pattern with direct instantiation

**Commands to run**:
```bash
# Test the example still works
uv run python examples/result_collection_demo.py

# Check for any remaining VerificationSuiteBuilder references
grep -n "VerificationSuiteBuilder" examples/result_collection_demo.py
```

**Expected result**: Example runs without errors and no VerificationSuiteBuilder references remain.

**Commit message**: "fix(examples): replace VerificationSuiteBuilder with direct VerificationSuite instantiation"

### Task 4: Update First Test File

**Goal**: Replace VerificationSuiteBuilder usage in test_assertion_result_collection.py.

**File to modify**: `tests/test_assertion_result_collection.py`

**Pattern to find**:
```python
from dqx.api import Context, MetricProvider, VerificationSuite, VerificationSuiteBuilder, check

# Later:
suite = VerificationSuiteBuilder("Test Suite", db).add_check(validate_data).build()
```

**Replace with**:
```python
from dqx.api import Context, MetricProvider, VerificationSuite, check

# Later:
suite = VerificationSuite([validate_data], db, "Test Suite")
```

**Steps**:
1. Open `tests/test_assertion_result_collection.py`
2. Remove `VerificationSuiteBuilder` from the import statement
3. Find all occurrences of `VerificationSuiteBuilder("Suite Name", db).add_check(check).build()`
4. Replace with `VerificationSuite([check], db, "Suite Name")`

**Commands to run**:
```bash
# Run the specific test file
uv run pytest tests/test_assertion_result_collection.py -xvs

# Check for remaining references
grep -n "VerificationSuiteBuilder" tests/test_assertion_result_collection.py
```

**Expected result**: All tests pass and no VerificationSuiteBuilder references remain.

**Commit message**: "fix(tests): replace VerificationSuiteBuilder in test_assertion_result_collection.py"

### Task 5: Update Second Test File

**Goal**: Replace VerificationSuiteBuilder usage in test_api_validation_integration.py.

**File to modify**: `tests/test_api_validation_integration.py`

**Pattern to find** (multiple occurrences):
```python
from dqx.api import Context, VerificationSuiteBuilder, check

# Later:
suite = VerificationSuiteBuilder("Valid Suite", db).add_check(check1).add_check(check2).build()
suite = VerificationSuiteBuilder("Invalid Suite", db).add_check(check1).add_check(check2).build()
suite = VerificationSuiteBuilder("Test Suite", db).add_check(empty_check).build()
```

**Replace with**:
```python
from dqx.api import Context, VerificationSuite, check

# Later:
suite = VerificationSuite([check1, check2], db, "Valid Suite")
suite = VerificationSuite([check1, check2], db, "Invalid Suite")
suite = VerificationSuite([empty_check], db, "Test Suite")
```

**Steps**:
1. Open `tests/test_api_validation_integration.py`
2. Remove `VerificationSuiteBuilder` from import, add `VerificationSuite`
3. Find all builder patterns and convert to direct instantiation
4. Pay attention to multiple `.add_check()` calls - collect all checks into a list

**Commands to run**:
```bash
# Run the specific test file
uv run pytest tests/test_api_validation_integration.py -xvs

# Check for remaining references
grep -n "VerificationSuiteBuilder" tests/test_api_validation_integration.py
```

**Expected result**: All tests pass and no VerificationSuiteBuilder references remain.

**Commit message**: "fix(tests): replace VerificationSuiteBuilder in test_api_validation_integration.py"

### Task 6: Run Full Test Suite

**Goal**: Ensure all tests pass after the removal.

**Commands to run**:
```bash
# Run all tests to ensure nothing is broken
uv run pytest

# Run type checking
uv run mypy src/dqx/api.py

# Run linting
uv run ruff check src/dqx/

# Run pre-commit hooks
bin/run-hooks.sh
```

**What to fix if tests fail**:
1. **Import errors**: Look for any remaining imports of VerificationSuiteBuilder
2. **Test failures**: Check if any tests indirectly depend on the builder
3. **Type errors**: Ensure no type annotations reference the removed class
4. **Linting errors**: Fix any code style issues

**Expected result**: All tests pass, no type errors, no linting issues.

**Commit message**: "test: verify all tests pass after VerificationSuiteBuilder removal"

### Task 7: Verify No Remaining References

**Goal**: Ensure VerificationSuiteBuilder is completely removed from the codebase.

**Commands to run**:
```bash
# Search entire codebase for any remaining references
grep -r "VerificationSuiteBuilder" . --exclude-dir=.git --exclude-dir=__pycache__

# Search for any remaining references in docs
find docs/ -name "*.md" -exec grep -l "VerificationSuiteBuilder" {} \;

# Check README for any examples
grep -n "VerificationSuiteBuilder" README.md
```

**What to do if references found**:
1. **Documentation**: Update any examples or API references
2. **Comments**: Remove or update any code comments mentioning the builder
3. **Docstrings**: Update any docstrings that reference the old pattern

**Expected result**: No references to VerificationSuiteBuilder remain anywhere in the codebase.

**Commit message**: "docs: remove any remaining VerificationSuiteBuilder references"

### Task 8: Update Documentation (if needed)

**Goal**: Update README or other documentation if it mentions VerificationSuiteBuilder.

**File to check**: `README.md`

**What to look for**:
- Examples showing VerificationSuiteBuilder usage
- API documentation mentioning the builder pattern
- Quick start guides using the old pattern

**What to update**:
Replace any builder examples with direct instantiation:

```python
# OLD (remove if found)
suite = VerificationSuiteBuilder("My Suite", db) \
    .add_check(check1) \
    .add_check(check2) \
    .build()

# NEW (replace with)
suite = VerificationSuite([check1, check2], db, "My Suite")
```

**Commands to run**:
```bash
# Check if README mentions the builder
grep -n "VerificationSuiteBuilder" README.md

# If found, edit the file and run
uv run python -c "print('README updated')"
```

**Commit message**: "docs: update README to use direct VerificationSuite instantiation"

### Task 9: Clean Up Test File

**Goal**: Remove the temporary test file created for TDD.

**File to remove**: `tests/test_verification_suite_builder_removal.py`

**Rationale**: This test was only needed to guide the removal process. Once the builder is gone, the test serves no ongoing purpose.

**Commands to run**:
```bash
# Remove the temporary test file
rm tests/test_verification_suite_builder_removal.py

# Verify tests still pass
uv run pytest
```

**Commit message**: "test: remove temporary VerificationSuiteBuilder removal test"

### Task 10: Final Validation and Issue Resolution

**Goal**: Run comprehensive validation and fix any remaining issues.

**Commands to run**:
```bash
# Run full test suite to ensure everything works
uv run pytest -v

# Run type checking on entire codebase
uv run mypy src/dqx/

# Run linting and auto-fix issues
uv run ruff check --fix src/dqx/ tests/ examples/

# Run pre-commit hooks (comprehensive validation)
bin/run-hooks.sh
```

**What to fix if issues arise**:

1. **Test Failures**:
   - Review failing test output carefully
   - Check if any tests indirectly use VerificationSuiteBuilder
   - Update import statements in any missed test files
   - Ensure all check function calls use correct VerificationSuite syntax

2. **Type Checking Errors**:
   - Remove any remaining type annotations that reference VerificationSuiteBuilder
   - Check if any return types or function parameters mention the removed class
   - Update any type hints in docstrings

3. **Linting Issues**:
   - Remove any unused imports of VerificationSuiteBuilder
   - Fix any code formatting issues introduced during editing
   - Remove any unused variables or functions

4. **Pre-commit Hook Failures**:
   - **isort**: Import sorting issues - let it auto-fix
   - **black**: Formatting issues - let it auto-fix
   - **flake8**: Code style violations - fix manually
   - **mypy**: Type checking - fix type annotations
   - **pytest**: Test failures - investigate and fix

**Step-by-step resolution**:
1. Run each command individually to isolate issues
2. Fix one type of issue at a time (tests, then types, then linting)
3. Re-run the failing tool after each fix to verify resolution
4. Only proceed to the next tool once the current one passes
5. If pre-commit fails, check which specific hook failed and address that tool

**Expected result**: All tools pass with no errors or warnings.

**Commit message**: "chore: final validation and cleanup after VerificationSuiteBuilder removal"

## Testing Strategy

### Test-Driven Development (TDD)

1. **Red**: Write failing test (Task 1)
2. **Green**: Make test pass by removing builder (Task 2)
3. **Refactor**: Update all usage sites (Tasks 3-5)
4. **Verify**: Ensure all tests pass (Task 6)

### Test Design Principles

- **Isolated**: Each test checks one specific aspect
- **Clear**: Test names explain what they verify
- **Fast**: No external dependencies or slow operations
- **Deterministic**: Same input always produces same result

### What Tests Cover

- VerificationSuiteBuilder no longer exists
- Direct VerificationSuite instantiation works
- All existing functionality remains intact
- No performance regression

## Risk Mitigation

### Potential Issues

1. **Missed References**: Some code might still try to import VerificationSuiteBuilder
   - **Mitigation**: Comprehensive grep search across codebase

2. **Test Failures**: Existing tests might break
   - **Mitigation**: Update all known usage sites before running full test suite

3. **Import Errors**: Other modules might re-export the builder
   - **Mitigation**: Check all `__init__.py` files for re-exports

### Rollback Plan

If issues arise:
1. Revert the commits in reverse order
2. The builder class can be restored from git history
3. All changes are atomic and easily reversible

## Definition of Done

- [ ] VerificationSuiteBuilder class removed from api.py
- [ ] All usage sites updated to use direct instantiation
- [ ] All tests pass (`uv run pytest`)
- [ ] Type checking passes (`uv run mypy`)
- [ ] Linting passes (`uv run ruff check`)
- [ ] Pre-commit hooks pass (`bin/run-hooks.sh`) ⭐ **FINAL VALIDATION**
- [ ] No remaining references in codebase (`grep -r "VerificationSuiteBuilder"`)
- [ ] Documentation updated (if needed)
- [ ] All commits follow conventional commit format
- [ ] **Task 10 completed successfully** - All automated checks pass

## Quick Reference

### Before (Builder Pattern)
```python
from dqx.api import VerificationSuiteBuilder

suite = VerificationSuiteBuilder("Suite Name", db) \
    .add_check(check1) \
    .add_check(check2) \
    .build()
```

### After (Direct Instantiation)
```python
from dqx.api import VerificationSuite

suite = VerificationSuite([check1, check2], db, "Suite Name")
```

### Files That Need Updates
1. `src/dqx/api.py` - Remove the class
2. `examples/result_collection_demo.py` - Update usage
3. `tests/test_assertion_result_collection.py` - Update usage
4. `tests/test_api_validation_integration.py` - Update usage
5. `README.md` - Update examples (if present)

### Key Commands
```bash
# Run tests
uv run pytest

# Type check
uv run mypy src/dqx/

# Lint
uv run ruff check --fix

# Search for references
grep -r "VerificationSuiteBuilder" .

# Pre-commit
bin/run-hooks.sh
```

This plan removes unnecessary complexity while maintaining all existing functionality, following DQX's philosophy of simple, direct approaches over clever patterns.
