# VerificationSuite Graph Improvements Implementation Plan v1

## Overview

This plan describes how to improve the VerificationSuite class by:
1. Adding a public `graph` property for cleaner access to the internal graph
2. Renaming the `collect()` method to `build_graph()` for clarity
3. Removing the unused `validate()` method (dead code)

**Important**: This is a breaking change with NO backward compatibility.

## Background Context

### What is VerificationSuite?
VerificationSuite is the main entry point for running data quality checks in DQX. It:
- Collects check functions that define data quality assertions
- Builds a dependency graph of these checks
- Executes the checks against data sources
- Returns results

### Current Issues
1. Code accesses the internal graph via `context._graph` in 6 places - not clean
2. The method name `collect()` doesn't clearly describe that it builds a graph
3. The `validate()` method exists but is never used outside of tests (dead code)

## Development Environment Setup

### Prerequisites
- Python 3.11 or 3.12
- Project uses `uv` package manager
- Shell is zsh (not bash)

### Commands You'll Need
```bash
# Run tests
uv run pytest tests/test_api.py -v

# Run specific test
uv run pytest tests/test_api.py::test_name -v

# Check types
uv run mypy src/dqx/api.py

# Check linting
uv run ruff check src/dqx/api.py

# Auto-fix linting
uv run ruff check --fix src/dqx/api.py

# Run all pre-commit hooks
./bin/run-hooks.sh
```

## Implementation Tasks

### Task 1: Write Failing Tests for Graph Property

**Why**: Following TDD, we write tests first to define expected behavior.

**File**: `tests/test_api.py`

**Add this test**:
```python
def test_verification_suite_graph_property(
    simple_check: CheckProducer, db: MetricDB
) -> None:
    """Test that VerificationSuite exposes graph property."""
    suite = VerificationSuite([simple_check], db, "Test Suite")

    # Graph should be accessible
    assert hasattr(suite, 'graph')

    # Should return a Graph instance
    from dqx.graph.traversal import Graph
    assert isinstance(suite.graph, Graph)

    # Should have the suite name as root
    assert suite.graph.root.name == "Test Suite"
```

**Run test to confirm it fails**:
```bash
uv run pytest tests/test_api.py::test_verification_suite_graph_property -v
```

**Commit**:
```bash
git add tests/test_api.py
git commit -m "test: add failing test for VerificationSuite.graph property"
```

### Task 2: Implement Graph Property

**File**: `src/dqx/api.py`

**Find the VerificationSuite class** (around line 300) and add this property after the `provider` property:

```python
@property
def graph(self) -> Graph:
    """
    The dependency graph containing all checks and assertions.

    This property provides direct access to the internal graph structure,
    which contains the root node and all registered checks. The graph
    is built when build_graph() (formerly collect()) is called.

    Returns:
        Graph instance with the root node and all registered checks

    Example:
        >>> suite = VerificationSuite(checks, db, "My Suite")
        >>> suite.build_graph(context, key)
        >>> print(suite.graph.root.name)  # "My Suite"
        >>> for check in suite.graph.root.children:
        ...     print(check.name)
    """
    return self._context._graph
```

**Import needed**:
Add to imports at top if not already there:
```python
from dqx.graph.traversal import Graph
```

**Run test to confirm it passes**:
```bash
uv run pytest tests/test_api.py::test_verification_suite_graph_property -v
```

**Commit**:
```bash
git add src/dqx/api.py
git commit -m "feat: add graph property to VerificationSuite"
```

### Task 3: Update Internal Graph References

**File**: `src/dqx/api.py`

**Replace these occurrences** of `context._graph` or `self._context._graph`:

1. In `validate()` method (~line 320):
   ```python
   # OLD: return self._validator.validate(temp_context._graph, self.provider)
   # NEW:
   return self._validator.validate(temp_context._graph, self.provider)
   # Note: This stays as is because temp_context is local
   ```

2. In `collect()` method (~line 340):
   ```python
   # OLD: report = self._validator.validate(context._graph, context.provider)
   # NEW: Keep as is - this uses the parameter context, not self._context
   ```

3. In `run()` method (~line 380):
   ```python
   # OLD: self._context._graph.impute_datasets(...)
   # NEW:
   self.graph.impute_datasets(list(datasources.keys()), self._context.provider)
   ```

4. In `run()` method (~line 390):
   ```python
   # OLD: self._context._graph.bfs(evaluator)
   # NEW:
   self.graph.bfs(evaluator)
   ```

5. In `collect_results()` method (~line 430):
   ```python
   # OLD: for assertion in self._context._graph.assertions():
   # NEW:
   for assertion in self.graph.assertions():
   ```

6. In `_create_check()` function (~line 500):
   ```python
   # OLD: node = context._graph.root.add_check(...)
   # NEW: Keep as is - this uses the parameter context, not self._context
   ```

**Run all tests to ensure nothing broke**:
```bash
uv run pytest tests/test_api.py -v
```

**Commit**:
```bash
git add src/dqx/api.py
git commit -m "refactor: use graph property instead of _context._graph"
```

### Task 4: Write Tests for Renaming collect to build_graph

**File**: `tests/test_api.py`

**Add test to verify build_graph works**:
```python
def test_verification_suite_build_graph_method(
    simple_check: CheckProducer, db: MetricDB
) -> None:
    """Test that build_graph method works (renamed from collect)."""
    suite = VerificationSuite([simple_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    context = Context("Test Suite", db)

    # Should have build_graph method
    assert hasattr(suite, 'build_graph')

    # Should not have collect method anymore
    assert not hasattr(suite, 'collect')

    # build_graph should work
    suite.build_graph(context, key)

    # Graph should be populated
    assert len(suite.graph.root.children) > 0
```

**Run test to see it fail**:
```bash
uv run pytest tests/test_api.py::test_verification_suite_build_graph_method -v
```

**Commit**:
```bash
git add tests/test_api.py
git commit -m "test: add failing test for build_graph method rename"
```

### Task 5: Rename collect to build_graph

**File**: `src/dqx/api.py`

1. **Find the collect method** (around line 330) and rename it:
   ```python
   # OLD: def collect(self, context: Context, key: ResultKey) -> None:
   # NEW:
   def build_graph(self, context: Context, key: ResultKey) -> None:
       """
       Build the dependency graph by executing all checks without running analysis.

       This method:
       1. Executes all check functions to populate the graph with assertions
       2. Validates the graph structure for errors or warnings
       3. Raises DQXError if validation fails

       Args:
           context: The execution context containing the graph
           key: The result key defining the time period and tags

       Raises:
           DQXError: If validation fails or duplicate checks are found

       Example:
           >>> suite = VerificationSuite(checks, db, "My Suite")
           >>> context = Context("My Suite", db)
           >>> key = ResultKey(date.today(), {"env": "prod"})
           >>> suite.build_graph(context, key)
       """
       # ... rest of the method stays the same
   ```

2. **Update the internal call in run() method** (around line 370):
   ```python
   # OLD: self.collect(self._context, key)
   # NEW:
   logger.info("Building dependency graph...")
   self.build_graph(self._context, key)
   ```

**Run tests**:
```bash
uv run pytest tests/test_api.py -v
```

**Commit**:
```bash
git add src/dqx/api.py
git commit -m "feat: rename collect() to build_graph() in VerificationSuite"
```

### Task 6: Update All Test References

**Files to update**: Run this search to find all occurrences:
```bash
grep -r "\.collect(" tests/ --include="*.py" | grep -v ".collect_" | head -20
```

**Key files with collect() calls**:
1. `tests/test_api.py` - Multiple occurrences
2. `tests/test_evaluator_validation.py` - 1 occurrence
3. `tests/test_dataset_validator_integration.py` - 3 occurrences
4. `tests/test_api_validation_integration.py` - 3 occurrences

**For each file**, replace:
```python
# OLD: suite.collect(...)
# NEW: suite.build_graph(...)
```

**Example for tests/test_api.py**:
```python
# Find lines like:
suite.collect(context, key=key)
# Replace with:
suite.build_graph(context, key=key)
```

**Run tests after each file update**:
```bash
uv run pytest tests/test_api.py -v
uv run pytest tests/test_evaluator_validation.py -v
uv run pytest tests/test_dataset_validator_integration.py -v
uv run pytest tests/test_api_validation_integration.py -v
```

**Commit after all test updates**:
```bash
git add tests/
git commit -m "test: update all test references from collect() to build_graph()"
```

### Task 7: Update Documentation

**Files to update**:

1. **README.md**:
   Find:
   ```python
   context = suite.collect(key)
   ```
   Replace with:
   ```python
   # Build the dependency graph
   suite.build_graph(context, key)
   ```

2. **docs/dataset_validation_guide.md**:
   - Find references to `suite.collect()` and update to `suite.build_graph()`
   - Remove any references to `suite.validate()` method

3. **Example files** in `examples/` directory - check if any use collect()

**Commit**:
```bash
git add README.md docs/ examples/
git commit -m "docs: update documentation for collect() to build_graph() rename"
```

### Task 8: Remove validate() Method and Update Tests

**Why**: The validate() method is dead code - never used outside of tests. Removing it simplifies the API.

**File**: `src/dqx/api.py`

**Remove the validate() method** (around line 320-335):
```python
# DELETE THIS ENTIRE METHOD:
def validate(self) -> ValidationReport:
    """
    Explicitly validate the suite configuration.

    Returns:
        ValidationReport containing any issues found
    """
    # Create temporary context to collect checks
    temp_context = Context(suite=self._name, db=self.provider._db)

    # Execute all checks to build graph
    for check_fn in self._checks:
        check_fn(self.provider, temp_context)

    # Run validation on the graph using the same provider that was used to register symbols
    return self._validator.validate(temp_context._graph, self.provider)
```

**Update test files that use validate()**:

1. **tests/test_api_validation_integration.py**:
   - Remove or update tests that call `suite.validate()`
   - These tests should instead test validation through `build_graph()` or `run()`

2. **tests/test_dataset_validator_integration.py**:
   - Update tests to use internal validation through `build_graph()` instead

**Search for validate() usage**:
```bash
grep -r "suite.validate()" tests/ --include="*.py"
```

**Run tests to ensure nothing breaks**:
```bash
uv run pytest tests/test_api.py -v
uv run pytest tests/test_api_validation_integration.py -v
uv run pytest tests/test_dataset_validator_integration.py -v
```

**Commit**:
```bash
git add src/dqx/api.py tests/
git commit -m "refactor: remove unused validate() method from VerificationSuite"
```

### Task 9: Final Testing and Validation

1. **Run all tests**:
   ```bash
   uv run pytest tests/ -v
   ```

2. **Run type checking**:
   ```bash
   uv run mypy src/dqx/api.py
   ```

3. **Run linting**:
   ```bash
   uv run ruff check src/dqx/api.py
   ```

4. **Run pre-commit hooks**:
   ```bash
   ./bin/run-hooks.sh
   ```

5. **Check test coverage**:
   ```bash
   uv run pytest tests/test_api.py -v --cov=dqx.api
   ```

### Task 10: Update Memory Bank

After all changes are complete, update the memory bank files:

1. **memory-bank/activeContext.md** - Add note about this change
2. **memory-bank/progress.md** - Update with this improvement

## Testing Strategy

### Unit Tests
- Test that `graph` property returns correct Graph instance
- Test that `build_graph()` method exists and works
- Test that `collect()` method no longer exists
- Test that `validate()` method no longer exists
- Test that validation still works through `build_graph()` and `run()`

### Integration Tests
- Run existing test suite to ensure no regressions
- Verify examples still work with new API

### Manual Testing
```python
# Create a simple test script
from dqx.api import VerificationSuite, check, Context
from dqx.orm.repositories import MetricDB
from dqx.common import ResultKey
import datetime

@check(name="Test Check")
def my_check(mp, ctx):
    ctx.assert_that(mp.sum("value")).where(
        name="Sum is positive"
    ).is_positive()

db = MetricDB()
suite = VerificationSuite([my_check], db, "Test")

# Test new graph property
print(suite.graph)  # Should work
print(suite.graph.root.name)  # Should print "Test"

# Test build_graph
context = Context("Test", db)
key = ResultKey(datetime.date.today(), {})
suite.build_graph(context, key)  # Should work

# Validate method should not exist
assert not hasattr(suite, 'validate')
```

## Rollback Plan

If issues arise:
1. Git revert the commits in reverse order
2. The changes are isolated to specific methods, making rollback straightforward

## Success Criteria

1. ✅ All existing tests pass
2. ✅ New graph property works correctly
3. ✅ build_graph() method works as expected
4. ✅ No references to collect() remain in code
5. ✅ Documentation is updated
6. ✅ Type checking passes
7. ✅ Linting passes
8. ✅ Test coverage maintained at 98%+

## Notes for Implementer

1. **Commit Frequently**: After each task, commit your changes
2. **Test Continuously**: Run tests after each change
3. **Read Error Messages**: Python/pytest give good error messages - read them carefully
4. **Use Type Hints**: The codebase uses type hints extensively - maintain them
5. **No Backward Compatibility**: This is a breaking change - don't add deprecation warnings

## Common Pitfalls to Avoid

1. **Don't forget imports**: When adding Graph type hint, import it
2. **Don't break other tests**: Run full test suite, not just new tests
3. **Don't skip TDD**: Write failing tests first, then implement
4. **Don't forget docs**: Update all documentation, not just code
5. **Case sensitivity**: Python is case-sensitive - `collect` vs `Collect`

## Questions You Might Have

**Q: What if I find other uses of collect() not mentioned?**
A: Update them all to build_graph(). Use grep to search thoroughly.

**Q: Should I update error messages that mention "collect"?**
A: Yes, update them to say "build_graph" instead.

**Q: What if tests fail after my changes?**
A: Read the error message carefully. Often it's a simple typo or missing import.

**Q: How do I know if my documentation is good?**
A: If another developer can understand what the method does without reading the code, it's good.

## Summary

This change improves the VerificationSuite API by:
1. Making the graph directly accessible via a property (cleaner than context._graph)
2. Renaming collect() to build_graph() (clearer intent)
3. Removing the unused validate() method (dead code elimination)

The changes are straightforward but require careful attention to update all references throughout the codebase.
