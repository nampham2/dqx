# VerificationSuite Graph Improvements Implementation Plan v2

## Overview

This plan describes how to improve the VerificationSuite class by:
1. Adding a public `graph` property with defensive error handling
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

### Analysis Confirmation
- **validate() is dead code**: Only appears in test files, never in production code
- **6 occurrences of _context._graph**: Confirmed in src/dqx/api.py
- **collect() method naming**: "build_graph" better describes its actual purpose
- **Breaking change approach**: Appropriate given project rules state "No backward compatibility is needed"

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

## Implementation Tasks - Priority Structure

### 🔴 Must-Have Tasks (Core Functionality)

#### Task 0: Pre-implementation Setup

**Why**: Ensure clean starting state and establish baseline metrics.

**Steps**:
```bash
# Ensure clean working directory
git status

# If uncommitted changes exist, stash them
git stash save "WIP: before graph improvements"

# Record baseline test coverage
uv run pytest tests/test_api.py -v --cov=dqx.api --cov-report=term-missing > coverage_baseline.txt

# Create feature branch
git checkout -b feature/verification-suite-graph-improvements
```

**Commit**: No commit needed for setup

#### Task 1: Write Failing Tests for Graph Property

**Why**: Following TDD, we write tests first to define expected behavior.

**File**: `tests/test_api.py`

**Add this test**:
```python
def test_verification_suite_graph_property(
    simple_check: CheckProducer, db: MetricDB
) -> None:
    """Test that VerificationSuite exposes graph property with proper error handling."""
    suite = VerificationSuite([simple_check], db, "Test Suite")

    # Graph should be accessible
    assert hasattr(suite, 'graph')

    # Should raise error before build_graph is called
    with pytest.raises(DQXError, match="Graph not built yet"):
        _ = suite.graph

    # After building graph, should work
    context = Context("Test Suite", db)
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.build_graph(context, key)  # Using new name already

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
git commit -m "test: add failing test for VerificationSuite.graph property with error handling"
```

#### Task 2: Implement Defensive Graph Property

**File**: `src/dqx/api.py`

**Add imports at top if not present**:
```python
from dqx.graph.traversal import Graph
```

**Find the VerificationSuite class** (around line 300) and add this property after the `provider` property:

```python
@property
def graph(self) -> Graph:
    """
    Read-only access to the dependency graph.

    This property provides direct access to the internal graph structure,
    which contains the root node and all registered checks. The graph
    is built when build_graph() is called.

    Returns:
        Graph instance with the root node and all registered checks

    Raises:
        DQXError: If VerificationSuite not initialized or graph not built yet

    Example:
        >>> suite = VerificationSuite(checks, db, "My Suite")
        >>> # Accessing before build_graph() raises error
        >>> try:
        ...     graph = suite.graph
        ... except DQXError:
        ...     print("Graph not built yet")
        >>>
        >>> # Build the graph first
        >>> suite.build_graph(context, key)
        >>> # Now access works
        >>> print(suite.graph.root.name)  # "My Suite"
        >>> for check in suite.graph.root.children:
        ...     print(check.name)
    """
    if not hasattr(self, '_context'):
        raise DQXError("VerificationSuite not initialized")

    # Check if graph has been built (has children or was run)
    if not self._context._graph.root.children and not self.is_evaluated:
        raise DQXError("Graph not built yet. Call build_graph() first.")

    return self._context._graph
```

**Run test to confirm it passes**:
```bash
uv run pytest tests/test_api.py::test_verification_suite_graph_property -v
```

**Commit**:
```bash
git add src/dqx/api.py
git commit -m "feat: add defensive graph property to VerificationSuite"
```

#### Task 3: Update Internal Graph References

**File**: `src/dqx/api.py`

**Replace these occurrences** of `self._context._graph`:

1. In `run()` method (~line 380):
   ```python
   # OLD: self._context._graph.impute_datasets(...)
   # NEW:
   self.graph.impute_datasets(list(datasources.keys()), self._context.provider)
   ```

2. In `run()` method (~line 390):
   ```python
   # OLD: self._context._graph.bfs(evaluator)
   # NEW:
   self.graph.bfs(evaluator)
   ```

3. In `collect_results()` method (~line 430):
   ```python
   # OLD: for assertion in self._context._graph.assertions():
   # NEW:
   for assertion in self.graph.assertions():
   ```

**Note**: Keep `context._graph` references that use parameter context unchanged.

**Run all tests to ensure nothing broke**:
```bash
uv run pytest tests/test_api.py -v
```

**Commit**:
```bash
git add src/dqx/api.py
git commit -m "refactor: use graph property instead of _context._graph"
```

#### Task 4: Write Tests for Renaming collect to build_graph

**File**: `tests/test_api.py`

**Add test to verify build_graph works and collect is gone**:
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

#### Task 5: Rename collect to build_graph

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

#### Task 6: Update All Code References

**First, search for error messages mentioning collect**:
```bash
grep -r "collect" src/ --include="*.py" | grep -i "error\|exception\|raise"
```

**Update test files** - Run this search to find all occurrences:
```bash
grep -r "\.collect(" tests/ --include="*.py" | grep -v ".collect_" | head -20
```

**Key files to update**:
1. `tests/test_api.py` - Multiple occurrences
2. `tests/test_evaluator_validation.py` - 1 occurrence
3. `tests/test_dataset_validator_integration.py` - 3 occurrences
4. `tests/test_api_validation_integration.py` - 3 occurrences
5. `tests/e2e/test_api_e2e.py` - Check for any usage

**For each file**, replace:
```python
# OLD: suite.collect(...)
# NEW: suite.build_graph(...)
```

**Run tests after each file update**:
```bash
uv run pytest tests/test_api.py -v
uv run pytest tests/test_evaluator_validation.py -v
uv run pytest tests/test_dataset_validator_integration.py -v
uv run pytest tests/test_api_validation_integration.py -v
uv run pytest tests/e2e/test_api_e2e.py -v
```

**Commit after all test updates**:
```bash
git add tests/
git commit -m "test: update all test references from collect() to build_graph()"
```

#### Task 8: Remove validate() Method and Update Tests

**Why**: The validate() method is dead code - never used outside of tests. Removing it simplifies the API.

**Step 1: Analyze validate() test usage**:
```bash
# Find all validate() usage in tests
grep -n "suite.validate()" tests/ --include="*.py" -B2 -A2
```

**Step 2: Update test files**

For tests that specifically test validation behavior:
- Convert them to test validation through `build_graph()`
- The `build_graph()` method already calls validation internally

For redundant tests:
- Remove them entirely

**Step 3: Remove validate() from src/dqx/api.py** (around line 320-335):
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

**Run tests to ensure nothing breaks**:
```bash
uv run pytest tests/test_api.py -v
uv run pytest tests/test_api_validation_integration.py -v
uv run pytest tests/test_dataset_validator_integration.py -v
```

**Commit**:
```bash
git add src/dqx/api.py tests/
git commit -m "refactor: remove unused validate() method and update tests"
```

### 🟡 Nice-to-Have Tasks (Documentation & Validation)

#### Task 7: Update Documentation

**Note**: Focus only on current documentation, not historical docs.

**Files to update**:

1. **README.md**:
   Search for `collect(` and update to `build_graph(`

2. **docs/dataset_validation_guide.md**:
   - Update references from `suite.collect()` to `suite.build_graph()`
   - Remove any references to `suite.validate()` method

3. **Example files** - check current examples only:
   ```bash
   grep -r "\.collect(" examples/ --include="*.py"
   ```

**Commit**:
```bash
git add README.md docs/ examples/
git commit -m "docs: update documentation for collect() to build_graph() rename"
```

#### Task 9: Final Testing and Validation

1. **Verify no collect() references remain** (except collect_results/collect_symbols):
   ```bash
   grep -r "\.collect(" . --include="*.py" --include="*.md" | grep -v "collect_results\|collect_symbols\|git\|docs/plans"
   ```

2. **Run all tests**:
   ```bash
   uv run pytest tests/ -v
   ```

3. **Compare coverage with baseline**:
   ```bash
   uv run pytest tests/test_api.py -v --cov=dqx.api --cov-report=term-missing > coverage_final.txt
   diff coverage_baseline.txt coverage_final.txt
   ```

4. **Run type checking**:
   ```bash
   uv run mypy src/dqx/api.py
   ```

5. **Run linting**:
   ```bash
   uv run ruff check src/dqx/api.py
   ```

6. **Run pre-commit hooks**:
   ```bash
   ./bin/run-hooks.sh
   ```

**Commit if any fixes needed**:
```bash
git add .
git commit -m "fix: address linting and type checking issues"
```

#### Task 10: Update Memory Bank

After all changes are complete, update the memory bank files:

1. **memory-bank/activeContext.md**:
   Add note about API change:
   ```markdown
   ## Recent API Changes
   - VerificationSuite.collect() renamed to build_graph() for clarity
   - VerificationSuite now exposes graph property for direct access
   - VerificationSuite.validate() removed (was unused)
   ```

2. **memory-bank/systemPatterns.md**:
   Document the defensive property pattern:
   ```markdown
   ## Defensive Property Pattern
   When exposing internal state via properties, include error handling:
   - Check initialization state
   - Verify preconditions are met
   - Provide clear error messages
   See: VerificationSuite.graph property implementation
   ```

3. **memory-bank/progress.md**:
   Record the breaking change

**Commit**:
```bash
git add memory-bank/
git commit -m "docs: update memory bank with API changes"
```

## Testing Strategy

### Unit Tests
- Test that `graph` property returns correct Graph instance
- Test that `graph` property raises errors appropriately
- Test that `build_graph()` method exists and works
- Test that `collect()` method no longer exists
- Test that `validate()` method no longer exists
- Test that validation still works through `build_graph()` and `run()`

### Integration Tests
- Run existing test suite to ensure no regressions
- Verify examples still work with new API

### Manual Testing Script
```python
# Create a simple test script
from dqx.api import VerificationSuite, check, Context
from dqx.orm.repositories import MetricDB
from dqx.common import ResultKey, DQXError
import datetime

@check(name="Test Check")
def my_check(mp, ctx):
    ctx.assert_that(mp.sum("value")).where(
        name="Sum is positive"
    ).is_positive()

db = MetricDB()
suite = VerificationSuite([my_check], db, "Test")

# Test new graph property - should fail before build_graph
try:
    graph = suite.graph
    print("ERROR: Should have raised DQXError")
except DQXError as e:
    print(f"✓ Expected error: {e}")

# Test build_graph
context = Context("Test", db)
key = ResultKey(datetime.date.today(), {})
suite.build_graph(context, key)  # Should work

# Now graph property should work
print(f"✓ Graph root name: {suite.graph.root.name}")

# Validate method should not exist
assert not hasattr(suite, 'validate'), "validate() should be removed"
print("✓ validate() method successfully removed")

# collect method should not exist
assert not hasattr(suite, 'collect'), "collect() should be removed"
print("✓ collect() method successfully removed")
```

## Success Criteria

### Must-Have Tasks
- ✅ Pre-implementation setup completed
- ✅ Graph property implemented with defensive error handling
- ✅ All internal _context._graph references updated
- ✅ collect() renamed to build_graph()
- ✅ All code references updated
- ✅ validate() method removed
- ✅ All tests passing

### Nice-to-Have Tasks
- ✅ Current documentation updated
- ✅ No stray references to old methods
- ✅ Type checking passes
- ✅ Linting passes
- ✅ Test coverage maintained or improved
- ✅ Memory bank updated

## Common Pitfalls to Avoid

1. **Don't forget imports**: When adding Graph type hint, import it
2. **Don't break other tests**: Run full test suite, not just new tests
3. **Don't skip TDD**: Write failing tests first, then implement
4. **Test the error cases**: Ensure graph property errors are tested
5. **Case sensitivity**: Python is case-sensitive - `collect` vs `Collect`

## Summary

This v2 plan incorporates all feedback to create a more robust implementation:
1. **Defensive graph property** prevents accessing unbuilt graphs
2. **Clear priority structure** separates must-have from nice-to-have tasks
3. **Better test migration strategy** for validate() removal
4. **Pre-implementation setup** ensures clean starting state
5. **Comprehensive validation** includes coverage comparison

The changes improve the VerificationSuite API by making it clearer, safer, and simpler.
