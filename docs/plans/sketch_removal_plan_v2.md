# Sketch Removal Implementation Plan v2

## Overview
This plan details the complete removal of sketch metrics functionality from the DQX project. Sketches (using the `datasketches` library) were used for approximate cardinality calculations but impose scalability constraints. This implementation will remove all traces of sketch functionality from the codebase.

**Version 2 Changes**: Reordered phases to maintain working codebase at each commit, enhanced validation checks, and improved integration testing between phases.

## Background
- Sketch metrics use the `datasketches` library for probabilistic data structures
- Currently only used for `ApproxCardinality` metric
- Removal is a breaking change but backward compatibility is NOT required
- This simplifies the architecture by focusing solely on SQL-based metrics
- Clean removal approach without deprecation warnings

## Task Groups

### Task Group 1: Remove Sketch Tests (Phase 1)
**Goal**: Remove all test cases related to sketch functionality to prepare for core removal.

#### Task 1.1: Remove Sketch State Tests
**File**: `tests/test_states.py`

Remove all test functions and fixtures related to `CardinalitySketch`:
- Remove imports: `from datasketches import cpc_sketch`
- Remove fixtures: `one_cpc_sketch`, `two_cpc_sketch`
- Remove test functions:
  - `test_cardinality_sketch_initialization`
  - `test_cardinality_identity`
  - `test_cardinality_sketch_serialization_deserialization`
  - `test_cardinality_sketch_merge`
  - `test_cardinality_sketch_identity_merge`
  - `test_cardinality_sketch_fit`
  - `test_cardinality_sketch_copy`
- Remove `CardinalitySketch` references from `test_state_equality`

#### Task 1.2: Remove Sketch Operations Tests
**File**: `tests/test_ops.py`

Remove all `ApproxCardinality` and `SketchOp` related tests:
- Remove from imports: `SketchOp`
- Remove test functions:
  - `test_approx_cardinality`
  - `test_approx_cardinality_not_equal`
  - `test_approx_cardinality_string_repr`
  - `test_approx_cardinality_sketch_ops`
- Update `test_op_hash_and_equality` to remove `ApproxCardinality` cases
- Remove `SketchOp` assertions from other tests

#### Task 1.3: Remove Sketch Specs Tests
**File**: `tests/test_specs.py`

Remove the entire `TestApproxCardinality` class and all its methods:
- Remove import: `ApproxCardinality`
- Remove from `test_metric_spec_types` list
- Remove from `test_metric_spec_equality_different_types`
- Remove from `test_registry_access` and `test_registry_content`
- Remove from parametrized tests in `test_metric_type_property`

**File**: `tests/test_specs_str.py`
- Remove import and test case for `ApproxCardinality`

#### Task 1.4: Remove from Provider and Analyzer Tests
**File**: `tests/test_provider.py`
- Remove `test_approx_cardinality` method and its mock patch

**File**: `tests/test_analyzer.py`
- Remove imports: `FakeSketchState`, `SketchOp`, `SketchState`
- Remove `FakeSketchState` class entirely
- Remove test methods:
  - `test_analyze_sketch_ops_empty`
  - `test_analyze_sketch_ops`
- Update analyzer tests to remove sketch ops references

#### Task 1.5: Remove from E2E Tests
**File**: `tests/e2e/test_api_e2e.py`
- Remove `sketch_check` function entirely
- Remove `sketch_check` from the `checks` list

**Integration Check**:
```bash
uv run pytest tests/ -v
```

**Commit after completion**: `test: remove all sketch-related test cases`

### Task Group 2: Remove Analyzer Support (Phase 2)
**Goal**: Remove sketch analysis functionality from the analyzer before removing the types it depends on.

#### Task 2.1: Remove Sketch Analysis Function
**File**: `src/dqx/analyzer.py`

Remove the entire `analyze_sketch_ops` function:
```python
# Remove this function:
def analyze_sketch_ops(ds: T, ops: Sequence[SketchOp], batch_size: int, nominal_date: datetime.date) -> None:
    # ... entire function content
```

Also remove:
- Import of `SketchOp` from ops
- Import of any sketch-related types

#### Task 2.2: Update Analyzer Run Method
**File**: `src/dqx/analyzer.py`

In the `Analyzer.run` method:
- Remove the section that filters and processes sketch ops:
```python
# Remove these lines:
sketch_ops = [op for op in all_ops if isinstance(op, SketchOp)]
analyze_sketch_ops(ds, sketch_ops, batch_size=100_000, nominal_date=key.yyyy_mm_dd)
```

**Integration Check**:
```bash
uv run mypy src/dqx
uv run pytest tests/ -v
```

**Commit after completion**: `refactor: remove sketch analysis from analyzer`

### Task Group 3: Remove Public API Components (Phase 3)
**Goal**: Remove sketch functionality from the public-facing API and update documentation.

#### Task 3.1: Remove from Specs Module
**File**: `src/dqx/specs.py`

Remove `ApproxCardinality` class:
```python
# Remove this entire class:
class ApproxCardinality(MetricSpec[states.CardinalitySketch]):
    # ... entire class content
```

Also remove:
- Import of `CardinalitySketch` from states
- `ApproxCardinality` from `__all__` export list
- Update the TODO comments about sketch inconsistency

#### Task 3.2: Remove from Provider Module
**File**: `src/dqx/provider.py`

Remove the `approx_cardinality` method:
```python
# Remove this method:
def approx_cardinality(self, column: str, ...) -> sp.Expr:
    # ... entire method content
```

#### Task 3.3: Update Documentation (Moved from Phase 6)
**File**: `README.md`

Remove or update:
1. In "Three Things to Know" section, remove the line:
   ```python
   mp.approx_cardinality("id")  # → 10,847
   ```

2. In "Available Metrics" section, remove:
   ```python
   mp.approx_cardinality("column")  # Distinct values (fast, approximate)
   ```

3. Search for any other `approx_cardinality` usage in examples and remove/replace them.

**File**: `docs/design.md`

In the "Supported Metrics" table, remove the row:
```
| Approx. Cardinality | `approx_cardinality` | Approximate distinct count |
```

#### Task 3.4: Update Module Exports
Ensure all `__all__` lists in affected modules no longer reference removed components.

**Integration Check**:
```bash
uv run mypy src/dqx
uv run pytest tests/ -v
```

**Commit after completion**: `feat!: remove ApproxCardinality from public API and docs`

### Task Group 4: Remove Core Operations (Phase 4)
**Goal**: Remove sketch operations and protocols from the ops module.

#### Task 4.1: Remove Sketch Protocol and Operations
**File**: `src/dqx/ops.py`

Remove:
1. Import statements:
   - `from dqx.states import CardinalitySketch, SketchState`

2. Update `OpsType` literal:
```python
# Change from:
OpsType = Literal["sql", "sketch"]
# To:
OpsType = Literal["sql"]
```

3. Check and update TypeVar bound:
```python
# Verify all usages before changing:
# Search for: TypeVar("T", bound=float | SketchState)
# Change to:
T = TypeVar("T", bound=float)
```

**Important**: Before modifying the TypeVar, search for all its usages:
```bash
grep -r "TypeVar.*bound.*SketchState" src/ tests/
grep -r "\[T\]" src/ tests/ --include="*.py"
```

4. Remove entire `SketchOp` protocol:
```python
# Remove this protocol:
@runtime_checkable
class SketchOp(Op[T], Protocol):
    # ... entire protocol content
```

5. Remove entire `ApproxCardinality` class:
```python
# Remove this class:
class ApproxCardinality(OpValueMixin[states.CardinalitySketch], SketchOp[states.CardinalitySketch]):
    # ... entire class content
```

**Integration Check**:
```bash
uv run mypy src/dqx
uv run pytest tests/ -v
```

**Commit after completion**: `refactor: remove SketchOp and ApproxCardinality from ops`

### Task Group 5: Remove State Components and Dependencies (Phase 5)
**Goal**: Remove sketch state management and the datasketches dependency.

#### Task 5.1: Remove Sketch States
**File**: `src/dqx/states.py`

Remove:
1. Import statement:
   - `from datasketches import cpc_sketch, cpc_union`

2. Remove entire `SketchState` protocol:
```python
# Remove this protocol:
@runtime_checkable
class SketchState(State, Protocol):
    def fit(self, batch: pa.RecordBatch) -> None: ...
```

3. Remove entire `CardinalitySketch` class:
```python
# Remove this class:
class CardinalitySketch(SketchState):
    # ... entire class content
```

#### Task 5.2: Remove Datasketches Dependency
**File**: `pyproject.toml`

1. Remove from dependencies:
```toml
# Remove this line:
"datasketches>=5.2.0",
```

2. Remove from mypy overrides:
```toml
# Remove "datasketches.*" from the module list:
module = ["msgpack.*", "datasketches.*", "pyarrow.*", "sympy.*", "sqlparse.*"]
# Change to:
module = ["msgpack.*", "pyarrow.*", "sympy.*", "sqlparse.*"]
```

**Integration Check**:
```bash
uv sync
uv run mypy src/dqx
uv run pytest tests/ -v
```

**Commit after completion**: `build: remove datasketches dependency and sketch states`

### Task Group 6: Final Validation (Phase 6)
**Goal**: Ensure complete removal and maintain code quality.

#### Task 6.1: Search for Remaining References
Run comprehensive searches to ensure no sketch references remain:

```bash
# Search for any remaining sketch references
grep -r "sketch" src/ tests/ --include="*.py" -i
grep -r "approx_cardinality" src/ tests/ docs/ --include="*.py" --include="*.md"
grep -r "datasketches" . --include="*.py" --include="*.toml"
grep -r "CardinalitySketch" src/ tests/ --include="*.py"
grep -r "SketchOp" src/ tests/ --include="*.py"

# String literal references
grep -r "approx_cardinality\|sketch" src/ tests/ --include="*.py" | grep -E "['\"]"

# Dynamic imports/getattr
grep -r "getattr\|__import__\|importlib" src/ tests/ --include="*.py" -B2 -A2 | grep -i sketch

# Registry or factory patterns
grep -r "register\|registry" src/ tests/ --include="*.py" | grep -i sketch
```

Fix any remaining references found.

#### Task 6.2: Run Quality Checks
```bash
# Run type checking
uv run mypy src/dqx tests/

# Run linting and auto-fix
uv run ruff check --fix

# Run all tests with coverage
uv run pytest tests/ -v --cov=dqx --cov-report=term-missing

# Run pre-commit hooks
./bin/run-hooks.sh
```

Ensure:
- All tests pass
- 100% code coverage is maintained
- No type errors
- No linting issues

**Final commit**: `chore: final cleanup after sketch removal`

## Testing Strategy

After each task group:
1. Run `uv run pytest tests/ -v` to ensure no regressions
2. Run `uv run mypy src/dqx` to check types
3. Run `uv run ruff check` to ensure code style
4. Verify the codebase is in a working state before committing

## Enhanced Validation Checklist

- [ ] All sketch-related code removed
- [ ] No `datasketches` dependency remains
- [ ] All tests pass with 100% coverage
- [ ] No type checking or linting errors
- [ ] Documentation is updated
- [ ] No performance regression in remaining functionality
- [ ] All TypeVar usages verified and updated
- [ ] No string literal references to sketch functionality
- [ ] No dynamic imports of sketch code
- [ ] Each phase commit leaves codebase in working state

## Rollback Plan

If issues arise during implementation:
1. Each task group is committed separately for easy rollback
2. Use `git revert` for the specific problematic commit
3. The modular approach allows partial rollback if needed
4. Reordered phases ensure no broken intermediate states

## Success Criteria

1. All sketch-related code is removed
2. No `datasketches` dependency remains
3. All tests pass with 100% coverage at each phase
4. No type checking or linting errors
5. Documentation is updated with API removal
6. No performance regression in remaining functionality
7. Clean git history with working codebase at each commit

## Notes for Implementation

- Follow TDD principles: update tests first, then implementation
- Make frequent commits as specified
- Each task group should be independently functional
- Use the project's code style and conventions
- No backward compatibility is required, so clean removal is preferred
- Run integration checks between each phase
- Verify TypeVar changes don't break generic type handling
