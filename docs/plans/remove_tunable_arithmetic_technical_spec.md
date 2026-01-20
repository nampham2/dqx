# Remove Tunable Arithmetic from DQL Bounds - Technical Specification

## Problem Statement

DQL currently allows arbitrary arithmetic expressions in tunable bounds, including references to other tunables. This creates unnecessary complexity and a parity gap with the Python API.

**Current behavior (to be removed):**
```dql
tunable BASE = 10 bounds [5, 20]
tunable SCALED = 20 bounds [BASE * 2, 100]  # Currently allowed
tunable DERIVED = 20 bounds [BASE + 5, BASE * 10]  # Currently allowed
tunable COMPUTED = 5 bounds [1 + 1, 10 * 2]  # Currently allowed
```

**Problems:**
1. **Complexity**: Requires full SymPy expression evaluation for bounds
2. **Circular dependencies**: Can create complex dependency graphs between tunables
3. **Parity gap**: Python API (`TunableInt`, `TunableFloat`) only accepts numeric literals for bounds
4. **Rarely used**: Only 2 test files use this feature, no production DQL files
5. **Unclear semantics**: When bounds reference other tunables, what happens when those tunables change?
6. **Type inference complexity**: Arithmetic in bounds complicates int vs float type inference

**Impact on tuning algorithms:**
- RL agents cannot reason about bounds that depend on other tunables
- Bounds should be fixed constraints, not dynamic expressions
- Simplifies action space for optimization algorithms

## Architecture Decisions

### Decision 1: Restrict Bounds to Numeric Literals Only

**Rationale:**
- Bounds should be **fixed constraints** that define the search space for optimization
- Dynamic bounds (that depend on other tunables) create a moving target for algorithms
- Simplifies parser, evaluator, and type inference logic
- Aligns with Python API design (which never supported this)

**Alternatives considered:**
1. **Keep arithmetic but disallow tunable references** - Still complex, minimal benefit
2. **Add validation to warn but not error** - Doesn't achieve simplification goal
3. **Support only simple arithmetic (no references)** - Inconsistent, still complex

**Decision:** Remove all arithmetic in bounds and values. Only accept signed numeric literals (`SIGNED_NUMBER` tokens).

### Decision 2: Grammar-Level Enforcement

**Rationale:**
- Catch invalid syntax at parse time (earliest possible)
- Clear error messages with syntax highlighting
- Prevents downstream evaluation errors
- Users get immediate feedback

**Implementation:**
- Grammar rule change from `expr` to `SIGNED_NUMBER` for tunable value and bounds
- Added `SIGNED_NUMBER` token to support negative literals
- Parser generates descriptive syntax errors
- No changes needed in evaluator (already handles literals)

### Decision 3: Breaking Change (Language Simplification)

**Rationale:**
- This is a **language simplification**, not a feature addition
- Only affects 2 test files, no production code
- Better to break now (pre-1.0) than carry technical debt
- Aligns with Python API semantics

**Migration path:**
- Users evaluate arithmetic manually and hardcode the result
- Example: `bounds [BASE * 2, 100]` → `bounds [20, 100]` (if BASE = 10)

**Version impact:**
- Breaking change - requires minor version bump (0.5.x → 0.6.0)
- Update CHANGELOG.md with clear migration guide
- Mark as **BREAKING CHANGE** in commit message

## API Design

### Grammar Changes

**Before:**
```lark
tunable: "tunable" IDENT "=" expr "bounds" "[" expr "," expr "]"
```

**After:**
```lark
tunable: "tunable" IDENT "=" SIGNED_NUMBER "bounds" "[" SIGNED_NUMBER "," SIGNED_NUMBER "]"

SIGNED_NUMBER: /-?[0-9]+(\.[0-9]+)?/
```

**Key changes:**
- Tunable value and bounds now use `SIGNED_NUMBER` token (signed numeric literals only)
- No longer accepts percentages, arithmetic expressions, or tunable references
- Supports negative values: `tunable OFFSET = -10 bounds [-20, 0]`
- Bounds are now syntactically restricted at grammar level

### Parser Changes

**Module:** `src/dqx/dql/parser.py`

**Current implementation:**
```python
def tunable(self, tree: Any) -> Tunable:
    # items: [name, value, min_bound, max_bound]
    # All are Expr objects (can be arithmetic)
    name = items[0]
    value = items[1]
    min_bound = items[2]  # Currently: Expr
    max_bound = items[3]  # Currently: Expr

    return Tunable(
        name=name,
        value=value,
        bounds=(min_bound, max_bound),
        loc=self._loc(tree),
    )
```

**After (no changes needed):**
- Grammar enforces literals, parser receives `NUMBER` or `PERCENT` tokens
- Transformer converts tokens to Expr objects with numeric text
- No parser logic changes required (grammar does the work)

### Evaluator Changes

**Module:** `src/dqx/api.py`

**Current implementation:**
```python
def _build_tunables_from_ast(self, tunables_ast: tuple[Any, ...]) -> dict[str, Tunable[Any]]:
    tunables_dict: dict[str, Tunable[Any]] = {}
    for t in tunables_ast:
        # Evaluate bounds and value using simple expressions
        # Can reference previously defined tunables
        min_val = self._eval_simple_expr(t.bounds[0], tunables_dict)
        max_val = self._eval_simple_expr(t.bounds[1], tunables_dict)
        value = self._eval_simple_expr(t.value, tunables_dict)
        # ...
```

**After (no changes needed):**
- `_eval_simple_expr()` already handles numeric literals correctly
- Will now only receive literals (no arithmetic)
- Simplifies execution but no code changes required

**Note:** The evaluator will be simpler in practice, but no code changes are needed because it already handles the restricted case correctly.

## Data Structures

### AST Node (No changes)

**Module:** `src/dqx/dql/ast.py`

```python
@dataclass(frozen=True)
class Tunable:
    """A tunable constant declaration with required bounds."""
    name: str
    value: Expr  # Can still be arithmetic (can reference other tunables)
    bounds: tuple[Expr, Expr]  # Now guaranteed to be numeric literals
    loc: SourceLocation | None = None
```

**No changes needed:**
- AST structure remains the same
- `bounds` still store `Expr` objects (but will contain only numeric text)
- Type hints and structure unchanged

## Error Messages

### Validation Rules

#### Rule 1: No arithmetic operators in bounds

**Invalid DQL:**
```dql
tunable X = 5 bounds [1 + 1, 10]
```

**Error message:**
```text
DQL Syntax Error at line 2, column 27:
    tunable X = 5 bounds [1 + 1, 10]
                          ^^^^^
Unexpected token: '+' in tunable bounds

Bounds must be numeric literals (numbers or percentages).
Expected: bounds [NUMBER, NUMBER] or bounds [PERCENT, PERCENT]

Example:
    tunable X = 5 bounds [0, 10]
    tunable X = 5% bounds [0%, 10%]
```

#### Rule 2: No tunable references in bounds

**Invalid DQL:**
```dql
tunable BASE = 10 bounds [5, 20]
tunable SCALED = 20 bounds [BASE * 2, 100]
```

**Error message:**
```text
DQL Syntax Error at line 3, column 32:
    tunable SCALED = 20 bounds [BASE * 2, 100]
                                ^^^^^^^^
Unexpected identifier: 'BASE' in tunable bounds

Bounds must be numeric literals (numbers or percentages).
Tunable references are not allowed in bounds.

If you need to use the value of BASE, calculate it manually:
    # If BASE = 10, then BASE * 2 = 20
    tunable SCALED = 20 bounds [20, 100]
```

#### Rule 3: No function calls in bounds

**Invalid DQL:**
```dql
tunable X = 5 bounds [min(1, 2), 10]
```

**Error message:**
```text
DQL Syntax Error at line 2, column 27:
    tunable X = 5 bounds [min(1, 2), 10]
                          ^^^^^^^^^^
Unexpected function call in tunable bounds

Bounds must be numeric literals (numbers or percentages).
Expected: bounds [NUMBER, NUMBER]

Example:
    tunable X = 5 bounds [1, 10]
```

## Integration Points

### Files to Modify

1. **Grammar**: `src/dqx/dql/grammar.lark`
   - Add `bound_value` rule
   - Update `tunable` rule to use `bound_value`

2. **Tests - Update existing arithmetic tests**: `tests/test_dql_arithmetic.py`
   - `test_tunable_with_arithmetic_in_bounds` - Remove (feature no longer supported)
   - `test_tunable_with_complex_arithmetic` - Remove (feature no longer supported)
   - `test_tunable_with_percentage` - Keep (percentages are still valid)
   - `test_tunable_with_invalid_expression` - Update (error message will change)

3. **Tests - Update API tests**: `tests/test_api_dql_tunables.py`
   - `test_arithmetic_in_tunable_bounds` - Remove (feature no longer supported)
   - Keep other tests (they use literals)

4. **DQL files**: All 5 existing DQL files already use literals - no changes needed
   - `tests/dql/banking_transactions.dql` ✓
   - `tests/dql/book_inventory.dql` ✓
   - `tests/dql/book_orders.dql` ✓
   - `tests/dql/video_streaming.dql` ✓
   - `tests/dql/commerce_suite.dql` ✓ (no tunables)

5. **Documentation**: `docs/design/dql-language.md`
   - Update tunables section to clarify bounds must be literals
   - Add migration guide for users with arithmetic in bounds

6. **Changelog**: `docs/changelog.md`
   - Add breaking change entry for 0.6.0

### Dependencies

**No impact on:**
- Python API tunables (`TunableInt`, `TunableFloat`) - already only accept literals
- Tunable value expressions - can still reference other tunables and use arithmetic
- Assertion conditions - can still use arbitrary expressions
- Metric expressions - no changes

**Affected components:**
- DQL parser (grammar.lark)
- DQL test files (2 files need updates)
- Documentation (1 file needs clarification)

## Performance Considerations

**Performance improvements:**
- **Simpler parsing**: Grammar-level restriction eliminates expression parsing for bounds
- **Faster validation**: No SymPy evaluation needed for bounds
- **Reduced complexity**: Type inference for bounds is trivial (int vs float)

**No performance regressions:**
- Tunable value expressions still support full arithmetic
- Assertion expressions unchanged
- Metric expressions unchanged

**Estimated impact:**
- Negligible runtime impact (bounds evaluated once at suite creation)
- Significant reduction in code complexity

## Non-Goals

**Out of scope for this change:**

1. **String tunables**: DQL never supported string bounds, not adding them
2. **Tunable value restrictions**: Value can still use arithmetic and reference other tunables
3. **Python API changes**: Python API already follows this restriction
4. **Percentage validation**: Grammar already handles `PERCENT` tokens, no changes needed
5. **Bounds validation logic**: Runtime validation (min <= max, value in bounds) unchanged

## Breaking Change Impact Assessment

### Affected Users

**Production code:** Zero impact
- No production DQL files use arithmetic in bounds (verified via codebase search)

**Test code:** Minimal impact
- 2 test files need updates (3 test methods total)
- Test updates are straightforward (remove or update)

**Documentation:** Minor updates
- 1 documentation file needs clarification
- Add migration guide section

### Migration Guide

#### For users with arithmetic in bounds

#### Step 1: Identify affected DQL files
```bash
# Search for tunables with non-literal bounds
grep -E "tunable.*bounds.*[\+\-\*/]" **/*.dql
```

#### Step 2: Evaluate arithmetic manually
```dql
# Before (DQX 0.5.x)
tunable BASE = 10 bounds [5, 20]
tunable SCALED = 20 bounds [BASE * 2, 100]

# After (DQX 0.6.0)
tunable BASE = 10 bounds [5, 20]
tunable SCALED = 20 bounds [20, 100]  # Calculated: 10 * 2 = 20
```

#### Step 3: Update DQL files
- Replace arithmetic expressions with evaluated results
- Use comments to document the calculation

#### Step 4: Test updated files
```python
suite = VerificationSuite(dql="path/to/updated.dql", db=db)
params = suite.get_tunable_params()
# Verify bounds are correct
```

### Rollback Strategy

**If issues arise:**
1. Revert grammar changes (restore `expr` in bounds)
2. Revert test updates
3. Release hotfix version 0.5.x

**Confidence:** High - change is isolated to grammar, no runtime logic changes

## Verification Strategy

### Test Coverage

**Unit tests:**
1. Grammar parsing with literal bounds (existing tests cover this)
2. Grammar parsing with arithmetic bounds (should fail)
3. Error messages for invalid bounds syntax

**Integration tests:**
1. All existing DQL files still parse correctly (they use literals)
2. Suite creation with updated grammar
3. Tunable evaluation with literal bounds

**Error message tests:**
1. Arithmetic operator in bounds
2. Tunable reference in bounds
3. Function call in bounds

### Manual Verification

**Checklist:**
- [ ] All 5 existing DQL files parse successfully
- [ ] Error messages are clear and actionable
- [ ] Documentation examples parse correctly
- [ ] Python API tunables still work (no impact expected)
- [ ] Type inference (int vs float) still works correctly

## Timeline Estimate

**Total effort:** 2-4 hours

- Phase 1: Grammar changes (30 min)
- Phase 2: Test updates (1 hour)
- Phase 3: Error message validation (30 min)
- Phase 4: Documentation updates (1 hour)
- Phase 5: Final verification (30 min)

## Success Criteria

- [ ] Grammar only accepts `NUMBER` or `PERCENT` in bounds
- [ ] Clear error messages when arithmetic used in bounds
- [ ] All existing DQL files parse successfully
- [ ] All tests pass (after removing arithmetic tests)
- [ ] Documentation updated with migration guide
- [ ] Zero impact on Python API
- [ ] Zero impact on tunable value expressions
- [ ] Type inference still works correctly
- [ ] 100% test coverage maintained
