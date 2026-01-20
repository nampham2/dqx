# Remove Tunable Arithmetic from DQL Bounds - Implementation Guide

## Overview

This guide provides step-by-step instructions for removing arithmetic expression support from DQL tunable bounds. The change simplifies the language by restricting bounds to numeric literals only, aligning with the Python API and improving clarity for optimization algorithms.

**Goal:** Restrict tunable bounds to `NUMBER` or `PERCENT` tokens only (no arithmetic, no tunable references).

**Approach:** Grammar-level enforcement with clear error messages.

## Prerequisites

**Files to read before starting:**
- `src/dqx/dql/grammar.lark` - Current tunable grammar definition
- `src/dqx/dql/ast.py` - AST node structures (Tunable class)
- `src/dqx/api.py` - Tunable evaluation logic (`_build_tunables_from_ast`, `_eval_simple_expr`)
- `tests/test_dql_arithmetic.py` - Tests that use arithmetic in bounds
- `tests/test_api_dql_tunables.py` - API integration tests for tunables

**Related components:**
- **DQL Parser**: Transforms grammar into AST nodes
- **Tunable Evaluator**: Evaluates expressions in `VerificationSuite._build_tunables_from_ast`
- **Python Tunables**: `TunableInt`, `TunableFloat` already follow this restriction

## Phase Breakdown

### Phase 1: Update Grammar and Add Validation Tests

**Goal:** Modify grammar to restrict bounds to literals and add tests that verify the restriction.

**Duration estimate:** 1-2 hours

**Files to create:**
- `tests/dql/test_tunable_bounds_validation.py` - New test file for bounds validation

**Files to modify:**
- `src/dqx/dql/grammar.lark` - Add `bound_value` rule

**Tests to write** (test names only):
```python
class TestTunableBoundsValidation:
    def test_tunable_with_numeric_literal_bounds(): ...
    def test_tunable_with_percent_literal_bounds(): ...
    def test_tunable_with_mixed_numeric_percent_bounds_fails(): ...
    def test_tunable_with_arithmetic_in_bounds_fails(): ...
    def test_tunable_with_tunable_reference_in_bounds_fails(): ...
    def test_tunable_with_function_call_in_bounds_fails(): ...
    def test_tunable_with_negative_literal_bounds(): ...
    def test_tunable_with_float_literal_bounds(): ...
    def test_tunable_with_integer_literal_bounds(): ...
    def test_error_message_for_arithmetic_in_bounds_is_clear(): ...
    def test_error_message_for_tunable_reference_in_bounds_is_clear(): ...
```

**Implementation notes:**
- Grammar change is straightforward: replace `expr` with `bound_value` in tunable rule
- New rule: `bound_value: NUMBER | PERCENT`
- Tests should verify both success cases (literals) and failure cases (arithmetic)
- Error messages should be validated (clear and actionable)

**Grammar change:**
```lark
# Before
tunable: "tunable" IDENT "=" expr "bounds" "[" expr "," expr "]"

# After
tunable: "tunable" IDENT "=" expr "bounds" "[" bound_value "," bound_value "]"

bound_value: NUMBER | PERCENT
```

**Success criteria:**
- [ ] Grammar updated with `bound_value` rule
- [ ] All validation tests pass
- [ ] Error messages are clear and actionable
- [ ] Coverage: 100% for new test file
- [ ] Pre-commit hooks: passing

**Commit message:** `feat(dql): restrict tunable bounds to numeric literals only`

---

### Phase 2: Update Existing Tests

**Goal:** Remove or update tests that use arithmetic in tunable bounds.

**Duration estimate:** 1 hour

**Files to modify:**
- `tests/test_dql_arithmetic.py` - Remove tests that use arithmetic in bounds
- `tests/test_api_dql_tunables.py` - Remove test for arithmetic in bounds

**Tests to remove:**
```python
# tests/test_dql_arithmetic.py
def test_tunable_with_arithmetic_in_bounds(): ...  # REMOVE
def test_tunable_with_complex_arithmetic(): ...     # REMOVE

# tests/test_api_dql_tunables.py
def test_arithmetic_in_tunable_bounds(): ...        # REMOVE
```

**Tests to keep (unchanged):**
```python
# tests/test_dql_arithmetic.py
def test_tunable_with_percentage(): ...             # KEEP (uses literals)
def test_tunable_with_invalid_expression(): ...     # KEEP (tests value expression)
```

**Implementation notes:**
- Remove entire test methods (not just assertions)
- Update test file docstrings if they reference removed features
- Verify test file still has meaningful coverage of arithmetic (in value expressions)
- All other tunable tests should pass unchanged (they use literals)

**Success criteria:**
- [ ] All tests using arithmetic in bounds removed
- [ ] All remaining tests pass
- [ ] Test file docstrings updated
- [ ] Coverage: 100% maintained for src/dqx/dql/
- [ ] Pre-commit hooks: passing

**Commit message:** `test(dql): remove tests for arithmetic in tunable bounds`

---

### Phase 3: Verify All DQL Files Parse Successfully

**Goal:** Ensure all existing DQL files (test fixtures and examples) still parse correctly.

**Duration estimate:** 30 minutes

**Files to verify:**
- `tests/dql/banking_transactions.dql` - Uses literals (should pass)
- `tests/dql/book_inventory.dql` - Uses literals (should pass)
- `tests/dql/book_orders.dql` - Uses literals (should pass)
- `tests/dql/video_streaming.dql` - Uses literals (should pass)
- `tests/dql/commerce_suite.dql` - No tunables (should pass)

**Tests to write:**
```python
# Add to tests/dql/test_parser.py
class TestTunableBoundsRegression:
    def test_all_example_dql_files_parse_successfully(): ...
    def test_banking_transactions_dql_tunables(): ...
    def test_book_inventory_dql_tunables(): ...
    def test_book_orders_dql_tunables(): ...
    def test_video_streaming_dql_tunables(): ...
```

**Implementation notes:**
- Parse each DQL file and verify no syntax errors
- Verify tunable bounds are extracted correctly
- This is a regression test suite (ensures no accidental breakage)

**Success criteria:**
- [ ] All 5 DQL files parse successfully
- [ ] Tunable bounds extracted correctly
- [ ] No syntax errors
- [ ] Coverage: 100% for regression tests
- [ ] Pre-commit hooks: passing

**Commit message:** `test(dql): add regression tests for tunable bounds parsing`

---

### Phase 4: Update Documentation

**Goal:** Update documentation to clarify bounds must be literals and provide migration guide.

**Duration estimate:** 1 hour

**Files to modify:**
- `docs/design/dql-language.md` - Update tunables section
- `docs/changelog.md` - Add breaking change entry

**Documentation updates:**

1. **dql-language.md - Tunables section (line ~365-415)**
   - Add clarification: "Bounds must be numeric literals (numbers or percentages). Arithmetic expressions and tunable references are not allowed in bounds."
   - Add examples of invalid bounds with explanations
   - Add migration guide for users upgrading from 0.5.x

2. **changelog.md - Add entry for 0.6.0**
   - Breaking change: Tunable bounds restricted to literals
   - Migration guide with before/after examples

**Content to add to dql-language.md:**

```markdown
#### Bounds Restrictions

Bounds must be **numeric literals only**. Arithmetic expressions and tunable references are not allowed in bounds.

**Valid bounds:**
```dql
tunable X = 5 bounds [0, 10]           # Integer literals ✓
tunable Y = 0.5 bounds [0.0, 1.0]      # Float literals ✓
tunable Z = 5% bounds [0%, 10%]        # Percent literals ✓
tunable W = 5 bounds [-10, 10]         # Negative literals ✓
```

**Invalid bounds:**
```dql
tunable X = 5 bounds [1 + 1, 10]       # Arithmetic ✗
tunable Y = 10 bounds [BASE * 2, 100]  # Tunable reference ✗
tunable Z = 5 bounds [min(1, 2), 10]   # Function call ✗
```

**Rationale:**
- Bounds define the search space for optimization algorithms
- Dynamic bounds (depending on other tunables) create a moving target
- Python API (`TunableInt`, `TunableFloat`) only accepts literals
- Simplifies parser and type inference

**Migration from DQX 0.5.x:**

If you have arithmetic in bounds, evaluate it manually:

```dql
# Before (DQX 0.5.x)
tunable BASE = 10 bounds [5, 20]
tunable SCALED = 20 bounds [BASE * 2, 100]

# After (DQX 0.6.0+)
tunable BASE = 10 bounds [5, 20]
tunable SCALED = 20 bounds [20, 100]  # Evaluated: 10 * 2 = 20
```

**Note:** Tunable **values** can still use arithmetic and reference other tunables:
```dql
tunable BASE = 10 bounds [5, 20]
tunable SCALED = BASE * 2 bounds [10, 40]  # Value can reference BASE ✓
```
```

**Success criteria:**
- [ ] Documentation clarifies bounds restriction
- [ ] Examples show valid and invalid syntax
- [ ] Migration guide provides clear before/after examples
- [ ] Changelog entry added for 0.6.0
- [ ] Pre-commit hooks: passing

**Commit message:** `docs(dql): clarify tunable bounds must be numeric literals`

---

### Phase 5: Final Verification and Integration

**Goal:** Run full test suite, verify coverage, and ensure no regressions.

**Duration estimate:** 30 minutes

**Verification checklist:**
- [ ] Run full test suite: `uv run pytest`
- [ ] Check coverage: `uv run pytest --cov=src/dqx --cov-report=term-missing`
- [ ] Run pre-commit hooks: `uv run pre-commit run --all-files`
- [ ] Verify no type errors: `uv run mypy src tests`
- [ ] Parse all DQL examples in documentation
- [ ] Manual smoke test: Create suite with literal bounds
- [ ] Manual smoke test: Try arithmetic in bounds (should fail with clear error)

**Integration tests:**
```python
# Add to tests/e2e/test_dql_verification_suite_e2e.py
class TestTunableBoundsE2E:
    def test_suite_with_literal_bounds_runs_successfully(): ...
    def test_suite_with_arithmetic_bounds_fails_gracefully(): ...
    def test_tunable_type_inference_with_literal_bounds(): ...
```

**Implementation notes:**
- Verify error messages are user-friendly
- Check that Python API tunables still work (no impact expected)
- Verify type inference (int vs float) works correctly with literals

**Success criteria:**
- [ ] All tests pass (100% pass rate)
- [ ] Coverage: 100% for src/dqx/dql/
- [ ] Pre-commit hooks: all passing
- [ ] No mypy errors
- [ ] Documentation examples parse successfully
- [ ] Manual smoke tests pass

**Commit message:** `test(dql): add e2e tests for tunable bounds validation`

---

## Phase Dependencies

**Sequential phases:**
1. Phase 1 must complete first (grammar changes)
2. Phases 2-4 can run in parallel after Phase 1
3. Phase 5 runs last (final verification)

**Dependency graph:**
```
Phase 1 (Grammar + Validation Tests)
   ├─→ Phase 2 (Update Existing Tests)
   ├─→ Phase 3 (Verify DQL Files)
   └─→ Phase 4 (Update Documentation)
           └─→ Phase 5 (Final Verification)
```

## Rollback Strategy

**If issues arise during implementation:**

1. **Revert grammar changes:**
   ```bash
   git revert <commit-hash>  # Grammar update commit
   ```

2. **Restore test files:**
   ```bash
   git checkout main -- tests/test_dql_arithmetic.py
   git checkout main -- tests/test_api_dql_tunables.py
   ```

3. **Skip documentation updates:**
   - Documentation is non-breaking, can be fixed in follow-up

**If issues arise after merge:**

1. **Emergency hotfix (if grammar breaks parsing):**
   - Release 0.5.x hotfix with reverted grammar
   - Investigate issue before re-attempting

2. **Documentation fix (if examples are wrong):**
   - Hot-patch documentation
   - No code rollback needed

**Risk assessment:** Low risk
- Grammar change is isolated and well-defined
- All existing DQL files use literals (verified)
- Test coverage ensures no regressions

## Estimated Total Time

**Phase estimates:**
- Phase 1: 1-2 hours (grammar + validation tests)
- Phase 2: 1 hour (update existing tests)
- Phase 3: 30 minutes (verify DQL files)
- Phase 4: 1 hour (documentation)
- Phase 5: 30 minutes (final verification)

**Total: 4-5 hours**

**Confidence level:** High (80-90%)
- Straightforward grammar change
- No complex logic changes
- Good test coverage
- Clear success criteria

## Breaking Change Communication

**Version bump:** 0.5.x → 0.6.0 (minor version for breaking change)

**Commit message format:**
```
feat(dql)!: restrict tunable bounds to numeric literals only

BREAKING CHANGE: Tunable bounds no longer support arithmetic expressions
or references to other tunables. Bounds must be numeric literals (numbers
or percentages).

Before:
    tunable BASE = 10 bounds [5, 20]
    tunable SCALED = 20 bounds [BASE * 2, 100]

After:
    tunable BASE = 10 bounds [5, 20]
    tunable SCALED = 20 bounds [20, 100]  # Evaluated manually

Rationale:
- Simplifies language and aligns with Python API
- Bounds should be fixed constraints, not dynamic expressions
- Improves clarity for optimization algorithms

Migration: Evaluate arithmetic manually and hardcode the result.

Closes #XXX
```

**Changelog entry (docs/changelog.md):**
```markdown
## [0.6.0] - 2026-01-XX

### Breaking Changes

- **DQL: Tunable bounds restricted to numeric literals** ([#XXX](link))
  - Tunable bounds no longer support arithmetic expressions or tunable references
  - Bounds must be `NUMBER` or `PERCENT` literals only
  - Rationale: Simplifies language, aligns with Python API, improves optimization
  - Migration: Evaluate arithmetic manually (e.g., `BASE * 2` → `20`)
  - See [dql-language.md](docs/design/dql-language.md#bounds-restrictions) for details
```

## Post-Implementation Checklist

After all phases complete:

- [ ] All tests pass (`uv run pytest`)
- [ ] Coverage 100% maintained
- [ ] Pre-commit hooks pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Breaking change clearly communicated
- [ ] Migration guide provided
- [ ] All DQL files parse successfully
- [ ] Error messages tested and validated
- [ ] Manual smoke tests performed

**Ready for PR:** When all checkboxes are complete.
