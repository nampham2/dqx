# Remove Tunable Arithmetic from DQL Bounds - Context Document

This document provides background context for implementing the removal of arithmetic expressions from DQL tunable bounds.

## DQX Architecture Overview

### Relevant Components

#### DQL Parser (`src/dqx/dql/`)

**Purpose:** Parses DQL source files into Abstract Syntax Tree (AST) nodes

**Key files:**
- `grammar.lark` - Lark grammar definition (EBNF-style)
- `parser.py` - Lark transformer that converts parse tree to AST
- `ast.py` - AST node dataclasses (Tunable, Suite, Check, Assertion, etc.)
- `errors.py` - DQL-specific error types (DQLSyntaxError, DQLError)

**How this feature relates:**
- Grammar defines tunable syntax: `tunable IDENT = SIGNED_NUMBER bounds [SIGNED_NUMBER, SIGNED_NUMBER]`
- Parser transforms grammar tokens into `Tunable` AST nodes
- This change restricts the value and bounds to signed numeric literals only (no percentages, no arithmetic)

#### Tunable Evaluator (`src/dqx/api.py`)

**Purpose:** Converts DQL AST to Python tunable objects

**Key methods:**
- `_build_dql_checks()` - Converts DQL Suite AST to Python checks
- `_build_tunables_from_ast()` - Creates `TunableInt`/`TunableFloat` from AST
- `_eval_simple_expr()` - Evaluates numeric expressions using SymPy
- `_eval_metric_expr()` - Evaluates metric expressions with tunable substitution

**How this feature relates:**
- Currently evaluates arbitrary arithmetic in bounds using `_eval_simple_expr()`
- After change, bounds will always be simple literals (no evaluation needed)
- Simplifies type inference (int vs float detection)

#### Python Tunable API (`src/dqx/tunables.py`)

**Purpose:** Type-safe tunable parameter implementations

**Key classes:**
- `TunableInt` - Integer tunables with bounds
- `TunableFloat` - Float tunables with bounds
- `TunablePercent` - Percentage tunables (0-1 internally)
- `TunableChoice` - Categorical tunables

**How this feature relates:**
- Python API already only accepts numeric literals for bounds
- This change brings DQL into parity with Python API
- No changes needed to tunable implementations

## Code Patterns to Follow

**IMPORTANT**: All patterns reference AGENTS.md standards.

### Pattern 1: Lark Grammar Rules

**When to use:** Defining new grammar rules or modifying existing ones

**Example from DQX** (grammar.lark):
```lark
# Existing pattern: expression-based rules
expr: term (ADD_OP term)*
term: factor (MUL_OP factor)*
factor: NUMBER | PERCENT | ident | ...

# New pattern: literal-only rules
bound_value: NUMBER | PERCENT
```

**Reference:** See AGENTS.md §formatting for grammar file conventions

**Apply to this feature:**
- Restrict tunable values and bounds to signed numeric literals only
- Change tunable grammar to use `SIGNED_NUMBER` token instead of `expr`
- No longer accept percentages or arithmetic expressions in tunables

### Pattern 2: Immutable AST Nodes

**Example from DQX** (ast.py):
```python
@dataclass(frozen=True)
class Tunable:
    """A tunable constant declaration with required bounds."""
    name: str
    value: Expr
    bounds: tuple[Expr, Expr]
    loc: SourceLocation | None = None
```

**Reference:** AGENTS.md §dataclasses

**Apply to this feature:**
- No changes needed to AST structure
- `bounds` field remains `tuple[Expr, Expr]`
- Expr objects will contain only numeric text after grammar change

### Pattern 3: Lark Transformer Methods

**Example from DQX** (parser.py):
```python
class DQLTransformer(Transformer):
    def NUMBER(self, token: Any) -> float | int:
        return _parse_number(str(token))

    def PERCENT(self, token: Any) -> float:
        return _parse_percent(str(token))

    @v_args(tree=True)
    def tunable(self, tree: Any) -> Tunable:
        items = [x for x in tree.children if x is not None]
        name = items[0]
        value = items[1]
        min_bound = items[2]  # Will be NUMBER or PERCENT token
        max_bound = items[3]  # Will be NUMBER or PERCENT token
        return Tunable(
            name=name,
            value=value,
            bounds=(min_bound, max_bound),
            loc=self._loc(tree),
        )
```

**Reference:** AGENTS.md §type-hints (all methods fully typed)

**Apply to this feature:**
- No parser changes needed (grammar does the work)
- Tokens are automatically converted by `NUMBER()` and `PERCENT()` methods
- Parser receives literals, wraps them in Expr objects

### Pattern 4: SymPy Expression Evaluation

**Example from DQX** (api.py):
```python
def _eval_simple_expr(self, expr: Any, tunables: dict[str, Tunable[Any]]) -> float:
    """Evaluate numeric expressions with full SymPy arithmetic support."""
    import sympy as sp
    from dqx.dql.errors import DQLError

    text = expr.text.strip()

    try:
        # Inject tunables as numeric values for immediate evaluation
        namespace = {name: tunable.value for name, tunable in tunables.items()}
        result = sp.sympify(text, locals=namespace, evaluate=True)

        # Preserve int vs float type
        if result.is_Integer:
            return int(result)
        return float(result)
    except (sp.SympifyError, Exception) as e:
        raise DQLError(f"Cannot evaluate expression: {text}", loc=expr.loc) from e
```

**Reference:** AGENTS.md §error-handling (use DQLError for domain errors)

**Apply to this feature:**
- This method will only receive literals after grammar change
- Evaluation becomes trivial (no arithmetic to evaluate)
- Error handling still works for edge cases

### Pattern 5: Clear Error Messages with Source Location

**Example from DQX** (errors.py usage):
```python
raise DQLSyntaxError(
    message=f"Unexpected token: {e.token!r}",
    loc=SourceLocation(line=e.line, column=e.column, filename=filename),
    source_line=source_line,
    suggestion=f"Expected one of: {expected}" if expected else None,
)
```

**Reference:** AGENTS.md §error-handling

**Apply to this feature:**
- Lark will generate syntax errors automatically when grammar doesn't match
- Error messages will indicate unexpected tokens in bounds
- Source location tracking is built into Lark parser

## Code Standards Reference

**All code must follow AGENTS.md standards**:

- **Import Order**: AGENTS.md §import-order
  - `from __future__ import annotations` at top
  - Standard library → third-party → local imports

- **Type Hints**: AGENTS.md §type-hints (strict mode)
  - All functions fully typed
  - Use `|` for unions (not `Union`)
  - Use `collections.abc` for abstract types

- **Docstrings**: AGENTS.md §docstrings (Google style)
  - All public functions documented
  - Args, Returns, Raises sections

- **Testing**: AGENTS.md §testing-standards
  - Mirror source structure: `src/dqx/dql/parser.py` → `tests/dql/test_parser.py`
  - Organize in classes: `class TestTunableBoundsValidation:`
  - Descriptive names: `test_tunable_with_arithmetic_in_bounds_fails`

- **Coverage**: AGENTS.md §coverage-requirements (100%)
  - All new code must be covered
  - Use `# pragma: no cover` only for unreachable defensive code

## Testing Patterns

**Reference:** AGENTS.md §testing-patterns

### Test Organization

Tests mirror source structure:
- `src/dqx/dql/grammar.lark` → `tests/dql/test_parser.py`
- `src/dqx/dql/ast.py` → `tests/dql/test_ast.py`

### Test Class Structure

```python
class TestTunableBoundsValidation:
    """Tests for tunable bounds validation."""

    def test_valid_numeric_bounds(self) -> None:
        """Test that numeric literal bounds are accepted."""
        dql = """
        suite "Test" {
            tunable X = 5 bounds [0, 10]
        }
        """
        suite = parse(dql)
        assert suite.tunables[0].name == "X"
        assert suite.tunables[0].bounds[0].text == "0"
        assert suite.tunables[0].bounds[1].text == "10"

    def test_invalid_arithmetic_bounds(self) -> None:
        """Test that arithmetic in bounds raises DQLSyntaxError."""
        dql = """
        suite "Test" {
            tunable X = 5 bounds [1 + 1, 10]
        }
        """
        with pytest.raises(DQLSyntaxError, match="bounds"):
            parse(dql)
```

**For this feature:**
- Create new test file: `tests/dql/test_tunable_bounds_validation.py`
- Test both success cases (literals) and failure cases (arithmetic)
- Validate error messages for clarity

### Fixtures

Use fixtures from `tests/fixtures/`:
- No new fixtures needed for this feature
- Use `tmp_path` fixture for creating DQL files

**Example:**
```python
def test_suite_with_literal_bounds(tmp_path: Path) -> None:
    dql_file = tmp_path / "test.dql"
    dql_file.write_text("""
        suite "Test" {
            tunable X = 5 bounds [0, 10]
        }
    """)
    suite = VerificationSuite(dql=dql_file, db=InMemoryMetricDB())
    assert suite._name == "Test"
```

## Common Pitfalls

### Pitfall 1: Forgetting to Update Test Files

**Problem:** Tests using arithmetic in bounds will break after grammar change

**Solution:**
- Identify all tests using arithmetic: `grep -r "bounds.*[\+\-\*/]" tests/`
- Remove or update each test
- Run full test suite after changes

**For this feature:**
- 2 test files need updates: `test_dql_arithmetic.py`, `test_api_dql_tunables.py`
- 3 test methods need removal

### Pitfall 2: Incomplete Error Message Validation

**Problem:** Error messages might be unclear or misleading

**Solution:**
- Test error messages explicitly (use `match` parameter in `pytest.raises`)
- Include suggestions for valid syntax
- Reference documentation examples

**For this feature:**
- Add tests that validate error message content
- Ensure messages mention "numeric literals" and "bounds"
- Provide examples of valid syntax in error messages

### Pitfall 3: Breaking Tunable Value Expressions

**Problem:** Accidentally restricting tunable values (not just bounds)

**Solution:**
- Only change bounds grammar rule, not value expression
- Keep existing tests for tunable values that reference other tunables
- Value can still use: `tunable Y = BASE * 2 bounds [10, 40]` ✓

**For this feature:**
- Grammar change: `bounds [bound_value, bound_value]` (restricted)
- No change: `= expr` (unrestricted)
- Test that values can still reference tunables

### Pitfall 4: Forgetting DQL Files in Tests

**Problem:** DQL test fixtures might use arithmetic in bounds

**Solution:**
- Search all `.dql` files: `find tests -name "*.dql" -exec grep "bounds" {} \;`
- Verify each file parses successfully after grammar change
- Update any files using arithmetic

**For this feature:**
- All 5 DQL files already use literals (verified)
- No updates needed, but add regression tests

### Pitfall 5: Incomplete Documentation

**Problem:** Users might not understand the breaking change

**Solution:**
- Update all documentation referencing tunables
- Provide clear migration guide with before/after examples
- Explain rationale for the change

**For this feature:**
- Update `docs/design/dql-language.md`
- Add changelog entry for 0.6.0
- Include migration examples

## Related PRs and Issues

**Similar changes (language restrictions):**
- If there are past PRs that restricted DQL syntax, reference them here
- Check git history for similar breaking changes
- Follow the same commit message format

**This is a new pattern:**
- First breaking change to tunable syntax
- Establishes precedent for simplifying DQL
- Future changes can reference this PR

## Grammar Development Pattern

**Lark grammar development workflow:**

1. **Update grammar file** (`grammar.lark`)
   - Add new rules or modify existing ones
   - Keep grammar readable with comments
   - Use consistent naming (lowercase for rules, UPPERCASE for tokens)

2. **Test grammar parsing** (add parser tests)
   - Test valid syntax (should parse successfully)
   - Test invalid syntax (should raise DQLSyntaxError)
   - Verify AST structure is correct

3. **Update transformer if needed** (`parser.py`)
   - Usually not needed (grammar changes are enough)
   - Only modify if AST transformation logic changes

4. **Run grammar tests**
   ```bash
   uv run pytest tests/dql/test_parser.py -v
   ```

**For this feature:**
- Step 1: Update grammar (add `bound_value` rule)
- Step 2: Add validation tests
- Step 3: No transformer changes needed
- Step 4: Run all DQL tests

## Documentation

After implementation, update:

### API Reference (`docs/design/dql-language.md`)

**Section: Tunables (line ~365-415)**

Add subsection "Bounds Restrictions" with:
- Clear explanation of literal-only requirement
- Examples of valid and invalid syntax
- Rationale for the restriction
- Migration guide from 0.5.x

### Changelog (`docs/changelog.md`)

Add entry for version 0.6.0:
- Breaking change announcement
- Migration instructions
- Link to detailed documentation

### Code Comments

Grammar file comments:
```lark
# Tunable bounds must be numeric literals (no arithmetic or references)
# This simplifies type inference and aligns with Python API
bound_value: NUMBER | PERCENT
```

## Breaking Change Protocol

**Reference:** AGENTS.md §commit-conventions

### Commit Message Format

Use `!` to mark breaking change:
```
feat(dql)!: restrict tunable bounds to numeric literals only

BREAKING CHANGE: Tunable bounds no longer support arithmetic expressions
or references to other tunables.
```

### Version Bump

- Current: 0.5.x
- After: 0.6.0 (minor version bump for breaking change)
- Follow semantic versioning (pre-1.0)

### Communication

- Update changelog with migration guide
- Update documentation before merge
- Consider deprecation warning in future (not applicable here since removing feature)

## Implementation Checklist

Before starting:
- [ ] Read grammar.lark to understand current tunable syntax
- [ ] Read parser.py to understand AST transformation
- [ ] Read api.py to understand tunable evaluation
- [ ] Identify all tests using arithmetic in bounds
- [ ] Verify all DQL files use literals

During implementation:
- [ ] Update grammar with `bound_value` rule
- [ ] Add validation tests for bounds
- [ ] Remove tests using arithmetic in bounds
- [ ] Update documentation
- [ ] Add changelog entry

After implementation:
- [ ] All tests pass
- [ ] Coverage 100%
- [ ] Pre-commit hooks pass
- [ ] Documentation updated
- [ ] Breaking change clearly communicated

## Performance Notes

**Impact:** Negligible (bounds evaluated once at suite creation)

**Before:**
- Grammar parses `expr` in bounds
- Evaluator uses SymPy to evaluate arithmetic
- Type inference analyzes expression result

**After:**
- Grammar parses `bound_value` (literals only)
- No SymPy evaluation needed (literals are already numeric)
- Type inference is trivial (check token type)

**Benefit:** Slight performance improvement, significant complexity reduction

## Testing Strategy

### Unit Tests

**Grammar parsing:**
- Valid: numeric literals, percentages, negative numbers
- Invalid: arithmetic, tunable references, function calls

**Error messages:**
- Clear indication of what's wrong
- Suggestion for valid syntax
- Example of correct usage

### Integration Tests

**DQL file parsing:**
- All existing DQL files parse successfully
- Suite creation works with literal bounds
- Tunable evaluation extracts correct bounds

**Type inference:**
- Integer literals → TunableInt
- Float literals → TunableFloat
- Percentage literals → TunableFloat (converted)

### End-to-End Tests

**Full verification suite:**
- Create suite with literal bounds
- Run suite with datasource
- Verify tunables work correctly in assertions
- Test `get_tunable_params()` returns correct bounds

**Error cases:**
- Suite creation fails gracefully with arithmetic in bounds
- Error message guides user to fix

### Regression Tests

**Existing functionality:**
- Tunable value expressions still support arithmetic
- Tunables can still reference other tunables (in value, not bounds)
- All existing DQL examples still work

## Summary

**Key takeaways:**
1. **Grammar-level enforcement** - Catch errors at parse time
2. **Minimal code changes** - Grammar does the heavy lifting
3. **Clear error messages** - Guide users to correct syntax
4. **Comprehensive testing** - Validate both success and failure cases
5. **Breaking change protocol** - Follow semantic versioning and documentation standards

**Implementation focus:**
- Grammar changes (5 lines)
- Test updates (remove 3 tests, add 11 validation tests)
- Documentation updates (1 section, 1 changelog entry)
- Verification (ensure all existing DQL files work)

**Expected outcome:**
- Simpler, more consistent language
- Parity with Python API
- Clearer semantics for optimization algorithms
- Maintained 100% test coverage
