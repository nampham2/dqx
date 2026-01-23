# PermanentProfile Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the `PermanentProfile` feature. We'll add a new profile type that is always active (no date-based activation), using the same Rule system as `SeasonalProfile`.

**Approach**: Test-Driven Development (TDD) - write tests first, then implement to pass them.

## Prerequisites

### Files to read before starting:

- `src/dqx/profiles.py` - Understand `Profile` protocol and `SeasonalProfile` structure
- `src/dqx/config.py` - Understand `load_profiles_from_config()` function (lines 281-401)
- `tests/test_profiles.py` - Understand profile testing patterns
- `tests/test_config.py` - Understand config loading tests (class `TestProfileLoading`)

### Related components:

- **Profile Protocol** (`profiles.py:78-94`): Defines contract all profiles must satisfy
- **SeasonalProfile** (`profiles.py:98-118`): Template for `PermanentProfile` implementation
- **Rule System** (`profiles.py:60-76, 120-183`): Shared between all profile types
- **Config Loading** (`config.py:281-401`): Parses YAML profiles into Python objects
- **VerificationSuite** (`api.py:908-936`): Integrates profiles into validation pipeline
- **Evaluator** (`evaluator.py:309-376`): Applies profile overrides during evaluation

## Phase Breakdown

### Phase 1: Core PermanentProfile Class

**Goal**: Create `PermanentProfile` dataclass with `is_active()` always returning `True`.

**Duration estimate**: 1 hour

**Files to create**:
- None (adding to existing files)

**Files to modify**:
- `src/dqx/profiles.py` - Add `PermanentProfile` dataclass after `SeasonalProfile`
- `tests/test_profiles.py` - Add `TestPermanentProfile` class

**Tests to write** (in `tests/test_profiles.py`):

```python
class TestPermanentProfile:
    """Tests for PermanentProfile."""

    def test_creation_with_name_and_rules(): ...

    def test_creation_with_default_rules(): ...

    def test_is_active_always_true(): ...

    def test_is_active_ignores_target_date(): ...

    def test_rules_property(): ...

    def test_protocol_compliance(): ...

    def test_frozen_dataclass(): ...
```

**Implementation notes**:

1. **Location**: Add `PermanentProfile` after `SeasonalProfile` (around line 119)
2. **Pattern**: Follow `SeasonalProfile` structure exactly:
   - Frozen dataclass with `name` and `rules` fields
   - `is_active()` method that accepts `target_date` but returns `True`
   - `rules` field with `field(default_factory=list)` for mutable default
3. **Docstring**: Use Google style with examples (see technical spec)
4. **Type hints**: Use `date` from `datetime` module, `list[Rule]` for rules

**Success criteria**:
- [ ] All Phase 1 tests passing
- [ ] Coverage: 100% for `PermanentProfile` class
- [ ] Pre-commit hooks: passing
- [ ] Mypy: no type errors

**Commit message**: `feat(profiles): add PermanentProfile class for always-active profiles`

---

### Phase 2: YAML Config Support

**Goal**: Enable loading `PermanentProfile` from YAML config files with `type: "permanent"`.

**Duration estimate**: 1.5 hours

**Files to modify**:
- `src/dqx/config.py` - Modify `load_profiles_from_config()` function
- `tests/test_config.py` - Add tests to `TestProfileLoading` class

**Tests to write** (in `tests/test_config.py:TestProfileLoading`):

```python
def test_load_permanent_profile_with_disable_rule(tmp_path: Path): ...

def test_load_permanent_profile_with_scale_rule(tmp_path: Path): ...

def test_load_permanent_profile_with_set_severity_rule(tmp_path: Path): ...

def test_load_permanent_profile_with_multiple_rules(tmp_path: Path): ...

def test_load_permanent_profile_with_empty_rules(tmp_path: Path): ...

def test_load_mixed_permanent_and_seasonal_profiles(tmp_path: Path): ...

def test_permanent_profile_with_start_date_raises_error(tmp_path: Path): ...

def test_permanent_profile_with_end_date_raises_error(tmp_path: Path): ...

def test_permanent_profile_with_both_dates_raises_error(tmp_path: Path): ...

def test_unknown_profile_type_raises_error(tmp_path: Path): ...

def test_permanent_profile_missing_name_raises_error(tmp_path: Path): ...

def test_permanent_profile_with_unknown_fields_raises_error(tmp_path: Path): ...

def test_permanent_profile_duplicate_names_raises_error(tmp_path: Path): ...
```

**Implementation notes**:

1. **Location**: Modify `load_profiles_from_config()` starting at line 336
2. **Changes**:
   - Extract `profile_type` (line ~338): Change from equality check to variable
   - Add conditional branch for `"permanent"` type
   - Update error message for unknown types: `"must be 'seasonal' or 'permanent'"`
   - Validate permanent profiles don't have `start_date` or `end_date` fields

3. **Pseudo-code structure**:
```python
profile_type = profile_dict.get("type", "seasonal")

if profile_type == "seasonal":
    # Existing SeasonalProfile parsing (lines 342-392)
    ...
elif profile_type == "permanent":
    # New PermanentProfile parsing
    # 1. Validate no start_date field
    # 2. Validate no end_date field
    # 3. Parse rules (reuse _parse_rules)
    # 4. Validate no unknown fields: {"name", "type", "rules"}
    # 5. Create PermanentProfile instance
    ...
else:
    raise DQXError(f"Profile '{name}': unknown type '{profile_type}' (must be 'seasonal' or 'permanent')")
```

4. **Error messages**:
   - If `start_date` present: `"Profile '{name}': 'start_date' not allowed for permanent profiles"`
   - If `end_date` present: `"Profile '{name}': 'end_date' not allowed for permanent profiles"`
   - Unknown type: `"Profile '{name}': unknown type '{profile_type}' (must be 'seasonal' or 'permanent')"`

5. **Import**: Add `from dqx.profiles import PermanentProfile` near line 304

**Success criteria**:
- [ ] All Phase 2 tests passing
- [ ] Coverage: 100% for new config parsing code
- [ ] Pre-commit hooks: passing
- [ ] Can load permanent profiles from YAML
- [ ] Appropriate errors for invalid configs

**Commit message**: `feat(config): add YAML support for PermanentProfile`

---

### Phase 3: Integration and E2E Tests

**Goal**: Verify `PermanentProfile` works correctly in full suite execution with profile compounding.

**Duration estimate**: 1.5 hours

**Files to modify**:
- `tests/test_profiles.py` - Add integration tests to `TestEvaluatorWithProfiles`
- `tests/test_config.py` - Add E2E test to `TestConfigIntegration`

**Tests to write**:

**In `tests/test_profiles.py:TestEvaluatorWithProfiles`**:

```python
def test_evaluator_with_permanent_profile_metric_multiplier(): ...

def test_evaluator_with_permanent_profile_disabled(): ...

def test_evaluator_with_permanent_profile_severity_override(): ...

def test_permanent_and_seasonal_profiles_compound(): ...

def test_permanent_profile_always_active_any_date(): ...

def test_multiple_permanent_profiles_compound(): ...

def test_permanent_profile_with_tag_selector(): ...

def test_permanent_profile_with_assertion_selector(): ...
```

**In `tests/test_config.py:TestConfigIntegration`**:

```python
def test_end_to_end_with_permanent_profile(tmp_path: Path): ...

def test_permanent_profile_disables_assertion_in_suite(tmp_path: Path): ...

def test_permanent_and_seasonal_profiles_from_config(tmp_path: Path): ...
```

**Implementation notes**:

1. **No source code changes** - this phase only adds tests
2. **Test patterns**: Follow existing `TestEvaluatorWithProfiles` patterns:
   - Use `_create_evaluator_with_metric()` helper
   - Create simple assertion graphs with `RootNode`, `CheckNode`, `AssertionNode`
   - Verify `_result` field values (`"PASSED"`, `"FAILED"`, `"SKIPPED"`)
3. **Focus on compounding**: Test that permanent + seasonal profiles multiply correctly
4. **Date variation**: Test permanent profile with various dates (past, present, future)

**Success criteria**:
- [ ] All Phase 3 tests passing
- [ ] Coverage: 100% maintained
- [ ] Pre-commit hooks: passing
- [ ] E2E validation confirms feature works in real suite

**Commit message**: `test(profiles): add integration tests for PermanentProfile`

---

### Phase 4: Documentation and Examples

**Goal**: Add docstrings, update documentation files, and provide usage examples.

**Duration estimate**: 1 hour

**Files to modify**:
- `src/dqx/profiles.py` - Enhance `PermanentProfile` docstring with full example
- `README.md` - Add `PermanentProfile` to feature list (if profiles mentioned)
- `docs/api-reference.md` - Document `PermanentProfile` API (if exists)

**Tasks**:

1. **Enhance `PermanentProfile` docstring**:
   - Add comprehensive example showing YAML + Python usage
   - Document use cases (account-specific, environment-specific)
   - Note interaction with `SeasonalProfile`

2. **Update module docstring** (`src/dqx/profiles.py`):
   - Add `PermanentProfile` to example if `SeasonalProfile` is shown
   - Update description to mention both profile types

3. **Check for documentation files**:
   - If `docs/guides/profiles.md` exists, add `PermanentProfile` section
   - If `examples/config.yaml` exists, add permanent profile example
   - If `docs/api-reference.md` exists, document `PermanentProfile` class

4. **Add inline code examples** (in docstring):

```python
"""
Permanent Profile Example:

Python API:
    from dqx.profiles import PermanentProfile, check, tag

    baseline = PermanentProfile(
        name="Production Baseline",
        rules=[
            check("Dev Only Check").disable(),
            tag("critical").set(severity="P0"),
        ],
    )

    suite = VerificationSuite(
        checks=[...],
        profiles=[baseline, seasonal_profile],  # Mix profile types
    )

YAML Config:
    profiles:
      - name: "Production Baseline"
        type: "permanent"
        rules:
          - action: "disable"
            target: "check"
            name: "Dev Only Check"

Use Cases:
    - Account-specific baseline configurations
    - Environment-specific overrides (dev/staging/prod)
    - Permanently disabled legacy checks
"""
```

**Success criteria**:
- [ ] Docstrings complete with examples
- [ ] Pre-commit hooks: passing
- [ ] Documentation updated (if applicable)
- [ ] API reference complete (if exists)

**Commit message**: `docs(profiles): document PermanentProfile usage and examples`

---

## Phase Dependencies

**Sequential dependencies**:
- Phase 2 depends on Phase 1 (config loading needs `PermanentProfile` class)
- Phase 3 depends on Phase 1 & 2 (integration tests need both class and config support)
- Phase 4 depends on Phase 1-3 (document completed implementation)

**Can be parallelized**: None (each phase builds on previous)

## Rollback Strategy

### If issues arise during implementation:

1. **Phase 1 issues**: Revert commit, class is isolated in `profiles.py`
2. **Phase 2 issues**: Revert commit, config loader has clear conditional branch
3. **Phase 3 issues**: Tests only, revert commit safely
4. **Phase 4 issues**: Documentation only, revert commit safely

**Low risk**: Feature is purely additive, no existing code modified (only extensions).

### Safe rollback procedure:

```bash
# Revert last commit
git revert HEAD

# Or revert specific commit
git revert <commit-hash>

# All tests should still pass (backward compatible)
uv run pytest
```

## Testing Strategy

### Unit Test Coverage

- `PermanentProfile` class (Phase 1): 7 tests
- Config loading (Phase 2): 13 tests
- Integration (Phase 3): 11 tests
- **Total**: ~31 new tests

### Coverage Target

**100% coverage required** for all new code:
- `PermanentProfile` dataclass
- Config parsing branch for permanent profiles
- All error paths

### Quality Gates

After each phase:

```bash
# 1. Run all tests
uv run pytest

# 2. Check coverage
uv run pytest --cov=src/dqx --cov-report=term-missing

# 3. Run pre-commit hooks
uv run pre-commit run --all-files
```

All must pass before proceeding to next phase.

## Common Pitfalls

### Pitfall 1: Forgetting `field(default_factory=list)` for rules

**Problem**: Using `rules: list[Rule] = []` creates shared mutable default.

**Solution**: Always use `field(default_factory=list)` for mutable defaults in dataclasses.

```python
# WRONG
rules: list[Rule] = []

# CORRECT
rules: list[Rule] = field(default_factory=list)
```

### Pitfall 2: Modifying existing profile type check

**Problem**: Changing existing `if profile_type != "seasonal"` breaks backward compatibility.

**Solution**: Extract to variable and use if-elif-else chain:

```python
# WRONG - breaks default behavior
if profile_type != "seasonal":
    raise DQXError(...)

# CORRECT - maintains default
profile_type = profile_dict.get("type", "seasonal")
if profile_type == "seasonal":
    ...
elif profile_type == "permanent":
    ...
else:
    raise DQXError(...)
```

### Pitfall 3: Not validating absence of date fields

**Problem**: Users might accidentally include dates in permanent profiles.

**Solution**: Explicitly check and raise clear errors:

```python
if "start_date" in profile_dict:
    raise DQXError(f"Profile '{name}': 'start_date' not allowed for permanent profiles")
```

### Pitfall 4: Incorrect valid_fields set for permanent profiles

**Problem**: Using same valid_fields as seasonal profiles allows date fields.

**Solution**: Define separate valid_fields for permanent profiles:

```python
# For permanent profiles only
valid_fields = {"name", "type", "rules"}  # No dates
_validate_no_unknown_fields(profile_dict, valid_fields, f"Profile '{name}'")
```

### Pitfall 5: Forgetting to import PermanentProfile in config.py

**Problem**: Config loader can't create instances without import.

**Solution**: Add to existing imports:

```python
from dqx.profiles import PermanentProfile, SeasonalProfile
```

## Debugging Tips

### If tests fail in Phase 1:

1. Check dataclass definition matches `Profile` protocol exactly
2. Verify `is_active()` signature: `def is_active(self, target_date: date) -> bool`
3. Ensure `frozen=True` for immutability
4. Run single test: `uv run pytest tests/test_profiles.py::TestPermanentProfile::test_creation_with_name_and_rules -v`

### If tests fail in Phase 2:

1. Check YAML syntax in test fixtures (indentation matters)
2. Verify error messages match exactly (use `match=` in `pytest.raises`)
3. Add debug print in config loader: `logger.debug(f"Parsing profile type: {profile_type}")`
4. Run single test: `uv run pytest tests/test_config.py::TestProfileLoading::test_load_permanent_profile_with_disable_rule -v`

### If tests fail in Phase 3:

1. Check profile is actually active: `assert profile.is_active(target_date)`
2. Verify rules match: `assert len(profile.rules) == expected_count`
3. Debug evaluator: Add logging in `Evaluator.visit()` to see override values
4. Test in isolation: Create minimal reproducer with single profile + assertion

## Verification Checklist

Before marking feature complete:

### Functionality
- [ ] `PermanentProfile` always returns `True` from `is_active()`
- [ ] Config loads permanent profiles with `type: "permanent"`
- [ ] Config rejects dates in permanent profiles with clear errors
- [ ] Permanent + seasonal profiles compound correctly
- [ ] Profiles work in full VerificationSuite execution

### Code Quality
- [ ] All tests passing (31+ new tests)
- [ ] Coverage: 100% for new code
- [ ] Pre-commit hooks: passing (format, lint, type check)
- [ ] No mypy errors
- [ ] Docstrings complete with examples

### Documentation
- [ ] `PermanentProfile` class documented
- [ ] Usage examples provided (Python + YAML)
- [ ] Error messages clear and actionable

### Compatibility
- [ ] Backward compatible with existing configs
- [ ] Works with existing `SeasonalProfile` instances
- [ ] No changes to public APIs (purely additive)

## Estimated Total Time

**Total: ~5 hours** (broken into 4 phases)

- Phase 1: 1 hour (core class)
- Phase 2: 1.5 hours (config support)
- Phase 3: 1.5 hours (integration tests)
- Phase 4: 1 hour (documentation)

**Realistic timeline**: 1 day (includes testing, iteration, code review)

## Next Steps After Implementation

1. **Code Review**: Submit PR with all 4 phases
2. **Manual Testing**: Test with real config file and suite
3. **Update Examples**: Add permanent profile to example configs
4. **Announce**: Document in changelog/release notes

## Questions to Resolve During Implementation

None anticipated - design is clear and follows existing patterns.

If questions arise:
1. Check `SeasonalProfile` implementation for guidance
2. Review Profile protocol for interface requirements
3. Consult AGENTS.md for code standards
