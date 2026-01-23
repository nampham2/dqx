# PermanentProfile Context for Implementation

This document provides background context for implementing the `PermanentProfile` feature in DQX.

## DQX Architecture Overview

### Relevant Components

#### Profile System (`src/dqx/profiles.py`)
- **Purpose**: Modify assertion behavior based on conditions (dates, rules)
- **Key classes**: `Profile` (protocol), `SeasonalProfile`, `Rule`, selectors
- **How PermanentProfile relates**: New profile type implementing same `Profile` protocol

**Core concepts**:
- **Profile Protocol**: Defines contract (`is_active()`, `name`, `rules`)
- **Selectors**: Target assertions by tag, check name, or assertion name
- **Rules**: Define actions (disable, scale metric, set severity)
- **Compounding**: Multiple profiles/rules multiply effects together

#### Config Loading (`src/dqx/config.py`)
- **Purpose**: Load tunables and profiles from YAML files
- **Key function**: `load_profiles_from_config()` - parses YAML into profile objects
- **How PermanentProfile relates**: Add parsing branch for `type: "permanent"`

**Parsing flow**:
1. Load YAML file → dict
2. Validate structure (tunables, profiles sections)
3. For each profile: parse type, dates, rules
4. Create profile instances
5. Return list of profiles

#### Verification Suite (`src/dqx/api.py`)
- **Purpose**: Main entry point for defining and running validations
- **Profile integration**: Accepts profiles in constructor, passes to Evaluator
- **How PermanentProfile relates**: No changes needed (accepts `Profile` protocol)

#### Evaluator (`src/dqx/evaluator.py`)
- **Purpose**: Execute assertions and apply profile overrides
- **Key function**: `visit()` - evaluates assertions with profile resolution
- **How PermanentProfile relates**: No changes needed (uses `profile.is_active()` polymorphically)

**Evaluation flow**:
1. Visit assertion node
2. Resolve profile overrides: call `resolve_overrides()`
3. Skip if disabled by profile
4. Evaluate metric expression
5. Apply metric multiplier and validator
6. Store result (PASSED/FAILED/SKIPPED)

### Profile Resolution (`resolve_overrides()`)

**Location**: `src/dqx/profiles.py` (lines 200-238)

**Logic**:
```python
def resolve_overrides(check_name, assertion, profiles, target_date):
    result = ResolvedOverrides()  # Start with defaults

    for profile in profiles:
        if not profile.is_active(target_date):  # <-- PermanentProfile always True
            continue

        for rule in profile.rules:
            if rule matches assertion:
                # Compound effects:
                if rule.disabled: result.disabled = True
                result.metric_multiplier *= rule.metric_multiplier
                if rule.severity: result.severity = rule.severity  # Last wins

    return result
```

**Key for PermanentProfile**: When `is_active()` returns `True`, all rules apply regardless of date.

## Code Patterns to Follow

**IMPORTANT**: All patterns reference AGENTS.md standards.

### Pattern 1: Frozen Dataclass for Immutable Data

**When to use**: Data structures that shouldn't change after creation (profiles, rules, results).

**Example from DQX** (2-5 lines only):
```python
@dataclass(frozen=True)
class SeasonalProfile:
    name: str
    start_date: date
    end_date: date
    rules: list[Rule] = field(default_factory=list)
```

**Reference**: See AGENTS.md §dataclasses for complete details

**Apply to PermanentProfile**: Use exact same pattern, omit date fields:
```python
@dataclass(frozen=True)
class PermanentProfile:
    name: str
    rules: list[Rule] = field(default_factory=list)
```

**Critical**: Use `field(default_factory=list)` for mutable defaults (not `rules: list[Rule] = []`).

### Pattern 2: Protocol-based Interfaces

**Example**:
```python
@runtime_checkable
class Profile(Protocol):
    name: str

    def is_active(self, target_date: date) -> bool: ...

    @property
    def rules(self) -> list[Rule]: ...
```

**Reference**: AGENTS.md §type-hints

**Apply to PermanentProfile**: Dataclass naturally satisfies protocol through structural typing:
- `name: str` field satisfies `name` attribute
- `is_active()` method satisfies protocol method
- `rules` field satisfies `rules` property (dataclass fields are properties)

### Pattern 3: Google-Style Docstrings

**Example**:
```python
def is_active(self, target_date: date) -> bool:
    """Return True if profile is active on the given date.

    Args:
        target_date: Date to check activation against.

    Returns:
        True if target_date falls within [start_date, end_date].
    """
```

**Reference**: AGENTS.md §docstrings (Google style)

**Apply to PermanentProfile**: Document that `target_date` is ignored:
```python
def is_active(self, target_date: date) -> bool:
    """Return True (always active).

    Args:
        target_date: Date to check activation (ignored for permanent profiles).

    Returns:
        Always True.
    """
```

### Pattern 4: Error Messages with Context

**Example from config.py**:
```python
raise DQXError(
    f"Profile '{name}': 'start_date' must be in ISO 8601 format (YYYY-MM-DD). "
    f"Got: {start_date_str}"
)
```

**Reference**: AGENTS.md §error-handling

**Apply to PermanentProfile**: Clear, actionable errors:
```python
if "start_date" in profile_dict:
    raise DQXError(
        f"Profile '{name}': 'start_date' not allowed for permanent profiles. "
        f"Remove date fields or use type: 'seasonal'"
    )
```

### Pattern 5: Type Hints and Forward References

**Example**:
```python
from __future__ import annotations  # Enable forward references

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dqx.graph.nodes import AssertionNode
```

**Reference**: AGENTS.md §type-hints

**Apply to PermanentProfile**: Already handled in `profiles.py` - no circular imports needed.

## Code Standards Reference

**All code must follow AGENTS.md standards**:
- **Import Order**: AGENTS.md §import-order (stdlib → third-party → local)
- **Type Hints**: AGENTS.md §type-hints (strict mode, use `|` for unions)
- **Docstrings**: AGENTS.md §docstrings (Google style, all public APIs)
- **Testing**: AGENTS.md §testing-standards (100% coverage)
- **Coverage**: AGENTS.md §coverage-requirements (no exceptions)

## Testing Patterns

**Reference**: AGENTS.md §testing-patterns

### Test Organization

DQX uses class-based test organization:

```python
class TestPermanentProfile:
    """Tests for PermanentProfile."""

    def test_creation_with_name_and_rules(self) -> None:
        """Test creating profile with name and rules."""
        # Arrange, Act, Assert

    def test_is_active_always_true(self) -> None:
        """Test is_active() returns True for any date."""
        # Test logic
```

**For PermanentProfile**:
- Create `TestPermanentProfile` class in `tests/test_profiles.py`
- Add tests to existing `TestProfileLoading` in `tests/test_config.py`
- Add tests to existing `TestEvaluatorWithProfiles` in `tests/test_profiles.py`

### Fixture Pattern: Temporary Config Files

```python
def test_load_profile(self, tmp_path: Path) -> None:
    """Test loading profile from YAML."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
tunables: {}
profiles:
  - name: "Test"
    type: "permanent"
    rules: []
""")

    config = load_config(config_file)
    profiles = load_profiles_from_config(config)

    assert len(profiles) == 1
    assert profiles[0].name == "Test"
```

**For PermanentProfile**: Follow exact same pattern for config tests.

### Evaluator Testing Pattern

```python
def test_evaluator_with_profile(self) -> None:
    """Test evaluator applies profile overrides."""
    # 1. Create profile
    profile = PermanentProfile(
        name="Test",
        rules=[tag("xmas").set(metric_multiplier=2.0)],
    )

    # 2. Create evaluator with profile
    evaluator, _ = self._create_evaluator_with_metric(
        symbol=x,
        metric_value=60.0,
        target_date=date(2024, 12, 25),
        profiles=[profile],
    )

    # 3. Create assertion node
    assertion_node = AssertionNode(...)

    # 4. Visit and assert
    evaluator.visit(assertion_node)
    assert assertion_node._result == "PASSED"
```

**For PermanentProfile**: Use existing `_create_evaluator_with_metric()` helper in test class.

## Common Pitfalls

### Pitfall 1: Mutable Default Arguments

**Problem**: Using `rules: list[Rule] = []` shares list across instances.

```python
# WRONG - mutable default
@dataclass
class PermanentProfile:
    rules: list[Rule] = []  # Shared across all instances!

# CORRECT - factory function
@dataclass
class PermanentProfile:
    rules: list[Rule] = field(default_factory=list)
```

**Solution**: Always use `field(default_factory=list)` for mutable defaults.

### Pitfall 2: Modifying Frozen Dataclass

**Problem**: `frozen=True` makes instances immutable - can't assign after creation.

```python
profile = PermanentProfile(name="Test")
profile.name = "New Name"  # ERROR: cannot assign to frozen dataclass
```

**Solution**: This is intentional - profiles should be immutable. Create new instance if needed.

### Pitfall 3: Circular Imports

**Problem**: DQX has complex dependencies between modules.

**Solution**: Use `TYPE_CHECKING` pattern (see AGENTS.md §type-hints):
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dqx.graph.nodes import AssertionNode
```

**For PermanentProfile**: Already handled in `profiles.py` - no new imports needed.

### Pitfall 4: Incorrect Protocol Compliance

**Problem**: Missing required attributes/methods from `Profile` protocol.

```python
# WRONG - missing is_active()
@dataclass
class PermanentProfile:
    name: str
    rules: list[Rule]
    # Missing: is_active() method!
```

**Solution**: Implement all protocol members:
- `name: str` (attribute)
- `is_active(target_date: date) -> bool` (method)
- `rules` (property or attribute)

### Pitfall 5: Type Validation in Config Loading

**Problem**: YAML parsers return `str`, not typed values.

```python
# profile_type is str from YAML
if profile_type == "permanent":  # CORRECT - string comparison
```

**Solution**: Always compare with string literals in config loading code.

### Pitfall 6: Date Import Source

**Problem**: Using wrong `date` type.

```python
from datetime import datetime  # WRONG
from datetime import date       # CORRECT
```

**Solution**: Import `date` from `datetime` module (see existing imports in `profiles.py`).

## Related PRs and Issues

**Similar features**:
- Initial Profile system implementation - Added `SeasonalProfile` and Rule system
- Config loading for profiles - Added YAML parsing support

**Pattern to follow**: Look at `SeasonalProfile` as reference implementation:
- Same structure (frozen dataclass)
- Same protocol compliance approach
- Same integration points (no changes to evaluator/suite)

## Implementation Sequence

### Phase 1: Core Class
1. Add `PermanentProfile` dataclass after `SeasonalProfile` in `profiles.py`
2. Implement `is_active()` to return `True`
3. Write unit tests in `tests/test_profiles.py`

### Phase 2: Config Support
1. Modify `load_profiles_from_config()` in `config.py`
2. Add conditional branch for `type: "permanent"`
3. Validate no date fields present
4. Write config loading tests in `tests/test_config.py`

### Phase 3: Integration
1. Write evaluator integration tests
2. Write E2E tests with full suite
3. Test profile compounding (permanent + seasonal)

### Phase 4: Documentation
1. Enhance docstrings with examples
2. Update module documentation
3. Add to API reference (if exists)

## Documentation

After implementation, update:
- `src/dqx/profiles.py` - Docstrings with usage examples
- `docs/api-reference.md` - API documentation (if exists)
- Inline docstrings (AGENTS.md §docstrings - Google style required)

## YAML Config Format Examples

### Basic Permanent Profile

```yaml
profiles:
  - name: "Production Baseline"
    type: "permanent"
    rules:
      - action: "disable"
        target: "check"
        name: "Dev Only Check"
```

### Mixed Profile Types

```yaml
profiles:
  # Permanent profile (always active)
  - name: "Account Baseline"
    type: "permanent"
    rules:
      - action: "scale"
        target: "tag"
        name: "volume"
        multiplier: 0.8

  # Seasonal profile (date-based)
  - name: "Holiday Season"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "scale"
        target: "tag"
        name: "volume"
        multiplier: 2.0
```

**Note**: When both profiles active, multipliers compound: `0.8 * 2.0 = 1.6x`

### Multiple Rules

```yaml
profiles:
  - name: "Production Config"
    type: "permanent"
    rules:
      - action: "disable"
        target: "check"
        name: "Experimental Check"

      - action: "set_severity"
        target: "tag"
        name: "critical"
        severity: "P0"

      - action: "scale"
        target: "tag"
        name: "volume"
        multiplier: 0.9
```

## Python API Examples

### Basic Usage

```python
from dqx.profiles import PermanentProfile, check, tag

baseline = PermanentProfile(
    name="Production Baseline",
    rules=[
        check("Dev Check").disable(),
        tag("critical").set(severity="P0"),
    ],
)
```

### With VerificationSuite

```python
from dqx.api import VerificationSuite
from dqx.profiles import PermanentProfile, SeasonalProfile

# Mix profile types
permanent = PermanentProfile(name="Baseline", rules=[...])
seasonal = SeasonalProfile(
    name="Holiday",
    start_date=date(2024, 12, 20),
    end_date=date(2025, 1, 5),
    rules=[...],
)

suite = VerificationSuite(
    checks=[...],
    db=db,
    profiles=[permanent, seasonal],  # Both types work together
)
```

### From Config File

```python
suite = VerificationSuite(
    checks=[...],
    db=db,
    config=Path("config.yaml"),  # Loads permanent + seasonal profiles
)
```

## Key Differences: PermanentProfile vs SeasonalProfile

| Aspect | PermanentProfile | SeasonalProfile |
|--------|------------------|-----------------|
| **Activation** | Always active | Date range |
| **Fields** | `name`, `rules` | `name`, `start_date`, `end_date`, `rules` |
| **is_active()** | Returns `True` | Checks `start_date <= target <= end_date` |
| **Use Case** | Baseline config | Seasonal adjustments |
| **Config Type** | `type: "permanent"` | `type: "seasonal"` (or default) |

## Testing Checklist

### Unit Tests
- [ ] Create with name and rules
- [ ] Create with default empty rules
- [ ] `is_active()` returns `True` for any date
- [ ] `is_active()` ignores target_date parameter
- [ ] `rules` property returns correct list
- [ ] Protocol compliance (isinstance check)
- [ ] Frozen dataclass (cannot modify after creation)

### Config Tests
- [ ] Load from YAML with `type: "permanent"`
- [ ] Load with various rule types (disable, scale, set_severity)
- [ ] Load mixed permanent + seasonal profiles
- [ ] Error: permanent with start_date
- [ ] Error: permanent with end_date
- [ ] Error: unknown profile type
- [ ] Error: missing name
- [ ] Error: unknown fields

### Integration Tests
- [ ] Evaluator applies metric_multiplier
- [ ] Evaluator skips disabled assertions
- [ ] Evaluator overrides severity
- [ ] Permanent + seasonal profiles compound
- [ ] Multiple permanent profiles compound
- [ ] Works with tag selectors
- [ ] Works with assertion selectors
- [ ] Always active regardless of date

### E2E Tests
- [ ] Full suite with permanent profile from config
- [ ] Permanent profile affects assertion results
- [ ] Mixed permanent + seasonal from config

## Quick Reference Commands

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_profiles.py

# Run specific test class
uv run pytest tests/test_profiles.py::TestPermanentProfile

# Run with coverage
uv run pytest --cov=src/dqx --cov-report=term-missing

# Format code
uv run ruff format

# Lint and fix
uv run ruff check --fix

# Type check
uv run mypy src tests

# Pre-commit hooks
uv run pre-commit run --all-files
```

## Summary

`PermanentProfile` is a straightforward addition to DQX's profile system:

1. **Simple**: Dataclass with `name` and `rules` (no dates)
2. **Consistent**: Follows exact same pattern as `SeasonalProfile`
3. **Integrated**: Works seamlessly with existing infrastructure
4. **Testable**: Clear test patterns from existing profile tests
5. **Documented**: Rich examples for both YAML and Python API

**Key insight**: `is_active()` always returns `True` - that's the entire distinction from `SeasonalProfile`.

Everything else (Rule system, selectors, compounding, config loading) works identically.
