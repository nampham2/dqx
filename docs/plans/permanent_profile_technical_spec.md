# PermanentProfile Technical Specification

## Problem Statement

DQX currently supports `SeasonalProfile` for date-based configuration changes (e.g., holiday season adjustments). However, users need a simpler profile type that is always active, regardless of date. This is required for:

1. **Account-Specific Configurations**: Different accounts may require different baseline thresholds or disabled checks
2. **Environment-Specific Rules**: Development, staging, and production environments need different validation rules
3. **Permanent Overrides**: Some checks need permanent modifications without managing date ranges

The `PermanentProfile` feature addresses this by providing a profile that is always active, using the same Rule system as `SeasonalProfile` but without date-based activation logic.

## Architecture Decisions

### Decision 1: Implement as Dataclass (Not Protocol Implementation)

**Rationale**: Follow the same pattern as `SeasonalProfile` - implement as a frozen dataclass that naturally satisfies the `Profile` protocol through structural subtyping.

**Alternatives considered**:
- Explicit Protocol implementation with inheritance - Rejected because DQX uses structural typing (Protocol) for flexibility
- Shared base class - Rejected to avoid tight coupling and maintain simplicity

**Benefits**:
- Consistent with existing `SeasonalProfile` design
- Leverages Python's structural subtyping (duck typing)
- Simpler implementation (no inheritance hierarchy)
- Easy to test and maintain

### Decision 2: `is_active()` Always Returns `True`

**Rationale**: The defining characteristic of `PermanentProfile` is that it's always active. The `target_date` parameter is accepted for protocol compliance but ignored.

**Alternatives considered**:
- Remove `target_date` parameter - Rejected because it would violate the `Profile` protocol
- Add optional date filtering - Rejected because this is `SeasonalProfile`'s purpose

### Decision 3: Config Type Identifier: `"permanent"`

**Rationale**: Use explicit type field in YAML to distinguish from `SeasonalProfile`.

**Config format**:
```yaml
profiles:
  - name: "Production Baseline"
    type: "permanent"  # New type identifier
    rules:
      - action: "disable"
        target: "check"
        name: "Development Only Check"
```

**Alternatives considered**:
- Infer type from absence of dates - Rejected because it's implicit and error-prone
- Use separate config section - Rejected to maintain unified profile list

### Decision 4: Share Existing Rule System

**Rationale**: Reuse `Rule`, `Selector`, `RuleBuilder` infrastructure without modification.

**Benefits**:
- No code duplication
- Consistent API across profile types
- Same validation and error handling
- Profiles compound together naturally

## API Design

### PermanentProfile Class

```python
@dataclass(frozen=True)
class PermanentProfile:
    """Profile that is always active (no date-based activation).

    Useful for account-specific or environment-specific baseline configurations
    that don't change seasonally.

    Example:
        baseline = PermanentProfile(
            name="Production Baseline",
            rules=[
                check("Dev Only Check").disable(),
                tag("critical").set(severity="P0"),
            ],
        )

    Attributes:
        name: Descriptive name for the profile.
        rules: List of rules to apply when profile is active.
    """

    name: str
    rules: list[Rule] = field(default_factory=list)

    def is_active(self, target_date: date) -> bool:
        """Return True (always active).

        Args:
            target_date: Date to check activation (ignored for permanent profiles).

        Returns:
            Always True.
        """
        return True
```

### Python API Usage

```python
from dqx.profiles import PermanentProfile, check, tag

# Create permanent profile
baseline = PermanentProfile(
    name="Production Baseline",
    rules=[
        check("Dev Only Check").disable(),
        tag("critical").set(severity="P0"),
        tag("volume").set(metric_multiplier=0.9),
    ],
)

# Use in VerificationSuite
suite = VerificationSuite(
    checks=[...],
    db=db,
    profiles=[baseline],  # Mix with SeasonalProfile instances
)
```

### YAML Config API

```yaml
tunables: {}

profiles:
  # Permanent profile (always active)
  - name: "Production Baseline"
    type: "permanent"
    rules:
      - action: "disable"
        target: "check"
        name: "Dev Only Check"

      - action: "set_severity"
        target: "tag"
        name: "critical"
        severity: "P0"

      - action: "scale"
        target: "tag"
        name: "volume"
        multiplier: 0.9

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

## Data Structures

### Type Aliases

No new type aliases needed. Uses existing `Profile` protocol.

### Protocol Compliance

`PermanentProfile` satisfies the `Profile` protocol:

```python
@runtime_checkable
class Profile(Protocol):
    name: str

    def is_active(self, target_date: date) -> bool: ...

    @property
    def rules(self) -> list[Rule]: ...
```

## Integration Points

### 1. `src/dqx/profiles.py`

**Location**: After `SeasonalProfile` definition (line ~118)

**Changes**:
- Add `PermanentProfile` dataclass
- No changes to existing code
- Exports: Add `PermanentProfile` to `__all__` (if present)

### 2. `src/dqx/config.py`

**Function**: `load_profiles_from_config()` (lines 281-401)

**Changes**:
- Add type validation for `"permanent"` (line ~338)
- Add parsing branch for permanent profiles
- Validate no date fields present in permanent profiles
- Validate no unknown fields

**Pseudo-code**:
```python
profile_type = profile_dict.get("type", "seasonal")

if profile_type == "seasonal":
    # Existing SeasonalProfile parsing...
    pass
elif profile_type == "permanent":
    # New PermanentProfile parsing
    # - Validate no start_date/end_date
    # - Parse rules (same logic)
    # - Create PermanentProfile
    pass
else:
    raise DQXError(f"Unknown profile type: {profile_type}")
```

### 3. `src/dqx/api.py`

**Location**: `VerificationSuite.__init__()` (lines 908-936)

**Changes**: None required - accepts any `Profile` protocol

**Validation**: Existing duplicate name check works for both types

### 4. `src/dqx/evaluator.py`

**Location**: `Evaluator.visit()` (lines 309-376)

**Changes**: None required - uses `profile.is_active()` polymorphically

### 5. `tests/test_profiles.py`

**New test class**: `TestPermanentProfile`

**Coverage**:
- Basic creation
- `is_active()` always returns True
- Rules property
- Protocol compliance

### 6. `tests/test_config.py`

**New tests in** `TestProfileLoading`:
- Load single permanent profile
- Load permanent + seasonal mix
- Error: permanent profile with start_date (invalid)
- Error: permanent profile with end_date (invalid)
- Error: unknown type

## Backward Compatibility

### Config Files

**Backward compatible**: Existing configs with only `seasonal` profiles work unchanged.

**Migration**: No migration needed. New `type: "permanent"` field is opt-in.

**Default behavior**: `type` defaults to `"seasonal"` (existing behavior preserved).

### Python API

**Backward compatible**: Existing code using `SeasonalProfile` continues working.

**Type hints**: `profiles: list[Profile]` accepts both types (structural subtyping).

## Error Handling and Validation

### Config Validation Errors

#### Invalid: Permanent profile with date fields

```yaml
profiles:
  - name: "Invalid"
    type: "permanent"
    start_date: "2024-01-01"  # ERROR
    rules: []
```

**Error**: `DQXError: Profile 'Invalid': 'start_date' not allowed for permanent profiles`

#### Invalid: Unknown type

```yaml
profiles:
  - name: "Invalid"
    type: "temporary"  # ERROR
    rules: []
```

**Error**: `DQXError: Profile 'Invalid': unknown type 'temporary' (must be 'seasonal' or 'permanent')`

#### Invalid: Missing name

```yaml
profiles:
  - type: "permanent"  # ERROR - no name
    rules: []
```

**Error**: `DQXError: Profile at index 0: 'name' is required`

### Python API Validation

#### Empty name

```python
PermanentProfile(name="", rules=[])  # Allowed by dataclass (validation in suite)
```

**Validation**: Existing suite-level validation catches empty names.

#### Invalid rules

```python
PermanentProfile(
    name="Test",
    rules=[Rule(selector=TagSelector("test"), metric_multiplier=-1.0)]  # Invalid
)
```

**Validation**: Existing rule validation in config parser. For Python API, validation occurs during evaluation (multiplier must be positive).

## Performance Considerations

### Minimal Overhead

- `is_active()` returns `True` immediately (no date arithmetic)
- Faster than `SeasonalProfile.is_active()` which checks date range
- No additional memory overhead

### Profile Resolution

- Permanent profiles are checked first (always active)
- Seasonal profiles checked conditionally (date-based)
- Order matters for rule compounding (existing behavior)

**Optimization opportunity**: Could separate permanent profiles to avoid repeated `is_active()` checks, but premature optimization (not implemented).

## Non-Goals

### Explicitly out of scope:

1. **Conditional activation** - Use `SeasonalProfile` for date-based conditions
2. **Priority/ordering semantics** - Profiles apply in list order (existing behavior)
3. **Profile groups/categories** - Single flat list of profiles
4. **Profile inheritance** - Each profile is independent
5. **Runtime enable/disable** - Profiles are set at suite creation

## Security Considerations

None. `PermanentProfile` has same security posture as `SeasonalProfile`:
- No external data access
- No code execution beyond validation functions
- Config files trusted (user-provided)

## Testing Strategy

### Unit Tests

1. `PermanentProfile` creation and attributes
2. `is_active()` always returns `True` (various dates)
3. Rules property access
4. Protocol compliance verification

### Integration Tests

1. Load from YAML config
2. Mix permanent + seasonal profiles
3. Profile compounding with both types
4. Error cases (invalid config)

### End-to-End Tests

1. Full suite with permanent profile
2. Assertions disabled by permanent profile
3. Metric multipliers from permanent + seasonal
4. Severity overrides

## Documentation Updates

### Files to update:

1. `src/dqx/profiles.py` - Add docstring examples
2. `docs/guides/profiles.md` - Add PermanentProfile section (if exists)
3. `docs/api-reference.md` - Document PermanentProfile class
4. Config example files in `examples/` (if exists)

### Example documentation:

```markdown
## Permanent Profiles

Use `PermanentProfile` for baseline configurations that don't change over time:

- Account-specific rule adjustments
- Environment-specific overrides (dev/staging/prod)
- Permanently disabled legacy checks

Unlike `SeasonalProfile`, permanent profiles are always active regardless of date.
```

## Migration Guide

**No migration needed** - feature is purely additive.

Users can opt-in by:
1. Adding `type: "permanent"` to config files
2. Using `PermanentProfile` class in Python API

## Open Questions

None. Design is straightforward extension of existing pattern.

## Future Enhancements

Potential future additions (not in scope):
1. Conditional permanent profiles (e.g., based on environment variable)
2. Profile validation at config load time (e.g., check referenced checks exist)
3. Profile debugging/inspection tools

## Summary

`PermanentProfile` is a minimal, focused addition that:
- Follows existing patterns (`SeasonalProfile` as template)
- Reuses existing infrastructure (Rule system, config loading)
- Maintains backward compatibility
- Requires no changes to core evaluation logic
- Provides clear separation between permanent and seasonal rules

Implementation complexity: **Low** - primarily adding a new dataclass and config parsing branch.
