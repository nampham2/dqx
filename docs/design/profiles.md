# Profiles: Configurable Data Quality Rules

## Problem

Data quality checks produce false positives during holiday seasons. Order volumes drop. User behavior shifts. Metrics that pass on normal days fail on Christmas.

Teams need two capabilities:

1. **Batch threshold changes** — Relax validation rules during specific date ranges
2. **Check enable/disable** — Turn off checks that do not apply during holidays

## Solution Overview

Profiles let users define rules that modify assertion behavior during specific periods. A profile activates based on the current date and applies threshold overrides or disables assertions.

```python
christmas = HolidayProfile(
    name="Christmas 2024",
    start_date=date(2024, 12, 20),
    end_date=date(2025, 1, 5),
    rules=[
        tag("xmas").set(threshold_multiplier=2.0),
        assertion("Volume Check", "Daily orders above minimum").disable(),
    ],
)

suite = VerificationSuite(
    checks=[volume_check, quality_check], db=db, name="My Suite", profiles=[christmas]
)
```

## Key Concepts

### DQX Background

DQX validates data quality through three constructs:

| Construct | Purpose | Example |
|-----------|---------|---------|
| **Check** | Groups related assertions | `@check(name="Volume Check")` |
| **Assertion** | Single validation rule | `ctx.assert_that(orders).is_gt(100)` |
| **Threshold** | Pass/fail boundary | The `100` in `is_gt(100)` |

A check contains one or more assertions. Each assertion compares a metric against a threshold.

```python
@check(name="Volume Check")
def volume_check(mp, ctx):
    ctx.assert_that(mp.sum("orders")).where(name="Daily orders above minimum").is_gt(
        100
    )  # threshold = 100
```

### Profile

A profile activates during a date range and applies rules to matching assertions.

```python
@runtime_checkable
class Profile(Protocol):
    name: str

    def is_active(self, target_date: date) -> bool: ...
    def rules(self) -> list[Rule]: ...
```

The protocol enables future profile types: `MaintenanceProfile`, `RegionProfile`, `ABTestProfile`.

### Rule

A rule selects assertions and specifies what to change.

```python
@dataclass(frozen=True)
class Rule:
    selector: Selector
    disabled: bool = False
    threshold: ThresholdOverride | None = None
```

### Selector

Selectors identify which assertions a rule targets. Two selector types exist:

**AssertionSelector** — Matches by check name and assertion name:

```python
assertion("Volume Check", "Daily orders above minimum")  # exact match
check("Volume Check")  # all assertions in check
```

**TagSelector** — Matches by tag:

```python
tag("xmas")
```

## Detailed Design

### Core Types

Create `src/dqx/profiles.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, runtime_checkable


Selector = AssertionSelector | TagSelector


@dataclass(frozen=True)
class AssertionSelector:
    """Matches assertions by check and assertion name."""

    check: str
    assertion: str | None = None

    def matches(self, check_name: str, assertion_name: str) -> bool:
        if check_name != self.check:
            return False
        if self.assertion is None:
            return True
        return assertion_name == self.assertion


@dataclass(frozen=True)
class TagSelector:
    """Matches assertions by tag."""

    tag: str

    def matches(self, tags: set[str]) -> bool:
        return self.tag in tags


@dataclass(frozen=True)
class ThresholdOverride:
    """Specifies how to modify a threshold."""

    value: float | None = None
    multiplier: float | None = None
    tol: float | None = None


@dataclass(frozen=True)
class Rule:
    """Pairs a selector with an action."""

    selector: Selector
    disabled: bool = False
    threshold: ThresholdOverride | None = None


@runtime_checkable
class Profile(Protocol):
    """Base protocol for all profile types."""

    name: str

    def is_active(self, target_date: date) -> bool: ...

    @property
    def rules(self) -> list[Rule]: ...


@dataclass
class HolidayProfile:
    """Profile active during a date range."""

    name: str
    start_date: date
    end_date: date
    rules: list[Rule] = field(default_factory=list)

    def is_active(self, target_date: date) -> bool:
        return self.start_date <= target_date <= self.end_date
```

### Builder Functions

Builders provide a fluent API for creating rules:

```python
class RuleBuilder:
    """Constructs rules with a fluent interface."""

    def __init__(self, selector: Selector):
        self._selector = selector

    def disable(self) -> Rule:
        return Rule(selector=self._selector, disabled=True)

    def set(
        self,
        *,
        threshold: float | None = None,
        threshold_multiplier: float | None = None,
        tol: float | None = None,
    ) -> Rule:
        return Rule(
            selector=self._selector,
            threshold=ThresholdOverride(
                value=threshold,
                multiplier=threshold_multiplier,
                tol=tol,
            ),
        )


def assertion(check: str, assertion: str | None = None) -> RuleBuilder:
    """Select by check and assertion name (exact match)."""
    return RuleBuilder(AssertionSelector(check=check, assertion=assertion))


def check(name: str) -> RuleBuilder:
    """Select all assertions in a check."""
    return RuleBuilder(AssertionSelector(check=name, assertion=None))


def tag(name: str) -> RuleBuilder:
    """Select assertions with a specific tag."""
    return RuleBuilder(TagSelector(tag=name))
```

### Adding Tags to Assertions

Extend `where()` to accept tags:

```python
# In src/dqx/api.py


def where(
    self,
    *,
    name: str,
    severity: SeverityLevel = "P1",
    tags: set[str] | None = None,  # NEW
) -> AssertionReady: ...
```

Store tags in `AssertionNode`:

```python
# In src/dqx/graph/nodes.py


class AssertionNode(BaseNode["CheckNode"]):
    def __init__(
        self,
        parent: CheckNode,
        actual: sp.Expr,
        name: str,
        validator: SymbolicValidator,
        severity: SeverityLevel = "P1",
        tags: set[str] | None = None,  # NEW
    ) -> None:
        ...
        self.tags = tags or set()
```

### Profile Resolution

The evaluator resolves rules before evaluating each assertion:

```python
@dataclass
class ResolvedOverrides:
    """Accumulated overrides from all matching rules."""

    disabled: bool = False
    threshold_value: float | None = None
    threshold_multiplier: float | None = None
    tol_value: float | None = None


def resolve_overrides(
    check_name: str,
    assertion: AssertionNode,
    profiles: list[Profile],
    target_date: date,
) -> ResolvedOverrides:
    """Apply all matching rules from active profiles."""

    result = ResolvedOverrides()

    for profile in profiles:
        if not profile.is_active(target_date):
            continue

        for rule in profile.rules:
            if not _matches(rule.selector, check_name, assertion):
                continue

            if rule.disabled:
                result.disabled = True

            if rule.threshold:
                _apply_threshold(result, rule.threshold)

    return result


def _matches(
    selector: Selector,
    check_name: str,
    assertion: AssertionNode,
) -> bool:
    match selector:
        case AssertionSelector():
            return selector.matches(check_name, assertion.name)
        case TagSelector():
            return selector.matches(assertion.tags)


def _apply_threshold(result: ResolvedOverrides, override: ThresholdOverride) -> None:
    if override.value is not None:
        result.threshold_value = override.value
    if override.multiplier is not None:
        result.threshold_multiplier = override.multiplier
    if override.tol is not None:
        result.tol_value = override.tol
```

### Rule Ordering

Rules apply in definition order. Later rules override earlier ones:

```python
rules = [
    tag("volume").set(threshold_multiplier=1.5),  # First: multiply by 1.5
    assertion("Check", "Orders").set(threshold=50),  # Second: override to 50
]
```

For the assertion named "Orders" with tag "volume", the final threshold is 50.

## Usage Examples

### Holiday Season Profile

```python
from datetime import date
from dqx.profiles import HolidayProfile, tag, assertion, check

christmas = HolidayProfile(
    name="Christmas 2024",
    start_date=date(2024, 12, 20),
    end_date=date(2025, 1, 5),
    rules=[
        # Relax all xmas assertions by 2x
        tag("xmas").set(threshold_multiplier=2.0),
        # Disable volume checks entirely
        check("Volume Check").disable(),
        # Set specific threshold for one assertion
        assertion("Quality Check", "Error rate below threshold").set(threshold=0.1),
    ],
)
```

### Defining Checks with Tags

```python
@check(name="Volume Check")
def volume_check(mp, ctx):
    ctx.assert_that(mp.sum("orders")).where(
        name="Daily orders above minimum", tags={"volume", "xmas"}
    ).is_gt(100)


@check(name="Quality Check")
def quality_check(mp, ctx):
    ctx.assert_that(mp.average("error_rate")).where(
        name="Error rate below threshold", tags={"quality"}
    ).is_lt(0.05)
```

### Running with Profiles

```python
suite = VerificationSuite(
    checks=[volume_check, quality_check],
    db=db,
    name="Daily Checks",
    profiles=[christmas],
)

key = ResultKey(date(2024, 12, 25), tags={})
suite.run(datasources, key)  # Profile activates, rules apply
```

### Multiple Profiles

```python
christmas = HolidayProfile(
    name="Christmas",
    start_date=date(2024, 12, 20),
    end_date=date(2025, 1, 5),
    rules=[tag("xmas").set(threshold_multiplier=2.0)],
)

black_friday = HolidayProfile(
    name="Black Friday",
    start_date=date(2024, 11, 29),
    end_date=date(2024, 12, 2),
    rules=[check("Volume Check").set(threshold_multiplier=3.0)],
)

suite = VerificationSuite(
    checks=[...],
    db=db,
    name="Suite",
    profiles=[christmas, black_friday],  # Both evaluated
)
```

## Files to Modify

| File | Change |
|------|--------|
| `src/dqx/profiles.py` | Create: Profile protocol, selectors, rules, builders |
| `src/dqx/api.py` | Add `tags` to `where()`, add `profiles` to `VerificationSuite` |
| `src/dqx/graph/nodes.py` | Add `tags` to `AssertionNode` |
| `src/dqx/evaluator.py` | Resolve and apply overrides before evaluation |
| `src/dqx/__init__.py` | Export profile types |

## Testing Strategy

### Unit Tests

1. **Selector matching** — Verify exact name matching and tag matching
2. **Rule application** — Confirm overrides apply correctly
3. **Profile activation** — Test date range logic
4. **Rule ordering** — Validate later rules override earlier ones

### Integration Tests

1. **Profile disables assertion** — Assertion skipped when rule disables it
2. **Profile modifies threshold** — Assertion uses overridden threshold
3. **Multiple profiles** — Rules from all active profiles apply
4. **No active profile** — Assertions evaluate normally

## Future Extensions

The `Profile` protocol enables additional profile types:

- **MaintenanceProfile** — Relax checks during scheduled maintenance
- **RegionProfile** — Apply region-specific thresholds
- **ABTestProfile** — Test different threshold configurations

Each implements `is_active()` and `rules` with its own activation logic.
