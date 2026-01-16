# Profiles: Configurable Data Quality Rules

## Problem

Data quality checks fail during holiday seasons. Order volumes drop. User behavior shifts. Metrics that pass on normal days fail on Christmas.

Teams need three capabilities:

1. **Disable checks** that do not apply during holidays
2. **Compensate metrics** by scaling values to account for expected changes
3. **Adjust severity** to reduce alert noise during known disruption periods

## Solution

Profiles modify assertion behavior during specific periods. A profile activates based on the current date and applies rules: disable assertions, scale metric values, or adjust severity.

```python
christmas = HolidayProfile(
    name="Christmas 2024",
    start_date=date(2024, 12, 20),
    end_date=date(2025, 1, 5),
    rules=[
        tag("xmas").set(metric_multiplier=2.0),
        tag("non-critical").set(severity="P3"),  # Downgrade during holidays
        check("Volume Check").disable(),
    ],
)

suite = VerificationSuite(
    checks=[volume_check, quality_check], db=db, name="My Suite", profiles=[christmas]
)
```

## Key Concepts

### DQX Constructs

| Construct | Purpose | Example |
|-----------|---------|---------|
| **Check** | Groups related assertions | `@check(name="Volume Check")` |
| **Assertion** | Single validation rule | `ctx.assert_that(orders).is_gt(100)` |
| **Threshold** | Pass/fail boundary | The `100` in `is_gt(100)` |

A check contains one or more assertions. Each assertion compares a metric against a threshold.

```python
@check(name="Volume Check")
def volume_check(mp, ctx):
    ctx.assert_that(mp.sum("orders")).where(
        name="Daily orders above minimum", tags={"volume", "xmas"}
    ).is_gt(100)
```

### Profile

A profile activates during a date range and applies rules to matching assertions.

```python
@runtime_checkable
class Profile(Protocol):
    name: str

    def is_active(self, target_date: date) -> bool: ...

    @property
    def rules(self) -> list[Rule]: ...
```

The protocol enables future profile types with different activation logic.

### Rule

A rule selects assertions and specifies an action.

```python
@dataclass(frozen=True)
class Rule:
    selector: Selector
    disabled: bool = False
    metric_multiplier: float = 1.0
    severity: SeverityLevel | None = None
```

### Selector

Selectors identify which assertions a rule targets.

**AssertionSelector** matches by check name and assertion name:

```python
assertion("Volume Check", "Daily orders above minimum")  # exact match
check("Volume Check")  # all assertions in check
```

**TagSelector** matches by tag:

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

    def matches(self, tags: frozenset[str]) -> bool:
        return self.tag in tags


@dataclass(frozen=True)
class Rule:
    """Pairs a selector with an action."""

    selector: Selector
    disabled: bool = False
    metric_multiplier: float = 1.0
    severity: SeverityLevel | None = None


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
        self, *, metric_multiplier: float = 1.0, severity: SeverityLevel | None = None
    ) -> Rule:
        return Rule(
            selector=self._selector,
            metric_multiplier=metric_multiplier,
            severity=severity,
        )


def assertion(check: str, name: str | None = None) -> RuleBuilder:
    """Select by check and assertion name."""
    return RuleBuilder(AssertionSelector(check=check, assertion=name))


def check(name: str) -> RuleBuilder:
    """Select all assertions in a check."""
    return RuleBuilder(AssertionSelector(check=name, assertion=None))


def tag(name: str) -> RuleBuilder:
    """Select assertions with a specific tag."""
    return RuleBuilder(TagSelector(tag=name))
```

### Profile Resolution

The evaluator resolves rules before evaluating each assertion:

```python
@dataclass
class ResolvedOverrides:
    """Accumulated overrides from all matching rules."""

    disabled: bool = False
    metric_multiplier: float = 1.0
    severity: SeverityLevel | None = None


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

            result.metric_multiplier *= rule.metric_multiplier

            if rule.severity is not None:
                result.severity = rule.severity

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
```

### Metric Multiplier

The multiplier scales the computed metric value before comparison. This compensates for expected metric changes during the profile period.

**Example:** Orders drop 50% during Christmas.

```python
# Assertion: orders > 100
# Christmas day: orders = 60

# Without profile: 60 > 100 → FAILED
# With metric_multiplier=2.0: 60 × 2.0 = 120 > 100 → PASSED
```

The evaluator applies the multiplier:

```python
# In Evaluator.visit():
match node._metric:
    case Success(value):
        adjusted = value * overrides.metric_multiplier
        passed = node.validator.fn(adjusted)
        node._result = "PASSED" if passed else "FAILED"
```

### Severity Override

The severity override changes an assertion's priority level. Use it to reduce alert noise during periods of expected disruption.

**Example:** Non-critical checks trigger pages during normal operations but should only log during holidays.

```python
# Assertion defined with severity P1
ctx.assert_that(orders).where(name="Order count", severity="P1", tags={"non-critical"})

# During Christmas, downgrade to P3
tag("non-critical").set(severity="P3")
```

The last matching rule determines severity. Unlike multipliers, severities do not compound.

```python
rules = [
    tag("volume").set(severity="P2"),
    tag("xmas").set(severity="P3"),
]
# Assertion with both tags: severity = P3 (last match wins)
```

### Rule Ordering

Rules apply in definition order. Later rules compound with earlier ones (multipliers multiply, severity uses last match):

```python
rules = [
    tag("volume").set(metric_multiplier=1.5),
    assertion("Check", "Orders").set(metric_multiplier=2.0),
]
# For "Orders" with tag "volume": multiplier = 1.5 × 2.0 = 3.0
```

## Usage Examples

### Holiday Profile

```python
from datetime import date
from dqx.profiles import HolidayProfile, tag, assertion, check

christmas = HolidayProfile(
    name="Christmas 2024",
    start_date=date(2024, 12, 20),
    end_date=date(2025, 1, 5),
    rules=[
        tag("xmas").set(metric_multiplier=2.0),
        check("Volume Check").disable(),
    ],
)
```

### Checks with Tags

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
    rules=[tag("xmas").set(metric_multiplier=2.0)],
)

black_friday = HolidayProfile(
    name="Black Friday",
    start_date=date(2024, 11, 29),
    end_date=date(2024, 12, 2),
    rules=[check("Volume Check").set(metric_multiplier=3.0)],
)

suite = VerificationSuite(
    checks=[...],
    db=db,
    name="Suite",
    profiles=[christmas, black_friday],
)
```

## Files to Modify

| File | Change |
|------|--------|
| `src/dqx/profiles.py` | Create: Profile protocol, selectors, rules, builders |
| `src/dqx/api.py` | Add `profiles` to `VerificationSuite` |
| `src/dqx/evaluator.py` | Resolve and apply `metric_multiplier` before validation |
| `src/dqx/__init__.py` | Export profile types |

## Testing Strategy

### Unit Tests

1. **Selector matching** — AssertionSelector and TagSelector match correctly
2. **Rule application** — metric_multiplier compounds across rules
3. **Profile activation** — date range logic works
4. **Rule ordering** — later rules compound with earlier ones

### Integration Tests

1. **Disabled assertion** — assertion skipped when rule disables it
2. **Scaled metric** — assertion uses adjusted value
3. **Multiple profiles** — rules from all active profiles apply
4. **No active profile** — assertions evaluate normally

## Future Extensions

The `Profile` protocol enables additional profile types:

- **MaintenanceProfile** — relax checks during scheduled maintenance
- **RegionProfile** — apply region-specific multipliers
- **ABTestProfile** — test different configurations

Each implements `is_active()` and `rules` with its own activation logic.
