"""Profiles for configurable data quality rules.

Profiles modify assertion behavior during specific periods. A profile activates
based on the current date and applies rules: disable assertions or scale metric values.

Example:
    christmas = HolidayProfile(
        name="Christmas 2024",
        start_date=date(2024, 12, 20),
        end_date=date(2025, 1, 5),
        rules=[
            tag("xmas").set(metric_multiplier=2.0),
            check("Volume Check").disable(),
        ],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dqx.graph.nodes import AssertionNode


@dataclass(frozen=True)
class AssertionSelector:
    """Matches assertions by check and assertion name."""

    check: str
    assertion: str | None = None

    def matches(self, check_name: str, assertion_name: str) -> bool:
        """Return True if selector matches the given check and assertion names."""
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
        """Return True if selector's tag is in the given tags set."""
        return self.tag in tags


Selector = AssertionSelector | TagSelector


@dataclass(frozen=True)
class Rule:
    """Pairs a selector with an action.

    Attributes:
        selector: Identifies which assertions this rule targets.
        disabled: If True, matching assertions are skipped.
        metric_multiplier: Scales the metric value before comparison.
    """

    selector: Selector
    disabled: bool = False
    metric_multiplier: float = 1.0


@runtime_checkable
class Profile(Protocol):
    """Base protocol for all profile types.

    Profiles modify assertion behavior during specific periods.
    Implement this protocol to create custom profile types.
    """

    name: str

    def is_active(self, target_date: date) -> bool:
        """Return True if profile is active on the given date."""
        ...

    @property
    def rules(self) -> list[Rule]:
        """Return the list of rules this profile applies."""
        ...


@dataclass
class HolidayProfile:
    """Profile active during a date range.

    Example:
        christmas = HolidayProfile(
            name="Christmas 2024",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[tag("xmas").set(metric_multiplier=2.0)],
        )
    """

    name: str
    start_date: date
    end_date: date
    rules: list[Rule] = field(default_factory=list)

    def is_active(self, target_date: date) -> bool:
        """Return True if target_date falls within the profile's date range."""
        return self.start_date <= target_date <= self.end_date


class RuleBuilder:
    """Constructs rules with a fluent interface.

    Example:
        tag("xmas").set(metric_multiplier=2.0)
        check("Volume Check").disable()
    """

    def __init__(self, selector: Selector) -> None:
        self._selector = selector

    def disable(self) -> Rule:
        """Create a rule that disables matching assertions."""
        return Rule(selector=self._selector, disabled=True)

    def set(self, *, metric_multiplier: float = 1.0) -> Rule:
        """Create a rule that scales the metric value by the given multiplier.

        Note: Calling set() without arguments creates a no-op rule (multiplier=1.0).
        This is valid for API consistency but has no effect on assertion evaluation.
        """
        return Rule(selector=self._selector, metric_multiplier=metric_multiplier)


def assertion(check: str, name: str | None = None) -> RuleBuilder:
    """Select assertions by check and assertion name.

    Args:
        check: Name of the check to match.
        name: Name of the assertion to match. If None, matches all assertions in the check.

    Example:
        assertion("Volume Check", "Daily orders above minimum").disable()
        assertion("Volume Check").set(metric_multiplier=2.0)  # all assertions
    """
    return RuleBuilder(AssertionSelector(check=check, assertion=name))


def check(name: str) -> RuleBuilder:
    """Select all assertions in a check.

    Args:
        name: Name of the check to match.

    Example:
        check("Volume Check").disable()
    """
    return RuleBuilder(AssertionSelector(check=name, assertion=None))


def tag(name: str) -> RuleBuilder:
    """Select assertions with a specific tag.

    Args:
        name: Tag to match.

    Example:
        tag("xmas").set(metric_multiplier=2.0)
    """
    return RuleBuilder(TagSelector(tag=name))


@dataclass
class ResolvedOverrides:
    """Accumulated overrides from all matching rules.

    Attributes:
        disabled: True if any matching rule disables the assertion.
        metric_multiplier: Product of all matching rules' multipliers.
    """

    disabled: bool = False
    metric_multiplier: float = 1.0


def resolve_overrides(
    check_name: str,
    assertion: "AssertionNode",
    profiles: list[Profile],
    target_date: date,
) -> ResolvedOverrides:
    """Apply all matching rules from active profiles.

    Rules apply in definition order. Multipliers compound (multiply together).
    Any disabled=True rule disables the assertion.

    Args:
        check_name: Name of the check containing the assertion.
        assertion: The assertion node to resolve overrides for.
        profiles: List of profiles to evaluate.
        target_date: Date to check profile activation against.

    Returns:
        ResolvedOverrides with accumulated disabled state and metric_multiplier.
    """
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

    return result


def _matches(
    selector: Selector,
    check_name: str,
    assertion: "AssertionNode",
) -> bool:
    """Return True if selector matches the given check and assertion."""
    match selector:
        case AssertionSelector():
            return selector.matches(check_name, assertion.name)
        case TagSelector():
            return selector.matches(assertion.tags)
