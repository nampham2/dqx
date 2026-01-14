"""AST node definitions for DQL.

All nodes are immutable dataclasses with optional source location tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class Severity(Enum):
    """Assertion severity levels."""

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


@dataclass(frozen=True)
class SourceLocation:
    """Source location for error reporting."""

    line: int
    column: int
    end_line: int | None = None
    end_column: int | None = None
    filename: str | None = None

    def __str__(self) -> str:
        loc = f"{self.filename}:" if self.filename else ""
        loc += f"{self.line}:{self.column}"
        return loc


@dataclass(frozen=True)
class Expr:
    """A metric expression (kept as string for sympy parsing later)."""

    text: str
    loc: SourceLocation | None = None


@dataclass(frozen=True)
class DateExpr:
    """A date expression - either a literal date or a function call."""

    value: date | str  # date literal or expression string like "nth_weekday(...)"
    offset: int = 0  # day offset (+/- N)
    loc: SourceLocation | None = None


@dataclass(frozen=True)
class Annotation:
    """An annotation like @experimental, @required, or @cost(fp=1, fn=100)."""

    name: str
    args: dict[str, int | float | str] = field(default_factory=dict)
    loc: SourceLocation | None = None


@dataclass(frozen=True)
class Sample:
    """Sampling configuration for an assertion."""

    value: float  # percentage (0-1) or row count
    is_percentage: bool = True
    seed: int | None = None
    loc: SourceLocation | None = None


@dataclass(frozen=True)
class Assertion:
    """An assertion within a check."""

    expr: Expr
    condition: str  # The condition operator: ">", ">=", "<", "<=", "==", "!=", "between", "is"
    threshold: Expr | None = None  # Right-hand side of comparison
    threshold_upper: Expr | None = None  # For "between A and B"
    keyword: str | None = None  # For "is positive", "is negative"
    name: str | None = None
    severity: Severity = Severity.P1
    tolerance: float | None = None
    tags: tuple[str, ...] = ()
    sample: Sample | None = None
    annotations: tuple[Annotation, ...] = ()
    loc: SourceLocation | None = None


@dataclass(frozen=True)
class Check:
    """A check block containing assertions."""

    name: str
    datasets: tuple[str, ...]
    assertions: tuple[Assertion, ...]
    loc: SourceLocation | None = None


@dataclass(frozen=True)
class Tunable:
    """A tunable constant declaration with required bounds."""

    name: str
    value: Expr
    bounds: tuple[Expr, Expr]  # Required: (min, max)
    loc: SourceLocation | None = None


@dataclass(frozen=True)
class DisableRule:
    """Disable a check or assertion."""

    target_type: str  # "check" or "assertion"
    target_name: str
    in_check: str | None = None  # For "assertion X in Y"
    loc: SourceLocation | None = None


@dataclass(frozen=True)
class ScaleRule:
    """Scale a check or tag by a multiplier."""

    selector_type: str  # "check" or "tag"
    selector_name: str
    multiplier: float
    loc: SourceLocation | None = None


@dataclass(frozen=True)
class SetSeverityRule:
    """Set severity for a selector."""

    selector_type: str  # "check" or "tag"
    selector_name: str
    severity: Severity
    loc: SourceLocation | None = None


# Type alias for all rule types
Rule = DisableRule | ScaleRule | SetSeverityRule


@dataclass(frozen=True)
class Profile:
    """A profile for modifying checks during specific periods."""

    name: str
    profile_type: str  # "holiday" or "recurring"
    from_date: DateExpr
    to_date: DateExpr
    rules: tuple[Rule, ...]
    loc: SourceLocation | None = None


@dataclass(frozen=True)
class Suite:
    """The root AST node representing a DQL suite."""

    name: str
    checks: tuple[Check, ...] = ()
    profiles: tuple[Profile, ...] = ()
    tunables: tuple[Tunable, ...] = ()
    availability_threshold: float | None = None  # e.g., 0.8 for 80%
    loc: SourceLocation | None = None
