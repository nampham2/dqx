"""AST node definitions for DQL.

All nodes are immutable dataclasses with optional source location tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
class Annotation:
    """An annotation like @experimental, @required, or @cost(fp=1, fn=100)."""

    name: str
    args: dict[str, int | float | str] = field(default_factory=dict)
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
    annotations: tuple[Annotation, ...] = ()
    loc: SourceLocation | None = None


@dataclass(frozen=True)
class Collection:
    """A collection statement within a check (noop assertion).

    Collections compute metrics without validation, always passing.
    Equivalent to Python API: ctx.assert_that(metric).where(...).noop()
    """

    expr: Expr
    name: str | None = None
    severity: Severity = Severity.P1
    tolerance: float | None = None  # Kept for structural consistency, not used
    tags: tuple[str, ...] = ()
    annotations: tuple[Annotation, ...] = ()
    loc: SourceLocation | None = None


@dataclass(frozen=True)
class Check:
    """A check block containing assertions and collections."""

    name: str
    datasets: tuple[str, ...]
    assertions: tuple[Assertion | Collection, ...]
    loc: SourceLocation | None = None


@dataclass(frozen=True)
class Tunable:
    """A tunable constant declaration with required bounds."""

    name: str
    value: Expr
    bounds: tuple[Expr, Expr]  # Required: (min, max)
    loc: SourceLocation | None = None


@dataclass(frozen=True)
class Suite:
    """The root AST node representing a DQL suite."""

    name: str
    checks: tuple[Check, ...] = ()
    tunables: tuple[Tunable, ...] = ()
    availability_threshold: float | None = None  # e.g., 0.8 for 80%
    loc: SourceLocation | None = None
