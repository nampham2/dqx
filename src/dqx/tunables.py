"""Tunable constants for RL agent integration.

This module provides an extensible type hierarchy for tunable parameters
that can be adjusted by RL agents within specified bounds.

Tunables can be used directly in SymPy expressions, automatically converting
to TunableSymbol objects that preserve the tunable reference for later
collection and manipulation.

Example:
    >>> from dqx.tunables import TunableFloat, TunableInt
    >>> threshold = TunableFloat("NULL_THRESHOLD", value=0.05, bounds=(0.0, 0.20))
    >>> threshold.set(0.03, agent="rl_optimizer", reason="Episode 42")
    >>> threshold.value
    0.03
    >>>
    >>> # Use tunable directly in expressions (no .value needed)
    >>> import sympy as sp
    >>> x = sp.Symbol("x")
    >>> expr = x - threshold  # threshold auto-converts to TunableSymbol
"""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import sympy as sp

if TYPE_CHECKING:
    import sympy.core.expr

T = TypeVar("T")


@dataclass
class TunableChange:
    """Record of a tunable parameter change.

    Attributes:
        timestamp: When the change occurred (UTC)
        old_value: Previous value
        new_value: New value after change
        agent: Who made the change ("human", "rl_optimizer", "autotuner")
        reason: Optional explanation for the change
    """

    timestamp: datetime.datetime
    old_value: Any
    new_value: Any
    agent: str
    reason: str | None = None


class TunableSymbol(sp.Symbol):
    """SymPy Symbol that wraps a Tunable parameter.

    This allows Tunables to be used directly in SymPy expressions while
    preserving the reference to the underlying Tunable object for later
    extraction and manipulation.

    The TunableSymbol appears as a regular symbol in expressions but maintains
    a reference to its source Tunable, enabling automatic collection of all
    tunables used in assertions.

    Example:
        >>> from dqx.tunables import TunableFloat
        >>> threshold = TunableFloat("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        >>> symbol = TunableSymbol(threshold)
        >>> x = sp.Symbol("x")
        >>> expr = x - symbol
        >>> # expr now contains the TunableSymbol which can be extracted later
        >>> # symbol.tunable gives access to the original Tunable object
    """

    def __new__(cls, tunable: Tunable[Any]) -> TunableSymbol:
        """Create a new TunableSymbol wrapping the given Tunable.

        Args:
            tunable: The Tunable object to wrap as a SymPy symbol.

        Returns:
            TunableSymbol: A SymPy Symbol subclass that preserves the tunable reference.

        Note:
            We use a unique dummy assumption (based on tunable ID) to ensure different
            Tunable instances with the same name create different TunableSymbol instances,
            preventing SymPy's caching from causing test isolation issues.
        """
        # Create a unique dummy assumption to prevent SymPy caching
        # This ensures different tunable instances with the same name get different symbols
        dummy_assumption = f"_tid{id(tunable)}"
        # Use a custom assumption that SymPy will treat as unique
        obj = sp.Symbol.__new__(cls, tunable.name, **{dummy_assumption: True})
        # Store reference to the tunable (uses object.__setattr__ to bypass SymPy's immutability)
        object.__setattr__(obj, "_tunable_ref", tunable)
        return obj

    @property
    def tunable(self) -> Tunable[Any]:
        """Get the underlying Tunable object.

        Returns:
            Tunable: The original Tunable instance wrapped by this symbol.
        """
        return object.__getattribute__(self, "_tunable_ref")

    @property
    def value(self) -> Any:
        """Get the current value of the tunable for evaluation.

        This property allows TunableSymbol to be substituted with its numeric
        value during expression evaluation, enabling seamless integration with
        SymPy's numeric evaluation system.

        Returns:
            The current value of the underlying tunable.
        """
        return self.tunable.value


@dataclass
class Tunable(ABC, Generic[T]):
    """Base class for all tunable parameters.

    Subclasses define validation rules for specific types.

    Attributes:
        name: Unique identifier for the tunable
        value: Current value
        history: List of changes made to this tunable
    """

    name: str
    value: T
    history: list[TunableChange] = field(default_factory=list, repr=False)

    @abstractmethod
    def validate(self, value: T) -> None:
        """
        Ensure a candidate value is valid for this tunable.

        Args:
            value (T): Candidate value to validate.

        Raises:
            ValueError: If the value is invalid for this tunable.
        """
        pass  # pragma: no cover

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the tunable into a dictionary suitable for RL action spaces and APIs.

        The returned mapping contains the tunable's identifying fields and any type-specific metadata
        required by external systems (for example, keys such as "type", "name", "value" and bounds or choices).

        Returns:
            dict[str, Any]: A dictionary representation of the tunable for use by RL agents or external APIs.
        """
        pass  # pragma: no cover

    def set(self, value: T, agent: str = "human", reason: str | None = None) -> None:
        """Set value with validation and history tracking.

        Args:
            value: New value to set
            agent: Who made the change ("human", "rl_optimizer", "autotuner")
            reason: Optional explanation for the change

        Raises:
            ValueError: If value fails validation
        """
        self.validate(value)
        old = self.value
        self.value = value
        self.history.append(
            TunableChange(
                timestamp=datetime.datetime.now(datetime.UTC),
                old_value=old,
                new_value=value,
                agent=agent,
                reason=reason,
            )
        )

    # Arithmetic operators to support using Tunables directly in expressions

    def __add__(self, other: Any) -> sympy.core.expr.Expr:
        """Support addition: threshold + x."""
        return TunableSymbol(self) + other

    def __radd__(self, other: Any) -> sympy.core.expr.Expr:
        """Support reverse addition: x + threshold."""
        return other + TunableSymbol(self)

    def __sub__(self, other: Any) -> sympy.core.expr.Expr:
        """Support subtraction: threshold - x."""
        return TunableSymbol(self) - other

    def __rsub__(self, other: Any) -> sympy.core.expr.Expr:
        """Support reverse subtraction: x - threshold."""
        return other - TunableSymbol(self)

    def __mul__(self, other: Any) -> sympy.core.expr.Expr:
        """Support multiplication: threshold * x."""
        return TunableSymbol(self) * other

    def __rmul__(self, other: Any) -> sympy.core.expr.Expr:
        """Support reverse multiplication: x * threshold."""
        return other * TunableSymbol(self)

    def __truediv__(self, other: Any) -> sympy.core.expr.Expr:
        """Support division: threshold / x."""
        return TunableSymbol(self) / other

    def __rtruediv__(self, other: Any) -> sympy.core.expr.Expr:
        """Support reverse division: x / threshold."""
        return other / TunableSymbol(self)

    def __neg__(self) -> sympy.core.expr.Expr:
        """Support negation: -threshold."""
        return -TunableSymbol(self)

    def __lt__(self, other: Any) -> bool:
        """Support < comparison using the tunable's value."""
        return self.value < other

    def __le__(self, other: Any) -> bool:
        """Support <= comparison using the tunable's value."""
        return self.value <= other

    def __gt__(self, other: Any) -> bool:
        """Support > comparison using the tunable's value."""
        return self.value > other

    def __ge__(self, other: Any) -> bool:
        """Support >= comparison using the tunable's value."""
        return self.value >= other


@dataclass
class TunableFloat(Tunable[float]):
    """Float tunable with min/max bounds.

    Example:
        >>> t = TunableFloat("tolerance", value=0.001, bounds=(0.0001, 0.01))
        >>> t.set(0.005)
        >>> t.value
        0.005
    """

    bounds: tuple[float, float] = (0.0, 1.0)

    @property
    def lower_bound(self) -> float:
        """
        Get the lower bound of the tunable's valid range.

        Returns:
            float: The lower bound of the allowed value range.
        """
        return self.bounds[0]

    @property
    def upper_bound(self) -> float:
        """
        Get the upper bound of the valid range.

        Returns:
            float: The upper bound value from `bounds`.
        """
        return self.bounds[1]

    def __post_init__(self) -> None:
        """
        Validate bounds and the initial value on post-initialization.

        Performs two checks: ensures the configured lower bound is not greater than the upper bound, and validates the current `value` using the tunable's `validate` method. Raises ValueError if the bounds are invalid or if the initial value fails validation.
        """
        if self.lower_bound > self.upper_bound:
            raise ValueError(f"Invalid bounds: min ({self.lower_bound}) > max ({self.upper_bound})")
        self.validate(self.value)

    def validate(self, value: float) -> None:
        """
        Ensure the given value lies within the tunable's inclusive bounds.

        Args:
            value: Candidate value to validate.

        Raises:
            ValueError: If `value` is less than `lower_bound` or greater than `upper_bound`.
        """
        if not self.lower_bound <= value <= self.upper_bound:
            raise ValueError(f"{self.name}: value {value} outside bounds {self.bounds}")

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the tunable float to a dictionary suitable for RL action spaces and APIs.

        Returns:
            dict[str, Any]: A mapping with keys:
                - "name" (str): the tunable's identifier.
                - "type" (str): the string "float".
                - "value" (float): the current numeric value.
                - "bounds" (tuple[float, float]): inclusive lower and upper bounds.
        """
        return {
            "name": self.name,
            "type": "float",
            "value": self.value,
            "bounds": self.bounds,
        }


@dataclass
class TunableInt(Tunable[int]):
    """Integer tunable with min/max bounds.

    Example:
        >>> t = TunableInt("min_rows", value=1000, bounds=(100, 10000))
        >>> t.set(500)
        >>> t.value
        500
    """

    bounds: tuple[int, int] = (0, 100)

    @property
    def lower_bound(self) -> int:
        """
        Retrieve the lower bound of the allowed integer range.

        Returns:
            int: The lower bound of the valid range.
        """
        return self.bounds[0]

    @property
    def upper_bound(self) -> int:
        """
        Get the inclusive upper bound of the valid range.

        Returns:
            int: The inclusive upper bound value from the tunable's bounds tuple.
        """
        return self.bounds[1]

    def __post_init__(self) -> None:
        """
        Validate bounds and the initial value on post-initialization.

        Performs two checks: ensures the configured lower bound is not greater than the upper bound, and validates the current `value` using the tunable's `validate` method. Raises ValueError if the bounds are invalid or if the initial value fails validation.
        """
        if self.lower_bound > self.upper_bound:
            raise ValueError(f"Invalid bounds: min ({self.lower_bound}) > max ({self.upper_bound})")
        self.validate(self.value)

    def validate(self, value: int) -> None:
        """
        Ensure `value` is an integer (not a bool) and falls within the tunable's bounds.

        Args:
            value (int): Candidate integer to validate.

        Raises:
            TypeError: If `value` is not an `int` or is a `bool`.
            ValueError: If `value` is less than `lower_bound` or greater than `upper_bound`.
        """
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{self.name}: value must be int, got {type(value).__name__}")
        if not self.lower_bound <= value <= self.upper_bound:
            raise ValueError(f"{self.name}: value {value} outside bounds {self.bounds}")

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the integer tunable for external use (for example, an RL action space).

        Returns:
            serialized (dict[str, Any]): Dictionary with keys:
                - "name" (str): tunable identifier
                - "type" (str): the literal "int"
                - "value" (int): current integer value
                - "bounds" (tuple[int, int]): (lower_bound, upper_bound)
        """
        return {
            "name": self.name,
            "type": "int",
            "value": self.value,
            "bounds": self.bounds,
        }


@dataclass
class TunableChoice(Tunable[str]):
    """Categorical tunable from fixed choices.

    Example:
        >>> t = TunableChoice("method", value="mean", choices=("mean", "median", "max"))
        >>> t.set("median")
        >>> t.value
        'median'
    """

    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """
        Perform post-initialization checks for a TunableChoice.

        Ensures the `choices` sequence is not empty and that the instance's current `value` is one of the allowed choices.

        Raises:
            ValueError: If `choices` is empty, or if `value` is not a member of `choices`.
        """
        if not self.choices:
            raise ValueError(f"{self.name}: choices cannot be empty")
        self.validate(self.value)

    def validate(self, value: str) -> None:
        """
        Ensure the provided value is a member of the tunable's allowed choices.

        Raises:
            ValueError: if the value is not one of the allowed choices.
        """
        if value not in self.choices:
            raise ValueError(f"{self.name}: value '{value}' not in choices {self.choices}")

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the choice tunable into a dictionary suitable for external APIs or an RL action space.

        Returns:
            serialized (dict[str, Any]): Dictionary with keys:
                - "name": tunable name (str)
                - "type": the string "choice"
                - "value": current selected value (str)
                - "choices": available choices (tuple[str, ...])
        """
        return {
            "name": self.name,
            "type": "choice",
            "value": self.value,
            "choices": self.choices,
        }
