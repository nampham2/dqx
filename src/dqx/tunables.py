"""Tunable constants for RL agent integration.

This module provides an extensible type hierarchy for tunable parameters
that can be adjusted by RL agents within specified bounds.

Example:
    >>> from dqx.tunables import TunablePercent, TunableInt
    >>> threshold = TunablePercent("NULL_THRESHOLD", value=0.05, bounds=(0.0, 0.20))
    >>> threshold.set(0.03, agent="rl_optimizer", reason="Episode 42")
    >>> threshold.value
    0.03
"""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

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


@dataclass
class Tunable(ABC, Generic[T]):
    """Base class for all tunable parameters.

    Subclasses define validation rules for specific types.

    Attributes:
        name: Unique identifier for the tunable
        _value: Current value (use .value property to access)
        _history: List of changes made to this tunable
    """

    name: str
    _value: T
    _history: list[TunableChange] = field(default_factory=list, repr=False)

    @property
    def value(self) -> T:
        """Get current value."""
        return self._value

    @abstractmethod
    def validate(self, value: T) -> None:
        """Validate value. Raises ValueError if invalid."""
        pass  # pragma: no cover

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize for RL action space / API."""
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
        old = self._value
        self._value = value
        self._history.append(
            TunableChange(
                timestamp=datetime.datetime.now(datetime.UTC),
                old_value=old,
                new_value=value,
                agent=agent,
                reason=reason,
            )
        )

    @property
    def history(self) -> list[TunableChange]:
        """Get copy of change history."""
        return self._history.copy()


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

    def __post_init__(self) -> None:
        if self.bounds[0] > self.bounds[1]:
            raise ValueError(f"Invalid bounds: min ({self.bounds[0]}) > max ({self.bounds[1]})")
        self.validate(self._value)

    def validate(self, value: float) -> None:
        """Validate value is within bounds."""
        if not self.bounds[0] <= value <= self.bounds[1]:
            raise ValueError(f"{self.name}: value {value} outside bounds {self.bounds}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize for RL action space."""
        return {
            "name": self.name,
            "type": "float",
            "value": self._value,
            "bounds": self.bounds,
        }


@dataclass
class TunablePercent(Tunable[float]):
    """Percentage tunable (0.0-1.0 internally, displayed as %).

    Example:
        >>> t = TunablePercent("threshold", value=0.05, bounds=(0.0, 0.20))
        >>> t.set(0.10)  # 10%
        >>> t.value
        0.1
    """

    bounds: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if self.bounds[0] > self.bounds[1]:
            raise ValueError(f"Invalid bounds: min ({self.bounds[0]}) > max ({self.bounds[1]})")
        self.validate(self._value)

    def validate(self, value: float) -> None:
        """Validate value is within bounds."""
        if not self.bounds[0] <= value <= self.bounds[1]:
            raise ValueError(
                f"{self.name}: value {value * 100:.1f}% outside bounds "
                f"[{self.bounds[0] * 100:.1f}%, {self.bounds[1] * 100:.1f}%]"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for RL action space."""
        return {
            "name": self.name,
            "type": "percent",
            "value": self._value,
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

    def __post_init__(self) -> None:
        if self.bounds[0] > self.bounds[1]:
            raise ValueError(f"Invalid bounds: min ({self.bounds[0]}) > max ({self.bounds[1]})")
        self.validate(self._value)

    def validate(self, value: int) -> None:
        """Validate value is an integer within bounds."""
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{self.name}: value must be int, got {type(value).__name__}")
        if not self.bounds[0] <= value <= self.bounds[1]:
            raise ValueError(f"{self.name}: value {value} outside bounds {self.bounds}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize for RL action space."""
        return {
            "name": self.name,
            "type": "int",
            "value": self._value,
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
        if not self.choices:
            raise ValueError(f"{self.name}: choices cannot be empty")
        self.validate(self._value)

    def validate(self, value: str) -> None:
        """Validate value is one of the allowed choices."""
        if value not in self.choices:
            raise ValueError(f"{self.name}: value '{value}' not in choices {self.choices}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize for RL action space."""
        return {
            "name": self.name,
            "type": "choice",
            "value": self._value,
            "choices": self.choices,
        }
