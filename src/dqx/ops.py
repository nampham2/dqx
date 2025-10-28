from __future__ import annotations

from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from dqx.common import DQXError
from dqx.utils import random_prefix

T = TypeVar("T", bound=float)


@runtime_checkable
class Op(Protocol[T]):
    @property
    def name(self) -> str: ...

    @property
    def prefix(self) -> str: ...

    def value(self) -> T: ...

    def assign(self, value: T) -> None: ...

    def clear(self) -> None: ...


@runtime_checkable
class SqlOp(Op[T], Protocol):
    """SqlOps are ops whose values are collected in a _single_ sql statement."""

    @property
    def sql_col(self) -> str: ...


class OpValueMixin(Generic[T]):
    def __init__(self) -> None:
        self._value: T | None = None

    def value(self) -> T:
        if self._value is None:
            raise DQXError(f"{self.__class__.__name__} op has not been collected yet!")
        return self._value

    def assign(self, value: T) -> None:
        self._value = value

    def clear(self) -> None:
        self._value = None


class NumRows(OpValueMixin[float], SqlOp[float]):
    def __init__(self) -> None:
        OpValueMixin.__init__(self)
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return "num_rows()"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NumRows):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


class Average(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column",)

    def __init__(self, column: str) -> None:
        OpValueMixin.__init__(self)
        self.column = column
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return f"average({self.column})"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Average):
            return NotImplemented
        return self.column == other.column

    def __hash__(self) -> int:
        return hash((self.name, self.column))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


class Minimum(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column",)

    def __init__(self, column: str) -> None:
        OpValueMixin.__init__(self)
        self.column = column
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return f"minimum({self.column})"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Minimum):
            return NotImplemented
        return self.column == other.column

    def __hash__(self) -> int:
        return hash((self.name, self.column))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


class Maximum(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column",)

    def __init__(self, column: str) -> None:
        OpValueMixin.__init__(self)
        self.column = column
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return f"maximum({self.column})"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Maximum):
            return NotImplemented
        return self.column == other.column

    def __hash__(self) -> int:
        return hash((self.name, self.column))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


class Sum(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column",)

    def __init__(self, column: str) -> None:
        OpValueMixin.__init__(self)
        self.column = column
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return f"sum({self.column})"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Sum):
            return NotImplemented
        return self.column == other.column

    def __hash__(self) -> int:
        return hash((self.name, self.column))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


class Variance(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column",)

    def __init__(self, column: str) -> None:
        OpValueMixin.__init__(self)
        self.column = column
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return f"variance({self.column})"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Variance):
            return NotImplemented
        return self.column == other.column

    def __hash__(self) -> int:
        return hash((self.name, self.column))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


class First(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column",)

    def __init__(self, column: str) -> None:
        OpValueMixin.__init__(self)
        self.column = column
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return f"first({self.column})"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, First):
            return NotImplemented
        return self.column == other.column

    def __hash__(self) -> int:
        return hash((self.name, self.column))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


class NullCount(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column",)

    def __init__(self, column: str) -> None:
        OpValueMixin.__init__(self)
        self.column = column
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return f"null_count({self.column})"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NullCount):
            return NotImplemented
        return self.column == other.column

    def __hash__(self) -> int:
        return hash((self.name, self.column))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


class NegativeCount(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column",)

    def __init__(self, column: str) -> None:
        OpValueMixin.__init__(self)
        self.column = column
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return f"negative_count({self.column})"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NegativeCount):
            return NotImplemented
        return self.column == other.column

    def __hash__(self) -> int:
        return hash((self.name, self.column))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


class UniqueCount(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column",)

    def __init__(self, column: str) -> None:
        OpValueMixin.__init__(self)
        self.column = column
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return f"unique_count({self.column})"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, UniqueCount):
            return NotImplemented
        return self.column == other.column

    def __hash__(self) -> int:
        return hash((self.name, self.column))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


class DuplicateCount(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("columns",)

    def __init__(self, columns: list[str]) -> None:
        OpValueMixin.__init__(self)
        if not columns:
            raise ValueError("DuplicateCount requires at least one column")
        # Sort columns to ensure consistent behavior regardless of input order
        self.columns = sorted(columns)
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return f"duplicate_count({','.join(self.columns)})"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DuplicateCount):
            return NotImplemented
        return self.columns == other.columns

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.columns)))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


class CountValues(OpValueMixin[float], SqlOp[float]):
    """Count occurrences of specific value(s) in a column.

    Accepts single values (int, str, or bool) or lists of values (list[int] or list[str]).
    Lists must be homogeneous - all integers or all strings, not mixed.
    Boolean values are supported as single values only, not in lists.
    String values will be properly escaped in SQL generation to prevent injection.
    """

    __match_args__ = ("column", "values")

    def __init__(self, column: str, values: int | str | bool | list[int] | list[str]) -> None:
        OpValueMixin.__init__(self)

        # Normalize to list for internal consistency
        # Declare _values with the broadest type first
        self._values: list[Any]

        if isinstance(values, (bool, int, str)):
            self._values = [values]
            self._is_single = True
        elif isinstance(values, list):
            if not values:
                raise ValueError("CountValues requires at least one value")

            # Check homogeneous types (also reject bools in lists)
            if any(isinstance(v, bool) for v in values):
                raise ValueError("CountValues list must contain all integers or all strings, not mixed")
            if not (all(isinstance(v, int) for v in values) or all(isinstance(v, str) for v in values)):
                raise ValueError("CountValues list must contain all integers or all strings, not mixed")

            self._values = values
            self._is_single = False
        else:
            raise ValueError(
                f"CountValues accepts int, str, bool, list[int], or list[str], got {type(values).__name__}"
            )

        self.column = column
        self.values = values  # Store original format for equality checks
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        if self._is_single:
            return f"count_values({self.column},{self._values[0]})"
        else:
            # Format as [val1,val2,...] without quotes
            values_str = "[" + ",".join(str(v) for v in self._values) + "]"
            return f"count_values({self.column},{values_str})"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        # For sql_col, we need a safe column name without special characters
        # Use the column name and a hash of the values to make it unique
        import hashlib

        values_hash = hashlib.md5(str(self.values).encode()).hexdigest()[:8]
        return f"{self.prefix}_count_values_{self.column}_{values_hash}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CountValues):
            return NotImplemented
        # Need to check both value and type to distinguish True from 1, False from 0
        return self.column == other.column and self.values == other.values and type(self.values) is type(other.values)

    def __hash__(self) -> int:
        # Convert lists to tuples for hashing
        hashable_values = self.values if not isinstance(self.values, list) else tuple(self.values)
        return hash((self.name, self.column, hashable_values))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()
