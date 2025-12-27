from __future__ import annotations

import hashlib
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from dqx.common import DQXError, Parameters
from dqx.utils import freeze_for_hashing, random_prefix

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

    @property
    def parameters(self) -> Parameters: ...


class OpValueMixin(Generic[T]):
    def __init__(self, parameters: Parameters | None = None) -> None:
        self._value: T | None = None
        self._parameters = parameters or {}

    @property
    def parameters(self) -> Parameters:
        return self._parameters

    def value(self) -> T:
        if self._value is None:
            raise DQXError(f"{self.__class__.__name__} op has not been collected yet!")
        return self._value

    def assign(self, value: T) -> None:
        self._value = value

    def clear(self) -> None:
        self._value = None


class NumRows(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("parameters",)  # Add for pattern matching

    def __init__(self, parameters: Parameters | None = None) -> None:
        """Initialize NumRows operation.

        Args:
            parameters: Optional parameters for CTE customization
        """
        OpValueMixin.__init__(self, parameters)
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
        return hash((self.name, tuple(sorted((k, freeze_for_hashing(v)) for k, v in self.parameters.items()))))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


class Average(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column", "parameters")

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        """Initialize Average operation.

        Args:
            column: Column name to calculate average
            parameters: Optional parameters for CTE customization
        """
        OpValueMixin.__init__(self, parameters)
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
        return hash(
            (self.name, self.column, tuple(sorted((k, freeze_for_hashing(v)) for k, v in self.parameters.items())))
        )

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


class CustomSQL(OpValueMixin[float], SqlOp[float]):
    """Custom SQL operation for user-defined SQL expressions.

    Allows defining custom SQL expressions as operations. The SQL expression
    is used as-is by the dialect, and any parameters are passed to the CTE
    level for filtering/grouping, just like all other operations.

    Example:
        # Simple usage
        CustomSQL("COUNT(DISTINCT user_id)")

        # With CTE parameters (not substituted in SQL)
        CustomSQL("PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)",
                  parameters={"region": "US"})
    """

    __match_args__ = ("sql_expression", "parameters")

    def __init__(self, sql_expression: str, parameters: Parameters | None = None) -> None:
        """Initialize CustomSQL operation.

        Args:
            sql_expression: SQL expression to be executed as-is
            parameters: Optional parameters for CTE customization
        """
        OpValueMixin.__init__(self, parameters)
        self.sql_expression = sql_expression
        self._prefix = random_prefix()

        # Generate a short identifier from the SQL expression
        # Use hash for uniqueness, limited to 8 chars

        sql_hash = hashlib.md5(sql_expression.encode()).hexdigest()[:8]
        self._sql_hash = sql_hash

    @property
    def name(self) -> str:
        # Use hash-based naming with parentheses format
        return f"custom_sql({self._sql_hash})"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        """
        Determine whether another object represents the same CustomSQL operation.

        Args:
            other: The object to compare against.

        Returns:
            `True` if `other` is a CustomSQL instance with an identical SQL expression and parameters, `False` otherwise.
        """
        if not isinstance(other, CustomSQL):
            return NotImplemented  # pragma: no cover
        return self.sql_expression == other.sql_expression and self.parameters == other.parameters

    def __hash__(self) -> int:
        """
        Compute a stable hash that uniquely identifies this CustomSQL instance.

        The hash is derived from the SQL expression and the operation's parameters; parameters are normalized and sorted by key so the result is independent of parameter insertion order.

        Returns:
            int: Integer hash based on the SQL expression and normalized parameters.
        """
        return hash(
            (self.sql_expression, tuple(sorted((k, freeze_for_hashing(v)) for k, v in self.parameters.items())))
        )

    def __repr__(self) -> str:
        return f"CustomSQL({self.sql_expression!r})"

    def __str__(self) -> str:
        return self.__repr__()


class Minimum(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column", "parameters")

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        """Initialize Minimum operation.

        Args:
            column: Column name to find minimum value
            parameters: Optional parameters for CTE customization
        """
        OpValueMixin.__init__(self, parameters)
        self.column = column
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return f"minimum({self.column})"

    @property
    def prefix(self) -> str:
        """
        Get the operation's unique per-instance prefix.

        Returns:
            prefix (str): Unique prefix string used to form SQL column aliases for this operation.
        """
        return self._prefix

    @property
    def sql_col(self) -> str:  # pragma: no cover
        """
        Constructs the SQL column identifier for the operation.

        Returns:
            sql_col (str): Unique SQL column alias formed by joining the operation's prefix and name with an underscore.
        """
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        """
        Determine whether another object represents the same Minimum operation.

        Args:
            other (Any): Object to compare against.

        Returns:
            `true` if `other` is a `Minimum` with an identical `column` attribute, `false` otherwise.
        """
        if not isinstance(other, Minimum):  # pragma: no cover
            return NotImplemented
        return self.column == other.column

    def __hash__(self) -> int:
        """
        Compute a deterministic hash for the operation instance based on its identifying attributes.

        Returns:
            int: Hash value derived from the operation's name, column, and its parameters (parameters are frozen for stable hashing).
        """
        return hash(
            (self.name, self.column, tuple(sorted((k, freeze_for_hashing(v)) for k, v in self.parameters.items())))
        )

    def __repr__(self) -> str:  # pragma: no cover
        """
        Return the operation's developer-facing representation.

        Returns:
            str: The operation's name (the value of the `name` property).
        """
        return self.name

    def __str__(self) -> str:  # pragma: no cover
        """
        Provide the object's informal string representation.

        Returns:
            A human-readable string representing the object.
        """
        return self.__repr__()


class Maximum(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column", "parameters")

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        """Initialize Maximum operation.

        Args:
            column: Column name to find maximum value
            parameters: Optional parameters for CTE customization
        """
        OpValueMixin.__init__(self, parameters)
        self.column = column
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return f"maximum({self.column})"

    @property
    def prefix(self) -> str:
        """
        Get the operation's unique per-instance prefix.

        Returns:
            prefix (str): Unique prefix string used to form SQL column aliases for this operation.
        """
        return self._prefix

    @property
    def sql_col(self) -> str:  # pragma: no cover
        """
        Constructs the SQL column identifier for the operation.

        Returns:
            sql_col (str): Unique SQL column alias formed by joining the operation's prefix and name with an underscore.
        """
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        """
        Determine whether another object represents the same Maximum operation.

        Args:
            other (Any): Object to compare against.

        Returns:
            `True` if `other` is a `Maximum` with the same `column`, `False` if `other` is a `Maximum` with a different `column`, `NotImplemented` if `other` is not a `Maximum`.
        """
        if not isinstance(other, Maximum):  # pragma: no cover
            return NotImplemented
        return self.column == other.column

    def __hash__(self) -> int:
        """
        Compute a deterministic hash for the operation instance based on its identifying attributes.

        Returns:
            int: Hash value derived from the operation's name, column, and its parameters (parameters are frozen for stable hashing).
        """
        return hash(
            (self.name, self.column, tuple(sorted((k, freeze_for_hashing(v)) for k, v in self.parameters.items())))
        )

    def __repr__(self) -> str:  # pragma: no cover
        """
        Return the operation's developer-facing representation.

        Returns:
            str: The operation's name (the value of the `name` property).
        """
        return self.name

    def __str__(self) -> str:  # pragma: no cover
        """
        Provide the object's informal string representation.

        Returns:
            A human-readable string representing the object.
        """
        return self.__repr__()


class Sum(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column", "parameters")

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        """Initialize Sum operation.

        Args:
            column: Column name to calculate sum
            parameters: Optional parameters for CTE customization
        """
        OpValueMixin.__init__(self, parameters)
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
        """
        Determine whether another object represents the same Sum operation.

        Args:
            other: The object to compare against.

        Returns:
            `True` if `other` is a `Sum` with the same `column`, `False` if `other` is a `Sum` with a different `column`. Returns `NotImplemented` if `other` is not a `Sum`.
        """
        if not isinstance(other, Sum):
            return NotImplemented  # pragma: no cover
        return self.column == other.column

    def __hash__(self) -> int:
        """
        Compute a deterministic hash for the operation instance based on its identifying attributes.

        Returns:
            int: Hash value derived from the operation's name, column, and its parameters (parameters are frozen for stable hashing).
        """
        return hash(
            (self.name, self.column, tuple(sorted((k, freeze_for_hashing(v)) for k, v in self.parameters.items())))
        )

    def __repr__(self) -> str:  # pragma: no cover
        """
        Return the operation's developer-facing representation.

        Returns:
            str: The operation's name (the value of the `name` property).
        """
        return self.name

    def __str__(self) -> str:  # pragma: no cover
        """
        Provide the object's informal string representation.

        Returns:
            A human-readable string representing the object.
        """
        return self.__repr__()


class Variance(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column", "parameters")

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        """Initialize Variance operation.

        Args:
            column: Column name to calculate variance
            parameters: Optional parameters for CTE customization
        """
        OpValueMixin.__init__(self, parameters)
        self.column = column
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return f"variance({self.column})"

    @property
    def prefix(self) -> str:
        """
        Get the operation's unique per-instance prefix.

        Returns:
            prefix (str): Unique prefix string used to form SQL column aliases for this operation.
        """
        return self._prefix

    @property
    def sql_col(self) -> str:  # pragma: no cover
        """
        Constructs the SQL column identifier for the operation.

        Returns:
            sql_col (str): Unique SQL column alias formed by joining the operation's prefix and name with an underscore.
        """
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        """
        Determine equality with another object by comparing the target column.

        Args:
            other (Any): Object to compare against.

        Returns:
            `True` if `other` is a `Variance` instance with the same `column`, `False` if it is a `Variance` with a different `column`. Returns `NotImplemented` when `other` is not a `Variance`.
        """
        if not isinstance(other, Variance):  # pragma: no cover
            return NotImplemented
        return self.column == other.column

    def __hash__(self) -> int:
        """
        Compute a deterministic hash for the operation instance based on its identifying attributes.

        Returns:
            int: Hash value derived from the operation's name, column, and its parameters (parameters are frozen for stable hashing).
        """
        return hash(
            (self.name, self.column, tuple(sorted((k, freeze_for_hashing(v)) for k, v in self.parameters.items())))
        )

    def __repr__(self) -> str:  # pragma: no cover
        """
        Return the operation's developer-facing representation.

        Returns:
            str: The operation's name (the value of the `name` property).
        """
        return self.name

    def __str__(self) -> str:  # pragma: no cover
        """
        Provide the object's informal string representation.

        Returns:
            A human-readable string representing the object.
        """
        return self.__repr__()


class First(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column", "order_by", "parameters")

    def __init__(self, column: str, order_by: str | None = None, parameters: Parameters | None = None) -> None:
        """
        Create a First operation that captures the target column and optional ordering used to select the first value.

        Args:
            column: Name of the column to extract the first value from.
            order_by: Optional column name to sort by before selecting the first value; if None, no explicit ordering is applied.
            parameters: Optional parameters used for customizing generated CTEs and SQL identifiers.
        """
        OpValueMixin.__init__(self, parameters)
        self.column = column
        self.order_by = order_by
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        """
        Operation name for the `first` aggregation, including `order_by` when provided.

        Returns:
            The operation name as a string; when `order_by` is set the name is formatted
            as "first(column, order_by=...)", otherwise as "first(column)".
        """
        if self.order_by:
            return f"first({self.column}, order_by={self.order_by})"
        return f"first({self.column})"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        """
        Determine equality between this First operation and another object.

        Args:
            other (Any): Object to compare against.

        Returns:
            `True` if `other` is a `First` with the same `column` and `order_by`, `False` if `other` is a `First` but differs, or `NotImplemented` when comparing to a non-`First` object.
        """
        if not isinstance(other, First):
            return NotImplemented  # pragma: no cover
        return self.column == other.column and self.order_by == other.order_by

    def __hash__(self) -> int:
        """
        Compute a stable hash for the operation instance.

        Returns:
            int: Hash value derived from the operation's name, column, order_by, and the operation parameters (parameters are normalized for stable hashing).
        """
        return hash(
            (
                self.name,
                self.column,
                self.order_by,
                tuple(sorted((k, freeze_for_hashing(v)) for k, v in self.parameters.items())),
            )
        )

    def __repr__(self) -> str:  # pragma: no cover
        """
        Return the operation's developer-facing representation.

        Returns:
            str: The operation's name (the value of the `name` property).
        """
        return self.name

    def __str__(self) -> str:  # pragma: no cover
        """
        Provide the object's informal string representation.

        Returns:
            A human-readable string representing the object.
        """
        return self.__repr__()


class NullCount(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column", "parameters")

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        """Initialize NullCount operation.

        Args:
            column: Column name to count null values
            parameters: Optional parameters for CTE customization
        """
        OpValueMixin.__init__(self, parameters)
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
        """
        Compare this NullCount to another object for equality based on the target column.

        Args:
            other (Any): Object to compare against.

        Returns:
            `True` if `other` is a `NullCount` with the same `column`, `False` if `other` is a `NullCount` with a different `column`, `NotImplemented` for other types.
        """
        if not isinstance(other, NullCount):
            return NotImplemented  # pragma: no cover
        return self.column == other.column

    def __hash__(self) -> int:
        """
        Compute a deterministic hash for the operation instance based on its identifying attributes.

        Returns:
            int: Hash value derived from the operation's name, column, and its parameters (parameters are frozen for stable hashing).
        """
        return hash(
            (self.name, self.column, tuple(sorted((k, freeze_for_hashing(v)) for k, v in self.parameters.items())))
        )

    def __repr__(self) -> str:  # pragma: no cover
        """
        Return the operation's developer-facing representation.

        Returns:
            str: The operation's name (the value of the `name` property).
        """
        return self.name

    def __str__(self) -> str:  # pragma: no cover
        """
        Provide the object's informal string representation.

        Returns:
            A human-readable string representing the object.
        """
        return self.__repr__()


class NegativeCount(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column", "parameters")

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        """Initialize NegativeCount operation.

        Args:
            column: Column name to count negative values
            parameters: Optional parameters for CTE customization
        """
        OpValueMixin.__init__(self, parameters)
        self.column = column
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return f"negative_count({self.column})"

    @property
    def prefix(self) -> str:
        """
        Get the operation's unique per-instance prefix.

        Returns:
            prefix (str): Unique prefix string used to form SQL column aliases for this operation.
        """
        return self._prefix

    @property
    def sql_col(self) -> str:  # pragma: no cover
        """
        Constructs the SQL column identifier for the operation.

        Returns:
            sql_col (str): Unique SQL column alias formed by joining the operation's prefix and name with an underscore.
        """
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        """
        Determine equality with another NegativeCount by comparing the column attribute.

        Args:
            other (Any): Object to compare against.

        Returns:
            True if `other` is a `NegativeCount` instance with the same `column`, `False` if it is a `NegativeCount` with a different `column`. Returns `NotImplemented` if `other` is not a `NegativeCount`.
        """
        if not isinstance(other, NegativeCount):  # pragma: no cover
            return NotImplemented
        return self.column == other.column

    def __hash__(self) -> int:
        """
        Compute a deterministic hash for the operation instance based on its identifying attributes.

        Returns:
            int: Hash value derived from the operation's name, column, and its parameters (parameters are frozen for stable hashing).
        """
        return hash(
            (self.name, self.column, tuple(sorted((k, freeze_for_hashing(v)) for k, v in self.parameters.items())))
        )

    def __repr__(self) -> str:  # pragma: no cover
        """
        Return the operation's developer-facing representation.

        Returns:
            str: The operation's name (the value of the `name` property).
        """
        return self.name

    def __str__(self) -> str:  # pragma: no cover
        """
        Provide the object's informal string representation.

        Returns:
            A human-readable string representing the object.
        """
        return self.__repr__()


class UniqueCount(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("column", "parameters")

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        """Initialize UniqueCount operation.

        Args:
            column: Column name to count unique values
            parameters: Optional parameters for CTE customization
        """
        OpValueMixin.__init__(self, parameters)
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
        """
        Compute a deterministic hash for the operation instance based on its identifying attributes.

        Returns:
            int: Hash value derived from the operation's name, column, and its parameters (parameters are frozen for stable hashing).
        """
        return hash(
            (self.name, self.column, tuple(sorted((k, freeze_for_hashing(v)) for k, v in self.parameters.items())))
        )

    def __repr__(self) -> str:  # pragma: no cover
        """
        Return the operation's developer-facing representation.

        Returns:
            str: The operation's name (the value of the `name` property).
        """
        return self.name

    def __str__(self) -> str:  # pragma: no cover
        """
        Provide the object's informal string representation.

        Returns:
            A human-readable string representing the object.
        """
        return self.__repr__()


class DuplicateCount(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("columns", "parameters")

    def __init__(self, columns: list[str], parameters: Parameters | None = None) -> None:
        """Initialize DuplicateCount operation.

        Args:
            columns: List of columns to check for duplicates
            parameters: Optional parameters for CTE customization
        """
        OpValueMixin.__init__(self, parameters)
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
        """
        Determine whether another object is equal to this DuplicateCount based on its columns.

        Args:
            other (Any): The object to compare against.

        Returns:
            True if `other` is a DuplicateCount with the same `columns`, `False` otherwise.
        """
        if not isinstance(other, DuplicateCount):
            return NotImplemented  # pragma: no cover
        return self.columns == other.columns

    def __hash__(self) -> int:
        """
        Compute a stable hash that uniquely identifies this DuplicateCount operation.

        The hash is derived from the operation's name, the tuple of columns, and the operation parameters (each value processed with freeze_for_hashing and the parameter pairs sorted) to ensure deterministic hashing regardless of parameter insertion order.

        Returns:
            int: Integer hash based on (name, columns, sorted hashed parameters).
        """
        return hash(
            (
                self.name,
                tuple(self.columns),
                tuple(sorted((k, freeze_for_hashing(v)) for k, v in self.parameters.items())),
            )
        )

    def __repr__(self) -> str:  # pragma: no cover
        """
        Return the operation's developer-facing representation.

        Returns:
            str: The operation's name (the value of the `name` property).
        """
        return self.name

    def __str__(self) -> str:  # pragma: no cover
        """
        Provide the object's informal string representation.

        Returns:
            A human-readable string representing the object.
        """
        return self.__repr__()


class CountValues(OpValueMixin[float], SqlOp[float]):
    """Count occurrences of specific value(s) in a column.

    Accepts single values (int, str, or bool) or lists of values (list[int] or list[str]).
    Lists must be homogeneous - all integers or all strings, not mixed.
    Boolean values are supported as single values only, not in lists.
    String values will be properly escaped in SQL generation to prevent injection.
    """

    __match_args__ = ("column", "values", "parameters")

    def __init__(
        self,
        column: str,
        values: int | str | bool | list[int] | list[str],
        parameters: Parameters | None = None,
    ) -> None:
        """Initialize CountValues operation.

        Args:
            column: Column name to count values in
            values: Values to count (int, str, bool, or list)
            parameters: Optional parameters for CTE customization
        """
        OpValueMixin.__init__(self, parameters)

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
        """
        Determine whether another object represents the same CountValues operation.

        Args:
            other (Any): Object to compare against.

        Returns:
            True if `other` is a CountValues with the same `column`, identical `values`, and the `values` have the same type; False otherwise.
        """
        if not isinstance(other, CountValues):
            return NotImplemented  # pragma: no cover
        # Need to check both value and type to distinguish True from 1, False from 0
        return self.column == other.column and self.values == other.values and type(self.values) is type(other.values)

    def __hash__(self) -> int:
        # Convert lists to tuples for hashing
        hashable_values = self.values if not isinstance(self.values, list) else tuple(self.values)
        return hash(
            (
                self.name,
                self.column,
                hashable_values,
                tuple(sorted((k, freeze_for_hashing(v)) for k, v in self.parameters.items())),
            )
        )

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


__all__ = [
    "Op",
    "SqlOp",
    "OpValueMixin",
    "NumRows",
    "Average",
    "CustomSQL",
    "Minimum",
    "Maximum",
    "Sum",
    "Variance",
    "First",
    "NullCount",
    "NegativeCount",
    "UniqueCount",
    "DuplicateCount",
    "CountValues",
]
