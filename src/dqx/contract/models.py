"""Contract dataclasses for DQX data quality contracts.

This module defines the full type system and check specifications used to
declare data contracts: schema types, validators, table-level checks,
column-level checks, column specs, SLA specs, and the top-level Contract.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

import sympy as sp
import yaml

from dqx.api import Context, DecoratedCheck, MetricProvider
from dqx.api import check as dqx_check
from dqx.common import SeverityLevel, validate_tags

# ---------------------------------------------------------------------------
# Error classes
# ---------------------------------------------------------------------------


class ContractValidationError(Exception):
    """Raised when a contract field fails validation."""


class SchemaValidationError(Exception):
    """Reserved for future use when schema validation is performed.

    This error will be raised during ``contract.to_checks()`` when the
    contract schema cannot be validated against the underlying data source
    (e.g., a column declared in the contract does not exist in the table).
    It is kept separate from :class:`ContractValidationError` to allow
    callers to distinguish between structural contract errors and
    schema-compatibility errors.
    """


class ContractWarning(UserWarning):
    """Emitted for contract configurations that are valid but potentially misconfigured.

    Use this class to filter DQX contract warnings independently of other
    ``UserWarning`` instances in your application.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_tags(tags: frozenset[str] | set[str] | None) -> frozenset[str]:
    """Normalize and validate tags, re-raising ValueError as ContractValidationError.

    Args:
        tags: Tag set to normalize.

    Returns:
        Validated frozenset of tags.

    Raises:
        ContractValidationError: If any tag is invalid.
    """
    try:
        return validate_tags(tags)
    except ValueError as exc:
        raise ContractValidationError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Type system
# ---------------------------------------------------------------------------

SimpleContractType = Literal["int", "float", "bool", "string", "bytes", "date", "time", "decimal", "timestamp"]

SIMPLE_TYPES: frozenset[str] = frozenset(get_args(SimpleContractType))


@dataclass(frozen=True)
class TimestampType:
    """A timestamp type with an optional timezone.

    ``tz=None`` (the default) means timezone-naive, corresponding to
    ``pa.timestamp(unit, tz=None)`` in PyArrow.  Set ``tz="UTC"`` for an
    explicit UTC timestamp, or any other IANA timezone string for a
    timezone-aware timestamp.

    The simple ``"timestamp"`` string form in YAML/user-facing API is
    normalised to ``TimestampType()`` (i.e. ``tz=None``) at
    :class:`ColumnSpec` construction time.

    Args:
        tz: IANA timezone string (e.g. ``"UTC"``, ``"America/New_York"``).
            Defaults to ``None`` (timezone-naive).

    Raises:
        ContractValidationError: If tz is an empty string.
    """

    tz: str | None = None

    def __post_init__(self) -> None:
        """Validate that tz, if provided, is non-empty."""
        if self.tz == "":
            raise ContractValidationError("TimestampType tz must be a non-empty string when specified")


@dataclass(frozen=True)
class ListType:
    """A list (array) type wrapping an element type.

    Args:
        value_type: The element type for list items.
    """

    value_type: ContractType


@dataclass(frozen=True)
class StructField:
    """A single named field within a StructType.

    Args:
        name: Field name (non-empty).
        type: The field's data type.
        description: Human-readable description (non-empty).

    Raises:
        ContractValidationError: If name or description is empty.
    """

    name: str
    type: ContractType
    description: str

    def __post_init__(self) -> None:
        """Validate name and description are non-empty."""
        if not self.name:
            raise ContractValidationError("StructField name must be non-empty")
        if not self.description:
            raise ContractValidationError("StructField description must be non-empty")


@dataclass(frozen=True)
class StructType:
    """A struct (record) type composed of named fields.

    Args:
        fields: Tuple of StructField instances (must be non-empty).

    Raises:
        ContractValidationError: If fields is empty.
    """

    fields: tuple[StructField, ...]

    def __post_init__(self) -> None:
        """Validate that at least one field is provided."""
        if not self.fields:
            raise ContractValidationError("StructType fields must be non-empty")


@dataclass(frozen=True)
class MapType:
    """A map (dictionary) type with key and value types.

    Args:
        key_type: The type for map keys.
        value_type: The type for map values.
    """

    key_type: ContractType
    value_type: ContractType


# Recursive union — requires `from __future__ import annotations`
ContractType = SimpleContractType | TimestampType | ListType | StructType | MapType

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MinValidator:
    """Validator that passes when the metric is >= (threshold - tolerance).

    Args:
        threshold: Lower bound (inclusive).
        tolerance: Floating-point comparison tolerance (must be >= 0).
            Defaults to 1e-9.

    Raises:
        ContractValidationError: If tolerance < 0.
    """

    threshold: float
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        """Validate tolerance is non-negative."""
        if self.tolerance < 0:
            raise ContractValidationError(f"MinValidator: tolerance must be >= 0, got {self.tolerance}")


@dataclass(frozen=True)
class MaxValidator:
    """Validator that passes when the metric is <= (threshold + tolerance).

    Args:
        threshold: Upper bound (inclusive).
        tolerance: Floating-point comparison tolerance (must be >= 0).
            Defaults to 1e-9.

    Raises:
        ContractValidationError: If tolerance < 0.
    """

    threshold: float
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        """Validate tolerance is non-negative."""
        if self.tolerance < 0:
            raise ContractValidationError(f"MaxValidator: tolerance must be >= 0, got {self.tolerance}")


@dataclass(frozen=True)
class BetweenValidator:
    """Validator that passes when the metric is within [low, high] (inclusive).

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        tolerance: Floating-point comparison tolerance (must be >= 0).
            Defaults to 1e-9.

    Raises:
        ContractValidationError: If low > high or tolerance < 0.
    """

    low: float
    high: float
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        """Validate low <= high and tolerance is non-negative."""
        if self.low > self.high:
            raise ContractValidationError(f"BetweenValidator: low {self.low} > high {self.high}")
        if self.tolerance < 0:
            raise ContractValidationError(f"BetweenValidator: tolerance must be >= 0, got {self.tolerance}")


@dataclass(frozen=True)
class NotBetweenValidator:
    """Validator that passes when the metric is outside (low, high) exclusively.

    Args:
        low: Lower bound of the excluded range.
        high: Upper bound of the excluded range.
        tolerance: Floating-point comparison tolerance (must be >= 0).
            Defaults to 1e-9.

    Raises:
        ContractValidationError: If low > high or tolerance < 0.
    """

    low: float
    high: float
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        """Validate low <= high and tolerance is non-negative."""
        if self.low > self.high:
            raise ContractValidationError(f"NotBetweenValidator: low {self.low} > high {self.high}")
        if self.tolerance < 0:
            raise ContractValidationError(f"NotBetweenValidator: tolerance must be >= 0, got {self.tolerance}")


@dataclass(frozen=True)
class EqualsValidator:
    """Validator that passes when the metric equals value (within tolerance).

    Args:
        value: Expected exact value.
        tolerance: Floating-point comparison tolerance (must be >= 0).
            Defaults to 1e-9.

    Raises:
        ContractValidationError: If tolerance < 0.
    """

    value: float
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        """Validate tolerance is non-negative."""
        if self.tolerance < 0:
            raise ContractValidationError(f"EqualsValidator: tolerance must be >= 0, got {self.tolerance}")


Validator = MinValidator | MaxValidator | BetweenValidator | NotBetweenValidator | EqualsValidator


# ---------------------------------------------------------------------------
# ReturnType
# ---------------------------------------------------------------------------

ReturnType = Literal["count", "pct"]

_RETURN_TYPES: frozenset[str] = frozenset({"count", "pct"})


def _validate_return_type(value: str) -> None:
    """Validate that value is a valid ReturnType.

    Args:
        value: The return_type value to validate.

    Raises:
        ContractValidationError: If value is not 'count' or 'pct'.
    """
    if value not in _RETURN_TYPES:
        raise ContractValidationError(f"return_type must be 'count' or 'pct', got '{value}'")


def _validate_single_validator(cls_name: str, validators: tuple[object, ...]) -> None:
    """Validate that at most one validator is provided.

    Args:
        cls_name: Name of the calling check class (for error messages).
        validators: Tuple of validators to check.

    Raises:
        ContractValidationError: If more than one validator is provided.
    """
    if len(validators) > 1:
        raise ContractValidationError(f"{cls_name} validators must have at most 1 entry, got {len(validators)}")


_AGGREGATION_VALUES: frozenset[str] = frozenset({"max", "min"})
_GRANULARITY_VALUES: frozenset[str] = frozenset({"hourly", "daily", "weekly", "monthly"})

FormatShortcut = Literal["email", "phone", "uuid", "url", "ipv4", "ipv6", "date", "datetime"]
_FORMAT_SHORTCUTS: frozenset[str] = frozenset({"email", "phone", "uuid", "url", "ipv4", "ipv6", "date", "datetime"})

_VALID_FLAG_NAMES: frozenset[str] = frozenset({"IGNORECASE", "MULTILINE", "DOTALL", "VERBOSE", "ASCII", "UNICODE"})

# ---------------------------------------------------------------------------
# Table-level check dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumRowsCheck:
    """Check that the number of table rows satisfies a validator.

    Args:
        name: Check name (non-empty).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty or validators has more than
            1 entry.
    """

    name: str
    validators: tuple[Validator, ...] = ()
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, validators length, and normalize tags."""
        if not self.name:
            raise ContractValidationError("NumRowsCheck name must be non-empty")
        _validate_single_validator("NumRowsCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, mp: MetricProvider, ctx: Context) -> None:
        """Emit assertions for this check into ``ctx`` using ``mp``.

        Args:
            mp: MetricProvider used to compute the row-count metric.
            ctx: Context in which assertions are registered.
        """
        _apply_validators(mp.num_rows(), ctx, self.name, self.severity, self.tags, self.validators)


@dataclass(frozen=True)
class TableDuplicatesCheck:
    """Check for duplicate rows across specified columns.

    Args:
        name: Check name (non-empty).
        columns: Tuple of column names to check for duplicates (non-empty;
            each name must be non-empty).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty, columns is empty, any
            column name is empty, or validators has more than 1 entry.
    """

    name: str
    columns: tuple[str, ...]
    validators: tuple[Validator, ...] = ()
    return_type: ReturnType = "count"
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, columns, return_type, validators length, and normalize tags."""
        _validate_return_type(self.return_type)
        if not self.name:
            raise ContractValidationError("TableDuplicatesCheck name must be non-empty")
        if not self.columns:
            raise ContractValidationError("TableDuplicatesCheck columns must be non-empty")
        for col in self.columns:
            if not col:
                raise ContractValidationError("Each column name in TableDuplicatesCheck must be non-empty")
        _validate_single_validator("TableDuplicatesCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, mp: MetricProvider, ctx: Context) -> None:
        """Emit assertions for this check into ``ctx`` using ``mp``.

        Args:
            mp: MetricProvider used to compute the duplicate-count metric.
            ctx: Context in which assertions are registered.
        """
        metric = mp.duplicate_count(list(self.columns))
        if self.return_type == "pct":
            # On empty tables num_rows() evaluates to 0; the resulting NULL metric
            # propagates as an EvaluationFailure (assertion fails).
            metric = metric / mp.num_rows()
        _apply_validators(metric, ctx, self.name, self.severity, self.tags, self.validators)


@dataclass(frozen=True)
class FreshnessCheck:
    """Check that data is fresh within a maximum age window.

    Args:
        name: Check name (non-empty).
        max_age_hours: Maximum acceptable age in hours (must be > 0).
        timestamp_column: Column used to determine freshness (non-empty).
        aggregation: Whether to use "max" or "min" of the timestamp column.
            Defaults to "max".
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty, max_age_hours <= 0,
            timestamp_column is empty, or aggregation is invalid.
    """

    name: str
    max_age_hours: float
    timestamp_column: str
    aggregation: Literal["max", "min"] = "max"
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, max_age_hours, timestamp_column, and aggregation."""
        if not self.name:
            raise ContractValidationError("FreshnessCheck name must be non-empty")
        if self.max_age_hours <= 0:
            raise ContractValidationError(f"FreshnessCheck max_age_hours must be > 0, got {self.max_age_hours}")
        if not self.timestamp_column:
            raise ContractValidationError("FreshnessCheck timestamp_column must be non-empty")
        if self.aggregation not in _AGGREGATION_VALUES:
            raise ContractValidationError(
                f"FreshnessCheck aggregation must be one of {sorted(_AGGREGATION_VALUES)}, got '{self.aggregation}'"
            )
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, mp: MetricProvider, ctx: Context) -> None:  # noqa: ARG002
        """Not yet supported — raises ``NotImplementedError``.

        Args:
            mp: Unused.
            ctx: Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("FreshnessCheck.to_dqx() is not yet supported")


@dataclass(frozen=True)
class CompletenessCheck:
    """Check that no partitions are missing within a lookback window.

    Args:
        name: Check name (non-empty).
        partition_column: Column used to detect partition gaps (non-empty).
        granularity: Partition granularity: "hourly", "daily", "weekly", or
            "monthly".
        lookback_days: Number of days to look back for gaps (must be > 0).
            Defaults to 30.
        allow_future_gaps: If True, gaps in the future are ignored. Defaults
            to True.
        max_gap_count: Maximum allowed number of gaps (must be >= 0). Defaults
            to 0.
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name or partition_column is empty,
            lookback_days <= 0, max_gap_count < 0, or granularity is invalid.
    """

    name: str
    partition_column: str
    granularity: Literal["hourly", "daily", "weekly", "monthly"]
    lookback_days: int = 30
    allow_future_gaps: bool = True
    max_gap_count: int = 0
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, partition_column, granularity, lookback_days, and max_gap_count."""
        if not self.name:
            raise ContractValidationError("CompletenessCheck name must be non-empty")
        if not self.partition_column:
            raise ContractValidationError("CompletenessCheck partition_column must be non-empty")
        if self.granularity not in _GRANULARITY_VALUES:
            raise ContractValidationError(
                f"CompletenessCheck granularity must be one of {sorted(_GRANULARITY_VALUES)}, got '{self.granularity}'"
            )
        if self.lookback_days <= 0:
            raise ContractValidationError(f"CompletenessCheck lookback_days must be > 0, got {self.lookback_days}")
        if self.max_gap_count < 0:
            raise ContractValidationError(f"CompletenessCheck max_gap_count must be >= 0, got {self.max_gap_count}")
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, mp: MetricProvider, ctx: Context) -> None:  # noqa: ARG002
        """Not yet supported — raises ``NotImplementedError``.

        Args:
            mp: Unused.
            ctx: Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("CompletenessCheck.to_dqx() is not yet supported")


TableCheck = NumRowsCheck | TableDuplicatesCheck | FreshnessCheck | CompletenessCheck

# ---------------------------------------------------------------------------
# Column-level check dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MissingCheck:
    """Check for missing (null) values in a column.

    Args:
        name: Check name (non-empty).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty or validators has more than
            1 entry.
    """

    name: str
    validators: tuple[Validator, ...] = ()
    return_type: ReturnType = "count"
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, return_type, validators length, and normalize tags."""
        _validate_return_type(self.return_type)
        if not self.name:
            raise ContractValidationError("MissingCheck name must be non-empty")
        _validate_single_validator("MissingCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, column: str, mp: MetricProvider, ctx: Context) -> None:
        """Emit assertions for this check into ``ctx`` using ``mp``.

        Args:
            column: Name of the column to check for missing values.
            mp: MetricProvider used to compute the null-count metric.
            ctx: Context in which assertions are registered.
        """
        metric = mp.null_count(column)
        if self.return_type == "pct":
            # On empty tables num_rows() evaluates to 0; the resulting NULL metric
            # propagates as an EvaluationFailure (assertion fails).
            metric = metric / mp.num_rows()
        _apply_validators(metric, ctx, self.name, self.severity, self.tags, self.validators)


@dataclass(frozen=True)
class ColumnDuplicatesCheck:
    """Check for duplicate values within a single column.

    Args:
        name: Check name (non-empty).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty or validators has more than
            1 entry.
    """

    name: str
    validators: tuple[Validator, ...] = ()
    return_type: ReturnType = "count"
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, return_type, validators length, and normalize tags."""
        _validate_return_type(self.return_type)
        if not self.name:
            raise ContractValidationError("ColumnDuplicatesCheck name must be non-empty")
        _validate_single_validator("ColumnDuplicatesCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, column: str, mp: MetricProvider, ctx: Context) -> None:
        """Emit assertions for this check into ``ctx`` using ``mp``.

        Args:
            column: Name of the column to check for duplicate values.
            mp: MetricProvider used to compute the duplicate-count metric.
            ctx: Context in which assertions are registered.
        """
        metric = mp.duplicate_count([column])
        if self.return_type == "pct":
            # On empty tables num_rows() evaluates to 0; the resulting NULL metric
            # propagates as an EvaluationFailure (assertion fails).
            metric = metric / mp.num_rows()
        _apply_validators(metric, ctx, self.name, self.severity, self.tags, self.validators)


@dataclass(frozen=True)
class WhitelistCheck:
    """Check that column values are within an allowed set.

    Args:
        name: Check name (non-empty).
        values: Allowed values (non-empty tuple).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        case_sensitive: Whether string comparisons are case-sensitive.
            Defaults to True.
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty, values is empty, or
            validators has more than 1 entry.
    """

    name: str
    values: tuple[str | int | float, ...]
    validators: tuple[Validator, ...] = ()
    return_type: ReturnType = "count"
    case_sensitive: bool = True
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, values, return_type, validators length, and normalize tags."""
        _validate_return_type(self.return_type)
        if not self.name:
            raise ContractValidationError("WhitelistCheck name must be non-empty")
        if not self.values:
            raise ContractValidationError("WhitelistCheck values must be non-empty")
        _validate_single_validator("WhitelistCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, column: str, mp: MetricProvider, ctx: Context) -> None:
        """Emit assertions for this check into ``ctx`` using ``mp``.

        The metric counts rows whose value is in the whitelist (matching rows).

        Args:
            column: Name of the column to check.
            mp: MetricProvider used to compute the count-values metric.
            ctx: Context in which assertions are registered.

        Raises:
            NotImplementedError: If ``case_sensitive=False`` is requested,
                as ``mp.count_values()`` does not support case-insensitive
                comparisons.
        """
        if not self.case_sensitive:
            raise NotImplementedError("WhitelistCheck.to_dqx() does not support case_sensitive=False")
        metric = mp.count_values(column, list(self.values))  # type: ignore[arg-type]
        if self.return_type == "pct":
            # Note: on empty tables num_rows() = 0; the resulting NULL metric
            # causes the assertion to fail with an EvaluationFailure.
            metric = metric / mp.num_rows()
        _apply_validators(metric, ctx, self.name, self.severity, self.tags, self.validators)


@dataclass(frozen=True)
class BlacklistCheck:
    """Check that column values are not in a forbidden set.

    Args:
        name: Check name (non-empty).
        values: Forbidden values (non-empty tuple).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        case_sensitive: Whether string comparisons are case-sensitive.
            Defaults to True.
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty, values is empty, or
            validators has more than 1 entry.
    """

    name: str
    values: tuple[str | int | float, ...]
    validators: tuple[Validator, ...] = ()
    return_type: ReturnType = "count"
    case_sensitive: bool = True
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, values, return_type, validators length, and normalize tags."""
        _validate_return_type(self.return_type)
        if not self.name:
            raise ContractValidationError("BlacklistCheck name must be non-empty")
        if not self.values:
            raise ContractValidationError("BlacklistCheck values must be non-empty")
        _validate_single_validator("BlacklistCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, column: str, mp: MetricProvider, ctx: Context) -> None:
        """Emit assertions for this check into ``ctx`` using ``mp``.

        The metric counts rows whose value is NOT in the blacklist (safe rows).

        Args:
            column: Name of the column to check.
            mp: MetricProvider used to compute the safe-row-count metric.
            ctx: Context in which assertions are registered.

        Raises:
            NotImplementedError: If ``case_sensitive=False`` is requested,
                as ``mp.count_values()`` does not support case-insensitive
                comparisons.
        """
        if not self.case_sensitive:
            raise NotImplementedError("BlacklistCheck.to_dqx() does not support case_sensitive=False")
        metric = mp.num_rows() - mp.count_values(column, list(self.values))  # type: ignore[arg-type]
        if self.return_type == "pct":
            # Note: on empty tables num_rows() = 0; the resulting NULL metric
            # causes the assertion to fail with an EvaluationFailure.
            metric = metric / mp.num_rows()
        _apply_validators(metric, ctx, self.name, self.severity, self.tags, self.validators)


@dataclass(frozen=True)
class PatternCheck:
    """Check that column values match a regex pattern or format shortcut.

    Exactly one of ``pattern`` or ``format`` must be set. Flags cannot be
    combined with a format shortcut.

    Args:
        name: Check name (non-empty).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        pattern: Regular expression pattern string (non-empty when provided).
        format: Predefined format shortcut (e.g. "email", "uuid").
        flags: Regex flags (only valid with ``pattern``).
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty, neither/both of
            pattern/format are set, flags are used with format, pattern
            is an empty string, format is unknown, pattern is invalid
            regex, flags contain unknown names, or validators has more
            than 1 entry.
    """

    name: str
    validators: tuple[Validator, ...] = ()
    pattern: str | None = None
    format: FormatShortcut | None = None
    flags: tuple[str, ...] = ()
    return_type: ReturnType = "count"
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, pattern/format exclusivity, return_type, regex syntax, flags, and validators length."""
        _validate_return_type(self.return_type)
        if not self.name:
            raise ContractValidationError("PatternCheck name must be non-empty")
        has_pattern = self.pattern is not None
        has_format = self.format is not None
        if has_pattern and has_format:
            raise ContractValidationError("PatternCheck: exactly one of 'pattern' or 'format' must be set, not both")
        if not has_pattern and not has_format:
            raise ContractValidationError("PatternCheck: exactly one of 'pattern' or 'format' must be set")
        if has_format and self.flags:
            raise ContractValidationError("PatternCheck: flags cannot be combined with a format shortcut")
        if has_pattern and self.pattern == "":
            raise ContractValidationError("PatternCheck: pattern must be non-empty when provided")
        if has_format and self.format not in _FORMAT_SHORTCUTS:
            raise ContractValidationError(
                f"PatternCheck: unknown format shortcut '{self.format}'; must be one of {sorted(_FORMAT_SHORTCUTS)}"
            )
        if has_pattern and self.pattern is not None:
            flags_bitmask = 0
            for flag in self.flags:
                if flag not in _VALID_FLAG_NAMES:
                    raise ContractValidationError(
                        f"PatternCheck: unknown flag '{flag}'; must be one of {sorted(_VALID_FLAG_NAMES)}"
                    )
                flag_value = getattr(re, flag, None)
                if flag_value is None:  # pragma: no cover
                    raise ContractValidationError(
                        f"PatternCheck: unknown flag '{flag}'; must be one of {sorted(_VALID_FLAG_NAMES)}"
                    )
                flags_bitmask |= int(flag_value)
            try:
                re.compile(self.pattern, flags_bitmask)
            except re.error as exc:
                raise ContractValidationError(f"PatternCheck: invalid regex pattern '{self.pattern}': {exc}") from exc
        _validate_single_validator("PatternCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, column: str, mp: MetricProvider, ctx: Context) -> None:  # noqa: ARG002
        """Not yet supported — raises ``NotImplementedError``.

        Args:
            column: Unused.
            mp: Unused.
            ctx: Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("PatternCheck.to_dqx() is not yet supported")


@dataclass(frozen=True)
class MinLengthCheck:
    """Check that column string values meet a minimum length.

    Args:
        name: Check name (non-empty).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty or validators has more than
            1 entry.
    """

    name: str
    validators: tuple[Validator, ...] = ()
    return_type: ReturnType = "count"
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, return_type, validators length, and normalize tags."""
        _validate_return_type(self.return_type)
        if not self.name:
            raise ContractValidationError("MinLengthCheck name must be non-empty")
        _validate_single_validator("MinLengthCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, column: str, mp: MetricProvider, ctx: Context) -> None:  # noqa: ARG002
        """Not yet supported — raises ``NotImplementedError``.

        Args:
            column: Unused.
            mp: Unused.
            ctx: Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("MinLengthCheck.to_dqx() is not yet supported")


@dataclass(frozen=True)
class MaxLengthCheck:
    """Check that column string values do not exceed a maximum length.

    Args:
        name: Check name (non-empty).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty or validators has more than
            1 entry.
    """

    name: str
    validators: tuple[Validator, ...] = ()
    return_type: ReturnType = "count"
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, return_type, validators length, and normalize tags."""
        _validate_return_type(self.return_type)
        if not self.name:
            raise ContractValidationError("MaxLengthCheck name must be non-empty")
        _validate_single_validator("MaxLengthCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, column: str, mp: MetricProvider, ctx: Context) -> None:  # noqa: ARG002
        """Not yet supported — raises ``NotImplementedError``.

        Args:
            column: Unused.
            mp: Unused.
            ctx: Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("MaxLengthCheck.to_dqx() is not yet supported")


@dataclass(frozen=True)
class AvgLengthCheck:
    """Check that the average string length satisfies a validator.

    Args:
        name: Check name (non-empty).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty or validators has more than
            1 entry.
    """

    name: str
    validators: tuple[Validator, ...] = ()
    return_type: ReturnType = "count"
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, return_type, validators length, and normalize tags."""
        _validate_return_type(self.return_type)
        if not self.name:
            raise ContractValidationError("AvgLengthCheck name must be non-empty")
        _validate_single_validator("AvgLengthCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, column: str, mp: MetricProvider, ctx: Context) -> None:  # noqa: ARG002
        """Not yet supported — raises ``NotImplementedError``.

        Args:
            column: Unused.
            mp: Unused.
            ctx: Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("AvgLengthCheck.to_dqx() is not yet supported")


@dataclass(frozen=True)
class CardinalityCheck:
    """Check the number of distinct values in a column.

    Args:
        name: Check name (non-empty).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty or validators has more than
            1 entry.
    """

    name: str
    validators: tuple[Validator, ...] = ()
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, validators length, and normalize tags."""
        if not self.name:
            raise ContractValidationError("CardinalityCheck name must be non-empty")
        _validate_single_validator("CardinalityCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, column: str, mp: MetricProvider, ctx: Context) -> None:
        """Emit assertions for this check into ``ctx`` using ``mp``.

        Args:
            column: Name of the column to check distinct-value count for.
            mp: MetricProvider used to compute the unique-count metric.
            ctx: Context in which assertions are registered.
        """
        _apply_validators(mp.unique_count(column), ctx, self.name, self.severity, self.tags, self.validators)


@dataclass(frozen=True)
class MinCheck:
    """Check the minimum value of a numeric column.

    Args:
        name: Check name (non-empty).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty or validators has more than
            1 entry.
    """

    name: str
    validators: tuple[Validator, ...] = ()
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, validators length, and normalize tags."""
        if not self.name:
            raise ContractValidationError("MinCheck name must be non-empty")
        _validate_single_validator("MinCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, column: str, mp: MetricProvider, ctx: Context) -> None:
        """Emit assertions for this check into ``ctx`` using ``mp``.

        Args:
            column: Name of the column to compute the minimum for.
            mp: MetricProvider used to compute the minimum metric.
            ctx: Context in which assertions are registered.
        """
        _apply_validators(mp.minimum(column), ctx, self.name, self.severity, self.tags, self.validators)


@dataclass(frozen=True)
class MaxCheck:
    """Check the maximum value of a numeric column.

    Args:
        name: Check name (non-empty).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty or validators has more than
            1 entry.
    """

    name: str
    validators: tuple[Validator, ...] = ()
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, validators length, and normalize tags."""
        if not self.name:
            raise ContractValidationError("MaxCheck name must be non-empty")
        _validate_single_validator("MaxCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, column: str, mp: MetricProvider, ctx: Context) -> None:
        """Emit assertions for this check into ``ctx`` using ``mp``.

        Args:
            column: Name of the column to compute the maximum for.
            mp: MetricProvider used to compute the maximum metric.
            ctx: Context in which assertions are registered.
        """
        _apply_validators(mp.maximum(column), ctx, self.name, self.severity, self.tags, self.validators)


@dataclass(frozen=True)
class MeanCheck:
    """Check the mean value of a numeric column.

    Args:
        name: Check name (non-empty).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty or validators has more than
            1 entry.
    """

    name: str
    validators: tuple[Validator, ...] = ()
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, validators length, and normalize tags."""
        if not self.name:
            raise ContractValidationError("MeanCheck name must be non-empty")
        _validate_single_validator("MeanCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, column: str, mp: MetricProvider, ctx: Context) -> None:
        """Emit assertions for this check into ``ctx`` using ``mp``.

        Args:
            column: Name of the column to compute the average for.
            mp: MetricProvider used to compute the average metric.
            ctx: Context in which assertions are registered.
        """
        _apply_validators(mp.average(column), ctx, self.name, self.severity, self.tags, self.validators)


@dataclass(frozen=True)
class SumCheck:
    """Check the sum of a numeric column.

    Args:
        name: Check name (non-empty).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty or validators has more than
            1 entry.
    """

    name: str
    validators: tuple[Validator, ...] = ()
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, validators length, and normalize tags."""
        if not self.name:
            raise ContractValidationError("SumCheck name must be non-empty")
        _validate_single_validator("SumCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, column: str, mp: MetricProvider, ctx: Context) -> None:
        """Emit assertions for this check into ``ctx`` using ``mp``.

        Args:
            column: Name of the column to sum.
            mp: MetricProvider used to compute the sum metric.
            ctx: Context in which assertions are registered.
        """
        _apply_validators(mp.sum(column), ctx, self.name, self.severity, self.tags, self.validators)


@dataclass(frozen=True)
class CountCheck:
    """Check the count of (non-null) values in a column.

    Args:
        name: Check name (non-empty).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty or validators has more than
            1 entry.
    """

    name: str
    validators: tuple[Validator, ...] = ()
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, validators length, and normalize tags."""
        if not self.name:
            raise ContractValidationError("CountCheck name must be non-empty")
        _validate_single_validator("CountCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, column: str, mp: MetricProvider, ctx: Context) -> None:
        """Emit assertions for this check into ``ctx`` using ``mp``.

        The metric is derived as ``num_rows - null_count`` — the count of
        non-null values in the column.

        Args:
            column: Name of the column to count non-null values for.
            mp: MetricProvider used to derive the non-null count metric.
            ctx: Context in which assertions are registered.
        """
        metric = mp.num_rows() - mp.null_count(column)
        _apply_validators(metric, ctx, self.name, self.severity, self.tags, self.validators)


@dataclass(frozen=True)
class VarianceCheck:
    """Check the variance of a numeric column.

    Args:
        name: Check name (non-empty).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty or validators has more than
            1 entry.
    """

    name: str
    validators: tuple[Validator, ...] = ()
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, validators length, and normalize tags."""
        if not self.name:
            raise ContractValidationError("VarianceCheck name must be non-empty")
        _validate_single_validator("VarianceCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, column: str, mp: MetricProvider, ctx: Context) -> None:
        """Emit assertions for this check into ``ctx`` using ``mp``.

        Args:
            column: Name of the column to compute variance for.
            mp: MetricProvider used to compute the variance metric.
            ctx: Context in which assertions are registered.
        """
        _apply_validators(mp.variance(column), ctx, self.name, self.severity, self.tags, self.validators)


@dataclass(frozen=True)
class StddevCheck:
    """Check the standard deviation of a numeric column.

    Args:
        name: Check name (non-empty).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty or validators has more than
            1 entry.
    """

    name: str
    validators: tuple[Validator, ...] = ()
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, validators length, and normalize tags."""
        if not self.name:
            raise ContractValidationError("StddevCheck name must be non-empty")
        _validate_single_validator("StddevCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, column: str, mp: MetricProvider, ctx: Context) -> None:
        """Emit assertions for this check into ``ctx`` using ``mp``.

        Computes the standard deviation as ``sp.sqrt(mp.variance(column))``,
        composing the built-in variance metric with a symbolic square-root
        rather than falling back to raw SQL.

        Args:
            column: Name of the column to compute standard deviation for.
            mp: MetricProvider used to compute the variance metric.
            ctx: Context in which assertions are registered.
        """
        _apply_validators(sp.sqrt(mp.variance(column)), ctx, self.name, self.severity, self.tags, self.validators)


@dataclass(frozen=True)
class PercentileCheck:
    """Check a specific percentile value of a numeric column.

    Args:
        name: Check name (non-empty).
        percentile: Percentile to compute, in [0.0, 1.0] inclusive (e.g. 0.5 for
            the median, 0.99 for the 99th percentile).
        validators: Tuple of at most one Validator. Empty tuple is a noop
            (check runs but never fails).
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty, percentile is out of
            range, or validators has more than 1 entry.
    """

    name: str
    percentile: float
    validators: tuple[Validator, ...] = ()
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, percentile range, validators length, and normalize tags."""
        if not self.name:
            raise ContractValidationError("PercentileCheck name must be non-empty")
        if not (0.0 <= self.percentile <= 1.0):
            raise ContractValidationError(f"PercentileCheck percentile must be in [0.0, 1.0], got {self.percentile}")
        _validate_single_validator("PercentileCheck", self.validators)
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    def to_dqx(self, column: str, mp: MetricProvider, ctx: Context) -> None:  # noqa: ARG002
        """Not yet supported — raises ``NotImplementedError``.

        Args:
            column: Unused.
            mp: Unused.
            ctx: Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("PercentileCheck.to_dqx() is not yet supported")


ColumnCheck = (
    MissingCheck
    | ColumnDuplicatesCheck
    | WhitelistCheck
    | BlacklistCheck
    | PatternCheck
    | MinLengthCheck
    | MaxLengthCheck
    | AvgLengthCheck
    | CardinalityCheck
    | MinCheck
    | MaxCheck
    | MeanCheck
    | SumCheck
    | CountCheck
    | VarianceCheck
    | StddevCheck
    | PercentileCheck
)

# ---------------------------------------------------------------------------
# ColumnSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a single column in a data contract.

    Args:
        name: Column name (non-empty).
        type: Column data type.
        description: Human-readable description (non-empty).
        nullable: Whether nulls are permitted. Defaults to True.
        metadata: Key-value metadata pairs. Defaults to empty tuple.
        checks: Column-level quality checks. Defaults to empty tuple.

    Raises:
        ContractValidationError: If name or description is empty.
    """

    name: str
    type: ContractType
    description: str
    nullable: bool = True
    metadata: tuple[tuple[str, str], ...] = ()
    checks: tuple[ColumnCheck, ...] = ()  # type: ignore[type-arg]

    def __post_init__(self) -> None:
        """Validate name, description, and simple type string."""
        if not self.name:
            raise ContractValidationError("ColumnSpec name must be non-empty")
        if not self.description:
            raise ContractValidationError("ColumnSpec description must be non-empty")
        # Normalize "timestamp" string to TimestampType() before the SIMPLE_TYPES check,
        # because "timestamp" IS in SIMPLE_TYPES but we always want the rich type object.
        if self.type == "timestamp":
            object.__setattr__(self, "type", TimestampType())
        if isinstance(self.type, str) and self.type not in SIMPLE_TYPES:
            raise ContractValidationError(
                f"ColumnSpec type '{self.type}' is not a valid simple type; "
                f"must be one of {sorted(SIMPLE_TYPES)} or a complex type"
            )


# ---------------------------------------------------------------------------
# SLASpec
# ---------------------------------------------------------------------------

_CRON_PATTERN = re.compile(r"^\S+ \S+ \S+ \S+ \S+$")

# Per-field patterns: allow *, numeric literals, ranges (n-m), steps (*/n, n/n, n-m/n)
# Month also allows JAN-DEC names; day-of-week allows SUN-SAT names (0-7)
_MONTH_NAMES = frozenset({"JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"})
_DOW_NAMES = frozenset({"SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"})

# Regex for a single cron step token (either * or numeric/name, with optional /step or -range/step)
_CRON_FIELD_RE = re.compile(
    r"^\*"  # wildcard
    r"(?:/\d+)?"  # optional step on wildcard
    r"$"
    r"|"
    r"^\d+"  # numeric literal
    r"(?:-\d+)?"  # optional range end
    r"(?:/\d+)?"  # optional step
    r"$"
)


def _cron_field_in_range(token: str, lo: int, hi: int, names: frozenset[str] | None = None) -> bool:
    """Return True if a single (non-list) cron token is numeric and within [lo, hi].

    Args:
        token: Single cron token (no commas).
        lo: Minimum allowed numeric value.
        hi: Maximum allowed numeric value.
        names: Optional set of allowed symbolic names (uppercase).

    Returns:
        True if the token is valid for the given range.
    """
    if token == "*":
        return True
    # Handle step on wildcard: */n
    if token.startswith("*/"):
        step_str = token[2:]
        return step_str.isdigit() and int(step_str) >= 1
    # Accept symbolic name
    if names and token.upper() in names:
        return True
    # Numeric / range / step patterns
    if not _CRON_FIELD_RE.match(token):
        return False
    # Parse numeric parts
    parts = token.split("/")
    range_part = parts[0]
    if len(parts) == 2:
        step_str = parts[1]
        # _CRON_FIELD_RE guarantees step_str is \d+, so isdigit() is always True here
        if int(step_str) < 1:
            return False
    range_ends = range_part.split("-")
    # Detect inverted range (e.g. "31-1")
    if len(range_ends) == 2:
        lo_val, hi_val = int(range_ends[0]), int(range_ends[1])
        if lo_val > hi_val:
            return False
    for part in range_ends:
        val = int(part)
        if not lo <= val <= hi:
            return False
    return True


def _validate_cron(schedule: str) -> None:
    """Validate a 5-field cron expression with per-field range checks.

    Validates that each field is syntactically valid and numerically within
    the allowed range for that position:

    * minute: 0-59
    * hour: 0-23
    * day-of-month: 1-31 (or *)
    * month: 1-12 (or JAN-DEC or *)
    * day-of-week: 0-7 (or SUN-SAT or *)

    List-based patterns (comma-separated values) in day-of-month and
    day-of-week are not supported.

    Args:
        schedule: Cron expression string to validate.

    Raises:
        ContractValidationError: If the expression is not a valid 5-field
            cron, uses unsupported list-based patterns, or contains
            out-of-range numeric values.
    """
    if not _CRON_PATTERN.match(schedule):
        raise ContractValidationError(f"Invalid cron expression: '{schedule}' must be a 5-field cron")
    minute, hour, day_of_month, month, day_of_week = schedule.split()
    if "," in day_of_month:
        raise ContractValidationError(f"Unsupported cron: list-based day-of-month '{day_of_month}'")
    if "," in day_of_week:
        raise ContractValidationError(f"Unsupported cron: list-based day-of-week '{day_of_week}'")
    if not _cron_field_in_range(minute, 0, 59):
        raise ContractValidationError(f"Invalid cron minute field '{minute}': must be 0-59, *, or a valid expression")
    if not _cron_field_in_range(hour, 0, 23):
        raise ContractValidationError(f"Invalid cron hour field '{hour}': must be 0-23, *, or a valid expression")
    if not _cron_field_in_range(day_of_month, 1, 31):
        raise ContractValidationError(
            f"Invalid cron day-of-month field '{day_of_month}': must be 1-31, *, or a valid expression"
        )
    if not _cron_field_in_range(month, 1, 12, _MONTH_NAMES):
        raise ContractValidationError(
            f"Invalid cron month field '{month}': must be 1-12, JAN-DEC, *, or a valid expression"
        )
    if not _cron_field_in_range(day_of_week, 0, 7, _DOW_NAMES):
        raise ContractValidationError(
            f"Invalid cron day-of-week field '{day_of_week}': must be 0-7, SUN-SAT, *, or a valid expression"
        )


def _is_hourly_or_daily(schedule: str) -> bool:
    """Return True if the cron schedule is hourly or daily (not catch-all, weekly, or monthly).

    An hourly or daily schedule satisfies all of:
    - ``minute`` is NOT ``"*"`` (has a specific minute, ruling out the every-minute catch-all)
    - ``day_of_month == "*"`` (not pinned to specific days of month)
    - ``month == "*"`` (not pinned to specific months)
    - ``day_of_week == "*"`` (not pinned to specific days of week)

    The pure catch-all ``"* * * * *"`` has a wildcard minute and returns ``False``.
    Weekly schedules (pinned ``day_of_week``) and monthly schedules (pinned
    ``day_of_month``) also return ``False``.

    Args:
        schedule: A validated 5-field cron expression.

    Returns:
        True if the schedule is an hourly or daily (but not catch-all) pattern.
    """
    parts = schedule.split()
    if len(parts) != 5:  # pragma: no cover
        return False
    minute, _hour, day_of_month, month, day_of_week = parts
    return minute != "*" and day_of_month == "*" and month == "*" and day_of_week == "*"


@dataclass(frozen=True)
class SLASpec:
    """Service Level Agreement specification.

    Args:
        schedule: 5-field cron expression defining the expected data arrival
            schedule (list-based day patterns are unsupported).
        lag_hours: Maximum acceptable lag in hours (must be >= 0).

    Raises:
        ContractValidationError: If lag_hours < 0 or schedule is invalid.
    """

    schedule: str
    lag_hours: float

    def __post_init__(self) -> None:
        """Validate lag_hours and cron schedule."""
        if self.lag_hours < 0:
            raise ContractValidationError(f"SLASpec lag_hours must be >= 0, got {self.lag_hours}")
        _validate_cron(self.schedule)
        # Emit a warning when lag_hours > 168 on hourly or daily schedules only.
        # Weekly, monthly, or catch-all schedules are excluded.
        if self.lag_hours > 168 and _is_hourly_or_daily(self.schedule):
            # stacklevel=2 points to the SLASpec(...) constructor call site.
            warnings.warn(
                f"SLASpec lag_hours={self.lag_hours} exceeds 168 hours (7 days) "
                f"on an hourly or daily schedule '{self.schedule}'. "
                "This may indicate a misconfigured SLA.",
                ContractWarning,
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def _parse_tags(raw: Any) -> frozenset[str]:
    """Parse a YAML tags value into a frozenset of strings.

    Args:
        raw: The raw YAML value for tags (list or None).

    Returns:
        frozenset of tag strings.

    Raises:
        ContractValidationError: If raw is a string or non-iterable (not a list).
    """
    if raw is None:
        return frozenset()
    if isinstance(raw, (str, bytes)):
        raise ContractValidationError(f"'tags' must be a list, got a string: {raw!r}")
    if not isinstance(raw, list):
        raise ContractValidationError(f"'tags' must be a list, got: {type(raw).__name__}")
    return frozenset(str(t) for t in raw)


def _parse_bool(value: Any, field_name: str, default: bool) -> bool:
    """Parse a boolean YAML value, supporting common string representations.

    Args:
        value: The raw YAML value (None, bool, int, or str).
        field_name: Name of the field for error messages.
        default: Default value to return if value is None.

    Returns:
        Parsed boolean value.

    Raises:
        ContractValidationError: If the value cannot be interpreted as a boolean.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        low = value.lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
    raise ContractValidationError(f"'{field_name}' must be a boolean (true/false), got: {value!r}")


def _require_str(value: Any, field_name: str) -> str:
    """Validate that a YAML value is a plain string scalar.

    Args:
        value: The raw YAML value to validate.
        field_name: Name of the field for error messages.

    Returns:
        The value cast to ``str``.

    Raises:
        ContractValidationError: If the value is not a ``str`` instance.
    """
    if not isinstance(value, str):
        raise ContractValidationError(f"'{field_name}' must be a string, got: {type(value).__name__} ({value!r})")
    return value


def _parse_severity(raw: Any) -> SeverityLevel:
    """Parse a YAML severity value, defaulting to 'P1'.

    Args:
        raw: The raw YAML value for severity.

    Returns:
        SeverityLevel string.

    Raises:
        ContractValidationError: If the severity value is not a valid SeverityLevel.
    """
    if raw is None:
        return "P1"
    severity = str(raw)
    valid_severities = set(get_args(SeverityLevel))
    if severity not in valid_severities:
        raise ContractValidationError(f"Unknown severity '{severity}'; must be one of {sorted(valid_severities)}")
    return severity  # type: ignore[return-value]


def _parse_type(raw: Any) -> ContractType:
    """Map a YAML type node to a ContractType.

    Args:
        raw: The raw YAML type value (str or dict).

    Returns:
        ContractType instance.

    Raises:
        ContractValidationError: If the type is unknown or invalid.
    """
    if isinstance(raw, str):
        if raw == "timestamp":
            return TimestampType()
        if raw in SIMPLE_TYPES:
            return raw  # type: ignore[return-value]
        raise ContractValidationError(
            f"Unknown type '{raw}'; must be one of {sorted(SIMPLE_TYPES)} or a complex type mapping"
        )
    if isinstance(raw, dict):
        kind = raw.get("kind")
        if kind == "timestamp":
            return TimestampType(tz=raw.get("tz"))
        if kind == "list":
            if "value_type" not in raw:
                raise ContractValidationError("ListType requires 'value_type'")
            return ListType(value_type=_parse_type(raw["value_type"]))
        if kind == "struct":
            if "fields" not in raw:
                raise ContractValidationError("StructType requires 'fields'")
            return StructType(fields=tuple(_parse_struct_field(f) for f in raw["fields"]))
        if kind == "map":
            if "key_type" not in raw:
                raise ContractValidationError("MapType requires 'key_type'")
            if "value_type" not in raw:
                raise ContractValidationError("MapType requires 'value_type'")
            return MapType(key_type=_parse_type(raw["key_type"]), value_type=_parse_type(raw["value_type"]))
        raise ContractValidationError(f"Unknown type kind: {raw.get('kind')}")
    raise ContractValidationError(f"Unknown type kind: {raw!r}")


def _parse_struct_field(raw: Any) -> StructField:
    """Parse a struct field dict into a StructField.

    Args:
        raw: Dict with 'name', 'type', and 'description' keys.

    Returns:
        StructField instance.

    Raises:
        ContractValidationError: If raw is not a mapping or required fields are missing.
    """
    if not isinstance(raw, dict):
        raise ContractValidationError(f"StructField entry must be a mapping, got: {type(raw).__name__}")
    for key in ("name", "type", "description"):
        if key not in raw:
            raise ContractValidationError(f"StructField missing required field: '{key}'")
    return StructField(
        name=_require_str(raw["name"], "StructField.name"),
        type=_parse_type(raw["type"]),
        description=_require_str(raw["description"], "StructField.description"),
    )


def _parse_float_field(value: Any, field_name: str) -> float:
    """Convert a YAML value to float, raising ContractValidationError on failure.

    Args:
        value: The raw YAML value to convert.
        field_name: Name of the field for error messages.

    Returns:
        Float representation of value.

    Raises:
        ContractValidationError: If the value cannot be converted to float.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"'{field_name}' must be a numeric value, got: {value!r}") from exc


def _parse_int_field(value: Any, field_name: str) -> int:
    """Convert a YAML value to int, raising ContractValidationError on failure.

    Rejects bool values (e.g. ``true``/``false``) and non-integral floats
    (e.g. ``1.9``).  Integral floats such as ``2.0`` are accepted and
    converted to ``2``.

    Args:
        value: The raw YAML value to convert.
        field_name: Name of the field for error messages.

    Returns:
        Integer representation of value.

    Raises:
        ContractValidationError: If the value is a bool, a non-integral float,
            or cannot be converted to int.
    """
    if isinstance(value, bool):
        raise ContractValidationError(f"'{field_name}' must be an integer value, got: {value!r}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ContractValidationError(
                f"'{field_name}' must be an integer value (non-integral float not allowed), got: {value!r}"
            )
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"'{field_name}' must be an integer value, got: {value!r}") from exc


def _parse_validator(raw: dict[str, Any]) -> Validator | None:
    """Parse a validator from a check YAML dict.

    At most one of min, max, between, not_between, equals may be present.

    Args:
        raw: The raw YAML dict for a check.

    Returns:
        Validator instance or None if no validator key found.

    Raises:
        ContractValidationError: If multiple validator keys are present, or if
            numeric values cannot be parsed, or if between/not_between do not
            provide exactly two values.
    """
    raw_tolerance = raw.get("tolerance")
    tolerance = _parse_float_field(raw_tolerance, "tolerance") if raw_tolerance is not None else 1e-9
    validator_keys = [k for k in ("min", "max", "between", "not_between", "equals") if k in raw]
    if len(validator_keys) > 1:
        raise ContractValidationError(f"Check specifies multiple validators: {validator_keys}")
    if not validator_keys:
        return None
    key = validator_keys[0]
    if key == "min":
        return MinValidator(threshold=_parse_float_field(raw["min"], "min"), tolerance=tolerance)
    if key == "max":
        return MaxValidator(threshold=_parse_float_field(raw["max"], "max"), tolerance=tolerance)
    if key == "between":
        bounds = raw["between"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ContractValidationError(f"'between' must be a list of exactly 2 numeric values, got: {bounds!r}")
        low = _parse_float_field(bounds[0], "between[0]")
        high = _parse_float_field(bounds[1], "between[1]")
        return BetweenValidator(low=low, high=high, tolerance=tolerance)
    if key == "not_between":
        bounds = raw["not_between"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ContractValidationError(f"'not_between' must be a list of exactly 2 numeric values, got: {bounds!r}")
        low = _parse_float_field(bounds[0], "not_between[0]")
        high = _parse_float_field(bounds[1], "not_between[1]")
        return NotBetweenValidator(low=low, high=high, tolerance=tolerance)
    # key == "equals"
    return EqualsValidator(value=_parse_float_field(raw["equals"], "equals"), tolerance=tolerance)


def _parse_table_check(raw: dict[str, Any]) -> TableCheck:
    """Parse a table-level check from a YAML dict.

    Args:
        raw: The raw YAML dict for a table check.

    Returns:
        TableCheck instance.

    Raises:
        ContractValidationError: If the check type is unknown or required fields are missing.
    """
    check_type = raw.get("type")
    if not check_type:
        raise ContractValidationError("Table check missing required field: 'type'")
    name = raw.get("name", "")
    severity = _parse_severity(raw.get("severity"))
    tags = _parse_tags(raw.get("tags"))
    validator = _parse_validator(raw)
    validators: tuple[Validator, ...] = (validator,) if validator is not None else ()

    if check_type == "num_rows":
        return NumRowsCheck(name=name, validators=validators, severity=severity, tags=tags)
    if check_type == "duplicates":
        columns_raw = raw.get("columns")
        if columns_raw is None:
            raise ContractValidationError("TableDuplicatesCheck missing required field: 'columns'")
        if isinstance(columns_raw, (str, bytes)):
            raise ContractValidationError(
                f"TableDuplicatesCheck 'columns' must be a list, got a string: {columns_raw!r}"
            )
        if not isinstance(columns_raw, list):
            raise ContractValidationError(
                f"TableDuplicatesCheck 'columns' must be a list, got: {type(columns_raw).__name__}"
            )
        return TableDuplicatesCheck(
            name=name,
            columns=tuple(columns_raw),
            validators=validators,
            return_type=raw.get("return", "count"),  # type: ignore[arg-type]
            severity=severity,
            tags=tags,
        )
    if check_type == "freshness":
        max_age_hours_raw = raw.get("max_age_hours")
        if max_age_hours_raw is None:
            raise ContractValidationError("FreshnessCheck missing required field: 'max_age_hours'")
        timestamp_column = raw.get("timestamp_column")
        if timestamp_column is None:
            raise ContractValidationError("FreshnessCheck missing required field: 'timestamp_column'")
        return FreshnessCheck(
            name=name,
            max_age_hours=_parse_float_field(max_age_hours_raw, "max_age_hours"),
            timestamp_column=timestamp_column,
            aggregation=raw.get("aggregation", "max"),  # type: ignore[arg-type]
            severity=severity,
            tags=tags,
        )
    if check_type == "completeness":
        partition_column = raw.get("partition_column")
        if partition_column is None:
            raise ContractValidationError("CompletenessCheck missing required field: 'partition_column'")
        granularity = raw.get("granularity")
        if granularity is None:
            raise ContractValidationError("CompletenessCheck missing required field: 'granularity'")
        return CompletenessCheck(
            name=name,
            partition_column=partition_column,
            granularity=granularity,  # type: ignore[arg-type]
            lookback_days=_parse_int_field(raw.get("lookback_days", 30), "lookback_days"),
            allow_future_gaps=_parse_bool(raw.get("allow_future_gaps"), "allow_future_gaps", default=True),
            max_gap_count=_parse_int_field(raw.get("max_gap_count", 0), "max_gap_count"),
            severity=severity,
            tags=tags,
        )
    raise ContractValidationError(f"Unknown table check type: '{check_type}'")


def _parse_column_check(raw: dict[str, Any]) -> ColumnCheck:
    """Parse a column-level check from a YAML dict.

    Args:
        raw: The raw YAML dict for a column check.

    Returns:
        ColumnCheck instance.

    Raises:
        ContractValidationError: If the check type is unknown or required fields are missing.
    """
    check_type = raw.get("type")
    if not check_type:
        raise ContractValidationError("Column check missing required field: 'type'")
    name = raw.get("name", "")
    severity = _parse_severity(raw.get("severity"))
    tags = _parse_tags(raw.get("tags"))
    validator = _parse_validator(raw)
    validators: tuple[Validator, ...] = (validator,) if validator is not None else ()
    return_type: str = raw.get("return", "count")

    if check_type == "missing":
        return MissingCheck(
            name=name,
            validators=validators,
            return_type=return_type,  # type: ignore[arg-type]
            severity=severity,
            tags=tags,
        )
    if check_type == "duplicates":
        return ColumnDuplicatesCheck(
            name=name,
            validators=validators,
            return_type=return_type,  # type: ignore[arg-type]
            severity=severity,
            tags=tags,
        )
    if check_type == "whitelist":
        values_raw = raw.get("values")
        if values_raw is None:
            raise ContractValidationError("WhitelistCheck missing required field: 'values'")
        if isinstance(values_raw, (str, bytes)):
            raise ContractValidationError(f"WhitelistCheck 'values' must be a list, got a string: {values_raw!r}")
        if not isinstance(values_raw, list):
            raise ContractValidationError(f"WhitelistCheck 'values' must be a list, got: {type(values_raw).__name__}")
        return WhitelistCheck(
            name=name,
            values=tuple(values_raw),
            validators=validators,
            return_type=return_type,  # type: ignore[arg-type]
            case_sensitive=_parse_bool(raw.get("case_sensitive"), "case_sensitive", default=True),
            severity=severity,
            tags=tags,
        )
    if check_type == "blacklist":
        values_raw = raw.get("values")
        if values_raw is None:
            raise ContractValidationError("BlacklistCheck missing required field: 'values'")
        if isinstance(values_raw, (str, bytes)):
            raise ContractValidationError(f"BlacklistCheck 'values' must be a list, got a string: {values_raw!r}")
        if not isinstance(values_raw, list):
            raise ContractValidationError(f"BlacklistCheck 'values' must be a list, got: {type(values_raw).__name__}")
        return BlacklistCheck(
            name=name,
            values=tuple(values_raw),
            validators=validators,
            return_type=return_type,  # type: ignore[arg-type]
            case_sensitive=_parse_bool(raw.get("case_sensitive"), "case_sensitive", default=True),
            severity=severity,
            tags=tags,
        )
    if check_type == "pattern":
        flags_raw = raw.get("flags", [])
        if isinstance(flags_raw, (str, bytes)):
            raise ContractValidationError(f"PatternCheck 'flags' must be a list, got a string: {flags_raw!r}")
        if not isinstance(flags_raw, list):
            raise ContractValidationError(f"PatternCheck 'flags' must be a list, got: {type(flags_raw).__name__}")
        return PatternCheck(
            name=name,
            validators=validators,
            pattern=raw.get("pattern"),
            format=raw.get("format"),  # type: ignore[arg-type]
            flags=tuple(flags_raw),
            return_type=return_type,  # type: ignore[arg-type]
            severity=severity,
            tags=tags,
        )
    if check_type == "min_length":
        return MinLengthCheck(
            name=name,
            validators=validators,
            return_type=return_type,  # type: ignore[arg-type]
            severity=severity,
            tags=tags,
        )
    if check_type == "max_length":
        return MaxLengthCheck(
            name=name,
            validators=validators,
            return_type=return_type,  # type: ignore[arg-type]
            severity=severity,
            tags=tags,
        )
    if check_type == "avg_length":
        return AvgLengthCheck(
            name=name,
            validators=validators,
            return_type=return_type,  # type: ignore[arg-type]
            severity=severity,
            tags=tags,
        )
    if check_type == "cardinality":
        return CardinalityCheck(name=name, validators=validators, severity=severity, tags=tags)
    if check_type == "min":
        return MinCheck(name=name, validators=validators, severity=severity, tags=tags)
    if check_type == "max":
        return MaxCheck(name=name, validators=validators, severity=severity, tags=tags)
    if check_type == "mean":
        return MeanCheck(name=name, validators=validators, severity=severity, tags=tags)
    if check_type == "sum":
        return SumCheck(name=name, validators=validators, severity=severity, tags=tags)
    if check_type == "count":
        return CountCheck(name=name, validators=validators, severity=severity, tags=tags)
    if check_type == "variance":
        return VarianceCheck(name=name, validators=validators, severity=severity, tags=tags)
    if check_type == "stddev":
        return StddevCheck(name=name, validators=validators, severity=severity, tags=tags)
    if check_type == "percentile":
        percentile_raw = raw.get("percentile")
        if percentile_raw is None:
            raise ContractValidationError("PercentileCheck missing required field: 'percentile'")
        return PercentileCheck(
            name=name,
            percentile=_parse_float_field(percentile_raw, "percentile"),
            validators=validators,
            severity=severity,
            tags=tags,
        )
    raise ContractValidationError(f"Unknown column check type: '{check_type}'")


def _parse_column(raw: Any) -> ColumnSpec:
    """Parse a column dict into a ColumnSpec.

    Args:
        raw: The raw YAML entry for a column. Must be a mapping.

    Returns:
        ColumnSpec instance.

    Raises:
        ContractValidationError: If raw is not a mapping or required fields are missing.
    """
    if not isinstance(raw, dict):
        raise ContractValidationError(f"Column entry must be a mapping, got: {type(raw).__name__}")
    for key in ("name", "type", "description"):
        if key not in raw:
            raise ContractValidationError(f"Column missing required field: '{key}'")

    nullable = _parse_bool(raw.get("nullable"), "nullable", default=True)

    # Parse column metadata
    raw_metadata = raw.get("metadata")
    if raw_metadata is None:
        raw_metadata = {}
    elif not isinstance(raw_metadata, dict):
        raise ContractValidationError("Column 'metadata' must be a mapping")
    col_metadata: tuple[tuple[str, str], ...] = tuple((str(k), str(v)) for k, v in raw_metadata.items())

    # Parse column checks
    raw_checks = raw.get("checks")
    if raw_checks is None:
        raw_checks = []
    elif not isinstance(raw_checks, list):
        raise ContractValidationError("Column 'checks' must be a list")
    col_checks: tuple[ColumnCheck, ...] = tuple(_parse_column_check(c) for c in raw_checks)  # type: ignore[misc]

    return ColumnSpec(
        name=_require_str(raw["name"], "Column.name"),
        type=_parse_type(raw["type"]),
        description=_require_str(raw["description"], "Column.description"),
        nullable=nullable,
        metadata=col_metadata,
        checks=col_checks,
    )


# ---------------------------------------------------------------------------
# Check translation helpers
# ---------------------------------------------------------------------------


def _apply_validators(
    metric: sp.Expr,
    ctx: Context,
    check_name: str,
    severity: SeverityLevel,
    tags: frozenset[str],
    validators: tuple[Validator, ...],
) -> None:
    """Apply all validators for a check as assertions in the context.

    Each validator becomes one named assertion within the same check node.
    ``NotBetweenValidator`` produces two assertions (lower and upper bounds).
    An empty validators tuple produces a single noop assertion.

    Args:
        metric: The already-computed symbolic metric expression to assert on.
        ctx: Context in which assertions are registered.
        check_name: Human-readable check name (used to generate assertion names).
        severity: Severity level propagated to every assertion.
        tags: Tag set propagated to every assertion.
        validators: Validators declared on the check. Empty means noop.
    """
    if not validators:
        ctx.assert_that(metric).config(name=check_name, severity=severity, tags=tags).noop()
        return

    for validator in validators:
        if isinstance(validator, MinValidator):
            assertion_name = f"{check_name} [min >= {validator.threshold}]"
            ctx.assert_that(metric).config(name=assertion_name, severity=severity, tags=tags).is_geq(
                validator.threshold, tol=validator.tolerance
            )
        elif isinstance(validator, MaxValidator):
            assertion_name = f"{check_name} [max <= {validator.threshold}]"
            ctx.assert_that(metric).config(name=assertion_name, severity=severity, tags=tags).is_leq(
                validator.threshold, tol=validator.tolerance
            )
        elif isinstance(validator, BetweenValidator):
            assertion_name = f"{check_name} [between {validator.low} and {validator.high}]"
            ctx.assert_that(metric).config(name=assertion_name, severity=severity, tags=tags).is_between(
                validator.low, validator.high, tol=validator.tolerance
            )
        elif isinstance(validator, NotBetweenValidator):
            assertion_name = f"{check_name} [not_between {validator.low} and {validator.high}]"
            ctx.assert_that(metric).config(name=assertion_name, severity=severity, tags=tags).is_not_between(
                validator.low, validator.high, tol=validator.tolerance
            )
        elif isinstance(validator, EqualsValidator):
            assertion_name = f"{check_name} [equals {validator.value}]"
            ctx.assert_that(metric).config(name=assertion_name, severity=severity, tags=tags).is_eq(
                validator.value, tol=validator.tolerance
            )


def _build_contract_check_fn(contract: Contract) -> Callable[[MetricProvider, Context], None]:
    """Build a single CheckProducer function covering all checks in a contract.

    The returned function, when called with ``(mp, ctx)``, iterates every
    table-level check and every column-level check on every column, calling
    each check's ``to_dqx`` method in order.  All assertions are emitted into
    the same ``@dqx_check`` node.

    Args:
        contract: The ``Contract`` whose checks are to be translated.

    Returns:
        A ``(MetricProvider, Context) -> None`` function ready to be wrapped
        by ``@dqx_check``.
    """

    def _check(mp: MetricProvider, ctx: Context) -> None:
        for table_check in contract.checks:
            table_check.to_dqx(mp, ctx)
        for col_spec in contract.columns:
            for col_check in col_spec.checks:
                col_check.to_dqx(col_spec.name, mp, ctx)

    _check.__name__ = contract.name
    return _check


@dataclass(frozen=True)
class Contract:
    """Top-level data contract definition.

    Args:
        name: Contract name (1-255 characters).
        version: Version string (non-empty).
        description: Human-readable description (non-empty).
        owner: Owning team or person (non-empty).
        dataset: Dataset identifier (non-empty, no whitespace).
        columns: Tuple of ColumnSpec instances (must be non-empty, no
            duplicate names).
        tags: Optional set of tags for categorization.
        sla: Optional SLA specification.
        partitioned_by: Column names used for partitioning. Each must
            reference a column in ``columns``.
        metadata: Key-value metadata pairs.
        checks: Table-level quality checks.

    Raises:
        ContractValidationError: For any validation failure.
    """

    name: str
    version: str
    description: str
    owner: str
    dataset: str
    columns: tuple[ColumnSpec, ...]
    tags: frozenset[str] = field(default_factory=frozenset)
    sla: SLASpec | None = None
    partitioned_by: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()
    checks: tuple[TableCheck, ...] = ()  # type: ignore[type-arg]

    def __post_init__(self) -> None:
        """Validate all contract fields."""
        if not self.name or len(self.name) > 255:
            raise ContractValidationError(f"Contract name must be between 1 and 255 characters, got {len(self.name)}")
        if not self.version:
            raise ContractValidationError("Contract version must be non-empty")
        if not self.description:
            raise ContractValidationError("Contract description must be non-empty")
        if not self.owner:
            raise ContractValidationError("Contract owner must be non-empty")
        if not self.dataset or any(c.isspace() for c in self.dataset):
            raise ContractValidationError(
                f"Contract dataset must be non-empty and contain no whitespace, got '{self.dataset}'"
            )
        if not self.columns:
            raise ContractValidationError("Contract columns must be non-empty")
        # Check for duplicate column names
        col_names = [col.name for col in self.columns]
        seen: set[str] = set()
        for col_name in col_names:
            if col_name in seen:
                raise ContractValidationError(f"Contract columns contain duplicate name: '{col_name}'")
            seen.add(col_name)
        # Validate partitioned_by columns exist
        col_name_set = frozenset(col_names)
        for part_col in self.partitioned_by:
            if part_col not in col_name_set:
                raise ContractValidationError(f"Contract partitioned_by column '{part_col}' not found in columns")
        # Validate table-level check column references
        for check in self.checks:
            if isinstance(check, TableDuplicatesCheck):
                for col in check.columns:
                    if col not in col_name_set:
                        raise ContractValidationError(
                            f"Contract TableDuplicatesCheck references unknown column '{col}'"
                        )
            elif isinstance(check, FreshnessCheck):
                if check.timestamp_column not in col_name_set:
                    raise ContractValidationError(
                        f"Contract FreshnessCheck references unknown column '{check.timestamp_column}'"
                    )
            elif isinstance(check, CompletenessCheck):
                if check.partition_column not in col_name_set:
                    raise ContractValidationError(
                        f"Contract CompletenessCheck references unknown column '{check.partition_column}'"
                    )
        # Non-partitioned SLA requires metadata.timestamp_column
        if self.sla is not None and not self.partitioned_by:
            metadata_dict = dict(self.metadata)
            ts_col = metadata_dict.get("timestamp_column")
            if ts_col is None:
                raise ContractValidationError(
                    "Contract: sla on a non-partitioned table requires metadata.timestamp_column"
                )
            if not ts_col:
                raise ContractValidationError("Contract: metadata.timestamp_column must be a non-empty string")
            if ts_col not in col_name_set:
                raise ContractValidationError(
                    f"Contract: metadata.timestamp_column '{ts_col}' does not reference a known column"
                )
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)

    @classmethod
    def from_yaml(cls, contract_yaml: Path) -> Contract:
        """Parse a YAML file and construct a Contract instance.

        Args:
            contract_yaml: Path to the YAML contract file.

        Returns:
            Contract: A fully validated Contract instance.

        Raises:
            ContractValidationError: If the file is not found, the YAML is
                invalid, required fields are missing, types are unknown, or
                any contract constraint is violated.
        """
        if not contract_yaml.exists():
            raise ContractValidationError(f"Contract file not found: {contract_yaml}")

        try:
            with contract_yaml.open() as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError as e:
            raise ContractValidationError(f"Invalid YAML in {contract_yaml}: {e}") from e

        if data is None:
            raise ContractValidationError(f"Contract file is empty: {contract_yaml}")

        if not isinstance(data, dict):
            raise ContractValidationError("Contract YAML must be a mapping")

        # Validate required root keys
        for key in ("name", "version", "description", "owner", "dataset", "columns"):
            if key not in data:
                raise ContractValidationError(f"Contract missing required field: '{key}'")

        # Validate columns is a list
        if not isinstance(data["columns"], list):
            raise ContractValidationError("Contract 'columns' must be a list")

        # Parse tags
        tags = _parse_tags(data.get("tags"))

        # Parse metadata block
        raw_metadata = data.get("metadata")
        if raw_metadata is None:
            raw_metadata = {}
        elif not isinstance(raw_metadata, dict):
            raise ContractValidationError("Contract 'metadata' must be a mapping")
        partitioned_by: tuple[str, ...] = ()
        if "partitioned_by" in raw_metadata:
            pb_raw = raw_metadata["partitioned_by"]
            if isinstance(pb_raw, (str, bytes)):
                raise ContractValidationError(f"Contract 'partitioned_by' must be a list, got a string: {pb_raw!r}")
            if not isinstance(pb_raw, list):
                raise ContractValidationError(f"Contract 'partitioned_by' must be a list, got: {type(pb_raw).__name__}")
            partitioned_by = tuple(str(c) for c in pb_raw)
        # All other k/v pairs (except partitioned_by) become contract metadata
        contract_metadata: tuple[tuple[str, str], ...] = tuple(
            (str(k), str(v)) for k, v in raw_metadata.items() if k != "partitioned_by"
        )

        # Parse optional SLA block
        sla: SLASpec | None = None
        if "sla" in data and data["sla"] is not None:
            raw_sla = data["sla"]
            if not isinstance(raw_sla, dict):
                raise ContractValidationError(f"Contract 'sla' must be a mapping, got: {type(raw_sla).__name__}")
            if "schedule" not in raw_sla:
                raise ContractValidationError("SLA block missing required field: 'schedule'")
            if "lag_hours" not in raw_sla:
                raise ContractValidationError("SLA block missing required field: 'lag_hours'")
            sla = SLASpec(schedule=raw_sla["schedule"], lag_hours=_parse_float_field(raw_sla["lag_hours"], "lag_hours"))

        # Parse optional top-level checks
        raw_checks = data.get("checks")
        if raw_checks is None:
            raw_checks = []
        elif not isinstance(raw_checks, list):
            raise ContractValidationError("Contract 'checks' must be a list")
        table_checks: tuple[TableCheck, ...] = tuple(_parse_table_check(c) for c in raw_checks)  # type: ignore[misc]

        # Parse columns
        columns: tuple[ColumnSpec, ...] = tuple(_parse_column(c) for c in data["columns"])

        return cls(
            name=_require_str(data["name"], "name"),
            version=_require_str(data["version"], "version"),
            description=_require_str(data["description"], "description"),
            owner=_require_str(data["owner"], "owner"),
            dataset=_require_str(data["dataset"], "dataset"),
            columns=columns,
            tags=tags,
            sla=sla,
            partitioned_by=partitioned_by,
            metadata=contract_metadata,
            checks=table_checks,
        )

    def to_checks(self) -> list[DecoratedCheck]:
        """Translate all contract checks into a single DecoratedCheck for VerificationSuite.

        All table-level and column-level checks defined in the contract are
        assembled into one check function and wrapped under a single
        ``@dqx_check`` node named after the contract.  Each check's
        ``to_dqx`` method is responsible for computing the appropriate metric
        and emitting assertions.

        Returns:
            list[DecoratedCheck]: A list containing exactly one ready-to-use
            check function that covers every check in the contract.
        """
        fn = _build_contract_check_fn(self)
        decorated = dqx_check(name=self.name, datasets=[self.dataset])(fn)
        return [decorated]
