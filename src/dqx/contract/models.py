"""Contract dataclasses for DQX data quality contracts.

This module defines the full type system and check specifications used to
declare data contracts: schema types, validator specs, table-level checks,
column-level checks, column specs, SLA specs, and the top-level Contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from dqx.common import SeverityLevel, validate_tags

# ---------------------------------------------------------------------------
# Error classes
# ---------------------------------------------------------------------------


class ContractValidationError(Exception):
    """Raised when a contract field fails validation."""


class SchemaValidationError(Exception):
    """Raised when a schema field fails validation."""


# ---------------------------------------------------------------------------
# Type system
# ---------------------------------------------------------------------------

SIMPLE_TYPES = frozenset({"int", "float", "bool", "string", "bytes", "date", "time", "timestamp", "decimal"})

SimpleContractType = Literal["int", "float", "bool", "string", "bytes", "date", "time", "timestamp", "decimal"]


@dataclass(frozen=True)
class TimestampType:
    """A timestamp type with an optional timezone.

    Args:
        tz: IANA timezone string (e.g. "UTC", "America/New_York"). If None,
            the timestamp is timezone-naive.

    Raises:
        ContractValidationError: If tz is an empty string.
    """

    tz: str | None = None

    def __post_init__(self) -> None:
        """Validate that tz, if provided, is non-empty."""
        if self.tz is not None and self.tz == "":
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
# ValidatorSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatorSpec:
    """Specification for a numeric range/value validator.

    All fields are optional; an all-None instance is a valid **noop** — the
    check runs and records the computed metric but never fails on the value
    comparison.  Useful for informational or schema-only checks where you want
    to observe a metric without enforcing a bound.

    Args:
        min: Lower bound (inclusive).
        max: Upper bound (inclusive).
        between: Closed interval ``(lo, hi)`` — mutually exclusive with
            ``min``/``max``.
        not_between: Exclusion interval ``(lo, hi)`` — mutually exclusive
            with ``min``, ``max``, and ``between``.
        equals: Exact equality target — mutually exclusive with ``min``,
            ``max``, ``between``, and ``not_between``.
        tolerance: Floating-point comparison tolerance (must be >= 0).

    Raises:
        ContractValidationError: For any invalid combination or value.
    """

    min: float | None = None
    max: float | None = None
    between: tuple[float, float] | None = None
    not_between: tuple[float, float] | None = None
    equals: float | None = None
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        """Validate field combinations and individual constraints."""
        # between conflicts
        if self.between is not None and (self.min is not None or self.max is not None):
            raise ContractValidationError("ValidatorSpec: 'between' cannot be combined with 'min' or 'max'")
        # not_between conflicts
        if self.not_between is not None and (self.min is not None or self.max is not None or self.between is not None):
            raise ContractValidationError(
                "ValidatorSpec: 'not_between' cannot be combined with 'min', 'max', or 'between'"
            )
        # equals mutual exclusivity
        if self.equals is not None and any(v is not None for v in (self.min, self.max, self.between, self.not_between)):
            raise ContractValidationError(
                "ValidatorSpec: 'equals' cannot be combined with 'min', 'max', 'between', or 'not_between'"
            )
        # between range check
        if self.between is not None and self.between[0] > self.between[1]:
            raise ContractValidationError(
                f"ValidatorSpec: between lower bound {self.between[0]} > upper bound {self.between[1]}"
            )
        # not_between range check
        if self.not_between is not None and self.not_between[0] > self.not_between[1]:
            raise ContractValidationError(
                f"ValidatorSpec: not_between lower bound {self.not_between[0]} > upper bound {self.not_between[1]}"
            )
        # tolerance must be non-negative
        if self.tolerance < 0:
            raise ContractValidationError(f"ValidatorSpec: tolerance must be >= 0, got {self.tolerance}")


# ---------------------------------------------------------------------------
# ReturnType
# ---------------------------------------------------------------------------

ReturnType = Literal["count", "pct"]

# ---------------------------------------------------------------------------
# Table-level check dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumRowsCheck:
    """Check that the number of table rows satisfies a ValidatorSpec.

    Args:
        name: Check name (non-empty).
        validator: Numeric validator for row count.
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty.
    """

    name: str
    validator: ValidatorSpec
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name and normalize tags."""
        if not self.name:
            raise ContractValidationError("NumRowsCheck name must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class TableDuplicatesCheck:
    """Check for duplicate rows across specified columns.

    Args:
        name: Check name (non-empty).
        columns: Tuple of column names to check for duplicates (non-empty;
            each name must be non-empty).
        validator: Numeric validator for duplicate count/percentage.
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty, columns is empty, or any
            column name is empty.
    """

    name: str
    columns: tuple[str, ...]
    validator: ValidatorSpec
    return_type: ReturnType = "count"
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, columns, and normalize tags."""
        if not self.name:
            raise ContractValidationError("TableDuplicatesCheck name must be non-empty")
        if not self.columns:
            raise ContractValidationError("TableDuplicatesCheck columns must be non-empty")
        for col in self.columns:
            if not col:
                raise ContractValidationError("Each column name in TableDuplicatesCheck must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


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
        ContractValidationError: If name is empty, max_age_hours <= 0, or
            timestamp_column is empty.
    """

    name: str
    max_age_hours: float
    timestamp_column: str
    aggregation: Literal["max", "min"] = "max"
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, max_age_hours, and timestamp_column."""
        if not self.name:
            raise ContractValidationError("FreshnessCheck name must be non-empty")
        if self.max_age_hours <= 0:
            raise ContractValidationError(f"FreshnessCheck max_age_hours must be > 0, got {self.max_age_hours}")
        if not self.timestamp_column:
            raise ContractValidationError("FreshnessCheck timestamp_column must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


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
            lookback_days <= 0, or max_gap_count < 0.
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
        """Validate name, partition_column, lookback_days, and max_gap_count."""
        if not self.name:
            raise ContractValidationError("CompletenessCheck name must be non-empty")
        if not self.partition_column:
            raise ContractValidationError("CompletenessCheck partition_column must be non-empty")
        if self.lookback_days <= 0:
            raise ContractValidationError(f"CompletenessCheck lookback_days must be > 0, got {self.lookback_days}")
        if self.max_gap_count < 0:
            raise ContractValidationError(f"CompletenessCheck max_gap_count must be >= 0, got {self.max_gap_count}")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


TableCheck = NumRowsCheck | TableDuplicatesCheck | FreshnessCheck | CompletenessCheck

# ---------------------------------------------------------------------------
# Column-level check dataclasses
# ---------------------------------------------------------------------------

FormatShortcut = Literal["email", "phone", "uuid", "url", "ipv4", "ipv6", "date", "datetime"]


@dataclass(frozen=True)
class MissingCheck:
    """Check for missing (null) values in a column.

    Args:
        name: Check name (non-empty).
        validator: Numeric validator for missing count/percentage.
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty.
    """

    name: str
    validator: ValidatorSpec
    return_type: ReturnType = "count"
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name and normalize tags."""
        if not self.name:
            raise ContractValidationError("MissingCheck name must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class ColumnDuplicatesCheck:
    """Check for duplicate values within a single column.

    Args:
        name: Check name (non-empty).
        validator: Numeric validator for duplicate count/percentage.
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty.
    """

    name: str
    validator: ValidatorSpec
    return_type: ReturnType = "count"
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name and normalize tags."""
        if not self.name:
            raise ContractValidationError("ColumnDuplicatesCheck name must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class WhitelistCheck:
    """Check that column values are within an allowed set.

    Args:
        name: Check name (non-empty).
        values: Allowed values (non-empty tuple).
        validator: Numeric validator for out-of-whitelist count/percentage.
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        case_sensitive: Whether string comparisons are case-sensitive.
            Defaults to True.
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty or values is empty.
    """

    name: str
    values: tuple[str | int | float, ...]
    validator: ValidatorSpec
    return_type: ReturnType = "count"
    case_sensitive: bool = True
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name and values."""
        if not self.name:
            raise ContractValidationError("WhitelistCheck name must be non-empty")
        if not self.values:
            raise ContractValidationError("WhitelistCheck values must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class BlacklistCheck:
    """Check that column values are not in a forbidden set.

    Args:
        name: Check name (non-empty).
        values: Forbidden values (non-empty tuple).
        validator: Numeric validator for blacklisted count/percentage.
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        case_sensitive: Whether string comparisons are case-sensitive.
            Defaults to True.
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty or values is empty.
    """

    name: str
    values: tuple[str | int | float, ...]
    validator: ValidatorSpec
    return_type: ReturnType = "count"
    case_sensitive: bool = True
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name and values."""
        if not self.name:
            raise ContractValidationError("BlacklistCheck name must be non-empty")
        if not self.values:
            raise ContractValidationError("BlacklistCheck values must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class PatternCheck:
    """Check that column values match a regex pattern or format shortcut.

    Exactly one of ``pattern`` or ``format`` must be set. Flags cannot be
    combined with a format shortcut.

    Args:
        name: Check name (non-empty).
        validator: Numeric validator for pattern-violation count/percentage.
        pattern: Regular expression pattern string (non-empty when provided).
        format: Predefined format shortcut (e.g. "email", "uuid").
        flags: Regex flags (only valid with ``pattern``).
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty, neither/both of
            pattern/format are set, flags are used with format, or pattern
            is an empty string.
    """

    name: str
    validator: ValidatorSpec
    pattern: str | None = None
    format: FormatShortcut | None = None
    flags: tuple[str, ...] = ()
    return_type: ReturnType = "count"
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name, pattern/format exclusivity, and flags."""
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
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class MinLengthCheck:
    """Check that column string values meet a minimum length.

    Args:
        name: Check name (non-empty).
        validator: Numeric validator for minimum length.
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty.
    """

    name: str
    validator: ValidatorSpec
    return_type: ReturnType = "count"
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name and normalize tags."""
        if not self.name:
            raise ContractValidationError("MinLengthCheck name must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class MaxLengthCheck:
    """Check that column string values do not exceed a maximum length.

    Args:
        name: Check name (non-empty).
        validator: Numeric validator for maximum length.
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty.
    """

    name: str
    validator: ValidatorSpec
    return_type: ReturnType = "count"
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name and normalize tags."""
        if not self.name:
            raise ContractValidationError("MaxLengthCheck name must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class AvgLengthCheck:
    """Check that the average string length satisfies a ValidatorSpec.

    Args:
        name: Check name (non-empty).
        validator: Numeric validator for average string length.
        return_type: Whether to return a count or percentage. Defaults to
            "count".
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty.
    """

    name: str
    validator: ValidatorSpec
    return_type: ReturnType = "count"
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name and normalize tags."""
        if not self.name:
            raise ContractValidationError("AvgLengthCheck name must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class CardinalityCheck:
    """Check the number of distinct values in a column.

    Args:
        name: Check name (non-empty).
        validator: Numeric validator for cardinality.
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty.
    """

    name: str
    validator: ValidatorSpec
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name and normalize tags."""
        if not self.name:
            raise ContractValidationError("CardinalityCheck name must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class MinCheck:
    """Check the minimum value of a numeric column.

    Args:
        name: Check name (non-empty).
        validator: Numeric validator for minimum value.
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty.
    """

    name: str
    validator: ValidatorSpec
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name and normalize tags."""
        if not self.name:
            raise ContractValidationError("MinCheck name must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class MaxCheck:
    """Check the maximum value of a numeric column.

    Args:
        name: Check name (non-empty).
        validator: Numeric validator for maximum value.
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty.
    """

    name: str
    validator: ValidatorSpec
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name and normalize tags."""
        if not self.name:
            raise ContractValidationError("MaxCheck name must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class MeanCheck:
    """Check the mean value of a numeric column.

    Args:
        name: Check name (non-empty).
        validator: Numeric validator for mean value.
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty.
    """

    name: str
    validator: ValidatorSpec
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name and normalize tags."""
        if not self.name:
            raise ContractValidationError("MeanCheck name must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class SumCheck:
    """Check the sum of a numeric column.

    Args:
        name: Check name (non-empty).
        validator: Numeric validator for sum value.
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty.
    """

    name: str
    validator: ValidatorSpec
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name and normalize tags."""
        if not self.name:
            raise ContractValidationError("SumCheck name must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class CountCheck:
    """Check the count of (non-null) values in a column.

    Args:
        name: Check name (non-empty).
        validator: Numeric validator for count value.
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty.
    """

    name: str
    validator: ValidatorSpec
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name and normalize tags."""
        if not self.name:
            raise ContractValidationError("CountCheck name must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class VarianceCheck:
    """Check the variance of a numeric column.

    Args:
        name: Check name (non-empty).
        validator: Numeric validator for variance.
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty.
    """

    name: str
    validator: ValidatorSpec
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name and normalize tags."""
        if not self.name:
            raise ContractValidationError("VarianceCheck name must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class StddevCheck:
    """Check the standard deviation of a numeric column.

    Args:
        name: Check name (non-empty).
        validator: Numeric validator for standard deviation.
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty.
    """

    name: str
    validator: ValidatorSpec
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name and normalize tags."""
        if not self.name:
            raise ContractValidationError("StddevCheck name must be non-empty")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class PercentileCheck:
    """Check a specific percentile value of a numeric column.

    Args:
        name: Check name (non-empty).
        percentile: Percentile to compute, in [0.0, 100.0] inclusive.
        validator: Numeric validator for the percentile value.
        severity: Severity level. Defaults to "P1".
        tags: Optional set of tags for categorization.

    Raises:
        ContractValidationError: If name is empty or percentile is out of
            range.
    """

    name: str
    percentile: float
    validator: ValidatorSpec
    severity: SeverityLevel = "P1"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate name and percentile range."""
        if not self.name:
            raise ContractValidationError("PercentileCheck name must be non-empty")
        if not (0.0 <= self.percentile <= 100.0):
            raise ContractValidationError(f"PercentileCheck percentile must be in [0.0, 100.0], got {self.percentile}")
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)


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
        """Validate name and description."""
        if not self.name:
            raise ContractValidationError("ColumnSpec name must be non-empty")
        if not self.description:
            raise ContractValidationError("ColumnSpec description must be non-empty")


# ---------------------------------------------------------------------------
# SLASpec
# ---------------------------------------------------------------------------

_CRON_PATTERN = re.compile(r"^\S+ \S+ \S+ \S+ \S+$")


def _validate_cron(schedule: str) -> None:
    """Validate a 5-field cron expression.

    Args:
        schedule: Cron expression string to validate.

    Raises:
        ContractValidationError: If the expression is not a valid 5-field
            cron or uses unsupported list-based patterns.
    """
    if not _CRON_PATTERN.match(schedule):
        raise ContractValidationError(f"Invalid cron expression: '{schedule}' must be a 5-field cron")
    fields = schedule.split()
    day_of_month, day_of_week = fields[2], fields[4]
    if "," in day_of_month and day_of_month != "*":
        raise ContractValidationError(f"Unsupported cron: list-based day-of-month '{day_of_month}'")
    if "," in day_of_week and day_of_week != "*":
        raise ContractValidationError(f"Unsupported cron: list-based day-of-week '{day_of_week}'")


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
        # TODO: emit a warning when lag_hours > 168 on hourly/daily schedules
        # (per spec sla.md validation rules — currently not enforced)
        _validate_cron(self.schedule)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Contract:
    """Top-level data contract definition.

    Args:
        name: Contract name (1–255 characters).
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
        # Non-partitioned SLA requires metadata.timestamp_column
        if self.sla is not None and not self.partitioned_by:
            metadata_keys = {k for k, _ in self.metadata}
            if "timestamp_column" not in metadata_keys:
                raise ContractValidationError(
                    "Contract: sla on a non-partitioned table requires metadata.timestamp_column"
                )
        validated = validate_tags(self.tags)
        object.__setattr__(self, "tags", validated)
