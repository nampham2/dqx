"""Contract dataclasses for DQX data quality contracts.

This module defines the full type system and check specifications used to
declare data contracts: schema types, validators, table-level checks,
column-level checks, column specs, SLA specs, and the top-level Contract.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

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
    """
    if raw is None:
        return frozenset()
    return frozenset(str(t) for t in raw)


def _parse_severity(raw: Any) -> SeverityLevel:
    """Parse a YAML severity value, defaulting to 'P1'.

    Args:
        raw: The raw YAML value for severity.

    Returns:
        SeverityLevel string.
    """
    if raw is None:
        return "P1"
    return str(raw)  # type: ignore[return-value]


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
        return raw  # type: ignore[return-value]
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


def _parse_struct_field(raw: dict[str, Any]) -> StructField:
    """Parse a struct field dict into a StructField.

    Args:
        raw: Dict with 'name', 'type', and 'description' keys.

    Returns:
        StructField instance.

    Raises:
        ContractValidationError: If required fields are missing.
    """
    for key in ("name", "type", "description"):
        if key not in raw:
            raise ContractValidationError(f"StructField missing required field: '{key}'")
    return StructField(
        name=raw["name"],
        type=_parse_type(raw["type"]),
        description=raw["description"],
    )


def _parse_validator(raw: dict[str, Any]) -> Validator | None:
    """Parse a validator from a check YAML dict.

    At most one of min, max, between, not_between, equals may be present.

    Args:
        raw: The raw YAML dict for a check.

    Returns:
        Validator instance or None if no validator key found.

    Raises:
        ContractValidationError: If multiple validator keys are present.
    """
    tolerance = float(raw.get("tolerance", 1e-9))
    validator_keys = [k for k in ("min", "max", "between", "not_between", "equals") if k in raw]
    if len(validator_keys) > 1:
        raise ContractValidationError(f"Check specifies multiple validators: {validator_keys}")
    if not validator_keys:
        return None
    key = validator_keys[0]
    if key == "min":
        return MinValidator(threshold=float(raw["min"]), tolerance=tolerance)
    if key == "max":
        return MaxValidator(threshold=float(raw["max"]), tolerance=tolerance)
    if key == "between":
        low, high = raw["between"]
        return BetweenValidator(low=float(low), high=float(high), tolerance=tolerance)
    if key == "not_between":
        low, high = raw["not_between"]
        return NotBetweenValidator(low=float(low), high=float(high), tolerance=tolerance)
    # key == "equals"
    return EqualsValidator(value=float(raw["equals"]), tolerance=tolerance)


def _parse_table_check(raw: dict[str, Any]) -> TableCheck:
    """Parse a table-level check from a YAML dict.

    Args:
        raw: The raw YAML dict for a table check.

    Returns:
        TableCheck instance.

    Raises:
        ContractValidationError: If the check type is unknown.
    """
    check_type = raw["type"]
    name = raw.get("name", "")
    severity = _parse_severity(raw.get("severity"))
    tags = _parse_tags(raw.get("tags"))
    validator = _parse_validator(raw)
    validators: tuple[Validator, ...] = (validator,) if validator is not None else ()

    if check_type == "num_rows":
        return NumRowsCheck(name=name, validators=validators, severity=severity, tags=tags)
    if check_type == "duplicates":
        return TableDuplicatesCheck(
            name=name,
            columns=tuple(raw["columns"]),
            validators=validators,
            return_type=raw.get("return", "count"),  # type: ignore[arg-type]
            severity=severity,
            tags=tags,
        )
    if check_type == "freshness":
        return FreshnessCheck(
            name=name,
            max_age_hours=float(raw["max_age_hours"]),
            timestamp_column=raw["timestamp_column"],
            aggregation=raw.get("aggregation", "max"),  # type: ignore[arg-type]
            severity=severity,
            tags=tags,
        )
    if check_type == "completeness":
        return CompletenessCheck(
            name=name,
            partition_column=raw["partition_column"],
            granularity=raw["granularity"],  # type: ignore[arg-type]
            lookback_days=int(raw.get("lookback_days", 30)),
            allow_future_gaps=bool(raw.get("allow_future_gaps", True)),
            max_gap_count=int(raw.get("max_gap_count", 0)),
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
        ContractValidationError: If the check type is unknown.
    """
    check_type = raw["type"]
    name = raw.get("name", "")
    severity = _parse_severity(raw.get("severity"))
    tags = _parse_tags(raw.get("tags"))
    validator = _parse_validator(raw)
    validators: tuple[Validator, ...] = (validator,) if validator is not None else ()
    return_type: str = raw.get("return_type", "count")

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
        return WhitelistCheck(
            name=name,
            values=tuple(raw["values"]),
            validators=validators,
            return_type=return_type,  # type: ignore[arg-type]
            case_sensitive=bool(raw.get("case_sensitive", True)),
            severity=severity,
            tags=tags,
        )
    if check_type == "blacklist":
        return BlacklistCheck(
            name=name,
            values=tuple(raw["values"]),
            validators=validators,
            return_type=return_type,  # type: ignore[arg-type]
            case_sensitive=bool(raw.get("case_sensitive", True)),
            severity=severity,
            tags=tags,
        )
    if check_type == "pattern":
        return PatternCheck(
            name=name,
            validators=validators,
            pattern=raw.get("pattern"),
            format=raw.get("format"),  # type: ignore[arg-type]
            flags=tuple(raw.get("flags", [])),
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
        return PercentileCheck(
            name=name,
            percentile=float(raw["percentile"]),
            validators=validators,
            severity=severity,
            tags=tags,
        )
    raise ContractValidationError(f"Unknown column check type: '{check_type}'")


def _parse_column(raw: dict[str, Any]) -> ColumnSpec:
    """Parse a column dict into a ColumnSpec.

    Args:
        raw: The raw YAML dict for a column.

    Returns:
        ColumnSpec instance.

    Raises:
        ContractValidationError: If required fields are missing.
    """
    for key in ("name", "type", "description"):
        if key not in raw:
            raise ContractValidationError(f"Column missing required field: '{key}'")

    nullable = bool(raw.get("nullable", True))

    # Parse column metadata
    raw_metadata = raw.get("metadata") or {}
    col_metadata: tuple[tuple[str, str], ...] = tuple((str(k), str(v)) for k, v in raw_metadata.items())

    # Parse column checks
    raw_checks = raw.get("checks") or []
    col_checks: tuple[ColumnCheck, ...] = tuple(_parse_column_check(c) for c in raw_checks)  # type: ignore[misc]

    return ColumnSpec(
        name=raw["name"],
        type=_parse_type(raw["type"]),
        description=raw["description"],
        nullable=nullable,
        metadata=col_metadata,
        checks=col_checks,
    )


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
        raw_metadata = data.get("metadata") or {}
        partitioned_by: tuple[str, ...] = ()
        if "partitioned_by" in raw_metadata:
            partitioned_by = tuple(str(c) for c in raw_metadata["partitioned_by"])
        # All other k/v pairs (except partitioned_by) become contract metadata
        contract_metadata: tuple[tuple[str, str], ...] = tuple(
            (str(k), str(v)) for k, v in raw_metadata.items() if k != "partitioned_by"
        )

        # Parse optional SLA block
        sla: SLASpec | None = None
        if "sla" in data and data["sla"] is not None:
            raw_sla = data["sla"]
            sla = SLASpec(schedule=raw_sla["schedule"], lag_hours=float(raw_sla["lag_hours"]))

        # Parse optional top-level checks
        raw_checks = data.get("checks") or []
        table_checks: tuple[TableCheck, ...] = tuple(_parse_table_check(c) for c in raw_checks)  # type: ignore[misc]

        # Parse columns
        columns: tuple[ColumnSpec, ...] = tuple(_parse_column(c) for c in data["columns"])

        return cls(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            owner=data["owner"],
            dataset=data["dataset"],
            columns=columns,
            tags=tags,
            sla=sla,
            partitioned_by=partitioned_by,
            metadata=contract_metadata,
            checks=table_checks,
        )
