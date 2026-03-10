"""Contract dataclasses for DQX data quality contracts.

This module defines the full type system and check specifications used to
declare data contracts: schema types, validators, table-level checks,
column-level checks, column specs, SLA specs, and the top-level Contract.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Literal, get_args

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
        if len(self.validators) > 1:
            raise ContractValidationError(
                f"NumRowsCheck validators must have at most 1 entry, got {len(self.validators)}"
            )
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
        if len(self.validators) > 1:
            raise ContractValidationError(
                f"TableDuplicatesCheck validators must have at most 1 entry, got {len(self.validators)}"
            )
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
        if len(self.validators) > 1:
            raise ContractValidationError(
                f"MissingCheck validators must have at most 1 entry, got {len(self.validators)}"
            )
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
        if len(self.validators) > 1:
            raise ContractValidationError(
                f"ColumnDuplicatesCheck validators must have at most 1 entry, got {len(self.validators)}"
            )
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
        if len(self.validators) > 1:
            raise ContractValidationError(
                f"WhitelistCheck validators must have at most 1 entry, got {len(self.validators)}"
            )
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
        if len(self.validators) > 1:
            raise ContractValidationError(
                f"BlacklistCheck validators must have at most 1 entry, got {len(self.validators)}"
            )
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
        else:
            for flag in self.flags:  # pragma: no cover
                if flag not in _VALID_FLAG_NAMES:
                    raise ContractValidationError(
                        f"PatternCheck: unknown flag '{flag}'; must be one of {sorted(_VALID_FLAG_NAMES)}"
                    )
        if len(self.validators) > 1:
            raise ContractValidationError(
                f"PatternCheck validators must have at most 1 entry, got {len(self.validators)}"
            )
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
        if len(self.validators) > 1:
            raise ContractValidationError(
                f"MinLengthCheck validators must have at most 1 entry, got {len(self.validators)}"
            )
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
        if len(self.validators) > 1:
            raise ContractValidationError(
                f"MaxLengthCheck validators must have at most 1 entry, got {len(self.validators)}"
            )
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
        if len(self.validators) > 1:
            raise ContractValidationError(
                f"AvgLengthCheck validators must have at most 1 entry, got {len(self.validators)}"
            )
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
        if len(self.validators) > 1:
            raise ContractValidationError(
                f"CardinalityCheck validators must have at most 1 entry, got {len(self.validators)}"
            )
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
        if len(self.validators) > 1:
            raise ContractValidationError(f"MinCheck validators must have at most 1 entry, got {len(self.validators)}")
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
        if len(self.validators) > 1:
            raise ContractValidationError(f"MaxCheck validators must have at most 1 entry, got {len(self.validators)}")
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
        if len(self.validators) > 1:
            raise ContractValidationError(f"MeanCheck validators must have at most 1 entry, got {len(self.validators)}")
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
        if len(self.validators) > 1:
            raise ContractValidationError(f"SumCheck validators must have at most 1 entry, got {len(self.validators)}")
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
        if len(self.validators) > 1:
            raise ContractValidationError(
                f"CountCheck validators must have at most 1 entry, got {len(self.validators)}"
            )
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
        if len(self.validators) > 1:
            raise ContractValidationError(
                f"VarianceCheck validators must have at most 1 entry, got {len(self.validators)}"
            )
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
        if len(self.validators) > 1:
            raise ContractValidationError(
                f"StddevCheck validators must have at most 1 entry, got {len(self.validators)}"
            )
        validated = _normalize_tags(self.tags)
        object.__setattr__(self, "tags", validated)


@dataclass(frozen=True)
class PercentileCheck:
    """Check a specific percentile value of a numeric column.

    Args:
        name: Check name (non-empty).
        percentile: Percentile to compute, in [0.0, 100.0] inclusive.
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
        if not (0.0 <= self.percentile <= 100.0):
            raise ContractValidationError(f"PercentileCheck percentile must be in [0.0, 100.0], got {self.percentile}")
        if len(self.validators) > 1:
            raise ContractValidationError(
                f"PercentileCheck validators must have at most 1 entry, got {len(self.validators)}"
            )
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
