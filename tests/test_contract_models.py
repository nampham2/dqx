"""Tests for contract dataclasses in dqx.contract.models."""

from __future__ import annotations

import dataclasses
from typing import cast

import pytest

from dqx.contract.models import (
    AvgLengthCheck,
    BlacklistCheck,
    CardinalityCheck,
    ColumnDuplicatesCheck,
    ColumnSpec,
    CompletenessCheck,
    Contract,
    ContractValidationError,
    ContractType,
    CountCheck,
    FreshnessCheck,
    ListType,
    MapType,
    MaxCheck,
    MaxLengthCheck,
    MeanCheck,
    MinCheck,
    MinLengthCheck,
    MissingCheck,
    NumRowsCheck,
    PatternCheck,
    PercentileCheck,
    SchemaValidationError,
    SLASpec,
    StddevCheck,
    StructField,
    StructType,
    SumCheck,
    TableDuplicatesCheck,
    TimestampType,
    ValidatorSpec,
    VarianceCheck,
    WhitelistCheck,
)


# ---------------------------------------------------------------------------
# TestContractErrors
# ---------------------------------------------------------------------------


class TestContractErrors:
    """Tests for custom exception classes."""

    def test_contract_validation_error_is_exception(self) -> None:
        """ContractValidationError should be an Exception."""
        err = ContractValidationError("bad contract")
        assert isinstance(err, Exception)
        assert str(err) == "bad contract"

    def test_schema_validation_error_is_exception(self) -> None:
        """SchemaValidationError should be an Exception."""
        err = SchemaValidationError("bad schema")
        assert isinstance(err, Exception)
        assert str(err) == "bad schema"

    def test_contract_validation_error_hierarchy(self) -> None:
        """ContractValidationError should not be a subclass of SchemaValidationError."""
        assert not issubclass(ContractValidationError, SchemaValidationError)

    def test_schema_validation_error_hierarchy(self) -> None:
        """SchemaValidationError should not be a subclass of ContractValidationError."""
        assert not issubclass(SchemaValidationError, ContractValidationError)


# ---------------------------------------------------------------------------
# TestValidatorSpec
# ---------------------------------------------------------------------------


class TestValidatorSpec:
    """Tests for ValidatorSpec dataclass."""

    def test_all_none_is_valid(self) -> None:
        """All-None ValidatorSpec is a valid noop validator — the check runs and records the metric but never fails."""
        v = ValidatorSpec()
        assert v.min is None
        assert v.max is None
        assert v.between is None
        assert v.not_between is None
        assert v.equals is None
        assert v.tolerance == pytest.approx(1e-9)

    def test_min_only(self) -> None:
        """ValidatorSpec with min only is valid."""
        v = ValidatorSpec(min=0.0)
        assert v.min == pytest.approx(0.0)

    def test_max_only(self) -> None:
        """ValidatorSpec with max only is valid."""
        v = ValidatorSpec(max=100.0)
        assert v.max == pytest.approx(100.0)

    def test_min_and_max(self) -> None:
        """ValidatorSpec with both min and max is valid."""
        v = ValidatorSpec(min=0.0, max=100.0)
        assert v.min == pytest.approx(0.0)
        assert v.max == pytest.approx(100.0)

    def test_min_greater_than_max_raises(self) -> None:
        """ValidatorSpec with min > max raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="min .* > max"):
            ValidatorSpec(min=100.0, max=1.0)

    def test_between_only(self) -> None:
        """ValidatorSpec with between only is valid."""
        v = ValidatorSpec(between=(0.0, 100.0))
        assert v.between == (pytest.approx(0.0), pytest.approx(100.0))

    def test_not_between_only(self) -> None:
        """ValidatorSpec with not_between only is valid."""
        v = ValidatorSpec(not_between=(10.0, 20.0))
        assert v.not_between == (pytest.approx(10.0), pytest.approx(20.0))

    def test_equals_only(self) -> None:
        """ValidatorSpec with equals only is valid."""
        v = ValidatorSpec(equals=42.0)
        assert v.equals == pytest.approx(42.0)

    def test_custom_tolerance(self) -> None:
        """ValidatorSpec accepts custom tolerance."""
        v = ValidatorSpec(tolerance=0.01)
        assert v.tolerance == pytest.approx(0.01)

    def test_zero_tolerance_is_valid(self) -> None:
        """ValidatorSpec with tolerance=0 is valid."""
        v = ValidatorSpec(tolerance=0.0)
        assert v.tolerance == pytest.approx(0.0)

    def test_between_with_min_raises(self) -> None:
        """between + min combo raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="between"):
            ValidatorSpec(between=(0.0, 10.0), min=0.0)

    def test_between_with_max_raises(self) -> None:
        """between + max combo raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="between"):
            ValidatorSpec(between=(0.0, 10.0), max=10.0)

    def test_between_with_min_and_max_raises(self) -> None:
        """between + min + max combo raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="between"):
            ValidatorSpec(between=(0.0, 10.0), min=0.0, max=10.0)

    def test_not_between_with_min_raises(self) -> None:
        """not_between + min combo raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="not_between"):
            ValidatorSpec(not_between=(0.0, 10.0), min=0.0)

    def test_not_between_with_max_raises(self) -> None:
        """not_between + max combo raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="not_between"):
            ValidatorSpec(not_between=(0.0, 10.0), max=10.0)

    def test_not_between_with_between_raises(self) -> None:
        """not_between + between combo raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="not_between"):
            ValidatorSpec(between=(0.0, 10.0), not_between=(20.0, 30.0))

    def test_not_between_with_min_max_and_between_raises(self) -> None:
        """Any conflicting combination raises ContractValidationError."""
        # between fires first (conflicts with min/max), then not_between would conflict too.
        # Either way a ContractValidationError is raised.
        with pytest.raises(ContractValidationError):
            ValidatorSpec(min=0.0, max=10.0, between=(0.0, 10.0), not_between=(20.0, 30.0))

    def test_between_inverted_range_raises(self) -> None:
        """between with low > high raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="between"):
            ValidatorSpec(between=(10.0, 0.0))

    def test_between_equal_bounds_is_valid(self) -> None:
        """between with equal bounds (lo == hi) is valid."""
        v = ValidatorSpec(between=(5.0, 5.0))
        assert v.between == (pytest.approx(5.0), pytest.approx(5.0))

    def test_not_between_inverted_range_raises(self) -> None:
        """not_between with low > high raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="not_between"):
            ValidatorSpec(not_between=(10.0, 0.0))

    def test_not_between_equal_bounds_is_valid(self) -> None:
        """not_between with equal bounds (lo == hi) is valid."""
        v = ValidatorSpec(not_between=(5.0, 5.0))
        assert v.not_between == (pytest.approx(5.0), pytest.approx(5.0))

    def test_negative_tolerance_raises(self) -> None:
        """Negative tolerance raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="tolerance"):
            ValidatorSpec(tolerance=-1e-9)

    def test_equals_with_min_raises(self) -> None:
        """equals + min combo raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="equals"):
            ValidatorSpec(equals=5.0, min=1.0)

    def test_equals_with_max_raises(self) -> None:
        """equals + max combo raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="equals"):
            ValidatorSpec(equals=5.0, max=10.0)

    def test_equals_with_between_raises(self) -> None:
        """equals + between combo raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="equals"):
            ValidatorSpec(equals=5.0, between=(1.0, 10.0))

    def test_equals_with_not_between_raises(self) -> None:
        """equals + not_between combo raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="equals"):
            ValidatorSpec(equals=5.0, not_between=(1.0, 10.0))

    def test_frozen(self) -> None:
        """ValidatorSpec is immutable (frozen dataclass)."""
        v = ValidatorSpec(min=0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            v.min = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestTimestampType
# ---------------------------------------------------------------------------


class TestTimestampType:
    """Tests for TimestampType dataclass."""

    def test_no_tz_is_timezone_naive(self) -> None:
        """TimestampType with no tz is timezone-naive (tz=None)."""
        t = TimestampType()
        assert t.tz is None

    def test_with_tz(self) -> None:
        """TimestampType with timezone string is valid."""
        t = TimestampType(tz="UTC")
        assert t.tz == "UTC"

    def test_with_complex_tz(self) -> None:
        """TimestampType with complex timezone is valid."""
        t = TimestampType(tz="America/New_York")
        assert t.tz == "America/New_York"

    def test_empty_tz_raises(self) -> None:
        """Empty timezone string raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="tz"):
            TimestampType(tz="")

    def test_frozen(self) -> None:
        """TimestampType is immutable."""
        t = TimestampType(tz="UTC")
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.tz = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestListType
# ---------------------------------------------------------------------------


class TestListType:
    """Tests for ListType dataclass."""

    def test_with_simple_type(self) -> None:
        """ListType wrapping a simple type string is valid."""
        lt = ListType(value_type="int")
        assert lt.value_type == "int"

    def test_with_timestamp_type(self) -> None:
        """ListType wrapping a TimestampType is valid."""
        lt = ListType(value_type=TimestampType(tz="UTC"))
        assert isinstance(lt.value_type, TimestampType)

    def test_nested_list_type(self) -> None:
        """ListType nesting another ListType is valid."""
        inner = ListType(value_type="string")
        outer = ListType(value_type=inner)
        assert isinstance(outer.value_type, ListType)

    def test_frozen(self) -> None:
        """ListType is immutable."""
        lt = ListType(value_type="int")
        with pytest.raises(dataclasses.FrozenInstanceError):
            lt.value_type = "string"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestStructField
# ---------------------------------------------------------------------------


class TestStructField:
    """Tests for StructField dataclass."""

    def test_valid_struct_field(self) -> None:
        """StructField with all required fields is valid."""
        sf = StructField(name="col1", type="int", description="An integer column")
        assert sf.name == "col1"
        assert sf.type == "int"
        assert sf.description == "An integer column"

    def test_nullable_kwarg_not_accepted(self) -> None:
        """StructField does not accept nullable keyword argument."""
        with pytest.raises(TypeError):
            StructField(name="x", type="int", description="d", nullable=True)  # type: ignore[call-arg]

    def test_empty_name_raises(self) -> None:
        """Empty name raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="name"):
            StructField(name="", type="int", description="desc")

    def test_empty_description_raises(self) -> None:
        """Empty description raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="description"):
            StructField(name="col1", type="int", description="")

    def test_frozen(self) -> None:
        """StructField is immutable."""
        sf = StructField(name="col1", type="int", description="desc")
        with pytest.raises(dataclasses.FrozenInstanceError):
            sf.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestStructType
# ---------------------------------------------------------------------------


class TestStructType:
    """Tests for StructType dataclass."""

    def test_valid_struct_type(self) -> None:
        """StructType with at least one field is valid."""
        sf = StructField(name="col1", type="int", description="An int")
        st = StructType(fields=(sf,))
        assert len(st.fields) == 1

    def test_multiple_fields(self) -> None:
        """StructType with multiple fields is valid."""
        sf1 = StructField(name="col1", type="int", description="A")
        sf2 = StructField(name="col2", type="string", description="B")
        st = StructType(fields=(sf1, sf2))
        assert len(st.fields) == 2

    def test_empty_fields_raises(self) -> None:
        """StructType with empty fields raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="fields"):
            StructType(fields=())

    def test_frozen(self) -> None:
        """StructType is immutable."""
        sf = StructField(name="col1", type="int", description="An int")
        st = StructType(fields=(sf,))
        with pytest.raises(dataclasses.FrozenInstanceError):
            st.fields = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestMapType
# ---------------------------------------------------------------------------


class TestMapType:
    """Tests for MapType dataclass."""

    def test_valid_map_type(self) -> None:
        """MapType with key and value types is valid."""
        mt = MapType(key_type="string", value_type="int")
        assert mt.key_type == "string"
        assert mt.value_type == "int"

    def test_complex_map_type(self) -> None:
        """MapType with complex nested types is valid."""
        mt = MapType(key_type="string", value_type=ListType(value_type="float"))
        assert isinstance(mt.value_type, ListType)

    def test_frozen(self) -> None:
        """MapType is immutable."""
        mt = MapType(key_type="string", value_type="int")
        with pytest.raises(dataclasses.FrozenInstanceError):
            mt.key_type = "int"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestNumRowsCheck
# ---------------------------------------------------------------------------


class TestNumRowsCheck:
    """Tests for NumRowsCheck dataclass."""

    def test_valid_num_rows_check(self) -> None:
        """NumRowsCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(min=1.0)
        check = NumRowsCheck(name="row_count", validator=v)
        assert check.name == "row_count"
        assert check.validator == v
        assert check.severity == "P1"
        assert check.tags == frozenset()

    def test_custom_severity_and_tags(self) -> None:
        """NumRowsCheck accepts custom severity and tags."""
        v = ValidatorSpec(min=1.0)
        check = NumRowsCheck(name="row_count", validator=v, severity="P0", tags=frozenset({"critical"}))
        assert check.severity == "P0"
        assert check.tags == frozenset({"critical"})

    def test_tags_as_set(self) -> None:
        """NumRowsCheck accepts a plain set for tags."""
        v = ValidatorSpec(min=1.0)
        # Pass a plain set — the __post_init__ should convert it via validate_tags
        tags_set: frozenset[str] = {"critical", "prod"}  # type: ignore[assignment]
        check = NumRowsCheck(name="row_count", validator=v, tags=tags_set)
        assert isinstance(check.tags, frozenset)
        assert check.tags == frozenset({"critical", "prod"})

    def test_empty_name_raises(self) -> None:
        """Empty name raises ContractValidationError."""
        v = ValidatorSpec(min=1.0)
        with pytest.raises(ContractValidationError, match="name"):
            NumRowsCheck(name="", validator=v)

    def test_invalid_tag_raises_contract_validation_error(self) -> None:
        """Invalid tag value raises ContractValidationError (not ValueError)."""
        v = ValidatorSpec(min=1.0)
        with pytest.raises(ContractValidationError):
            NumRowsCheck(name="check", validator=v, tags=frozenset({"invalid tag!"}))

    def test_frozen(self) -> None:
        """NumRowsCheck is immutable."""
        v = ValidatorSpec(min=1.0)
        check = NumRowsCheck(name="row_count", validator=v)
        with pytest.raises(dataclasses.FrozenInstanceError):
            check.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestTableDuplicatesCheck
# ---------------------------------------------------------------------------


class TestTableDuplicatesCheck:
    """Tests for TableDuplicatesCheck dataclass."""

    def test_valid_table_duplicates_check(self) -> None:
        """TableDuplicatesCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(equals=0.0)
        check = TableDuplicatesCheck(name="dedup_check", columns=("id", "date"), validator=v)
        assert check.name == "dedup_check"
        assert check.columns == ("id", "date")
        assert check.validator == v
        assert check.return_type == "count"

    def test_pct_return_type(self) -> None:
        """TableDuplicatesCheck accepts return_type='pct'."""
        v = ValidatorSpec(equals=0.0)
        check = TableDuplicatesCheck(name="dedup_check", columns=("id",), validator=v, return_type="pct")
        assert check.return_type == "pct"

    def test_empty_name_raises(self) -> None:
        """Empty name raises ContractValidationError."""
        v = ValidatorSpec(equals=0.0)
        with pytest.raises(ContractValidationError, match="name"):
            TableDuplicatesCheck(name="", columns=("id",), validator=v)

    def test_empty_columns_raises(self) -> None:
        """Empty columns tuple raises ContractValidationError."""
        v = ValidatorSpec(equals=0.0)
        with pytest.raises(ContractValidationError, match="columns"):
            TableDuplicatesCheck(name="dedup_check", columns=(), validator=v)

    def test_empty_column_name_raises(self) -> None:
        """Column with empty string name raises ContractValidationError."""
        v = ValidatorSpec(equals=0.0)
        with pytest.raises(ContractValidationError, match="column"):
            TableDuplicatesCheck(name="dedup_check", columns=("id", ""), validator=v)

    def test_frozen(self) -> None:
        """TableDuplicatesCheck is immutable."""
        v = ValidatorSpec(equals=0.0)
        check = TableDuplicatesCheck(name="dedup_check", columns=("id",), validator=v)
        with pytest.raises(dataclasses.FrozenInstanceError):
            check.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestFreshnessCheck
# ---------------------------------------------------------------------------


class TestFreshnessCheck:
    """Tests for FreshnessCheck dataclass."""

    def test_valid_freshness_check(self) -> None:
        """FreshnessCheck with valid fields is constructed correctly."""
        check = FreshnessCheck(name="freshness", max_age_hours=24.0, timestamp_column="updated_at")
        assert check.name == "freshness"
        assert check.max_age_hours == pytest.approx(24.0)
        assert check.timestamp_column == "updated_at"
        assert check.aggregation == "max"

    def test_min_aggregation(self) -> None:
        """FreshnessCheck accepts aggregation='min'."""
        check = FreshnessCheck(name="freshness", max_age_hours=1.0, timestamp_column="ts", aggregation="min")
        assert check.aggregation == "min"

    def test_empty_name_raises(self) -> None:
        """Empty name raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="name"):
            FreshnessCheck(name="", max_age_hours=24.0, timestamp_column="ts")

    def test_zero_max_age_hours_raises(self) -> None:
        """max_age_hours=0 raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="max_age_hours"):
            FreshnessCheck(name="freshness", max_age_hours=0.0, timestamp_column="ts")

    def test_negative_max_age_hours_raises(self) -> None:
        """Negative max_age_hours raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="max_age_hours"):
            FreshnessCheck(name="freshness", max_age_hours=-1.0, timestamp_column="ts")

    def test_empty_timestamp_column_raises(self) -> None:
        """Empty timestamp_column raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="timestamp_column"):
            FreshnessCheck(name="freshness", max_age_hours=24.0, timestamp_column="")

    def test_invalid_aggregation_raises(self) -> None:
        """Invalid aggregation value raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="aggregation"):
            FreshnessCheck(  # type: ignore[arg-type]
                name="freshness",
                max_age_hours=24.0,
                timestamp_column="ts",
                aggregation="latest",  # type: ignore[arg-type]
            )

    def test_frozen(self) -> None:
        """FreshnessCheck is immutable."""
        check = FreshnessCheck(name="freshness", max_age_hours=24.0, timestamp_column="ts")
        with pytest.raises(dataclasses.FrozenInstanceError):
            check.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestCompletenessCheck
# ---------------------------------------------------------------------------


class TestCompletenessCheck:
    """Tests for CompletenessCheck dataclass."""

    def test_valid_completeness_check(self) -> None:
        """CompletenessCheck with valid fields is constructed correctly."""
        check = CompletenessCheck(name="completeness", partition_column="date", granularity="daily")
        assert check.name == "completeness"
        assert check.partition_column == "date"
        assert check.granularity == "daily"
        assert check.lookback_days == 30
        assert check.allow_future_gaps is True
        assert check.max_gap_count == 0

    def test_all_granularities(self) -> None:
        """CompletenessCheck accepts all valid granularity values."""
        for gran in ("hourly", "daily", "weekly", "monthly"):
            check = CompletenessCheck(name="c", partition_column="date", granularity=gran)  # type: ignore[arg-type]
            assert check.granularity == gran

    def test_custom_lookback_days(self) -> None:
        """CompletenessCheck accepts custom lookback_days."""
        check = CompletenessCheck(name="c", partition_column="date", granularity="daily", lookback_days=7)
        assert check.lookback_days == 7

    def test_custom_max_gap_count(self) -> None:
        """CompletenessCheck accepts custom max_gap_count."""
        check = CompletenessCheck(name="c", partition_column="date", granularity="daily", max_gap_count=2)
        assert check.max_gap_count == 2

    def test_zero_max_gap_count_is_valid(self) -> None:
        """CompletenessCheck with max_gap_count=0 is valid."""
        check = CompletenessCheck(name="c", partition_column="date", granularity="daily", max_gap_count=0)
        assert check.max_gap_count == 0

    def test_empty_name_raises(self) -> None:
        """Empty name raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="name"):
            CompletenessCheck(name="", partition_column="date", granularity="daily")

    def test_empty_partition_column_raises(self) -> None:
        """Empty partition_column raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="partition_column"):
            CompletenessCheck(name="c", partition_column="", granularity="daily")

    def test_zero_lookback_days_raises(self) -> None:
        """lookback_days=0 raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="lookback_days"):
            CompletenessCheck(name="c", partition_column="date", granularity="daily", lookback_days=0)

    def test_negative_lookback_days_raises(self) -> None:
        """Negative lookback_days raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="lookback_days"):
            CompletenessCheck(name="c", partition_column="date", granularity="daily", lookback_days=-1)

    def test_negative_max_gap_count_raises(self) -> None:
        """Negative max_gap_count raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="max_gap_count"):
            CompletenessCheck(name="c", partition_column="date", granularity="daily", max_gap_count=-1)

    def test_invalid_granularity_raises(self) -> None:
        """Invalid granularity value raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="granularity"):
            CompletenessCheck(  # type: ignore[arg-type]
                name="c",
                partition_column="date",
                granularity="quarterly",  # type: ignore[arg-type]
            )

    def test_frozen(self) -> None:
        """CompletenessCheck is immutable."""
        check = CompletenessCheck(name="c", partition_column="date", granularity="daily")
        with pytest.raises(dataclasses.FrozenInstanceError):
            check.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestMissingCheck
# ---------------------------------------------------------------------------


class TestMissingCheck:
    """Tests for MissingCheck dataclass."""

    def test_valid_missing_check(self) -> None:
        """MissingCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(equals=0.0)
        check = MissingCheck(name="missing", validator=v)
        assert check.name == "missing"
        assert check.return_type == "count"

    def test_pct_return_type(self) -> None:
        """MissingCheck accepts return_type='pct'."""
        v = ValidatorSpec(equals=0.0)
        check = MissingCheck(name="missing", validator=v, return_type="pct")
        assert check.return_type == "pct"

    def test_empty_name_raises(self) -> None:
        """Empty name raises ContractValidationError."""
        v = ValidatorSpec(equals=0.0)
        with pytest.raises(ContractValidationError, match="name"):
            MissingCheck(name="", validator=v)

    def test_frozen(self) -> None:
        """MissingCheck is immutable."""
        v = ValidatorSpec(equals=0.0)
        check = MissingCheck(name="missing", validator=v)
        with pytest.raises(dataclasses.FrozenInstanceError):
            check.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestColumnDuplicatesCheck
# ---------------------------------------------------------------------------


class TestColumnDuplicatesCheck:
    """Tests for ColumnDuplicatesCheck dataclass."""

    def test_valid_column_duplicates_check(self) -> None:
        """ColumnDuplicatesCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(equals=0.0)
        check = ColumnDuplicatesCheck(name="col_dedup", validator=v)
        assert check.name == "col_dedup"
        assert check.return_type == "count"

    def test_empty_name_raises(self) -> None:
        """Empty name raises ContractValidationError."""
        v = ValidatorSpec(equals=0.0)
        with pytest.raises(ContractValidationError, match="name"):
            ColumnDuplicatesCheck(name="", validator=v)

    def test_frozen(self) -> None:
        """ColumnDuplicatesCheck is immutable."""
        v = ValidatorSpec(equals=0.0)
        check = ColumnDuplicatesCheck(name="col_dedup", validator=v)
        with pytest.raises(dataclasses.FrozenInstanceError):
            check.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestWhitelistCheck
# ---------------------------------------------------------------------------


class TestWhitelistCheck:
    """Tests for WhitelistCheck dataclass."""

    def test_valid_whitelist_check(self) -> None:
        """WhitelistCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(equals=0.0)
        check = WhitelistCheck(name="whitelist", values=("active", "inactive"), validator=v)
        assert check.name == "whitelist"
        assert check.values == ("active", "inactive")
        assert check.case_sensitive is True

    def test_numeric_values(self) -> None:
        """WhitelistCheck accepts numeric values."""
        v = ValidatorSpec(equals=0.0)
        check = WhitelistCheck(name="whitelist", values=(1, 2, 3), validator=v)
        assert check.values == (1, 2, 3)

    def test_mixed_values(self) -> None:
        """WhitelistCheck accepts mixed type values."""
        v = ValidatorSpec(equals=0.0)
        check = WhitelistCheck(name="whitelist", values=("a", 1, 2.5), validator=v)
        assert check.values == ("a", 1, 2.5)

    def test_case_insensitive(self) -> None:
        """WhitelistCheck accepts case_sensitive=False."""
        v = ValidatorSpec(equals=0.0)
        check = WhitelistCheck(name="whitelist", values=("Active",), validator=v, case_sensitive=False)
        assert check.case_sensitive is False

    def test_empty_name_raises(self) -> None:
        """Empty name raises ContractValidationError."""
        v = ValidatorSpec(equals=0.0)
        with pytest.raises(ContractValidationError, match="name"):
            WhitelistCheck(name="", values=("a",), validator=v)

    def test_empty_values_raises(self) -> None:
        """Empty values tuple raises ContractValidationError."""
        v = ValidatorSpec(equals=0.0)
        with pytest.raises(ContractValidationError, match="values"):
            WhitelistCheck(name="whitelist", values=(), validator=v)

    def test_frozen(self) -> None:
        """WhitelistCheck is immutable."""
        v = ValidatorSpec(equals=0.0)
        check = WhitelistCheck(name="whitelist", values=("a",), validator=v)
        with pytest.raises(dataclasses.FrozenInstanceError):
            check.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestBlacklistCheck
# ---------------------------------------------------------------------------


class TestBlacklistCheck:
    """Tests for BlacklistCheck dataclass."""

    def test_valid_blacklist_check(self) -> None:
        """BlacklistCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(equals=0.0)
        check = BlacklistCheck(name="blacklist", values=("deleted", "banned"), validator=v)
        assert check.name == "blacklist"
        assert check.values == ("deleted", "banned")
        assert check.case_sensitive is True

    def test_empty_name_raises(self) -> None:
        """Empty name raises ContractValidationError."""
        v = ValidatorSpec(equals=0.0)
        with pytest.raises(ContractValidationError, match="name"):
            BlacklistCheck(name="", values=("a",), validator=v)

    def test_empty_values_raises(self) -> None:
        """Empty values tuple raises ContractValidationError."""
        v = ValidatorSpec(equals=0.0)
        with pytest.raises(ContractValidationError, match="values"):
            BlacklistCheck(name="blacklist", values=(), validator=v)

    def test_frozen(self) -> None:
        """BlacklistCheck is immutable."""
        v = ValidatorSpec(equals=0.0)
        check = BlacklistCheck(name="blacklist", values=("a",), validator=v)
        with pytest.raises(dataclasses.FrozenInstanceError):
            check.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestPatternCheck
# ---------------------------------------------------------------------------


class TestPatternCheck:
    """Tests for PatternCheck dataclass."""

    def test_valid_pattern_check_with_pattern(self) -> None:
        """PatternCheck with explicit pattern is valid."""
        v = ValidatorSpec(equals=0.0)
        check = PatternCheck(name="pattern_check", validator=v, pattern=r"\d+")
        assert check.pattern == r"\d+"
        assert check.format is None

    def test_valid_pattern_check_with_format(self) -> None:
        """PatternCheck with format shortcut is valid."""
        v = ValidatorSpec(equals=0.0)
        check = PatternCheck(name="email_check", validator=v, format="email")
        assert check.format == "email"
        assert check.pattern is None

    def test_all_format_shortcuts(self) -> None:
        """PatternCheck accepts all valid format shortcuts."""
        v = ValidatorSpec(equals=0.0)
        for fmt in ("email", "phone", "uuid", "url", "ipv4", "ipv6", "date", "datetime"):
            check = PatternCheck(name="check", validator=v, format=fmt)  # type: ignore[arg-type]
            assert check.format == fmt

    def test_flags_with_pattern(self) -> None:
        """PatternCheck with flags and pattern is valid."""
        v = ValidatorSpec(equals=0.0)
        check = PatternCheck(name="pattern_check", validator=v, pattern=r"\d+", flags=("IGNORECASE",))
        assert check.flags == ("IGNORECASE",)

    def test_neither_pattern_nor_format_raises(self) -> None:
        """Neither pattern nor format raises ContractValidationError."""
        v = ValidatorSpec(equals=0.0)
        with pytest.raises(ContractValidationError, match="pattern"):
            PatternCheck(name="pattern_check", validator=v)

    def test_both_pattern_and_format_raises(self) -> None:
        """Both pattern and format raises ContractValidationError."""
        v = ValidatorSpec(equals=0.0)
        with pytest.raises(ContractValidationError, match="pattern"):
            PatternCheck(name="pattern_check", validator=v, pattern=r"\d+", format="email")

    def test_flags_with_format_raises(self) -> None:
        """Flags with format shortcut raises ContractValidationError."""
        v = ValidatorSpec(equals=0.0)
        with pytest.raises(ContractValidationError, match="flags"):
            PatternCheck(name="email_check", validator=v, format="email", flags=("IGNORECASE",))

    def test_empty_pattern_raises(self) -> None:
        """Empty pattern string raises ContractValidationError."""
        v = ValidatorSpec(equals=0.0)
        with pytest.raises(ContractValidationError, match="pattern"):
            PatternCheck(name="pattern_check", validator=v, pattern="")

    def test_empty_name_raises(self) -> None:
        """Empty name raises ContractValidationError."""
        v = ValidatorSpec(equals=0.0)
        with pytest.raises(ContractValidationError, match="name"):
            PatternCheck(name="", validator=v, pattern=r"\d+")

    def test_frozen(self) -> None:
        """PatternCheck is immutable."""
        v = ValidatorSpec(equals=0.0)
        check = PatternCheck(name="pattern_check", validator=v, pattern=r"\d+")
        with pytest.raises(dataclasses.FrozenInstanceError):
            check.name = "other"  # type: ignore[misc]

    def test_invalid_regex_pattern_raises(self) -> None:
        """Invalid regex pattern raises ContractValidationError."""
        v = ValidatorSpec(equals=0.0)
        with pytest.raises(ContractValidationError, match="invalid regex"):
            PatternCheck(name="check", validator=v, pattern="[")

    def test_invalid_flag_name_raises(self) -> None:
        """Unknown flag name raises ContractValidationError."""
        v = ValidatorSpec(equals=0.0)
        with pytest.raises(ContractValidationError, match="unknown flag"):
            PatternCheck(name="check", validator=v, pattern=r"\d+", flags=("BADFLG",))

    def test_invalid_format_shortcut_raises(self) -> None:
        """Unknown format shortcut raises ContractValidationError."""
        v = ValidatorSpec(equals=0.0)
        with pytest.raises(ContractValidationError, match="unknown format"):
            PatternCheck(name="check", validator=v, format="ssn")  # type: ignore[arg-type]

    def test_all_valid_flag_names(self) -> None:
        """All recognized flag names are accepted."""
        v = ValidatorSpec(equals=0.0)
        for flag in ("IGNORECASE", "MULTILINE", "DOTALL", "VERBOSE", "ASCII", "UNICODE", "LOCALE"):
            check = PatternCheck(name="check", validator=v, pattern=r"\w+", flags=(flag,))
            assert flag in check.flags


# ---------------------------------------------------------------------------
# TestMinLengthCheck
# ---------------------------------------------------------------------------


class TestMinLengthCheck:
    """Tests for MinLengthCheck dataclass."""

    def test_valid_min_length_check(self) -> None:
        """MinLengthCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(min=1.0)
        check = MinLengthCheck(name="min_len", validator=v)
        assert check.name == "min_len"

    def test_empty_name_raises(self) -> None:
        """Empty name raises ContractValidationError."""
        v = ValidatorSpec(min=1.0)
        with pytest.raises(ContractValidationError, match="name"):
            MinLengthCheck(name="", validator=v)

    def test_frozen(self) -> None:
        """MinLengthCheck is immutable."""
        v = ValidatorSpec(min=1.0)
        check = MinLengthCheck(name="min_len", validator=v)
        with pytest.raises(dataclasses.FrozenInstanceError):
            check.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestMaxLengthCheck
# ---------------------------------------------------------------------------


class TestMaxLengthCheck:
    """Tests for MaxLengthCheck dataclass."""

    def test_valid_max_length_check(self) -> None:
        """MaxLengthCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(max=255.0)
        check = MaxLengthCheck(name="max_len", validator=v)
        assert check.name == "max_len"

    def test_empty_name_raises(self) -> None:
        """Empty name raises ContractValidationError."""
        v = ValidatorSpec(max=255.0)
        with pytest.raises(ContractValidationError, match="name"):
            MaxLengthCheck(name="", validator=v)

    def test_frozen(self) -> None:
        """MaxLengthCheck is immutable."""
        v = ValidatorSpec(max=255.0)
        check = MaxLengthCheck(name="max_len", validator=v)
        with pytest.raises(dataclasses.FrozenInstanceError):
            check.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestAvgLengthCheck
# ---------------------------------------------------------------------------


class TestAvgLengthCheck:
    """Tests for AvgLengthCheck dataclass."""

    def test_valid_avg_length_check(self) -> None:
        """AvgLengthCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(between=(5.0, 50.0))
        check = AvgLengthCheck(name="avg_len", validator=v)
        assert check.name == "avg_len"

    def test_default_return_type(self) -> None:
        """AvgLengthCheck default return_type is 'count'."""
        v = ValidatorSpec(between=(5.0, 50.0))
        check = AvgLengthCheck(name="avg_len", validator=v)
        assert check.return_type == "count"

    def test_pct_return_type(self) -> None:
        """AvgLengthCheck accepts return_type='pct'."""
        v = ValidatorSpec(between=(5.0, 50.0))
        check = AvgLengthCheck(name="avg_len", validator=v, return_type="pct")
        assert check.return_type == "pct"

    def test_empty_name_raises(self) -> None:
        """Empty name raises ContractValidationError."""
        v = ValidatorSpec(between=(5.0, 50.0))
        with pytest.raises(ContractValidationError, match="name"):
            AvgLengthCheck(name="", validator=v)

    def test_frozen(self) -> None:
        """AvgLengthCheck is immutable."""
        v = ValidatorSpec(between=(5.0, 50.0))
        check = AvgLengthCheck(name="avg_len", validator=v)
        with pytest.raises(dataclasses.FrozenInstanceError):
            check.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestCardinalityCheck
# ---------------------------------------------------------------------------


class TestCardinalityCheck:
    """Tests for CardinalityCheck dataclass."""

    def test_valid_cardinality_check(self) -> None:
        """CardinalityCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(max=1000.0)
        check = CardinalityCheck(name="cardinality", validator=v)
        assert check.name == "cardinality"

    def test_empty_name_raises(self) -> None:
        """Empty name raises ContractValidationError."""
        v = ValidatorSpec(max=1000.0)
        with pytest.raises(ContractValidationError, match="name"):
            CardinalityCheck(name="", validator=v)

    def test_frozen(self) -> None:
        """CardinalityCheck is immutable."""
        v = ValidatorSpec(max=1000.0)
        check = CardinalityCheck(name="cardinality", validator=v)
        with pytest.raises(dataclasses.FrozenInstanceError):
            check.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestStatisticalChecks  (min, max, mean, sum, count, variance, stddev)
# ---------------------------------------------------------------------------


class TestStatisticalChecks:
    """Tests for statistical check dataclasses (Min, Max, Mean, Sum, Count, Variance, Stddev)."""

    def test_valid_min_check(self) -> None:
        """MinCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(min=0.0)
        check = MinCheck(name="min_check", validator=v)
        assert check.name == "min_check"

    def test_min_check_empty_name_raises(self) -> None:
        """MinCheck empty name raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="name"):
            MinCheck(name="", validator=ValidatorSpec())

    def test_valid_max_check(self) -> None:
        """MaxCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(max=100.0)
        check = MaxCheck(name="max_check", validator=v)
        assert check.name == "max_check"

    def test_max_check_empty_name_raises(self) -> None:
        """MaxCheck empty name raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="name"):
            MaxCheck(name="", validator=ValidatorSpec())

    def test_valid_mean_check(self) -> None:
        """MeanCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(between=(10.0, 90.0))
        check = MeanCheck(name="mean_check", validator=v)
        assert check.name == "mean_check"

    def test_mean_check_empty_name_raises(self) -> None:
        """MeanCheck empty name raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="name"):
            MeanCheck(name="", validator=ValidatorSpec())

    def test_valid_sum_check(self) -> None:
        """SumCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(min=0.0)
        check = SumCheck(name="sum_check", validator=v)
        assert check.name == "sum_check"

    def test_sum_check_empty_name_raises(self) -> None:
        """SumCheck empty name raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="name"):
            SumCheck(name="", validator=ValidatorSpec())

    def test_valid_count_check(self) -> None:
        """CountCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(min=1.0)
        check = CountCheck(name="count_check", validator=v)
        assert check.name == "count_check"

    def test_count_check_empty_name_raises(self) -> None:
        """CountCheck empty name raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="name"):
            CountCheck(name="", validator=ValidatorSpec())

    def test_valid_variance_check(self) -> None:
        """VarianceCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(max=100.0)
        check = VarianceCheck(name="variance_check", validator=v)
        assert check.name == "variance_check"

    def test_variance_check_empty_name_raises(self) -> None:
        """VarianceCheck empty name raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="name"):
            VarianceCheck(name="", validator=ValidatorSpec())

    def test_valid_stddev_check(self) -> None:
        """StddevCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(max=10.0)
        check = StddevCheck(name="stddev_check", validator=v)
        assert check.name == "stddev_check"

    def test_stddev_check_empty_name_raises(self) -> None:
        """StddevCheck empty name raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="name"):
            StddevCheck(name="", validator=ValidatorSpec())

    def test_statistical_checks_frozen(self) -> None:
        """Statistical checks are all immutable."""
        v = ValidatorSpec()
        checks: list[MinCheck | MaxCheck | MeanCheck | SumCheck | CountCheck | VarianceCheck | StddevCheck] = [
            MinCheck(name="x", validator=v),
            MaxCheck(name="x", validator=v),
            MeanCheck(name="x", validator=v),
            SumCheck(name="x", validator=v),
            CountCheck(name="x", validator=v),
            VarianceCheck(name="x", validator=v),
            StddevCheck(name="x", validator=v),
        ]
        for check in checks:
            with pytest.raises(dataclasses.FrozenInstanceError):
                check.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestPercentileCheck
# ---------------------------------------------------------------------------


class TestPercentileCheck:
    """Tests for PercentileCheck dataclass."""

    def test_valid_percentile_check(self) -> None:
        """PercentileCheck with valid fields is constructed correctly."""
        v = ValidatorSpec(max=100.0)
        check = PercentileCheck(name="p50", percentile=50.0, validator=v)
        assert check.name == "p50"
        assert check.percentile == pytest.approx(50.0)

    def test_percentile_zero(self) -> None:
        """PercentileCheck with percentile=0.0 is valid."""
        v = ValidatorSpec()
        check = PercentileCheck(name="p0", percentile=0.0, validator=v)
        assert check.percentile == pytest.approx(0.0)

    def test_percentile_100(self) -> None:
        """PercentileCheck with percentile=100.0 is valid."""
        v = ValidatorSpec()
        check = PercentileCheck(name="p100", percentile=100.0, validator=v)
        assert check.percentile == pytest.approx(100.0)

    def test_empty_name_raises(self) -> None:
        """Empty name raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="name"):
            PercentileCheck(name="", percentile=50.0, validator=ValidatorSpec())

    def test_percentile_below_zero_raises(self) -> None:
        """percentile < 0.0 raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="percentile"):
            PercentileCheck(name="p", percentile=-0.1, validator=ValidatorSpec())

    def test_percentile_above_100_raises(self) -> None:
        """percentile > 100.0 raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="percentile"):
            PercentileCheck(name="p", percentile=100.1, validator=ValidatorSpec())

    def test_frozen(self) -> None:
        """PercentileCheck is immutable."""
        v = ValidatorSpec()
        check = PercentileCheck(name="p50", percentile=50.0, validator=v)
        with pytest.raises(dataclasses.FrozenInstanceError):
            check.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestColumnSpec
# ---------------------------------------------------------------------------


class TestColumnSpec:
    """Tests for ColumnSpec dataclass."""

    def test_valid_column_spec(self) -> None:
        """ColumnSpec with all required fields is constructed correctly."""
        col = ColumnSpec(name="user_id", type="int", description="The user's ID")
        assert col.name == "user_id"
        assert col.type == "int"
        assert col.description == "The user's ID"
        assert col.nullable is True
        assert col.metadata == ()
        assert col.checks == ()

    def test_not_nullable(self) -> None:
        """ColumnSpec with nullable=False is valid."""
        col = ColumnSpec(name="id", type="string", description="PK", nullable=False)
        assert col.nullable is False

    def test_with_metadata(self) -> None:
        """ColumnSpec with metadata tuple is valid."""
        col = ColumnSpec(name="col", type="string", description="desc", metadata=(("key", "value"),))
        assert col.metadata == (("key", "value"),)

    def test_with_checks(self) -> None:
        """ColumnSpec with checks is valid."""
        v = ValidatorSpec(equals=0.0)
        check = MissingCheck(name="missing", validator=v)
        col = ColumnSpec(name="col", type="string", description="desc", checks=(check,))
        assert len(col.checks) == 1

    def test_empty_name_raises(self) -> None:
        """Empty name raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="name"):
            ColumnSpec(name="", type="int", description="An int")

    def test_empty_description_raises(self) -> None:
        """Empty description raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="description"):
            ColumnSpec(name="col1", type="int", description="")

    def test_invalid_simple_type_raises(self) -> None:
        """Invalid simple type string raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="not a valid simple type"):
            ColumnSpec(name="x", type="strng", description="d")  # type: ignore[arg-type]

    def test_with_timestamp_type(self) -> None:
        """ColumnSpec with TimestampType is valid."""
        col = ColumnSpec(name="ts", type=TimestampType(tz="UTC"), description="A timestamp")
        assert isinstance(col.type, TimestampType)

    def test_timestamp_string_normalizes_to_timestamp_type(self) -> None:
        """ColumnSpec with type 'timestamp' is normalized to TimestampType()."""
        col = ColumnSpec(name="ts", type="timestamp", description="A timestamp")  # type: ignore[arg-type]
        assert col.type == TimestampType()
        assert isinstance(col.type, TimestampType)
        assert col.type.tz is None

    def test_with_list_type(self) -> None:
        """ColumnSpec with ListType is valid."""
        col = ColumnSpec(name="tags", type=ListType(value_type="string"), description="Tags list")
        assert isinstance(col.type, ListType)

    def test_with_struct_type(self) -> None:
        """ColumnSpec with StructType is valid."""
        sf = StructField(name="x", type="int", description="An int field")
        col = ColumnSpec(name="data", type=StructType(fields=(sf,)), description="Struct column")
        assert isinstance(col.type, StructType)

    def test_with_map_type(self) -> None:
        """ColumnSpec with MapType is valid."""
        col = ColumnSpec(name="attrs", type=MapType(key_type="string", value_type="string"), description="Attributes")
        assert isinstance(col.type, MapType)

    def test_frozen(self) -> None:
        """ColumnSpec is immutable."""
        col = ColumnSpec(name="col", type="int", description="An int")
        with pytest.raises(dataclasses.FrozenInstanceError):
            col.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestSLASpec
# ---------------------------------------------------------------------------


class TestSLASpec:
    """Tests for SLASpec dataclass."""

    def test_valid_sla_spec(self) -> None:
        """SLASpec with valid cron and lag is constructed correctly."""
        sla = SLASpec(schedule="0 6 * * *", lag_hours=2.0)
        assert sla.schedule == "0 6 * * *"
        assert sla.lag_hours == pytest.approx(2.0)

    def test_zero_lag_hours_is_valid(self) -> None:
        """SLASpec with lag_hours=0 is valid."""
        sla = SLASpec(schedule="0 6 * * *", lag_hours=0.0)
        assert sla.lag_hours == pytest.approx(0.0)

    def test_negative_lag_hours_raises(self) -> None:
        """Negative lag_hours raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="lag_hours"):
            SLASpec(schedule="0 6 * * *", lag_hours=-1.0)

    def test_invalid_cron_format_raises(self) -> None:
        """Non-5-field cron raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="cron"):
            SLASpec(schedule="not a cron", lag_hours=1.0)

    def test_four_field_cron_raises(self) -> None:
        """4-field cron raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="cron"):
            SLASpec(schedule="0 6 * *", lag_hours=1.0)

    def test_six_field_cron_raises(self) -> None:
        """6-field cron raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="cron"):
            SLASpec(schedule="0 0 6 * * *", lag_hours=1.0)

    def test_day_of_month_list_raises(self) -> None:
        """List-based day-of-month raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="day-of-month"):
            SLASpec(schedule="0 6 1,15 * *", lag_hours=1.0)

    def test_day_of_week_list_raises(self) -> None:
        """List-based day-of-week raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="day-of-week"):
            SLASpec(schedule="0 6 * * 1,3", lag_hours=1.0)

    def test_wildcard_day_of_month_is_valid(self) -> None:
        """Wildcard * in day-of-month is valid."""
        sla = SLASpec(schedule="0 6 * * *", lag_hours=1.0)
        assert sla.schedule == "0 6 * * *"

    def test_various_valid_crons(self) -> None:
        """Various valid 5-field cron expressions are accepted."""
        for cron in ("* * * * *", "0 0 1 1 *", "30 8 * * 1", "*/15 * * * *"):
            sla = SLASpec(schedule=cron, lag_hours=0.0)
            assert sla.schedule == cron

    def test_frozen(self) -> None:
        """SLASpec is immutable."""
        sla = SLASpec(schedule="0 6 * * *", lag_hours=2.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            sla.lag_hours = 3.0  # type: ignore[misc]

    def test_out_of_range_minute_raises(self) -> None:
        """Out-of-range minute field raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="minute"):
            SLASpec(schedule="60 0 * * *", lag_hours=1.0)

    def test_out_of_range_hour_raises(self) -> None:
        """Out-of-range hour field raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="hour"):
            SLASpec(schedule="0 24 * * *", lag_hours=1.0)

    def test_out_of_range_day_of_month_raises(self) -> None:
        """Out-of-range day-of-month field raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="day-of-month"):
            SLASpec(schedule="0 0 32 * *", lag_hours=1.0)

    def test_out_of_range_month_raises(self) -> None:
        """Out-of-range month field raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="month"):
            SLASpec(schedule="0 0 * 13 *", lag_hours=1.0)

    def test_out_of_range_day_of_week_raises(self) -> None:
        """Out-of-range day-of-week field raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="day-of-week"):
            SLASpec(schedule="0 0 * * 8", lag_hours=1.0)

    def test_garbage_field_raises(self) -> None:
        """Garbage cron fields like 'foo bar baz qux quux' raise ContractValidationError."""
        with pytest.raises(ContractValidationError, match=r"cron|minute"):
            SLASpec(schedule="foo bar baz qux quux", lag_hours=1.0)

    def test_all_out_of_range_raises(self) -> None:
        """All out-of-range fields like '99 99 99 99 99' raise ContractValidationError."""
        with pytest.raises(ContractValidationError, match=r"minute|hour|day"):
            SLASpec(schedule="99 99 99 99 99", lag_hours=1.0)

    def test_symbolic_month_names_valid(self) -> None:
        """Symbolic month names (JAN-DEC) are accepted."""
        sla = SLASpec(schedule="0 6 1 JAN *", lag_hours=1.0)
        assert sla.schedule == "0 6 1 JAN *"

    def test_symbolic_day_of_week_names_valid(self) -> None:
        """Symbolic day-of-week names (SUN-SAT) are accepted."""
        sla = SLASpec(schedule="0 6 * * MON", lag_hours=1.0)
        assert sla.schedule == "0 6 * * MON"

    def test_step_expression_valid(self) -> None:
        """Step expressions like '*/15' are accepted."""
        sla = SLASpec(schedule="*/15 * * * *", lag_hours=0.0)
        assert sla.schedule == "*/15 * * * *"

    def test_invalid_step_zero_raises(self) -> None:
        """Step value of 0 in a cron field raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="minute"):
            SLASpec(schedule="0/0 * * * *", lag_hours=1.0)

    def test_inverted_cron_range_raises(self) -> None:
        """Inverted range (lo > hi) in a cron field raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="day-of-month"):
            SLASpec(schedule="0 0 31-1 * *", lag_hours=1.0)

    def test_lag_hours_above_168_on_specific_schedule_warns(self) -> None:
        """lag_hours > 168 on a non-catch-all schedule emits a UserWarning."""
        with pytest.warns(UserWarning, match="lag_hours=200"):
            SLASpec(schedule="0 6 * * *", lag_hours=200.0)

    def test_lag_hours_above_168_on_catch_all_no_warning(self) -> None:
        """lag_hours > 168 on catch-all schedule '* * * * *' does NOT warn."""
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")  # any warning would become an error
            sla = SLASpec(schedule="* * * * *", lag_hours=200.0)
        assert sla.lag_hours == pytest.approx(200.0)

    def test_lag_hours_exactly_168_no_warning(self) -> None:
        """lag_hours == 168 does NOT warn (threshold is strictly > 168)."""
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            sla = SLASpec(schedule="0 6 * * *", lag_hours=168.0)
        assert sla.lag_hours == pytest.approx(168.0)

    def test_symbolic_month_name_with_day_valid(self) -> None:
        """Specific day+month using symbolic names is accepted."""
        sla = SLASpec(schedule="0 6 1 JAN *", lag_hours=1.0)
        assert sla.schedule == "0 6 1 JAN *"


# ---------------------------------------------------------------------------
# TestContract
# ---------------------------------------------------------------------------


def _make_column(name: str = "user_id", col_type: str = "int", description: str = "User ID") -> ColumnSpec:
    """Helper to create a valid ColumnSpec."""
    return ColumnSpec(name=name, type=cast(ContractType, col_type), description=description)


def _make_contract(**overrides: object) -> Contract:
    """Helper to create a valid Contract with defaults."""
    defaults: dict[str, object] = {
        "name": "my_contract",
        "version": "1.0",
        "description": "A test contract",
        "owner": "team-data",
        "dataset": "my_dataset",
        "columns": (_make_column(),),
    }
    defaults.update(overrides)
    return Contract(**defaults)  # type: ignore[arg-type]


class TestContract:
    """Tests for Contract dataclass."""

    def test_valid_contract(self) -> None:
        """Contract with valid fields is constructed correctly."""
        contract = _make_contract()
        assert contract.name == "my_contract"
        assert contract.version == "1.0"
        assert contract.description == "A test contract"
        assert contract.owner == "team-data"
        assert contract.dataset == "my_dataset"
        assert len(contract.columns) == 1
        assert contract.tags == frozenset()
        assert contract.sla is None
        assert contract.partitioned_by == ()
        assert contract.metadata == ()
        assert contract.checks == ()

    def test_with_sla(self) -> None:
        """Contract with SLASpec is valid when partitioned_by is set (no timestamp_column required)."""
        sla = SLASpec(schedule="0 6 * * *", lag_hours=2.0)
        col_date = _make_column(name="date", col_type="date", description="Partition date")
        col_id = _make_column(name="user_id")
        contract = _make_contract(columns=(col_id, col_date), sla=sla, partitioned_by=("date",))
        assert contract.sla == sla

    def test_with_tags(self) -> None:
        """Contract with tags is valid."""
        contract = _make_contract(tags=frozenset({"production", "finance"}))
        assert contract.tags == frozenset({"production", "finance"})

    def test_with_partitioned_by(self) -> None:
        """Contract with partitioned_by referencing existing columns is valid."""
        col_date = _make_column(name="date", col_type="date", description="Partition date")
        col_id = _make_column(name="user_id")
        contract = _make_contract(columns=(col_id, col_date), partitioned_by=("date",))
        assert contract.partitioned_by == ("date",)

    def test_with_metadata(self) -> None:
        """Contract with metadata tuples is valid."""
        contract = _make_contract(metadata=(("team", "data-eng"), ("env", "prod")))
        assert contract.metadata == (("team", "data-eng"), ("env", "prod"))

    def test_with_table_checks(self) -> None:
        """Contract with table-level checks is valid."""
        check = NumRowsCheck(name="row_count", validator=ValidatorSpec(min=1.0))
        contract = _make_contract(checks=(check,))
        assert len(contract.checks) == 1

    def test_multiple_columns(self) -> None:
        """Contract with multiple columns is valid."""
        cols = (
            _make_column(name="id", col_type="int", description="ID"),
            _make_column(name="name", col_type="string", description="Name"),
        )
        contract = _make_contract(columns=cols)
        assert len(contract.columns) == 2

    # --- Validation errors ---

    def test_empty_name_raises(self) -> None:
        """Empty name raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="name"):
            _make_contract(name="")

    def test_name_too_long_raises(self) -> None:
        """name > 255 chars raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="name"):
            _make_contract(name="a" * 256)

    def test_name_exactly_255_is_valid(self) -> None:
        """name of exactly 255 chars is valid."""
        contract = _make_contract(name="a" * 255)
        assert len(contract.name) == 255

    def test_empty_version_raises(self) -> None:
        """Empty version raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="version"):
            _make_contract(version="")

    def test_empty_description_raises(self) -> None:
        """Empty description raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="description"):
            _make_contract(description="")

    def test_empty_owner_raises(self) -> None:
        """Empty owner raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="owner"):
            _make_contract(owner="")

    def test_empty_dataset_raises(self) -> None:
        """Empty dataset raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="dataset"):
            _make_contract(dataset="")

    def test_dataset_with_whitespace_raises(self) -> None:
        """Dataset containing whitespace raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="dataset"):
            _make_contract(dataset="my dataset")

    def test_dataset_with_tab_raises(self) -> None:
        """Dataset with tab character raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="dataset"):
            _make_contract(dataset="my\tdataset")

    def test_empty_columns_raises(self) -> None:
        """Empty columns tuple raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="columns"):
            _make_contract(columns=())

    def test_duplicate_column_names_raises(self) -> None:
        """Duplicate column names raise ContractValidationError."""
        col1 = _make_column(name="id")
        col2 = _make_column(name="id", col_type="string", description="Duplicate ID")
        with pytest.raises(ContractValidationError, match="duplicate"):
            _make_contract(columns=(col1, col2))

    def test_partitioned_by_unknown_column_raises(self) -> None:
        """partitioned_by referencing unknown column raises ContractValidationError."""
        with pytest.raises(ContractValidationError, match="partitioned_by"):
            _make_contract(partitioned_by=("nonexistent_column",))

    def test_partitioned_by_multiple_valid_columns(self) -> None:
        """partitioned_by with multiple valid columns is accepted."""
        col_year = _make_column(name="year", col_type="int", description="Year")
        col_month = _make_column(name="month", col_type="int", description="Month")
        col_id = _make_column(name="id")
        contract = _make_contract(
            columns=(col_id, col_year, col_month),
            partitioned_by=("year", "month"),
        )
        assert contract.partitioned_by == ("year", "month")

    def test_frozen(self) -> None:
        """Contract is immutable."""
        contract = _make_contract()
        with pytest.raises(dataclasses.FrozenInstanceError):
            contract.name = "other"  # type: ignore[misc]

    def test_sla_with_partitioned_by_no_timestamp_column_in_metadata_is_valid(self) -> None:
        """Contract with SLA + partitioned_by but no metadata.timestamp_column is valid."""
        sla = SLASpec(schedule="0 6 * * *", lag_hours=2.0)
        col_date = _make_column(name="date", col_type="date", description="Partition date")
        col_id = _make_column(name="user_id")
        contract = _make_contract(columns=(col_id, col_date), sla=sla, partitioned_by=("date",))
        assert contract.sla == sla

    def test_sla_without_partitioned_by_missing_metadata_timestamp_column_raises(self) -> None:
        """Contract with SLA + no partitioned_by + no metadata.timestamp_column raises ContractValidationError."""
        sla = SLASpec(schedule="0 6 * * *", lag_hours=0.0)
        with pytest.raises(ContractValidationError, match="timestamp_column"):
            _make_contract(sla=sla)

    def test_sla_without_partitioned_by_with_metadata_timestamp_column_is_valid(self) -> None:
        """Contract with SLA + no partitioned_by + metadata.timestamp_column referencing a real column is valid."""
        sla = SLASpec(schedule="0 6 * * *", lag_hours=0.0)
        col_ts = _make_column(name="last_updated", col_type="timestamp", description="Update timestamp")
        col_id = _make_column()
        contract = _make_contract(
            columns=(col_id, col_ts),
            sla=sla,
            metadata=(("timestamp_column", "last_updated"),),
        )
        assert contract.sla == sla

    def test_no_sla_no_metadata_timestamp_column_required(self) -> None:
        """Contract with no SLA and no metadata is valid."""
        contract = _make_contract()
        assert contract.sla is None

    def test_sla_without_partitioned_by_empty_timestamp_column_value_raises(self) -> None:
        """Contract with SLA + no partitioned_by + empty timestamp_column value raises ContractValidationError."""
        sla = SLASpec(schedule="0 6 * * *", lag_hours=0.0)
        with pytest.raises(ContractValidationError, match="timestamp_column"):
            _make_contract(sla=sla, metadata=(("timestamp_column", ""),))

    def test_sla_without_partitioned_by_nonexistent_timestamp_column_raises(self) -> None:
        """Contract with SLA + no partitioned_by + unknown column in timestamp_column raises ContractValidationError."""
        sla = SLASpec(schedule="0 6 * * *", lag_hours=0.0)
        with pytest.raises(ContractValidationError, match="does not reference a known column"):
            _make_contract(sla=sla, metadata=(("timestamp_column", "missing_col"),))
