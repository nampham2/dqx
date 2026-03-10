"""Tests for Contract.to_checks() and check dataclass to_dqx() methods."""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest

from dqx.api import VerificationSuite
from dqx.common import AssertionResult, ResultKey
from dqx.contract.models import (
    AvgLengthCheck,
    BetweenValidator,
    BlacklistCheck,
    CardinalityCheck,
    ColumnDuplicatesCheck,
    ColumnSpec,
    CompletenessCheck,
    Contract,
    CountCheck,
    EqualsValidator,
    FreshnessCheck,
    MaxCheck,
    MaxLengthCheck,
    MaxValidator,
    MeanCheck,
    MinCheck,
    MinLengthCheck,
    MinValidator,
    MissingCheck,
    NotBetweenValidator,
    NumRowsCheck,
    PatternCheck,
    PercentileCheck,
    StddevCheck,
    SumCheck,
    TableDuplicatesCheck,
    VarianceCheck,
    WhitelistCheck,
)
from dqx.datasource import DuckRelationDataSource
from dqx.display import print_assertion_results
from dqx.orm.repositories import InMemoryMetricDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_contract(**overrides: object) -> Contract:
    """Build a minimal valid Contract, applying keyword overrides."""
    defaults: dict[str, object] = {
        "name": "Test Contract",
        "version": "1.0.0",
        "description": "Test",
        "owner": "team",
        "dataset": "orders",
        "columns": (ColumnSpec(name="order_id", type="int", description="Order ID", nullable=False),),
        "checks": (),
    }
    defaults.update(overrides)
    return Contract(**defaults)  # type: ignore[arg-type]


def _run_and_collect(
    contract: Contract,
    data: pa.Table,
) -> list[AssertionResult]:
    """Run ``contract.to_checks()`` against ``data`` and return results."""
    datasource = DuckRelationDataSource.from_arrow(data, contract.dataset)
    db = InMemoryMetricDB()
    suite = VerificationSuite(checks=contract.to_checks(), db=db, name="test suite")
    key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={})
    suite.run([datasource], key)
    result = suite.collect_results()
    print_assertion_results(result)
    return result


def _simple_table(n: int = 5) -> pa.Table:
    """Return a table with ``order_id`` column [0..n-1]."""
    return pa.table({"order_id": list(range(n))})


# ---------------------------------------------------------------------------
# TestToChecksStructure
# ---------------------------------------------------------------------------


class TestToChecksStructure:
    """Tests for the structure of the list returned by to_checks()."""

    def test_always_returns_one_decorated_check(self) -> None:
        """to_checks() returns exactly one DecoratedCheck regardless of check count."""
        contract = _minimal_contract(
            checks=(NumRowsCheck(name="Row count", validators=(MinValidator(threshold=1.0),)),)
        )
        assert len(contract.to_checks()) == 1

    def test_returns_one_check_even_with_multiple_table_checks(self) -> None:
        """Multiple NumRowsChecks still produce exactly one DecoratedCheck."""
        contract = _minimal_contract(
            checks=(
                NumRowsCheck(name="Min rows", validators=(MinValidator(threshold=1.0),)),
                NumRowsCheck(name="Max rows", validators=(MaxValidator(threshold=1000),)),
            )
        )
        assert len(contract.to_checks()) == 1

    def test_returns_one_check_even_with_no_checks(self) -> None:
        """A contract with no checks still produces exactly one DecoratedCheck."""
        contract = _minimal_contract(checks=())
        assert len(contract.to_checks()) == 1

    def test_decorated_check_name_is_contract_name(self) -> None:
        """The DecoratedCheck's __name__ is the contract name."""
        contract = _minimal_contract(
            name="Orders Quality Contract",
            checks=(NumRowsCheck(name="Row count", validators=(MinValidator(threshold=1.0),)),),
        )
        assert contract.to_checks()[0].__name__ == "Orders Quality Contract"

    def test_all_assertions_share_contract_check_name(self) -> None:
        """Every AssertionResult.check equals the contract name."""
        contract = _minimal_contract(
            name="Orders Quality Contract",
            checks=(
                NumRowsCheck(name="Min rows", validators=(MinValidator(threshold=1.0),)),
                NumRowsCheck(name="Max rows", validators=(MaxValidator(threshold=1000),)),
            ),
        )
        results = _run_and_collect(contract, _simple_table(5))
        assert all(r.check == "Orders Quality Contract" for r in results)

    def test_dataset_binding_uses_contract_dataset(self) -> None:
        """The DecoratedCheck is bound to contract.dataset."""
        contract = _minimal_contract(
            dataset="shipments",
            checks=(NumRowsCheck(name="Row count", validators=(MinValidator(threshold=1.0),)),),
        )
        data = pa.table({"order_id": [1, 2, 3]})
        datasource = DuckRelationDataSource.from_arrow(data, "shipments")
        db = InMemoryMetricDB()
        suite = VerificationSuite(checks=contract.to_checks(), db=db, name="suite")
        suite.run([datasource], ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={}))
        assert len(suite.collect_results()) == 1


# ---------------------------------------------------------------------------
# TestNumRowsToDqx
# ---------------------------------------------------------------------------


class TestNumRowsToDqx:
    """Tests for NumRowsCheck.to_dqx()."""

    def test_noop_produces_one_passed_assertion(self) -> None:
        """NumRowsCheck with no validators produces one noop (PASSED) assertion."""
        contract = _minimal_contract(checks=(NumRowsCheck(name="Noop check", validators=()),))
        results = _run_and_collect(contract, _simple_table(5))
        assert len(results) == 1
        assert results[0].status == "PASSED"

    def test_min_validator_passes(self) -> None:
        """MinValidator passes when row count >= threshold."""
        contract = _minimal_contract(checks=(NumRowsCheck(name="Min", validators=(MinValidator(threshold=3.0),)),))
        results = _run_and_collect(contract, _simple_table(5))
        assert results[0].status == "PASSED"

    def test_min_validator_fails(self) -> None:
        """MinValidator fails when row count < threshold."""
        contract = _minimal_contract(checks=(NumRowsCheck(name="Min", validators=(MinValidator(threshold=10.0),)),))
        results = _run_and_collect(contract, _simple_table(5))
        assert results[0].status == "FAILED"

    def test_max_validator_passes(self) -> None:
        """MaxValidator passes when row count <= threshold."""
        contract = _minimal_contract(checks=(NumRowsCheck(name="Max", validators=(MaxValidator(threshold=10),)),))
        results = _run_and_collect(contract, _simple_table(5))
        assert results[0].status == "PASSED"

    def test_max_validator_fails(self) -> None:
        """MaxValidator fails when row count > threshold."""
        contract = _minimal_contract(checks=(NumRowsCheck(name="Max", validators=(MaxValidator(threshold=3),)),))
        results = _run_and_collect(contract, _simple_table(5))
        assert results[0].status == "FAILED"

    def test_between_validator_passes(self) -> None:
        """BetweenValidator passes when row count is within [low, high]."""
        contract = _minimal_contract(
            checks=(NumRowsCheck(name="Between", validators=(BetweenValidator(low=3, high=10),)),)
        )
        results = _run_and_collect(contract, _simple_table(5))
        assert results[0].status == "PASSED"

    def test_between_validator_fails(self) -> None:
        """BetweenValidator fails when row count is outside [low, high]."""
        contract = _minimal_contract(
            checks=(NumRowsCheck(name="Between", validators=(BetweenValidator(low=10, high=20),)),)
        )
        results = _run_and_collect(contract, _simple_table(5))
        assert results[0].status == "FAILED"

    def test_not_between_produces_single_assertion(self) -> None:
        """NotBetweenValidator produces exactly one assertion with OR semantics."""
        contract = _minimal_contract(
            checks=(NumRowsCheck(name="Not between", validators=(NotBetweenValidator(low=10, high=20),)),)
        )
        results = _run_and_collect(contract, _simple_table(5))
        assert len(results) == 1

    def test_not_between_passes_when_below_range(self) -> None:
        """NotBetweenValidator passes when metric is below the lower bound."""
        contract = _minimal_contract(
            checks=(NumRowsCheck(name="Not between", validators=(NotBetweenValidator(low=10, high=20),)),)
        )
        # 5 rows < 10 → outside range → PASSED
        results = _run_and_collect(contract, _simple_table(5))
        assert results[0].status == "PASSED"

    def test_not_between_passes_when_above_range(self) -> None:
        """NotBetweenValidator passes when metric is above the upper bound."""
        contract = _minimal_contract(
            checks=(NumRowsCheck(name="Not between", validators=(NotBetweenValidator(low=10, high=20),)),)
        )
        # 25 rows > 20 → outside range → PASSED
        results = _run_and_collect(contract, _simple_table(25))
        assert results[0].status == "PASSED"

    def test_not_between_fails_when_inside_range(self) -> None:
        """NotBetweenValidator fails when metric is inside the [low, high] range."""
        contract = _minimal_contract(
            checks=(NumRowsCheck(name="Not between", validators=(NotBetweenValidator(low=10, high=20),)),)
        )
        # 15 rows is within [10, 20] → FAILED
        results = _run_and_collect(contract, _simple_table(15))
        assert results[0].status == "FAILED"

    def test_not_between_assertion_name_encodes_bounds(self) -> None:
        """NotBetweenValidator assertion name includes both bounds."""
        contract = _minimal_contract(
            checks=(NumRowsCheck(name="Out of range", validators=(NotBetweenValidator(low=10, high=20),)),)
        )
        results = _run_and_collect(contract, _simple_table(5))
        assert results[0].assertion == "Out of range [not_between 10 and 20]"

    def test_equals_validator_passes(self) -> None:
        """EqualsValidator passes when row count equals expected value."""
        contract = _minimal_contract(checks=(NumRowsCheck(name="Equals", validators=(EqualsValidator(value=5),)),))
        results = _run_and_collect(contract, _simple_table(5))
        assert results[0].status == "PASSED"

    def test_assertion_name_encodes_check_name_and_validator(self) -> None:
        """Assertion name includes NumRowsCheck name and validator threshold."""
        contract = _minimal_contract(checks=(NumRowsCheck(name="Volume", validators=(MinValidator(threshold=10.0),)),))
        results = _run_and_collect(contract, _simple_table(5))
        assert results[0].assertion == "Volume [min >= 10.0]"

    def test_severity_propagates(self) -> None:
        """NumRowsCheck severity propagates to every assertion."""
        contract = _minimal_contract(
            checks=(NumRowsCheck(name="P0 check", severity="P0", validators=(MinValidator(threshold=1.0),)),)
        )
        results = _run_and_collect(contract, _simple_table(5))
        assert results[0].severity == "P0"

    def test_tags_propagate(self) -> None:
        """NumRowsCheck tags propagate to every assertion."""
        contract = _minimal_contract(
            checks=(NumRowsCheck(name="Tagged", tags=frozenset({"core"}), validators=(MinValidator(threshold=1.0),)),)
        )
        results = _run_and_collect(contract, _simple_table(5))
        assert results[0].assertion_tags == frozenset({"core"})


# ---------------------------------------------------------------------------
# TestTableDuplicatesToDqx
# ---------------------------------------------------------------------------


class TestTableDuplicatesToDqx:
    """Tests for TableDuplicatesCheck.to_dqx()."""

    def test_duplicate_count_zero_passes(self) -> None:
        """Zero duplicates passes a max <= 0 validator."""
        contract = _minimal_contract(
            checks=(
                TableDuplicatesCheck(
                    name="No dupes",
                    columns=("order_id",),
                    validators=(MaxValidator(threshold=0.0),),
                ),
            )
        )
        data = pa.table({"order_id": [1, 2, 3, 4, 5]})
        results = _run_and_collect(contract, data)
        assert results[0].status == "PASSED"

    def test_duplicate_count_nonzero_fails(self) -> None:
        """Duplicates present fail a max <= 0 validator."""
        contract = _minimal_contract(
            checks=(
                TableDuplicatesCheck(
                    name="No dupes",
                    columns=("order_id",),
                    validators=(MaxValidator(threshold=0.0),),
                ),
            )
        )
        data = pa.table({"order_id": [1, 1, 2, 3, 4]})
        results = _run_and_collect(contract, data)
        assert results[0].status == "FAILED"

    def test_assertion_name_encodes_check_name(self) -> None:
        """Assertion name includes TableDuplicatesCheck name and validator."""
        contract = _minimal_contract(
            checks=(
                TableDuplicatesCheck(
                    name="Dup check",
                    columns=("order_id",),
                    validators=(MaxValidator(threshold=0.0),),
                ),
            )
        )
        data = pa.table({"order_id": [1, 2, 3]})
        results = _run_and_collect(contract, data)
        assert results[0].assertion == "Dup check [max <= 0.0]"


# ---------------------------------------------------------------------------
# TestFreshnessAndCompletenessToDqx
# ---------------------------------------------------------------------------


class TestFreshnessAndCompletenessToDqx:
    """Tests that unsupported checks raise NotImplementedError from to_dqx()."""

    def test_freshness_raises_not_implemented(self) -> None:
        """FreshnessCheck.to_dqx() raises NotImplementedError."""
        check = FreshnessCheck(name="Fresh", max_age_hours=24.0, timestamp_column="ts")
        with pytest.raises(NotImplementedError, match="FreshnessCheck"):
            check.to_dqx(None, None)  # type: ignore[arg-type]

    def test_completeness_raises_not_implemented(self) -> None:
        """CompletenessCheck.to_dqx() raises NotImplementedError."""
        check = CompletenessCheck(name="Complete", partition_column="date", granularity="daily")
        with pytest.raises(NotImplementedError, match="CompletenessCheck"):
            check.to_dqx(None, None)  # type: ignore[arg-type]

    def test_freshness_in_contract_raises_not_implemented(self) -> None:
        """A contract with FreshnessCheck raises NotImplementedError when run."""
        contract = _minimal_contract(
            columns=(
                ColumnSpec(name="order_id", type="int", description="ID", nullable=False),
                ColumnSpec(name="ts", type="timestamp", description="Timestamp", nullable=False),
            ),
            checks=(FreshnessCheck(name="Fresh", max_age_hours=24.0, timestamp_column="ts"),),
        )
        data = pa.table(
            {"order_id": [1, 2], "ts": pa.array([datetime.datetime(2024, 1, 1)] * 2, type=pa.timestamp("us"))}
        )
        with pytest.raises(NotImplementedError, match="FreshnessCheck"):
            _run_and_collect(contract, data)


# ---------------------------------------------------------------------------
# TestMissingCheckToDqx
# ---------------------------------------------------------------------------


class TestMissingCheckToDqx:
    """Tests for MissingCheck.to_dqx()."""

    def test_no_nulls_passes_max_zero(self) -> None:
        """Zero nulls passes a max <= 0 validator."""
        contract = _minimal_contract(
            columns=(ColumnSpec(name="val", type="float", description="Value", nullable=True),),
            checks=(),
        )
        contract = _minimal_contract(
            columns=(
                ColumnSpec(
                    name="val",
                    type="float",
                    description="Value",
                    nullable=True,
                    checks=(MissingCheck(name="No nulls", validators=(MaxValidator(threshold=0.0),)),),
                ),
            ),
        )
        data = pa.table({"val": [1.0, 2.0, 3.0]})
        results = _run_and_collect(contract, data)
        assert results[0].status == "PASSED"

    def test_nulls_present_fails_max_zero(self) -> None:
        """Nulls present fail a max <= 0 validator."""
        contract = _minimal_contract(
            columns=(
                ColumnSpec(
                    name="val",
                    type="float",
                    description="Value",
                    nullable=True,
                    checks=(MissingCheck(name="No nulls", validators=(MaxValidator(threshold=0.0),)),),
                ),
            ),
        )
        data = pa.table({"val": pa.array([1.0, None, 3.0], type=pa.float64())})
        results = _run_and_collect(contract, data)
        assert results[0].status == "FAILED"

    def test_pct_return_type_divides_by_num_rows(self) -> None:
        """MissingCheck with return_type='pct' reports proportion of nulls."""
        contract = _minimal_contract(
            columns=(
                ColumnSpec(
                    name="val",
                    type="float",
                    description="Value",
                    nullable=True,
                    checks=(
                        MissingCheck(
                            name="Null pct",
                            return_type="pct",
                            validators=(MaxValidator(threshold=0.5),),
                        ),
                    ),
                ),
            ),
        )
        # 1 null out of 4 = 25% < 50% → PASSED
        data = pa.table({"val": pa.array([1.0, None, 3.0, 4.0], type=pa.float64())})
        results = _run_and_collect(contract, data)
        assert results[0].status == "PASSED"


# ---------------------------------------------------------------------------
# TestColumnDuplicatesToDqx
# ---------------------------------------------------------------------------


class TestColumnDuplicatesToDqx:
    """Tests for ColumnDuplicatesCheck.to_dqx()."""

    def test_no_duplicates_passes_max_zero(self) -> None:
        """Zero column duplicates passes a max <= 0 validator."""
        contract = _minimal_contract(
            columns=(
                ColumnSpec(
                    name="order_id",
                    type="int",
                    description="ID",
                    nullable=False,
                    checks=(ColumnDuplicatesCheck(name="No dupes", validators=(MaxValidator(threshold=0.0),)),),
                ),
            ),
        )
        data = pa.table({"order_id": [1, 2, 3]})
        results = _run_and_collect(contract, data)
        assert results[0].status == "PASSED"

    def test_duplicates_fail_max_zero(self) -> None:
        """Column duplicates present fail a max <= 0 validator."""
        contract = _minimal_contract(
            columns=(
                ColumnSpec(
                    name="order_id",
                    type="int",
                    description="ID",
                    nullable=False,
                    checks=(ColumnDuplicatesCheck(name="No dupes", validators=(MaxValidator(threshold=0.0),)),),
                ),
            ),
        )
        data = pa.table({"order_id": [1, 1, 2]})
        results = _run_and_collect(contract, data)
        assert results[0].status == "FAILED"

    def test_pct_return_type_divides_by_num_rows(self) -> None:
        """ColumnDuplicatesCheck with return_type='pct' reports proportion of duplicates."""
        contract = _minimal_contract(
            columns=(
                ColumnSpec(
                    name="order_id",
                    type="int",
                    description="ID",
                    nullable=False,
                    checks=(
                        ColumnDuplicatesCheck(
                            name="Dup pct",
                            return_type="pct",
                            validators=(MaxValidator(threshold=0.5),),
                        ),
                    ),
                ),
            ),
        )
        # 1 duplicate out of 3 rows ≈ 33% < 50% → PASSED
        data = pa.table({"order_id": [1, 1, 2]})
        results = _run_and_collect(contract, data)
        assert results[0].status == "PASSED"


# ---------------------------------------------------------------------------
# TestWhitelistAndBlacklistToDqx
# ---------------------------------------------------------------------------


class TestWhitelistAndBlacklistToDqx:
    """Tests for WhitelistCheck.to_dqx() and BlacklistCheck.to_dqx()."""

    def test_whitelist_all_values_match_passes(self) -> None:
        """WhitelistCheck: all rows matching whitelist values passes min >= n."""
        contract = _minimal_contract(
            columns=(
                ColumnSpec(
                    name="status",
                    type="string",
                    description="Status",
                    nullable=False,
                    checks=(
                        WhitelistCheck(
                            name="Valid statuses",
                            values=("active", "inactive"),
                            validators=(MinValidator(threshold=3.0),),
                        ),
                    ),
                ),
            ),
        )
        data = pa.table({"status": ["active", "inactive", "active"]})
        results = _run_and_collect(contract, data)
        assert results[0].status == "PASSED"

    def test_whitelist_no_match_fails(self) -> None:
        """WhitelistCheck: no matching values fails min >= threshold."""
        contract = _minimal_contract(
            columns=(
                ColumnSpec(
                    name="status",
                    type="string",
                    description="Status",
                    nullable=False,
                    checks=(
                        WhitelistCheck(
                            name="Valid statuses",
                            values=("active", "inactive"),
                            validators=(MinValidator(threshold=1.0),),
                        ),
                    ),
                ),
            ),
        )
        data = pa.table({"status": ["unknown", "other", "bad"]})
        results = _run_and_collect(contract, data)
        assert results[0].status == "FAILED"

    def test_blacklist_all_safe_rows_passes(self) -> None:
        """BlacklistCheck: safe (non-blacklisted) row count passes min >= n."""
        contract = _minimal_contract(
            columns=(
                ColumnSpec(
                    name="status",
                    type="string",
                    description="Status",
                    nullable=False,
                    checks=(
                        BlacklistCheck(
                            name="No banned",
                            values=("banned", "deleted"),
                            validators=(MinValidator(threshold=3.0),),
                        ),
                    ),
                ),
            ),
        )
        # All 3 rows are safe (not in blacklist)
        data = pa.table({"status": ["active", "inactive", "pending"]})
        results = _run_and_collect(contract, data)
        assert results[0].status == "PASSED"

    def test_blacklist_some_unsafe_rows_fails(self) -> None:
        """BlacklistCheck: presence of blacklisted rows reduces safe count."""
        contract = _minimal_contract(
            columns=(
                ColumnSpec(
                    name="status",
                    type="string",
                    description="Status",
                    nullable=False,
                    checks=(
                        BlacklistCheck(
                            name="No banned",
                            values=("banned",),
                            validators=(MinValidator(threshold=3.0),),
                        ),
                    ),
                ),
            ),
        )
        # 1 banned row → safe count = 2, which fails min >= 3
        data = pa.table({"status": ["active", "banned", "inactive"]})
        results = _run_and_collect(contract, data)
        assert results[0].status == "FAILED"


# ---------------------------------------------------------------------------
# TestUnsupportedColumnChecksToDqx
# ---------------------------------------------------------------------------


class TestUnsupportedColumnChecksToDqx:
    """Tests that unsupported column checks raise NotImplementedError."""

    def test_pattern_check_raises_not_implemented(self) -> None:
        """PatternCheck.to_dqx() raises NotImplementedError."""
        check = PatternCheck(name="Pattern", pattern=r"\d+")
        with pytest.raises(NotImplementedError, match="PatternCheck"):
            check.to_dqx("col", None, None)  # type: ignore[arg-type]

    def test_min_length_check_raises_not_implemented(self) -> None:
        """MinLengthCheck.to_dqx() raises NotImplementedError."""
        check = MinLengthCheck(name="Min length")
        with pytest.raises(NotImplementedError, match="MinLengthCheck"):
            check.to_dqx("col", None, None)  # type: ignore[arg-type]

    def test_max_length_check_raises_not_implemented(self) -> None:
        """MaxLengthCheck.to_dqx() raises NotImplementedError."""
        check = MaxLengthCheck(name="Max length")
        with pytest.raises(NotImplementedError, match="MaxLengthCheck"):
            check.to_dqx("col", None, None)  # type: ignore[arg-type]

    def test_avg_length_check_raises_not_implemented(self) -> None:
        """AvgLengthCheck.to_dqx() raises NotImplementedError."""
        check = AvgLengthCheck(name="Avg length")
        with pytest.raises(NotImplementedError, match="AvgLengthCheck"):
            check.to_dqx("col", None, None)  # type: ignore[arg-type]

    def test_percentile_check_raises_not_implemented(self) -> None:
        """PercentileCheck.to_dqx() raises NotImplementedError."""
        check = PercentileCheck(name="P99", percentile=0.99)
        with pytest.raises(NotImplementedError, match="PercentileCheck"):
            check.to_dqx("col", None, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestNumericColumnChecksToDqx
# ---------------------------------------------------------------------------


class TestNumericColumnChecksToDqx:
    """Tests for numeric column-level checks: MinCheck, MaxCheck, MeanCheck,
    SumCheck, CountCheck, VarianceCheck, CardinalityCheck."""

    def _contract_with_column_check(self, col_check: object) -> Contract:
        return _minimal_contract(
            columns=(
                ColumnSpec(
                    name="val",
                    type="float",
                    description="Value",
                    nullable=True,
                    checks=(col_check,),  # type: ignore[arg-type]
                ),
            ),
        )

    def test_min_check_passes(self) -> None:
        """MinCheck: column minimum satisfies validator."""
        contract = self._contract_with_column_check(MinCheck(name="Col min", validators=(MinValidator(threshold=1.0),)))
        data = pa.table({"val": pa.array([2.0, 3.0, 4.0], type=pa.float64())})
        results = _run_and_collect(contract, data)
        assert results[0].status == "PASSED"

    def test_max_check_passes(self) -> None:
        """MaxCheck: column maximum satisfies validator."""
        contract = self._contract_with_column_check(
            MaxCheck(name="Col max", validators=(MaxValidator(threshold=10.0),))
        )
        data = pa.table({"val": pa.array([1.0, 5.0, 9.0], type=pa.float64())})
        results = _run_and_collect(contract, data)
        assert results[0].status == "PASSED"

    def test_mean_check_passes(self) -> None:
        """MeanCheck: column mean satisfies validator."""
        contract = self._contract_with_column_check(
            MeanCheck(name="Col mean", validators=(EqualsValidator(value=3.0),))
        )
        data = pa.table({"val": pa.array([1.0, 3.0, 5.0], type=pa.float64())})
        results = _run_and_collect(contract, data)
        assert results[0].status == "PASSED"

    def test_sum_check_passes(self) -> None:
        """SumCheck: column sum satisfies validator."""
        contract = self._contract_with_column_check(SumCheck(name="Col sum", validators=(EqualsValidator(value=6.0),)))
        data = pa.table({"val": pa.array([1.0, 2.0, 3.0], type=pa.float64())})
        results = _run_and_collect(contract, data)
        assert results[0].status == "PASSED"

    def test_count_check_passes(self) -> None:
        """CountCheck: non-null count satisfies validator."""
        contract = self._contract_with_column_check(
            CountCheck(name="Non-null count", validators=(EqualsValidator(value=2.0),))
        )
        # 2 non-null out of 3 rows
        data = pa.table({"val": pa.array([1.0, None, 3.0], type=pa.float64())})
        results = _run_and_collect(contract, data)
        assert results[0].status == "PASSED"

    def test_variance_check_passes(self) -> None:
        """VarianceCheck: column variance satisfies validator."""
        contract = self._contract_with_column_check(
            VarianceCheck(name="Variance", validators=(MinValidator(threshold=0.0),))
        )
        data = pa.table({"val": pa.array([1.0, 2.0, 3.0, 4.0, 5.0], type=pa.float64())})
        results = _run_and_collect(contract, data)
        assert results[0].status == "PASSED"

    def test_cardinality_check_passes(self) -> None:
        """CardinalityCheck: unique count satisfies validator."""
        contract = self._contract_with_column_check(
            CardinalityCheck(name="Unique vals", validators=(EqualsValidator(value=3.0),))
        )
        data = pa.table({"val": pa.array([1.0, 2.0, 3.0, 1.0, 2.0], type=pa.float64())})
        results = _run_and_collect(contract, data)
        assert results[0].status == "PASSED"

    def test_stddev_check_passes(self) -> None:
        """StddevCheck: column stddev (via custom SQL) satisfies validator."""
        contract = self._contract_with_column_check(
            StddevCheck(name="Stddev", validators=(MinValidator(threshold=0.0),))
        )
        data = pa.table({"val": pa.array([1.0, 2.0, 3.0, 4.0, 5.0], type=pa.float64())})
        results = _run_and_collect(contract, data)
        assert results[0].status == "PASSED"


# ---------------------------------------------------------------------------
# TestMultipleChecksInOneContract
# ---------------------------------------------------------------------------


class TestMultipleChecksInOneContract:
    """Tests that table + column checks all run together under one @check."""

    def test_table_and_column_checks_all_in_one_check_node(self) -> None:
        """Table-level and column-level checks all share the same check name."""
        contract = _minimal_contract(
            name="Combined Contract",
            columns=(
                ColumnSpec(
                    name="val",
                    type="float",
                    description="Value",
                    nullable=True,
                    checks=(MissingCheck(name="No nulls", validators=(MaxValidator(threshold=0.0),)),),
                ),
            ),
            checks=(NumRowsCheck(name="Row count", validators=(MinValidator(threshold=1.0),)),),
        )
        data = pa.table({"val": pa.array([1.0, 2.0, 3.0], type=pa.float64())})
        results = _run_and_collect(contract, data)
        assert len(results) == 2
        assert all(r.check == "Combined Contract" for r in results)

    def test_assertion_names_are_distinct_per_check(self) -> None:
        """Different checks produce distinct assertion names."""
        contract = _minimal_contract(
            columns=(
                ColumnSpec(
                    name="val",
                    type="float",
                    description="Value",
                    nullable=True,
                    checks=(MissingCheck(name="No nulls", validators=(MaxValidator(threshold=0.0),)),),
                ),
            ),
            checks=(NumRowsCheck(name="Row count", validators=(MinValidator(threshold=1.0),)),),
        )
        data = pa.table({"val": pa.array([1.0, 2.0], type=pa.float64())})
        results = _run_and_collect(contract, data)
        assertion_names = {r.assertion for r in results}
        assert "Row count [min >= 1.0]" in assertion_names
        assert "No nulls [max <= 0.0]" in assertion_names

    def test_multiple_columns_with_checks(self) -> None:
        """Checks on multiple columns all land in one @check node."""
        contract = _minimal_contract(
            columns=(
                ColumnSpec(
                    name="a",
                    type="int",
                    description="A",
                    nullable=False,
                    checks=(MinCheck(name="A min", validators=(MinValidator(threshold=0),)),),
                ),
                ColumnSpec(
                    name="b",
                    type="int",
                    description="B",
                    nullable=False,
                    checks=(MaxCheck(name="B max", validators=(MaxValidator(threshold=100),)),),
                ),
            ),
        )
        data = pa.table({"a": [1, 2, 3], "b": [10, 20, 30]})
        results = _run_and_collect(contract, data)
        assert len(results) == 2
        assert all(r.check == "Test Contract" for r in results)
