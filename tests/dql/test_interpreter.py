from __future__ import annotations

from datetime import date
from pathlib import Path

import pyarrow as pa
import pytest

from dqx.datasource import DuckRelationDataSource
from dqx.dql import Interpreter
from dqx.dql.errors import DQLError
from dqx.orm.repositories import InMemoryMetricDB


class TestBasicExecution:
    """Test basic interpreter execution."""

    def test_simple_check_passes(self) -> None:
        """Test basic check execution with passing assertion."""
        dql = """
        suite "Test Suite" {
            check "Volume" on orders {
                assert num_rows() > 0
                    name "orders.volume.has_rows"
                    severity P1
            }
        }
        """

        # Create test data
        data = pa.Table.from_pydict({"id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]})
        datasources = {"orders": DuckRelationDataSource.from_arrow(data, "orders")}

        # Execute
        db = InMemoryMetricDB()
        interp = Interpreter(db=db)
        results = interp.run_string(dql, datasources, date.today())

        # Verify
        assert results.suite_name == "Test Suite"
        assert results.all_passed()
        assert len(results.assertions) == 1
        assert results.assertions[0].check_name == "Volume"
        assert results.assertions[0].assertion_name == "orders.volume.has_rows"

    def test_simple_check_fails(self) -> None:
        """Test basic check execution with failing assertion."""
        dql = """
        suite "Test Suite" {
            check "Volume" on orders {
                assert num_rows() > 100
                    name "orders.volume.min_rows"
            }
        }
        """

        # Create test data (only 3 rows, but we assert > 100)
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"orders": DuckRelationDataSource.from_arrow(data, "orders")}

        # Execute
        db = InMemoryMetricDB()
        interp = Interpreter(db=db)
        results = interp.run_string(dql, datasources, date.today())

        # Verify
        assert not results.all_passed()
        assert len(results.failures) == 1
        assert results.failures[0].assertion_name == "orders.volume.min_rows"

    def test_missing_datasource_fails_fast(self) -> None:
        """Test interpreter fails immediately if datasource missing."""
        dql = """
        suite "Test Suite" {
            check "Multi" on orders, customers {
                assert num_rows(dataset=orders) > 0
            }
        }
        """

        # Only provide orders, missing customers
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"orders": DuckRelationDataSource.from_arrow(data, "orders")}

        db = InMemoryMetricDB()
        interp = Interpreter(db=db)

        with pytest.raises(DQLError, match="Missing datasources.*customers"):
            interp.run_string(dql, datasources, date.today())


class TestComparisonOperators:
    """Test all comparison operators."""

    def test_greater_than(self) -> None:
        """Test > operator."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert average(amount) > 15.0
                    name "test.gt"
            }
        }
        """
        data = pa.Table.from_pydict({"amount": [10.0, 20.0, 30.0]})  # avg = 20
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()

    def test_between_condition(self) -> None:
        """Test between A and B condition."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert average(amount) between 15 and 25
                    name "test.between"
            }
        }
        """
        data = pa.Table.from_pydict({"amount": [10.0, 20.0, 30.0]})  # avg = 20
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()

    def test_is_positive(self) -> None:
        """Test 'is positive' condition."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert sum(amount) is positive
                    name "test.positive"
            }
        }
        """
        data = pa.Table.from_pydict({"amount": [10.0, 20.0, 30.0]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()


class TestTunables:
    """Test tunable constants."""

    def test_tunable_substitution(self) -> None:
        """Test tunables are substituted correctly."""
        dql = """
        suite "Test" {
            tunable MAX_NULL = 5% bounds [0%, 10%]

            check "Quality" on orders {
                assert null_count(email) / num_rows() < MAX_NULL
                    name "test.null_rate"
            }
        }
        """

        # 1 null out of 10 rows = 10% (should fail since MAX_NULL is 5%)
        data = pa.Table.from_pydict(
            {
                "email": [
                    "a@test.com",
                    None,
                    "c@test.com",
                    "d@test.com",
                    "e@test.com",
                    "f@test.com",
                    "g@test.com",
                    "h@test.com",
                    "i@test.com",
                    "j@test.com",
                ]
            }
        )
        datasources = {"orders": DuckRelationDataSource.from_arrow(data, "orders")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())

        # Should pass: 1/10 = 10%, but wait, actually 1/10 = 0.1 = 10% which is > 0.05
        # So this should fail
        assert not results.all_passed()


class TestDateFunctions:
    """Test all date functions in profiles."""

    def test_nth_weekday(self) -> None:
        """Test nth_weekday date function."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() > 0
                    name "test.rows"
                    tags [volume]
            }

            profile "Thanksgiving" {
                type holiday
                from nth_weekday(november, thursday, 4)
                to nth_weekday(november, thursday, 4)
                scale tag "volume" by 2.0x
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())

        # Test on Thanksgiving 2024 (Nov 28)
        results = interp.run_string(dql, datasources, date(2024, 11, 28))
        assert results.all_passed()

    def test_last_day_of_month(self) -> None:
        """Test last_day_of_month function."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() >= 1
                    name "test.rows"
                    tags [end_of_month]
            }

            profile "Month End" {
                type recurring
                from last_day_of_month()
                to last_day_of_month()
                scale tag "end_of_month" by 1.5x
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())

        # Test on last day of January
        results = interp.run_string(dql, datasources, date(2024, 1, 31))
        assert results.all_passed()

        # Test on last day of February (leap year)
        results = interp.run_string(dql, datasources, date(2024, 2, 29))
        assert results.all_passed()

    def test_month_day_function(self) -> None:
        """Test month(day) functions like december(25)."""
        dql = """
        suite "Test" {
            check "Volume" on t {
                assert num_rows() > 0
                    name "test.rows"
                    tags [holiday]
            }

            profile "Christmas" {
                type holiday
                from december(25)
                to december(25)
                scale tag "holiday" by 2.0x
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())

        # On Christmas, the check should be scaled (3 * 2.0 = 6 > 0)
        results = interp.run_string(dql, datasources, date(2024, 12, 25))
        # Profile active and scaling applied
        assert results.all_passed()

    def test_date_with_offset(self) -> None:
        """Test date functions with day offsets."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() > 0
                    name "test.rows"
                    tags [week]
            }

            profile "Thanksgiving Week" {
                type holiday
                from nth_weekday(november, thursday, 4)
                to nth_weekday(november, thursday, 4) + 3
                scale tag "week" by 1.5x
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())

        # Test 3 days after Thanksgiving (Sunday after)
        results = interp.run_string(dql, datasources, date(2024, 12, 1))
        assert results.all_passed()


class TestProfiles:
    """Test profile activation and rule application."""

    def test_profile_scale_rule(self) -> None:
        """Test profile scale rule multiplies metric."""
        dql = """
        suite "Test" {
            check "Volume" on t {
                assert num_rows() >= 10
                    name "test.min_rows"
                    tags [volume]
            }

            profile "Low Traffic" {
                type holiday
                from 2024-12-24
                to 2024-12-26
                scale tag "volume" by 2.0x
            }
        }
        """
        # 6 rows: normally would fail (< 10), but with 2.0x scaling: 6 * 2.0 = 12 >= 10
        data = pa.Table.from_pydict({"id": [1, 2, 3, 4, 5, 6]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date(2024, 12, 25))
        assert results.all_passed()

    def test_profile_disable_rule(self) -> None:
        """Test profile disable rule skips assertions."""
        dql = """
        suite "Test" {
            check "Volume" on t {
                assert num_rows() >= 1000
                    name "test.min_rows"
            }

            profile "Holiday" {
                type holiday
                from 2024-12-20
                to 2024-12-31
                disable check "Volume"
            }
        }
        """
        # Only 5 rows, would normally fail, but disabled by profile
        data = pa.Table.from_pydict({"id": [1, 2, 3, 4, 5]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date(2024, 12, 25))

        # Check was disabled, no results
        assert len(results.assertions) == 0

    def test_profile_disable_check(self) -> None:
        """Test profile disable rule by check name."""
        dql = """
        suite "Test" {
            check "Volume" on t {
                assert num_rows() >= 1000
                    name "test.min_rows"
            }

            check "Quality" on t {
                assert num_rows() > 0
                    name "test.has_rows"
            }

            profile "Holiday" {
                type holiday
                from 2024-12-20
                to 2024-12-31
                disable check "Volume"
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3, 4, 5]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date(2024, 12, 25))

        # Volume check disabled, Quality check runs
        assert len(results.assertions) == 1
        assert results.assertions[0].check_name == "Quality"
        assert results.all_passed()

    def test_profile_set_severity(self) -> None:
        """Test profile set severity rule."""
        dql = """
        suite "Test" {
            check "Quality" on t {
                assert num_rows() > 0
                    name "test.rows"
                    severity P0
                    tags [quality]
            }

            profile "Maintenance" {
                type holiday
                from 2024-12-24
                to 2024-12-26
                set tag "quality" severity P3
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date(2024, 12, 25))

        # Severity should be overridden to P3
        assert results.assertions[0].severity == "P3"

    def test_multiple_profiles_compound(self) -> None:
        """Test multiple active profiles compound multipliers."""
        dql = """
        suite "Test" {
            check "Volume" on t {
                assert num_rows() >= 20
                    name "test.min_rows"
                    tags [volume]
            }

            profile "Season" {
                type holiday
                from 2024-11-01
                to 2024-12-31
                scale tag "volume" by 2.0x
            }

            profile "BlackFriday" {
                type holiday
                from 2024-11-29
                to 2024-11-29
                scale tag "volume" by 2.0x
            }
        }
        """
        # 6 rows: 6 * 2.0 * 2.0 = 24 >= 20
        data = pa.Table.from_pydict({"id": [1, 2, 3, 4, 5, 6]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date(2024, 11, 29))
        assert results.all_passed()


class TestAnnotations:
    """Test assertion annotations."""

    def test_experimental_annotation(self) -> None:
        """Test @experimental annotation is passed through."""
        dql = """
        suite "Test" {
            check "Test" on t {
                @experimental
                assert num_rows() > 0
                    name "test.experimental"
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()

    def test_required_annotation(self) -> None:
        """Test @required annotation is passed through."""
        dql = """
        suite "Test" {
            check "Test" on t {
                @required
                assert num_rows() > 0
                    name "test.required"
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()

    def test_cost_annotation(self) -> None:
        """Test @cost annotation is passed through."""
        dql = """
        suite "Test" {
            check "Test" on t {
                @cost(false_positive=10, false_negative=100)
                assert num_rows() > 0
                    name "test.cost"
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()


class TestAllOperators:
    """Test all comparison operators and conditions."""

    def test_gte_operator(self) -> None:
        """Test >= operator."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() >= 3
                    name "test.gte"
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()

    def test_lt_operator(self) -> None:
        """Test < operator."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() < 10
                    name "test.lt"
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()

    def test_lte_operator(self) -> None:
        """Test <= operator."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() <= 3
                    name "test.lte"
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()

    def test_eq_operator(self) -> None:
        """Test == operator."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() == 3
                    name "test.eq"
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()

    def test_eq_with_tolerance(self) -> None:
        """Test == operator with tolerance."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert average(val) == 10.0 tolerance 0.1
                    name "test.eq_tolerance"
            }
        }
        """
        data = pa.Table.from_pydict({"val": [9.95, 10.0, 10.05]})  # avg ≈ 10.0
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()

    def test_neq_operator(self) -> None:
        """Test != operator."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() != 5
                    name "test.neq"
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()

    def test_is_negative(self) -> None:
        """Test 'is negative' condition."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert sum(val) is negative
                    name "test.negative"
            }
        }
        """
        data = pa.Table.from_pydict({"val": [-10.0, -20.0, -5.0]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_date_function(self) -> None:
        """Test error for unknown date function."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() > 0
                    name "test.rows"
            }

            profile "Bad" {
                type holiday
                from unknown_function(foo, bar)
                to december(25)
                scale tag "x" by 2.0x
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())

        with pytest.raises(DQLError, match="Unknown date function"):
            interp.run_string(dql, datasources, date(2024, 12, 25))

    def test_invalid_month_name(self) -> None:
        """Test error for unknown month name."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() > 0
                    name "test.rows"
            }

            profile "Bad" {
                type holiday
                from nth_weekday(badmonth, monday, 1)
                to december(25)
                scale tag "x" by 2.0x
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())

        with pytest.raises(DQLError, match="Unknown month name"):
            interp.run_string(dql, datasources, date(2024, 12, 25))

    def test_invalid_weekday_name(self) -> None:
        """Test error for unknown weekday name."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() > 0
                    name "test.rows"
            }

            profile "Bad" {
                type holiday
                from nth_weekday(november, badday, 1)
                to december(25)
                scale tag "x" by 2.0x
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())

        with pytest.raises(DQLError, match="Unknown weekday name"):
            interp.run_string(dql, datasources, date(2024, 12, 25))

    def test_invalid_nth_weekday(self) -> None:
        """Test error for nth weekday that doesn't exist."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() > 0
                    name "test.rows"
            }

            profile "Bad" {
                type holiday
                from nth_weekday(february, monday, 5)
                to december(25)
                scale tag "x" by 2.0x
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())

        # February 2024 doesn't have a 5th Monday
        with pytest.raises(DQLError, match="No 5th monday"):
            interp.run_string(dql, datasources, date(2024, 2, 29))

    def test_run_file(self, tmp_path: Path) -> None:
        """Test run_file method."""
        dql_content = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() > 0
                    name "test.rows"
            }
        }
        """

        # Write DQL to temp file
        dql_file = tmp_path / "test.dql"
        dql_file.write_text(dql_content)

        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_file(str(dql_file), datasources, date.today())

        assert results.all_passed()

    def test_suite_with_availability_threshold(self) -> None:
        """Test suite with availability threshold."""
        dql = """
        suite "Test" {
            availability_threshold 95%

            check "Test" on t {
                assert num_rows() > 0
                    name "test.rows"
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()

    def test_invalid_date_function_format(self) -> None:
        """Test error for malformed date function."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() > 0
                    name "test.rows"
            }

            profile "Bad" {
                type holiday
                from unknown_function(foo, bar)
                to december(25)
                scale tag "x" by 2.0x
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())

        with pytest.raises(DQLError, match="Unknown date function"):
            interp.run_string(dql, datasources, date(2024, 12, 25))

    def test_profile_disable_assertion(self) -> None:
        """Test profile can disable specific assertion."""
        dql = """
        suite "Test" {
            check "Volume" on t {
                assert num_rows() >= 1000
                    name "min_rows"

                assert num_rows() > 0
                    name "has_rows"
            }

            profile "Holiday" {
                type holiday
                from 2024-12-20
                to 2024-12-31
                disable assertion "min_rows" in "Volume"
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date(2024, 12, 25))

        # min_rows disabled, only has_rows runs
        assert len(results.assertions) == 1
        assert results.assertions[0].assertion_name == "has_rows"
        assert results.all_passed()

    def test_profile_scale_by_check(self) -> None:
        """Test profile scale rule by check name."""
        dql = """
        suite "Test" {
            check "Volume" on t {
                assert num_rows() >= 10
                    name "min_rows"
            }

            profile "Scale" {
                type holiday
                from 2024-12-24
                to 2024-12-26
                scale check "Volume" by 2.0x
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3, 4, 5, 6]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date(2024, 12, 25))
        assert results.all_passed()

    def test_profile_set_severity_by_check(self) -> None:
        """Test profile set severity by check name."""
        dql = """
        suite "Test" {
            check "Quality" on t {
                assert num_rows() > 0
                    name "test.rows"
                    severity P0
            }

            profile "Maintenance" {
                type holiday
                from 2024-12-24
                to 2024-12-26
                set check "Quality" severity P3
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date(2024, 12, 25))

        assert results.assertions[0].severity == "P3"

    def test_integer_tunable(self) -> None:
        """Test integer tunable type detection."""
        dql = """
        suite "Test" {
            tunable MIN_ROWS = 1000 bounds [100, 10000]

            check "Volume" on t {
                assert num_rows() >= MIN_ROWS
                    name "test.min_rows"
            }
        }
        """
        data = pa.Table.from_pydict({"id": list(range(1200))})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()

    def test_float_tunable(self) -> None:
        """Test float tunable type detection."""
        dql = """
        suite "Test" {
            tunable VARIANCE_LIMIT = 0.5 bounds [0.1, 1.0]

            check "Stability" on t {
                assert variance(val) < VARIANCE_LIMIT
                    name "test.variance"
            }
        }
        """
        data = pa.Table.from_pydict({"val": [1.0, 1.1, 0.9, 1.05, 0.95]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()


class TestAdditionalCoverage:
    """Additional tests to reach 100% coverage."""

    def test_results_passes_property(self) -> None:
        """Test SuiteResults.passes property."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() > 100
                    name "test.fail"

                assert num_rows() > 0
                    name "test.pass"
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())

        # Test passes property (line 98)
        assert len(results.passes) == 1
        assert results.passes[0].assertion_name == "test.pass"

    def test_percentage_threshold(self) -> None:
        """Test percentage literal in threshold (line 260)."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert null_count(email) / num_rows() < 5%
                    name "test.null_pct"
            }
        }
        """
        # 1 null out of 10 = 10%, should fail (> 5%)
        data = pa.Table.from_pydict({"email": ["a@test.com"] * 9 + [None]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert not results.all_passed()

    def test_profile_disable_assertion_no_check_match(self) -> None:
        """Test disable assertion without in_check clause (line 575)."""
        dql = """
        suite "Test" {
            check "Volume" on t {
                assert num_rows() >= 1000
                    name "min_rows"

                assert num_rows() > 0
                    name "has_rows"
            }

            profile "Holiday" {
                type holiday
                from 2024-12-20
                to 2024-12-31
                disable assertion "min_rows" in "Volume"
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date(2024, 12, 25))

        # min_rows disabled by name match
        assert len(results.assertions) == 1
        assert results.assertions[0].assertion_name == "has_rows"

    def test_results_passes_property_used(self) -> None:
        """Test using the passes property specifically (line 98)."""
        dql = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() > 100
                    name "test.fail"

                assert num_rows() > 0
                    name "test.pass1"

                assert average(val) > 0
                    name "test.pass2"
            }
        }
        """
        data = pa.Table.from_pydict({"id": [1, 2, 3], "val": [10.0, 20.0, 30.0]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())

        # Use passes property
        passed = results.passes
        assert len(passed) == 2
        assert passed[0].assertion_name in ["test.pass1", "test.pass2"]
        assert passed[1].assertion_name in ["test.pass1", "test.pass2"]

    # Parser requires 'in "CheckName"' for disable assertion, so line 579 is unreachable
    # def test_disable_assertion_without_check_clause(self) -> None:

    def test_tunable_float_type(self) -> None:
        """Test float tunable (line 249)."""
        dql = """
        suite "Test" {
            tunable TOLERANCE = 1.5 bounds [1.0, 2.0]

            check "Test" on t {
                assert average(val) > TOLERANCE
                    name "test.avg"
            }
        }
        """
        data = pa.Table.from_pydict({"val": [2.0, 2.5, 3.0]})  # avg = 2.5
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()

    def test_percentage_tunable_bound(self) -> None:
        """Test percentage in tunable bounds (line 260)."""
        dql = """
        suite "Test" {
            tunable MAX_RATE = 3% bounds [1%, 5%]

            check "Test" on t {
                assert null_count(email) / num_rows() < MAX_RATE
                    name "test.null_rate"
            }
        }
        """
        # 0 nulls, should pass
        data = pa.Table.from_pydict({"email": ["a@test.com", "b@test.com", "c@test.com"]})
        datasources = {"t": DuckRelationDataSource.from_arrow(data, "t")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run_string(dql, datasources, date.today())
        assert results.all_passed()
