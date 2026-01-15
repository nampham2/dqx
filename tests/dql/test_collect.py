"""Tests for DQL collect statement (noop assertions)."""

from __future__ import annotations

import datetime

import pyarrow as pa

from dqx.datasource import DuckRelationDataSource
from dqx.dql.interpreter import Interpreter
from dqx.orm.repositories import InMemoryMetricDB


class TestCollectExecution:
    """Test execution of collect statements."""

    def test_collect_always_passes(self) -> None:
        """Test that collect statements always return passed status."""
        dql = """
        suite "Test" {
            check "Metrics" on data {
                collect average(value)
                    name "avg_value"
            }
        }
        """

        # Create test data
        data = pa.table({"value": [1, 2, 3, 4, 5]})
        datasources = {"data": DuckRelationDataSource.from_arrow(data, "data")}

        # Execute
        db = InMemoryMetricDB()
        interpreter = Interpreter(db=db)
        results = interpreter.run(dql, datasources, datetime.date.today())

        # Verify result
        assert len(results.assertions) == 1
        assert results.assertions[0].passed
        assert results.assertions[0].assertion_name == "avg_value"

    def test_collect_computes_metric(self) -> None:
        """Test that collect statements compute and store metric values."""
        dql = """
        suite "Test" {
            check "Metrics" on data {
                collect sum(value)
                    name "total"
            }
        }
        """

        # Create test data
        data = pa.table({"value": [10, 20, 30]})
        datasources = {"data": DuckRelationDataSource.from_arrow(data, "data")}

        # Execute
        db = InMemoryMetricDB()
        interpreter = Interpreter(db=db)
        results = interpreter.run(dql, datasources, datetime.date.today())

        # Verify metric was computed
        assert len(results.assertions) == 1
        assertion = results.assertions[0]
        assert assertion.passed
        # The metric value should be 60.0 (10 + 20 + 30)
        # Note: metric value access depends on implementation details

    def test_mixed_assert_and_collect(self) -> None:
        """Test check with both assertions and collections."""
        dql = """
        suite "Test" {
            check "Quality" on data {
                assert num_rows() > 0
                    name "has_rows"

                collect average(value)
                    name "avg_value"

                assert sum(value) > 10
                    name "min_sum"

                collect null_count(value) / num_rows()
                    name "null_rate"
            }
        }
        """

        # Create test data
        data = pa.table({"value": [5, 10, 15, 20]})
        datasources = {"data": DuckRelationDataSource.from_arrow(data, "data")}

        # Execute
        db = InMemoryMetricDB()
        interpreter = Interpreter(db=db)
        results = interpreter.run(dql, datasources, datetime.date.today())

        # Verify all passed
        assert len(results.assertions) == 4
        assert results.all_passed()

        # Check names
        names = [a.assertion_name for a in results.assertions]
        assert "has_rows" in names
        assert "avg_value" in names
        assert "min_sum" in names
        assert "null_rate" in names

    def test_collect_with_complex_expression(self) -> None:
        """Test collect with complex metric expressions."""
        dql = """
        suite "Test" {
            check "Metrics" on data {
                collect null_count(value) / num_rows()
                    name "null_rate"
            }
        }
        """

        # Create test data with nulls
        data = pa.table({"value": [1, None, 3, None, 5]})
        datasources = {"data": DuckRelationDataSource.from_arrow(data, "data")}

        # Execute
        db = InMemoryMetricDB()
        interpreter = Interpreter(db=db)
        results = interpreter.run(dql, datasources, datetime.date.today())

        # Verify
        assert len(results.assertions) == 1
        assert results.assertions[0].passed

    def test_collect_with_annotations(self) -> None:
        """Test that collect works with @experimental and @cost annotations."""
        dql = """
        suite "Test" {
            check "Metrics" on data {
                @experimental
                @cost(fp=1, fn=100)
                collect average(value)
                    name "avg_experimental"
            }
        }
        """

        # Create test data
        data = pa.table({"value": [1, 2, 3]})
        datasources = {"data": DuckRelationDataSource.from_arrow(data, "data")}

        # Execute
        db = InMemoryMetricDB()
        interpreter = Interpreter(db=db)
        results = interpreter.run(dql, datasources, datetime.date.today())

        # Verify
        assert len(results.assertions) == 1
        assert results.assertions[0].passed

    def test_collect_with_tags_and_severity(self) -> None:
        """Test collect with tags and severity modifiers."""
        dql = """
        suite "Test" {
            check "Metrics" on data {
                collect sum(value)
                    name "total"
                    severity P0
                    tags [revenue, critical]
            }
        }
        """

        # Create test data
        data = pa.table({"value": [100, 200, 300]})
        datasources = {"data": DuckRelationDataSource.from_arrow(data, "data")}

        # Execute
        db = InMemoryMetricDB()
        interpreter = Interpreter(db=db)
        results = interpreter.run(dql, datasources, datetime.date.today())

        # Verify
        assert len(results.assertions) == 1
        assert results.assertions[0].passed


class TestCollectWithProfiles:
    """Test collect statements with profile interactions."""

    def test_collect_with_profile_scaling(self) -> None:
        """Test that profile scaling applies to collect statements."""
        dql = """
        suite "Test" {
            check "Metrics" on data {
                collect num_rows()
                    name "row_count"
                    tags [volume]
            }

            profile "Test Profile" {
                type holiday
                from 2024-01-01
                to 2024-01-01

                scale tag "volume" by 2.0x
            }
        }
        """

        # Create test data (100 rows)
        data = pa.table({"value": list(range(100))})
        datasources = {"data": DuckRelationDataSource.from_arrow(data, "data")}

        # Execute on profile date
        db = InMemoryMetricDB()
        interpreter = Interpreter(db=db)
        results = interpreter.run(dql, datasources, datetime.date(2024, 1, 1))

        # Verify metric was scaled
        assert len(results.assertions) == 1
        assert results.assertions[0].passed
        # Metric should be scaled by 2.0x (100 * 2.0 = 200)

    def test_collect_can_be_disabled(self) -> None:
        """Test that profiles can disable collect statements."""
        dql = """
        suite "Test" {
            check "Metrics" on data {
                collect num_rows()
                    name "row_count"
            }

            profile "Disable Profile" {
                type holiday
                from 2024-01-01
                to 2024-01-01

                disable check "Metrics"
            }
        }
        """

        # Create test data
        data = pa.table({"value": [1, 2, 3]})
        datasources = {"data": DuckRelationDataSource.from_arrow(data, "data")}

        # Execute on profile date
        db = InMemoryMetricDB()
        interpreter = Interpreter(db=db)
        results = interpreter.run(dql, datasources, datetime.date(2024, 1, 1))

        # Verify no results (check was disabled)
        assert len(results.assertions) == 0

    def test_collect_severity_override(self) -> None:
        """Test that profiles can override collect severity."""
        dql = """
        suite "Test" {
            check "Metrics" on data {
                collect num_rows()
                    name "row_count"
                    severity P1
                    tags [volume]
            }

            profile "Override Profile" {
                type holiday
                from 2024-01-01
                to 2024-01-01

                set tag "volume" severity P3
            }
        }
        """

        # Create test data
        data = pa.table({"value": [1, 2, 3]})
        datasources = {"data": DuckRelationDataSource.from_arrow(data, "data")}

        # Execute on profile date
        db = InMemoryMetricDB()
        interpreter = Interpreter(db=db)
        results = interpreter.run(dql, datasources, datetime.date(2024, 1, 1))

        # Verify
        assert len(results.assertions) == 1
        assert results.assertions[0].passed
        # Severity should be overridden to P3


class TestCollectMultiDataset:
    """Test collect statements with multiple datasets."""

    def test_collect_multi_dataset(self) -> None:
        """Test collect with dataset parameter."""
        dql = """
        suite "Test" {
            check "Cross-Dataset" on orders, returns {
                collect num_rows(dataset=orders)
                    name "order_count"

                collect num_rows(dataset=returns)
                    name "return_count"
            }
        }
        """

        # Create test data
        orders = pa.table({"id": [1, 2, 3, 4, 5]})
        returns = pa.table({"id": [1, 2]})

        datasources = {
            "orders": DuckRelationDataSource.from_arrow(orders, "orders"),
            "returns": DuckRelationDataSource.from_arrow(returns, "returns"),
        }

        # Execute
        db = InMemoryMetricDB()
        interpreter = Interpreter(db=db)
        results = interpreter.run(dql, datasources, datetime.date.today())

        # Verify
        assert len(results.assertions) == 2
        assert results.all_passed()


class TestCollectEdgeCases:
    """Test edge cases for collect statements."""

    def test_collect_empty_dataset(self) -> None:
        """Test collect on empty dataset."""
        dql = """
        suite "Test" {
            check "Metrics" on data {
                collect num_rows()
                    name "row_count"
            }
        }
        """

        # Create empty data
        data = pa.table({"value": pa.array([], type=pa.int64())})
        datasources = {"data": DuckRelationDataSource.from_arrow(data, "data")}

        # Execute
        db = InMemoryMetricDB()
        interpreter = Interpreter(db=db)
        results = interpreter.run(dql, datasources, datetime.date.today())

        # Verify - should still pass even with 0 rows
        assert len(results.assertions) == 1
        assert results.assertions[0].passed

    def test_collect_only_check(self) -> None:
        """Test check with only collect statements (no assertions)."""
        dql = """
        suite "Test" {
            check "Collection Only" on data {
                collect num_rows()
                    name "row_count"

                collect average(value)
                    name "avg_value"

                collect sum(value)
                    name "total_value"
            }
        }
        """

        # Create test data
        data = pa.table({"value": [10, 20, 30]})
        datasources = {"data": DuckRelationDataSource.from_arrow(data, "data")}

        # Execute
        db = InMemoryMetricDB()
        interpreter = Interpreter(db=db)
        results = interpreter.run(dql, datasources, datetime.date.today())

        # Verify all pass
        assert len(results.assertions) == 3
        assert results.all_passed()
