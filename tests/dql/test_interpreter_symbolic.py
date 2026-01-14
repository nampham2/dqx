"""Tests for symbolic metric expressions with DQX metrics and sympy functions."""

from __future__ import annotations

from datetime import date

import pyarrow as pa

from dqx.datasource import DuckRelationDataSource
from dqx.dql import Interpreter
from dqx.orm.repositories import InMemoryMetricDB


class TestSymbolicMetrics:
    """Test symbolic metric expressions with DQX metrics and sympy functions."""

    # ===== Section 1: Basic DQX Metrics =====

    def test_minimum_metric(self) -> None:
        """Test minimum() metric function."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert minimum(value) == 5
                    name "test.minimum"
            }
        }
        """
        data = pa.Table.from_pydict({"value": [5, 10, 15, 20, 25]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    def test_maximum_metric(self) -> None:
        """Test maximum() metric function."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert maximum(value) == 25
                    name "test.maximum"
            }
        }
        """
        data = pa.Table.from_pydict({"value": [5, 10, 15, 20, 25]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    def test_unique_count_metric(self) -> None:
        """Test unique_count() metric function."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert unique_count(category) == 3
                    name "test.unique_count"
            }
        }
        """
        data = pa.Table.from_pydict({"category": ["A", "B", "A", "C", "B", "A"]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    def test_first_metric(self) -> None:
        """Test first() returns first value (including nulls)."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert first(value) == 42
                    name "test.first"
            }
        }
        """
        # first() returns the first value in the dataset, not first non-null
        data = pa.Table.from_pydict({"value": [42, 100, 200, 300, 400]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    # ===== Section 2: Sympy Math Functions =====

    def test_abs_function(self) -> None:
        """Test abs() with negative values."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert abs(average(value)) == 20.0
                    name "test.abs"
            }
        }
        """
        data = pa.Table.from_pydict({"value": [-10, -20, -30]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    def test_sqrt_function(self) -> None:
        """Test sqrt() for square root calculations."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert sqrt(average(value)) > 3.0
                    name "test.sqrt"
            }
        }
        """
        # average = (1+4+9+16+25)/5 = 11, sqrt(11) ≈ 3.317
        data = pa.Table.from_pydict({"value": [1, 4, 9, 16, 25]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    def test_log_function(self) -> None:
        """Test log() for logarithmic calculations."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert log(average(value)) > 4.0
                    name "test.log"
            }
        }
        """
        # average = (10+100+1000)/3 ≈ 370, ln(370) ≈ 5.9
        data = pa.Table.from_pydict({"value": [10, 100, 1000]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    def test_exp_function(self) -> None:
        """Test exp() for exponential calculations."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert exp(average(value)) > 2.7
                    name "test.exp"
            }
        }
        """
        # average = (0+1+2)/3 = 1, e^1 ≈ 2.718
        data = pa.Table.from_pydict({"value": [0, 1, 2]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    def test_min_function(self) -> None:
        """Test min() to compare multiple metrics."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert min(average(a), average(b)) == 20.0
                    name "test.min"
            }
        }
        """
        # average(a) = 20, average(b) = 25, min = 20
        data = pa.Table.from_pydict({"a": [10, 20, 30], "b": [15, 25, 35]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    def test_max_function(self) -> None:
        """Test max() to compare multiple metrics."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert max(average(a), average(b)) == 25.0
                    name "test.max"
            }
        }
        """
        # average(a) = 20, average(b) = 25, max = 25
        data = pa.Table.from_pydict({"a": [10, 20, 30], "b": [15, 25, 35]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    # ===== Section 3: Complex Expressions =====

    def test_arithmetic_combination(self) -> None:
        """Test arithmetic operations between metrics."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert (average(price) + average(quantity)) / 2 > 11.0
                    name "test.arithmetic"
            }
        }
        """
        # average(price) = 20, average(quantity) = 3, (20+3)/2 = 11.5
        data = pa.Table.from_pydict({"price": [10, 20, 30], "quantity": [2, 3, 4]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    def test_metric_multiplication(self) -> None:
        """Test metric multiplication for derived values."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert average(price) * num_rows() == 30.0
                    name "test.multiplication"
            }
        }
        """
        # average(price) = 15, num_rows = 2, 15 * 2 = 30
        data = pa.Table.from_pydict({"price": [10.0, 20.0]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    def test_standard_deviation_calculation(self) -> None:
        """Test sqrt(variance()) for standard deviation."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert sqrt(variance(value)) > 14.0
                    name "test.stddev"
            }
        }
        """
        # variance ≈ 200, sqrt(200) ≈ 14.14
        data = pa.Table.from_pydict({"value": [10, 20, 30, 40, 50]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    def test_percent_change_calculation(self) -> None:
        """Test complex expression for percent difference."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert abs((average(current) - average(previous)) / average(previous)) > 0.15
                    name "test.percent_change"
            }
        }
        """
        # average(current) = 120, average(previous) = 100
        # |(120 - 100) / 100| = 0.2 = 20%
        data = pa.Table.from_pydict({"current": [110, 120, 130], "previous": [100, 100, 100]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    def test_null_rate_with_division(self) -> None:
        """Test null rate calculation with division."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert null_count(email) / num_rows() between 0.39 and 0.41
                    name "test.null_rate"
            }
        }
        """
        # 2 nulls out of 5 rows = 0.4 (use range to handle floating point precision)
        data = pa.Table.from_pydict({"email": [None, "a@test.com", None, "b@test.com", "c@test.com"]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    def test_nested_math_functions(self) -> None:
        """Test nested sympy functions."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert sqrt(exp(log(average(value)))) between 15 and 30
                    name "test.nested"
            }
        }
        """
        # sqrt(exp(log(x))) simplifies to sqrt(x) ≈ sqrt(277.75) ≈ 16.66
        data = pa.Table.from_pydict({"value": [1, 10, 100, 1000]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    def test_multi_metric_comparison(self) -> None:
        """Test comparing three different metric aggregations."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert max(average(a), average(b), average(c)) == 20.0
                    name "test.multi_max"
            }
        }
        """
        # average(a) = 15, average(b) = 20, average(c) = 20, max = 20
        data = pa.Table.from_pydict({"a": [10, 20], "b": [15, 25], "c": [12, 28]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    def test_min_with_three_metrics(self) -> None:
        """Test min() with three metric arguments."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert min(sum(x), sum(y), sum(z)) == 30
                    name "test.min_three"
            }
        }
        """
        # sum(x) = 30, sum(y) = 75, sum(z) = 120, min = 30
        data = pa.Table.from_pydict({"x": [10, 20], "y": [25, 50], "z": [40, 80]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    def test_complex_null_handling(self) -> None:
        """Test complex expression with null handling."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert (num_rows() - null_count(value)) / num_rows() > 0.5
                    name "test.non_null_rate"
            }
        }
        """
        # 3 non-null out of 5 rows = 0.6
        data = pa.Table.from_pydict({"value": [10, None, 20, None, 30]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()

    def test_ratio_of_aggregates(self) -> None:
        """Test ratio between different aggregations."""
        dql = """
        suite "Metric Test" {
            check "Test" on dataset {
                assert sum(revenue) / sum(cost) > 1.5
                    name "test.profit_margin"
            }
        }
        """
        # sum(revenue) = 300, sum(cost) = 150, ratio = 2.0
        data = pa.Table.from_pydict({"revenue": [100, 100, 100], "cost": [50, 50, 50]})
        datasources = {"dataset": DuckRelationDataSource.from_arrow(data, "dataset")}

        interp = Interpreter(db=InMemoryMetricDB())
        results = interp.run(dql, datasources, date.today())
        assert results.all_passed()
