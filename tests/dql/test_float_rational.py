"""Additional tests to cover Float/Rational conversion paths.

These tests specifically target the rarely-hit Float/Rational type conversion
paths in the _convert_kwargs and _convert_value helper functions.
"""

from __future__ import annotations

from datetime import date

import pyarrow as pa

from dqx.datasource import DuckRelationDataSource
from dqx.dql import Interpreter
from dqx.orm.repositories import InMemoryMetricDB


class TestFloatRationalConversions:
    """Tests to hit Float/Rational sympy type conversion paths."""

    def test_lag_with_float_triggers_rational_conversion(self) -> None:
        """Test that might trigger Float/Rational in kwargs.

        This attempts to hit lines 525-526 in interpreter.py where
        value.is_Float or value.is_Rational returns True.
        """
        dql = """
        suite "Test" {
            check "Test" on orders {
                # Use lag parameter which gets converted through _convert_kwargs
                assert average(price, lag=1) > 0
                    name "test1"

                # Use n parameter in stddev
                assert stddev(average(price), n=7) < 1000
                    name "test2"

                # Use both offset and n
                assert stddev(average(tax), offset=1, n=5) < 100
                    name "test3"
            }
        }
        """
        data = pa.Table.from_pydict({"id": list(range(20)), "price": [100.0] * 20, "tax": [10.0] * 20})
        ds = DuckRelationDataSource.from_arrow(data, "orders")

        db = InMemoryMetricDB()
        interp = Interpreter(db=db)

        results = interp.run(dql, {"orders": ds}, date.today())
        assert len(results.assertions) == 3

    def test_count_values_with_various_types(self) -> None:
        """Test count_values to trigger _convert_value paths.

        This attempts to hit lines 570-574 where Float/Rational values
        are converted to int for count_values.
        """
        dql = """
        suite "Test" {
            check "Test" on orders {
                # Integer value
                assert count_values(quantity, 5) >= 0
                    name "int_value"

                # Another integer
                assert count_values(quantity, 0) >= 0
                    name "zero_value"

                # String value
                assert count_values(status, "active") >= 0
                    name "str_value"
            }
        }
        """
        data = pa.Table.from_pydict(
            {"quantity": [0, 5, 10, 5, 0], "status": ["active", "inactive", "active", "active", "inactive"]}
        )
        ds = DuckRelationDataSource.from_arrow(data, "orders")

        db = InMemoryMetricDB()
        interp = Interpreter(db=db)

        results = interp.run(dql, {"orders": ds}, date.today())
        assert len(results.assertions) == 3

    def test_complex_arithmetic_with_parameters(self) -> None:
        """Test complex expressions that might create Float/Rational sympy types."""
        dql = """
        suite "Test" {
            check "Test" on orders {
                # Complex expression with lag
                assert (average(price, lag=1) + average(price, lag=2)) / 2 > 0
                    name "avg_lag"

                # Multiple dataset parameters
                assert average(price, dataset=orders) > 0
                    name "dataset_param"
            }
        }
        """
        data = pa.Table.from_pydict({"id": list(range(10)), "price": [100.0] * 10})
        ds = DuckRelationDataSource.from_arrow(data, "orders")

        db = InMemoryMetricDB()
        interp = Interpreter(db=db)

        results = interp.run(dql, {"orders": ds}, date.today())
        assert len(results.assertions) == 2
