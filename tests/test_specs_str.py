"""Tests for __str__ methods in specs module."""

import pytest

from dqx.specs import (
    Average,
    First,
    Maximum,
    Minimum,
    NegativeCount,
    NullCount,
    NumRows,
    Sum,
    Variance,
)


class TestMetricSpecStrMethods:
    """Test __str__ methods for all MetricSpec classes."""

    def test_num_rows_str(self) -> None:
        """Test NumRows.__str__ returns the name."""
        metric = NumRows()
        assert str(metric) == "num_rows()"

    def test_first_str(self) -> None:
        """Test First.__str__ returns the name with column."""
        metric = First("user_id")
        assert str(metric) == "first(user_id)"

    def test_average_str(self) -> None:
        """Test Average.__str__ returns the name with column."""
        metric = Average("price")
        assert str(metric) == "average(price)"

    def test_variance_str(self) -> None:
        """Test Variance.__str__ returns the name with column."""
        metric = Variance("amount")
        assert str(metric) == "variance(amount)"

    def test_minimum_str(self) -> None:
        """Test Minimum.__str__ returns the name with column."""
        metric = Minimum("score")
        assert str(metric) == "minimum(score)"

    def test_maximum_str(self) -> None:
        """Test Maximum.__str__ returns the name with column."""
        metric = Maximum("revenue")
        assert str(metric) == "maximum(revenue)"

    def test_sum_str(self) -> None:
        """Test Sum.__str__ returns the name with column."""
        metric = Sum("total")
        assert str(metric) == "sum(total)"

    def test_null_count_str(self) -> None:
        """Test NullCount.__str__ returns the name with column."""
        metric = NullCount("email")
        assert str(metric) == "null_count(email)"

    def test_negative_count_str(self) -> None:
        """Test NegativeCount.__str__ returns the name with column."""
        metric = NegativeCount("balance")
        assert str(metric) == "non_negative(balance)"

    @pytest.mark.parametrize(
        "metric_class,column,expected",
        [
            (NumRows, None, "num_rows()"),
            (First, "col1", "first(col1)"),
            (Average, "col2", "average(col2)"),
            (Variance, "col3", "variance(col3)"),
            (Minimum, "col4", "minimum(col4)"),
            (Maximum, "col5", "maximum(col5)"),
            (Sum, "col6", "sum(col6)"),
            (NullCount, "col7", "null_count(col7)"),
            (NegativeCount, "col8", "non_negative(col8)"),
        ],
    )
    def test_all_metrics_str(self, metric_class: type, column: str | None, expected: str) -> None:
        """Parametrized test for all metric __str__ methods."""
        if column is None:
            metric = metric_class()
        else:
            metric = metric_class(column)
        assert str(metric) == expected
