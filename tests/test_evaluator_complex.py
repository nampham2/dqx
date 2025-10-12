"""Tests for complex number handling in the Evaluator."""

from datetime import date
from unittest.mock import Mock

import sympy as sp
from returns.result import Failure, Success

from dqx.common import ResultKey
from dqx.evaluator import Evaluator
from dqx.provider import MetricProvider


class TestEvaluatorComplexNumberHandling:
    """Test complex number detection and handling in the Evaluator."""

    def test_sqrt_negative_returns_complex_failure(self) -> None:
        """Test that square root of negative value returns complex number failure."""
        # Setup provider with a negative metric
        provider = MetricProvider(db=Mock())
        key = ResultKey(yyyy_mm_dd=date.today(), tags={})

        negative_metric = provider.sum("returns", dataset="financials")
        provider._symbol_index[negative_metric].fn = lambda k: Success(-100.0)

        evaluator = Evaluator(provider, key)

        # sqrt(-100) = 10i
        expr = sp.sqrt(negative_metric)
        result = evaluator.evaluate(expr)

        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        assert "complex: 0.0 + 10.0i" in failures[0].error_message

    def test_log_negative_returns_complex_failure(self) -> None:
        """Test that log of negative value returns complex number failure."""
        provider = MetricProvider(db=Mock())
        key = ResultKey(yyyy_mm_dd=date.today(), tags={})

        negative_balance = provider.sum("balance", dataset="accounts")
        provider._symbol_index[negative_balance].fn = lambda k: Success(-50.0)

        evaluator = Evaluator(provider, key)

        # log(-50) produces complex result
        expr = sp.log(negative_balance)
        result = evaluator.evaluate(expr)

        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        # log(-50) = log(50) + πi ≈ 3.912023 + 3.141593i
        assert "complex:" in failures[0].error_message
        assert "3.14159" in failures[0].error_message  # π (checking first 6 digits)

    def test_negative_power_fractional_returns_complex_failure(self) -> None:
        """Test that negative number to fractional power returns complex failure."""
        provider = MetricProvider(db=Mock())
        key = ResultKey(yyyy_mm_dd=date.today(), tags={})

        negative_value = provider.sum("value", dataset="metrics")
        provider._symbol_index[negative_value].fn = lambda k: Success(-8.0)

        evaluator = Evaluator(provider, key)

        # (-8)^(1/3) has complex solutions
        expr = negative_value ** (1 / 3)
        result = evaluator.evaluate(expr)

        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        assert "complex:" in failures[0].error_message

    def test_positive_sqrt_returns_success(self) -> None:
        """Test that square root of positive value succeeds."""
        provider = MetricProvider(db=Mock())
        key = ResultKey(yyyy_mm_dd=date.today(), tags={})

        positive_metric = provider.sum("revenue", dataset="sales")
        provider._symbol_index[positive_metric].fn = lambda k: Success(100.0)

        evaluator = Evaluator(provider, key)

        # sqrt(100) = 10
        expr = sp.sqrt(positive_metric)
        result = evaluator.evaluate(expr)

        assert isinstance(result, Success)
        assert result.unwrap() == 10.0

    def test_complex_expression_with_imaginary_unit(self) -> None:
        """Test expression that explicitly includes imaginary unit."""
        provider = MetricProvider(db=Mock())
        key = ResultKey(yyyy_mm_dd=date.today(), tags={})

        real_metric = provider.sum("real_part", dataset="complex_data")
        provider._symbol_index[real_metric].fn = lambda k: Success(5.0)

        evaluator = Evaluator(provider, key)

        # 5 + 3i
        expr = real_metric + 3 * sp.I
        result = evaluator.evaluate(expr)

        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        assert "complex: 5.0 + 3.0i" in failures[0].error_message

    def test_complex_arithmetic_operations(self) -> None:
        """Test that complex arithmetic operations are caught."""
        provider = MetricProvider(db=Mock())
        key = ResultKey(yyyy_mm_dd=date.today(), tags={})

        metric_a = provider.sum("a", dataset="test")
        metric_b = provider.sum("b", dataset="test")
        provider._symbol_index[metric_a].fn = lambda k: Success(-1.0)
        provider._symbol_index[metric_b].fn = lambda k: Success(2.0)

        evaluator = Evaluator(provider, key)

        # sqrt(-1) * 2 = 2i
        expr = sp.sqrt(metric_a) * metric_b
        result = evaluator.evaluate(expr)

        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        assert "complex: 0.0 + 2.0i" in failures[0].error_message
