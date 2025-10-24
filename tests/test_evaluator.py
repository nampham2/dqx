from datetime import date
from unittest.mock import Mock

import pytest
import sympy as sp
from returns.result import Failure, Success

from dqx.common import DQXError, EvaluationFailure, ResultKey
from dqx.evaluator import Evaluator
from dqx.provider import MetricProvider, SymbolicMetric
from dqx.specs import Average


class TestEvaluatorFailureHandling:
    """Test that Evaluator returns EvaluationFailure objects."""

    def test_metric_failure_returns_evaluation_failure(self) -> None:
        """Test that metric failures are converted to EvaluationFailure."""
        provider = Mock(spec=MetricProvider)
        key = ResultKey(yyyy_mm_dd=date.today(), tags={})
        evaluator = Evaluator(provider, key, "Test Suite")

        # Create a symbol that will fail
        symbol = sp.Symbol("x_1")
        symbolic_metric = SymbolicMetric(
            name="average(price)",
            symbol=symbol,
            fn=lambda k: Failure("Database error"),
            key_provider=Mock(),
            metric_spec=Average("price"),
            dataset="orders",
        )

        # Mock the provider methods
        provider.symbolic_metrics = [symbolic_metric]
        provider.get_symbol.return_value = symbolic_metric

        # Mock the metrics collection to return a failure
        evaluator._metrics = {symbol: Failure("Database error")}

        # Evaluate an expression with the failing symbol
        result = evaluator.evaluate(symbol)

        # Assert we get EvaluationFailure objects
        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        failure = failures[0]
        assert isinstance(failure, EvaluationFailure)
        assert failure.error_message == "One or more metrics failed to evaluate"
        assert failure.expression == str(symbol)
        assert len(failure.symbols) == 1

        symbol_info = failure.symbols[0]
        assert symbol_info.name == "x_1"
        assert "average(price)" in symbol_info.metric or symbol_info.metric == str(Average("price"))
        assert symbol_info.dataset == "orders"
        assert isinstance(symbol_info.value, Failure)
        assert symbol_info.value.failure() == "Database error"

    def test_expression_nan_returns_evaluation_failure(self) -> None:
        """Test that NaN results are converted to EvaluationFailure."""
        provider = Mock(spec=MetricProvider)
        key = ResultKey(yyyy_mm_dd=date.today(), tags={})
        evaluator = Evaluator(provider, key, "Test Suite")

        # Create symbols that will divide to NaN
        x1 = sp.Symbol("x_1")
        x2 = sp.Symbol("x_2")

        # Create symbolic metrics
        metric_spec1 = Mock(__str__=Mock(return_value="average(price)"))
        metric_spec2 = Mock(__str__=Mock(return_value="sum(quantity)"))

        sm1 = SymbolicMetric(
            name="average(price)",
            symbol=x1,
            fn=lambda k: Success(0.0),
            key_provider=Mock(),
            metric_spec=metric_spec1,
            dataset="orders",
        )
        sm2 = SymbolicMetric(
            name="sum(quantity)",
            symbol=x2,
            fn=lambda k: Success(0.0),
            key_provider=Mock(),
            metric_spec=metric_spec2,
            dataset="inventory",
        )

        provider.symbolic_metrics = [sm1, sm2]
        provider.get_symbol.side_effect = lambda s: sm1 if s == x1 else sm2

        # Mock successful metric collection
        evaluator._metrics = {x1: Success(0.0), x2: Success(0.0)}

        # Evaluate 0/0 which gives NaN
        result = evaluator.evaluate(sp.zoo * x1 / x2)  # Use zoo to ensure NaN

        # Assert we get EvaluationFailure
        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        failure = failures[0]
        assert failure.error_message == "Validating value is NaN"
        assert failure.expression == "zoo*x_1/x_2"
        assert len(failure.symbols) == 2

        # Check symbols (order may vary)
        symbol_names = {s.name for s in failure.symbols}
        assert symbol_names == {"x_1", "x_2"}

        for s in failure.symbols:
            if s.name == "x_1":
                assert s.metric == "average(price)"
                assert s.value == Success(0.0)
            elif s.name == "x_2":
                assert s.metric == "sum(quantity)"
                assert s.value == Success(0.0)

    def test_symbol_not_in_provider(self) -> None:
        """Test error when symbol is not found in provider."""
        provider = Mock(spec=MetricProvider)
        key = ResultKey(yyyy_mm_dd=date.today(), tags={})
        evaluator = Evaluator(provider, key, "Test Suite")

        # Create a symbol that doesn't exist in provider
        unknown_symbol = sp.Symbol("x_unknown")

        # Mock provider to raise error
        provider.get_symbol.side_effect = DQXError("Symbol not found")

        # Mock metrics without the symbol
        evaluator._metrics = {}

        # Attempt to evaluate should handle the error gracefully
        with pytest.raises(DQXError, match="Symbol"):
            evaluator.evaluate(unknown_symbol)

    def test_complex_expression_with_multiple_operators(self) -> None:
        """Test complex expression like (a + b) / (c - d)."""
        provider = Mock(spec=MetricProvider)
        key = ResultKey(yyyy_mm_dd=date.today(), tags={})
        evaluator = Evaluator(provider, key, "Test Suite")

        # Create symbols
        a, b, c, d = sp.symbols("a b c d")

        # Create symbolic metrics
        metrics = {
            a: ("metric_a", "dataset1", 10.0),
            b: ("metric_b", "dataset1", 20.0),
            c: ("metric_c", "dataset2", 15.0),
            d: ("metric_d", "dataset2", 15.0),  # Will make denominator 0
        }

        for sym, (name, dataset, value) in metrics.items():

            def metric_fn(k: ResultKey, val: float = value) -> Success[float]:
                return Success(val)

            SymbolicMetric(
                name=str(sym),
                symbol=sym,
                fn=metric_fn,
                key_provider=Mock(),
                metric_spec=Mock(name=name),
                dataset=dataset,
            )

        def get_symbol_mock(s: sp.Symbol) -> Mock:
            # Create a mock spec that returns the metric name when converted to string
            mock_spec = Mock()
            mock_spec.__str__ = Mock(return_value=metrics[s][0])  # type: ignore
            return Mock(name=str(s), dataset=metrics[s][1], metric_spec=mock_spec)

        provider.get_symbol.side_effect = get_symbol_mock

        # Mock metrics
        evaluator._metrics = {sym: Success(val[2]) for sym, val in metrics.items()}

        # Evaluate (a + b) / (c - d) = 30 / 0 = infinity
        result = evaluator.evaluate((a + b) / (c - d))

        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        assert (
            "Cannot convert complex to float" in failures[0].error_message
            or failures[0].error_message == "Validating value is infinity"
        )

    def test_empty_expression(self) -> None:
        """Test expression with no free symbols."""
        provider = Mock(spec=MetricProvider)
        key = ResultKey(yyyy_mm_dd=date.today(), tags={})
        evaluator = Evaluator(provider, key, "Test Suite")

        # Evaluate a constant expression
        result = evaluator.evaluate(sp.sympify(42))

        assert isinstance(result, Success)
        assert result.unwrap() == 42.0

    def test_constant_expression_nan(self) -> None:
        """Test NaN from constant expression like 0/0."""
        provider = Mock(spec=MetricProvider)
        key = ResultKey(yyyy_mm_dd=date.today(), tags={})
        evaluator = Evaluator(provider, key, "Test Suite")

        # Evaluate 0/0 which should result in NaN
        result = evaluator.evaluate(sp.sympify("0/0"))

        # Should return EvaluationFailure with no symbols
        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        failure = failures[0]
        assert failure.error_message == "Validating value is NaN"
        assert failure.expression == "nan"  # sympy evaluates 0/0 to nan
        assert len(failure.symbols) == 0  # No symbols in constant expression
