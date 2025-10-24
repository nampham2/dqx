"""Additional tests to improve coverage for evaluator.py."""

from datetime import date
from unittest.mock import Mock, patch

import pytest
import sympy as sp
from returns.result import Failure, Success

from dqx.common import DQXError, ResultKey
from dqx.evaluator import Evaluator
from dqx.provider import MetricProvider, SymbolicMetric


def test_evaluator_sympify_non_basic_expression() -> None:
    """Test that non-sympy Basic expressions are converted using sympify in _gather."""
    provider = Mock(spec=MetricProvider)
    key = ResultKey(yyyy_mm_dd=date.today(), tags={})
    evaluator = Evaluator(provider, key, "Test Suite")

    # Create a symbol
    x = sp.Symbol("x")

    # Create symbolic metric
    symbolic_metric = SymbolicMetric(
        name="x",
        symbol=x,
        fn=lambda k: Success(5.0),
        key_provider=Mock(),
        metric_spec=Mock(__str__=Mock(return_value="test_metric")),
        dataset="test_dataset",
    )

    provider.symbolic_metrics = [symbolic_metric]
    provider.get_symbol.return_value = symbolic_metric

    # Mock metrics to include our symbol
    evaluator._metrics = {x: Success(5.0)}

    # The sympify happens in _gather, not evaluate
    # We need to test _gather directly with a non-Basic expression
    # True is not a sp.Basic, so it will be sympified
    symbol_values, symbol_infos = evaluator._gather(True)

    # True becomes S.true which has no free symbols
    assert len(symbol_values) == 0
    assert len(symbol_infos) == 0

    # Test with a more complex case - a Python int
    # This tests the sympify path for non-Basic values
    symbol_values2, symbol_infos2 = evaluator._gather(42)
    assert len(symbol_values2) == 0  # No symbols in constant
    assert len(symbol_infos2) == 0


def test_evaluator_symbol_not_in_metrics() -> None:
    """Test error when symbol is not found in collected metrics."""
    provider = Mock(spec=MetricProvider)
    key = ResultKey(yyyy_mm_dd=date.today(), tags={})
    evaluator = Evaluator(provider, key, "Test Suite")

    # Create a symbol that won't be in metrics
    missing_symbol = sp.Symbol("missing")

    # Create a mock for the missing symbol's metric
    missing_metric = Mock()
    missing_metric.name = "missing"

    # Set up provider to return the metric when asked
    provider.get_symbol.return_value = missing_metric

    # Initialize with empty metrics
    evaluator._metrics = {}

    # Try to evaluate an expression with the missing symbol
    with pytest.raises(DQXError, match="Symbol missing not found in collected metrics"):
        evaluator.evaluate(missing_symbol)


def test_evaluator_infinity_result() -> None:
    """Test that infinity results are converted to EvaluationFailure."""
    provider = Mock(spec=MetricProvider)
    key = ResultKey(yyyy_mm_dd=date.today(), tags={})
    evaluator = Evaluator(provider, key, "Test Suite")

    # Create a symbol
    x = sp.Symbol("x")

    # Create symbolic metric with a very large value
    symbolic_metric = SymbolicMetric(
        name="x",
        symbol=x,
        fn=lambda k: Success(1e308),  # Very large number
        key_provider=Mock(),
        metric_spec=Mock(__str__=Mock(return_value="metric_x")),
        dataset="dataset1",
    )

    provider.symbolic_metrics = [symbolic_metric]
    provider.get_symbol.return_value = symbolic_metric
    evaluator._metrics = {x: Success(1e308)}

    # Evaluate x * 10 which should overflow to infinity
    result = evaluator.evaluate(x * 10)

    assert isinstance(result, Failure)
    failures = result.failure()
    assert len(failures) == 1
    assert failures[0].error_message == "Validating value is infinity"


def test_evaluator_unexpected_exception() -> None:
    """Test that unexpected exceptions during evaluation are caught."""
    provider = Mock(spec=MetricProvider)
    key = ResultKey(yyyy_mm_dd=date.today(), tags={})
    evaluator = Evaluator(provider, key, "Test Suite")

    # Create a symbol
    x = sp.Symbol("x")

    # Create symbolic metric
    symbolic_metric = SymbolicMetric(
        name="x",
        symbol=x,
        fn=lambda k: Success(5.0),
        key_provider=Mock(),
        metric_spec=Mock(__str__=Mock(return_value="test_metric")),
        dataset="test_dataset",
    )

    provider.symbolic_metrics = [symbolic_metric]
    provider.get_symbol.return_value = symbolic_metric
    evaluator._metrics = {x: Success(5.0)}

    # Mock sp.N to raise an exception
    with patch("sympy.N", side_effect=RuntimeError("Unexpected error")):
        result = evaluator.evaluate(x)

    assert isinstance(result, Failure)
    failures = result.failure()
    assert len(failures) == 1
    assert "Error evaluating expression: Unexpected error" in failures[0].error_message
    assert failures[0].expression == "x"
    assert len(failures[0].symbols) == 1


def test_evaluator_complex_infinity_zoo() -> None:
    """Test that complex infinity (zoo) is handled correctly."""
    provider = Mock(spec=MetricProvider)
    key = ResultKey(yyyy_mm_dd=date.today(), tags={})
    evaluator = Evaluator(provider, key, "Test Suite")

    # Create a symbol
    x = sp.Symbol("x")

    # Create symbolic metric
    symbolic_metric = SymbolicMetric(
        name="x",
        symbol=x,
        fn=lambda k: Success(1.0),  # Use 1.0 instead of 0.0
        key_provider=Mock(),
        metric_spec=Mock(__str__=Mock(return_value="test_metric")),
        dataset="test_dataset",
    )

    provider.symbolic_metrics = [symbolic_metric]
    provider.get_symbol.return_value = symbolic_metric
    evaluator._metrics = {x: Success(1.0)}

    # Create an expression that directly evaluates to zoo
    # The evaluator handles zoo as a special case before converting to float
    # We need to test the code path that checks for zoo
    # Create a mock that makes the expression evaluate to zoo
    with patch("sympy.Basic.subs") as mock_subs:
        mock_subs.return_value = sp.zoo
        result = evaluator.evaluate(x)

    assert isinstance(result, Failure)
    failures = result.failure()
    assert len(failures) == 1
    assert failures[0].error_message == "Validating value is infinity"


def test_evaluator_complex_number_result() -> None:
    """Test that complex number results are handled correctly."""
    provider = Mock(spec=MetricProvider)
    key = ResultKey(yyyy_mm_dd=date.today(), tags={})
    evaluator = Evaluator(provider, key, "Test Suite")

    # Create a symbol
    x = sp.Symbol("x")

    # Create symbolic metric
    symbolic_metric = SymbolicMetric(
        name="x",
        symbol=x,
        fn=lambda k: Success(-1.0),
        key_provider=Mock(),
        metric_spec=Mock(__str__=Mock(return_value="test_metric")),
        dataset="test_dataset",
    )

    provider.symbolic_metrics = [symbolic_metric]
    provider.get_symbol.return_value = symbolic_metric
    evaluator._metrics = {x: Success(-1.0)}

    # sqrt(-1) = i (imaginary unit)
    expr = sp.sqrt(x)

    result = evaluator.evaluate(expr)

    assert isinstance(result, Failure)
    failures = result.failure()
    assert len(failures) == 1
    assert "Validating value is complex: 0.0 + 1.0i" in failures[0].error_message


def test_evaluator_expression_with_constants() -> None:
    """Test evaluation of expression mixing symbols and constants."""
    provider = Mock(spec=MetricProvider)
    key = ResultKey(yyyy_mm_dd=date.today(), tags={})
    evaluator = Evaluator(provider, key, "Test Suite")

    # Create a symbol
    x = sp.Symbol("x")

    # Create symbolic metric
    symbolic_metric = SymbolicMetric(
        name="x",
        symbol=x,
        fn=lambda k: Success(10.0),
        key_provider=Mock(),
        metric_spec=Mock(__str__=Mock(return_value="test_metric")),
        dataset="test_dataset",
    )

    provider.symbolic_metrics = [symbolic_metric]
    provider.get_symbol.return_value = symbolic_metric
    evaluator._metrics = {x: Success(10.0)}

    # Evaluate 2 * x + 5
    expr = 2 * x + 5
    result = evaluator.evaluate(expr)

    assert isinstance(result, Success)
    assert result.unwrap() == 25.0


def test_evaluator_collect_symbols() -> None:
    """Test the collect_symbols method."""
    provider = Mock(spec=MetricProvider)
    key = ResultKey(yyyy_mm_dd=date(2024, 1, 15), tags={"env": "prod"})
    evaluator = Evaluator(provider, key, "Test Suite")

    # Create symbols
    x = sp.Symbol("x")
    y = sp.Symbol("y")

    # Create symbolic metrics
    sm_x = SymbolicMetric(
        name="metric_x",  # This is what appears in si.metric
        symbol=x,
        fn=lambda k: Success(5.0),
        key_provider=Mock(),
        metric_spec=Mock(),
        dataset="dataset1",
    )
    sm_y = SymbolicMetric(
        name="metric_y",  # This is what appears in si.metric
        symbol=y,
        fn=lambda k: Failure("Error loading y"),
        key_provider=Mock(),
        metric_spec=Mock(),
        dataset="dataset2",
    )

    # Fix for mypy: don't assign to provider attributes directly
    def get_symbol_impl(s: sp.Symbol) -> SymbolicMetric:
        return sm_x if s == x else sm_y

    provider.get_symbol.side_effect = get_symbol_impl
    evaluator._metrics = {x: Success(5.0), y: Failure("Error loading y")}

    # Collect symbols from expression
    expr = x + y * 2
    symbol_infos = evaluator.collect_symbols(expr)

    assert len(symbol_infos) == 2

    # Check symbol info contains correct data
    symbol_names = {si.name for si in symbol_infos}
    assert symbol_names == {"x", "y"}

    for si in symbol_infos:
        assert si.yyyy_mm_dd == date(2024, 1, 15)
        assert si.suite == "Test Suite"
        assert si.tags == {"env": "prod"}

        if si.name == "x":
            assert si.metric == "metric_x"  # si.metric is the name field from SymbolicMetric
            assert si.dataset == "dataset1"
            assert si.value == Success(5.0)
        else:  # y
            assert si.metric == "metric_y"  # si.metric is the name field from SymbolicMetric
            assert si.dataset == "dataset2"
            assert si.value == Failure("Error loading y")
